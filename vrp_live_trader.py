#!/usr/bin/env python3
# ============================================================
#  VRP Live Trader — Alpaca
#  60% SPY buy-and-hold  +  20%/slot IWM options (2 slots)
#
#  Install:
#    pip install alpaca-py yfinance numpy pandas scipy
#
#  Setup:
#    1. Create a free account at alpaca.markets
#    2. Generate API keys (paper first, live later)
#    3. Set environment variables OR edit the CONFIG block below
#       export ALPACA_API_KEY="your_key"
#       export ALPACA_SECRET_KEY="your_secret"
#       export ALPACA_PAPER=true          # set false for live
#
#  Run:  python vrp_live_trader.py
#    Schedule via cron: 30 9 * * 1-5  (9:30 AM ET, Mon-Fri)
#    Or run manually each morning before 10 AM ET.
#
#  How it works:
#    Each run takes ~60 seconds. It:
#      1. Reads state.json (tracks both slots)
#      2. Manages open positions (profit target / DTE exit)
#      3. Opens new positions if a slot is empty
#      4. Manages SPY allocation (buy if under-allocated)
#      5. Writes updated state.json and appends to trade_log.csv
#
#  State file (state.json):
#    Persists between runs. Tracks slot A and B positions,
#    SPY shares held, and cumulative P&L. Delete it to reset.
#
#  IMPORTANT — read before going live:
#    - Paper trade for at least 3 months first
#    - Verify fills match expected credit before scaling up
#    - Options Level 3 required for iron condors on live account
#    - IWM options are American-style (early assignment risk near DTE)
#    - Always review the log before market close on entry days
# ============================================================

import os, sys, json, math, logging
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

# ── Alpaca imports ────────────────────────────────────────────
from alpaca.trading.client     import TradingClient
from alpaca.trading.requests   import (
    MarketOrderRequest, LimitOrderRequest,
    GetOptionContractsRequest, GetOrdersRequest, ClosePositionRequest,
    OptionLegRequest,
)
from alpaca.trading.enums      import (
    OrderSide, OrderType, TimeInForce, OrderClass,
    AssetStatus, ContractType,
)
from alpaca.data.historical    import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests      import (
    StockLatestQuoteRequest, StockBarsRequest,
    OptionLatestQuoteRequest, OptionSnapshotRequest,
)
from alpaca.data.timeframe     import TimeFrame


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION — edit or use environment variables
# ══════════════════════════════════════════════════════════════

API_KEY    = os.environ.get("ALPACA_API_KEY",    "YOUR_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY")
PAPER      = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

CONFIG = {
    # ── Portfolio allocation ──────────────────────────────────
    "spy_weight":      0.60,   # 60% SPY buy-and-hold
    "slot_risk":       0.20,   # 20% per IWM options slot

    # ── Options parameters ────────────────────────────────────
    "slot_dte":        14,     # target DTE at entry
    "dte_tolerance":   3,      # accept contracts within ±3 DTE of target
    "exit_dte":        3,      # force-close at ≤3 DTE
    "min_hold_days":   3,      # minimum days before profit target fires
    "profit_target":   0.50,   # close at 50% of credit received
    "stagger_days":    7,      # Slot B opens 7 days after Slot A

    # ── Regime signals ────────────────────────────────────────
    "trend_fast":      20,     # SMA fast period
    "trend_slow":      50,     # SMA slow period
    "hv_lookback":     20,     # historical vol lookback
    "ivr_lookback":    252,    # IVR percentile window
    "atr_fast":        5,
    "atr_slow":        20,
    "vrp_factor":      1.18,   # IV = HV × factor (IWM VRP adjustment)

    # ── Spread parameters ─────────────────────────────────────
    "spread_pct":      0.025,  # strike width as % of underlying
    "r":               0.04,   # risk-free rate for B-S

    # ── Execution ─────────────────────────────────────────────
    # Credit limit orders: submit at mid, then widen by this if unfilled
    "limit_offset":    0.05,   # widen by $0.05 if not filled in 30s
    "max_fill_retries": 3,

    # ── Files ─────────────────────────────────────────────────
    "state_file":     "state.json",
    "log_file":       "trade_log.csv",
}


# ══════════════════════════════════════════════════════════════
#  REGIME MAP (identical to backtest)
# ══════════════════════════════════════════════════════════════

REGIME_MAP = {
    ("bullish","low","contracting"): "long_dte_condor",
    ("bullish","low","expanding"):   "long_dte_condor",
    ("bullish","low","unknown"):     "long_dte_condor",
    ("bullish","mid","contracting"): "long_dte_condor",
    ("bullish","mid","expanding"):   "long_dte_condor",
    ("bullish","mid","unknown"):     "long_dte_condor",
    ("bullish","high","contracting"):"skewed_put",
    ("bullish","high","expanding"):  "skewed_put",
    ("bullish","high","unknown"):    "skewed_put",
    ("bearish","high","contracting"):"skewed_put",
    ("bearish","high","expanding"):  "skewed_put",
    ("bearish","high","unknown"):    "skewed_put",
    ("bearish","mid","contracting"): "skewed_put",
    ("bearish","mid","expanding"):   "iron_condor",
    ("bearish","mid","unknown"):     "iron_condor",
    ("bearish","low","contracting"): "skewed_put",
    ("bearish","low","expanding"):   "skewed_put",
    ("bearish","low","unknown"):     "skewed_put",
    ("neutral","high","contracting"):"skewed_put",
    ("neutral","high","expanding"):  "skewed_put",
    ("neutral","high","unknown"):    "skewed_put",
    ("neutral","mid","contracting"): "long_dte_condor",
    ("neutral","mid","expanding"):   "iron_condor",
    ("neutral","mid","unknown"):     "iron_condor",
    ("neutral","low","contracting"): "long_dte_condor",
    ("neutral","low","expanding"):   "long_dte_condor",
    ("neutral","low","unknown"):     "long_dte_condor",
    ("unknown","high","contracting"):"iron_condor",
    ("unknown","high","expanding"):  "iron_condor",
    ("unknown","high","unknown"):    "iron_condor",
    ("unknown","mid","contracting"): "iron_condor",
    ("unknown","mid","expanding"):   "iron_condor",
    ("unknown","mid","unknown"):     "iron_condor",
    ("unknown","low","contracting"): "iron_condor",
    ("unknown","low","expanding"):   "iron_condor",
    ("unknown","low","unknown"):     "iron_condor",
}

# Spread definitions — DTE will be set to CONFIG["slot_dte"] at runtime
SPREAD_PARAMS = {
    "long_dte_condor": {"put_delta":0.20,"call_delta":0.20,"spread_pct":0.035,"put_width_mult":1.0},
    "skewed_put":      {"put_delta":0.20,"call_delta":0.20,"spread_pct":0.030,"put_width_mult":1.5},
    "iron_condor":     {"put_delta":0.20,"call_delta":0.20,"spread_pct":0.030,"put_width_mult":1.0},
    "bull_put":        {"put_delta":0.25,"call_delta":0.00,"spread_pct":0.030,"put_width_mult":1.0},
}


# ══════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("vrp_trader.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("vrp")


# ══════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════

def load_state() -> dict:
    path = Path(CONFIG["state_file"])
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "slots": {
            "A": {"position": None, "next_entry": None},
            "B": {"position": None, "next_entry": None},
        },
        "spy_shares":     0.0,
        "initial_capital": None,     # set on first run
        "cumulative_pnl":  0.0,
        "trade_count":     0,
    }


def save_state(state: dict):
    with open(CONFIG["state_file"], "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(record: dict):
    path = Path(CONFIG["log_file"])
    df   = pd.DataFrame([record])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# ══════════════════════════════════════════════════════════════
#  BLACK-SCHOLES HELPERS (for strike selection)
# ══════════════════════════════════════════════════════════════

def bs_delta(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0: return 0.0
    try:
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        return (norm.cdf(d1) - 1.0) if option_type=="put" else norm.cdf(d1)
    except Exception:
        return 0.0


def find_strike_by_delta(S, r, sigma, T, target_delta, option_type="put"):
    if T <= 0 or sigma <= 0:
        return S * (1 - target_delta) if option_type=="put" else S * (1 + target_delta)
    try:
        return brentq(
            lambda K: abs(bs_delta(S, K, T, r, sigma, option_type)) - target_delta,
            S*0.40, S*1.60, xtol=0.01
        )
    except ValueError:
        return S*(1-target_delta*1.5) if option_type=="put" else S*(1+target_delta*1.5)


# ══════════════════════════════════════════════════════════════
#  REGIME DETECTION
# ══════════════════════════════════════════════════════════════

def get_regime(cfg: dict) -> dict:
    """
    Download recent IWM and VIX data via yfinance and compute
    the same regime signals as the backtest.
    Returns: {"trend", "vol", "atr", "hv", "spread_chosen"}
    """
    lookback_days = cfg["ivr_lookback"] + 60
    end   = datetime.today()
    start = end - timedelta(days=lookback_days * 1.5)

    iwm = yf.download("IWM",  start=start, end=end,
                      auto_adjust=True, progress=False)["Close"].squeeze()
    vix = yf.download("^VIX", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"].squeeze()

    # Align
    df = pd.DataFrame({"iwm": iwm, "vix": vix}).ffill().dropna()
    px  = df["iwm"]
    vix = df["vix"]

    lr       = np.log(px / px.shift(1))
    hv       = lr.rolling(cfg["hv_lookback"]).std() * np.sqrt(252)
    hv_min   = hv.rolling(cfg["ivr_lookback"]).min()
    hv_max   = hv.rolling(cfg["ivr_lookback"]).max()

    vix_min  = vix.rolling(cfg["ivr_lookback"]).min()
    vix_max  = vix.rolling(cfg["ivr_lookback"]).max()
    vix_pct  = ((vix - vix_min) / (vix_max - vix_min + 1e-9)).clip(0,1)

    sma_fast = px.rolling(cfg["trend_fast"]).mean()
    sma_slow = px.rolling(cfg["trend_slow"]).mean()
    atr_fast = px.diff().abs().rolling(cfg["atr_fast"]).mean()
    atr_slow = px.diff().abs().rolling(cfg["atr_slow"]).mean()

    # Latest values
    p  = px.iloc[-1]
    f  = sma_fast.iloc[-1];  s  = sma_slow.iloc[-1]
    vp = vix_pct.iloc[-1]
    af = atr_fast.iloc[-1];  as_ = atr_slow.iloc[-1]
    hv_val = hv.iloc[-1]
    vix_val = vix.iloc[-1]

    # Trend
    if p > f > s:       trend = "bullish"
    elif p < f < s:     trend = "bearish"
    else:               trend = "neutral"

    # Vol
    if vp > 0.67:       vol = "high"
    elif vp > 0.33:     vol = "mid"
    else:               vol = "low"

    # ATR
    if pd.isna(af) or pd.isna(as_) or as_ == 0:
        atr = "unknown"
    else:
        atr = "expanding" if af > as_ else "contracting"

    spread = REGIME_MAP.get((trend, vol, atr),
             REGIME_MAP.get((trend, vol, "unknown"), "iron_condor"))

    iv = float(hv_val) * cfg["vrp_factor"] if not pd.isna(hv_val) else 0.20

    return {
        "trend":    trend,
        "vol":      vol,
        "atr":      atr,
        "hv":       float(hv_val) if not pd.isna(hv_val) else 0.20,
        "iv":       iv,
        "vix":      float(vix_val),
        "price":    float(p),
        "spread":   spread,
    }


# ══════════════════════════════════════════════════════════════
#  OPTION CONTRACT SELECTION
# ══════════════════════════════════════════════════════════════

def find_contract(trade_client, opt_data_client,
                  underlying: str, option_type: str,
                  target_delta: float, target_strike: float,
                  target_expiry: date) -> dict:
    """
    Find the nearest IWM option contract to target_strike
    expiring closest to target_expiry.
    Returns dict with {symbol, strike, expiry, bid, ask, mid}
    """
    exp_min = (target_expiry - timedelta(days=CONFIG["dte_tolerance"])).isoformat()
    exp_max = (target_expiry + timedelta(days=CONFIG["dte_tolerance"])).isoformat()

    try:
        contracts = trade_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[underlying],
                expiration_date_gte=exp_min,
                expiration_date_lte=exp_max,
                type=ContractType.PUT if option_type=="put" else ContractType.CALL,
                status=AssetStatus.ACTIVE,
            )
        )
    except Exception as e:
        log.error(f"Contract lookup failed: {e}")
        return None

    if not contracts.option_contracts:
        log.warning(f"No {option_type} contracts found for {underlying} near {target_expiry}")
        return None

    # Pick contract closest to target strike
    best = min(
        contracts.option_contracts,
        key=lambda c: abs(float(c.strike_price) - target_strike)
    )

    # Get latest quote for mid price
    try:
        snap = opt_data_client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=[best.symbol])
        )
        quote = snap[best.symbol].latest_quote
        bid, ask = float(quote.bid_price), float(quote.ask_price)
        mid = (bid + ask) / 2
    except Exception:
        bid = ask = mid = 0.0

    return {
        "symbol":  best.symbol,
        "strike":  float(best.strike_price),
        "expiry":  best.expiration_date,
        "type":    option_type,
        "bid":     bid,
        "ask":     ask,
        "mid":     mid,
    }


# ══════════════════════════════════════════════════════════════
#  ORDER ENTRY
# ══════════════════════════════════════════════════════════════

def submit_spread(trade_client, spread_type: str, legs_info: list,
                  contracts: int, net_credit: float) -> str:
    """
    Submit a multi-leg credit spread order.
    legs_info: list of {"symbol", "side": "sell"/"buy"}
    net_credit: positive number (credit we expect to receive)
    Returns order ID or None on failure.
    """
    legs = []
    for leg in legs_info:
        legs.append(OptionLegRequest(
            symbol=leg["symbol"],
            side=OrderSide.SELL if leg["side"]=="sell" else OrderSide.BUY,
            ratio_qty=1,
        ))

    # Alpaca mleg credit spread: limit_price = positive net credit
    # (the minimum credit you are willing to receive per share)
    limit_price = round(abs(net_credit), 2)
    if limit_price <= 0:
        log.error("  Net credit is zero or negative, cannot submit")
        return None

    try:
        order = trade_client.submit_order(
            LimitOrderRequest(
                qty=contracts,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG,
                limit_price=limit_price,
                legs=legs,
            )
        )
        log.info(f"  Order submitted: {order.id}  limit=${limit_price}  qty={contracts}")
        return str(order.id)
    except Exception as e:
        log.error(f"  Order submission failed: {e}")
        return None


def close_position_by_legs(trade_client, position_state: dict) -> bool:
    """Close an existing spread position at market."""
    leg_symbols = position_state.get("leg_symbols", [])
    success = True
    for sym in leg_symbols:
        try:
            trade_client.close_position(sym)
            log.info(f"  Closed leg: {sym}")
        except Exception as e:
            log.error(f"  Failed to close {sym}: {e}")
            success = False
    return success


# ══════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════

def compute_contracts(portfolio_value: float, max_loss_per_spread: float) -> int:
    """
    Number of contracts such that max_loss * contracts * 100
    = slot_risk * portfolio_value
    """
    if max_loss_per_spread <= 0:
        return 0
    budget = portfolio_value * CONFIG["slot_risk"]
    n = int(budget / (max_loss_per_spread * 100))
    return max(1, min(n, 10))   # hard cap at 10 contracts


# ══════════════════════════════════════════════════════════════
#  SPY ALLOCATION MANAGER
# ══════════════════════════════════════════════════════════════

def manage_spy_allocation(trade_client, data_client, state: dict,
                          portfolio_value: float):
    """
    Ensure ~60% of portfolio is in SPY.
    On first run: buys the initial SPY position.
    Subsequent runs: rebalances if >5% off target.
    """
    target_value = portfolio_value * CONFIG["spy_weight"]
    current_shares = state.get("spy_shares", 0.0)

    # Get current SPY price — try quote first, fall back to last daily bar
    spy_price = 0.0
    try:
        quote = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=["SPY"])
        )
        ask = float(quote["SPY"].ask_price)
        bid = float(quote["SPY"].bid_price)
        spy_price = ask if ask > 0 else bid
    except Exception:
        pass

    if spy_price <= 0:
        # Market closed — use last daily bar close
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            bars = data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=["SPY"],
                    timeframe=TimeFrame.Day,
                    limit=1,
                )
            )
            spy_price = float(bars["SPY"][-1].close)
        except Exception as e:
            log.error(f"Could not get SPY price from bars: {e}")
            return

    if spy_price <= 0:
        log.error("SPY price is still zero after fallback, skipping allocation")
        return

    current_value = current_shares * spy_price

    if current_shares == 0:
        # First run — buy initial SPY position
        shares_to_buy = int(target_value / spy_price)
        if shares_to_buy < 1:
            log.warning("Portfolio too small to buy even 1 SPY share")
            return
        try:
            order = trade_client.submit_order(
                MarketOrderRequest(
                    symbol="SPY",
                    qty=shares_to_buy,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            state["spy_shares"] = shares_to_buy
            log.info(f"  SPY initial buy: {shares_to_buy} shares @ ~${spy_price:.2f}")
        except Exception as e:
            log.error(f"  SPY buy failed: {e}")
    else:
        # Rebalance check — only if >5% off target
        drift = abs(current_value - target_value) / target_value
        if drift > 0.05:
            target_shares = int(target_value / spy_price)
            delta_shares  = target_shares - int(current_shares)
            if delta_shares > 0:
                try:
                    trade_client.submit_order(
                        MarketOrderRequest(
                            symbol="SPY", qty=delta_shares,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                        )
                    )
                    state["spy_shares"] += delta_shares
                    log.info(f"  SPY rebalance: bought {delta_shares} shares")
                except Exception as e:
                    log.error(f"  SPY rebalance failed: {e}")
            elif delta_shares < 0:
                try:
                    trade_client.submit_order(
                        MarketOrderRequest(
                            symbol="SPY", qty=abs(delta_shares),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )
                    )
                    state["spy_shares"] += delta_shares
                    log.info(f"  SPY rebalance: sold {abs(delta_shares)} shares")
                except Exception as e:
                    log.error(f"  SPY rebalance failed: {e}")


# ══════════════════════════════════════════════════════════════
#  SLOT MANAGEMENT
# ══════════════════════════════════════════════════════════════

def manage_slot(trade_client, opt_data_client, slot_id: str,
                slot_state: dict, portfolio_value: float,
                today: date) -> dict:
    """
    Check if open position needs to be closed.
    Returns updated slot_state.
    """
    pos = slot_state.get("position")
    if pos is None:
        return slot_state

    entry_date = date.fromisoformat(pos["entry_date"])
    expiry     = date.fromisoformat(pos["expiry"])
    days_held  = (today - entry_date).days
    dte        = (expiry - today).days

    log.info(f"  Slot {slot_id}: {pos['spread']}  DTE={dte}  held={days_held}d  "
             f"credit=${pos['credit_received']:.2f}")

    # Rule 1: force close near expiry
    if dte <= CONFIG["exit_dte"]:
        log.info(f"  Slot {slot_id}: DTE exit (dte={dte})")
        if close_position_by_legs(trade_client, pos):
            pnl = _estimate_close_pnl(opt_data_client, pos, "dte_exit")
            _record_close(slot_state, today, "dte_exit", pnl)
    # Rule 2: profit target
    elif days_held >= CONFIG["min_hold_days"]:
        current_value = _get_spread_value(opt_data_client, pos)
        if current_value is not None:
            profit = pos["credit_received"] - current_value
            target = pos["credit_received"] * CONFIG["profit_target"]
            if profit >= target:
                log.info(f"  Slot {slot_id}: profit target  "
                         f"profit=${profit:.2f} >= target=${target:.2f}")
                if close_position_by_legs(trade_client, pos):
                    pnl = profit * pos["contracts"] * 100
                    _record_close(slot_state, today, "profit_target", pnl)

    return slot_state


def _get_spread_value(opt_data_client, pos: dict) -> float:
    """Get current mid-price sum of all legs."""
    try:
        syms = pos.get("leg_symbols", [])
        if not syms:
            return None
        snaps = opt_data_client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=syms)
        )
        total = 0.0
        for sym, is_short in zip(syms, pos.get("leg_sides", [])):
            q = snaps[sym].latest_quote
            mid = (float(q.bid_price) + float(q.ask_price)) / 2
            total += -mid if is_short else mid
        return max(-total, 0.0)
    except Exception as e:
        log.error(f"  Could not get spread value: {e}")
        return None


def _estimate_close_pnl(opt_data_client, pos: dict, reason: str) -> float:
    val = _get_spread_value(opt_data_client, pos)
    if val is None:
        return 0.0
    return (pos["credit_received"] - val) * pos["contracts"] * 100


def _record_close(slot_state: dict, today: date, reason: str, pnl: float):
    slot_state["position"]["close_date"]   = today.isoformat()
    slot_state["position"]["close_reason"] = reason
    slot_state["position"]["realized_pnl"] = pnl
    slot_state["closed_trades"] = slot_state.get("closed_trades", [])
    slot_state["closed_trades"].append(dict(slot_state["position"]))
    slot_state["position"]        = None
    slot_state["next_entry"]      = (today + timedelta(days=1)).isoformat()
    log.info(f"  Closed: {reason}  P&L=${pnl:,.2f}")


# ══════════════════════════════════════════════════════════════
#  SLOT ENTRY
# ══════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    """Return True if US options market is currently open (9:30-4:00 PM ET)."""
    from datetime import timezone
    import zoneinfo
    try:
        et = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        # fallback: UTC-5 (rough EST, ignores DST)
        et = timezone(timedelta(hours=-5))
    now_et = datetime.now(et)
    # Weekday 0=Mon, 4=Fri
    if now_et.weekday() > 4:
        return False
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_et <= market_close


def enter_slot(trade_client, opt_data_client, slot_id: str,
               slot_state: dict, regime: dict,
               portfolio_value: float, today: date) -> dict:
    """
    Open a new options position in this slot based on current regime.
    Only enters during market hours — options credit is unreliable otherwise.
    """
    # Check gate
    next_entry = slot_state.get("next_entry")
    if next_entry and date.fromisoformat(next_entry) > today:
        log.info(f"  Slot {slot_id}: gated until {next_entry}")
        return slot_state

    # Only enter during market hours
    if not is_market_open():
        log.info(f"  Slot {slot_id}: market closed — skipping entry, will retry next run")
        return slot_state

    spread_name = regime["spread"]
    params      = SPREAD_PARAMS[spread_name]
    S           = regime["price"]
    iv          = regime["iv"]
    T           = CONFIG["slot_dte"] / 365

    target_expiry = today + timedelta(days=CONFIG["slot_dte"])
    log.info(f"  Slot {slot_id}: entering {spread_name}  "
             f"regime={regime['trend']}+{regime['vol']}+{regime['atr']}  "
             f"target_expiry={target_expiry}")

    # ── Find strikes ──────────────────────────────────────────
    put_short_target = find_strike_by_delta(
        S, CONFIG["r"], iv, T, params["put_delta"], "put")
    put_width  = max(5.0, round(S * params["spread_pct"] * params["put_width_mult"]))
    put_long_target = put_short_target - put_width

    # ── Look up contracts ─────────────────────────────────────
    short_put = find_contract(trade_client, opt_data_client, "IWM", "put",
                              params["put_delta"], put_short_target, target_expiry)
    long_put  = find_contract(trade_client, opt_data_client, "IWM", "put",
                              params["put_delta"] * 0.5, put_long_target, target_expiry)

    if not short_put or not long_put:
        log.error(f"  Slot {slot_id}: could not find put contracts, skipping")
        return slot_state

    legs_info = [
        {"symbol": short_put["symbol"], "side": "sell"},
        {"symbol": long_put["symbol"],  "side": "buy"},
    ]
    leg_symbols = [short_put["symbol"], long_put["symbol"]]
    leg_sides   = [True, False]   # True = short
    net_credit  = short_put["mid"] - long_put["mid"]
    call_short  = None; call_long = None

    # ── Call side for condors ─────────────────────────────────
    if params["call_delta"] > 0:
        call_width = max(5.0, round(S * params["spread_pct"]))
        call_short_target = find_strike_by_delta(
            S, CONFIG["r"], iv, T, params["call_delta"], "call")
        call_long_target = call_short_target + call_width

        short_call = find_contract(trade_client, opt_data_client, "IWM", "call",
                                   params["call_delta"], call_short_target, target_expiry)
        long_call  = find_contract(trade_client, opt_data_client, "IWM", "call",
                                   params["call_delta"] * 0.5, call_long_target, target_expiry)

        if short_call and long_call:
            legs_info  += [{"symbol": short_call["symbol"], "side": "sell"},
                           {"symbol": long_call["symbol"],  "side": "buy"}]
            leg_symbols += [short_call["symbol"], long_call["symbol"]]
            leg_sides   += [True, False]
            net_credit  += short_call["mid"] - long_call["mid"]
            call_short   = short_call
            call_long    = long_call
        else:
            log.warning(f"  Slot {slot_id}: call contracts not found, using put spread only")

    if net_credit <= 0:
        log.warning(f"  Slot {slot_id}: zero/negative credit ${net_credit:.2f}, skipping")
        return slot_state

    # ── Size ──────────────────────────────────────────────────
    max_loss   = put_width - net_credit
    contracts  = compute_contracts(portfolio_value, max_loss)
    if contracts < 1:
        log.warning(f"  Slot {slot_id}: insufficient capital for 1 contract")
        return slot_state

    # ── Submit ────────────────────────────────────────────────
    order_id = submit_spread(trade_client, spread_name, legs_info,
                             contracts, net_credit)
    if not order_id:
        # Order failed — gate slot until tomorrow so we retry with fresh quotes
        log.warning(f"  Slot {slot_id}: order failed, retrying next market session")
        slot_state["next_entry"] = (today + timedelta(days=1)).isoformat()
        return slot_state

    # ── Record state ──────────────────────────────────────────
    slot_state["position"] = {
        "entry_date":       today.isoformat(),
        "expiry":           (short_put["expiry"] if hasattr(short_put["expiry"],'isoformat')
                             else str(short_put["expiry"])),
        "spread":           spread_name,
        "trend_regime":     regime["trend"],
        "vol_regime":       regime["vol"],
        "atr_regime":       regime["atr"],
        "contracts":        contracts,
        "credit_received":  round(net_credit, 4),
        "max_loss":         round(max_loss, 4),
        "leg_symbols":      leg_symbols,
        "leg_sides":        leg_sides,
        "order_id":         order_id,
        "underlying_px":    S,
    }
    slot_state["next_entry"] = None

    # Log trade
    log_trade({
        "date":           today.isoformat(),
        "action":         "open",
        "slot":           slot_id,
        "spread":         spread_name,
        "trend":          regime["trend"],
        "vol":            regime["vol"],
        "atr":            regime["atr"],
        "contracts":      contracts,
        "credit":         round(net_credit, 4),
        "max_loss":       round(max_loss, 4),
        "underlying_px":  round(S, 2),
        "order_id":       order_id,
    })

    log.info(f"  Slot {slot_id}: opened {spread_name}  "
             f"{contracts} contracts  credit=${net_credit:.2f}  "
             f"max_loss=${max_loss:.2f}")
    return slot_state


# ══════════════════════════════════════════════════════════════
#  MAIN DAILY RUNNER
# ══════════════════════════════════════════════════════════════

def run():
    today = date.today()
    log.info("=" * 60)
    log.info(f"VRP Live Trader — {today}  {'PAPER' if PAPER else 'LIVE'}")
    log.info("=" * 60)

    # ── Initialise Alpaca clients ─────────────────────────────
    trade_client    = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    stock_data      = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    opt_data        = OptionHistoricalDataClient(API_KEY, SECRET_KEY)

    # ── Account value ─────────────────────────────────────────
    account = trade_client.get_account()
    portfolio_value = float(account.portfolio_value)
    log.info(f"Portfolio value: ${portfolio_value:,.2f}  "
             f"{'Paper' if PAPER else 'Live'}")

    # ── Load state ────────────────────────────────────────────
    state = load_state()
    if state["initial_capital"] is None:
        state["initial_capital"] = portfolio_value
        log.info(f"First run — initial capital set to ${portfolio_value:,.2f}")

    # ── Slot B gate: stagger 7 days after Slot A ──────────────
    slot_a_state = state["slots"]["A"]
    slot_b_state = state["slots"]["B"]

    # If Slot B has never been opened, set gate to today + stagger
    if (slot_b_state.get("next_entry") is None and
            slot_b_state.get("position") is None):
        slot_b_start = (today + timedelta(days=CONFIG["stagger_days"])).isoformat()
        slot_b_state["next_entry"] = slot_b_start
        log.info(f"Slot B first entry gated to {slot_b_start}")

    # ── Regime detection ──────────────────────────────────────
    log.info("Computing regime...")
    regime = get_regime(CONFIG)
    log.info(f"  Trend={regime['trend']}  Vol={regime['vol']}  "
             f"ATR={regime['atr']}  →  {regime['spread']}")
    log.info(f"  IWM=${regime['price']:.2f}  VIX={regime['vix']:.1f}  "
             f"HV={regime['hv']*100:.1f}%  IV={regime['iv']*100:.1f}%")

    # ── Manage open positions ─────────────────────────────────
    log.info("Managing open positions...")
    for slot_id, slot_st in [("A", slot_a_state), ("B", slot_b_state)]:
        if slot_st.get("position"):
            updated = manage_slot(trade_client, opt_data,
                                  slot_id, slot_st, portfolio_value, today)
            state["slots"][slot_id] = updated
        else:
            log.info(f"  Slot {slot_id}: no open position")

    # ── Open new positions where slots are empty ──────────────
    log.info("Checking for entry opportunities...")
    for slot_id, slot_st in [("A", slot_a_state), ("B", slot_b_state)]:
        if slot_st.get("position") is None:
            updated = enter_slot(trade_client, opt_data,
                                 slot_id, slot_st, regime,
                                 portfolio_value, today)
            state["slots"][slot_id] = updated

    # ── Manage SPY allocation ─────────────────────────────────
    log.info("Checking SPY allocation...")
    manage_spy_allocation(trade_client, stock_data, state, portfolio_value)

    # ── Save state ────────────────────────────────────────────
    save_state(state)
    log.info("State saved.")

    # ── Summary ───────────────────────────────────────────────
    log.info("─" * 60)
    for slot_id in ["A","B"]:
        pos = state["slots"][slot_id].get("position")
        if pos:
            expiry = pos["expiry"]
            dte = (date.fromisoformat(str(expiry)) - today).days
            log.info(f"  Slot {slot_id}: OPEN — {pos['spread']}  "
                     f"{pos['contracts']} contracts  DTE={dte}  "
                     f"credit=${pos['credit_received']:.2f}")
        else:
            nxt = state["slots"][slot_id].get("next_entry","immediately")
            log.info(f"  Slot {slot_id}: EMPTY  next entry: {nxt}")
    log.info(f"  SPY shares: {state['spy_shares']:.0f}")
    log.info("Done.")


if __name__ == "__main__":
    run()
