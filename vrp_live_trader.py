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
    # No SPY — all-options portfolio
    # IWM: 17.5% per slot × 2 slots = 35%
    # XBI: 22.5% per slot × 2 slots = 45%
    # Cash reserve: 20% (buffer + future XLE allocation)
    "iwm_slot_risk":   0.175,  # 17.5% per slot  (84% of half-Kelly)
    "xbi_slot_risk":   0.225,  # 22.5% per slot  (62% of half-Kelly)

    # ── Options parameters ────────────────────────────────────
    "slot_dte":        14,     # target DTE at entry
    "dte_tolerance":   3,      # accept contracts within ±3 DTE of target
    "exit_dte":        3,      # force-close at ≤3 DTE
    "min_hold_days":   3,      # minimum days before profit target fires
    "profit_target":   0.65,   # close at 65% of credit — optimal by Sharpe (19yr backtest)
                               # improves Sharpe 0.50→1.46 at live sizing, MaxDD -3.7%→-1.2%
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

# ── Regime map — optimised via 44-ticker cross-validation (2005-2024) ──────
# Changes vs original (9 of 18 regimes):
#   bullish+high+expanding:   skewed_put  → iron_condor  (same WR, +P&L)
#   bearish+high+contracting: skewed_put  → bull_put     (+2.6pp WR, +$23K)
#   bearish+high+expanding:   skewed_put  → bull_put     (+2.9pp WR, +$10K)
#   bearish+mid+contracting:  skewed_put  → iron_condor  (same WR, +$132K)
#   bearish+low+contracting:  skewed_put  → iron_condor  (same WR, +$124K)
#   bearish+low+expanding:    skewed_put  → bull_put     (+7.0pp WR, +$13K)
#   neutral+high+contracting: skewed_put  → bull_put     (+2.7pp WR, +$8K)
#   neutral+high+expanding:   skewed_put  → bull_put     (+7.0pp WR, +$12K)
#   neutral+mid+contracting:  long_dte    → iron_condor  (same WR, +$7K)
# Key finding: skewed_put systematically underperforms iron_condor on P&L
# in most regimes. bull_put is reserved for bearish/high-vol where directional
# put premium genuinely dominates. Does NOT affect any open positions —
# only applies at the moment of entering a new trade.
REGIME_MAP = {
    # Bullish regimes — long_dte_condor optimal in low/mid vol
    ("bullish","low","contracting"): "long_dte_condor",
    ("bullish","low","expanding"):   "long_dte_condor",
    ("bullish","low","unknown"):     "long_dte_condor",
    ("bullish","mid","contracting"): "long_dte_condor",
    ("bullish","mid","expanding"):   "long_dte_condor",
    ("bullish","mid","unknown"):     "long_dte_condor",
    ("bullish","high","contracting"):"skewed_put",      # low frequency, keep conservative
    ("bullish","high","expanding"):  "iron_condor",     # CHANGED: same WR, better P&L
    ("bullish","high","unknown"):    "skewed_put",

    # Bearish regimes — iron_condor or bull_put, skewed_put retired
    ("bearish","high","contracting"):"bull_put",        # CHANGED: +2.6pp WR, +$23K P&L
    ("bearish","high","expanding"):  "bull_put",        # CHANGED: +2.9pp WR, +$10K P&L
    ("bearish","high","unknown"):    "bull_put",
    ("bearish","mid","contracting"): "iron_condor",     # CHANGED: same WR, +$132K P&L
    ("bearish","mid","expanding"):   "iron_condor",     # unchanged — already optimal
    ("bearish","mid","unknown"):     "iron_condor",
    ("bearish","low","contracting"): "iron_condor",     # CHANGED: same WR, +$124K P&L
    ("bearish","low","expanding"):   "bull_put",        # CHANGED: +7.0pp WR, +$13K P&L
    ("bearish","low","unknown"):     "iron_condor",

    # Neutral regimes — iron_condor default, bull_put for high-vol
    ("neutral","high","contracting"):"bull_put",        # CHANGED: +2.7pp WR, +$8K P&L
    ("neutral","high","expanding"):  "bull_put",        # CHANGED: +7.0pp WR, +$12K P&L
    ("neutral","high","unknown"):    "bull_put",
    ("neutral","mid","contracting"): "iron_condor",     # CHANGED: long_dte → iron_condor
    ("neutral","mid","expanding"):   "iron_condor",     # unchanged — already optimal
    ("neutral","mid","unknown"):     "iron_condor",
    ("neutral","low","contracting"): "long_dte_condor", # unchanged — highest volume regime
    ("neutral","low","expanding"):   "long_dte_condor", # unchanged — iron_condor ties it
    ("neutral","low","unknown"):     "long_dte_condor",

    # Unknown/fallback
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
            "IWM_A": {"position": None, "next_entry": None},
            "IWM_B": {"position": None, "next_entry": None},
            "XBI_A": {"position": None, "next_entry": None},
            "XBI_B": {"position": None, "next_entry": None},
        },
        "initial_capital": None,
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


def save_spread_diagram(slot_id: str, spread_name: str, legs_info: list,
                        short_put: dict, long_put: dict,
                        short_call: dict, long_call: dict,
                        S: float, net_credit: float, max_loss: float,
                        contracts: int, today: date, regime: dict,
                        expiry_str: str):
    """
    Generate and save a spread payoff diagram as a PNG.
    Saved to: diagrams/YYYY-MM-DD_SlotID_spread.png
    Also appended to spread_log.csv for a running record.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        log.warning("  matplotlib not installed — skipping diagram (pip install matplotlib)")
        return

    # ── Collect all strikes ───────────────────────────────────
    legs = []
    for info, is_short, label in [
        (short_put, True,  "Short Put"),
        (long_put,  False, "Long Put"),
        (short_call,True,  "Short Call"),
        (long_call, False, "Long Call"),
    ]:
        if info:
            legs.append({"strike": info["strike"], "type": info["type"],
                         "is_short": is_short, "label": label,
                         "mid": info.get("mid", 0)})

    strikes = [l["strike"] for l in legs if l["strike"] > 0]
    if not strikes:
        log.warning("  No valid strikes — skipping diagram")
        return

    lo = min(strikes); hi = max(strikes)
    pad = max((hi - lo) * 1.5, S * 0.12)
    px_range = np.linspace(lo - pad, hi + pad, 500)

    # ── P&L at expiry ─────────────────────────────────────────
    def intrinsic(px, strike, opt_type):
        if opt_type == "put":  return max(strike - px, 0)
        if opt_type == "call": return max(px - strike, 0)
        return 0.0

    pnl = np.array([
        (net_credit + sum(
            (-intrinsic(px, l["strike"], l["type"]) if l["is_short"]
             else  intrinsic(px, l["strike"], l["type"]))
            for l in legs
        )) * contracts * 100
        for px in px_range
    ])

    # ── Plot ──────────────────────────────────────────────────
    G = "#1D9E75"; R = "#E24B4A"; B = "#378ADD"; A = "#EF9F27"
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    for spine in ax.spines.values():
        spine.set_color("#2d3150")
    ax.tick_params(colors="#7b82a0")
    ax.xaxis.label.set_color("#7b82a0")
    ax.yaxis.label.set_color("#7b82a0")

    # Fill profitable / loss zones
    ax.fill_between(px_range, pnl, 0,
                    where=pnl >= 0, alpha=0.18, color=G, linewidth=0)
    ax.fill_between(px_range, pnl, 0,
                    where=pnl < 0,  alpha=0.18, color=R, linewidth=0)

    # Payoff line — colour by sign
    for i in range(len(px_range)-1):
        col = G if pnl[i] >= 0 else R
        ax.plot(px_range[i:i+2], pnl[i:i+2], color=col, lw=2.2)

    # Zero line
    ax.axhline(0, color="#2d3150", lw=1.0, zorder=1)

    # Current price line
    ax.axvline(S, color=A, lw=1.5, ls="--", alpha=0.85, label=f"Entry ${S:.2f}")

    # Strike markers
    colors_map = {(True,"put"): R, (False,"put"): G,
                  (True,"call"): R, (False,"call"): G}
    for l in legs:
        if l["strike"] <= 0: continue
        c = colors_map.get((l["is_short"], l["type"]), "#888")
        ax.axvline(l["strike"], color=c, lw=1.0, ls=":", alpha=0.7)
        pnl_at_strike = float(np.interp(l["strike"], px_range, pnl))
        ax.annotate(
            f"{l['label']}\n${l['strike']:.0f}",
            xy=(l["strike"], pnl_at_strike),
            xytext=(0, 18 if l["is_short"] else -28),
            textcoords="offset points",
            ha="center", fontsize=7.5, color=c,
            arrowprops=dict(arrowstyle="-", color=c, alpha=0.5, lw=0.8),
        )

    # Credit / max-loss labels
    max_profit = net_credit * contracts * 100
    max_loss_d = -max_loss   * contracts * 100
    ax.axhline(max_profit, color=G, lw=0.8, ls=":", alpha=0.5)
    ax.axhline(max_loss_d, color=R, lw=0.8, ls=":", alpha=0.5)
    ax.text(px_range[-1], max_profit + abs(max_profit)*0.04,
            f"Max profit ${max_profit:,.0f}", color=G,
            fontsize=8, ha="right", va="bottom")
    ax.text(px_range[-1], max_loss_d - abs(max_loss_d)*0.04,
            f"Max loss  ${max_loss_d:,.0f}", color=R,
            fontsize=8, ha="right", va="top")

    # Labels
    ax.set_xlabel("Underlying price at expiry", color="#7b82a0", fontsize=9)
    ax.set_ylabel("P&L ($)", color="#7b82a0", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _:
        f"${y:,.0f}" if y >= 0 else f"-${abs(y):,.0f}"))
    ax.grid(True, color="#2d3150", linewidth=0.5, alpha=0.6)

    spread_display = spread_name.replace("_", " ").title()
    title = (f"Slot {slot_id} — {spread_display}  |  "
             f"{regime['trend'].capitalize()}+{regime['vol']}  |  "
             f"Credit ${net_credit:.4f}  ×  {contracts} contracts  |  "
             f"Expires {expiry_str}")
    ax.set_title(title, color="#e8eaf0", fontsize=9, pad=10)

    ax.legend(fontsize=8, framealpha=0.2, labelcolor="#e8eaf0",
              facecolor="#1a1d27", edgecolor="#2d3150")

    # ── Save ──────────────────────────────────────────────────
    diagrams_dir = Path("diagrams")
    diagrams_dir.mkdir(exist_ok=True)
    filename = f"{today.isoformat()}_{slot_id}_{spread_name}.png"
    filepath = diagrams_dir / filename

    fig.tight_layout()
    fig.savefig(filepath, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Diagram saved: {filepath}")

    # ── Append to spread_log.csv ───────────────────────────────
    log_path = Path("spread_log.csv")
    row = {
        "date":          today.isoformat(),
        "slot":          slot_id,
        "spread":        spread_name,
        "trend_regime":  regime["trend"],
        "vol_regime":    regime["vol"],
        "atr_regime":    regime["atr"],
        "underlying_px": round(S, 2),
        "net_credit":    round(net_credit, 4),
        "max_loss":      round(max_loss, 4),
        "contracts":     contracts,
        "expiry":        expiry_str,
        "diagram_file":  str(filepath),
        "legs":          " | ".join(
            f"{'S' if l['is_short'] else 'L'} {l['type']} ${l['strike']:.0f}"
            for l in legs if l["strike"] > 0
        ),
    }
    pd.DataFrame([row]).to_csv(
        log_path, mode="a", header=not log_path.exists(), index=False
    )
    log.info(f"  Spread logged: spread_log.csv")


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

def get_regime(cfg: dict, ticker: str = "IWM") -> dict:
    """
    Compute regime signals for the given ticker (IWM or XBI).
    XBI uses the same equity regime map — biotech follows risk-on/off
    cycles driven by the same VIX and SMA signals as small-caps.
    Returns: {"trend", "vol", "atr", "hv", "iv", "vix", "price", "spread"}
    """
    lookback_days = cfg["ivr_lookback"] + 60
    end   = datetime.today()
    start = end - timedelta(days=lookback_days * 1.5)

    underlying = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)["Close"].squeeze()
    vix = yf.download("^VIX", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"].squeeze()

    # Align
    df = pd.DataFrame({"px": underlying, "vix": vix}).ffill().dropna()
    px  = df["px"]
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

# Market-impact caps by ticker (0.1% of avg daily options volume).
# These only activate at very large portfolio sizes — they are a safety
# rail against bugs, not a growth limiter for normal account sizes.
#   IWM activates at ~$1.4M portfolio  (500K daily vol × 0.1%)
#   XBI activates at ~$160K portfolio  (75K daily vol × 0.1%)
#   Unknown tickers: conservative 50-contract cap
MARKET_IMPACT_CAP = {
    "IWM":  500,   # 0.1% of ~500K daily options volume
    "XBI":   75,   # 0.1% of ~75K daily options volume
    "QQQ":  500,
    "SPY":  500,
    "AAPL": 200,
    "XLE":  150,
}
DEFAULT_CONTRACT_CAP = 50


def compute_contracts(portfolio_value: float, max_loss_per_spread: float,
                      slot_risk: float = None, ticker: str = "IWM") -> int:
    """
    Number of contracts = floor(slot_budget / max_loss_per_contract).

    No hard 10-contract cap — contracts scale naturally with portfolio so
    the strategy compounds without an artificial growth ceiling.

    A generous ticker-specific market-impact cap acts as a safety rail
    against code bugs, not a growth limiter. At any realistic retail
    portfolio size the formula-driven count will be well below these caps.

    Contract counts at typical portfolio sizes:
      $10K:  IWM=3   XBI=4
      $25K:  IWM=9   XBI=11
      $50K:  IWM=18  XBI=23
      $100K: IWM=36  XBI=46
      $250K: IWM=91  XBI=75 (XBI market-impact cap)
    """
    if max_loss_per_spread <= 0:
        return 0
    risk   = slot_risk if slot_risk else CONFIG["iwm_slot_risk"]
    budget = portfolio_value * risk
    n      = int(budget / (max_loss_per_spread * 100))
    cap    = MARKET_IMPACT_CAP.get(ticker, DEFAULT_CONTRACT_CAP)
    return max(1, min(n, cap))


# ══════════════════════════════════════════════════════════════
#  SPY ALLOCATION MANAGER — REMOVED
#  Portfolio is now 100% options (IWM + XBI) with 20% cash buffer.
#  SPY buy-and-hold replaced by higher-edge options on two tickers.
# ══════════════════════════════════════════════════════════════

def manage_spy_allocation(trade_client, data_client, state: dict,
                          portfolio_value: float):
    """Stub — SPY allocation removed. All capital deployed in IWM+XBI options."""
    return  # no-op
    """
    Ensure ~60% of portfolio is in SPY.
    On first run: buys the initial SPY position.
    Subsequent runs: rebalances if >5% off target.
    """
    target_value = portfolio_value * 0.60   # legacy reference, function is no-op

    # Always read actual SPY shares from Alpaca — never trust state.json alone.
    # Prevents double-buying when state.json is stale between runs.
    try:
        positions = trade_client.get_all_positions()
        actual_shares = 0.0
        for pos in positions:
            if pos.symbol == "SPY":
                actual_shares = float(pos.qty)
                break
        state["spy_shares"] = actual_shares
        log.info(f"  SPY actual shares from Alpaca: {actual_shares:.0f}")
    except Exception as e:
        log.warning(f"  Could not read SPY position from Alpaca: {e}")
        actual_shares = state.get("spy_shares", 0.0)

    current_shares = actual_shares

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
#  DAILY DIAGRAM UPDATE  (called every morning for open slots)
# ══════════════════════════════════════════════════════════════

def update_daily_diagrams(opt_data_client, state: dict, today: date):
    """
    Redraw the spread diagram for every open slot using live option
    quotes + current DTE.  Shows where the position stands RIGHT NOW,
    not just what it looks like at expiry.

    Saved to: diagrams/YYYY-MM-DD_SlotID_live.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import norm
        from scipy.optimize import brentq
    except ImportError:
        log.warning("  matplotlib/scipy not installed — skipping daily diagrams")
        return

    def bs(S, K, T, r, sig, opt_type):
        if T <= 0:
            return max(S - K, 0) if opt_type == "call" else max(K - S, 0)
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * math.sqrt(T))
            d2 = d1 - sig * math.sqrt(T)
            if opt_type == "call":
                return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        except Exception:
            return 0.0

    G = "#1D9E75"; R = "#E24B4A"; B = "#378ADD"; A = "#EF9F27"
    diagrams_dir = Path("diagrams")
    diagrams_dir.mkdir(exist_ok=True)

    for slot_id, slot_st in state["slots"].items():
        pos = slot_st.get("position")
        if not pos:
            continue

        # ── Get live quotes for all legs ──────────────────────
        syms     = pos.get("leg_symbols", [])
        sides    = pos.get("leg_sides", [])
        credit   = float(pos.get("credit_received", 0))
        ml       = float(pos.get("max_loss", 0))
        contr    = int(pos.get("contracts", 1))
        expiry   = date.fromisoformat(str(pos["expiry"]))
        dte      = max((expiry - today).days, 0)
        entry_px = float(pos.get("underlying_px", 0))
        spread   = pos.get("spread", "unknown")

        # Infer ticker from leg symbols
        ticker = "IWM"
        for t in ["IWM", "XBI", "QQQ", "SPY"]:
            if syms and syms[0].startswith(t):
                ticker = t; break

        # Get live quotes
        live_mids = {}
        live_px   = entry_px
        try:
            snaps = opt_data_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=syms)
            )
            for sym in syms:
                q = snaps[sym].latest_quote
                bid = float(q.bid_price); ask = float(q.ask_price)
                live_mids[sym] = (bid + ask) / 2 if bid > 0 or ask > 0 else 0.0
        except Exception as e:
            log.warning(f"  Could not get live quotes for {slot_id}: {e}")

        # Get live underlying price
        try:
            import yfinance as yf
            raw = yf.download(ticker, period="2d", auto_adjust=True,
                              progress=False)["Close"].squeeze()
            live_px = float(raw.dropna().iloc[-1])
        except Exception:
            pass

        # Current spread value from live quotes
        current_cost = 0.0
        for sym, is_short in zip(syms, sides):
            mid = live_mids.get(sym, 0.0)
            current_cost += -mid if is_short else mid
        # current_cost is negative (it costs money to close short spreads)
        # P&L = credit_received + current_cost  (current_cost is negative = good)
        current_pnl_ps = credit + current_cost
        current_pnl_total = current_pnl_ps * contr * 100
        pct_captured = (current_pnl_ps / credit * 100) if credit > 0 else 0

        # Parse strikes from leg symbols
        legs = []
        for sym, is_short in zip(syms, sides):
            try:
                opt_type = "call" if "C" in sym[-10:] else "put"
                strike = int(sym[-8:]) / 1000
            except Exception:
                opt_type = "put"; strike = 0
            legs.append({"strike": strike, "type": opt_type, "is_short": is_short})

        # Estimate implied vol from live quotes (use first short leg)
        sig = 0.25  # fallback
        try:
            T = max(dte / 365, 1/365)
            for sym, is_short, leg in zip(syms, sides, legs):
                if is_short and leg["strike"] > 0 and live_px > 0:
                    target = live_mids.get(sym, 0)
                    if target > 0:
                        def obj(v):
                            return bs(live_px, leg["strike"], T, 0.04, v,
                                      leg["type"]) - target
                        sig = brentq(obj, 0.05, 3.0, xtol=0.001)
                    break
        except Exception:
            pass

        # ── Build P&L curves ──────────────────────────────────
        strikes = [l["strike"] for l in legs if l["strike"] > 0]
        if not strikes:
            continue
        lo = min(strikes); hi = max(strikes)
        pad = max((hi - lo) * 1.6, live_px * 0.12)
        px_range = np.linspace(lo - pad, hi + pad, 500)
        T_now    = max(dte / 365, 1/365)
        T_entry  = 14 / 365

        def spread_pnl_at(px_arr, T_val, iv):
            out = []
            for px in px_arr:
                val = credit
                for leg in legs:
                    k = leg["strike"]
                    if k <= 0: continue
                    p = bs(px, k, T_val, 0.04, iv, leg["type"])
                    val += -p if leg["is_short"] else p
                out.append(val * contr * 100)
            return np.array(out)

        pnl_expiry   = spread_pnl_at(px_range, 0,       sig)
        pnl_now      = spread_pnl_at(px_range, T_now,   sig)
        pnl_entry_iv = spread_pnl_at(px_range, T_entry, 0.22)

        # ── Plot ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")
        for sp in ax.spines.values():
            sp.set_color("#2d3150")
        ax.tick_params(colors="#7b82a0")

        # Fills
        ax.fill_between(px_range, pnl_now, 0, where=pnl_now >= 0,
                        alpha=0.15, color=G, linewidth=0)
        ax.fill_between(px_range, pnl_now, 0, where=pnl_now < 0,
                        alpha=0.15, color=R, linewidth=0)

        # Entry shape (dashed grey)
        ax.plot(px_range, pnl_entry_iv, color="#555577",
                lw=1.2, ls="--", label="At entry (14 DTE)")

        # Expiry shape (blue)
        ax.plot(px_range, pnl_expiry, color=B,
                lw=1.5, ls=":", label="At expiry (0 DTE)", alpha=0.7)

        # Live shape (green/red segmented)
        for i in range(len(px_range) - 1):
            c = G if pnl_now[i] >= 0 else R
            ax.plot(px_range[i:i+2], pnl_now[i:i+2], color=c, lw=2.4)

        # Zero line
        ax.axhline(0, color="#2d3150", lw=1.0)

        # Strike lines
        color_map = {(True,"put"):R,(False,"put"):G,(True,"call"):R,(False,"call"):G}
        for leg in legs:
            k = leg["strike"]
            if k <= 0: continue
            c = color_map.get((leg["is_short"], leg["type"]), "#888")
            ax.axvline(k, color=c, lw=0.9, ls=":", alpha=0.6)
            lbl = ("Short" if leg["is_short"] else "Long") + f" {leg['type'].title()}\n${k:.0f}"
            pv = float(np.interp(k, px_range, pnl_now))
            yoff = 22 if leg["is_short"] else -32
            ax.annotate(lbl, xy=(k, pv), xytext=(0, yoff),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=c,
                        arrowprops=dict(arrowstyle="-", color=c, alpha=0.4, lw=0.7))

        # Live price line (amber)
        ax.axvline(live_px, color=A, lw=1.8, ls="--", alpha=0.9,
                   label=f"Live ${live_px:.2f}")

        # Entry price line (faint)
        if abs(live_px - entry_px) > 0.5:
            ax.axvline(entry_px, color="#555577", lw=1.0, ls=":",
                       alpha=0.5, label=f"Entry ${entry_px:.2f}")

        # Max profit / loss markers
        mp = credit * contr * 100
        ml_d = -ml * contr * 100
        ax.axhline(mp, color=G, lw=0.7, ls=":", alpha=0.4)
        ax.axhline(ml_d, color=R, lw=0.7, ls=":", alpha=0.4)
        ax.text(px_range[-1], mp, f" Max profit ${mp:,.0f}", color=G,
                fontsize=8, va="bottom", ha="right")
        ax.text(px_range[-1], ml_d, f" Max loss ${ml_d:,.0f}", color=R,
                fontsize=8, va="top", ha="right")

        # ── Annotations ───────────────────────────────────────
        pnl_color = G if current_pnl_total >= 0 else R
        ax.text(0.02, 0.97,
                f"P&L today: {'+'if current_pnl_total>=0 else ''}"
                f"${current_pnl_total:,.0f}  ({pct_captured:.0f}% of credit)",
                transform=ax.transAxes, color=pnl_color,
                fontsize=9, va="top", fontweight="bold")
        ax.text(0.02, 0.90,
                f"DTE: {dte}  |  IV: {sig*100:.0f}%  |  "
                f"65% target: +${credit*0.65*contr*100:,.0f}",
                transform=ax.transAxes,
                color="#7b82a0", fontsize=8, va="top")

        ax.set_xlabel("Underlying price", color="#7b82a0", fontsize=9)
        ax.set_ylabel("P&L ($)", color="#7b82a0", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda y, _: f"${y:,.0f}" if y >= 0 else f"-${abs(y):,.0f}"))
        ax.grid(True, color="#2d3150", linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8, framealpha=0.15, labelcolor="#e8eaf0",
                  facecolor="#1a1d27", edgecolor="#2d3150", loc="upper right")

        spread_disp = spread.replace("_", " ").title()
        title = (f"Slot {slot_id} ({ticker}) — {spread_disp}  "
                 f"|  {dte} DTE  |  Credit ${credit:.4f} × {contr}c  "
                 f"|  Expires {expiry}")
        ax.set_title(title, color="#e8eaf0", fontsize=9, pad=10)

        fig.tight_layout()
        fpath = diagrams_dir / f"{today.isoformat()}_{slot_id}_live.png"
        fig.savefig(fpath, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Live diagram saved: {fpath}  "
                 f"(P&L {'+' if current_pnl_total>=0 else ''}${current_pnl_total:,.0f}, "
                 f"{pct_captured:.0f}% captured)")


# ══════════════════════════════════════════════════════════════
#  PORTFOLIO EQUITY CHART  (cumulative P&L from trade_log.csv)
# ══════════════════════════════════════════════════════════════

def update_portfolio_chart(portfolio_value: float, state: dict, today: date):
    """
    Build and save a daily equity curve chart from trade_log.csv plus
    today's live portfolio value from Alpaca.

    Saved to: diagrams/portfolio_equity.png  (overwritten daily)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
    except ImportError:
        log.warning("  matplotlib not installed — skipping portfolio chart")
        return

    log_path = Path(CONFIG["log_file"])
    initial  = state.get("initial_capital") or portfolio_value

    G = "#1D9E75"; R = "#E24B4A"; B = "#378ADD"; A = "#EF9F27"

    # ── Build equity curve from trade log ─────────────────────
    dates_list  = [date.fromisoformat(str(state.get("start_date", today)))]
    equity_list = [initial]

    if log_path.exists():
        try:
            log_df = pd.read_csv(log_path)
            log_df["date"] = pd.to_datetime(log_df["date"]).dt.date
            # Only closed trades have real P&L
            closed = log_df[log_df["action"] == "close"].copy() \
                if "action" in log_df.columns else log_df.copy()
            if "pnl" in closed.columns:
                closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0)
                closed = closed.sort_values("date")
                running = initial
                for _, row in closed.iterrows():
                    running += row["pnl"]
                    dates_list.append(row["date"])
                    equity_list.append(running)
        except Exception as e:
            log.warning(f"  Could not read trade log for portfolio chart: {e}")

    # Always include today's actual Alpaca portfolio value as the last point
    if dates_list[-1] != today:
        dates_list.append(today)
        equity_list.append(portfolio_value)
    else:
        equity_list[-1] = portfolio_value  # update today with live value

    # ── Plot ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1d27")
        for sp in ax.spines.values():
            sp.set_color("#2d3150")
        ax.tick_params(colors="#7b82a0")
        ax.grid(True, color="#2d3150", linewidth=0.4, alpha=0.5)

    import datetime as dt_mod
    date_nums = [dt_mod.datetime.combine(d, dt_mod.time()) for d in dates_list]

    eq = np.array(equity_list)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100

    total_pnl  = equity_list[-1] - initial
    total_ret  = total_pnl / initial * 100
    max_dd     = dd.min()
    n_days     = max((today - dates_list[0]).days, 1)
    cagr       = ((equity_list[-1] / initial) ** (365 / n_days) - 1) * 100 \
                 if n_days > 5 else 0

    # ── Equity curve (top panel) ──────────────────────────────
    ax1.fill_between(date_nums, eq, initial,
                     where=eq >= initial, alpha=0.15, color=G, linewidth=0)
    ax1.fill_between(date_nums, eq, initial,
                     where=eq < initial,  alpha=0.15, color=R, linewidth=0)
    ax1.plot(date_nums, eq, color=G if total_pnl >= 0 else R,
             lw=2.2, label="Portfolio value")
    ax1.axhline(initial, color="#555577", lw=1.0, ls="--",
                label=f"Initial ${initial:,.0f}", alpha=0.7)

    # Mark each closed trade
    if len(dates_list) > 2:
        for i in range(1, len(dates_list) - 1):
            pnl = equity_list[i] - equity_list[i-1]
            c   = G if pnl >= 0 else R
            ax1.plot(date_nums[i], equity_list[i], "o",
                     color=c, ms=5, zorder=5)

    # Latest value annotation
    ax1.annotate(
        f"${equity_list[-1]:,.0f}\n{total_ret:+.1f}%",
        xy=(date_nums[-1], equity_list[-1]),
        xytext=(-60, 20 if total_pnl >= 0 else -40),
        textcoords="offset points",
        color=G if total_pnl >= 0 else R,
        fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#7b82a0", lw=0.8)
    )

    ax1.set_ylabel("Portfolio value ($)", color="#7b82a0", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda y, _: f"${y:,.0f}"))
    ax1.legend(fontsize=8, framealpha=0.15, labelcolor="#e8eaf0",
               facecolor="#1a1d27", edgecolor="#2d3150")

    # Header stats
    stats_txt = (f"Total P&L: {'+'if total_pnl>=0 else ''}${total_pnl:,.0f}  |  "
                 f"Return: {total_ret:+.1f}%  |  "
                 f"Max DD: {max_dd:.1f}%  |  "
                 f"CAGR: {cagr:.0f}%  |  "
                 f"Trades: {max(len(dates_list)-2,0)}")
    ax1.set_title(f"VRP Portfolio Equity Curve  —  {today}\n{stats_txt}",
                  color="#e8eaf0", fontsize=9, pad=10)

    # ── Drawdown (bottom panel) ────────────────────────────────
    ax2.fill_between(date_nums, dd, 0,
                     where=dd <= 0, alpha=0.55, color=R, linewidth=0)
    ax2.plot(date_nums, dd, color=R, lw=1.5)
    ax2.axhline(0, color="#2d3150", lw=0.8)
    ax2.set_ylabel("Drawdown %", color="#7b82a0", fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax2.set_xlabel("Date", color="#7b82a0", fontsize=8)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout(h_pad=0.5)
    diagrams_dir = Path("diagrams")
    diagrams_dir.mkdir(exist_ok=True)
    fpath = diagrams_dir / "portfolio_equity.png"
    fig.savefig(fpath, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Portfolio chart saved: {fpath}  "
             f"(value=${portfolio_value:,.0f}, "
             f"PnL={'+'if total_pnl>=0 else ''}${total_pnl:,.0f})")


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
               portfolio_value: float, today: date,
               ticker: str = "IWM",
               cfg_override: dict = None) -> dict:
    """
    Open a new options position in this slot based on current regime.
    ticker: underlying ETF (IWM or XBI)
    cfg_override: allows per-slot risk_pct to differ from CONFIG
    """
    cfg = cfg_override if cfg_override else CONFIG
    # Check gate
    next_entry = slot_state.get("next_entry")
    if next_entry and date.fromisoformat(next_entry) > today:
        log.info(f"  Slot {slot_id}: gated until {next_entry}")
        return slot_state

    # Only enter during market hours
    if not is_market_open():
        log.info(f"  Slot {slot_id}: market closed — skipping entry, will retry next run")
        return slot_state

    # Safety: count already-open option legs for this ticker.
    # Prevents doubling if state.json was stale between simultaneous runs.
    try:
        open_positions = trade_client.get_all_positions()
        ticker_legs = sum(1 for p in open_positions
                         if p.symbol.startswith(ticker) and len(p.symbol) > len(ticker))
        max_legs = 8   # 2 slots × 4 legs per condor
        if ticker_legs >= max_legs:
            log.warning(f"  Slot {slot_id}: {ticker_legs} {ticker} legs open already — skipping")
            return slot_state
    except Exception:
        pass

    spread_name = regime["spread"]
    params      = SPREAD_PARAMS[spread_name]
    S           = regime["price"]
    iv          = regime["iv"]
    T           = (cfg if cfg_override else CONFIG)["slot_dte"] / 365

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
    short_put = find_contract(trade_client, opt_data_client, ticker, "put",
                              params["put_delta"], put_short_target, target_expiry)
    long_put  = find_contract(trade_client, opt_data_client, ticker, "put",
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

        short_call = find_contract(trade_client, opt_data_client, ticker, "call",
                                   params["call_delta"], call_short_target, target_expiry)
        long_call  = find_contract(trade_client, opt_data_client, ticker, "call",
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
    contracts  = compute_contracts(portfolio_value, max_loss,
                                   slot_risk=(cfg_override or CONFIG).get("risk_pct"),
                                   ticker=ticker)
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

    # Save spread diagram + append to spread_log.csv
    expiry_str = (short_put["expiry"].isoformat()
                  if hasattr(short_put["expiry"], "isoformat")
                  else str(short_put["expiry"]))
    save_spread_diagram(
        slot_id=slot_id,
        spread_name=spread_name,
        legs_info=legs_info,
        short_put=short_put,
        long_put=long_put,
        short_call=short_call if params["call_delta"] > 0 else None,
        long_call=long_call   if params["call_delta"] > 0 else None,
        S=S,
        net_credit=net_credit,
        max_loss=max_loss,
        contracts=contracts,
        today=today,
        regime=regime,
        expiry_str=expiry_str,
    )

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

    # ── Slot configuration ────────────────────────────────────
    # 4 slots: IWM_A, IWM_B (17.5% each), XBI_A, XBI_B (22.5% each)
    # Each ticker's B slot is staggered 7 days after its A slot.
    TICKER_SLOTS = [
        ("IWM_A", "IWM", CONFIG["iwm_slot_risk"]),
        ("IWM_B", "IWM", CONFIG["iwm_slot_risk"]),
        ("XBI_A", "XBI", CONFIG["xbi_slot_risk"]),
        ("XBI_B", "XBI", CONFIG["xbi_slot_risk"]),
    ]
    # Gate B slots on first run
    for _sid, _tkr, _ in TICKER_SLOTS:
        slot_st = state["slots"][_sid]
        if _sid.endswith("_B") and slot_st.get("next_entry") is None and slot_st.get("position") is None:
            gate = (today + timedelta(days=CONFIG["stagger_days"])).isoformat()
            slot_st["next_entry"] = gate
            log.info(f"Slot {_sid} ({_tkr}) first entry gated to {gate}")

    # ── Regime detection (one per ticker) ────────────────────
    log.info("Computing regimes...")
    regimes = {}
    for ticker in ["IWM", "XBI"]:
        r = get_regime(CONFIG, ticker)
        regimes[ticker] = r
        log.info(f"  {ticker}: Trend={r['trend']}  Vol={r['vol']}  ATR={r['atr']}  → {r['spread']}")
        log.info(f"    ${r['price']:.2f}  VIX={r['vix']:.1f}  HV={r['hv']*100:.1f}%")

    # ── Manage open positions ─────────────────────────────────
    log.info("Managing open positions...")
    for slot_id, ticker, slot_risk in TICKER_SLOTS:
        slot_st = state["slots"][slot_id]
        if slot_st.get("position"):
            updated = manage_slot(trade_client, opt_data,
                                  slot_id, slot_st, portfolio_value, today)
            state["slots"][slot_id] = updated
        else:
            log.info(f"  Slot {slot_id}: no open position")

    # ── Open new positions ────────────────────────────────────
    log.info("Checking entry opportunities...")
    for slot_id, ticker, slot_risk in TICKER_SLOTS:
        slot_st = state["slots"][slot_id]
        if slot_st.get("position") is None:
            regime = regimes[ticker]
            # Override risk_pct for this specific slot
            cfg_override = {**CONFIG, "risk_pct": slot_risk}
            updated = enter_slot(trade_client, opt_data,
                                 slot_id, slot_st, regime,
                                 portfolio_value, today,
                                 ticker=ticker,
                                 cfg_override=cfg_override)
            state["slots"][slot_id] = updated

    # ── Save state ────────────────────────────────────────────
    save_state(state)
    log.info("State saved.")

    # ── Summary ───────────────────────────────────────────────
    log.info("─" * 60)
    for slot_id, ticker, slot_risk in TICKER_SLOTS:
        pos = state["slots"][slot_id].get("position")
        if pos:
            try:
                dte = (date.fromisoformat(str(pos["expiry"])) - today).days
            except Exception:
                dte = "?"
            log.info(f"  Slot {slot_id} ({ticker} {slot_risk*100:.1f}%): OPEN — "
                     f"{pos['spread']}  {pos['contracts']} contracts  "
                     f"DTE={dte}  credit=${pos['credit_received']:.2f}")
        else:
            nxt = state["slots"][slot_id].get("next_entry","immediately")
            log.info(f"  Slot {slot_id} ({ticker} {slot_risk*100:.1f}%): EMPTY  next: {nxt}")
    log.info(f"  Portfolio: ${portfolio_value:,.0f}  |  Cash reserve ~20%")

    # ── Daily diagrams & portfolio chart ─────────────────────
    log.info("Generating daily live diagrams...")
    update_daily_diagrams(opt_data, state, today)

    log.info("Generating portfolio equity chart...")
    update_portfolio_chart(portfolio_value, state, today)

    log.info("Done.")


if __name__ == "__main__":
    run()
