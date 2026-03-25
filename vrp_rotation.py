# ============================================================
#  Options Rotation Backtester
#  Pure Python — yfinance + Black-Scholes
#
#  Philosophy (completely different from the signal-based v1-v4):
#
#    Instead of waiting for a "perfect" setup, this strategy opens
#    a new spread every ROTATION_DAYS regardless of market conditions.
#    Only one position is open at a time. The edge (if it exists) is
#    structural — selling options premium consistently over hundreds
#    of cycles, not timing entries.
#
#    This allows you to:
#      1. Test many spread configurations on the same timeline
#      2. Cross-tab performance by spread type × market regime
#      3. Identify which spreads work in which environments
#      4. Build toward a regime-conditional spread selector
#
#  Spread library (all configurable):
#    - iron_condor       — sell both sides, neutral
#    - bull_put          — sell put spread, bullish
#    - bear_call         — sell call spread, bearish
#    - wide_condor       — lower delta (0.15), more room
#    - tight_condor      — higher delta (0.30), more credit
#    - skewed_put        — asymmetric: wider put side (harvest skew)
#
#  Every trade is tagged with:
#    - Trend regime  (bullish / bearish / neutral — 20/50 SMA)
#    - Vol regime    (high / mid / low — VIX terciles)
#    - IVR           (rolling percentile of own HV)
#    - ATR regime    (expanding / contracting — 5d vs 20d ATR)
#
#  Install:
#    pip install yfinance numpy pandas scipy matplotlib seaborn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import date, timedelta
import math
import itertools

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

CONFIG = {
    "ticker":           "QQQ",
    "start_date":       "2018-01-01",
    "end_date":         "2024-01-01",
    "initial_capital":  100_000,

    # ── Rotation ─────────────────────────────────────────────
    # Open a new spread every N calendar days after the previous
    # one closes (or at startup). Only ONE position open at a time.
    "rotation_days":    28,

    # ── Spread to run ─────────────────────────────────────────
    # Set this to any key from SPREAD_LIBRARY below, or use
    # run_all_spreads() to backtest every spread in sequence.
    "active_spread":    "iron_condor",

    # ── Shared exit rules (apply to all spread types) ────────
    "profit_target":    0.50,       # Close at 50% of max profit
    "stop_loss_mult":   2.0,        # Close at 2× credit received
    "exit_dte":         7,          # Force-close within 7 DTE
    "min_hold_days":    4,          # Min days before profit target fires

    # ── Position sizing ───────────────────────────────────────
    "risk_pct":         0.02,       # Max 2% of portfolio at risk per trade
    "max_contracts":    20,

    # ── Pricing ───────────────────────────────────────────────
    "risk_free_rate":   0.04,
    "slippage_pct":     0.05,       # 5% of mid — widen fills conservatively
    "commission":       0.65,       # $ per contract per leg

    # ── Lookbacks for signals / regime tagging ────────────────
    "hv_lookback":      20,
    "ivr_lookback":     252,
    "trend_fast":       20,
    "trend_slow":       50,
    "atr_fast":         5,
    "atr_slow":         20,

    # ── VRP factor: IV = HV × vrp_factor ─────────────────────
    # QQQ IV historically runs ~15-20% above realized vol
    "vrp_factor":       1.18,
}


# ══════════════════════════════════════════════════════════════
#  SPREAD LIBRARY
#  Each spread defines its own structure. Parameters:
#    dte         — days to expiry at entry
#    put_delta   — short put delta  (0 if no put leg)
#    call_delta  — short call delta (0 if no call leg)
#    spread_pct  — width as % of underlying price
#    put_width_mult  — put spread width multiplier vs call side
#                      (>1 = wider put side, harvests put skew)
# ══════════════════════════════════════════════════════════════

SPREAD_LIBRARY = {
    "iron_condor": {
        "description": "Symmetric iron condor — 0.20 delta both sides",
        "dte":          35,
        "put_delta":    0.20,
        "call_delta":   0.20,
        "spread_pct":   0.03,
        "put_width_mult": 1.0,
    },
    "wide_condor": {
        "description": "Wide iron condor — 0.15 delta, more breathing room",
        "dte":          35,
        "put_delta":    0.15,
        "call_delta":   0.15,
        "spread_pct":   0.03,
        "put_width_mult": 1.0,
    },
    "tight_condor": {
        "description": "Tight iron condor — 0.30 delta, more credit",
        "dte":          35,
        "put_delta":    0.30,
        "call_delta":   0.30,
        "spread_pct":   0.03,
        "put_width_mult": 1.0,
    },
    "skewed_put": {
        "description": "Skewed condor — wider put side to harvest put skew premium",
        "dte":          35,
        "put_delta":    0.20,
        "call_delta":   0.20,
        "spread_pct":   0.03,
        "put_width_mult": 1.5,   # put wing 1.5× wider than call wing
    },
    "bull_put": {
        "description": "Bull put spread — sell put side only, bullish bias",
        "dte":          35,
        "put_delta":    0.25,
        "call_delta":   0.0,     # no call leg
        "spread_pct":   0.03,
        "put_width_mult": 1.0,
    },
    "bear_call": {
        "description": "Bear call spread — sell call side only, bearish bias",
        "dte":          35,
        "put_delta":    0.0,     # no put leg
        "call_delta":   0.25,
        "spread_pct":   0.03,
        "put_width_mult": 1.0,
    },
    "short_dte_condor": {
        "description": "Fast-decay condor — 21 DTE, higher theta/day",
        "dte":          21,
        "put_delta":    0.20,
        "call_delta":   0.20,
        "spread_pct":   0.025,
        "put_width_mult": 1.0,
    },
    "long_dte_condor": {
        "description": "Slow condor — 45 DTE, more time for recovery",
        "dte":          45,
        "put_delta":    0.20,
        "call_delta":   0.20,
        "spread_pct":   0.035,
        "put_width_mult": 1.0,
    },
    "high_credit_condor": {
        "description": "Max credit condor — 0.30 delta, 2% spread width",
        "dte":          35,
        "put_delta":    0.30,
        "call_delta":   0.30,
        "spread_pct":   0.02,
        "put_width_mult": 1.0,
    },
}


# ══════════════════════════════════════════════════════════════
#  BLACK-SCHOLES ENGINE
# ══════════════════════════════════════════════════════════════

def bs_price(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0, K - S) if option_type == "put" else max(0, S - K)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "put":
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    except Exception:
        return 0.0

def bs_delta(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return (norm.cdf(d1) - 1.0) if option_type == "put" else norm.cdf(d1)
    except Exception:
        return 0.0

def find_strike_by_delta(S, r, sigma, T, target_delta, option_type="put"):
    if T <= 0 or sigma <= 0:
        offset = S * target_delta
        return S - offset if option_type == "put" else S + offset
    try:
        return brentq(
            lambda K: abs(bs_delta(S, K, T, r, sigma, option_type)) - target_delta,
            S * 0.40, S * 1.60, xtol=0.01
        )
    except ValueError:
        offset = S * target_delta * 1.5
        return S - offset if option_type == "put" else S + offset


# ══════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class Leg:
    option_type:  str
    strike:       float
    expiry:       date
    is_short:     bool
    entry_price:  float

@dataclass
class Trade:
    # Identity
    key:            str
    spread_name:    str
    entry_date:     date
    expiry:         date

    # Structure
    legs:           list
    credit:         float       # net credit per share
    max_loss:       float       # max loss per share
    contracts:      int

    # Regime tags at entry
    trend_regime:   str         # bullish / bearish / neutral
    vol_regime:     str         # high / mid / low
    ivr:            float       # 0–1
    atr_regime:     str         # expanding / contracting
    underlying_px:  float
    iv_at_entry:    float
    hv_at_entry:    float
    vrp_at_entry:   float       # iv - hv

    # Short-strike references for delta monitoring
    put_short_strike:  float = 0.0
    call_short_strike: float = 0.0

    # Outcome (filled on close)
    close_date:     Optional[date] = None
    close_reason:   str = ""
    realized_pnl:   float = 0.0
    days_held:      int = 0


# ══════════════════════════════════════════════════════════════
#  BACKTESTER
# ══════════════════════════════════════════════════════════════

class RotationBacktester:
    def __init__(self, config: dict, spread_name: str):
        self.cfg        = config
        self.spread     = SPREAD_LIBRARY[spread_name]
        self.spread_name = spread_name
        self.r          = config["risk_free_rate"]

        self.portfolio_value = config["initial_capital"]
        self.cash            = config["initial_capital"]
        self.position        = None          # at most one open trade
        self.closed_trades   = []
        self.equity_curve    = []
        self.next_entry_date = None          # calendar gate

    # ── Main loop ─────────────────────────────────────────────

    def run(self, closes: pd.DataFrame, signals: dict) -> dict:
        warmup = self.cfg["ivr_lookback"] + 5

        for i, today in enumerate(closes.index):
            if i < warmup:
                self.equity_curve.append((today, self.portfolio_value))
                continue

            S         = closes.loc[today, self.cfg["ticker"]]
            vix_today = closes.loc[today, "VIX"]
            sig       = {k: v.loc[today] for k, v in signals.items()
                         if today in v.index}

            # Step 1 — manage open position
            if self.position:
                self._manage(today, S, vix_today, sig)

            # Step 2 — mark portfolio
            self._mark(today, S, vix_today, sig)
            self.equity_curve.append((today, self.portfolio_value))

            # Step 3 — open new position if rotation gate is clear
            if self.position is None:
                if self.next_entry_date is None or today >= self.next_entry_date:
                    self._enter(today, S, vix_today, sig, closes)

        # Force-close any remaining position
        if self.position is not None and not closes.empty:
            last = closes.index[-1]
            S    = closes.loc[last, self.cfg["ticker"]]
            vix  = closes.loc[last, "VIX"]
            self._force_close(last, S, vix, "end_of_backtest")

        return self._build_results()

    # ── Entry ─────────────────────────────────────────────────

    def _enter(self, today, S, vix_today, sig, closes):
        sp    = self.spread
        T     = sp["dte"] / 365
        hv    = sig.get("hv", 0.20)
        if pd.isna(hv) or hv <= 0:
            hv = 0.20
        iv    = hv * self.cfg["vrp_factor"]

        base_width     = max(5.0, round(S * sp["spread_pct"]))
        put_width      = round(base_width * sp["put_width_mult"])
        call_width     = base_width

        legs = []
        credit = 0.0
        put_short_K = call_short_K = 0.0

        # ── Put legs ─────────────────────────────────────────
        if sp["put_delta"] > 0:
            pk = find_strike_by_delta(S, self.r, iv, T, sp["put_delta"], "put")
            pk = round(pk)
            pl = pk - put_width
            sp_p = bs_price(S, pk, T, self.r, iv, "put")
            lp_p = bs_price(S, pl, T, self.r, iv, "put")
            credit += sp_p - lp_p
            legs += [
                Leg("put", pk, today + timedelta(days=sp["dte"]), True,  sp_p),
                Leg("put", pl, today + timedelta(days=sp["dte"]), False, lp_p),
            ]
            put_short_K = pk

        # ── Call legs ─────────────────────────────────────────
        if sp["call_delta"] > 0:
            ck = find_strike_by_delta(S, self.r, iv, T, sp["call_delta"], "call")
            ck = round(ck)
            cl = ck + call_width
            sc_p = bs_price(S, ck, T, self.r, iv, "call")
            lc_p = bs_price(S, cl, T, self.r, iv, "call")
            credit += sc_p - lc_p
            legs += [
                Leg("call", ck, today + timedelta(days=sp["dte"]), True,  sc_p),
                Leg("call", cl, today + timedelta(days=sp["dte"]), False, lc_p),
            ]
            call_short_K = ck

        # Apply slippage to entry credit
        credit = credit * (1 - self.cfg["slippage_pct"])

        if credit <= 0:
            # Schedule retry next day
            self.next_entry_date = today + timedelta(days=1)
            return

        max_loss  = max(put_width, call_width) - credit
        contracts = self._size(max_loss)
        if contracts < 1:
            self.next_entry_date = today + timedelta(days=1)
            return

        commission = self.cfg["commission"] * len(legs) * contracts
        self.cash -= commission
        self.cash += credit * contracts * 100

        # Regime tagging
        hv_val   = sig.get("hv", 0.20)
        ivr_val  = sig.get("ivr", 0.50)
        trend    = self._regime_trend(sig)
        vol_reg  = self._regime_vol(vix_today, sig)
        atr_reg  = self._regime_atr(sig)

        expiry = today + timedelta(days=sp["dte"])
        self.position = Trade(
            key              = f"{self.spread_name}_{today}",
            spread_name      = self.spread_name,
            entry_date       = today,
            expiry           = expiry,
            legs             = legs,
            credit           = credit,
            max_loss         = max_loss,
            contracts        = contracts,
            trend_regime     = trend,
            vol_regime       = vol_reg,
            ivr              = ivr_val,
            atr_regime       = atr_reg,
            underlying_px    = S,
            iv_at_entry      = iv,
            hv_at_entry      = float(hv_val) if not pd.isna(hv_val) else 0.0,
            vrp_at_entry     = iv - (float(hv_val) if not pd.isna(hv_val) else 0.0),
            put_short_strike = put_short_K,
            call_short_strike= call_short_K,
        )

    # ── Position management ───────────────────────────────────

    def _manage(self, today, S, vix_today, sig):
        pos       = self.position
        dte       = (pos.expiry - today).days
        days_held = (today - pos.entry_date).days
        iv_now    = max(vix_today / 100, 0.05)
        cost      = self._spread_value(pos, today, S, vix_today)
        profit    = pos.credit - cost
        loss      = cost - pos.credit

        # Rule 1 — force close near expiry
        if dte <= self.cfg["exit_dte"]:
            self._close(today, S, vix_today, "dte_exit")
            return

        # Rule 2 — profit target (with min hold)
        if (days_held >= self.cfg["min_hold_days"] and
                profit >= pos.credit * self.cfg["profit_target"]):
            self._close(today, S, vix_today, "profit_target")
            return

        # Rule 3 — stop loss
        if loss >= pos.credit * self.cfg["stop_loss_mult"]:
            self._close(today, S, vix_today, "stop_loss")
            return

    def _close(self, today, S, vix_today, reason):
        pos       = self.position
        cost      = self._spread_value(pos, today, S, vix_today)
        cost      = cost * (1 + self.cfg["slippage_pct"])
        commission = self.cfg["commission"] * len(pos.legs) * pos.contracts
        pnl = (pos.credit - cost) * pos.contracts * 100 - commission

        self.cash -= pos.credit * pos.contracts * 100  # return original credit
        self.cash += pnl + pos.credit * pos.contracts * 100  # add net result

        pos.close_date   = today
        pos.close_reason = reason
        pos.realized_pnl = pnl
        pos.days_held    = (today - pos.entry_date).days
        self.closed_trades.append(pos)
        self.position = None

        # Schedule next entry after rotation_days
        self.next_entry_date = today + timedelta(days=self.cfg["rotation_days"])

    def _force_close(self, today, S, vix_today, reason):
        if self.position:
            self._close(today, S, vix_today, reason)

    # ── Valuation ─────────────────────────────────────────────

    def _spread_value(self, pos, today, S, vix_today):
        """Current cost to close the spread (mark-to-model)."""
        try:
            T  = max((pos.expiry - today).days / 365, 1/365)
            iv = max(vix_today / 100 * self.cfg["vrp_factor"], 0.05)
            total = 0.0
            for leg in pos.legs:
                p = bs_price(S, leg.strike, T, self.r, iv, leg.option_type)
                total += -p if leg.is_short else p
            return max(-total, 0.0)
        except Exception:
            return pos.credit

    def _mark(self, today, S, vix_today, sig):
        unrealized = 0.0
        if self.position:
            val = self._spread_value(self.position, today, S, vix_today)
            unrealized = (self.position.credit - val) * self.position.contracts * 100
        self.portfolio_value = self.cash + unrealized

    # ── Regime tagging ────────────────────────────────────────

    def _regime_trend(self, sig) -> str:
        sma_fast = sig.get("sma_fast", np.nan)
        sma_slow = sig.get("sma_slow", np.nan)
        price    = sig.get("price",    np.nan)
        if any(pd.isna(v) for v in [sma_fast, sma_slow, price]):
            return "unknown"
        if price > sma_fast > sma_slow:
            return "bullish"
        if price < sma_fast < sma_slow:
            return "bearish"
        return "neutral"

    def _regime_vol(self, vix_today, sig) -> str:
        vix_pct = sig.get("vix_pct", 0.5)
        if pd.isna(vix_pct):
            return "unknown"
        if vix_pct > 0.67:
            return "high"
        if vix_pct > 0.33:
            return "mid"
        return "low"

    def _regime_atr(self, sig) -> str:
        atr_fast = sig.get("atr_fast", np.nan)
        atr_slow = sig.get("atr_slow", np.nan)
        if pd.isna(atr_fast) or pd.isna(atr_slow) or atr_slow == 0:
            return "unknown"
        return "expanding" if atr_fast > atr_slow else "contracting"

    # ── Sizing ────────────────────────────────────────────────

    def _size(self, max_loss_per_spread) -> int:
        if max_loss_per_spread <= 0:
            return 0
        budget = self.portfolio_value * self.cfg["risk_pct"]
        n = int(budget / (max_loss_per_spread * 100))
        return max(1, min(n, self.cfg["max_contracts"]))

    # ── Results ───────────────────────────────────────────────

    def _build_results(self) -> dict:
        equity = pd.Series(
            {d: v for d, v in self.equity_curve},
            name="portfolio_value"
        )
        equity.index = pd.to_datetime(equity.index)

        if not self.closed_trades:
            return {"equity": equity, "trades": pd.DataFrame(),
                    "spread_name": self.spread_name}

        trades = pd.DataFrame([{
            "spread_name":    t.spread_name,
            "entry_date":     t.entry_date,
            "close_date":     t.close_date,
            "close_reason":   t.close_reason,
            "days_held":      t.days_held,
            "contracts":      t.contracts,
            "credit":         round(t.credit, 4),
            "max_loss":       round(t.max_loss, 4),
            "pnl":            round(t.realized_pnl, 2),
            "trend_regime":   t.trend_regime,
            "vol_regime":     t.vol_regime,
            "ivr":            round(t.ivr, 3),
            "atr_regime":     t.atr_regime,
            "underlying_px":  round(t.underlying_px, 2),
            "iv_at_entry":    round(t.iv_at_entry, 4),
            "hv_at_entry":    round(t.hv_at_entry, 4),
            "vrp_at_entry":   round(t.vrp_at_entry, 4),
            "put_short_strike": t.put_short_strike,
            "call_short_strike": t.call_short_strike,
        } for t in self.closed_trades])

        return {"equity": equity, "trades": trades,
                "spread_name": self.spread_name}


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & SIGNAL COMPUTATION
# ══════════════════════════════════════════════════════════════

def load_data(config: dict) -> pd.DataFrame:
    print("Downloading data via yfinance...")
    tickers = [config["ticker"], "^VIX"]
    raw = yf.download(tickers, start=config["start_date"],
                      end=config["end_date"],
                      auto_adjust=True, progress=False)
    closes = raw["Close"].copy()
    closes.columns = [c.replace("^", "") for c in closes.columns]
    closes.index = pd.to_datetime(closes.index).date
    closes = closes.ffill().dropna()
    print(f"  {len(closes)} trading days: {closes.index[0]} → {closes.index[-1]}")
    return closes


def build_signals(closes: pd.DataFrame, config: dict) -> dict:
    """Compute all regime-tagging and pricing signals."""
    ticker = config["ticker"]
    px     = closes[ticker]
    vix    = closes["VIX"]
    lr     = np.log(px / px.shift(1))

    hv        = lr.rolling(config["hv_lookback"]).std() * np.sqrt(252)
    hv_min    = hv.rolling(config["ivr_lookback"]).min()
    hv_max    = hv.rolling(config["ivr_lookback"]).max()
    ivr       = ((hv - hv_min) / (hv_max - hv_min + 1e-9)).clip(0, 1)

    vix_min   = vix.rolling(config["ivr_lookback"]).min()
    vix_max   = vix.rolling(config["ivr_lookback"]).max()
    vix_pct   = ((vix - vix_min) / (vix_max - vix_min + 1e-9)).clip(0, 1)

    sma_fast  = px.rolling(config["trend_fast"]).mean()
    sma_slow  = px.rolling(config["trend_slow"]).mean()
    atr_fast  = px.diff().abs().rolling(config["atr_fast"]).mean()
    atr_slow  = px.diff().abs().rolling(config["atr_slow"]).mean()

    # Align all series to closes index
    return {
        "hv":       hv,
        "ivr":      ivr,
        "vix_pct":  vix_pct,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "price":    px,
        "atr_fast": atr_fast,
        "atr_slow": atr_slow,
    }


# ══════════════════════════════════════════════════════════════
#  PERFORMANCE ANALYTICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(equity: pd.Series, trades: pd.DataFrame,
                    initial_capital: float) -> dict:
    ret    = equity.pct_change().dropna()
    years  = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.01)
    tot    = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cagr   = ((equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1) * 100
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
    down   = ret[ret < 0].std()
    sortino = (ret.mean() / down) * np.sqrt(252) if down > 0 else 0
    dd     = (equity - equity.cummax()) / equity.cummax()
    mdd    = dd.min() * 100
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    m = {
        "Total Return":    f"{tot:.1f}%",
        "CAGR":            f"{cagr:.1f}%",
        "Sharpe":          f"{sharpe:.2f}",
        "Sortino":         f"{sortino:.2f}",
        "Max Drawdown":    f"{mdd:.1f}%",
        "Calmar":          f"{calmar:.2f}",
    }

    if not trades.empty:
        wins    = trades[trades["pnl"] > 0]
        losses  = trades[trades["pnl"] <= 0]
        wr      = len(wins) / len(trades) * 100
        pf      = (wins["pnl"].sum() / abs(losses["pnl"].sum())
                   if losses["pnl"].sum() != 0 else float("inf"))
        m.update({
            "Trades":          str(len(trades)),
            "Win Rate":        f"{wr:.1f}%",
            "Avg Win":         f"${wins['pnl'].mean():,.0f}" if len(wins) else "$0",
            "Avg Loss":        f"${losses['pnl'].mean():,.0f}" if len(losses) else "$0",
            "Profit Factor":   f"{pf:.2f}",
            "Avg Hold":        f"{trades['days_held'].mean():.1f}d",
            "Total P&L":       f"${trades['pnl'].sum():,.0f}",
        })
    return m, dd


# ══════════════════════════════════════════════════════════════
#  REGIME × SPREAD CROSS-TAB ANALYSIS
# ══════════════════════════════════════════════════════════════

def regime_analysis(all_trades: pd.DataFrame):
    """
    Cross-tab P&L and win rate by spread × regime combination.
    This is the core output for identifying when each spread works.
    """
    print("\n" + "═" * 70)
    print("  REGIME ANALYSIS — where does each spread have edge?")
    print("═" * 70)

    for regime_col in ["trend_regime", "vol_regime", "atr_regime"]:
        print(f"\n── By {regime_col} ──────────────────────────────────────────")
        pivot = all_trades.groupby(["spread_name", regime_col]).agg(
            trades   = ("pnl", "count"),
            win_rate = ("pnl", lambda x: f"{(x > 0).mean()*100:.0f}%"),
            avg_pnl  = ("pnl", lambda x: f"${x.mean():,.0f}"),
            total    = ("pnl", lambda x: f"${x.sum():,.0f}"),
        )
        print(pivot.to_string())

    # Best spread × regime combos by win rate (min 5 trades)
    print("\n── Top 10 spread × regime combos (min 5 trades) ────────────────")
    combos = all_trades.groupby(["spread_name", "trend_regime", "vol_regime"]).agg(
        trades   = ("pnl", "count"),
        win_rate = ("pnl", lambda x: (x > 0).mean() * 100),
        avg_pnl  = ("pnl", "mean"),
        total    = ("pnl", "sum"),
    ).reset_index()
    combos = combos[combos["trades"] >= 5].sort_values("win_rate", ascending=False)
    if not combos.empty:
        print(combos.head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════
#  TEARSHEET
# ══════════════════════════════════════════════════════════════

def plot_single_tearsheet(result: dict, metrics: dict, drawdowns: pd.Series):
    trades     = result["trades"]
    equity     = result["equity"]
    spread_name = result["spread_name"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 20))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)
    G, R, B, A = "#1D9E75", "#E24B4A", "#378ADD", "#EF9F27"

    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity / 1000, color=G, lw=1.5)
    ax1.fill_between(equity.index, equity / 1000, equity.min() / 1000,
                     alpha=0.07, color=G)
    ax1.set_title(f"Equity curve — {spread_name}", fontsize=13, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}K"))

    # Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(drawdowns.index, drawdowns * 100, 0, color=R, alpha=0.5)
    ax2.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    if not trades.empty:
        # P&L per trade
        ax3 = fig.add_subplot(gs[2, 0])
        cols = [G if p > 0 else R for p in trades["pnl"]]
        ax3.bar(range(len(trades)), trades["pnl"], color=cols, width=0.8)
        ax3.axhline(0, color="#888", lw=0.8, ls="--")
        ax3.set_title("P&L per trade", fontsize=13, fontweight="bold")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # Exit reason
        ax4 = fig.add_subplot(gs[2, 1])
        rc = trades["close_reason"].value_counts()
        reason_colors = {"profit_target": G, "stop_loss": R,
                         "dte_exit": B, "end_of_backtest": "#888"}
        ax4.bar(rc.index, rc.values,
                color=[reason_colors.get(r, A) for r in rc.index], width=0.5)
        ax4.set_title("Exit reasons", fontsize=13, fontweight="bold")
        ax4.tick_params(axis="x", rotation=20)

        # Monthly P&L
        ax5 = fig.add_subplot(gs[3, 0])
        trades["month"] = pd.to_datetime(trades["close_date"]).dt.to_period("M")
        monthly = trades.groupby("month")["pnl"].sum()
        ax5.bar(monthly.index.astype(str), monthly.values,
                color=[G if v >= 0 else R for v in monthly.values], width=0.7)
        ax5.axhline(0, color="#888", lw=0.8, ls="--")
        ax5.set_title("Monthly P&L", fontsize=13, fontweight="bold")
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax5.tick_params(axis="x", rotation=45)

        # P&L by trend regime
        ax6 = fig.add_subplot(gs[3, 1])
        regime_pnl = trades.groupby("trend_regime")["pnl"].agg(["sum", "count", "mean"])
        ax6.bar(regime_pnl.index, regime_pnl["sum"],
                color=[G if v >= 0 else R for v in regime_pnl["sum"]], width=0.4)
        ax6.axhline(0, color="#888", lw=0.8, ls="--")
        ax6.set_title("Total P&L by trend regime", fontsize=13, fontweight="bold")
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    mtext = "\n".join(f"{k:<18} {v}" for k, v in metrics.items())
    fig.text(0.01, 0.985, f"Summary — {spread_name}", fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.01, 0.968, mtext, fontsize=10, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f0",
                       edgecolor="#ccc", alpha=0.85))

    fname = f"tearsheet_{spread_name}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {fname}")


def plot_comparison_tearsheet(all_results: list):
    """Side-by-side equity curves and win-rate bar for all spreads."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    G, R = "#1D9E75", "#E24B4A"

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # Equity curves
    ax = axes[0]
    for (result, _, _), col in zip(all_results, colors):
        eq = result["equity"]
        ax.plot(eq.index, eq / 1000, lw=1.4,
                label=result["spread_name"], color=col)
    ax.set_title("All spreads — equity curves", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}K"))
    ax.legend(fontsize=9, ncol=3)

    # Win rate + total P&L comparison
    ax2 = axes[1]
    names  = [r["spread_name"] for r, _, _ in all_results]
    totals = []
    wrs    = []
    for result, _, _ in all_results:
        t = result["trades"]
        if t.empty:
            totals.append(0); wrs.append(0)
        else:
            totals.append(t["pnl"].sum())
            wrs.append((t["pnl"] > 0).mean() * 100)

    x = np.arange(len(names))
    ax2.bar(x - 0.2, totals, width=0.35,
            color=[G if v >= 0 else R for v in totals], label="Total P&L ($)")
    ax2b = ax2.twinx()
    ax2b.plot(x, wrs, "o--", color="#378ADD", lw=1.5, ms=6, label="Win rate %")
    ax2b.set_ylabel("Win rate (%)", color="#378ADD")
    ax2b.set_ylim(0, 110)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=25, ha="right")
    ax2.set_title("Spread comparison — total P&L and win rate", fontsize=14, fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("comparison_all_spreads.png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Saved comparison_all_spreads.png")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def run_single(config: dict, spread_name: str,
               closes: pd.DataFrame, signals: dict) -> tuple:
    print(f"  Running: {spread_name}  "
          f"({SPREAD_LIBRARY[spread_name]['description']})")
    bt     = RotationBacktester(config, spread_name)
    result = bt.run(closes, signals)
    metrics, dd = compute_metrics(result["equity"], result["trades"],
                                  config["initial_capital"])
    return result, metrics, dd


def run_all_spreads(config: dict, closes: pd.DataFrame,
                    signals: dict) -> list:
    all_results = []
    all_trades  = []

    for name in SPREAD_LIBRARY:
        result, metrics, dd = run_single(config, name, closes, signals)
        all_results.append((result, metrics, dd))

        t = result["trades"]
        if not t.empty:
            all_trades.append(t)

        # Print summary
        print(f"    P&L=${result['trades']['pnl'].sum():,.0f}  "
              f"WR={metrics.get('Win Rate','N/A')}  "
              f"Trades={metrics.get('Trades','0')}") if not result["trades"].empty \
            else print(f"    No trades generated.")

    return all_results, pd.concat(all_trades) if all_trades else pd.DataFrame()


def main():
    print("=" * 65)
    print("  Options Rotation Backtester")
    print(f"  Ticker: {CONFIG['ticker']}  "
          f"Rotation: every {CONFIG['rotation_days']} days")
    print("=" * 65)

    closes  = load_data(CONFIG)
    signals = build_signals(closes, CONFIG)

    print("\nBacktesting all spreads...")
    all_results, all_trades = run_all_spreads(CONFIG, closes, signals)

    # Per-spread tearsheets
    print("\nGenerating tearsheets...")
    for result, metrics, dd in all_results:
        if not result["trades"].empty:
            plot_single_tearsheet(result, metrics, dd)

    # Comparison chart
    plot_comparison_tearsheet(all_results)

    # Regime analysis
    if not all_trades.empty:
        regime_analysis(all_trades)
        all_trades.to_csv("all_trades_rotation.csv", index=False)
        print("\nFull trade log saved to all_trades_rotation.csv")

    # Summary table
    print("\n" + "═" * 65)
    print(f"  {'Spread':<22} {'Trades':>7} {'Win%':>7} "
          f"{'Sharpe':>8} {'Total P&L':>12}")
    print("  " + "─" * 60)
    for result, metrics, _ in all_results:
        t = result["trades"]
        if t.empty:
            print(f"  {result['spread_name']:<22} {'0':>7}")
            continue
        print(f"  {result['spread_name']:<22} "
              f"{metrics.get('Trades','0'):>7} "
              f"{metrics.get('Win Rate','?'):>7} "
              f"{metrics.get('Sharpe','?'):>8} "
              f"{metrics.get('Total P&L','?'):>12}")
    print("═" * 65)
    print("\nDone.")


if __name__ == "__main__":
    main()
