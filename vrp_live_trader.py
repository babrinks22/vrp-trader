#!/usr/bin/env python3
# ============================================================
#  VRP Live Trader — Alpaca
#  4×IWM iron condor, 30 DTE, 65% profit target, skip-2 gate
#
#  Install:
#    pip install alpaca-py yfinance numpy pandas scipy matplotlib
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
#      1. Reads state.json (tracks all 4 slots)
#      2. Manages open positions (profit target / DTE exit)
#      3. Opens new positions if a slot is empty (subject to 7d stagger)
#      4. Detects option assignments and alerts
#      5. Writes updated state.json and appends to trade_log.csv
#
#  CLI flags:
#    python vrp_live_trader.py            normal daily run
#    python vrp_live_trader.py --dump-state   pretty-print state.json and exit
#
#  State file (state.json):
#    Persists between runs. Tracks all 4 IWM slots (A/B/C/D),
#    cumulative P&L, per-slot stats, and trade count.
#    Delete it to reset.
#
#  IMPORTANT — read before going live:
#    - Paper trade for at least 3 months first
#    - Verify fills match expected credit before scaling up
#    - Options Level 3 required for iron condors on live account
#    - IWM options are American-style (early assignment risk near DTE)
#    - Always review the log before market close on entry days
# ============================================================

import os, sys, json, math, time, logging, argparse, urllib.request
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

try:
    import zoneinfo
except ImportError:
    zoneinfo = None

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
    # IWM only — 4 slots × 17.5% = 70% deployed, 30% cash reserve
    # Why IWM only: cleaner fills at 2c per slot, no inverted-strike chain
    # issues, lower slippage (12% vs 44-57% for XBI at small size)
    "iwm_slot_risk":   0.175,  # 17.5% per slot (4 slots = 70% deployed)

    # ── Options parameters ────────────────────────────────────
    "slot_dte":        30,     # target DTE at entry — 30 DTE optimal:
                               # (1) fewer trades/yr → fewer fees
                               # (2) more legs expire worthless on Alpaca ($0 close)
                               # (3) more time for position to recover before expiry
    "dte_tolerance":   3,      # accept contracts within ±3 DTE of target
    "exit_dte":        3,      # force-close at ≤3 DTE
    "min_hold_days":   0,      # minimum days before profit target fires
    "profit_target":   -0.50,   # close at 65% of credit — optimal by Sharpe (19yr backtest)
                               # improves Sharpe 0.50→1.46 at live sizing, MaxDD -3.7%→-1.2%
    "stagger_days":    7,      # Minimum days between any two same-ticker entries

    # ── Regime signals ────────────────────────────────────────
    # Trend SMA pair: 14/40 (NOT 20/50). Chosen from the tight trend-SMA
    # sweep and confirmed by cross-instrument validation on an 8-ETF basket
    # (DIA/EFA/EEM/XLF/XLK/XLU/FXI/EWJ) — 14/40 sits in the high-Sharpe
    # plateau on both IWM/XLE and the unrelated ETFs, and is the most
    # sub-period-consistent pair tested. 20/50 ranked far lower on both.
    # Keep 14/40 for all future versions. See etf_validate.py / trend_sweep.py.
    "trend_fast":      14,     # SMA fast period
    "trend_slow":      40,     # SMA slow period
    "hv_lookback":     20,     # historical vol lookback
    "ivr_lookback":    252,    # IVR percentile window
    "atr_fast":        5,
    "atr_slow":        20,
    "vrp_factor":      1.18,   # IV = HV × factor (IWM VRP adjustment)

    # ── Spread parameters ─────────────────────────────────────
    # Per-spread minimum credits — iron condor collects ~2× the credit of
    # single-side spreads (4 vs 2 legs), so the IC gate is correspondingly
    # higher. Values empirically calibrated as a high-EV-margin filter:
    # they pass only trades with enough cushion that the expected-value
    # math survives realistic VRP variance.
    #
    # RATIO-BASED (credit as a fraction of wing width), NOT a flat dollar
    # amount. A flat $ floor silently punishes lower-priced underlyings: at
    # identical risk/reward, XLE (~$60) collects far fewer dollars of credit
    # than IWM (~$273) purely because its strikes and wings are smaller — so
    # a flat floor blocks economically sound XLE trades and holds XLE to a
    # ~3x stricter standard. A ratio is price-invariant. Values calibrated
    # to the effective ratio bar the old $0.90 / $0.50 floors produced for
    # the average trade (see credit_gate_audit.py).
    "min_credit_ratio_ic":   0.01,   # IC  credit must be >= 15%  of wing width
    "min_credit_ratio_pcs":  0.09,   # PCS credit must be >= 9%   of wing width
    "min_credit_ratio_ccs":  0.085,  # CCS credit must be >= 8.5% of wing width

    # ── Indicator parameters (V-bottom protection for CCS) ────
    "rsi_period":          14,   # Wilder RSI period on underlying close
    "rsi_ccs_threshold":   43,   # block CCS entries when RSI < this
                                 # (avoids selling calls into oversold bottoms
                                 # vulnerable to V-bottom reversal)
    "vix_fresh_lookback":  60,   # window for fresh-VIX-high panic-peak detection
    "panic_call_delta":    0.10, # widened short-call delta during
                                 # bearish + fresh-60d-VIX-high
    # ── Iron-condor regime-change gate (data-validated) ───────
    "vix_rising_lookback": 30,    # IC blocked if VIX rose over this many sessions
    "ic_down_days_window": 10,   # window for the down-day count
    "ic_down_days_block":  6,    # IC blocked if >= this many down days in window
    "r":               0.04,   # risk-free rate fallback if live ^IRX fetch fails
    "max_quote_spread_pct": 0.30,  # skip options where (ask-bid)/mid > this (#14)

    # ── Execution ─────────────────────────────────────────────
    # Credit limit orders: submit at mid, then widen by this if unfilled
    "limit_offset":    0.05,   # widen by $0.05 if not filled in 30s
    "max_fill_retries": 3,

    # ── Files ─────────────────────────────────────────────────
    "state_file":     "state.json",
    "log_file":       "trade_log.csv",
}


# ══════════════════════════════════════════════════════════════
#  REGIME / SPREAD CONFIG  (regime-mapped — see PATCH discussion)
# ══════════════════════════════════════════════════════════════
#
# Strategy:
#   - IC  in neutral regimes with stable/falling vol  → premium harvest
#   - PCS in bullish regimes                          → directional vol selling
#   - CCS in bearish regimes (with RSI gate)          → crash protection
#   - SKIP in neutral + expanding vol                 → uncertain direction
#
# Refinements baked in:
#   - CCS requires RSI(14) ≥ rsi_ccs_threshold (avoids V-bottom reversals)
#   - In bearish + fresh-60d-VIX-high: short call delta = panic_call_delta
#     (V-spike protection via wider call wing)
#
# Out-of-sample sweep verified the 43-45 RSI threshold plateau is real
# signal (see backtest_v5). 43 chosen as conservative edge of plateau.

SPREAD_PARAMS = {
    "iron_condor":        {"put_delta": 0.20, "call_delta": 0.20, "put_width_mult": 1.0},
    "put_credit_spread":  {"put_delta": 0.20, "call_delta": 0.00, "put_width_mult": 1.0},
    "call_credit_spread": {"put_delta": 0.00, "call_delta": 0.20, "put_width_mult": 1.0},
}

# Regime → spread map. (trend, vol_level, atr_direction) → spread or "SKIP".
REGIME_MAP = {
    # Bullish: PCS by default — call side most likely to be tested in uptrends
    ("bullish", "low",  "contracting"): "put_credit_spread",
    ("bullish", "low",  "expanding"):   "put_credit_spread",
    ("bullish", "mid",  "contracting"): "iron_condor",   # boring uptrend → full IC
    ("bullish", "mid",  "expanding"):   "put_credit_spread",
    ("bullish", "high", "contracting"): "put_credit_spread",
    ("bullish", "high", "expanding"):   "put_credit_spread",

    # Neutral: IC when vol contracting, SKIP when expanding (legacy skip-2)
    ("neutral", "low",  "contracting"): "iron_condor",
    ("neutral", "low",  "expanding"):   "iron_condor",
    ("neutral", "mid",  "contracting"): "iron_condor",
    ("neutral", "mid",  "expanding"):   "iron_condor",
    ("neutral", "high", "contracting"): "iron_condor",
    ("neutral", "high", "expanding"):   "iron_condor",

    # Bearish: CCS by default — short calls profit from decline
    ("bearish", "low",  "contracting"): "call_credit_spread",
    ("bearish", "low",  "expanding"):   "SKIP",          # legacy skip-2 entry
    ("bearish", "mid",  "contracting"): "call_credit_spread",
    ("bearish", "mid",  "expanding"):   "call_credit_spread",
    ("bearish", "high", "contracting"): "call_credit_spread",
    ("bearish", "high", "expanding"):   "call_credit_spread",
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
            "IWM_A": {"position": None, "next_entry": None, "closed_trades": []},
            "XLE_B": {"position": None, "next_entry": None, "closed_trades": []},
            "IWM_C": {"position": None, "next_entry": None, "closed_trades": []},
            "XLE_D": {"position": None, "next_entry": None, "closed_trades": []},
        },
        "initial_capital": None,
        "cumulative_pnl":  0.0,
        "trade_count":     0,
        "equity_history":  [],   # list of {"date": YYYY-MM-DD, "value": float}
                                  # populated daily for the equity chart
    }


def save_state(state: dict):
    with open(CONFIG["state_file"], "w") as f:
        json.dump(state, f, indent=2, default=str)


# Canonical columns — every row writes all of these, blank for non-applicable fields.
# open rows:  credit, credit_mid, max_loss, underlying_px, slippage_pct, order_id
# close rows: credit, max_loss, close_reason, realized_pnl, entry_date, days_held, order_id
TRADE_LOG_COLS = [
    "date", "action", "slot", "spread", "trend", "vol", "atr",
    "contracts", "credit", "credit_mid", "max_loss",
    "underlying_px", "slippage_pct",
    "close_reason", "realized_pnl", "entry_date", "days_held",
    "order_id",
]


def _validate_or_rotate_trade_log(path: Path):
    """
    If trade_log.csv exists with a header that doesn't match TRADE_LOG_COLS
    exactly, rotate it to trade_log_legacy_<YYYYMMDD>.csv and start fresh.
    This is a one-shot self-heal for the schema drift that accumulated when
    columns were added over time without rewriting the header.
    """
    if not path.exists():
        return
    try:
        with open(path, "r", newline="") as fh:
            first_line = fh.readline().strip()
        expected = ",".join(TRADE_LOG_COLS)
        if first_line == expected:
            return   # header matches, nothing to do
        # Mismatch — rotate
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        rotated = path.with_name(f"trade_log_legacy_{ts}.csv")
        path.rename(rotated)
        log.warning(f"  trade_log.csv header drift detected. "
                    f"Rotated old file → {rotated.name}. "
                    f"A fresh trade_log.csv will be created on the next write "
                    f"using the canonical {len(TRADE_LOG_COLS)}-column schema.")
    except Exception as e:
        log.warning(f"  Could not validate trade_log.csv header: {e}")


def log_trade(record: dict):
    """Append one row to trade_log.csv using the fixed canonical schema."""
    import csv as _csv
    path = Path(CONFIG["log_file"])
    _validate_or_rotate_trade_log(path)
    row  = {col: record.get(col, "") for col in TRADE_LOG_COLS}
    write_header = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=TRADE_LOG_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
#  RISK-FREE RATE (cached from ^IRX, refreshed daily)
# ══════════════════════════════════════════════════════════════

_RATE_CACHE_FILE = Path(".rfr_cache.json")


def get_risk_free_rate() -> float:
    """
    Return the current 3-month T-bill yield as a decimal (e.g. 0.045).
    Cached to disk for 24 hours to avoid hammering yfinance on every brentq call.
    Falls back to CONFIG["r"] if the fetch fails.
    """
    today_iso = date.today().isoformat()
    try:
        if _RATE_CACHE_FILE.exists():
            cached = json.loads(_RATE_CACHE_FILE.read_text())
            if cached.get("date") == today_iso:
                return float(cached["rate"])
    except Exception:
        pass

    rate = CONFIG["r"]
    try:
        # ^IRX is the 13-week T-bill index, quoted in percent (e.g. 4.5 = 4.5%)
        bars = yf.download("^IRX", period="5d",
                           auto_adjust=True, progress=False)["Close"].squeeze()
        latest = float(bars.dropna().iloc[-1])
        if 0 < latest < 25:
            rate = latest / 100.0
        else:
            log.warning(f"  ^IRX returned implausible value {latest}; using fallback {rate}")
    except Exception as e:
        log.warning(f"  ^IRX fetch failed: {e}; using fallback {rate}")

    try:
        _RATE_CACHE_FILE.write_text(json.dumps({"date": today_iso, "rate": rate}))
    except Exception:
        pass
    return rate


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
    """
    Solve for the strike at which |delta| == target_delta.

    Bracket widened to [S*0.10, S*2.50] (was [S*0.40, S*1.60]) to handle
    high-IV regimes where 0.20-delta puts can sit far below S. The old bracket
    failed silently into the linear-approximation fallback, returning a strike
    ~30% closer to ATM than intended on a vol spike.
    """
    if T <= 0 or sigma <= 0:
        return S * (1 - target_delta) if option_type == "put" else S * (1 + target_delta)
    try:
        return brentq(
            lambda K: abs(bs_delta(S, K, T, r, sigma, option_type)) - target_delta,
            S * 0.10, S * 2.50, xtol=0.01,
        )
    except ValueError:
        log.warning(f"  find_strike_by_delta brentq failed for S={S:.2f} "
                    f"sigma={sigma:.3f} T={T:.4f} target_delta={target_delta:.2f} "
                    f"type={option_type} — using linear fallback")
        return S * (1 - target_delta * 1.5) if option_type == "put" else S * (1 + target_delta * 1.5)


def bs_price(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes price for a European put or call."""
    if T <= 0:
        return max(K - S, 0) if option_type == "put" else max(S - K, 0)
    if sigma <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except Exception:
        return 0.0


def _implied_vol_from_price(S, K, T, r, price, opt_type="put"):
    """
    Solve for the implied volatility that prices the option at `price`.
    Returns the IV (e.g. 0.18) or None on failure.

    Used by enter_slot's IV-refinement step (#15) so strike selection uses the
    actual market-implied vol rather than HV × VRP factor.
    """
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return None
    try:
        return brentq(
            lambda v: bs_price(S, K, T, r, v, opt_type) - price,
            1e-4, 5.0, xtol=1e-4,
        )
    except (ValueError, RuntimeError):
        return None


# ══════════════════════════════════════════════════════════════
#  INDICATORS — RSI + VIX fresh-high + strategy selector
# ══════════════════════════════════════════════════════════════

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Wilder's RSI on a price series. Returns the latest value.
    Uses EMA with α = 1/period (canonical Wilder smoothing).
    Returns 50.0 on insufficient data.
    """
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def is_vix_fresh_high(vix_series: pd.Series, lookback: int = 60) -> bool:
    """
    True if today's VIX equals the maximum over the last `lookback` trading days.
    Used to trigger panic-peak protection (wider short-call delta in bearish regimes).
    """
    if len(vix_series) < lookback:
        return False
    today_vix = vix_series.iloc[-1]
    window_max = vix_series.iloc[-lookback:].max()
    return bool(today_vix >= window_max and not pd.isna(today_vix))


def is_vix_rising(vix_series: pd.Series, lookback: int = 5) -> bool:
    """
    True if VIX has risen over the last `lookback` trading sessions.

    Used as an iron-condor entry gate. Backtest + out-of-sample analysis
    (2008-2016 real data) showed IC trades entered while VIX was rising had a
    ~68% win rate vs ~96-100% when VIX was falling/flat. Rising VIX signals
    stress building before it is fully priced into IV, so the IC wings are
    more likely to be breached. See indicator_analysis.py for the study.
    """
    if len(vix_series) < lookback + 1:
        return False
    today = vix_series.iloc[-1]
    prior = vix_series.iloc[-lookback - 1]
    if pd.isna(today) or pd.isna(prior):
        return False
    return bool(today > prior)


def count_down_days(price_series: pd.Series, lookback: int = 10) -> int:
    """
    Count negative-return sessions in the last `lookback` trading days.

    Used as a secondary iron-condor entry gate. When 6+ of the last 10
    sessions were down, the market is in a confirmed short-term downtrend
    that the SMA-based regime classifier can miss (it may still read
    "neutral" during a trend pause). Backtest showed IC win rate of ~75%
    when down_days >= 6 vs ~87% otherwise.
    """
    if len(price_series) < lookback + 1:
        return 0
    rets = price_series.diff().iloc[-lookback:]
    return int((rets < 0).sum())


def select_strategy(regime: dict, vix_fresh_high: bool,
                    rsi_today: float, cfg: dict = None,
                    vix_rising: bool = False,
                    down_days: int = 0) -> dict:
    """
    Map a regime + indicator state to a concrete strategy decision.

    Returns a dict with keys:
      spread       : "iron_condor" | "put_credit_spread" | "call_credit_spread" | None
      put_delta    : target delta for short put leg (0 if no put side)
      call_delta   : target delta for short call leg (0 if no call side)
      skip_reason  : None if entering, else a short string explaining the skip

    Inputs:
      regime          : dict from get_regime() — needs 'trend', 'vol', 'atr'
      vix_fresh_high  : bool from is_vix_fresh_high()
      rsi_today       : float in [0, 100]
      cfg             : optional config override (defaults to CONFIG)
      vix_rising      : bool from is_vix_rising() — IC entry gate
      down_days       : int from count_down_days() — IC entry gate
    """
    cfg = cfg or CONFIG
    regime_key = (regime["trend"], regime["vol"], regime["atr"])
    spread = REGIME_MAP.get(regime_key)   # None if the regime is genuinely absent

    if spread is None:
        # The regime tuple is NOT a key in REGIME_MAP at all. This is not a
        # normal no-trade cell — it usually means an 'unknown' regime (data
        # warmup or a data problem) or a genuine gap in the map. Flag it
        # distinctly so it is not mistaken for a deliberate SKIP.
        return {"spread": None, "put_delta": 0.0, "call_delta": 0.0,
                "skip_reason": f"regime {regime_key} NOT RECOGNISED — not a key "
                               f"in REGIME_MAP (likely an 'unknown' regime from "
                               f"data warmup, or a map gap — worth a look)"}

    if spread == "SKIP":
        # The regime IS in the map and is a designated no-trade cell. This is
        # intended, normal behaviour (e.g. neutral trend + expanding range).
        return {"spread": None, "put_delta": 0.0, "call_delta": 0.0,
                "skip_reason": f"regime {regime_key} is a designated no-trade "
                               f"(SKIP) regime — holding cash by design"}

    # V-bottom gate: CCS requires RSI to have recovered from oversold zone
    if spread == "call_credit_spread" and rsi_today < cfg["rsi_ccs_threshold"]:
        return {"spread": None, "put_delta": 0.0, "call_delta": 0.0,
                "skip_reason": f"RSI {rsi_today:.1f} < {cfg['rsi_ccs_threshold']} "
                               f"(V-bottom risk on CCS)"}

    # ── Iron-condor regime-change gate ───────────────────────────
    # ICs are the strategy's most fragile structure during regime change:
    # both wings are exposed, so a directional break loses on one side
    # with no offsetting gain. Two indicators of "regime breaking" block
    # IC entry (data-validated, see indicator_analysis.py):
    #   - VIX rising over the last 5 sessions
    #   - 6+ of the last 10 sessions were down
    # When either fires, hold cash rather than sell a fragile IC.
    if spread == "iron_condor":
        if vix_rising:
            return {"spread": None, "put_delta": 0.0, "call_delta": 0.0,
                    "skip_reason": "IC blocked: VIX rising over last 5 sessions "
                                   "(regime-change risk)"}
        if down_days >= cfg["ic_down_days_block"]:
            return {"spread": None, "put_delta": 0.0, "call_delta": 0.0,
                    "skip_reason": f"IC blocked: {down_days} down days in last 10 "
                                   f"(confirmed short-term downtrend)"}

    params = SPREAD_PARAMS[spread]
    put_delta  = params["put_delta"]
    call_delta = params["call_delta"]

    # Panic-peak: widen short call when bearish + fresh-60d VIX high
    # (call_delta=0.10 instead of 0.20 — roughly 2× further OTM for V-spike cushion)
    if (regime["trend"] == "bearish" and vix_fresh_high
            and call_delta > 0):
        call_delta = cfg["panic_call_delta"]

    return {"spread": spread, "put_delta": put_delta,
            "call_delta": call_delta, "skip_reason": None}


# ══════════════════════════════════════════════════════════════
#  REGIME DETECTION
# ══════════════════════════════════════════════════════════════

_PRICE_CACHE_DIR = Path(".price_cache")


def _fetch_history(symbol: str, lookback_days: int,
                   max_retries: int = 3) -> pd.Series:
    """
    Download daily close history for `symbol` with disk caching and retry.

    Cache strategy:
      - Stored at .price_cache/{symbol}.csv as date,close rows
      - On each call we read the cache, find the last cached date, and
        fetch only the missing tail (or full lookback if no cache)
      - This trims ~250 daily rows × 22 KB/yr down to ~1 KB/day of network

    Retry strategy:
      - yfinance occasionally returns empty (rate limit / outage)
      - 3 attempts with exponential backoff (1s, 2s, 4s)
      - On total failure, returns whatever cache we have; if no cache, raises
    """
    _PRICE_CACHE_DIR.mkdir(exist_ok=True)
    cache_path = _PRICE_CACHE_DIR / f"{symbol.replace('^', '_').replace('/', '_')}.csv"

    cached_series = None
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached_series = cached["close"].dropna()
            cached_series.index = pd.to_datetime(cached_series.index)
        except Exception as e:
            log.warning(f"  cache read failed for {symbol}: {e}")
            cached_series = None

    # Decide fetch window
    today = pd.Timestamp.today().normalize()
    if cached_series is not None and len(cached_series) > 0:
        last_cached = pd.Timestamp(cached_series.index.max()).normalize()
        # if we have data through yesterday, only fetch the last few days
        fetch_start = max(last_cached - pd.Timedelta(days=2),
                          today - pd.Timedelta(days=lookback_days * 1.5))
    else:
        fetch_start = today - pd.Timedelta(days=lookback_days * 1.5)

    new_series = None
    for attempt in range(max_retries):
        try:
            raw = yf.download(symbol, start=fetch_start.strftime("%Y-%m-%d"),
                              end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                raise RuntimeError("empty result")
            new_series = raw["Close"].squeeze().dropna()
            new_series.index = pd.to_datetime(new_series.index)
            break
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"  yfinance {symbol} attempt {attempt+1}/{max_retries} "
                        f"failed: {e} — sleeping {wait}s")
            time.sleep(wait)

    # Merge cache + fresh
    if new_series is not None and len(new_series) > 0:
        if cached_series is not None:
            combined = pd.concat([cached_series, new_series])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_series
        # Write back
        try:
            out = combined.to_frame("close")
            out.index.name = "date"
            out.to_csv(cache_path)
        except Exception as e:
            log.warning(f"  cache write failed for {symbol}: {e}")
        return combined

    # Fetch failed; fall back to cache if we have it
    if cached_series is not None and len(cached_series) > 0:
        log.warning(f"  using stale cache for {symbol} "
                    f"(last bar {cached_series.index.max().date()})")
        return cached_series

    raise RuntimeError(f"Could not fetch {symbol} and no cache available")


def get_regime(cfg: dict, ticker: str = "IWM") -> dict:
    """
    Compute regime signals for the given ticker.
    Returns: {"trend", "vol", "atr", "hv", "iv", "vix", "price", "spread", "skip_entry"}
    """
    lookback_days = cfg["ivr_lookback"] + 60

    underlying = _fetch_history(ticker, lookback_days)
    vix        = _fetch_history("^VIX", lookback_days)

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

    # ── Spread selection ─────────────────────────────────────────
    # No longer hardcoded. The returned regime dict is consumed by
    # select_strategy() in enter_slot, which maps to IC/PCS/CCS/SKIP
    # based on REGIME_MAP plus the RSI and fresh-VIX-high gates.
    spread = "regime_dispatched"   # placeholder; actual choice in select_strategy

    # ── Indicator values for strategy selection ─────────────────
    rsi_today = compute_rsi(px, cfg["rsi_period"])
    vix_fresh = is_vix_fresh_high(vix, cfg["vix_fresh_lookback"])
    vix_rising = is_vix_rising(vix, cfg["vix_rising_lookback"])
    down_days  = count_down_days(px, cfg["ic_down_days_window"])

    iv = float(hv_val) * cfg["vrp_factor"] if not pd.isna(hv_val) else 0.20

    return {
        "trend":          trend,
        "vol":            vol,
        "atr":            atr,
        "hv":             float(hv_val) if not pd.isna(hv_val) else 0.20,
        "iv":             iv,
        "vix":            float(vix_val),
        "price":          float(p),
        "spread":         spread,       # placeholder, see select_strategy()
        "skip_entry":     False,         # legacy field — kept for backwards compat;
                                         # actual skip handled by select_strategy()
        "rsi":            rsi_today,
        "vix_fresh_high": vix_fresh,
        "vix_rising":     vix_rising,    # IC entry gate
        "down_days":      down_days,     # IC entry gate
    }


# ══════════════════════════════════════════════════════════════
#  OPTION CONTRACT SELECTION
# ══════════════════════════════════════════════════════════════

def find_contract(trade_client, opt_data_client,
                  underlying: str, option_type: str,
                  target_delta: float, target_strike: float,
                  target_expiry: date) -> dict:
    """
    Find the option contract closest to target_strike expiring closest to
    target_expiry, subject to a bid-ask quality filter.

    Quality filter: rejects contracts where (ask - bid) / mid exceeds
    CONFIG["max_quote_spread_pct"]. On a thin chain the absolute-closest
    strike may have a $0 bid and $5 ask — a "mid" of $2.50 is fictional and
    will never fill at limit. Better to pick the next-closest strike with a
    real two-sided market than to submit an order that never executes.

    Returns dict with {symbol, strike, expiry, type, bid, ask, mid} or None.
    """
    exp_min = (target_expiry - timedelta(days=CONFIG["dte_tolerance"])).isoformat()
    exp_max = (target_expiry + timedelta(days=CONFIG["dte_tolerance"])).isoformat()

    # Bound the query to a strike band around the target. WITHOUT a strike
    # filter, get_option_contracts returns only its first page (limit
    # default = 100) ordered strike-ascending. For an underlying with many
    # strikes (IWM, XLE) the OTM calls — where the short call actually
    # lives — fall past contract #100 and are never returned, so every
    # call leg snaps to the highest *returned* strike: a degenerate,
    # inverted call spread. Bounding by strike keeps the result small and
    # guarantees the target's neighbourhood is present. (Puts happened to
    # work only because their strikes sit low in the ascending list.)
    STRIKE_BAND = 45.0   # $ above/below target — wide enough for sparse chains
    strike_lo = max(0.5, target_strike - STRIKE_BAND)
    strike_hi = target_strike + STRIKE_BAND

    try:
        contracts = trade_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[underlying],
                expiration_date_gte=exp_min,
                expiration_date_lte=exp_max,
                type=ContractType.PUT if option_type == "put" else ContractType.CALL,
                status=AssetStatus.ACTIVE,
                strike_price_gte=f"{strike_lo:.2f}",
                strike_price_lte=f"{strike_hi:.2f}",
                limit=500,
            )
        )
    except Exception as e:
        log.error(f"Contract lookup failed: {e}")
        return None

    if not contracts.option_contracts:
        log.warning(f"No {option_type} contracts found for {underlying} near "
                    f"{target_expiry} in strike band "
                    f"${strike_lo:.0f}-${strike_hi:.0f}")
        return None

    # Sort by closeness to target strike — we'll walk this list until we
    # find one with a healthy two-sided market.
    candidates = sorted(
        contracts.option_contracts,
        key=lambda c: abs(float(c.strike_price) - target_strike),
    )

    max_spread_pct = CONFIG.get("max_quote_spread_pct", 0.30)
    fallback = None    # remember the closest one in case nothing passes the filter

    for c in candidates[:8]:    # check up to 8 nearby strikes before giving up
        try:
            snap = opt_data_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=[c.symbol])
            )
            quote = snap[c.symbol].latest_quote
            bid, ask = float(quote.bid_price), float(quote.ask_price)
        except Exception:
            bid = ask = 0.0

        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0
        record = {
            "symbol": c.symbol,
            "strike": float(c.strike_price),
            "expiry": c.expiration_date,
            "type":   option_type,
            "bid":    bid,
            "ask":    ask,
            "mid":    mid,
        }

        # Remember the closest contract even if it fails the filter — we'll
        # use it as a last resort rather than returning None.
        if fallback is None:
            fallback = record

        if mid <= 0:
            continue   # no market at all, skip

        spread_pct = (ask - bid) / mid
        if spread_pct <= max_spread_pct:
            if c.symbol != candidates[0].symbol:
                log.info(f"  find_contract: skipped {candidates[0].symbol} "
                         f"(wide spread) → picked {c.symbol} "
                         f"(spread {spread_pct*100:.0f}% ≤ {max_spread_pct*100:.0f}%)")
            return record

    # Nothing passed the quality filter — return the closest strike with a warning
    if fallback is not None:
        log.warning(f"  find_contract: no contract near ${target_strike:.0f} "
                    f"passed bid-ask filter (max {max_spread_pct*100:.0f}%) — "
                    f"using fallback {fallback['symbol']} anyway "
                    f"(bid={fallback['bid']:.2f} ask={fallback['ask']:.2f})")
    return fallback


# ══════════════════════════════════════════════════════════════
#  ORDER ENTRY
# ══════════════════════════════════════════════════════════════

def submit_spread(trade_client, spread_type: str, legs_info: list,
                  contracts: int, net_credit: float) -> tuple:
    """
    Submit a multi-leg credit spread order and wait for fill confirmation.

    Returns (order_id, actual_fill_credit) or (None, None) on failure.
    actual_fill_credit is the true per-share credit received at fill —
    this is what gets stored in state.json as credit_received, NOT the
    mid-price estimate used as the limit.  Using the actual fill ensures
    profit target calculations are always based on real execution prices.

    Fill-reading logic:
      - After submit, poll the order for up to 45 s (3 × 15 s intervals)
      - If the mleg order reports filled_avg_price, use that directly
      - Otherwise, fall back to reading individual leg fills and summing them
      - If the order is still open after 45 s (partial or pending), widen the
        limit by CONFIG["limit_offset"] and retry up to MAX_RETRIES times
    """
    legs = []
    for leg in legs_info:
        legs.append(OptionLegRequest(
            symbol=leg["symbol"],
            side=OrderSide.SELL if leg["side"]=="sell" else OrderSide.BUY,
            ratio_qty=1,
        ))

    limit_price = round(abs(net_credit), 2)
    if limit_price <= 0:
        log.error("  Net credit is zero or negative, cannot submit")
        return None, None

    MAX_RETRIES        = CONFIG.get("max_fill_retries", 3)
    # CREDIT_CONCESSION: we move the limit DOWN (accept less credit) to make
    # the order fill. For credit orders, a smaller limit = easier fill.
    CREDIT_CONCESSION  = CONFIG.get("limit_offset", 0.05)
    POLL_WAIT          = 15   # seconds between fill checks

    for attempt in range(MAX_RETRIES):
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
            order_id = str(order.id)
            log.info(f"  Order submitted: {order_id}  limit=${limit_price}  "
                     f"qty={contracts}  attempt={attempt+1}")
        except Exception as e:
            log.error(f"  Order submission failed (attempt {attempt+1}): {e}")
            return None, None

        # Poll for fill
        filled_credit = None
        for poll in range(3):
            time.sleep(POLL_WAIT)
            try:
                o = trade_client.get_order_by_id(order_id)
                status = str(o.status).lower()

                if "filled" in status or "complete" in status:
                    # ── Method 1: mleg reports filled_avg_price directly ──
                    fp = getattr(o, "filled_avg_price", None)
                    if fp is not None:
                        try:
                            fp_raw = float(fp)
                            # Sign sanity: a credit mleg should produce
                            # negative fp (Alpaca convention) OR positive
                            # depending on SDK version. Log raw so we can audit.
                            filled_credit = abs(fp_raw)
                            log.info(f"  Fill confirmed (mleg avg): "
                                     f"raw={fp_raw:+.4f} → credit=${filled_credit:.4f}/share  "
                                     f"(limit ${limit_price:.2f}, "
                                     f"slippage {(limit_price-filled_credit)/limit_price*100:.1f}%)")
                            break
                        except (TypeError, ValueError):
                            pass

                    # ── Method 2: sum individual leg fills ─────────────────
                    # Try parent order's .legs attribute first; if that's
                    # empty/missing, query child orders by parent_id.
                    leg_fills = {}
                    try:
                        legs_data = getattr(o, "legs", []) or []
                        for l in legs_data:
                            sym  = str(l.symbol)
                            side = str(l.side).lower()
                            fp_l = getattr(l, "filled_avg_price", None)
                            if fp_l is not None:
                                leg_fills[sym] = ("sell" in side, float(fp_l))
                    except Exception:
                        pass

                    # Fallback: query all orders linked by parent_id
                    if not leg_fills:
                        try:
                            from alpaca.trading.requests import GetOrdersRequest as _GOR
                            children = trade_client.get_orders(
                                filter=_GOR(parent_id=order_id)
                            ) if hasattr(trade_client, "get_orders") else []
                            for ch in children:
                                fp_l = getattr(ch, "filled_avg_price", None)
                                if fp_l is None:
                                    continue
                                sym  = str(ch.symbol)
                                side = str(ch.side).lower()
                                leg_fills[sym] = ("sell" in side, float(fp_l))
                        except Exception as _e:
                            log.debug(f"  parent_id leg query failed: {_e}")

                    if leg_fills:
                        total = sum(
                            fp if is_sell else -fp
                            for is_sell, fp in leg_fills.values()
                        )
                        filled_credit = total
                        log.info(f"  Fill confirmed (leg sum): "
                                 f"${filled_credit:.4f}/share  "
                                 f"(limit was ${limit_price:.2f}, "
                                 f"slippage {(limit_price-filled_credit)/limit_price*100:.1f}%)")
                        break

                    # Filled but can't read price — use limit as fallback
                    filled_credit = limit_price
                    log.warning(f"  Order filled but could not read fill price — "
                                f"using limit ${limit_price:.2f} as fallback")
                    break

                elif "cancel" in status or "reject" in status or "expired" in status:
                    log.warning(f"  Order {order_id} {status}")
                    filled_credit = None
                    break

                else:
                    log.info(f"  Order {order_id} status={status} "
                             f"(poll {poll+1}/3, waiting {POLL_WAIT}s...)")

            except Exception as e:
                log.error(f"  Could not get order status: {e}")

        if filled_credit is not None and filled_credit > 0:
            return order_id, filled_credit

        # Not filled — concede credit (lower the limit) and retry.
        # For credit orders: limit ↓ → fills easier.
        if attempt < MAX_RETRIES - 1:
            limit_price = round(limit_price - CREDIT_CONCESSION, 2)
            log.info(f"  Not filled — conceding ${CREDIT_CONCESSION:.2f} of credit, "
                     f"new limit ${limit_price:.2f} (attempt {attempt+2}/{MAX_RETRIES})")
            try:
                trade_client.cancel_order_by_id(order_id)
            except Exception:
                pass

    log.error(f"  Spread order not filled after {MAX_RETRIES} attempts")
    return None, None


def check_assignment(trade_client, state: dict, today: date) -> list:
    """
    Scan Alpaca positions for evidence of option assignment.

    Assignment turns a short option into a stock position. When detected:
      - The stock position itself is flagged (the bot cannot manage it).
      - The affected slot in state.json is flagged with assignment_alert.
      - A WARNING is written to the log.
      - A list of alert strings is returned so the caller can send a push.
      - The alert is written to assignment_alert.txt only once per assignment
        (tracked by symbol+qty hash in state["seen_assignments"]); subsequent
        daily runs won't re-spam the file.

    The bot does NOT attempt to close stock positions automatically —
    assignment requires human judgement.

    Returns: list of alert message strings (empty = no assignment detected).
    """
    alerts = []
    try:
        positions = trade_client.get_all_positions()
    except Exception as e:
        log.warning(f"  Assignment check: could not read positions: {e}")
        return alerts

    seen = set(state.get("seen_assignments", []))

    # Tracked tickers: anything our slots use. Currently IWM-only.
    TRACKED = {sid.split("_")[0] for sid in state.get("slots", {}).keys()} or {"IWM"}

    for pos in positions:
        sym = str(pos.symbol).upper()
        if sym not in TRACKED:
            continue

        qty       = float(pos.qty)
        mkt_val   = float(getattr(pos, "market_value", 0) or 0)
        unreal_pl = float(getattr(pos, "unrealized_pl", 0) or 0)
        price     = float(getattr(pos, "current_price", 0) or 0)
        direction = "LONG" if qty > 0 else "SHORT"

        msg = (f"ASSIGNMENT DETECTED: {sym} stock position found  "
               f"{direction} {abs(qty):.0f} shares @ ${price:.2f}  "
               f"market_value=${mkt_val:,.0f}  P&L=${unreal_pl:,.0f}  "
               f"— manual intervention required: exercise protective leg "
               f"or close stock position NOW")
        log.warning(f"  {msg}")
        alerts.append(msg)

        # Dedupe key: ticker + signed-qty captures both the "what" and a
        # natural identity. If the assignment changes (partial close, etc.)
        # this key changes and we'll re-alert.
        dedupe_key = f"{sym}:{direction}:{int(abs(qty))}:{today.isoformat()[:7]}"
        if dedupe_key not in seen:
            try:
                alert_path = Path("assignment_alert.txt")
                with open(alert_path, "a") as f:
                    f.write(f"[{datetime.now().isoformat()}] {msg}\n")
            except Exception:
                pass
            seen.add(dedupe_key)

        # Mark the affected slot(s) in state
        for slot_id, slot in state.get("slots", {}).items():
            if slot_id.startswith(sym) and slot.get("position"):
                slot["position"]["assignment_alert"] = today.isoformat()

    state["seen_assignments"] = sorted(seen)
    if not alerts:
        log.info("  Assignment check: no stock positions found — all clear")
    return alerts


def _cancel_and_wait(trade_client, order_id: str, max_wait_s: int = 20):
    """
    Cancel an order and poll until it reaches a terminal state, so that the
    contracts it was holding (Alpaca 'held_for_orders') are released back to
    available quantity.

    Returns the final lowercased status string (e.g. 'canceled', 'filled')
    or None if the terminal state could not be confirmed in time.

    This MUST be called before any fallback order is placed on the same
    legs. An un-cancelled working order keeps the contracts reserved, and a
    follow-up order then fails with 'insufficient qty available'
    (held_for_orders == position size, available == 0).
    """
    try:
        trade_client.cancel_order_by_id(order_id)
    except Exception as e:
        # A cancel can legitimately fail if the order already terminal —
        # we still poll below to find out its real state.
        log.warning(f"  Cancel request for {order_id} failed: {e}")
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        time.sleep(2)
        try:
            o = trade_client.get_order_by_id(order_id)
            st = str(o.status).lower()
            if any(k in st for k in ("cancel", "fill", "expired",
                                     "rejected", "done")):
                log.info(f"  Order {order_id} reached terminal state "
                         f"'{st}' — contracts released")
                return st
        except Exception:
            # Order no longer retrievable → treat as gone (qty released)
            log.info(f"  Order {order_id} no longer retrievable — "
                     f"treating as cancelled")
            return "canceled"
    log.warning(f"  Order {order_id} not confirmed terminal after "
                f"{max_wait_s}s — a fallback order may still fail")
    return None


def _close_legs_mleg_limit(trade_client, opt_data_client,
                           legs_to_close: list, contracts: int) -> tuple:
    """
    Close a set of legs as a single MLEG limit order, walking the debit
    limit toward market on retries. Mirrors the entry-side submit_spread
    ladder so the close side gets the same slippage protection.

    legs_to_close : list of (symbol, is_short) tuples
    Returns (success: bool, closed_symbols: list).
    A False return means the caller should fall back to market orders.
    """
    if not legs_to_close:
        return True, []

    # Build the closing MLEG legs — reverse of the original sides:
    #   short leg (was sold)  → BUY  to close
    #   long  leg (was bought)→ SELL to close
    close_legs = []
    for sym, is_short in legs_to_close:
        close_legs.append(OptionLegRequest(
            symbol=sym,
            side=OrderSide.BUY if is_short else OrderSide.SELL,
            ratio_qty=1,
        ))

    # Estimate the mid debit (what it costs to buy the spread back) from quotes
    syms = [s for s, _ in legs_to_close]
    mid_debit = None
    try:
        snaps = opt_data_client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=syms))
        debit = 0.0
        ok = True
        for sym, is_short in legs_to_close:
            snap = snaps.get(sym) if hasattr(snaps, "get") else snaps[sym]
            q = getattr(snap, "latest_quote", None) if snap else None
            if q is None:
                ok = False; break
            bid = float(q.bid_price); ask = float(q.ask_price)
            if bid <= 0 and ask <= 0:
                ok = False; break
            mid = (bid + ask) / 2.0
            # short → buy back → adds to debit; long → sell → reduces debit
            debit += mid if is_short else -mid
        if ok:
            mid_debit = max(0.01, round(debit, 2))
    except Exception as e:
        log.debug(f"  Close-quote fetch failed: {e}")

    if mid_debit is None:
        # Could not price the spread — signal caller to use market fallback
        log.warning("  Could not price spread for limit close — "
                    "will fall back to market orders")
        return False, []

    MAX_RETRIES = CONFIG.get("max_fill_retries", 3)
    CONCESSION  = CONFIG.get("limit_offset", 0.05)
    POLL_WAIT   = 15
    limit_price = mid_debit

    for attempt in range(MAX_RETRIES):
        try:
            order = trade_client.submit_order(
                LimitOrderRequest(
                    qty=contracts,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.MLEG,
                    limit_price=limit_price,
                    legs=close_legs,
                )
            )
            order_id = str(order.id)
            log.info(f"  Close order submitted: {order_id}  "
                     f"debit limit=${limit_price:.2f}  qty={contracts}  "
                     f"attempt={attempt+1}/{MAX_RETRIES}")
        except Exception as e:
            log.error(f"  Close order submission failed (attempt {attempt+1}): {e}")
            return False, []

        # Poll for fill (3 × 15 s = 45 s, same cadence as entry)
        filled = False
        for poll in range(3):
            time.sleep(POLL_WAIT)
            try:
                o = trade_client.get_order_by_id(order_id)
                status = str(o.status).lower()
                if "filled" in status or "complete" in status:
                    fp = getattr(o, "filled_avg_price", None)
                    if fp is not None:
                        try:
                            paid = abs(float(fp))
                            slip = ((paid - mid_debit) / mid_debit * 100
                                    if mid_debit > 0 else 0.0)
                            log.info(f"  Close filled: debit ${paid:.4f}/share  "
                                     f"(mid was ${mid_debit:.2f}, "
                                     f"slippage {slip:+.1f}%)")
                        except (TypeError, ValueError):
                            log.info(f"  Close filled (limit ${limit_price:.2f})")
                    else:
                        log.info(f"  Close filled (price unread, "
                                 f"limit ${limit_price:.2f})")
                    filled = True
                    break
                elif "cancel" in status or "reject" in status or "expired" in status:
                    log.warning(f"  Close order {order_id} {status}")
                    break
                else:
                    log.info(f"  Close order status={status} "
                             f"(poll {poll+1}/3, waiting {POLL_WAIT}s...)")
            except Exception as e:
                log.error(f"  Could not get close order status: {e}")

        if filled:
            return True, syms

        # Not filled. CANCEL this order and WAIT for the contracts to be
        # released — on EVERY attempt, including the last. Leaving a
        # working order alive holds the legs (held_for_orders) and blocks
        # the market-order fallback with 'insufficient qty available'.
        final_status = _cancel_and_wait(trade_client, order_id)
        if final_status is not None and "fill" in final_status:
            # Race: the order filled between the last poll and the cancel.
            # The position IS closed — report success.
            log.info(f"  Close order {order_id} filled during cancellation "
                     f"— close succeeded")
            return True, syms

        # Concede for the next attempt (if any): RAISE the debit limit.
        if attempt < MAX_RETRIES - 1:
            limit_price = round(limit_price + CONCESSION, 2)
            log.info(f"  Close not filled — conceding ${CONCESSION:.2f} more, "
                     f"new debit limit ${limit_price:.2f} "
                     f"(attempt {attempt+2}/{MAX_RETRIES})")

    log.warning(f"  Close MLEG limit not filled after {MAX_RETRIES} attempts — "
                f"order cancelled, caller will fall back to market orders")
    return False, []


def close_position_by_legs(trade_client, opt_data_client,
                           position_state: dict) -> tuple:
    """
    Close an existing spread position — slippage-optimized.

    Execution strategy:
      1. Read current Alpaca positions + quote each leg.
      2. Legs worth < MIN_CLOSE_VALUE/share are left to expire worthless
         (no closing fee — letting a $0.01 option expire beats paying to close it).
      3. Legs with real value are closed as a single MLEG LIMIT order whose
         debit limit walks toward market on retries — this mirrors the
         entry-side ladder and pays the spread's combined bid-ask ONCE
         instead of paying each leg's bid-ask separately via market orders.
      4. If the MLEG limit can't fill (or quotes are unavailable), fall back
         to per-leg market orders so the position is always flattened.

    Returns (success, failed_legs):
      - success: True iff every leg was confirmed closed (or expired/missing)
      - failed_legs: list of leg symbols that could not be closed

    Best-effort semantics: the slot is freed even when some legs failed,
    but the caller uses failed_legs to tag the trade as a partial close.
    """
    MIN_CLOSE_VALUE = 0.03   # $/share — don't chase options below this
    leg_symbols  = position_state.get("leg_symbols", [])
    leg_sides    = position_state.get("leg_sides",   [])
    contracts    = position_state.get("contracts",   1)
    closed_count = 0
    failed_legs  = []

    # Read current position sizes from Alpaca
    try:
        open_pos = {p.symbol: p for p in trade_client.get_all_positions()}
    except Exception:
        open_pos = {}

    # Partition legs: already-gone / worthless (let expire) / worth-closing
    worth_closing = []   # list of (symbol, is_short)
    for sym, is_short in zip(leg_symbols, leg_sides):
        if sym not in open_pos:
            log.info(f"  Leg {sym}: not in Alpaca positions — already closed/expired")
            closed_count += 1
            continue
        try:
            pos_val   = abs(float(open_pos[sym].market_value))
            per_share = pos_val / (abs(float(open_pos[sym].qty)) * 100)
            if per_share < MIN_CLOSE_VALUE:
                log.info(f"  Leg {sym}: value ${per_share:.3f}/share < "
                         f"${MIN_CLOSE_VALUE} — letting expire worthless")
                closed_count += 1
                continue
        except Exception:
            pass
        worth_closing.append((sym, is_short))

    if worth_closing:
        # ── Try MLEG limit close first (slippage-optimized) ─────────
        ok, closed_syms = _close_legs_mleg_limit(
            trade_client, opt_data_client, worth_closing, contracts)
        if ok:
            closed_count += len(closed_syms)
            log.info(f"  MLEG limit close succeeded — {len(closed_syms)} legs")
        else:
            # ── Fallback: per-leg market orders (guarantees flat) ───
            # CRITICAL ORDERING: close SHORT legs first, long legs last.
            # Selling a long (protective) leg while its short is still
            # open momentarily creates a NAKED SHORT → Alpaca rejects it
            # ('account not eligible to trade uncovered option contracts'
            #  / 'insufficient buying power for cash-secured put').
            # Buying back the shorts first is always margin-safe; the
            # remaining longs can then be sold freely.
            log.warning("  Falling back to per-leg market orders for close "
                        "(shorts first, then longs)")
            ordered = sorted(worth_closing,
                             key=lambda x: 0 if x[1] else 1)  # x[1] = is_short
            for sym, is_short in ordered:
                done = False
                for attempt in range(3):
                    try:
                        trade_client.close_position(sym)
                        log.info(f"  Closed leg (market fallback): {sym} "
                                 f"({'short' if is_short else 'long'})")
                        closed_count += 1
                        done = True
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(1.5)
                        else:
                            log.error(f"  Failed to close {sym} after 3 attempts: {e}")
                if not done:
                    failed_legs.append(sym)

    total = len(leg_symbols)
    success = len(failed_legs) == 0

    if failed_legs:
        log.warning(f"  Partial close: {closed_count}/{total} legs closed. "
                    f"Failed: {failed_legs}. Freeing slot anyway — "
                    f"check Alpaca manually for remaining legs.")
    else:
        log.info(f"  All {closed_count}/{total} legs closed/expired successfully.")

    return success, failed_legs


# ══════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════

# Market-impact cap (0.1% of avg daily options volume).
# IWM activates at ~$1.4M portfolio  (500K daily vol × 0.1%).
# This only activates at very large portfolio sizes — a safety rail against
# bugs, not a growth limiter for normal account sizes.
# Other tickers can be added here if the portfolio ever expands beyond IWM.
MARKET_IMPACT_CAP = {
    "IWM":  500,   # 0.1% of ~500K daily options volume
    "XLE":  300,   # 0.1% of ~300K daily options volume on XLE (less liquid than IWM)
}
DEFAULT_CONTRACT_CAP = 50


def compute_contracts(portfolio_value: float, max_loss_per_spread: float,
                      slot_risk: float = None, ticker: str = "IWM") -> int:
    """
    Number of contracts = floor(slot_budget / max_loss_per_contract).

    No hard contract cap — contracts scale naturally with portfolio so
    the strategy compounds without an artificial growth ceiling.

    A generous ticker-specific market-impact cap acts as a safety rail
    against code bugs, not a growth limiter. At any realistic retail
    portfolio size the formula-driven count will be well below these caps.

    Contract counts at typical portfolio sizes (IWM, ~$10 put width, $0.90 credit):
      $10K:  ~2 contracts per slot
      $25K:  ~5 contracts per slot
      $50K:  ~10 contracts per slot
      $100K: ~19 contracts per slot
      $250K: ~48 contracts per slot
    """
    if max_loss_per_spread <= 0:
        return 0
    risk   = slot_risk if slot_risk else CONFIG["iwm_slot_risk"]
    budget = portfolio_value * risk
    n      = int(budget / (max_loss_per_spread * 100))
    cap    = MARKET_IMPACT_CAP.get(ticker, DEFAULT_CONTRACT_CAP)
    return max(1, min(n, cap))


# ══════════════════════════════════════════════════════════════
#  (SPY allocation manager removed — portfolio is 100% IWM options)
# ══════════════════════════════════════════════════════════════
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

    NOTE (#28): IV is re-fit per slot via brentq on every run. With 4 slots
    that's 4 brentq calls × 1ms each — negligible at current scale. If this
    ever expands to many tickers/slots, consider caching the implied vol per
    expiry across slots or fitting once on a single representative leg.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("  matplotlib not installed — skipping daily diagrams")
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
        for t in ["IWM", "XLE", "XBI", "QQQ", "SPY"]:
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

        # Get live underlying price (uses cached _fetch_history for #13)
        try:
            raw = _fetch_history(ticker, lookback_days=10)
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

    # ── Build equity curve from state["equity_history"] ──────────
    # equity_history is appended every daily run, giving one point per
    # trading day. Falls back to trade_log.csv close events if history
    # is empty (e.g. before first run after upgrade).
    eq_history = state.get("equity_history", []) or []

    dates_list  = []
    equity_list = []

    if eq_history:
        # Sort by date in case of out-of-order entries
        sorted_history = sorted(eq_history, key=lambda r: r.get("date", ""))
        for entry in sorted_history:
            try:
                d = date.fromisoformat(str(entry["date"]))
                v = float(entry["value"])
                dates_list.append(d)
                equity_list.append(v)
            except Exception:
                continue

    # If equity_history is empty (first run after upgrade, no points yet),
    # fall back to reconstructing from trade_log.csv close events
    if not dates_list and log_path.exists():
        try:
            log_df = pd.read_csv(log_path, on_bad_lines="skip")
            log_df["date"] = pd.to_datetime(log_df["date"], errors="coerce").dt.date
            log_df = log_df.dropna(subset=["date"])
            closed = log_df[log_df.get("action", "") == "close"].copy() \
                     if "action" in log_df.columns else log_df.copy()
            if "realized_pnl" in closed.columns:
                closed["realized_pnl"] = pd.to_numeric(
                    closed["realized_pnl"], errors="coerce").fillna(0)
                closed = closed.sort_values("date")
                dates_list.append(date.fromisoformat(str(state.get("start_date", today))))
                equity_list.append(initial)
                running = initial
                for _, row in closed.iterrows():
                    running += row["realized_pnl"]
                    dates_list.append(row["date"])
                    equity_list.append(running)
        except Exception as e:
            log.warning(f"  Could not read trade log for portfolio chart: {e}")

    # Seed initial point if list is still empty (first ever run)
    if not dates_list:
        dates_list.append(today)
        equity_list.append(initial)

    # Always ensure today's actual Alpaca portfolio value is the last point
    if dates_list[-1] != today:
        dates_list.append(today)
        equity_list.append(portfolio_value)
    else:
        equity_list[-1] = portfolio_value

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

    date_nums = [datetime.combine(d, datetime.min.time()) for d in dates_list]

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

    # Count actual closed trades from the log (robust to malformed rows)
    n_trades = 0
    if log_path.exists():
        try:
            log_df_for_count = pd.read_csv(log_path, on_bad_lines="skip")
            if "action" in log_df_for_count.columns:
                n_trades = int((log_df_for_count["action"] == "close").sum())
        except Exception:
            pass

    # Header stats
    stats_txt = (f"Total P&L: {'+'if total_pnl>=0 else ''}${total_pnl:,.0f}  |  "
                 f"Return: {total_ret:+.1f}%  |  "
                 f"Max DD: {max_dd:.1f}%  |  "
                 f"CAGR: {cagr:.0f}%  |  "
                 f"Trades: {n_trades}")
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
                today: date, state: dict = None) -> dict:
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
        success, failed_legs = close_position_by_legs(trade_client, opt_data_client, pos)
        total_legs = len(pos.get("leg_symbols", []))
        if failed_legs and total_legs and len(failed_legs) >= total_legs:
            # NOTHING closed — the position is fully intact. Do NOT record a
            # P&L and do NOT free the slot (that would desync state from
            # Alpaca and log a phantom result). Keep the position; it will
            # be retried next run. Alert URGENTLY — this is near expiry.
            log.error(f"  Slot {slot_id}: DTE-exit close FAILED ENTIRELY "
                      f"({len(failed_legs)}/{total_legs} legs still open). "
                      f"Position INTACT, slot NOT freed — will retry next run. "
                      f"NEAR EXPIRY: CHECK ALPACA MANUALLY NOW.")
            if state is not None:
                state.setdefault("close_failures", []).append({
                    "slot": slot_id, "date": today.isoformat(),
                    "kind": "dte_exit", "legs": failed_legs})
        else:
            pnl = _estimate_close_pnl(opt_data_client, pos)
            reason = "dte_exit" if success else "dte_exit_partial"
            _record_close(slot_state, today, reason, pnl,
                          slot_id=slot_id, state=state,
                          failed_legs=failed_legs)

    # Rule 2: profit target
    elif days_held >= CONFIG["min_hold_days"]:
        current_value = _get_spread_value(opt_data_client, pos)
        if current_value is not None:
            profit = pos["credit_received"] - current_value
            target = pos["credit_received"] * CONFIG["profit_target"]
            if profit >= target:
                log.info(f"  Slot {slot_id}: profit target  "
                         f"profit=${profit:.2f} >= target=${target:.2f}")
                success, failed_legs = close_position_by_legs(trade_client, opt_data_client, pos)
                total_legs = len(pos.get("leg_symbols", []))
                if failed_legs and total_legs and len(failed_legs) >= total_legs:
                    # NOTHING closed — position fully intact. Do NOT log a
                    # phantom profit and do NOT free the slot. Keep the
                    # position; profit target will re-trigger next run with
                    # the (now fixed) close path.
                    log.error(f"  Slot {slot_id}: profit-target close FAILED "
                              f"ENTIRELY ({len(failed_legs)}/{total_legs} legs "
                              f"still open). Position INTACT, slot NOT freed, "
                              f"no P&L recorded — will retry next run. "
                              f"CHECK ALPACA MANUALLY.")
                    if state is not None:
                        state.setdefault("close_failures", []).append({
                            "slot": slot_id, "date": today.isoformat(),
                            "kind": "profit_target", "legs": failed_legs})
                else:
                    pnl = profit * pos["contracts"] * 100
                    reason = "profit_target" if success else "profit_target_partial"
                    _record_close(slot_state, today, reason, pnl,
                                  slot_id=slot_id, state=state,
                                  failed_legs=failed_legs)

    return slot_state


def _get_spread_value(opt_data_client, pos: dict) -> float:
    """
    Get current per-share value of the spread (i.e. what it would cost to
    buy back). For an open spread this is the sum of mid prices weighted
    by leg side (short = +mid for the closer, long = -mid).

    Returns None on quote failure for a live position.

    NOTE: This function should NOT be called for expired positions —
    use _terminal_spread_value() for those.
    """
    try:
        syms = pos.get("leg_symbols", [])
        if not syms:
            return None
        snaps = opt_data_client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=syms)
        )
        total = 0.0
        for sym, is_short in zip(syms, pos.get("leg_sides", [])):
            if sym not in snaps:
                # Quote missing — could be expired or just unavailable
                log.warning(f"  _get_spread_value: missing snapshot for {sym}")
                return None
            q = snaps[sym].latest_quote
            mid = (float(q.bid_price) + float(q.ask_price)) / 2
            total += -mid if is_short else mid
        return max(-total, 0.0)
    except Exception as e:
        log.error(f"  Could not get spread value: {e}")
        return None


def _parse_strike_from_occ(sym: str) -> tuple:
    """
    Parse the option type and strike from an OCC symbol.
    Format: TICKERYYMMDD[C|P]NNNNNNNN where the last 8 digits are strike × 1000.
    Returns (opt_type, strike) or ("?", 0.0) on parse error.
    """
    try:
        # The C/P is at position -9 from the end (8 digits of strike follow)
        ct  = sym[-9]
        opt = "call" if ct == "C" else "put"
        k   = int(sym[-8:]) / 1000.0
        return opt, k
    except Exception:
        return "?", 0.0


def _terminal_spread_value(pos: dict, underlying_close: float) -> float:
    """
    Compute the terminal per-share value of the spread at expiry, given the
    underlying close price. Each leg's payoff is intrinsic value at expiry:
      put:  max(K - S, 0)
      call: max(S - K, 0)
    Short legs add a cost-to-close (positive), long legs reduce it.
    """
    total = 0.0
    for sym, is_short in zip(pos.get("leg_symbols", []), pos.get("leg_sides", [])):
        opt, k = _parse_strike_from_occ(sym)
        if opt == "put":
            payoff = max(k - underlying_close, 0.0)
        elif opt == "call":
            payoff = max(underlying_close - k, 0.0)
        else:
            log.warning(f"  _terminal_spread_value: could not parse {sym}")
            payoff = 0.0
        total += payoff if is_short else -payoff
    return max(total, 0.0)


def _fetch_underlying_close(ticker: str) -> float:
    """Pull the most recent daily close for the underlying."""
    try:
        ser = _fetch_history(ticker, lookback_days=10)
        return float(ser.dropna().iloc[-1])
    except Exception as e:
        log.error(f"  _fetch_underlying_close({ticker}) failed: {e}")
        return 0.0


def _estimate_close_pnl(opt_data_client, pos: dict) -> float:
    """
    Best estimate of realized P&L at close. For live positions: use current
    quotes. For expired positions (DTE ≤ 0): use terminal payoff from the
    underlying's close price vs strikes — this is the correct accounting and
    fixes the bug where every expired close logged $0 P&L.
    """
    try:
        expiry = date.fromisoformat(str(pos.get("expiry", "")))
        dte = (expiry - date.today()).days
    except Exception:
        dte = 99   # unknown; treat as live

    if dte <= 0:
        # Expired: compute terminal value from underlying close.
        # Infer ticker from leg symbols.
        leg_syms = pos.get("leg_symbols", [])
        ticker = "IWM"
        if leg_syms:
            for t in ["IWM", "XLE", "XBI", "QQQ", "SPY"]:
                if leg_syms[0].startswith(t):
                    ticker = t
                    break
        S_close = _fetch_underlying_close(ticker)
        if S_close <= 0:
            log.warning("  _estimate_close_pnl: could not get underlying close; reporting $0")
            return 0.0
        terminal_val = _terminal_spread_value(pos, S_close)
        pnl_per_share = pos["credit_received"] - terminal_val
        log.info(f"  Terminal P&L: {ticker} close=${S_close:.2f}  "
                 f"terminal_val=${terminal_val:.4f}  "
                 f"credit=${pos['credit_received']:.4f}  "
                 f"per_share=${pnl_per_share:+.4f}")
        return pnl_per_share * pos["contracts"] * 100

    # Live: quote-based valuation
    val = _get_spread_value(opt_data_client, pos)
    if val is None:
        log.warning("  _estimate_close_pnl: live quote unavailable; reporting $0")
        return 0.0
    return (pos["credit_received"] - val) * pos["contracts"] * 100


def _record_close(slot_state: dict, today: date, reason: str, pnl: float,
                  slot_id: str = "", state: dict = None,
                  failed_legs: list = None):
    """Record close in slot_state, write to trade_log.csv, update counters."""
    failed_legs = failed_legs or []
    pos = slot_state["position"]
    pos["close_date"]   = today.isoformat()
    pos["close_reason"] = reason
    pos["realized_pnl"] = pnl
    if failed_legs:
        pos["failed_legs"] = failed_legs

    slot_state["closed_trades"] = slot_state.get("closed_trades", [])
    slot_state["closed_trades"].append(dict(pos))
    slot_state["position"]   = None
    slot_state["next_entry"] = _next_trading_day(today)

    try:
        entry_d = date.fromisoformat(pos["entry_date"])
        days_h  = (today - entry_d).days
    except Exception:
        days_h  = ""

    log_trade({
        "date":         today.isoformat(),
        "action":       "close",
        "slot":         slot_id,
        "spread":       pos.get("spread", ""),
        "trend":        pos.get("trend_regime", ""),
        "vol":          pos.get("vol_regime", ""),
        "atr":          pos.get("atr_regime", ""),
        "contracts":    pos.get("contracts", 0),
        "credit":       pos.get("credit_received", 0),
        "max_loss":     pos.get("max_loss", 0),
        "close_reason": reason,
        "realized_pnl": round(pnl, 2),
        "entry_date":   pos.get("entry_date", ""),
        "days_held":    days_h,
        "order_id":     pos.get("order_id", ""),
    })

    if state is not None:
        state["cumulative_pnl"] = round(state.get("cumulative_pnl", 0.0) + pnl, 2)
        state["trade_count"]    = state.get("trade_count", 0) + 1
        log.info(f"  Closed: {reason}  P&L=${pnl:,.2f}  |  "
                 f"cumulative=${state['cumulative_pnl']:,.2f}  "
                 f"trades={state['trade_count']}")
        msg_extra = ""
        priority  = "high" if pnl >= 0 else "urgent"
        tags      = "chart_with_upwards_trend" if pnl >= 0 else "warning"
        if failed_legs:
            msg_extra = f"  ⚠ {len(failed_legs)} leg(s) FAILED to close: {failed_legs}"
            priority  = "urgent"
            tags      = "rotating_light,warning"
        send_alert(
            title=f"VRP: {slot_id} closed ({'profit' if pnl >= 0 else 'LOSS'})",
            message=(f"{slot_id}: closed {reason}  P&L=${pnl:,.2f}  "
                     f"cumulative=${state['cumulative_pnl']:,.2f}  "
                     f"trades={state['trade_count']}{msg_extra}"),
            priority=priority,
            tags=tags,
        )
    else:
        log.info(f"  Closed: {reason}  P&L=${pnl:,.2f}")


# ══════════════════════════════════════════════════════════════
#  SLOT ENTRY
# ══════════════════════════════════════════════════════════════

# Full-day market closures (NYSE). Keep this list rolling 2 years forward.
# Update annually — sourced from https://www.nyse.com/markets/hours-calendars
NYSE_HOLIDAYS = {
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
    # 2027
    "2027-01-01",
    "2027-01-18",
    "2027-02-15",
    "2027-03-26",
    "2027-05-31",
    "2027-06-18",  # Juneteenth observed (Sat)
    "2027-07-05",  # Independence Day observed
    "2027-09-06",
    "2027-11-25",
    "2027-12-24",  # Christmas observed (Sat)
}

# Half-day closes (1:00 PM ET): day before Independence Day, day after
# Thanksgiving, Christmas Eve when on a weekday.
NYSE_EARLY_CLOSE = {
    "2026-07-02",  # July 3 observed full closure, so this is N/A; placeholder
    "2026-11-27",  # day after Thanksgiving
    "2026-12-24",  # Christmas Eve
    "2027-11-26",
    "2027-12-23",  # Dec 24 falls on Friday but closed in observance of Christmas
}


def is_market_open() -> bool:
    """
    Return True if US options market is currently open.
    Accounts for weekends, full-day NYSE holidays, and half-day closes (1pm ET).
    """
    if zoneinfo is not None:
        et = zoneinfo.ZoneInfo("America/New_York")
    else:
        # fallback: UTC-5 (ignores DST — fine for is-it-open check, edge cases rare)
        et = timezone(timedelta(hours=-5))

    now_et = datetime.now(et)
    if now_et.weekday() > 4:
        return False

    iso = now_et.date().isoformat()
    if iso in NYSE_HOLIDAYS:
        return False

    market_open = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    if iso in NYSE_EARLY_CLOSE:
        market_close = now_et.replace(hour=13, minute=0, second=0, microsecond=0)
    else:
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_et <= market_close


def _next_trading_day(d) -> str:
    """
    Return the next calendar date that is a weekday AND not an NYSE holiday.
    Used wherever next_entry is set so slots never get gated on a non-trading
    day that would silently delay entry by extra days.
    """
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5 or nxt.isoformat() in NYSE_HOLIDAYS:
        nxt += timedelta(days=1)
    return nxt.isoformat()


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
        slot_state["next_entry"] = _next_trading_day(today)
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
            slot_state["next_entry"] = _next_trading_day(today)
            return slot_state
    except Exception:
        pass

    # ── Strategy selection from regime map + indicator gates ─────
    # Replaces the old skip-2 gate. select_strategy returns the spread
    # type, the target deltas, and a skip_reason if entry should be
    # blocked (either the regime maps to SKIP, or the RSI gate is firing
    # on a CCS-candidate cell).
    decision = select_strategy(
        regime,
        vix_fresh_high=regime.get("vix_fresh_high", False),
        rsi_today=regime.get("rsi", 50.0),
        cfg=cfg,
        vix_rising=regime.get("vix_rising", False),
        down_days=regime.get("down_days", 0),
    )

    if decision["skip_reason"] is not None:
        log.info(f"  Slot {slot_id}: SKIP — {decision['skip_reason']}.  "
                 f"Holding cash, will retry tomorrow.")
        slot_state["next_entry"] = _next_trading_day(today)
        return slot_state

    spread_name = decision["spread"]
    # Build params from decision — call_delta may have been widened by
    # panic-peak detection, so we override SPREAD_PARAMS for this entry.
    params = {
        "put_delta":      decision["put_delta"],
        "call_delta":     decision["call_delta"],
        "put_width_mult": SPREAD_PARAMS[spread_name]["put_width_mult"],
    }
    log.info(f"  Slot {slot_id}: select_strategy → {spread_name}  "
             f"put_delta={params['put_delta']:.2f}  "
             f"call_delta={params['call_delta']:.2f}  "
             f"(RSI={regime.get('rsi', 0):.1f}, "
             f"vix_fresh_high={regime.get('vix_fresh_high', False)})")

    S           = regime["price"]
    iv          = regime["iv"]   # initial HV×factor estimate
    rfr         = get_risk_free_rate()
    T           = cfg["slot_dte"] / 365

    target_expiry = today + timedelta(days=cfg["slot_dte"])
    log.info(f"  Slot {slot_id}: entering {spread_name}  "
             f"regime={regime['trend']}+{regime['vol']}+{regime['atr']}  "
             f"rfr={rfr*100:.2f}%  iv_est={iv:.3f}  target_expiry={target_expiry}")

    # ── Wing width — shared across put and call sides ────────────
    # Scales with implied vol so C/R ratio stays consistent across regimes.
    # Formula: wing = S × IV × sqrt(DTE/252) × VOL_WING_MULT
    # Clamp: min $5, max 10% of S (prevents absurd wings at extreme IV)
    VOL_WING_MULT = 0.80   # 0.8 × 1-SD move = conservative wing boundary
    iv_wing = S * iv * math.sqrt(cfg["slot_dte"] / 252) * VOL_WING_MULT

    # Initialise leg containers — populated below depending on spread type
    legs_info   = []
    leg_symbols = []
    leg_sides   = []
    net_credit  = 0.0
    short_put = long_put = short_call = long_call = None
    put_width = call_width = 0.0

    # ── PUT SIDE (for iron_condor and put_credit_spread) ─────────
    if params["put_delta"] > 0:
        put_short_target = find_strike_by_delta(
            S, rfr, iv, T, params["put_delta"], "put")

        # IV refinement (#15): probe candidate, solve for implied vol,
        # re-pick strike. Corrects for IV ≠ HV×factor in shifted surfaces.
        try:
            probe = find_contract(trade_client, opt_data_client, ticker, "put",
                                  params["put_delta"], put_short_target, target_expiry)
            if probe and probe.get("mid", 0) > 0 and probe.get("strike", 0) > 0:
                iv_implied = _implied_vol_from_price(
                    S=S, K=probe["strike"], T=T, r=rfr,
                    price=probe["mid"], opt_type="put",
                )
                if iv_implied is not None and 0.05 < iv_implied < 2.5:
                    log.info(f"  IV refined: HV-est={iv:.3f} → implied={iv_implied:.3f} "
                             f"(from probe {probe['symbol']} mid=${probe['mid']:.2f})")
                    iv = iv_implied
                    # Recompute wing with refined IV
                    iv_wing = S * iv * math.sqrt(cfg["slot_dte"] / 252) * VOL_WING_MULT
                    put_short_target = find_strike_by_delta(
                        S, rfr, iv, T, params["put_delta"], "put")
        except Exception as e:
            log.debug(f"  IV refinement skipped: {e}")

        put_width = max(5.0, min(round(iv_wing * params["put_width_mult"]),
                                  round(S * 0.10)))
        put_long_target = put_short_target - put_width

        short_put = find_contract(trade_client, opt_data_client, ticker, "put",
                                  params["put_delta"], put_short_target, target_expiry)
        long_put  = find_contract(trade_client, opt_data_client, ticker, "put",
                                  params["put_delta"] * 0.5, put_long_target, target_expiry)

        if not short_put or not long_put:
            log.error(f"  Slot {slot_id}: could not find put contracts, skipping")
            slot_state["next_entry"] = _next_trading_day(today)
            return slot_state

        if long_put["strike"] >= short_put["strike"]:
            log.warning(f"  Slot {slot_id}: put spread inverted — "
                        f"long put ${long_put['strike']} ≥ short put "
                        f"${short_put['strike']}. Skipping entry.")
            slot_state["next_entry"] = _next_trading_day(today)
            return slot_state

        legs_info   += [{"symbol": short_put["symbol"], "side": "sell"},
                        {"symbol": long_put["symbol"],  "side": "buy"}]
        leg_symbols += [short_put["symbol"], long_put["symbol"]]
        leg_sides   += [True, False]   # True = short
        net_credit  += short_put["mid"] - long_put["mid"]

    # ── CALL SIDE (for iron_condor and call_credit_spread) ───────
    if params["call_delta"] > 0:
        call_width = max(5.0, min(round(iv_wing), round(S * 0.10)))
        call_short_target = find_strike_by_delta(
            S, rfr, iv, T, params["call_delta"], "call")
        call_long_target = call_short_target + call_width

        short_call = find_contract(trade_client, opt_data_client, ticker, "call",
                                   params["call_delta"], call_short_target, target_expiry)
        long_call  = find_contract(trade_client, opt_data_client, ticker, "call",
                                   params["call_delta"] * 0.5, call_long_target, target_expiry)

        # Sanity check: long call must be ABOVE short call
        if short_call and long_call:
            if long_call["strike"] <= short_call["strike"]:
                log.warning(f"  Slot {slot_id}: call spread inverted — "
                            f"long call ${long_call['strike']} ≤ short call "
                            f"${short_call['strike']}. Dropping call side.")
                short_call = long_call = None

        if not (short_call and long_call):
            # The call side could not be built. An iron condor is a
            # NEUTRAL, two-sided position; silently dropping the call
            # side turns it into a directional put credit spread that the
            # regime logic never selected and the risk model never sized.
            # With the strike-band query fix this should be rare (genuine
            # liquidity hole only) — when it does happen, SKIP rather than
            # place an unintended directional bet.
            if spread_name == "iron_condor":
                log.warning(f"  Slot {slot_id}: iron_condor call side could not "
                            f"be built — SKIPPING entry (will not silently "
                            f"degrade a neutral IC into a directional put "
                            f"spread). Retrying next trading day.")
            else:
                # CCS REQUIRES call legs — abort
                log.error(f"  Slot {slot_id}: CCS requires call legs but none found. Skipping.")
            slot_state["next_entry"] = _next_trading_day(today)
            return slot_state

        if short_call and long_call:
            legs_info   += [{"symbol": short_call["symbol"], "side": "sell"},
                            {"symbol": long_call["symbol"],  "side": "buy"}]
            leg_symbols += [short_call["symbol"], long_call["symbol"]]
            leg_sides   += [True, False]
            net_credit  += short_call["mid"] - long_call["mid"]

    if net_credit <= 0:
        log.warning(f"  Slot {slot_id}: zero/negative credit ${net_credit:.2f} — "
                    f"likely inverted strike snap on thin options chain. Skipping.")
        slot_state["next_entry"] = _next_trading_day(today)
        return slot_state

    # ── Wing width (needed for the ratio gate AND for sizing) ────
    # For single-side spreads, the binding wing is that side's width.
    # For an iron condor, use the larger wing as the binding constraint.
    if spread_name == "put_credit_spread":
        wing_for_max_loss = put_width
    elif spread_name == "call_credit_spread":
        wing_for_max_loss = call_width
    else:  # iron_condor — call wing may differ if put-side IV refinement ran
        wing_for_max_loss = max(put_width, call_width) if call_width > 0 else put_width

    # ── Per-spread RATIO credit gate ─────────────────────────────
    # Credit must be at least X% of the wing width. This is price-
    # invariant: IWM (~$273) and XLE (~$60) are held to the SAME
    # risk/reward standard. A flat dollar floor would hold XLE to a
    # ~3x stricter bar purely because of its lower price — blocking
    # sound XLE trades (see credit_gate_audit.py for the evidence).
    min_ratio_map = {
        "iron_condor":        cfg["min_credit_ratio_ic"],
        "put_credit_spread":  cfg["min_credit_ratio_pcs"],
        "call_credit_spread": cfg["min_credit_ratio_ccs"],
    }
    min_ratio = min_ratio_map.get(spread_name, 0.09)
    credit_ratio = (net_credit / wing_for_max_loss) if wing_for_max_loss > 0 else 0.0
    if credit_ratio < min_ratio:
        log.info(f"  Slot {slot_id}: credit ${net_credit:.3f} is only "
                 f"{credit_ratio*100:.1f}% of the ${wing_for_max_loss:.1f} wing "
                 f"— below the {min_ratio*100:.1f}% minimum for {spread_name}. "
                 f"Skipping — retry tomorrow.")
        slot_state["next_entry"] = _next_trading_day(today)
        return slot_state

    log.info(f"  Slot {slot_id}: credit quality OK — ${net_credit:.3f}/share "
             f"= {credit_ratio*100:.1f}% of the ${wing_for_max_loss:.1f} wing "
             f"(min {min_ratio*100:.1f}%)")

    # ── Size ──────────────────────────────────────────────────
    max_loss   = wing_for_max_loss - net_credit
    contracts  = compute_contracts(portfolio_value, max_loss,
                                   slot_risk=(cfg_override or CONFIG).get("risk_pct"),
                                   ticker=ticker)
    if contracts < 1:
        log.warning(f"  Slot {slot_id}: insufficient capital for 1 contract")
        slot_state["next_entry"] = _next_trading_day(today)
        return slot_state

    # ── Submit and read actual fill ──────────────────────────
    order_id, filled_credit = submit_spread(trade_client, spread_name,
                                            legs_info, contracts, net_credit)
    if not order_id:
        # Order failed — gate slot until tomorrow so we retry with fresh quotes
        log.warning(f"  Slot {slot_id}: order failed, retrying next market session")
        slot_state["next_entry"] = _next_trading_day(today)
        return slot_state

    # Use the actual fill price for credit_received and max_loss.
    # net_credit is the mid-price estimate used as the limit; filled_credit
    # is what was actually received.  Profit target and diagrams use
    # filled_credit so every calculation is anchored to real execution.
    actual_credit = filled_credit if filled_credit else net_credit
    actual_max_loss = put_width - actual_credit
    if actual_credit != net_credit:
        slip_pct = (net_credit - actual_credit) / net_credit * 100
        log.info(f"  Slot {slot_id}: fill slippage "
                 f"${net_credit:.4f} (mid) → ${actual_credit:.4f} (fill)  "
                 f"= {slip_pct:.1f}%")

    # ── Record state ──────────────────────────────────────────
    # Pick the canonical expiry source: short_put for IC/PCS, short_call for CCS
    expiry_contract = short_put if short_put else short_call
    slot_state["position"] = {
        "entry_date":       today.isoformat(),
        "expiry":           (expiry_contract["expiry"] if hasattr(expiry_contract["expiry"],'isoformat')
                             else str(expiry_contract["expiry"])),
        "spread":           spread_name,
        "trend_regime":     regime["trend"],
        "vol_regime":       regime["vol"],
        "atr_regime":       regime["atr"],
        "contracts":        contracts,
        "credit_received":  round(actual_credit, 4),    # actual fill, not mid
        "credit_mid":       round(net_credit, 4),        # mid at quote time (for reference)
        "max_loss":         round(actual_max_loss, 4),   # recalculated from actual fill
        "leg_symbols":      leg_symbols,
        "leg_sides":        leg_sides,
        "order_id":         order_id,
        "underlying_px":    S,
        # ── Indicator state at entry (for analysis / post-hoc review) ─
        "rsi_at_entry":     regime.get("rsi", None),
        "vix_fresh_high":   regime.get("vix_fresh_high", False),
        "call_delta":       params["call_delta"],   # may be panic delta
        "put_delta":        params["put_delta"],
    }
    slot_state["next_entry"] = None

    # Log trade — record both mid and actual fill for slippage tracking
    log_trade({
        "date":           today.isoformat(),
        "action":         "open",
        "slot":           slot_id,
        "spread":         spread_name,
        "trend":          regime["trend"],
        "vol":            regime["vol"],
        "atr":            regime["atr"],
        "contracts":      contracts,
        "credit":         round(actual_credit, 4),
        "credit_mid":     round(net_credit, 4),
        "slippage_pct":   round((net_credit-actual_credit)/net_credit*100, 1)
                          if net_credit > 0 else 0,
        "max_loss":       round(actual_max_loss, 4),
        "underlying_px":  round(S, 2),
        "order_id":       order_id,
    })

    # Save spread diagram + append to spread_log.csv
    # Use the same canonical-expiry contract picked above for the position state
    expiry_str = (expiry_contract["expiry"].isoformat()
                  if hasattr(expiry_contract["expiry"], "isoformat")
                  else str(expiry_contract["expiry"]))
    save_spread_diagram(
        slot_id=slot_id,
        spread_name=spread_name,
        legs_info=legs_info,
        short_put=short_put,           # None for CCS
        long_put=long_put,             # None for CCS
        short_call=short_call if params["call_delta"] > 0 else None,
        long_call=long_call   if params["call_delta"] > 0 else None,
        S=S,
        net_credit=actual_credit,   # diagram uses actual fill price
        max_loss=actual_max_loss,
        contracts=contracts,
        today=today,
        regime=regime,
        expiry_str=expiry_str,
    )

    log.info(f"  Slot {slot_id}: opened {spread_name}  "
             f"{contracts} contracts  "
             f"credit=${actual_credit:.4f} (mid ${net_credit:.4f})  "
             f"max_loss=${actual_max_loss:.4f}")
    send_alert(
        title=f"VRP: {slot_id} opened",
        message=(f"{slot_id} {ticker}: opened {spread_name} "
                 f"{contracts}c @ ${actual_credit:.2f} credit  "
                 f"regime={regime['trend']}+{regime['vol']}+{regime['atr']}  "
                 f"exp {target_expiry}"),
        tags="chart_with_upwards_trend",
    )
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
    # Use last_equity (prior-close settled value) for position sizing.
    # portfolio_value includes live option mark-to-market and drops
    # immediately when new spreads are opened (bid-ask slippage on valuation),
    # causing apparent -5-7% swings that incorrectly shrink next-entry sizing.
    # last_equity is the settled cash+equity value from prior close and is
    # the correct stable base for computing how many contracts to enter.
    portfolio_value = float(account.last_equity)
    mtm_value       = float(account.portfolio_value)
    log.info(f"Portfolio (last equity):   ${portfolio_value:,.2f}  "
             f"({'Paper' if PAPER else 'Live'})")
    log.info(f"Portfolio (live MTM):      ${mtm_value:,.2f}  "
             f"(not used for sizing — includes option bid-ask drag)")

    # ── Load state ────────────────────────────────────────────
    state = load_state()
    if state["initial_capital"] is None:
        state["initial_capital"] = portfolio_value
        log.info(f"First run — initial capital set to ${portfolio_value:,.2f}")

    # ── Slot configuration ────────────────────────────────────
    # 2× IWM + 2× XLE at 17.5% each = 70% deployed, 30% cash reserve.
    # Real-data backtest 2008-2016 ($9k start) showed this config:
    #   CAGR 24.87% vs 4× IWM 14.2% vs S&P 4.5%
    #   Sharpe 1.26 vs IWM-only 0.94 (higher despite worse max DD)
    #   The XLE slots profited from oil's 2009-2014 bull AND from CCS during
    #   the 2014-16 oil crash (100% WR on XLE_CCS).
    # Tradeoff: max DD -42% vs IWM-only -30% during XLE-specific shocks.
    # See backtest_9k_xle.py for full analysis. Slots staggered 7 days.
    TICKER_SLOTS = [
        ("IWM_A", "IWM", CONFIG["iwm_slot_risk"]),
        ("XLE_B", "XLE", CONFIG["iwm_slot_risk"]),
        ("IWM_C", "IWM", CONFIG["iwm_slot_risk"]),
        ("XLE_D", "XLE", CONFIG["iwm_slot_risk"]),
    ]
    # On first run, gate B/C/D by 7/14/21 days so they open in sequence
    FIRST_RUN_GATES = {"XLE_B": 7, "IWM_C": 14, "XLE_D": 21}
    for _sid, _tkr, _ in TICKER_SLOTS:
        slot_st = state["slots"].setdefault(_sid, {"position": None, "next_entry": None, "closed_trades": []})
        if _sid in FIRST_RUN_GATES and slot_st.get("next_entry") is None and slot_st.get("position") is None:
            gate = (today + timedelta(days=FIRST_RUN_GATES[_sid])).isoformat()
            slot_st["next_entry"] = gate
            log.info(f"Slot {_sid} ({_tkr}) first-run gate → {gate}")

    # ── Assignment check (before managing slots) ─────────────
    # Catches overnight assignments early so manage_slot doesn't operate
    # on broken state. Also runs once — the duplicate end-of-run call was
    # removed when the two assignment functions were merged (#19).
    log.info("Checking for assignments...")
    assignment_alerts = check_assignment(trade_client, state, today)

    # ── Regime detection (one per distinct ticker in TICKER_SLOTS) ─
    log.info("Computing regimes...")
    regimes = {}
    distinct_tickers = sorted({t for _, t, _ in TICKER_SLOTS})
    for ticker in distinct_tickers:
        r = get_regime(CONFIG, ticker)
        regimes[ticker] = r
        log.info(f"  {ticker}: Trend={r['trend']}  Vol={r['vol']}  ATR={r['atr']}  → {r['spread']}")
        log.info(f"    ${r['price']:.2f}  VIX={r['vix']:.1f}  HV={r['hv']*100:.1f}%  "
                 f"RSI={r.get('rsi', 50):.1f}  fresh_VIX_high={r.get('vix_fresh_high', False)}")

    # ── Manage open positions ─────────────────────────────────
    # First pass: slots in the current TICKER_SLOTS config
    log.info("Managing open positions...")
    managed = set()
    for slot_id, ticker, slot_risk in TICKER_SLOTS:
        slot_st = state["slots"].setdefault(slot_id, {"position": None, "next_entry": None, "closed_trades": []})
        managed.add(slot_id)
        if slot_st.get("position"):
            updated = manage_slot(trade_client, opt_data,
                                  slot_id, slot_st, portfolio_value, today,
                                  state=state)
            state["slots"][slot_id] = updated
        else:
            log.info(f"  Slot {slot_id}: no open position")

    # Second pass: legacy slots (still in state.json but removed from TICKER_SLOTS).
    # We continue to manage these to close — but no new entries open here.
    for slot_id in list(state["slots"].keys()):
        if slot_id in managed:
            continue
        slot_st = state["slots"][slot_id]
        if slot_st.get("position"):
            log.warning(f"  Slot {slot_id}: LEGACY slot (not in current TICKER_SLOTS) — "
                        f"managing existing position to close. No new entries will open here.")
            updated = manage_slot(trade_client, opt_data,
                                  slot_id, slot_st, portfolio_value, today,
                                  state=state)
            state["slots"][slot_id] = updated

    # ── Open new positions ────────────────────────────────────
    # Rolling stagger across all same-ticker slots, looking at entries in
    # both OPEN positions AND recently CLOSED trades (closed_trades history).
    #
    # Why include closed trades: if A/B/C/D all close on the same day (e.g.
    # a market shock takes them all to max loss), the old logic — which only
    # looked at currently-open positions — would let all 4 slots re-enter
    # the same day, undoing the diversification benefit. With closed-trade
    # history included, the next entry is gated by the most recent ENTRY
    # date across all slots, regardless of whether that position is still open.
    STAGGER = timedelta(days=CONFIG["stagger_days"])

    log.info("Checking entry opportunities...")
    for slot_id, ticker, slot_risk in TICKER_SLOTS:
        slot_st = state["slots"][slot_id]
        if slot_st.get("position") is None:

            # ── Rolling stagger check (all same-ticker slots) ─────────
            same_ticker_entries = []
            for other_id, other_tkr, _ in TICKER_SLOTS:
                if other_id == slot_id or other_tkr != ticker:
                    continue
                other_slot = state["slots"][other_id]

                # Open position's entry_date
                op = other_slot.get("position")
                if op and op.get("entry_date"):
                    try:
                        same_ticker_entries.append(date.fromisoformat(op["entry_date"]))
                    except Exception:
                        pass

                # Closed-trade history's entry_date — only consider entries
                # within the stagger window (older entries can't gate us anyway)
                cutoff = today - STAGGER
                for ct in other_slot.get("closed_trades", []):
                    ed = ct.get("entry_date")
                    if not ed:
                        continue
                    try:
                        d = date.fromisoformat(ed)
                        if d >= cutoff:
                            same_ticker_entries.append(d)
                    except Exception:
                        pass

            if same_ticker_entries:
                most_recent = max(same_ticker_entries)
                earliest    = most_recent + STAGGER
                if today < earliest:
                    gated_until = earliest.isoformat()
                    cur_ne = slot_st.get("next_entry")
                    if not cur_ne or gated_until > cur_ne:
                        slot_st["next_entry"] = gated_until
                        log.info(f"  Slot {slot_id}: most recent {ticker} entry "
                                 f"was {most_recent} (open or closed) — stagger "
                                 f"gate until {gated_until} "
                                 f"({CONFIG['stagger_days']}d)")
                    state["slots"][slot_id] = slot_st
                    continue

            regime = regimes[ticker]
            # Override risk_pct for this specific slot
            cfg_override = {**CONFIG, "risk_pct": slot_risk}
            updated = enter_slot(trade_client, opt_data,
                                 slot_id, slot_st, regime,
                                 portfolio_value, today,
                                 ticker=ticker,
                                 cfg_override=cfg_override)
            state["slots"][slot_id] = updated

    # ── Send per-assignment alerts (assignment_alerts gathered at start) ─
    for alert in assignment_alerts:
        send_alert(title="⚠ Assignment detected", message=alert,
                   priority="urgent", tags="rotating_light,warning")

    # ── Append today's portfolio value to equity history ──────
    # This gives the equity chart daily granularity that doesn't depend on
    # the trade_log.csv being clean. One point per trading day the bot runs.
    eh = state.setdefault("equity_history", [])
    today_iso = today.isoformat()
    # Replace existing entry for today (if the bot ran twice) or append
    if eh and eh[-1].get("date") == today_iso:
        eh[-1] = {"date": today_iso, "value": float(portfolio_value)}
    else:
        eh.append({"date": today_iso, "value": float(portfolio_value)})
    # Cap retention at 5 years of daily points to keep state.json reasonable
    if len(eh) > 1260:
        state["equity_history"] = eh[-1260:]

    # ── Save state ────────────────────────────────────────────
    save_state(state)
    log.info("State saved.")

    # ── Summary ───────────────────────────────────────────────
    log.info("-" * 60)
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

    # ── Daily summary push notification ──────────────────────
    open_slots = sum(1 for s in state["slots"].values() if s.get("position"))
    closed_today = []
    for sl_id, sl in state["slots"].items():
        for ct in sl.get("closed_trades", []):
            if ct.get("close_date") == today.isoformat():
                closed_today.append(f"{sl_id} ${ct['realized_pnl']:+,.0f}")
    cum_pnl = state.get("cumulative_pnl", 0)

    summary_lines = [
        f"Open positions: {open_slots}/4",
        f"Portfolio: ${portfolio_value:,.0f}  cumulative P&L: ${cum_pnl:+,.0f}",
    ]
    if closed_today:
        summary_lines.insert(0, "Closed today: " + "  |  ".join(closed_today))
    if assignment_alerts:
        summary_lines.insert(0, "ASSIGNMENTS NEED ATTENTION")

    priority = "urgent" if assignment_alerts else ("high" if closed_today else "default")
    tags     = "rotating_light" if assignment_alerts else (
               "chart_with_upwards_trend" if closed_today else "robot")
    send_alert(
        title=f"VRP {'ALERT' if assignment_alerts else 'daily'} — {today}",
        message="\n".join(summary_lines),
        priority=priority,
        tags=tags,
    )




# ══════════════════════════════════════════════════════════════
#  PUSH NOTIFICATIONS (ntfy.sh)
# ══════════════════════════════════════════════════════════════

def send_alert(title: str, message: str, priority: str = "default",
               tags: str = "robot"):
    """
    Send a push notification via ntfy.sh.

    Free, no signup, no API key — just pick a unique topic name.
    Install the ntfy app (iOS/Android) and subscribe to your topic.

    Set NTFY_TOPIC environment variable to your chosen topic name.
    Example:  NTFY_TOPIC=vrp-trader-benny-alerts

    Priority levels: min, low, default, high, urgent
    Tags map to emoji in the app: robot, warning, white_check_mark,
                                   rotating_light, chart_with_upwards_trend
    """
    topic = os.environ.get("NTFY_TOPIC", "")
    if not topic:
        log.debug("  NTFY_TOPIC not set — skipping push notification")
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={
                "Title":    title,
                "Priority": priority,
                "Tags":     tags,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log.info(f"  Alert sent: [{title}]")
            else:
                log.warning(f"  Alert send failed: HTTP {resp.status}")
    except Exception as e:
        log.warning(f"  Alert send failed: {e}")

# ══════════════════════════════════════════════════════════════
#  CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════

def dump_state_cli() -> int:
    """
    Pretty-print state.json for debugging. Returns process exit code.
    Useful when CI fails and you want to see what slots are open without
    SSH-ing or parsing JSON by eye.
    """
    state = load_state()
    today = date.today()
    print(f"State file: {CONFIG['state_file']}")
    print(f"Today:      {today}")
    print(f"Initial:    ${state.get('initial_capital') or 0:,.2f}")
    print(f"Cumulative: ${state.get('cumulative_pnl', 0):+,.2f}")
    print(f"Trades:     {state.get('trade_count', 0)}")
    print()
    for sid, slot in state.get("slots", {}).items():
        pos = slot.get("position")
        if pos:
            try:
                dte = (date.fromisoformat(str(pos["expiry"])) - today).days
            except Exception:
                dte = "?"
            print(f"  {sid}: OPEN  {pos.get('spread', '?')}  "
                  f"{pos.get('contracts', 0)}c  "
                  f"DTE={dte}  credit=${pos.get('credit_received', 0):.4f}  "
                  f"max_loss=${pos.get('max_loss', 0):.4f}  "
                  f"entry={pos.get('entry_date', '?')}")
            if pos.get("assignment_alert"):
                print(f"      ⚠ ASSIGNMENT ALERT raised on {pos['assignment_alert']}")
        else:
            print(f"  {sid}: empty  next_entry={slot.get('next_entry', 'now')}  "
                  f"closed={len(slot.get('closed_trades', []))}")
    if state.get("seen_assignments"):
        print()
        print("Seen assignments:")
        for k in state["seen_assignments"]:
            print(f"  {k}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRP Live Trader — Alpaca")
    parser.add_argument("--dump-state", action="store_true",
                        help="Pretty-print state.json and exit without trading")
    args = parser.parse_args()
    if args.dump_state:
        sys.exit(dump_state_cli())
    run()
