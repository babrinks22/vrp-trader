# ============================================================
#  Fix 2 of 3 — Over-concentration (one spread dominates)
#
#  Problem:
#    Even after Fix 1 (bullish+low → bull_put), the map only uses
#    2 dimensions: trend × vol. That creates large buckets where
#    a single spread covers very different market conditions.
#    bull_put would now be 22/36 trades (61%) — the same
#    concentration problem, just shifted to a different spread.
#
#    The ATR regime signal (expanding vs contracting) is already
#    computed but unused. It's the right tiebreaker:
#      - Contracting ATR = momentum is slowing, range is narrowing
#        → symmetric condor can widen its tent safely
#      - Expanding ATR = intraday ranges growing, trending harder
#        → directional spread aligned with trend is safer
#
#  Change: 3-key lookup (trend, vol, atr) instead of 2-key
#
#    bullish + low + contracting  → long_dte_condor
#      (slow uptrend, quiet, narrowing range → theta play)
#    bullish + low + expanding    → bull_put
#      (faster uptrend, call side too hot → drop the call leg)
#    bearish + mid + contracting  → skewed_put
#      (bearish but settling down → capture put skew, wider wing)
#    bearish + mid + expanding    → iron_condor
#      (bearish + vol expanding → symmetric hedge both directions)
#    + all other prior assignments preserved
#
#  Run: python vrp_fix2_differentiation.py
#  Requires: vrp_fix1_bullish_low.py and vrp_regime_selector.py
# ============================================================

import sys
sys.path.insert(0, '.')

from vrp_fix1_bullish_low import Fix1Backtester, REGIME_MAP_FIX1
from vrp_regime_selector  import plot_selector_tearsheet, plot_comparison
from vrp_rotation import (
    CONFIG, SPREAD_LIBRARY, RotationBacktester,
    load_data, build_signals, compute_metrics,
)
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ── 3-key map: (trend, vol, atr) ─────────────────────────────
#
#  Keys present here override the 2-key fallback.
#  Any (trend, vol, atr) combo NOT listed falls back to
#  REGIME_MAP_FIX1 using just (trend, vol).

REGIME_MAP_FIX2 = {
    # bullish + low — split by momentum
    ("bullish", "low",  "contracting"): "long_dte_condor",  # slow grind → theta
    ("bullish", "low",  "expanding"):   "bull_put",          # faster trend → no call leg
    ("bullish", "low",  "unknown"):     "bull_put",          # default to directional

    # bullish + mid — split by momentum
    ("bullish", "mid",  "contracting"): "long_dte_condor",
    ("bullish", "mid",  "expanding"):   "bull_put",
    ("bullish", "mid",  "unknown"):     "long_dte_condor",

    # bearish + mid — split by momentum
    ("bearish", "mid",  "contracting"): "skewed_put",   # settling → harvest skew
    ("bearish", "mid",  "expanding"):   "iron_condor",  # vol growing → hedge both
    ("bearish", "mid",  "unknown"):     "iron_condor",

    # bearish + low — split by momentum
    ("bearish", "low",  "contracting"): "bull_put",     # bearish but quiet — mean-rev
    ("bearish", "low",  "expanding"):   "skewed_put",   # picking up steam → skew play
    ("bearish", "low",  "unknown"):     "bull_put",

    # neutral + low — split by momentum
    ("neutral", "low",  "contracting"): "long_dte_condor",
    ("neutral", "low",  "expanding"):   "iron_condor",  # neutral but moving → tighter
    ("neutral", "low",  "unknown"):     "long_dte_condor",

    # high vol — both bearish+high and neutral+high
    ("bearish", "high", "contracting"): "skewed_put",
    ("bearish", "high", "expanding"):   "skewed_put",
    ("bearish", "high", "unknown"):     "skewed_put",
    ("neutral", "high", "contracting"): "skewed_put",
    ("neutral", "high", "expanding"):   "skewed_put",
    ("neutral", "high", "unknown"):     "skewed_put",

    # bullish + high
    ("bullish", "high", "contracting"): "iron_condor",
    ("bullish", "high", "expanding"):   "iron_condor",
    ("bullish", "high", "unknown"):     "iron_condor",

    # neutral + mid
    ("neutral", "mid",  "contracting"): "long_dte_condor",
    ("neutral", "mid",  "expanding"):   "iron_condor",
    ("neutral", "mid",  "unknown"):     "iron_condor",
}


# ── Subclass using 3-key map with 2-key fallback ──────────────

class Fix2Backtester(Fix1Backtester):
    def _enter(self, today, S, vix_today, sig, closes):
        trend = self._regime_trend(sig)
        vol   = self._regime_vol(vix_today, sig)
        atr   = self._regime_atr(sig)

        # Try 3-key first, fall back to 2-key (Fix1 map)
        chosen = REGIME_MAP_FIX2.get(
            (trend, vol, atr),
            REGIME_MAP_FIX1.get((trend, vol), "iron_condor")
        )

        self.spread      = SPREAD_LIBRARY[chosen]
        self.spread_name = chosen

        RotationBacktester._enter(self, today, S, vix_today, sig, closes)

        if self.position is not None:
            self.position.spread_name = chosen
            self.position.key         = f"fix2_{chosen}_{today}"

            # Also store ATR regime for analysis
            # (reuse atr_regime field already on Trade)
            self.position.atr_regime = atr


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Fix 2: ATR tiebreaker for better differentiation")
    print("  3-key lookup (trend × vol × atr)")
    print("=" * 65)
    print("\nKey new assignments:")
    print("  bullish+low+contracting → long_dte_condor (slow grind)")
    print("  bullish+low+expanding   → bull_put         (fast trend)")
    print("  bearish+mid+contracting → skewed_put       (put skew)")
    print("  bearish+mid+expanding   → iron_condor      (hedge both)")
    print("  all high-vol regimes    → skewed_put        (replaces high_credit)")

    closes  = load_data(CONFIG)
    signals = build_signals(closes, CONFIG)

    print("\nRunning strategies...")

    print("  [1] Fix2 selector (3-key map)...")
    bt2 = Fix2Backtester(CONFIG)
    r2  = bt2.run(closes, signals)
    r2["spread_name"] = "fix2_selector"
    m2, dd2 = compute_metrics(r2["equity"], r2["trades"], CONFIG["initial_capital"])

    print("  [2] Fix1 selector (2-key, bullish+low fixed)...")
    from vrp_fix1_bullish_low import Fix1Backtester
    bt1 = Fix1Backtester(CONFIG)
    r1  = bt1.run(closes, signals)
    r1["spread_name"] = "fix1_selector"
    m1, dd1 = compute_metrics(r1["equity"], r1["trades"], CONFIG["initial_capital"])

    print("  [3] Original selector (baseline)...")
    from vrp_regime_selector import RegimeSelectorBacktester
    bt_orig = RegimeSelectorBacktester(CONFIG)
    r_orig  = bt_orig.run(closes, signals)
    r_orig["spread_name"] = "original_selector"
    m_orig, _ = compute_metrics(
        r_orig["equity"], r_orig["trades"], CONFIG["initial_capital"])

    print("\n" + "=" * 65)
    print(f"  {'Strategy':<30} {'Trades':>7} {'Win%':>7} "
          f"{'Sharpe':>8} {'Total P&L':>12}")
    print("  " + "-" * 62)
    for name, m in [
        ("Fix2 (3-key: trend×vol×atr)", m2),
        ("Fix1 (bullish+low→bull_put)", m1),
        ("Original selector",            m_orig),
    ]:
        print(f"  {name:<30} "
              f"{m.get('Trades','0'):>7} "
              f"{m.get('Win Rate','?'):>7} "
              f"{m.get('Sharpe','?'):>8} "
              f"{m.get('Total P&L','?'):>12}")
    print("=" * 65)

    # Spread selection breakdown
    if not r2["trades"].empty:
        t = r2["trades"]
        print("\nFix2 — spread selection frequency:")
        freq = t["spread_name"].value_counts()
        for name, count in freq.items():
            pct  = count / len(t) * 100
            pnl  = t[t["spread_name"]==name]["pnl"].sum()
            wr   = (t[t["spread_name"]==name]["pnl"] > 0).mean() * 100
            print(f"  {name:<22}  {count:>3} trades ({pct:.0f}%)  "
                  f"WR={wr:.0f}%  P&L=${pnl:,.0f}")

        # Year-by-year
        print("\nFix2 — year-by-year P&L:")
        t["year"] = pd.to_datetime(t["entry_date"]).dt.year
        print(t.groupby("year")["pnl"].sum().apply(lambda x: f"${x:,.0f}").to_string())

        t.to_csv("fix2_trades.csv", index=False)
        print("\nTrade log saved to fix2_trades.csv")

    plot_selector_tearsheet(r2, m2, dd2)

    all_r = {
        "fix2_selector":    {**r2,    "sharpe": float(m2.get("Sharpe", 0))},
        "fix1_selector":    {**r1,    "sharpe": float(m1.get("Sharpe", 0))},
        "original_selector":{**r_orig,"sharpe": float(m_orig.get("Sharpe", 0))},
    }
    plot_comparison(all_r, {})
    print("\nDone.")


if __name__ == "__main__":
    main()
