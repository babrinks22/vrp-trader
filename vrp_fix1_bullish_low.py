# ============================================================
#  Fix 1 of 3 — Wrong spread assignment in bullish + low vol
#
#  Problem:
#    bullish + low vol was routed to long_dte_condor.
#    That slot accounted for 50% of all trades (18/36) and
#    broke even: 66.7% WR, total P&L = -$38, all 3 stop losses.
#
#    Root cause: when price is above both SMAs in a quiet market,
#    the call side of a symmetric condor is perpetually at risk
#    as the market grinds higher. A bull_put has no call leg —
#    it only needs price to stay above the short put strike,
#    which is exactly what a trending, low-vol market does.
#
#  Change (1 line in REGIME_MAP):
#    ("bullish", "low"):  "long_dte_condor"   ← before
#    ("bullish", "low"):  "bull_put"           ← after
#
#  Everything else is identical to vrp_regime_selector.py.
#  Run: python vrp_fix1_bullish_low.py
# ============================================================

import sys
sys.path.insert(0, '.')

from vrp_regime_selector import (
    RegimeSelectorBacktester, FixedStopBacktester,
    plot_selector_tearsheet, plot_comparison,
    FIXED_STOP_SPREADS, FIXED_STOP_PCT,
)
from vrp_rotation import (
    CONFIG, SPREAD_LIBRARY, RotationBacktester,
    load_data, build_signals, compute_metrics, run_single,
)
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ── The only change: one line in the regime map ───────────────

REGIME_MAP_FIX1 = {
    ("bullish",  "low"):  "bull_put",          # WAS: long_dte_condor
    ("bullish",  "mid"):  "long_dte_condor",
    ("bullish",  "high"): "iron_condor",
    ("bearish",  "high"): "high_credit_condor",
    ("bearish",  "mid"):  "iron_condor",
    ("bearish",  "low"):  "bull_put",
    ("neutral",  "low"):  "long_dte_condor",
    ("neutral",  "mid"):  "iron_condor",
    ("neutral",  "high"): "high_credit_condor",
    ("unknown",  "low"):  "iron_condor",
    ("unknown",  "mid"):  "iron_condor",
    ("unknown",  "high"): "iron_condor",
}


# ── Subclass that uses the patched map ────────────────────────

class Fix1Backtester(RegimeSelectorBacktester):
    def _enter(self, today, S, vix_today, sig, closes):
        trend  = self._regime_trend(sig)
        vol    = self._regime_vol(vix_today, sig)
        chosen = REGIME_MAP_FIX1.get((trend, vol), "iron_condor")

        self.spread      = SPREAD_LIBRARY[chosen]
        self.spread_name = chosen

        # Call grandparent _enter (RotationBacktester) directly
        RotationBacktester._enter(self, today, S, vix_today, sig, closes)

        if self.position is not None:
            self.position.spread_name = chosen
            self.position.key         = f"fix1_{chosen}_{today}"


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Fix 1: bullish+low → bull_put")
    print("  All other regime assignments unchanged.")
    print("=" * 65)

    closes  = load_data(CONFIG)
    signals = build_signals(closes, CONFIG)

    print("\nRunning strategies...")

    # Fix 1 selector
    print("  [1] Fix1 selector (bullish+low → bull_put)...")
    bt1 = Fix1Backtester(CONFIG)
    r1  = bt1.run(closes, signals)
    r1["spread_name"] = "fix1_selector"
    m1, dd1 = compute_metrics(r1["equity"], r1["trades"], CONFIG["initial_capital"])

    # Original selector (baseline)
    print("  [2] Original selector (baseline)...")
    from vrp_regime_selector import RegimeSelectorBacktester as OrigSel
    bt_orig = OrigSel(CONFIG)
    r_orig  = bt_orig.run(closes, signals)
    r_orig["spread_name"] = "original_selector"
    m_orig, dd_orig = compute_metrics(
        r_orig["equity"], r_orig["trades"], CONFIG["initial_capital"])

    # long_dte standalone (best single)
    print("  [3] long_dte standalone...")
    bt_ld = RotationBacktester(CONFIG, "long_dte_condor")
    r_ld  = bt_ld.run(closes, signals)
    m_ld, dd_ld = compute_metrics(
        r_ld["equity"], r_ld["trades"], CONFIG["initial_capital"])

    # ── Results table ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'Strategy':<28} {'Trades':>7} {'Win%':>7} "
          f"{'Sharpe':>8} {'Total P&L':>12}")
    print("  " + "-" * 60)
    for name, m in [
        ("Fix1 (bullish+low→bull_put)", m1),
        ("Original selector",            m_orig),
        ("long_dte standalone",          m_ld),
    ]:
        print(f"  {name:<28} "
              f"{m.get('Trades','0'):>7} "
              f"{m.get('Win Rate','?'):>7} "
              f"{m.get('Sharpe','?'):>8} "
              f"{m.get('Total P&L','?'):>12}")
    print("=" * 65)

    # Spread selection frequency
    if not r1["trades"].empty:
        t = r1["trades"]
        print("\nFix1 — spread selection frequency:")
        freq = t["spread_name"].value_counts()
        for name, count in freq.items():
            pct = count / len(t) * 100
            pnl = t[t["spread_name"]==name]["pnl"].sum()
            wr  = (t[t["spread_name"]==name]["pnl"] > 0).mean() * 100
            print(f"  {name:<22}  {count:>3} trades ({pct:.0f}%)  "
                  f"WR={wr:.0f}%  P&L=${pnl:,.0f}")

        t.to_csv("fix1_trades.csv", index=False)
        print("\nTrade log saved to fix1_trades.csv")

    # Tearsheet
    plot_selector_tearsheet(r1, m1, dd1)

    # Comparison
    all_r = {
        "fix1_selector":    {**r1,    "sharpe": float(m1.get("Sharpe", 0))},
        "original_selector":{**r_orig,"sharpe": float(m_orig.get("Sharpe", 0))},
        "long_dte_condor":  {**r_ld,  "sharpe": float(m_ld.get("Sharpe", 0))},
    }
    plot_comparison(all_r, {})
    print("\nDone.")


if __name__ == "__main__":
    main()
