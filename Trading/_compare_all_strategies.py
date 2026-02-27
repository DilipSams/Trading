"""
All-Strategy Performance Comparison Dashboard.

Runs every available strategy and displays a unified comparison table
with CAGR, Sharpe, Sortino, MaxDD, and multi-period returns.

Usage:
    python _compare_all_strategies.py
"""
import sys
import time
import numpy as np
import pandas as pd
from collections import OrderedDict

# ── Strategy imports ─────────────────────────────────────────────────
sys.path.insert(0, ".")
from strategy_research import (
    StrategyEngine, ETFStrategyEngine, compute_metrics,
)
from waverider import (
    WaveRiderStrategy, WaveRiderConfig, load_universe, load_spy,
    compute_nav_metrics,
)


def fmt_ret(val, width=9):
    """Format return % compactly."""
    if not np.isfinite(val):
        return f"{'n/a':>{width}s}"
    if abs(val) >= 1_000_000:
        return f"{val/1e6:>+{width-2}.1f}M%"
    if abs(val) >= 10_000:
        return f"{val/1e3:>+{width-2}.0f}k%"
    if abs(val) >= 1_000:
        return f"{val:>+{width-1},.0f}%"
    return f"{val:>+{width-1}.1f}%"


def nav_metrics(nav, label=""):
    """Compute standardized metrics from a NAV series."""
    nav = nav.dropna()
    if len(nav) < 20:
        return None
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    if n_years <= 0 or nav.iloc[0] <= 0:
        return None
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    if total_return <= -1:
        # NAV went to zero or negative
        return None
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    daily_r = nav.pct_change().dropna()
    sharpe = float(daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0
    down = daily_r[daily_r < 0]
    sortino = float(daily_r.mean() * 252 / (down.std() * np.sqrt(252))) if len(down) > 0 and down.std() > 0 else 0
    max_dd = float((nav / nav.cummax() - 1).min())

    return {
        "label": label,
        "nav": nav,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "total_return": total_return,
        "n_years": n_years,
        "start": nav.index[0],
        "end": nav.index[-1],
    }


def period_return(nav, end_date, years):
    """Compute return over last N years ending at end_date."""
    cutoff = end_date - pd.DateOffset(years=years)
    sub = nav[nav.index >= cutoff]
    if len(sub) < 20:
        return np.nan
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


# =====================================================================
print("=" * 130)
print("  ALL-STRATEGY PERFORMANCE COMPARISON DASHBOARD")
print("=" * 130)

# ── 1. Load engines ──────────────────────────────────────────────────
print("\n  Loading strategy engines...")
t0 = time.time()

# Stock + ETF strategy engine (strategy_research.py)
engine = StrategyEngine(dynamic_universe=True)
etf_engine = ETFStrategyEngine(strategy_engine=engine)

# WaveRider engine (waverider.py)
wr_prices, wr_rankings = load_universe()
wr_spy = load_spy()

print(f"  All engines loaded in {time.time()-t0:.1f}s\n")

# ── 2. Run all strategies ────────────────────────────────────────────
print("  Running strategies...", flush=True)
results = OrderedDict()  # name -> metrics dict


def add_result(name, nav, label=None):
    """Add a strategy result to the comparison."""
    if nav is None or (hasattr(nav, '__len__') and len(nav) == 0):
        print(f"    {name:<40s} SKIPPED (no NAV data)", flush=True)
        return
    m = nav_metrics(nav, label or name)
    if m is not None:
        results[name] = m
        print(f"    {name:<40s} CAGR: {m['cagr']*100:>+7.1f}%  Sharpe: {m['sharpe']:.2f}", flush=True)


# --- Benchmarks ---
add_result("SPY Buy & Hold", engine.buy_and_hold("SPY"))
add_result("QQQ Buy & Hold", engine.buy_and_hold("QQQ"))

# --- Stock strategies (strategy_research.py) ---
nav, _ = engine.sma_crossover_spy(50, 200)
add_result("SMA 50/200 SPY", nav)

nav, _ = engine.sma_crossover_basket(50, 200, top_n=20)
add_result("SMA 50/200 Basket (T20)", nav)

nav, _ = engine.sector_rotation()
add_result("Sector Rotation (T3)", nav)

nav, _ = engine.dual_momentum(top_n=20)
add_result("Dual Momentum (T20)", nav)

nav, _ = engine.improved_momentum(top_n=10)
add_result("Improved Momentum T10", nav)

nav, _ = engine.improved_momentum(top_n=5)
add_result("Improved Momentum T5", nav)

nav, _ = engine.improved_momentum(top_n=5, use_meme_score=True,
                                   meme_exclude=70, meme_max1=50, meme_max2=30)
add_result("Improved Mom T5 + MemeFilter", nav)

nav, _ = engine.low_volatility(top_n=20)
add_result("Low Volatility (T20)", nav)

nav, _ = engine.mean_reversion_dip()
add_result("Mean Reversion Dip Buy", nav)

nav, _ = engine.multibagger_screen()
add_result("Multibagger Screen", nav)

nav, _ = engine.combined_multibagger_rotation()
add_result("Combined Multibagger+Rotation", nav)

add_result("Equal-Weight Sectors", engine.equal_weight_sectors())

# --- ETF strategies ---
nav, _ = etf_engine.dual_momentum_etf()
add_result("Dual Momentum ETF (Antonacci)", nav)

nav, _ = etf_engine.sector_momentum_etf(top_n=3)
add_result("Sector Momentum ETF (T3)", nav)

# Hybrid core+satellite variants
for core, pct, sat_n in [("QQQ", 0.60, 5), ("QQQ", 0.50, 5), ("XLK", 0.60, 5), ("QQQ", 0.60, 10)]:
    nav, _ = etf_engine.qqq_core_momentum_satellite(core_etf=core, core_pct=pct, sat_top_n=sat_n)
    add_result(f"Core {core} {int(pct*100)}% + Sat T{sat_n}", nav)

# --- WaveRider variants ---
for lbl, cfg in [
    ("WaveRider T5 BearVol2x (21d)", WaveRiderConfig()),
    ("WaveRider T5 Unlevered (21d)", WaveRiderConfig()),
    ("WaveRider T5 BearVol2x (EOM)", WaveRiderConfig(rebalance_eom=True)),
]:
    strat = WaveRiderStrategy(cfg)
    res = strat.backtest(wr_prices, wr_spy, wr_rankings)
    if "Unlevered" in lbl:
        add_result(lbl, res.nav_unlevered)
    else:
        add_result(lbl, res.nav_leveraged)

print(f"\n  {len(results)} strategies computed.\n")

# ── 3. Summary table ─────────────────────────────────────────────────
# Find common end date for period returns
if not results:
    print("\n  No strategies produced valid results. Exiting.")
    sys.exit(0)

all_ends = [m["end"] for m in results.values()]
end_date = min(all_ends)  # use earliest end so periods are comparable

print("=" * 130)
print("  STRATEGY COMPARISON — LIFETIME METRICS")
print("=" * 130)
print(f"\n  {'#':<3s} {'Strategy':<40s} {'CAGR':>8s} {'Sharpe':>7s} {'Sortino':>8s} "
      f"{'MaxDD':>7s} {'TotRet':>10s} {'Years':>6s}")
print(f"  {'-'*3} {'-'*40} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*10} {'-'*6}")

# Sort by CAGR descending
sorted_results = sorted(results.items(), key=lambda x: x[1]["cagr"], reverse=True)

for rank, (name, m) in enumerate(sorted_results, 1):
    tr = m["total_return"] * 100
    tr_s = fmt_ret(tr, 10)
    print(f"  {rank:<3d} {name:<40s} {m['cagr']*100:>+7.1f}% {m['sharpe']:>7.2f} "
          f"{m['sortino']:>8.2f} {m['max_dd']*100:>6.1f}% {tr_s} {m['n_years']:>5.0f}")

# ── 4. Multi-period return table ─────────────────────────────────────
print(f"\n\n{'=' * 130}")
print("  MULTI-PERIOD RETURNS (total return over each lookback window)")
print(f"{'=' * 130}")
print(f"\n  {'#':<3s} {'Strategy':<40s} {'YTD':>8s} {'1Y':>8s} {'3Y':>9s} "
      f"{'5Y':>9s} {'10Y':>10s} {'Full':>10s}")
print(f"  {'-'*3} {'-'*40} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")

for rank, (name, m) in enumerate(sorted_results, 1):
    nav = m["nav"]
    end = nav.index[-1]

    # YTD
    ytd_dates = nav[nav.index.year == end.year]
    ytd = (ytd_dates.iloc[-1] / ytd_dates.iloc[0] - 1) * 100 if len(ytd_dates) > 5 else np.nan

    r1 = period_return(nav, end, 1)
    r3 = period_return(nav, end, 3)
    r5 = period_return(nav, end, 5)
    r10 = period_return(nav, end, 10)
    full = m["total_return"] * 100

    def _f(v, w=8):
        return fmt_ret(v, w) if not np.isnan(v) else f"{'n/a':>{w}s}"

    print(f"  {rank:<3d} {name:<40s} {_f(ytd)} {_f(r1)} {_f(r3, 9)} "
          f"{_f(r5, 9)} {_f(r10, 10)} {fmt_ret(full, 10)}")

# ── 5. Risk-adjusted ranking ─────────────────────────────────────────
print(f"\n\n{'=' * 130}")
print("  RISK-ADJUSTED RANKING (by Sharpe ratio)")
print(f"{'=' * 130}")
sorted_sharpe = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
print(f"\n  {'#':<3s} {'Strategy':<40s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>7s} {'Sortino':>8s}")
print(f"  {'-'*3} {'-'*40} {'-'*7} {'-'*8} {'-'*7} {'-'*8}")
for rank, (name, m) in enumerate(sorted_sharpe, 1):
    print(f"  {rank:<3d} {name:<40s} {m['sharpe']:>7.2f} {m['cagr']*100:>+7.1f}% "
          f"{m['max_dd']*100:>6.1f}% {m['sortino']:>8.2f}")

print(f"\n  NOTE: Strategies have different date ranges. Lifetime metrics reflect each strategy's full history.")
print(f"  Multi-period returns use each strategy's own end date as the anchor.")
print()
