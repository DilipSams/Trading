"""
Wave Rider T5 MS BearVol2x — Year-by-year performance, holdings, trades, SPY comparison.

Uses the canonical WaveRiderStrategy from waverider.py (single source of truth).
"""
import numpy as np
import pandas as pd
import re
from collections import Counter

from waverider import (
    WaveRiderStrategy, WaveRiderConfig, BacktestResult,
    load_universe, load_spy, load_risk_free, clean_uid,
    compute_nav_metrics,
)

# --- Load data ---
prices, rankings = load_universe()
spy_price = load_spy()

# --- Run backtest ---
config = WaveRiderConfig()
strategy = WaveRiderStrategy(config)
result = strategy.backtest(prices, spy_price, rankings)

dates = result.dates
n_years = (dates[-1] - dates[0]).days / 365.25
rf_daily = load_risk_free(dates)

# --- Metrics ---
m_lev = compute_nav_metrics(result.nav_leveraged, "BearVol2x")
m_unlev = compute_nav_metrics(result.nav_unlevered, "Unlevered")

spy_start = spy_price.asof(dates[0])
spy_end = spy_price.asof(dates[-1])
spy_cagr = ((spy_end / spy_start) ** (1 / n_years) - 1) if (pd.notna(spy_start) and spy_start > 0) else 0.106

# --- Year-by-year output ---
print("=" * 160)
print("  WAVE RIDER T5 MS BearVol2x -- YEAR-BY-YEAR PERFORMANCE (Recommended Strategy)")
print("  Ranking: multi-timeframe Carhart momentum / vol, SMA200 filter, hysteresis band")
print("  MemeScore: 6-factor (vol + parabolic + stretch + mom-conc + vol-accel + tenure)")
print(f"  Thresholds: Exclude>{config.meme_exclude} | Max1>{config.meme_max1} | Max2>{config.meme_max2}")
print("  Leverage: vol-target 20%, max 2x, bear gate 0.5x when SPY < SMA200")
print("=" * 160)

print(f"\n  {'Year':<6s} {'WR T5 2x':>9s} {'WR T5 1x':>9s} {'SPY':>9s} {'vs SPY':>8s} "
      f"{'AvgLev':>7s} {'Trades':>7s} {'#Stks':>6s}  Monthly Holdings")
print(f"  {'-' * 150}")

years = sorted(set(d.year for d in dates))
cumulative_trades = 0
yearly_data = []

for year in years:
    yr_dates = dates[dates.year == year]
    if len(yr_dates) < 20:
        continue

    yr_ret_l = (result.nav_leveraged.loc[yr_dates[-1]] / result.nav_leveraged.loc[yr_dates[0]] - 1) * 100
    yr_ret_u = (result.nav_unlevered.loc[yr_dates[-1]] / result.nav_unlevered.loc[yr_dates[0]] - 1) * 100

    spy_yr = spy_price[spy_price.index.year == year]
    spy_ret = (spy_yr.iloc[-1] / spy_yr.iloc[0] - 1) * 100 if len(spy_yr) > 20 else np.nan
    vs_spy = yr_ret_l - spy_ret if not pd.isna(spy_ret) else np.nan

    avg_lev = result.leverage_series.loc[yr_dates].mean()

    yr_rebal = [rd for rd in result.rebalance_dates if rd.year == year]
    yr_trades = sum(result.trades_log.get(rd, 0) for rd in yr_rebal)
    cumulative_trades += yr_trades

    all_held = set()
    monthly_parts = []
    for rd in yr_rebal:
        syms = result.holdings_log[rd]
        all_held.update(syms)
        clean = [clean_uid(s) for s in syms]
        monthly_parts.append(f"{rd.month:02d}:{'+'.join(clean)}")

    n_unique = len(all_held)
    holdings_str = ", ".join(monthly_parts)
    if len(holdings_str) > 80:
        holdings_str = holdings_str[:77] + "..."

    spy_str = f"{spy_ret:>+8.1f}%" if not pd.isna(spy_ret) else "     n/a"
    vs_str = f"{vs_spy:>+7.1f}%" if not pd.isna(vs_spy) else "     n/a"
    print(f"  {year:<6d} {yr_ret_l:>+8.1f}% {yr_ret_u:>+8.1f}% {spy_str} {vs_str} "
          f"{avg_lev:>6.2f}x {yr_trades:>5d}   {n_unique:>4d}   {holdings_str}")

    yearly_data.append({
        "year": year, "ret_2x": yr_ret_l, "ret_1x": yr_ret_u,
        "spy_ret": spy_ret, "vs_spy": vs_spy, "trades": yr_trades,
        "n_stocks": n_unique, "avg_lev": avg_lev,
    })

# --- Summary ---
total_years = len(yearly_data)
beat_count = sum(1 for d in yearly_data if d["vs_spy"] is not None and not pd.isna(d["vs_spy"]) and d["vs_spy"] > 0)
total_with_spy = sum(1 for d in yearly_data if d["spy_ret"] is not None and not pd.isna(d["spy_ret"]))
positive_years = sum(1 for d in yearly_data if d["ret_2x"] > 0)
best_yr = max(yearly_data, key=lambda d: d["ret_2x"])
worst_yr = min(yearly_data, key=lambda d: d["ret_2x"])

avg_ret_2x = np.mean([d["ret_2x"] for d in yearly_data])
avg_ret_1x = np.mean([d["ret_1x"] for d in yearly_data])
avg_trades = np.mean([d["trades"] for d in yearly_data])
avg_stocks = np.mean([d["n_stocks"] for d in yearly_data])

print(f"\n  {'=' * 120}")
print(f"  SUMMARY -- WAVE RIDER T5 MS BearVol2x")
print(f"  {'=' * 120}")
print(f"  Period:          {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({n_years:.1f} years)")
print()
print(f"  --- BearVol2x (leveraged) ---")
print(f"  CAGR:            {m_lev['cagr']*100:+.1f}% (SPY: {spy_cagr*100:+.1f}%)  ->  {m_lev['cagr']/spy_cagr:.1f}x SPY")
print(f"  Total Return:    {m_lev['total_return']*100:+,.0f}%")
print(f"  Sharpe Ratio:    {m_lev['sharpe']:.2f}")
print(f"  Sortino Ratio:   {m_lev['sortino']:.2f}")
print(f"  Max Drawdown:    {m_lev['max_dd']*100:.1f}%")
print(f"  Avg Leverage:    {result.leverage_series.mean():.2f}x")
print()
print(f"  --- Unlevered (1x) ---")
print(f"  CAGR:            {m_unlev['cagr']*100:+.1f}%")
print(f"  Total Return:    {m_unlev['total_return']*100:+,.0f}%")
print(f"  Sharpe Ratio:    {m_unlev['sharpe']:.2f}")
print(f"  Max Drawdown:    {m_unlev['max_dd']*100:.1f}%")
print()
print(f"  --- Year Stats ---")
print(f"  Beat SPY:        {beat_count}/{total_with_spy} years ({beat_count/total_with_spy*100:.0f}%)")
print(f"  Positive years:  {positive_years}/{total_years} ({positive_years/total_years*100:.0f}%)")
print(f"  Best year:       {best_yr['year']} ({best_yr['ret_2x']:+.1f}%)")
print(f"  Worst year:      {worst_yr['year']} ({worst_yr['ret_2x']:+.1f}%)")
print(f"  Avg annual ret:  {avg_ret_2x:+.1f}% (2x) / {avg_ret_1x:+.1f}% (1x)")
print()
print(f"  --- Trading Activity ---")
print(f"  Total trades:    {cumulative_trades} ({cumulative_trades / 2:.0f} round-trips)")
print(f"  Total rebalances:{len(result.rebalance_dates)}")
print(f"  Avg trades/year: {avg_trades:.1f}")
print(f"  Avg stocks/year: {avg_stocks:.1f} unique")
print(f"  Turnover:        ~{cumulative_trades / n_years / config.top_n / 2 * 100:.0f}% annual")

# Meme filter impact
all_filtered = []
for rd in result.rebalance_dates:
    all_filtered.extend(result.filtered_log.get(rd, []))
total_filtered = len(all_filtered)
pct_filtered = sum(1 for rd in result.rebalance_dates if result.filtered_log.get(rd, [])) / len(result.rebalance_dates) * 100

print()
print(f"  --- Meme Score Filter Impact ---")
print(f"  Stocks removed:  {total_filtered} removals across {len(result.rebalance_dates)} rebalances")
print(f"  Months affected: {pct_filtered:.0f}% of rebalance months had a stock filtered")
if all_filtered:
    top_filt = Counter(all_filtered).most_common(10)
    filt_str = ", ".join(f"{clean_uid(s)}({c}x)" for s, c in top_filt)
    print(f"  Most filtered:   {filt_str}")

print(f"\n  * = delisted stock (identified by UID suffix)")
