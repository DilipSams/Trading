"""
Wave Rider T5 MS BearVol2x — Monthly Rebalance Signal Generator.

Run this script to get the current recommended portfolio, leverage level,
and buy/sell instructions. Replays full history to preserve hysteresis state.

Usage:
    python waverider_live.py
"""
import numpy as np
import pandas as pd

from waverider import (
    WaveRiderStrategy, WaveRiderConfig, PortfolioSignal,
    load_universe, load_spy, clean_uid, compute_nav_metrics,
)


def classify(score: float, cfg: WaveRiderConfig) -> str:
    if score <= cfg.meme_max2:
        return "Clean Growth"
    elif score <= cfg.meme_max1:
        return "Gray Zone"
    elif score <= cfg.meme_exclude:
        return "Meme-Adjacent"
    return "Pure Meme"


def main():
    print("  Loading data...", end="", flush=True)
    prices, rankings = load_universe()
    spy_price = load_spy()
    print(" done.")

    cfg = WaveRiderConfig()
    strategy = WaveRiderStrategy(cfg)

    print("  Computing signals and replaying history...", end="", flush=True)
    signal = strategy.current_portfolio(prices, spy_price, rankings)
    print(" done.\n")

    # Also get backtest for YTD/recent stats
    result = strategy.backtest(prices, spy_price, rankings)
    m_lev = compute_nav_metrics(result.nav_leveraged)

    # ===================================================================
    # Header
    # ===================================================================
    print("=" * 70)
    print("  WAVE RIDER T5 MS BearVol2x -- MONTHLY REBALANCE SIGNAL")
    print(f"  Signal date: {signal.date.strftime('%Y-%m-%d')}")
    print("=" * 70)

    # ===================================================================
    # Current portfolio
    # ===================================================================
    n = len(signal.holdings_clean)
    weight_pct = 100.0 / n if n > 0 else 0
    print(f"\n  CURRENT PORTFOLIO ({n} stocks, equal weight {weight_pct:.0f}% each):")
    for i, sym in enumerate(signal.holdings_clean, 1):
        ms = signal.meme_scores.get(sym, 0)
        cls = classify(ms, cfg)
        print(f"    {i}. {sym:<8s} MemeScore: {ms:.0f}  ({cls})")

    # ===================================================================
    # Leverage
    # ===================================================================
    vol_lev = cfg.target_vol / signal.realized_vol * cfg.target_leverage if signal.realized_vol > 0.01 else cfg.target_leverage
    vol_lev_capped = min(cfg.target_leverage, vol_lev)

    print(f"\n  LEVERAGE: {signal.leverage:.2f}x")
    print(f"    Realized vol (21d): {signal.realized_vol:.2f} ann.")
    print(f"    Vol-target leverage: {vol_lev_capped:.2f}x (= {cfg.target_vol}/{signal.realized_vol:.2f} * {cfg.target_leverage})")
    bear_str = "ON  (SPY < SMA200) -> capped at 0.5x" if signal.bear_regime else "OFF (SPY > SMA200)"
    print(f"    Bear gate: {bear_str}")
    eff = signal.leverage * 100
    print(f"    Effective exposure: {signal.leverage:.2f}x * 100% = {eff:.0f}% invested")

    # ===================================================================
    # Trades
    # ===================================================================
    print(f"\n  TRADES THIS MONTH:")
    if signal.buys:
        for sym in signal.buys:
            print(f"    BUY:  {sym}")
    if signal.sells:
        for sym in signal.sells:
            print(f"    SELL: {sym}")
    holds = [s for s in signal.holdings_clean if s not in signal.buys]
    if holds:
        print(f"    HOLD: {', '.join(holds)}")
    if not signal.buys and not signal.sells:
        print(f"    No changes this month.")

    # ===================================================================
    # Top candidates
    # ===================================================================
    print(f"\n  TOP 20 CANDIDATES (ranked by composite momentum score):")
    print(f"    {'Rank':>4s}  {'Symbol':<8s} {'Composite':>9s}  {'MemeScore':>9s}  {'Classification':<16s} {'Status'}")
    print(f"    {'-' * 68}")
    for j, c in enumerate(signal.candidates, 1):
        status = "HELD" if c["in_portfolio"] else ""
        print(f"    {j:>4d}  {c['symbol']:<8s} {c['composite']:>9.2f}  {c['meme_score']:>9.0f}  "
              f"{c['classification']:<16s} {status}")

    # ===================================================================
    # YTD performance
    # ===================================================================
    current_year = signal.date.year
    yr_dates = result.dates[result.dates.year == current_year]
    if len(yr_dates) > 5:
        ytd_ret = (result.nav_leveraged.loc[yr_dates[-1]] / result.nav_leveraged.loc[yr_dates[0]] - 1) * 100
        spy_yr = spy_price[spy_price.index.year == current_year]
        spy_ytd = (spy_yr.iloc[-1] / spy_yr.iloc[0] - 1) * 100 if len(spy_yr) > 5 else 0
        print(f"\n  YTD PERFORMANCE: {ytd_ret:+.1f}% (SPY: {spy_ytd:+.1f}%)")

    # ===================================================================
    # Lifetime summary
    # ===================================================================
    print(f"\n  LIFETIME ({m_lev['n_years']:.0f} years): "
          f"CAGR {m_lev['cagr']*100:+.1f}% | "
          f"Sharpe {m_lev['sharpe']:.2f} | "
          f"MaxDD {m_lev['max_dd']*100:.1f}%")

    print()


if __name__ == "__main__":
    main()
