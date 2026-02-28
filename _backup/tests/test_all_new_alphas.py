"""
Comprehensive Test: All New Alphas on MSFT
==========================================

Tests all 5 new alphas:
1. Amihud Liquidity (Expected IC: +0.015-0.030)
2. Short-Term Reversal (Expected IC: +0.020)
3. Vol Term Structure (Expected IC: +0.008)
4. Volume-Price Divergence (Expected IC: +0.008)
5. Calendar Effects (Expected IC: +0.007-0.010)

Decision Gate: IC > +0.003 for quick wins, IC > +0.010 for high-value
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys
sys.path.insert(0, '.')

from alphago_architecture import ArchitectureConfig
from alphago_new_alphas import (
    AmihudLiquidityAlpha,
    ShortTermReversalAlpha,
    VolTermStructureAlpha,
    VolumePriceDivergenceAlpha,
    CalendarAlpha
)


def test_alpha_ic(alpha, df, symbol='MSFT'):
    """
    Test alpha IC on real market data.

    Args:
        alpha: Alpha instance
        df: DataFrame with OHLCV data
        symbol: Symbol name for logging

    Returns:
        Dict with IC metrics
    """
    print(f"\n{'='*70}")
    print(f"Testing {alpha.name} on {symbol}")
    print(f"{'='*70}")

    closes = df['Close'].values
    volumes = df['Volume'].values if 'Volume' in df.columns else None
    timestamps = df.index

    n = len(closes)
    print(f"Bars: {n}")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Generate signals
    signals = []
    timestamps_valid = []

    for i in range(100, n - 20):  # Leave room for forward returns
        sig = alpha.generate(
            closes=closes[:i+1],
            volumes=volumes[:i+1] if volumes is not None else None,
            bar_idx=i,
            timestamp=timestamps[i]
        )

        if sig.is_active and abs(sig.mu) > 1e-6:
            signals.append(sig.mu)
            timestamps_valid.append(i)

    if len(signals) < 50:
        print(f"[FAIL] Insufficient signals: {len(signals)}")
        return None

    signals = np.array(signals)
    print(f"Valid signals: {len(signals)}")
    print(f"Signal stats: mean={signals.mean():+.4f}, std={signals.std():.4f}")

    # Compute forward returns at alpha's horizon
    horizon = alpha.horizon
    forward_rets = []

    for idx in timestamps_valid:
        if idx + horizon < n:
            fwd_ret = np.log(closes[idx + horizon] / closes[idx])
            forward_rets.append(fwd_ret)
        else:
            forward_rets.append(0.0)  # Pad

    forward_rets = np.array(forward_rets[:len(signals)])

    # Information Coefficient
    ic = np.corrcoef(signals, forward_rets)[0, 1]

    # T-statistic
    n_samples = len(signals)
    t_stat = ic * np.sqrt(n_samples - 2) / np.sqrt(1 - ic**2 + 1e-10)

    # Significance
    significant = abs(t_stat) > 1.96  # 95% confidence

    print(f"\nResults:")
    print(f"  Horizon: {horizon} bars")
    print(f"  IC: {ic:+.4f}")
    print(f"  t-stat: {t_stat:+.2f}")
    print(f"  Significant: {'YES' if significant else 'NO'} (|t| > 1.96)")

    # Decision
    if abs(ic) >= 0.010:
        print(f"  [PASS] HIGH VALUE - IC {abs(ic):.4f} >= 0.010 target")
        decision = "IMPLEMENT"
    elif abs(ic) >= 0.003:
        print(f"  [PASS] QUICK WIN - IC {abs(ic):.4f} >= 0.003 target")
        decision = "IMPLEMENT"
    else:
        print(f"  [FAIL] IC {abs(ic):.4f} below 0.003 threshold")
        decision = "REJECT"

    return {
        'alpha': alpha.name,
        'symbol': symbol,
        'ic': ic,
        't_stat': t_stat,
        'n_signals': len(signals),
        'horizon': horizon,
        'decision': decision
    }


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE ALPHA TEST - ALL NEW ALPHAS ON MSFT")
    print("="*70)

    # Download MSFT data
    print("\nDownloading MSFT data (10 years)...")
    ticker = yf.Ticker("MSFT")
    df = ticker.history(period="10y")

    if df.empty or len(df) < 500:
        print("[ERROR] Failed to download data")
        return

    print(f"Downloaded: {len(df)} bars")

    # Create config
    acfg = ArchitectureConfig()

    # Create alphas
    alphas = [
        AmihudLiquidityAlpha(acfg),
        ShortTermReversalAlpha(acfg),
        VolTermStructureAlpha(acfg),
        VolumePriceDivergenceAlpha(acfg),
        CalendarAlpha(acfg),
    ]

    # Test each alpha
    results = []
    for alpha in alphas:
        try:
            result = test_alpha_ic(alpha, df, symbol='MSFT')
            if result:
                results.append(result)
        except Exception as e:
            print(f"[ERROR] {alpha.name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: ALL ALPHAS ON MSFT")
    print(f"{'='*70}")
    print(f"{'Alpha':<30} {'Horizon':>8} {'IC':>10} {'t-stat':>10} {'Decision':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['alpha']:<30} {r['horizon']:>8} {r['ic']:>10.4f} {r['t_stat']:>10.2f} {r['decision']:>12}")

    # Combined metrics
    total_ic = sum(abs(r['ic']) for r in results)
    avg_ic = total_ic / len(results) if results else 0

    print("-" * 70)
    print(f"{'TOTAL ABS IC':<30} {'':<8} {total_ic:>10.4f}")
    print(f"{'AVERAGE ABS IC':<30} {'':<8} {avg_ic:>10.4f}")

    # Count by decision
    implement_count = sum(1 for r in results if r['decision'] == 'IMPLEMENT')
    reject_count = sum(1 for r in results if r['decision'] == 'REJECT')

    print(f"\n{'='*70}")
    print("FINAL DECISION")
    print(f"{'='*70}")
    print(f"Alphas to IMPLEMENT: {implement_count}/{len(results)}")
    print(f"Alphas to REJECT:    {reject_count}/{len(results)}")
    print(f"Total IC Gain:       {total_ic:+.4f}")
    print(f"Expected Sharpe Gain: {total_ic * 15:+.2f} (IC × 15 heuristic)")

    if implement_count >= 3:
        print(f"\n[SUCCESS] {implement_count} alphas passed tests!")
        print(f"Recommendation: Add to production pipeline")
    elif implement_count >= 1:
        print(f"\n[PARTIAL] {implement_count} alphas passed, {reject_count} failed")
        print(f"Recommendation: Implement passing alphas only")
    else:
        print(f"\n[FAIL] All alphas failed IC tests")
        print(f"Recommendation: Do not implement, revisit hypotheses")


if __name__ == "__main__":
    main()
