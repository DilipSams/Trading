"""
Test Vol Features on Real Market Data
======================================

Tests Vol-of-Vol, Gap Analysis, and Volume Anomaly on MSFT 10-year data.

Decision gate: IC > +0.003 for features
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys
sys.path.insert(0, '.')

from alphago_vol_features import VolOfVolFeature, GapAnalysis, VolumeAnomalyFeature


def test_feature_ic(feature_name, signals, forward_returns):
    """
    Test if a feature has predictive power.

    Args:
        feature_name: Name of the feature
        signals: Array of feature values
        forward_returns: Array of forward returns to predict

    Returns:
        Dict with IC metrics
    """
    if len(signals) < 50:
        print(f"[FAIL] {feature_name}: Insufficient signals")
        return None

    # Information Coefficient
    ic = np.corrcoef(signals, forward_returns)[0, 1]

    # T-statistic
    n_samples = len(signals)
    t_stat = ic * np.sqrt(n_samples - 2) / np.sqrt(1 - ic**2 + 1e-10)

    # Significance
    significant = abs(t_stat) > 1.96  # 95% confidence

    print(f"\n{feature_name}:")
    print(f"  IC: {ic:+.4f}")
    print(f"  t-stat: {t_stat:+.2f}")
    print(f"  Significant: {'YES' if significant else 'NO'}")

    # Decision
    if abs(ic) >= 0.003:
        print(f"  [PASS] IC {abs(ic):.4f} >= 0.003 target")
        decision = "IMPLEMENT"
    else:
        print(f"  [FAIL] IC {abs(ic):.4f} below 0.003 threshold")
        decision = "REJECT"

    return {
        'feature': feature_name,
        'ic': ic,
        't_stat': t_stat,
        'n_signals': len(signals),
        'decision': decision
    }


def main():
    print("\n" + "="*70)
    print("VOL FEATURES VALIDATION - MSFT 10 YEARS")
    print("="*70)

    # Download MSFT data
    print("\nDownloading MSFT data (10 years)...")
    ticker = yf.Ticker("MSFT")
    df = ticker.history(period="10y")

    if df.empty or len(df) < 500:
        print("[ERROR] Failed to download data")
        return

    print(f"Downloaded: {len(df)} bars")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    closes = df['Close'].values
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values

    n = len(closes)
    horizon = 5  # 5-day forward return

    # ===================================================================
    # TEST 1: VOL-OF-VOL FEATURE
    # ===================================================================
    print("\n" + "="*70)
    print("TEST 1: Vol-of-Vol Feature")
    print("="*70)

    vov = VolOfVolFeature(vol_window=21, vov_window=21)
    vov_signals = []
    vov_timestamps = []

    for i in range(100, n - horizon):
        result = vov.update(closes[:i+1], i)
        vov_z = result['vol_of_vol_zscore']

        if abs(vov_z) > 0.1:  # Filter near-zero signals
            vov_signals.append(vov_z)
            vov_timestamps.append(i)

    # Compute forward returns
    vov_forward_rets = []
    for idx in vov_timestamps:
        if idx + horizon < n:
            fwd_ret = np.log(closes[idx + horizon] / closes[idx])
            vov_forward_rets.append(fwd_ret)

    vov_signals = np.array(vov_signals[:len(vov_forward_rets)])
    vov_forward_rets = np.array(vov_forward_rets)

    vov_result = test_feature_ic("Vol-of-Vol Z-score", vov_signals, vov_forward_rets)

    # ===================================================================
    # TEST 2: GAP ANALYSIS FEATURES
    # ===================================================================
    print("\n" + "="*70)
    print("TEST 2: Gap Analysis Features")
    print("="*70)

    gap = GapAnalysis(lookback=63)
    gap_signals = []
    gap_timestamps = []

    for i in range(100, n - horizon):
        result = gap.compute(
            opens[:i+1], highs[:i+1], lows[:i+1], closes[:i+1]
        )

        # Test: Overnight dominance as predictor
        gap_signals.append(result['overnight_dominance'])
        gap_timestamps.append(i)

    # Compute forward returns
    gap_forward_rets = []
    for idx in gap_timestamps:
        if idx + horizon < n:
            fwd_ret = np.log(closes[idx + horizon] / closes[idx])
            gap_forward_rets.append(fwd_ret)

    gap_signals = np.array(gap_signals[:len(gap_forward_rets)])
    gap_forward_rets = np.array(gap_forward_rets)

    gap_result = test_feature_ic("Overnight Dominance", gap_signals, gap_forward_rets)

    # ===================================================================
    # TEST 3: VOLUME ANOMALY FEATURE
    # ===================================================================
    print("\n" + "="*70)
    print("TEST 3: Volume Anomaly Feature")
    print("="*70)

    va = VolumeAnomalyFeature(lookback=63)
    vol_signals = []
    vol_timestamps = []

    for i in range(100, n - horizon):
        result = va.compute(volumes[:i+1])
        vol_z = result['volume_zscore']

        if abs(vol_z) > 0.1:  # Filter near-zero signals
            vol_signals.append(vol_z)
            vol_timestamps.append(i)

    # Compute forward returns
    vol_forward_rets = []
    for idx in vol_timestamps:
        if idx + horizon < n:
            fwd_ret = np.log(closes[idx + horizon] / closes[idx])
            vol_forward_rets.append(fwd_ret)

    vol_signals = np.array(vol_signals[:len(vol_forward_rets)])
    vol_forward_rets = np.array(vol_forward_rets)

    vol_result = test_feature_ic("Volume Z-score", vol_signals, vol_forward_rets)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "="*70)
    print("SUMMARY: VOL FEATURES ON MSFT")
    print("="*70)

    results = [r for r in [vov_result, gap_result, vol_result] if r is not None]

    print(f"{'Feature':<30} {'IC':>10} {'t-stat':>10} {'Decision':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['feature']:<30} {r['ic']:>10.4f} {r['t_stat']:>10.2f} {r['decision']:>12}")

    # Count
    implement_count = sum(1 for r in results if r['decision'] == 'IMPLEMENT')
    reject_count = sum(1 for r in results if r['decision'] == 'REJECT')

    print("\n" + "="*70)
    print("FINAL DECISION")
    print("="*70)
    print(f"Features to IMPLEMENT: {implement_count}/{len(results)}")
    print(f"Features to REJECT:    {reject_count}/{len(results)}")

    if implement_count >= 2:
        print(f"\n[SUCCESS] {implement_count} features passed tests!")
        print(f"Recommendation: Add to feature engineering pipeline")
    elif implement_count >= 1:
        print(f"\n[PARTIAL] {implement_count} feature passed, {reject_count} failed")
        print(f"Recommendation: Implement passing features only")
    else:
        print(f"\n[FAIL] All features failed IC tests")
        print(f"Recommendation: Do not implement, revisit hypotheses")


if __name__ == "__main__":
    main()
