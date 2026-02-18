"""
Test DD Duration + Vol Feature
================================

Feature: Combine price drawdown duration with volatility regime

Hypothesis:
- High DD duration + High vol = Dangerous (reduce exposure)
- High DD duration + Low vol = Recovery (cautiously increase)
- Short DD + Any vol = Normal trading

Implementation:
- DD Duration: Count bars since price peak (symbol-level proxy)
- Vol Regime: Z-score of realized vol vs 60-bar history
- Combined Signal: DD_duration_norm × Vol_regime_z

Decision Gate: If feature improves risk management (reduces DD in backtests)
"""

import numpy as np
import pandas as pd
import yfinance as yf


def compute_dd_duration(closes):
    """
    Compute bars since price peak (drawdown duration).

    Returns:
        Array of drawdown duration in bars
    """
    dd_duration = np.zeros(len(closes))
    peak_idx = 0

    for i in range(len(closes)):
        if closes[i] >= closes[peak_idx]:
            peak_idx = i
            dd_duration[i] = 0
        else:
            dd_duration[i] = i - peak_idx

    return dd_duration


def compute_vol_regime(closes, vol_window=20, lookback=60):
    """
    Compute volatility regime z-score.

    Vol_regime = (current_vol - mean_vol_60) / std_vol_60
    """
    log_rets = np.diff(np.log(closes + 1e-12), prepend=np.log(closes[0]))
    vol_estimates = np.zeros(len(closes))
    vol_regime_z = np.zeros(len(closes))

    # Compute rolling vol
    for i in range(vol_window, len(closes)):
        recent_rets = log_rets[max(0, i-vol_window):i+1]
        vol_estimates[i] = np.std(recent_rets) * np.sqrt(252)

    # Compute z-score
    for i in range(lookback, len(closes)):
        vol_history = vol_estimates[max(0, i-lookback):i]
        vol_mean = np.mean(vol_history)
        vol_std = np.std(vol_history) + 1e-10
        vol_regime_z[i] = (vol_estimates[i] - vol_mean) / vol_std

    return vol_regime_z


def compute_dd_vol_feature(closes, vol_window=20, lookback=60, dd_norm_factor=60):
    """
    Compute DD Duration × Vol Regime feature.

    Args:
        closes: Price array
        vol_window: Window for vol estimation
        lookback: Lookback for vol z-score
        dd_norm_factor: Normalize DD duration by this many bars

    Returns:
        {
            'dd_duration': Raw DD duration in bars,
            'vol_regime_z': Vol z-score,
            'dd_duration_norm': Normalized DD duration [0, 2],
            'dd_vol_signal': Combined feature,
        }
    """
    dd_duration = compute_dd_duration(closes)
    vol_regime_z = compute_vol_regime(closes, vol_window, lookback)

    # Normalize DD duration: 0 bars = 0, 60 bars = 1, 120+ bars = 2
    dd_duration_norm = np.clip(dd_duration / dd_norm_factor, 0, 2)

    # Combined signal: High DD + High Vol = strongly negative
    dd_vol_signal = dd_duration_norm * vol_regime_z

    return {
        'dd_duration': dd_duration,
        'vol_regime_z': vol_regime_z,
        'dd_duration_norm': dd_duration_norm,
        'dd_vol_signal': dd_vol_signal,
    }


def test_dd_vol_feature(symbol='MSFT', period='10y'):
    """
    Test DD+Vol feature on real market data.

    Tests:
    1. Does feature predict future drawdowns?
    2. Does it help reduce risk exposure in dangerous regimes?
    """
    print(f"\n{'='*70}")
    print(f"DD DURATION + VOL FEATURE TEST - {symbol}")
    print(f"{'='*70}")

    # Download data
    print(f"\nDownloading {symbol} data ({period})...")
    df = yf.Ticker(symbol).history(period=period)
    closes = df['Close'].values
    print(f"Bars: {len(closes)}")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Compute features
    result = compute_dd_vol_feature(closes)
    dd_duration = result['dd_duration']
    vol_regime_z = result['vol_regime_z']
    dd_vol_signal = result['dd_vol_signal']

    # Compute forward drawdowns (next 20 bars)
    horizon = 20
    forward_dds = []
    for i in range(len(closes) - horizon):
        future_prices = closes[i:i+horizon+1]
        peak = np.max(future_prices)
        trough = np.min(future_prices)
        forward_dd = (trough - peak) / peak if peak > 0 else 0
        forward_dds.append(forward_dd)

    # Align arrays
    dd_vol_signals = dd_vol_signal[:-horizon]
    forward_dds = np.array(forward_dds)

    # Filter valid signals (after warmup)
    valid_idx = (dd_duration[:-horizon] > 0) | (abs(vol_regime_z[:-horizon]) > 0.1)
    dd_vol_signals = dd_vol_signals[valid_idx]
    forward_dds = forward_dds[valid_idx]

    # Test correlation: Does high DD+Vol predict worse drawdowns?
    if len(dd_vol_signals) > 50:
        corr = np.corrcoef(dd_vol_signals, forward_dds)[0, 1]
        n = len(dd_vol_signals)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2 + 1e-10)

        print(f"\n{'='*70}")
        print(f"PREDICTIVE POWER TEST")
        print(f"{'='*70}")
        print(f"Samples:          {n}")
        print(f"Correlation:      {corr:+.4f} (DD+Vol -> Future DD)")
        print(f"t-stat:           {t_stat:+.2f}")
        print(f"Significant:      {'YES' if abs(t_stat) > 1.96 else 'NO'}")
        print(f"\nInterpretation:")
        if corr < -0.01:
            print(f"  [GOOD] High DD+Vol predicts WORSE future drawdowns")
            print(f"         (Negative correlation: high feature -> deeper DD)")
        elif corr > 0.01:
            print(f"  [WARNING] High DD+Vol predicts BETTER future returns")
            print(f"            (Unexpected: may need signal inversion)")
        else:
            print(f"  [NEUTRAL] No predictive power for future drawdowns")

    # Feature statistics
    print(f"\n{'='*70}")
    print(f"FEATURE STATISTICS")
    print(f"{'='*70}")
    print(f"DD Duration:")
    print(f"  Mean:    {np.mean(dd_duration):.1f} bars")
    print(f"  Median:  {np.median(dd_duration):.1f} bars")
    print(f"  Max:     {np.max(dd_duration):.0f} bars")
    print(f"  % in DD: {np.sum(dd_duration > 0) / len(dd_duration) * 100:.1f}%")

    print(f"\nVol Regime Z:")
    print(f"  Mean:    {np.mean(vol_regime_z[vol_regime_z != 0]):.2f}")
    print(f"  Std:     {np.std(vol_regime_z[vol_regime_z != 0]):.2f}")
    print(f"  Min:     {np.min(vol_regime_z):.2f}")
    print(f"  Max:     {np.max(vol_regime_z):.2f}")

    print(f"\nDD+Vol Signal:")
    print(f"  Mean:    {np.mean(dd_vol_signal):.3f}")
    print(f"  Std:     {np.std(dd_vol_signal):.3f}")
    print(f"  Min:     {np.min(dd_vol_signal):.3f}")
    print(f"  Max:     {np.max(dd_vol_signal):.3f}")

    # Regime analysis
    print(f"\n{'='*70}")
    print(f"REGIME ANALYSIS")
    print(f"{'='*70}")

    # Dangerous regime: High DD (>30 bars) + High Vol (z > 1)
    dangerous = (dd_duration > 30) & (vol_regime_z > 1.0)
    dangerous_pct = np.sum(dangerous) / len(dangerous) * 100

    # Recovery regime: High DD (>30 bars) + Low Vol (z < -0.5)
    recovery = (dd_duration > 30) & (vol_regime_z < -0.5)
    recovery_pct = np.sum(recovery) / len(recovery) * 100

    print(f"Dangerous Regime (DD>30 bars + Vol>1sd): {dangerous_pct:.1f}% of time")
    print(f"Recovery Regime (DD>30 bars + Vol<-0.5sd): {recovery_pct:.1f}% of time")

    if dangerous_pct > 0:
        # Check average forward DD in dangerous vs normal regime
        # Align arrays properly
        dangerous_aligned = dangerous[:len(forward_dds)]
        dangerous_fwd_dd = forward_dds[dangerous_aligned]
        normal_fwd_dd = forward_dds[~dangerous_aligned]

        if len(dangerous_fwd_dd) > 10 and len(normal_fwd_dd) > 10:
            print(f"\nFuture DD in Dangerous Regime: {np.mean(dangerous_fwd_dd)*100:.2f}%")
            print(f"Future DD in Normal Regime:    {np.mean(normal_fwd_dd)*100:.2f}%")
            print(f"Difference:                    {(np.mean(dangerous_fwd_dd) - np.mean(normal_fwd_dd))*100:.2f}%")

    return {
        'symbol': symbol,
        'correlation': corr if len(dd_vol_signals) > 50 else 0.0,
        't_stat': t_stat if len(dd_vol_signals) > 50 else 0.0,
        'dangerous_pct': dangerous_pct,
        'recovery_pct': recovery_pct,
    }


def main():
    print("=" * 70)
    print("DD DURATION + VOL FEATURE VALIDATION")
    print("=" * 70)

    results = []
    for symbol in ['MSFT', 'AAPL', 'SPY', 'GOOGL']:
        result = test_dd_vol_feature(symbol, period='10y')
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: DD+VOL FEATURE ACROSS SYMBOLS")
    print(f"{'='*70}")
    print(f"{'Symbol':<10} {'Correlation':>12} {'t-stat':>10} {'Dangerous %':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['symbol']:<10} {r['correlation']:>12.4f} {r['t_stat']:>10.2f} {r['dangerous_pct']:>11.1f}%")

    avg_corr = np.mean([r['correlation'] for r in results])
    print("-" * 70)
    print(f"{'AVERAGE':<10} {avg_corr:>12.4f}")

    print(f"\n{'='*70}")
    print(f"DECISION")
    print(f"{'='*70}")
    if abs(avg_corr) > 0.05:
        print(f"[PASS] Feature shows predictive power")
        print(f"Recommendation: IMPLEMENT as risk management feature")
        print(f"  - Add to FEATURE_COLUMNS")
        print(f"  - Scale positions inversely with DD+Vol signal")
    else:
        print(f"[WEAK] Feature shows weak predictive power")
        print(f"Recommendation: Implement for monitoring, not for RL features")


if __name__ == "__main__":
    main()
