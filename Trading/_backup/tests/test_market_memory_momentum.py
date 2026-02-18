"""
Test Market Memory Momentum Alpha Hypothesis
=============================================

Hypothesis: Momentum signals gated by short-term (variance ratio) + long-term (Hurst)
memory will have higher IC and prevent crashes.

Expected Results:
- Traditional momentum IC: 0.020 to 0.040
- Memory-gated momentum IC: 0.030 to 0.050 (improvement)
- Crash periods: Memory signal reduces exposure when autocorrelation weakens

Decision Gate: Proceed if IC improvement > 0.010 and t-stat > 1.5
"""

import numpy as np
import pandas as pd
from scipy import stats

def compute_variance_ratio(returns, lag=5):
    """
    Compute variance ratio to detect short-term autocorrelation.
    VR > 1: positive autocorrelation (momentum)
    VR < 1: negative autocorrelation (reversal)
    """
    if len(returns) < lag * 5:
        return 1.0

    # 1-period variance
    var_1 = np.var(returns)
    if var_1 < 1e-10:
        return 1.0

    # k-period variance (non-overlapping)
    k_period_rets = []
    for i in range(0, len(returns) - lag, lag):
        k_ret = np.sum(returns[i:i+lag])
        k_period_rets.append(k_ret)

    if len(k_period_rets) < 3:
        return 1.0

    var_k = np.var(k_period_rets)
    vr = var_k / (lag * var_1)
    return vr


def compute_hurst_exponent(prices, window=126):
    """
    Compute Hurst exponent via R/S analysis.
    H > 0.5: trending (long memory)
    H < 0.5: mean-reverting
    H = 0.5: random walk
    """
    if len(prices) < window:
        return 0.5

    prices = prices[-window:]
    log_prices = np.log(prices + 1e-12)

    # Rescaled range analysis
    lags = [8, 16, 32, 64, 126]
    rs_values = []

    for lag in lags:
        if lag > len(log_prices) // 2:
            continue

        # Split into chunks
        n_chunks = len(log_prices) // lag
        if n_chunks < 2:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = log_prices[i*lag:(i+1)*lag]
            if len(chunk) < lag:
                continue

            # Mean-adjusted series
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk

            # Cumulative sum
            cum_dev = np.cumsum(deviations)

            # Range
            R = np.max(cum_dev) - np.min(cum_dev)

            # Standard deviation
            S = np.std(chunk)

            if S > 1e-10:
                rs_chunk.append(R / S)

        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))

    if len(rs_values) < 3:
        return 0.5

    # Fit log(R/S) = H * log(lag) + c
    lags_log = np.log([x[0] for x in rs_values])
    rs_log = np.log([x[1] for x in rs_values])

    if np.std(lags_log) < 1e-10:
        return 0.5

    hurst, _ = np.polyfit(lags_log, rs_log, 1)
    return float(np.clip(hurst, 0.0, 1.0))


def test_traditional_momentum(prices, lookback=126):
    """Traditional momentum signal."""
    if len(prices) < lookback + 1:
        return 0.0

    mom_return = np.log(prices[-1] / prices[-lookback])
    return float(np.sign(mom_return) * abs(mom_return) ** 0.5 * 0.15)


def test_memory_gated_momentum(prices, returns, lookback=126):
    """Memory-enhanced momentum signal."""
    if len(prices) < lookback + 50:
        return 0.0, {}

    # 1. Traditional momentum
    mom_return = np.log(prices[-1] / prices[-lookback])

    # 2. Short-term memory (variance ratio)
    recent_rets = returns[-100:] if len(returns) > 100 else returns
    vr = compute_variance_ratio(recent_rets, lag=5)
    short_memory = vr - 1.0  # >0 = momentum, <0 = reversal

    # 3. Long-term memory (Hurst)
    hurst = compute_hurst_exponent(prices, window=min(126, len(prices)))
    long_memory = hurst - 0.5  # >0 = trending, <0 = mean-reverting

    # 4. Gate signal by memory
    if short_memory > 0 and long_memory > 0:
        memory_strength = min(short_memory, long_memory * 2.0)
        memory_strength = np.clip(memory_strength, 0.0, 1.0)
        mu = float(np.sign(mom_return) * abs(mom_return) ** 0.5
                  * memory_strength * 0.15)
    else:
        mu = 0.0
        memory_strength = 0.0

    metadata = {
        'vr': vr,
        'short_memory': short_memory,
        'hurst': hurst,
        'long_memory': long_memory,
        'memory_strength': memory_strength,
    }

    return mu, metadata


def run_backtest(prices, returns):
    """
    Backtest both strategies and compute IC.
    """
    n = len(prices)
    if n < 200:
        return None

    signals_trad = []
    signals_memory = []
    forward_returns = []

    for i in range(150, n - 15):
        # Generate signals
        trad_mu = test_traditional_momentum(prices[:i])
        memory_mu, _ = test_memory_gated_momentum(prices[:i], returns[:i])

        # Forward 15-bar return
        fwd_ret = np.log(prices[min(i+15, n-1)] / prices[i])

        signals_trad.append(trad_mu)
        signals_memory.append(memory_mu)
        forward_returns.append(fwd_ret)

    return {
        'signals_trad': np.array(signals_trad),
        'signals_memory': np.array(signals_memory),
        'forward_returns': np.array(forward_returns),
    }


def compute_ic_stats(signals, returns):
    """Compute IC, t-stat, hit rate."""
    if len(signals) != len(returns) or len(signals) < 30:
        return {'ic': 0.0, 't_stat': 0.0, 'n': 0}

    # Remove zeros
    mask = np.abs(signals) > 1e-10
    sig_filt = signals[mask]
    ret_filt = returns[mask]

    if len(sig_filt) < 30:
        return {'ic': 0.0, 't_stat': 0.0, 'n': len(sig_filt)}

    ic = np.corrcoef(sig_filt, ret_filt)[0, 1]
    if np.isnan(ic):
        ic = 0.0

    # T-statistic
    n = len(sig_filt)
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)

    # Hit rate
    correct = np.sum(np.sign(sig_filt) == np.sign(ret_filt))
    hit_rate = correct / n if n > 0 else 0.0

    return {
        'ic': float(ic),
        't_stat': float(t_stat),
        'n': int(n),
        'hit_rate': float(hit_rate),
    }


def main():
    print("\n" + "="*70)
    print("MARKET MEMORY MOMENTUM HYPOTHESIS TEST")
    print("="*70)

    # Generate synthetic data with regime changes
    np.random.seed(42)
    n_bars = 2000

    # Create price series with momentum regime and reversal regime
    prices = [100.0]
    returns = []

    for i in range(n_bars):
        if i < 1000:
            # Momentum regime (positive autocorrelation)
            if len(returns) > 0:
                drift = 0.0005 + 0.3 * returns[-1]  # Autocorrelation
            else:
                drift = 0.0005
        else:
            # Crash/reversal regime (negative autocorrelation)
            if len(returns) > 0:
                drift = -0.0005 - 0.2 * returns[-1]  # Reversal
            else:
                drift = -0.0005

        ret = drift + np.random.randn() * 0.015
        returns.append(ret)
        prices.append(prices[-1] * np.exp(ret))

    prices = np.array(prices)
    returns = np.array(returns)

    print(f"Generated {n_bars} bars of synthetic data")
    print(f"  Regime 1 (bars 0-1000): Momentum (positive autocorrelation)")
    print(f"  Regime 2 (bars 1000-2000): Crash/Reversal (negative autocorrelation)")

    # Run backtest
    results = run_backtest(prices, returns)

    # Compute IC for both strategies
    trad_stats = compute_ic_stats(results['signals_trad'], results['forward_returns'])
    memory_stats = compute_ic_stats(results['signals_memory'], results['forward_returns'])

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Traditional Momentum:")
    print(f"  IC:        {trad_stats['ic']:.4f}")
    print(f"  t-stat:    {trad_stats['t_stat']:.2f}")
    print(f"  Hit Rate:  {trad_stats['hit_rate']:.1%}")
    print(f"  N:         {trad_stats['n']}")

    print(f"\nMemory-Gated Momentum:")
    print(f"  IC:        {memory_stats['ic']:.4f}")
    print(f"  t-stat:    {memory_stats['t_stat']:.2f}")
    print(f"  Hit Rate:  {memory_stats['hit_rate']:.1%}")
    print(f"  N:         {memory_stats['n']}")

    ic_improvement = memory_stats['ic'] - trad_stats['ic']
    print(f"\nIC Improvement: {ic_improvement:+.4f}")

    # Decision gate
    print(f"\n{'='*70}")
    print("DECISION GATE")
    print(f"{'='*70}")
    print(f"Target IC Improvement:  +0.010")
    print(f"Actual IC Improvement:  {ic_improvement:+.4f}")
    print(f"Target t-stat:          1.5")
    print(f"Actual t-stat:          {memory_stats['t_stat']:.2f}")

    if ic_improvement >= 0.010 and memory_stats['t_stat'] >= 1.5:
        print(f"[PASS] Hypothesis confirmed!")
        print(f"   Recommendation: IMPLEMENT Market Memory Momentum Alpha")
    elif ic_improvement >= 0.005:
        print(f"[PARTIAL] IC improvement positive but below target")
        print(f"   Recommendation: Consider implementing with further testing")
    else:
        print(f"[FAIL] IC improvement insufficient")
        print(f"   Recommendation: DO NOT implement")


if __name__ == "__main__":
    main()
