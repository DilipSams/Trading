"""
Test Market Memory Momentum on REAL Data (MSFT)
================================================

Test hypothesis on actual market data instead of synthetic.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from test_market_memory_momentum import (
    compute_variance_ratio,
    compute_hurst_exponent,
    test_traditional_momentum,
    test_memory_gated_momentum,
    compute_ic_stats,
)


def download_data(symbol='MSFT', period='10y'):
    """Download real market data."""
    print(f"Downloading {symbol} data ({period})...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        raise ValueError(f"No data downloaded for {symbol}")

    print(f"Downloaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def run_real_data_test(symbol='MSFT'):
    """Run Market Memory Momentum test on real data."""
    print("\n" + "="*70)
    print(f"MARKET MEMORY MOMENTUM TEST - REAL DATA ({symbol})")
    print("="*70)

    # Download data
    df = download_data(symbol, period='10y')

    prices = df['Close'].values
    returns = np.diff(np.log(prices + 1e-12))

    print(f"\nData Statistics:")
    print(f"  Bars:           {len(prices)}")
    print(f"  Daily Vol:      {np.std(returns):.4f}")
    print(f"  Annualized Vol: {np.std(returns) * np.sqrt(252):.2%}")
    print(f"  Total Return:   {(prices[-1] / prices[0] - 1):.1%}")

    # Run backtest
    lookback = 126
    signals_trad = []
    signals_memory = []
    forward_returns = []
    metadata_list = []

    n = len(prices)
    for i in range(150, n - 15):
        # Traditional momentum
        trad_mu = test_traditional_momentum(prices[:i], lookback=lookback)

        # Memory-gated momentum
        memory_mu, metadata = test_memory_gated_momentum(
            prices[:i], returns[:i-1], lookback=lookback
        )

        # Forward 15-bar return
        fwd_ret = np.log(prices[min(i+15, n-1)] / prices[i])

        signals_trad.append(trad_mu)
        signals_memory.append(memory_mu)
        forward_returns.append(fwd_ret)
        metadata_list.append(metadata)

    signals_trad = np.array(signals_trad)
    signals_memory = np.array(signals_memory)
    forward_returns = np.array(forward_returns)

    # Compute IC stats
    trad_stats = compute_ic_stats(signals_trad, forward_returns)
    memory_stats = compute_ic_stats(signals_memory, forward_returns)

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

    # Analyze metadata
    if metadata_list:
        vr_values = [m['vr'] for m in metadata_list if 'vr' in m]
        hurst_values = [m['hurst'] for m in metadata_list if 'hurst' in m]

        print(f"\nMemory Metrics:")
        print(f"  Variance Ratio:  {np.mean(vr_values):.3f} ± {np.std(vr_values):.3f}")
        print(f"  Hurst Exponent:  {np.mean(hurst_values):.3f} ± {np.std(hurst_values):.3f}")
        print(f"  % Time Trading:  {memory_stats['n'] / trad_stats['n']:.1%}")

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
        print(f"[PASS] Hypothesis confirmed on {symbol} real data!")
        print(f"   Recommendation: IMPLEMENT Market Memory Momentum Alpha")
        return True
    elif ic_improvement >= 0.005:
        print(f"[PARTIAL] IC improvement positive but below target")
        print(f"   Recommendation: Test on more symbols before deciding")
        return None
    else:
        print(f"[FAIL] IC improvement insufficient on {symbol}")
        print(f"   Recommendation: DO NOT implement")
        return False


if __name__ == "__main__":
    result = run_real_data_test('MSFT')
