"""
Test Market Memory Momentum (INVERTED) on Multiple Symbols
============================================================

Test the inverted signal hypothesis on SPY, AAPL, MSFT, GOOGL to confirm
that negative IC is consistent and can be inverted for positive alpha.
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


def test_single_symbol(symbol, period='10y'):
    """Test Market Memory Momentum on a single symbol."""
    print(f"\n{'='*70}")
    print(f"Testing {symbol}")
    print(f"{'='*70}")

    try:
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty or len(df) < 300:
            print(f"  ERROR: Insufficient data for {symbol}")
            return None

        prices = df['Close'].values
        returns = np.diff(np.log(prices + 1e-12))

        print(f"  Bars: {len(prices)}")
        print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Total Return: {(prices[-1] / prices[0] - 1):.1%}")

        # Run backtest
        lookback = 126
        signals_trad = []
        signals_memory = []
        signals_memory_inverted = []
        forward_returns = []

        n = len(prices)
        for i in range(150, n - 15):
            # Traditional momentum
            trad_mu = test_traditional_momentum(prices[:i], lookback=lookback)

            # Memory-gated momentum (original)
            memory_mu, metadata = test_memory_gated_momentum(
                prices[:i], returns[:i-1], lookback=lookback
            )

            # Inverted memory signal
            memory_mu_inverted = -memory_mu if memory_mu != 0 else 0.0

            # Forward 15-bar return
            fwd_ret = np.log(prices[min(i+15, n-1)] / prices[i])

            signals_trad.append(trad_mu)
            signals_memory.append(memory_mu)
            signals_memory_inverted.append(memory_mu_inverted)
            forward_returns.append(fwd_ret)

        signals_trad = np.array(signals_trad)
        signals_memory = np.array(signals_memory)
        signals_memory_inverted = np.array(signals_memory_inverted)
        forward_returns = np.array(forward_returns)

        # Compute IC stats
        trad_stats = compute_ic_stats(signals_trad, forward_returns)
        memory_stats = compute_ic_stats(signals_memory, forward_returns)
        memory_inv_stats = compute_ic_stats(signals_memory_inverted, forward_returns)

        print(f"\n  Traditional Momentum:")
        print(f"    IC: {trad_stats['ic']:+.4f}, t-stat: {trad_stats['t_stat']:+.2f}, N: {trad_stats['n']}")

        print(f"\n  Memory-Gated (Original):")
        print(f"    IC: {memory_stats['ic']:+.4f}, t-stat: {memory_stats['t_stat']:+.2f}, N: {memory_stats['n']}")

        print(f"\n  Memory-Gated (INVERTED):")
        print(f"    IC: {memory_inv_stats['ic']:+.4f}, t-stat: {memory_inv_stats['t_stat']:+.2f}, N: {memory_inv_stats['n']}")

        ic_improvement = memory_inv_stats['ic'] - trad_stats['ic']
        print(f"\n  IC Improvement (inverted vs traditional): {ic_improvement:+.4f}")

        return {
            'symbol': symbol,
            'trad_ic': trad_stats['ic'],
            'memory_ic': memory_stats['ic'],
            'memory_inv_ic': memory_inv_stats['ic'],
            'memory_inv_tstat': memory_inv_stats['t_stat'],
            'n_trades': memory_inv_stats['n'],
            'ic_improvement': ic_improvement,
        }

    except Exception as e:
        print(f"  ERROR testing {symbol}: {e}")
        return None


def main():
    print("\n" + "="*70)
    print("MARKET MEMORY MOMENTUM - MULTI-SYMBOL VALIDATION")
    print("Testing INVERTED signal on SPY, AAPL, MSFT, GOOGL, NVDA")
    print("="*70)

    symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'NVDA']
    results = []

    for symbol in symbols:
        result = test_single_symbol(symbol, period='10y')
        if result:
            results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: INVERTED MARKET MEMORY MOMENTUM")
    print(f"{'='*70}")
    print(f"{'Symbol':<8} {'Trad IC':>8} {'Mem IC':>8} {'Inv IC':>8} {'t-stat':>8} {'Improve':>8} {'Decision':>10}")
    print("-" * 70)

    total_ic = 0.0
    total_tstat = 0.0
    passing = 0

    for r in results:
        decision = "PASS" if r['memory_inv_ic'] > 0.010 and abs(r['memory_inv_tstat']) > 1.5 else "FAIL"
        if decision == "PASS":
            passing += 1

        print(f"{r['symbol']:<8} {r['trad_ic']:>8.4f} {r['memory_ic']:>8.4f} {r['memory_inv_ic']:>8.4f} "
              f"{r['memory_inv_tstat']:>8.2f} {r['ic_improvement']:>8.4f} {decision:>10}")

        total_ic += r['memory_inv_ic']
        total_tstat += abs(r['memory_inv_tstat'])

    avg_ic = total_ic / len(results) if results else 0.0
    avg_tstat = total_tstat / len(results) if results else 0.0

    print("-" * 70)
    print(f"{'AVERAGE':<8} {'':<8} {'':<8} {avg_ic:>8.4f} {avg_tstat:>8.2f} {'':<8} {passing}/{len(results)} PASS")

    # Decision gate
    print(f"\n{'='*70}")
    print("FINAL DECISION GATE")
    print(f"{'='*70}")
    print(f"Target: IC > +0.010, t-stat > 1.5, majority of symbols PASS")
    print(f"Result: Avg IC = {avg_ic:+.4f}, Avg t-stat = {avg_tstat:.2f}, {passing}/{len(results)} PASS")

    if avg_ic >= 0.010 and avg_tstat >= 1.5 and passing >= len(results) * 0.6:
        print(f"\n[PASS] Inverted Market Memory Momentum validated!")
        print(f"   Recommendation: IMPLEMENT with signal inversion")
    elif avg_ic >= 0.005:
        print(f"\n[PARTIAL] Inverted signal shows promise but needs more validation")
        print(f"   Recommendation: Test on more symbols or longer periods")
    else:
        print(f"\n[FAIL] Inverted signal does not meet criteria")
        print(f"   Recommendation: DO NOT implement")


if __name__ == "__main__":
    main()
