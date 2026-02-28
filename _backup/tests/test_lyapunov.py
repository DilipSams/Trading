"""
Test Lyapunov Exponent Feature
================================

Measures chaos/predictability in price series via largest Lyapunov exponent.

Lyapunov Exponent (lambda):
- lambda > 0: Chaotic (exponentially diverging, unpredictable)
- lambda = 0: Neutral (random walk)
- lambda < 0: Stable (mean-reverting, predictable)

Hypothesis:
- Trade when lambda < 0.05 (predictable regime)
- Avoid when lambda > 0.1 (chaotic regime)
- Scale positions inversely with lambda

Algorithm: Rosenstein et al. (1993) - Practical method for calculating LLE
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple


def compute_lyapunov_exponent(
    prices: np.ndarray,
    window: int = 100,
    embed_dim: int = 3,
    delay: int = 1,
    min_neighbor_dist: int = 5
) -> float:
    """
    Compute largest Lyapunov exponent using Rosenstein algorithm.

    Steps:
    1. Create time-delay embedding in phase space
    2. For each point, find nearest neighbor (excluding temporal neighbors)
    3. Track log(divergence) over time
    4. Linear regression slope = Lyapunov exponent

    Args:
        prices: Price array
        window: Look-back window for computation
        embed_dim: Embedding dimension (3-5 recommended)
        delay: Time delay for embedding (1 for daily data)
        min_neighbor_dist: Minimum temporal distance between neighbors

    Returns:
        lambda_max: Largest Lyapunov exponent
    """
    if len(prices) < window + embed_dim * delay:
        return 0.0

    # Use log returns for stationarity
    log_rets = np.diff(np.log(prices[-window:] + 1e-12))

    if len(log_rets) < embed_dim * delay + 10:
        return 0.0

    # Phase space embedding (delay embedding)
    embedded = []
    for i in range(len(log_rets) - embed_dim * delay):
        point = [log_rets[i + j * delay] for j in range(embed_dim)]
        embedded.append(point)

    embedded = np.array(embedded)

    if len(embedded) < 20:
        return 0.0

    # For each point, find nearest neighbor and track divergence
    divergences = []

    for i in range(len(embedded) - 10):  # Need future points to track divergence
        # Compute distances to all other points
        distances = np.linalg.norm(embedded - embedded[i], axis=1)

        # Exclude self and temporal neighbors (within min_neighbor_dist)
        exclude_start = max(0, i - min_neighbor_dist)
        exclude_end = min(len(embedded), i + min_neighbor_dist + 1)
        distances[exclude_start:exclude_end] = np.inf

        # Find nearest neighbor
        if np.all(np.isinf(distances)):
            continue

        nearest_idx = np.argmin(distances)
        initial_distance = distances[nearest_idx]

        if initial_distance < 1e-10:  # Too close, skip
            continue

        # Track divergence for up to 10 steps
        for dt in range(1, min(11, len(embedded) - max(i, nearest_idx))):
            if i + dt < len(embedded) and nearest_idx + dt < len(embedded):
                current_dist = np.linalg.norm(
                    embedded[i + dt] - embedded[nearest_idx + dt]
                )

                if current_dist > 1e-10:
                    # Log divergence normalized by initial distance
                    log_div = np.log(current_dist / initial_distance)
                    divergences.append((dt, log_div))

    if len(divergences) < 20:
        return 0.0

    # Linear regression: log(divergence) vs time
    # Slope = Lyapunov exponent
    times, log_divs = zip(*divergences)
    times = np.array(times)
    log_divs = np.array(log_divs)

    # Remove outliers (divergence too large or too small)
    valid = (log_divs > -5) & (log_divs < 5)
    times = times[valid]
    log_divs = log_divs[valid]

    if len(times) < 10:
        return 0.0

    # Linear fit
    slope, intercept = np.polyfit(times, log_divs, 1)

    return float(slope)


def test_lyapunov_regime(symbol='MSFT', period='10y'):
    """
    Test if Lyapunov exponent helps identify predictable vs chaotic regimes.

    Tests:
    1. Correlation with forward returns
    2. Regime classification (low lambda = predictable, high lambda = chaotic)
    3. Sharpe ratio in low-lambda vs high-lambda regimes
    """
    print(f"\n{'='*70}")
    print(f"LYAPUNOV EXPONENT TEST - {symbol}")
    print(f"{'='*70}")

    # Download data
    print(f"\nDownloading {symbol} data ({period})...")
    df = yf.Ticker(symbol).history(period=period)
    closes = df['Close'].values
    print(f"Bars: {len(closes)}")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Compute rolling Lyapunov exponent
    print(f"\nComputing Lyapunov exponents...")
    window = 100
    lyapunov_vals = []
    lyapunov_indices = []

    for i in range(window, len(closes)):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(closes)} ({i/len(closes)*100:.0f}%)", end='\r')

        lam = compute_lyapunov_exponent(
            closes[:i+1],
            window=window,
            embed_dim=3,
            delay=1
        )
        lyapunov_vals.append(lam)
        lyapunov_indices.append(i)

    lyapunov_vals = np.array(lyapunov_vals)
    print(f"\nComputed {len(lyapunov_vals)} Lyapunov values")

    # Compute forward returns (5-bar)
    horizon = 5
    forward_rets = []
    for idx in lyapunov_indices:
        if idx + horizon < len(closes):
            fwd_ret = np.log(closes[idx + horizon] / closes[idx])
            forward_rets.append(fwd_ret)
        else:
            forward_rets.append(np.nan)

    forward_rets = np.array(forward_rets)

    # Filter valid
    valid = ~np.isnan(forward_rets) & (lyapunov_vals != 0)
    lyapunov_vals_valid = lyapunov_vals[valid]
    forward_rets_valid = forward_rets[valid]

    print(f"\n{'='*70}")
    print(f"LYAPUNOV STATISTICS")
    print(f"{'='*70}")
    print(f"Samples:  {len(lyapunov_vals_valid)}")
    print(f"Mean:     {np.mean(lyapunov_vals_valid):+.4f}")
    print(f"Median:   {np.median(lyapunov_vals_valid):+.4f}")
    print(f"Std:      {np.std(lyapunov_vals_valid):.4f}")
    print(f"Min:      {np.min(lyapunov_vals_valid):+.4f}")
    print(f"Max:      {np.max(lyapunov_vals_valid):+.4f}")

    # Correlation with returns
    if len(lyapunov_vals_valid) > 50:
        corr = np.corrcoef(lyapunov_vals_valid, forward_rets_valid)[0, 1]
        n = len(lyapunov_vals_valid)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2 + 1e-10)

        print(f"\n{'='*70}")
        print(f"PREDICTIVE POWER TEST")
        print(f"{'='*70}")
        print(f"IC (Lyapunov vs Forward Returns): {corr:+.4f}")
        print(f"t-stat:                           {t_stat:+.2f}")
        print(f"Significant:                      {'YES' if abs(t_stat) > 1.96 else 'NO'}")

    # Regime analysis
    print(f"\n{'='*70}")
    print(f"REGIME ANALYSIS")
    print(f"{'='*70}")

    # Define regimes
    low_threshold = np.percentile(lyapunov_vals_valid, 33)
    high_threshold = np.percentile(lyapunov_vals_valid, 67)

    print(f"Low Lambda (Predictable):  lambda < {low_threshold:.4f}")
    print(f"Medium Lambda:             {low_threshold:.4f} < lambda < {high_threshold:.4f}")
    print(f"High Lambda (Chaotic):     lambda > {high_threshold:.4f}")

    # Separate returns by regime
    low_lambda_regime = lyapunov_vals_valid < low_threshold
    high_lambda_regime = lyapunov_vals_valid > high_threshold

    low_returns = forward_rets_valid[low_lambda_regime]
    high_returns = forward_rets_valid[high_lambda_regime]

    if len(low_returns) > 10 and len(high_returns) > 10:
        low_sharpe = np.mean(low_returns) / (np.std(low_returns) + 1e-10) * np.sqrt(252/horizon)
        high_sharpe = np.mean(high_returns) / (np.std(high_returns) + 1e-10) * np.sqrt(252/horizon)

        print(f"\nLow Lambda Regime (Predictable):")
        print(f"  Sample count: {len(low_returns)}")
        print(f"  Mean return:  {np.mean(low_returns)*100:.3f}%")
        print(f"  Volatility:   {np.std(low_returns)*100:.3f}%")
        print(f"  Sharpe:       {low_sharpe:.2f}")

        print(f"\nHigh Lambda Regime (Chaotic):")
        print(f"  Sample count: {len(high_returns)}")
        print(f"  Mean return:  {np.mean(high_returns)*100:.3f}%")
        print(f"  Volatility:   {np.std(high_returns)*100:.3f}%")
        print(f"  Sharpe:       {high_sharpe:.2f}")

        print(f"\nSharpe Difference: {low_sharpe - high_sharpe:+.2f}")
        if low_sharpe > high_sharpe + 0.2:
            print(f"  [GOOD] Predictable regime has better Sharpe (+{low_sharpe - high_sharpe:.2f})")
            decision = "IMPLEMENT"
        elif high_sharpe > low_sharpe + 0.2:
            print(f"  [WARNING] Chaotic regime has better Sharpe (+{high_sharpe - low_sharpe:.2f})")
            print(f"            (Unexpected: may indicate signal works inversely)")
            decision = "IMPLEMENT_INVERTED"
        else:
            print(f"  [NEUTRAL] No significant Sharpe difference")
            decision = "SKIP"

        return {
            'symbol': symbol,
            'correlation': corr,
            't_stat': t_stat,
            'low_sharpe': low_sharpe,
            'high_sharpe': high_sharpe,
            'sharpe_diff': low_sharpe - high_sharpe,
            'decision': decision,
        }
    else:
        return {
            'symbol': symbol,
            'correlation': corr if len(lyapunov_vals_valid) > 50 else 0.0,
            't_stat': t_stat if len(lyapunov_vals_valid) > 50 else 0.0,
            'low_sharpe': 0.0,
            'high_sharpe': 0.0,
            'sharpe_diff': 0.0,
            'decision': 'INSUFFICIENT_DATA',
        }


def main():
    print("=" * 70)
    print("LYAPUNOV EXPONENT FEATURE VALIDATION")
    print("=" * 70)
    print("\nNote: This is computationally intensive (may take several minutes)")

    # Test on single symbol first
    result = test_lyapunov_regime('MSFT', period='5y')  # Use 5y for speed

    print(f"\n{'='*70}")
    print(f"DECISION")
    print(f"{'='*70}")

    if result['decision'] == 'IMPLEMENT':
        print(f"[PASS] Lyapunov exponent shows regime differentiation")
        print(f"Recommendation: IMPLEMENT as regime filter")
        print(f"  - Add to FEATURE_COLUMNS as 'lyapunov'")
        print(f"  - Use for position scaling: low lambda -> increase sizing")
    elif result['decision'] == 'IMPLEMENT_INVERTED':
        print(f"[PASS] Lyapunov exponent works inversely")
        print(f"Recommendation: IMPLEMENT with signal inversion")
        print(f"  - Add to FEATURE_COLUMNS as 'lyapunov_inv'")
        print(f"  - Use inverted: high lambda -> increase sizing")
    else:
        print(f"[SKIP] Lyapunov exponent shows weak predictive power")
        print(f"Recommendation: Do not implement (too computational for benefit)")


if __name__ == "__main__":
    main()
