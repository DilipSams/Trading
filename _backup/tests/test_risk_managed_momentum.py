"""
Test Risk-Managed Momentum Hypothesis
======================================

Hypothesis: Scaling momentum signal by inverse volatility prevents crashes
and improves risk-adjusted returns.

Expected Results:
- Traditional momentum: Sharpe ~0.5, max DD ~30%
- Risk-managed momentum: Sharpe ~0.8, max DD ~15%
- Crash periods: Risk-managed reduces exposure

Decision Gate: Proceed if Sharpe improvement > 0.2 OR max DD reduction > 10%
"""

import numpy as np
import pandas as pd

def compute_momentum_signal(prices, lookback=126):
    """Traditional momentum signal."""
    if len(prices) < lookback + 21:
        return 0.0

    # Fast MA vs Slow MA
    fast = np.mean(prices[-21:])
    slow = np.mean(prices[-lookback:])

    raw_signal = (fast - slow) / (slow + 1e-12)
    mu = float(np.clip(raw_signal, -3.0, 3.0)) * 0.05
    return mu


def compute_risk_managed_momentum(prices, returns, lookback=126):
    """
    Risk-managed momentum: scale by inverse volatility.

    When momentum vol is high -> crashes likely -> scale down
    When momentum vol is low -> safe to be aggressive
    """
    if len(prices) < lookback + 21:
        return 0.0

    # Get traditional signal
    raw_mu = compute_momentum_signal(prices, lookback)

    if len(returns) < 126:
        return raw_mu

    # Compute recent momentum strategy volatility
    # (not price volatility, but momentum return volatility)
    mom_rets = []
    recent_prices = prices[-126:]
    for i in range(21, len(recent_prices)):
        # 21-bar momentum return
        mom_ret = np.log(recent_prices[i] / recent_prices[max(0, i-21)])
        mom_rets.append(mom_ret)

    if len(mom_rets) < 10:
        return raw_mu

    mom_vol = float(np.std(mom_rets))
    target_vol = 0.12 / np.sqrt(252)  # 12% annual target

    # Scale factor: target_vol / realized_vol, capped at 2x
    if mom_vol > 1e-10:
        scale = min(target_vol / mom_vol, 2.0)
    else:
        scale = 1.0

    return raw_mu * scale


def backtest_strategy(prices, returns, strategy_func):
    """
    Backtest a momentum strategy.
    Returns: equity curve, positions, returns
    """
    n = len(prices)
    positions = []
    strategy_returns = []
    equity = [100.0]

    for i in range(150, n - 1):
        # Generate signal
        signal = strategy_func(prices[:i], returns[:i-1])

        # Position: signal is target exposure (-1 to +1)
        position = np.clip(signal / 0.05, -1.0, 1.0)  # Normalize to ±1
        positions.append(position)

        # Realized return
        bar_return = returns[i]
        strategy_ret = position * bar_return
        strategy_returns.append(strategy_ret)

        # Update equity
        equity.append(equity[-1] * (1 + strategy_ret))

    return {
        'equity': np.array(equity),
        'positions': np.array(positions),
        'returns': np.array(strategy_returns),
    }


def compute_performance_stats(returns, equity):
    """Compute Sharpe, max DD, etc."""
    if len(returns) < 10:
        return {'sharpe': 0.0, 'max_dd': 0.0, 'total_ret': 0.0}

    # Sharpe (annualized)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(252) if std_ret > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-10)
    max_dd = abs(np.min(drawdown))

    # Total return
    total_ret = (equity[-1] / equity[0]) - 1.0

    return {
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'total_ret': float(total_ret),
        'mean_ret_daily': float(mean_ret),
        'vol_daily': float(std_ret),
    }


def main():
    print("\n" + "="*70)
    print("RISK-MANAGED MOMENTUM HYPOTHESIS TEST")
    print("="*70)

    # Generate synthetic data with vol regime changes
    np.random.seed(42)
    n_bars = 2000

    prices = [100.0]
    returns = []
    vol_regime = 0.010  # Low vol

    for i in range(n_bars):
        # Add momentum drift
        if len(returns) >= 21:
            momentum = np.mean(returns[-21:])
        else:
            momentum = 0.0

        drift = 0.0002 + 0.5 * momentum  # Positive autocorrelation

        # Vol regime changes
        if i == 1000:
            vol_regime = 0.030  # HIGH VOL REGIME (crash zone)
        elif i == 1500:
            vol_regime = 0.010  # Back to low vol

        ret = drift + np.random.randn() * vol_regime
        returns.append(ret)
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)
    returns = np.array(returns)

    print(f"Generated {n_bars} bars of synthetic data")
    print(f"  Regime 1 (bars 0-1000):    Low vol (1% daily)")
    print(f"  Regime 2 (bars 1000-1500): HIGH VOL CRASH (3% daily)")
    print(f"  Regime 3 (bars 1500-2000): Low vol recovery (1% daily)")

    # Backtest both strategies
    print(f"\n{'='*70}")
    print("BACKTESTING...")
    print(f"{'='*70}")

    trad_results = backtest_strategy(
        prices, returns,
        lambda p, r: compute_momentum_signal(p)
    )

    risk_results = backtest_strategy(
        prices, returns,
        lambda p, r: compute_risk_managed_momentum(p, r)
    )

    # Compute stats
    trad_stats = compute_performance_stats(
        trad_results['returns'],
        trad_results['equity']
    )

    risk_stats = compute_performance_stats(
        risk_results['returns'],
        risk_results['equity']
    )

    print(f"\nTraditional Momentum:")
    print(f"  Sharpe Ratio:      {trad_stats['sharpe']:.2f}")
    print(f"  Max Drawdown:      {trad_stats['max_dd']:.1%}")
    print(f"  Total Return:      {trad_stats['total_ret']:.1%}")
    print(f"  Daily Mean Return: {trad_stats['mean_ret_daily']:.4f}")
    print(f"  Daily Volatility:  {trad_stats['vol_daily']:.4f}")

    print(f"\nRisk-Managed Momentum:")
    print(f"  Sharpe Ratio:      {risk_stats['sharpe']:.2f}")
    print(f"  Max Drawdown:      {risk_stats['max_dd']:.1%}")
    print(f"  Total Return:      {risk_stats['total_ret']:.1%}")
    print(f"  Daily Mean Return: {risk_stats['mean_ret_daily']:.4f}")
    print(f"  Daily Volatility:  {risk_stats['vol_daily']:.4f}")

    sharpe_improvement = risk_stats['sharpe'] - trad_stats['sharpe']
    dd_improvement = trad_stats['max_dd'] - risk_stats['max_dd']

    print(f"\nImprovements:")
    print(f"  Sharpe:       {sharpe_improvement:+.2f}")
    print(f"  Max DD:       {dd_improvement:+.1%} (reduction)")

    # Decision gate
    print(f"\n{'='*70}")
    print("DECISION GATE")
    print(f"{'='*70}")
    print(f"Target Sharpe Improvement:  +0.2")
    print(f"Actual Sharpe Improvement:  {sharpe_improvement:+.2f}")
    print(f"Target DD Reduction:        10%")
    print(f"Actual DD Reduction:        {dd_improvement:.1%}")

    if sharpe_improvement >= 0.2 or dd_improvement >= 0.10:
        print(f"[PASS] Hypothesis confirmed!")
        print(f"   Recommendation: IMPLEMENT Risk-Managed Momentum")
    elif sharpe_improvement >= 0.1 or dd_improvement >= 0.05:
        print(f"[PARTIAL] Improvement positive but below target")
        print(f"   Recommendation: Consider implementing")
    else:
        print(f"[FAIL] Improvement insufficient")
        print(f"   Recommendation: DO NOT implement")


if __name__ == "__main__":
    main()
