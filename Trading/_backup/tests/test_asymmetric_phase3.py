"""
Phase 3: Performance Comparison Test
====================================

Compare asymmetric stops vs fixed stops across multiple scenarios.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from alphago_stop_loss import AsymmetricStopLoss, StopLossConfig


def simulate_trade_sequence(num_trades=100, stop_manager=None, use_stops=True):
    """
    Simulate a sequence of trades with realistic price paths.

    Returns:
        metrics: Dict with performance statistics
    """
    np.random.seed(42)

    trades = []
    equity_curve = [100000]  # Start with $100k

    for trade_num in range(num_trades):
        # Random entry
        entry_price = 100.0
        position_size = 1000  # $1000 per trade

        # Generate realistic price path (50 bars)
        # 50% trending, 30% choppy, 20% reversal
        scenario = np.random.choice(['trend', 'chop', 'reversal'], p=[0.5, 0.3, 0.2])

        if scenario == 'trend':
            # Trending: 60% up, 40% down
            direction = 1 if np.random.random() < 0.6 else -1
            drift = direction * 0.002  # 0.2% per bar
            volatility = 0.01
            prices = [entry_price]
            for _ in range(50):
                ret = np.random.normal(drift, volatility)
                prices.append(prices[-1] * (1 + ret))

        elif scenario == 'chop':
            # Choppy: mean-reverting
            prices = [entry_price]
            for _ in range(50):
                ret = np.random.normal(-0.0005 * (prices[-1] - entry_price) / entry_price, 0.012)
                prices.append(prices[-1] * (1 + ret))

        else:  # reversal
            # Initial move then reversal
            initial_direction = 1 if np.random.random() < 0.5 else -1
            prices = [entry_price]
            for i in range(50):
                if i < 15:
                    drift = initial_direction * 0.003
                else:
                    drift = -initial_direction * 0.002
                ret = np.random.normal(drift, 0.01)
                prices.append(prices[-1] * (1 + ret))

        # Simulate trade with or without stops
        peak_pnl = 0.0
        atr = entry_price * 0.015  # 1.5% ATR

        exit_bar = 50  # Default: hold to end
        exit_price = prices[-1]
        exit_reason = 'time_exit'

        if use_stops and stop_manager is not None:
            for bar, price in enumerate(prices[1:], 1):
                pnl_pct = (price - entry_price) / entry_price

                if pnl_pct > peak_pnl:
                    peak_pnl = pnl_pct

                result = stop_manager.compute_stop(
                    position_pnl_pct=pnl_pct,
                    entry_price=entry_price,
                    current_price=price,
                    atr=atr,
                    peak_pnl_pct=peak_pnl,
                    is_long=True,
                    realized_vol=0.15
                )

                if result['should_exit']:
                    exit_bar = bar
                    exit_price = result['stop_price']
                    exit_reason = f"{result['stop_type']}_stop"
                    stop_manager.record_stop_hit(result['stop_type'], pnl_pct)
                    break

        # Calculate P&L
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_dollars = position_size * pnl_pct

        trades.append({
            'scenario': scenario,
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'exit_bar': exit_bar,
            'exit_reason': exit_reason,
            'peak_pnl': peak_pnl,
        })

        equity_curve.append(equity_curve[-1] + pnl_dollars)

    # Calculate metrics
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max

    metrics = {
        'num_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'win_loss_ratio': abs(np.mean(wins) / np.mean(losses)) if wins and losses else 0,
        'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
        'sharpe': np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0,
        'max_drawdown': abs(np.min(drawdowns)),
        'avg_trade': np.mean(pnls),
        'stop_stats': stop_manager.get_statistics() if stop_manager else None,
    }

    return metrics, trades


def run_phase3_comparison():
    """Run Phase 3: Baseline vs Enhanced comparison."""
    print("=" * 70)
    print("PHASE 3: PERFORMANCE COMPARISON")
    print("=" * 70)
    print("\nSimulating 100 trades with realistic price paths...")
    print("Scenarios: 50% trending, 30% choppy, 20% reversal\n")

    # Baseline: No stops (hold to end)
    print("[1/2] Running baseline (no stops)...")
    baseline_metrics, _ = simulate_trade_sequence(
        num_trades=100,
        stop_manager=None,
        use_stops=False
    )

    # Enhanced: Asymmetric stops
    print("[2/2] Running enhanced (asymmetric stops)...")
    config = StopLossConfig(
        loss_stop_pct=0.02,
        profit_trail_pct=0.05,
    )
    stop_manager = AsymmetricStopLoss(config)
    enhanced_metrics, _ = simulate_trade_sequence(
        num_trades=100,
        stop_manager=stop_manager,
        use_stops=True
    )

    # Display comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    metrics_to_compare = [
        ('Total Return', 'total_return', '%', 100),
        ('Sharpe Ratio', 'sharpe', '', 1),
        ('Max Drawdown', 'max_drawdown', '%', 100),
        ('Win Rate', 'win_rate', '%', 100),
        ('Avg Win', 'avg_win', '%', 100),
        ('Avg Loss', 'avg_loss', '%', 100),
        ('Win/Loss Ratio', 'win_loss_ratio', 'x', 1),
        ('Avg Trade', 'avg_trade', '%', 100),
    ]

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Enhanced':>12} {'Change':>12}")
    print("-" * 70)

    for name, key, unit, mult in metrics_to_compare:
        baseline_val = baseline_metrics[key] * mult
        enhanced_val = enhanced_metrics[key] * mult

        if baseline_val != 0:
            change_pct = ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
            change_str = f"{change_pct:+.1f}%"
        else:
            change_str = "N/A"

        if unit == '%':
            print(f"{name:<20} {baseline_val:>11.2f}% {enhanced_val:>11.2f}% {change_str:>12}")
        elif unit == 'x':
            print(f"{name:<20} {baseline_val:>11.2f}x {enhanced_val:>11.2f}x {change_str:>12}")
        else:
            print(f"{name:<20} {baseline_val:>12.3f} {enhanced_val:>12.3f} {change_str:>12}")

    # Stop statistics
    if enhanced_metrics['stop_stats']:
        print("\n" + "=" * 70)
        print("STOP LOSS STATISTICS")
        print("=" * 70)
        stats = enhanced_metrics['stop_stats']
        print(f"\nTotal Stops Hit:      {stats['total_stops']}")
        print(f"  Loss Stops:         {stats['loss_stops']} ({stats['loss_stop_pct']:.1f}%)")
        print(f"  Trail Stops:        {stats['trail_stops']} ({stats['trail_stop_pct']:.1f}%)")
        print(f"\nAvg P&L at Stop:")
        print(f"  Loss Stops:         {stats['avg_loss_at_stop']*100:+.2f}%")
        print(f"  Trail Stops:        {stats['avg_profit_at_stop']*100:+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    improvements = []
    if enhanced_metrics['sharpe'] > baseline_metrics['sharpe']:
        improvements.append(f"Sharpe +{(enhanced_metrics['sharpe'] - baseline_metrics['sharpe']):.2f}")
    if enhanced_metrics['max_drawdown'] < baseline_metrics['max_drawdown']:
        dd_reduction = (baseline_metrics['max_drawdown'] - enhanced_metrics['max_drawdown']) / baseline_metrics['max_drawdown'] * 100
        improvements.append(f"Max DD -{dd_reduction:.1f}%")
    if abs(enhanced_metrics['avg_loss']) < abs(baseline_metrics['avg_loss']):
        loss_reduction = (abs(baseline_metrics['avg_loss']) - abs(enhanced_metrics['avg_loss'])) / abs(baseline_metrics['avg_loss']) * 100
        improvements.append(f"Avg Loss -{loss_reduction:.1f}%")

    if improvements:
        print("\n[SUCCESS] Asymmetric stops show improvement:")
        for imp in improvements:
            print(f"  - {imp}")
    else:
        print("\n[NEUTRAL] Performance similar to baseline")
        print("  (This is expected with only 100 random trades)")

    print("\n" + "=" * 70)
    print("[PHASE 3 COMPLETE]")
    print("=" * 70)

    return baseline_metrics, enhanced_metrics


if __name__ == "__main__":
    run_phase3_comparison()
