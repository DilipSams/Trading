"""
Phase 4: Grid Search for Optimal Parameters
============================================

Test different stop loss configurations to find optimal parameters.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from alphago_stop_loss import AsymmetricStopLoss, StopLossConfig
from test_asymmetric_phase3 import simulate_trade_sequence


def grid_search_parameters():
    """
    Grid search over stop loss parameter space.

    Tests:
    - Loss stop: 1.5%, 2.0%, 2.5%
    - Trail stop: 3%, 5%, 7%
    - 9 total combinations
    """
    print("=" * 70)
    print("PHASE 4: GRID SEARCH FOR OPTIMAL PARAMETERS")
    print("=" * 70)
    print("\nTesting parameter combinations:")
    print("  Loss Stops:  1.5%, 2.0%, 2.5%")
    print("  Trail Stops: 3%, 5%, 7%")
    print("  Total combinations: 9")
    print("\nRunning 100 simulated trades per configuration...\n")

    loss_stops = [0.015, 0.020, 0.025]
    trail_stops = [0.03, 0.05, 0.07]

    results = []

    for i, loss_pct in enumerate(loss_stops, 1):
        for j, trail_pct in enumerate(trail_stops, 1):
            config_num = (i - 1) * 3 + j
            print(f"[{config_num}/9] Testing Loss={loss_pct*100:.1f}%, Trail={trail_pct*100:.1f}%...", end=" ")

            config = StopLossConfig(
                loss_stop_pct=loss_pct,
                profit_trail_pct=trail_pct,
            )
            stop_manager = AsymmetricStopLoss(config)

            metrics, _ = simulate_trade_sequence(
                num_trades=100,
                stop_manager=stop_manager,
                use_stops=True
            )

            results.append({
                'loss_pct': loss_pct,
                'trail_pct': trail_pct,
                'sharpe': metrics['sharpe'],
                'max_dd': metrics['max_drawdown'],
                'total_return': metrics['total_return'],
                'win_loss_ratio': metrics['win_loss_ratio'],
                'avg_loss': metrics['avg_loss'],
                'avg_win': metrics['avg_win'],
                'stop_stats': metrics['stop_stats'],
            })

            print(f"Sharpe: {metrics['sharpe']:.3f}, Max DD: {metrics['max_drawdown']*100:.2f}%")

    # Find optimal configurations
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)

    # Sort by different metrics
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    best_dd = min(results, key=lambda x: x['max_dd'])
    best_win_loss = max(results, key=lambda x: x['win_loss_ratio'])
    smallest_loss = max(results, key=lambda x: x['avg_loss'])  # Closest to 0

    print("\n[1] BEST SHARPE RATIO")
    print(f"    Loss Stop: {best_sharpe['loss_pct']*100:.1f}%, Trail Stop: {best_sharpe['trail_pct']*100:.1f}%")
    print(f"    Sharpe: {best_sharpe['sharpe']:.3f}")
    print(f"    Max DD: {best_sharpe['max_dd']*100:.2f}%")
    print(f"    Return: {best_sharpe['total_return']*100:+.2f}%")

    print("\n[2] LOWEST MAX DRAWDOWN")
    print(f"    Loss Stop: {best_dd['loss_pct']*100:.1f}%, Trail Stop: {best_dd['trail_pct']*100:.1f}%")
    print(f"    Max DD: {best_dd['max_dd']*100:.2f}%")
    print(f"    Sharpe: {best_dd['sharpe']:.3f}")
    print(f"    Return: {best_dd['total_return']*100:+.2f}%")

    print("\n[3] BEST WIN/LOSS RATIO")
    print(f"    Loss Stop: {best_win_loss['loss_pct']*100:.1f}%, Trail Stop: {best_win_loss['trail_pct']*100:.1f}%")
    print(f"    Win/Loss: {best_win_loss['win_loss_ratio']:.2f}x")
    print(f"    Sharpe: {best_win_loss['sharpe']:.3f}")
    print(f"    Avg Loss: {best_win_loss['avg_loss']*100:.2f}%")

    print("\n[4] SMALLEST AVERAGE LOSS")
    print(f"    Loss Stop: {smallest_loss['loss_pct']*100:.1f}%, Trail Stop: {smallest_loss['trail_pct']*100:.1f}%")
    print(f"    Avg Loss: {smallest_loss['avg_loss']*100:.2f}%")
    print(f"    Sharpe: {smallest_loss['sharpe']:.3f}")

    # Full results table
    print("\n" + "=" * 70)
    print("FULL RESULTS TABLE")
    print("=" * 70)
    print(f"\n{'Loss%':<8} {'Trail%':<8} {'Sharpe':<10} {'Max DD%':<10} {'Return%':<10} {'W/L':<8} {'Avg Loss%':<12}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: -x['sharpe']):
        print(f"{r['loss_pct']*100:<8.1f} {r['trail_pct']*100:<8.1f} "
              f"{r['sharpe']:<10.3f} {r['max_dd']*100:<10.2f} "
              f"{r['total_return']*100:<10.2f} {r['win_loss_ratio']:<8.2f} "
              f"{r['avg_loss']*100:<12.2f}")

    # Heatmap visualization
    print("\n" + "=" * 70)
    print("SHARPE RATIO HEATMAP")
    print("=" * 70)
    print("\n                Trail Stop Percentage")
    print("           3.0%      5.0%      7.0%")
    print("         " + "-" * 32)

    for loss_pct in loss_stops:
        sharpes = [r['sharpe'] for r in results
                  if r['loss_pct'] == loss_pct]
        row = f"{loss_pct*100:.1f}%  |  "
        for sharpe in sharpes:
            row += f"{sharpe:6.3f}    "
        print(row)

    print("\n" + "=" * 70)
    print("MAX DRAWDOWN HEATMAP (lower is better)")
    print("=" * 70)
    print("\n                Trail Stop Percentage")
    print("           3.0%      5.0%      7.0%")
    print("         " + "-" * 32)

    for loss_pct in loss_stops:
        dds = [r['max_dd']*100 for r in results
               if r['loss_pct'] == loss_pct]
        row = f"{loss_pct*100:.1f}%  |  "
        for dd in dds:
            row += f"{dd:6.2f}    "
        print(row)

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Score each config: 40% Sharpe, 30% DD, 30% Win/Loss
    best_sharpe_val = max(r['sharpe'] for r in results)
    best_dd_val = min(r['max_dd'] for r in results)
    best_wl_val = max(r['win_loss_ratio'] for r in results)

    for r in results:
        sharpe_score = r['sharpe'] / best_sharpe_val if best_sharpe_val > 0 else 0
        dd_score = 1 - (r['max_dd'] / best_dd_val) if best_dd_val > 0 else 0
        wl_score = r['win_loss_ratio'] / best_wl_val if best_wl_val > 0 else 0

        r['composite_score'] = 0.4 * sharpe_score + 0.3 * dd_score + 0.3 * wl_score

    recommended = max(results, key=lambda x: x['composite_score'])

    print(f"\nOptimal Configuration (Composite Score):")
    print(f"  Loss Stop:     {recommended['loss_pct']*100:.1f}%")
    print(f"  Trail Stop:    {recommended['trail_pct']*100:.1f}%")
    print(f"\nExpected Performance:")
    print(f"  Sharpe Ratio:  {recommended['sharpe']:.3f}")
    print(f"  Max Drawdown:  {recommended['max_dd']*100:.2f}%")
    print(f"  Total Return:  {recommended['total_return']*100:+.2f}%")
    print(f"  Win/Loss:      {recommended['win_loss_ratio']:.2f}x")
    print(f"  Avg Loss:      {recommended['avg_loss']*100:.2f}%")
    print(f"  Avg Win:       {recommended['avg_win']*100:+.2f}%")

    # Stop hit statistics
    if recommended['stop_stats']:
        stats = recommended['stop_stats']
        print(f"\nStop Statistics:")
        print(f"  Loss Stops:    {stats['loss_stops']} ({stats['loss_stop_pct']:.1f}%)")
        print(f"  Trail Stops:   {stats['trail_stops']} ({stats['trail_stop_pct']:.1f}%)")

    print("\n" + "=" * 70)
    print("INSIGHTS")
    print("=" * 70)

    # Analyze patterns
    tight_configs = [r for r in results if r['loss_pct'] == 0.015]
    loose_configs = [r for r in results if r['loss_pct'] == 0.025]

    avg_sharpe_tight = np.mean([r['sharpe'] for r in tight_configs])
    avg_sharpe_loose = np.mean([r['sharpe'] for r in loose_configs])

    print(f"\nTighter loss stops (1.5%):  Avg Sharpe = {avg_sharpe_tight:.3f}")
    print(f"Looser loss stops (2.5%):   Avg Sharpe = {avg_sharpe_loose:.3f}")

    if avg_sharpe_tight > avg_sharpe_loose:
        print("  -> Tighter stops perform better (cut losses faster)")
    else:
        print("  -> Looser stops perform better (less whipsaw)")

    narrow_trail = [r for r in results if r['trail_pct'] == 0.03]
    wide_trail = [r for r in results if r['trail_pct'] == 0.07]

    avg_dd_narrow = np.mean([r['max_dd'] for r in narrow_trail])
    avg_dd_wide = np.mean([r['max_dd'] for r in wide_trail])

    print(f"\nNarrow trail (3%):  Avg Max DD = {avg_dd_narrow*100:.2f}%")
    print(f"Wide trail (7%):    Avg Max DD = {avg_dd_wide*100:.2f}%")

    if avg_dd_narrow < avg_dd_wide:
        print("  -> Tighter trailing locks profits better")
    else:
        print("  -> Wider trailing gives room to run")

    print("\n" + "=" * 70)
    print("[PHASE 4 COMPLETE]")
    print("=" * 70)

    return results, recommended


if __name__ == "__main__":
    results, recommended = grid_search_parameters()
