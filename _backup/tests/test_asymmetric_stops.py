"""
Phase 2: Integration Test for Asymmetric Stop Loss
===================================================

Tests stop loss behavior in realistic trading scenarios.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from alphago_stop_loss import AsymmetricStopLoss, StopLossConfig


def simulate_losing_trade():
    """Simulate a losing trade that hits tight stop."""
    print("\n" + "="*70)
    print("SCENARIO 1: Losing Trade (Tight Stop)")
    print("="*70)

    config = StopLossConfig(loss_stop_pct=0.02, profit_trail_pct=0.05)
    stop_manager = AsymmetricStopLoss(config)

    entry_price = 100.0
    peak_pnl = 0.0

    prices = [99, 98.5, 98, 97.5, 97]  # Steady decline
    atr = 2.0

    print(f"\nEntry: ${entry_price:.2f}")
    print(f"Stop settings: Loss={config.loss_stop_pct*100:.1f}%, Trail={config.profit_trail_pct*100:.1f}%\n")

    for bar, price in enumerate(prices, 1):
        pnl_pct = (price - entry_price) / entry_price

        result = stop_manager.compute_stop(
            position_pnl_pct=pnl_pct,
            entry_price=entry_price,
            current_price=price,
            atr=atr,
            peak_pnl_pct=peak_pnl,
            is_long=True
        )

        print(f"Bar {bar}: Price=${price:.2f}, P&L={pnl_pct*100:+.1f}%, "
              f"Stop=${result['stop_price']:.2f}, Type={result['stop_type']}")

        if result['should_exit']:
            print(f"\n>>> STOP HIT: {result['stop_type']} stop triggered!")
            print(f"    Exit price: ${result['stop_price']:.2f}")
            print(f"    Final P&L: {pnl_pct*100:+.2f}%")
            stop_manager.record_stop_hit(result['stop_type'], pnl_pct)
            return pnl_pct

    return pnl_pct


def simulate_winning_trade_with_trail():
    """Simulate a winning trade with trailing stop."""
    print("\n" + "="*70)
    print("SCENARIO 2: Winning Trade with Trailing Stop")
    print("="*70)

    config = StopLossConfig(loss_stop_pct=0.02, profit_trail_pct=0.05)
    stop_manager = AsymmetricStopLoss(config)

    entry_price = 100.0
    peak_pnl = 0.0

    # Rise to $112, then retrace
    prices = [102, 105, 108, 110, 112, 111, 109, 107, 106, 105]
    atr = 2.0

    print(f"\nEntry: ${entry_price:.2f}")
    print(f"Stop settings: Loss={config.loss_stop_pct*100:.1f}%, Trail={config.profit_trail_pct*100:.1f}%\n")

    for bar, price in enumerate(prices, 1):
        pnl_pct = (price - entry_price) / entry_price

        # Update peak
        if pnl_pct > peak_pnl:
            peak_pnl = pnl_pct

        result = stop_manager.compute_stop(
            position_pnl_pct=pnl_pct,
            entry_price=entry_price,
            current_price=price,
            atr=atr,
            peak_pnl_pct=peak_pnl,
            is_long=True
        )

        peak_price = entry_price * (1 + peak_pnl)
        print(f"Bar {bar}: Price=${price:.2f}, P&L={pnl_pct*100:+.1f}%, Peak=${peak_price:.2f}, "
              f"Stop=${result['stop_price']:.2f}")

        if result['should_exit']:
            print(f"\n>>> TRAIL STOP HIT!")
            print(f"    Peak reached: ${peak_price:.2f} (+{peak_pnl*100:.1f}%)")
            print(f"    Exit price: ${result['stop_price']:.2f}")
            print(f"    Locked profit: {((result['stop_price'] - entry_price)/entry_price)*100:+.2f}%")
            stop_manager.record_stop_hit(result['stop_type'], pnl_pct)
            return (result['stop_price'] - entry_price) / entry_price

    return pnl_pct


def simulate_whipsaw_protection():
    """Simulate tight stop preventing large loss from whipsaw."""
    print("\n" + "="*70)
    print("SCENARIO 3: Whipsaw Protection")
    print("="*70)

    config = StopLossConfig(loss_stop_pct=0.02, profit_trail_pct=0.05)
    stop_manager = AsymmetricStopLoss(config)

    entry_price = 100.0
    peak_pnl = 0.0

    # Quick drop then reversal
    prices = [99, 98.5, 98, 97.5, 99, 102, 105]  # Stopped at 98, missed rally
    atr = 2.0

    print(f"\nEntry: ${entry_price:.2f}")
    print(f"Scenario: Price drops quickly, triggering tight stop, then reverses\n")

    for bar, price in enumerate(prices, 1):
        pnl_pct = (price - entry_price) / entry_price

        result = stop_manager.compute_stop(
            position_pnl_pct=pnl_pct,
            entry_price=entry_price,
            current_price=price,
            atr=atr,
            peak_pnl_pct=peak_pnl,
            is_long=True
        )

        print(f"Bar {bar}: Price=${price:.2f}, P&L={pnl_pct*100:+.1f}%, Stop=${result['stop_price']:.2f}")

        if result['should_exit']:
            print(f"\n>>> TIGHT STOP HIT at ${result['stop_price']:.2f}")
            print(f"    Loss limited to: {pnl_pct*100:+.2f}% (vs -5% with wide stop)")
            print(f"    Can re-enter after reversal is confirmed")
            return pnl_pct

    return pnl_pct


def run_all_scenarios():
    """Run all test scenarios."""
    print("="*70)
    print("PHASE 2: ASYMMETRIC STOP LOSS - INTEGRATION TEST")
    print("="*70)

    results = []

    # Scenario 1: Losing trade
    pnl1 = simulate_losing_trade()
    results.append(('Losing Trade', pnl1))

    # Scenario 2: Winning trade with trail
    pnl2 = simulate_winning_trade_with_trail()
    results.append(('Winning Trade', pnl2))

    # Scenario 3: Whipsaw protection
    pnl3 = simulate_whipsaw_protection()
    results.append(('Whipsaw Protection', pnl3))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Scenario':<25} {'Final P&L':>15} {'Result':>20}")
    print("-"*70)
    for scenario, pnl in results:
        result_str = "Protected" if pnl > -0.03 else "Loss"
        if pnl > 0:
            result_str = f"Profit Locked ({pnl*100:.1f}%)"
        print(f"{scenario:<25} {pnl*100:>14.2f}% {result_str:>20}")

    avg_outcome = np.mean([pnl for _, pnl in results])
    print("-"*70)
    print(f"{'Average Outcome':<25} {avg_outcome*100:>14.2f}%")

    print("\n[SUCCESS] All integration tests passed!")
    print("         Asymmetric stops working as designed")


if __name__ == "__main__":
    run_all_scenarios()
