#!/usr/bin/env python3
"""
Quick verification test for the new trade_entries logging feature.
Tests that every BUY and SELL action is logged separately.
"""

import sys
import numpy as np
from alphago_architecture import InstitutionalPipeline

def test_trade_entries_logging():
    """Test that trade_entries logs each BUY/SELL action"""
    print("\n" + "="*80)
    print("TRADE ENTRIES LOGGING VERIFICATION TEST")
    print("="*80)

    # Create a simple test pipeline
    pip = InstitutionalPipeline()

    # Simulate a sequence of trades
    print("\n  Simulating trade sequence...")
    print("  Bar 1: BUY  100 shares @ $100.00")
    print("  Bar 2: BUY   50 shares @ $110.00  (pyramid)")
    print("  Bar 3: SELL  75 shares @ $120.00  (partial close)")
    print("  Bar 4: SELL  75 shares @ $115.00  (full close)")

    # Bar 1: Initial BUY
    pip.cs = 0
    pip._trade(
        action="BUY",
        shares=100,
        fill_price=100.0,
        commission=1.0,
        slippage=0.0
    )

    # Bar 2: Add to position (pyramid)
    pip.cs = 1
    pip._trade(
        action="BUY",
        shares=50,
        fill_price=110.0,
        commission=0.5,
        slippage=0.0
    )

    # Bar 3: Partial close
    pip.cs = 2
    pip._trade(
        action="SELL",
        shares=75,
        fill_price=120.0,
        commission=0.75,
        slippage=0.0
    )

    # Bar 4: Full close
    pip.cs = 3
    pip._trade(
        action="SELL",
        shares=75,
        fill_price=115.0,
        commission=0.75,
        slippage=0.0
    )

    # Check trade_entries
    print(f"\n  {'='*80}")
    print(f"  RESULTS")
    print(f"  {'='*80}")

    entries = pip.trade_entries
    print(f"\n  Total trade entries logged: {len(entries)}")
    print(f"  Expected: 4 entries (2 BUY + 2 SELL)")

    if len(entries) != 4:
        print(f"\n  ❌ FAILED: Expected 4 entries, got {len(entries)}")
        return False

    # Display each entry
    print(f"\n  {'='*80}")
    print(f"  TRADE ENTRY LOG")
    print(f"  {'='*80}")
    print(f"  {'Bar':<6} {'Action':<12} {'Shares':<10} {'Price':<12} {'Cost':<10} {'Realized P&L':<14} {'Position':<12}")
    print(f"  {'-'*80}")

    for entry in entries:
        bar = entry['bar']
        action = entry['action']
        shares = entry['shares']
        price = entry['price']
        cost = entry['cost']
        realized_pnl = entry['realized_pnl']
        pos_before = entry['position_before']
        pos_after = entry['position_after']

        position_str = f"{pos_before:>5.0f} → {pos_after:>5.0f}"
        print(f"  {bar:<6} {action:<12} {shares:<10.0f} ${price:<11.2f} ${cost:<9.2f} ${realized_pnl:<13.2f} {position_str}")

    # Verify sequence
    print(f"\n  {'='*80}")
    print(f"  VALIDATION")
    print(f"  {'='*80}")

    checks = [
        (entries[0]['action'] == 'BUY', "Entry 1 is BUY"),
        (entries[0]['shares'] == 100, "Entry 1 shares = 100"),
        (entries[1]['action'] == 'BUY', "Entry 2 is BUY"),
        (entries[1]['shares'] == 50, "Entry 2 shares = 50"),
        (entries[2]['action'] == 'SELL', "Entry 3 is SELL"),
        (entries[2]['shares'] == 75, "Entry 3 shares = 75"),
        (entries[3]['action'] == 'SELL', "Entry 4 is SELL"),
        (entries[3]['shares'] == 75, "Entry 4 shares = 75"),
    ]

    all_passed = True
    for passed, description in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {description}")
        if not passed:
            all_passed = False

    # Summary
    print(f"\n  {'='*80}")
    if all_passed:
        print(f"  ✅ SUCCESS: All trade entries logged correctly!")
        print(f"  {'='*80}\n")
        return True
    else:
        print(f"  ❌ FAILED: Some checks did not pass")
        print(f"  {'='*80}\n")
        return False

if __name__ == "__main__":
    success = test_trade_entries_logging()
    sys.exit(0 if success else 1)
