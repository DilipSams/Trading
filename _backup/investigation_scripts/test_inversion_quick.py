#!/usr/bin/env python
"""
Quick test to verify signal inversions are working.
Runs 1 iteration with 1000 steps on 5 symbols to quickly verify implementation.
"""

import subprocess
import sys

def run_quick_test(label, extra_args=""):
    """Run a quick 1-iteration test."""
    cmd = [
        sys.executable,
        "alphago_layering.py",
        "--iterations", "1",
        "--steps-per-iter", "1000",
        "--n-symbols", "5"
    ]

    if extra_args:
        cmd.extend(extra_args.split())

    print(f"\n{'='*80}")
    print(f"Running: {label}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract seasonality and vol_premium IC values from output
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if 'MULTI-HORIZON ALPHA PROFILING' in line:
            # Print next 20 lines (the IC table)
            print("\n" + "="*80)
            print(f"RESULTS: {label}")
            print("="*80)
            for j in range(min(20, len(lines) - i)):
                print(lines[i + j])
            break

    return result.returncode

if __name__ == "__main__":
    print("""
================================================================================
            QUICK INVERSION VERIFICATION TEST

  Runs 3 quick tests (1 iteration, 1000 steps, 5 symbols):
  1. Baseline (no inversions)
  2. Seasonality inverted
  3. Both inversions enabled

  Expected: IC signs should flip for inverted alphas
  Time: ~5-10 minutes total
================================================================================
    """)

    # Test 1: Baseline
    run_quick_test("Baseline (no inversions)")

    # Test 2: Seasonality inverted
    run_quick_test("Seasonality Inverted", "--invert-seasonality")

    # Test 3: Both inversions
    run_quick_test("Both Inversions", "--invert-seasonality --invert-vol-premium")

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("""
If inversions are working:
  - Seasonality IC should have opposite sign when --invert-seasonality is used
  - Vol premium IC should have opposite sign when --invert-vol-premium is used

If you see the signs flip, the implementation is correct!

Next step: Run full investigations with:
  python run_investigations.bat    (Windows)
  bash run_investigations.sh       (Linux/Mac)
    """)
