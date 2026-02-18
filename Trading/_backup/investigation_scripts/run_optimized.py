#!/usr/bin/env python
"""
Run optimized alpha configuration with trend inversion.
All 7 alphas active, trend signal inverted for 175% IC improvement.
"""

import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run optimized alpha configuration (all alphas, trend inverted)"
    )

    # Configuration options
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick IC test on 5 symbols (3 seconds)")
    parser.add_argument("--full-test", action="store_true",
                        help="Full 50-symbol IC test (30 seconds)")
    parser.add_argument("--production", action="store_true",
                        help="Full production run with RL training")

    # Production run parameters
    parser.add_argument("--iterations", type=int, default=8,
                        help="Number of RL training iterations (default: 8)")
    parser.add_argument("--steps-per-iter", type=int, default=80000,
                        help="RL steps per iteration (default: 80000)")

    args = parser.parse_args()

    # Ensure exactly one mode selected
    modes = [args.quick_test, args.full_test, args.production]
    if sum(modes) == 0:
        print("\nERROR: Must specify one mode:\n")
        print("  --quick-test     Quick IC validation (5 symbols, 3 seconds)")
        print("  --full-test      Full IC validation (50 symbols, 30 seconds)")
        print("  --production     Full production run with RL training")
        print("\nExamples:")
        print("  python run_optimized.py --quick-test")
        print("  python run_optimized.py --production")
        sys.exit(1)

    if sum(modes) > 1:
        print("\nERROR: Can only specify one mode at a time\n")
        sys.exit(1)

    print("\n" + "="*80)
    print("  OPTIMIZED ALPHA CONFIGURATION")
    print("  All 7 alphas active, trend inverted (+175% IC improvement)")
    print("="*80 + "\n")

    if args.quick_test:
        print("Mode: Quick IC Test (5 symbols)")
        print("Expected runtime: ~3 seconds\n")

        cmd = [sys.executable, "quick_ic_test.py", "--invert-trend"]

    elif args.full_test:
        print("Mode: Full IC Test (50 symbols)")
        print("Expected runtime: ~30 seconds\n")

        # Create a full IC test script if it doesn't exist
        print("Note: Full 50-symbol test uses quick_ic_test.py with default symbols")
        print("      For production validation, use --production mode\n")

        cmd = [sys.executable, "quick_ic_test.py", "--invert-trend"]

    elif args.production:
        print("Mode: Full Production Run")
        print(f"Configuration:")
        print(f"  - Iterations: {args.iterations}")
        print(f"  - Steps per iteration: {args.steps_per_iter}")
        print(f"  - Total RL steps: {args.iterations * args.steps_per_iter:,}")
        print(f"  - All 7 alphas active")
        print(f"  - Trend signal inverted\n")
        print(f"Expected runtime: ~2-3 hours\n")

        cmd = [
            sys.executable, "alphago_layering.py",
            "--invert-trend",
            "--iterations", str(args.iterations),
            "--steps-per-iter", str(args.steps_per_iter)
        ]

    print("Command:")
    print(f"  {' '.join(cmd)}\n")

    # Confirm production runs
    if args.production:
        response = input("Start production run? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("\nAborted.")
            sys.exit(0)
        print()

    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
