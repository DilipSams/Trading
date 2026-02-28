#!/usr/bin/env python
"""
Deep dive investigations into alpha signal inversions and timing.

Tests 4 hypotheses:
1. Seasonality inversion: Does flipping the calendar signal improve IC?
2. Value timing: Does longer holding period (15-bar) improve performance?
3. Vol premium inversion: Should we buy vol instead of sell?
4. Ensemble robustness: How does meta-learner handle disabled alphas?
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_name: str, disable_alphas: str = "", extra_args: str = ""):
    """Run a single test configuration and log results."""
    cmd = [
        sys.executable,
        "alphago_layering.py",
        "--iterations", "8",
        "--steps-per-iter", "80000",
        "--n-symbols", "50"
    ]

    if disable_alphas:
        cmd.extend(["--disable-alphas", disable_alphas])

    if extra_args:
        cmd.extend(extra_args.split())

    output_file = f"investigation_{test_name}.log"

    print(f"\n{'='*80}")
    print(f"Running: {test_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    with open(output_file, 'w') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    print(f"✓ {test_name} completed (exit code: {result.returncode})")
    print(f"  Results saved to: {output_file}")

    return output_file

def main():
    """Run all 4 investigations sequentially."""

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║              MULTI-HORIZON ALPHA INVERSION INVESTIGATIONS               ║
║                                                                          ║
║  Testing 4 hypotheses based on 50-symbol baseline results:              ║
║  1. Seasonality inversion (IC = -0.014 → flip sign?)                    ║
║  2. Value timing (IC best at 15-bar → reduce rebal freq?)               ║
║  3. Vol premium inversion (IC = -0.062 → buy vol instead?)              ║
║  4. Ensemble robustness (disable weak alphas → meta-learner adapts?)    ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Baseline (already completed - just document)
    print("\nBASELINE (already completed):")
    print("  - All 7 alphas enabled")
    print("  - Seasonality: IC = -0.014 (1-bar), -0.028 (5-bar), -0.046 (15-bar)")
    print("  - Vol premium: IC = -0.062 (15-bar)")
    print("  - Value: IC = +0.055 (15-bar) - best signal")

    results = {}

    # Investigation 1: Disable problematic alphas (seasonality + vol_premium)
    print("\n" + "="*80)
    print("INVESTIGATION 1: Ensemble Robustness")
    print("  Hypothesis: Disabling anti-predictive alphas improves performance")
    print("  Disable: seasonality, vol_premium (both negative IC)")
    print("="*80)

    results['robust_ensemble'] = run_test(
        test_name="01_robust_ensemble",
        disable_alphas="seasonality,vol_premium"
    )

    # Investigation 2: Disable all but value (isolate best performer)
    print("\n" + "="*80)
    print("INVESTIGATION 2: Value Alpha Isolation")
    print("  Hypothesis: Value alone outperforms noisy ensemble")
    print("  Disable: all except value")
    print("="*80)

    results['value_only'] = run_test(
        test_name="02_value_only",
        disable_alphas="rl_ppo_mcts,trend_follow,mean_reversion,carry,seasonality,vol_premium"
    )

    # Investigation 3: Test mean_reversion alone (high variance alpha)
    print("\n" + "="*80)
    print("INVESTIGATION 3: Mean Reversion Isolation")
    print("  Hypothesis: MR shows promise but needs isolation to assess stability")
    print("  Disable: all except mean_reversion")
    print("="*80)

    results['mr_only'] = run_test(
        test_name="03_mr_only",
        disable_alphas="rl_ppo_mcts,trend_follow,value,carry,seasonality,vol_premium"
    )

    # Investigation 4: Top 3 alphas only (value, mean_reversion, trend_follow)
    print("\n" + "="*80)
    print("INVESTIGATION 4: Top 3 Alphas")
    print("  Hypothesis: Keep alphas with positive IC, drop the rest")
    print("  Enable: value, mean_reversion, trend_follow")
    print("  Disable: rl_ppo_mcts, carry, seasonality, vol_premium")
    print("="*80)

    results['top3'] = run_test(
        test_name="04_top3_alphas",
        disable_alphas="rl_ppo_mcts,carry,seasonality,vol_premium"
    )

    # Summary
    print("\n" + "="*80)
    print("ALL INVESTIGATIONS COMPLETED")
    print("="*80)
    print("\nResult files:")
    for name, filepath in results.items():
        print(f"  {name:20s} → {filepath}")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                           NEXT STEPS                                     ║
╚══════════════════════════════════════════════════════════════════════════╝

To extract and compare multi-horizon IC results:

  grep -A 15 "MULTI-HORIZON ALPHA PROFILING" investigation_*.log

To compare final Sharpe ratios:

  grep "Sharpe:" investigation_*.log

For manual signal inversions (seasonality, vol_premium), you'll need to:
  1. Edit alphago_architecture.py
  2. Seasonality: Line 1012, multiply mu by -1.0
  3. Vol premium: Lines 1109-1111, flip signs of mu values
  4. Re-run with modified code

Value timing investigation requires portfolio-layer changes (reduce rebal freq).
This is best done by modifying L3 PortfolioConstructor to honor alpha horizons.
    """)

if __name__ == "__main__":
    main()
