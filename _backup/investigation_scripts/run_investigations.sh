#!/bin/bash
# Quick investigation runner
# Tests the 4 key hypotheses from multi-horizon profiling

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              MULTI-HORIZON ALPHA INVESTIGATIONS                          ║"
echo "║                                                                          ║"
echo "║  Running 4 critical tests to validate signal inversions                 ║"
echo "║  Estimated time: 8-10 hours total (2-2.5 hours each)                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: Signal Inversions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Signal Inversions (Seasonality + Vol Premium)"
echo "  Hypothesis: IC flips from negative to positive"
echo "  Started: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --invert-seasonality \
  --invert-vol-premium \
  > investigation_01_inversions.log 2>&1

echo "✓ Test 1 completed: $(date)"
echo ""

# Test 2: Robust Ensemble (disable bad alphas)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Robust Ensemble (Disable Seasonality + Vol Premium)"
echo "  Hypothesis: Sharpe improves by >20%"
echo "  Started: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --disable-alphas "seasonality,vol_premium" \
  > investigation_02_robust_ensemble.log 2>&1

echo "✓ Test 2 completed: $(date)"
echo ""

# Test 3: Value Only
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Value Alpha Isolation"
echo "  Hypothesis: Value alone achieves Sharpe > 0.5"
echo "  Started: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --disable-alphas "rl_ppo_mcts,trend_follow,mean_reversion,carry,seasonality,vol_premium" \
  > investigation_03_value_only.log 2>&1

echo "✓ Test 3 completed: $(date)"
echo ""

# Test 4: Optimal Configuration (inversions + disabled weak alphas)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Optimal Configuration"
echo "  Hypothesis: Combined inversions + disable RL/carry = best Sharpe"
echo "  Started: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --invert-seasonality \
  --invert-vol-premium \
  --disable-alphas "rl_ppo_mcts,carry" \
  > investigation_04_optimal.log 2>&1

echo "✓ Test 4 completed: $(date)"
echo ""

# Generate summary
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                     ALL TESTS COMPLETED                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Extracting results..."

# Extract multi-horizon IC comparisons
echo "=== MULTI-HORIZON IC COMPARISON ===" > investigation_summary.txt
echo "" >> investigation_summary.txt

echo "BASELINE (from full_multi_horizon_test.log):" >> investigation_summary.txt
grep -A 10 "MULTI-HORIZON ALPHA PROFILING" full_multi_horizon_test.log | head -20 >> investigation_summary.txt
echo "" >> investigation_summary.txt

echo "TEST 1 (Inversions):" >> investigation_summary.txt
grep -A 10 "MULTI-HORIZON ALPHA PROFILING" investigation_01_inversions.log | head -20 >> investigation_summary.txt
echo "" >> investigation_summary.txt

echo "TEST 2 (Robust Ensemble):" >> investigation_summary.txt
grep -A 10 "MULTI-HORIZON ALPHA PROFILING" investigation_02_robust_ensemble.log | head -20 >> investigation_summary.txt
echo "" >> investigation_summary.txt

echo "TEST 3 (Value Only):" >> investigation_summary.txt
grep -A 10 "MULTI-HORIZON ALPHA PROFILING" investigation_03_value_only.log | head -20 >> investigation_summary.txt
echo "" >> investigation_summary.txt

echo "TEST 4 (Optimal):" >> investigation_summary.txt
grep -A 10 "MULTI-HORIZON ALPHA PROFILING" investigation_04_optimal.log | head -20 >> investigation_summary.txt
echo "" >> investigation_summary.txt

# Extract Sharpe comparisons
echo "=== SHARPE RATIO COMPARISON ===" >> investigation_summary.txt
echo "" >> investigation_summary.txt
echo "Baseline:        $(grep 'Sharpe:' full_multi_horizon_test.log | head -1)" >> investigation_summary.txt
echo "Test 1 (Invert): $(grep 'Sharpe:' investigation_01_inversions.log | head -1)" >> investigation_summary.txt
echo "Test 2 (Robust): $(grep 'Sharpe:' investigation_02_robust_ensemble.log | head -1)" >> investigation_summary.txt
echo "Test 3 (Value):  $(grep 'Sharpe:' investigation_03_value_only.log | head -1)" >> investigation_summary.txt
echo "Test 4 (Optim):  $(grep 'Sharpe:' investigation_04_optimal.log | head -1)" >> investigation_summary.txt

echo ""
echo "✓ Summary saved to: investigation_summary.txt"
echo ""
cat investigation_summary.txt
