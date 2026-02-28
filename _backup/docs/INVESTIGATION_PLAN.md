# Multi-Horizon Alpha Investigation Plan

## Why is RL_PPO_MCTS showing IC = 0.000?

**Observed**: `rl_ppo_mcts` shows IC = +0.000 (t=+0.0) across all three horizons (1-bar, 5-bar, 15-bar)

**Explanation**: The RL agent's predictions have **zero correlation** with realized returns. This indicates:

### Root Causes:

1. **RL agent hasn't learned yet (most likely)**
   - The multi-horizon profiling runs on the **validation set** AFTER training
   - The RL agent sees **new symbols** during validation that it wasn't trained on
   - If the RL agent overfits to the training symbols, it will fail to generalize
   - IC = 0.000 means the RL predictions are essentially random noise on out-of-sample data

2. **Training configuration issues**
   - 8 iterations × 80,000 steps = 640,000 steps total
   - With 16 parallel environments, this is 40,000 gradient updates
   - May not be enough for the RL agent to converge
   - RL agents typically need millions of steps to learn robust patterns

3. **Signal extraction timing**
   - The multi-horizon profiling extracts the RL agent's `mu` (return prediction)
   - If the RL agent outputs mu=0 or very small values, IC will be near zero
   - Check: Are the RL agent's mu values actually non-zero during validation?

4. **Observation space mismatch**
   - RL agent is trained on specific technical features
   - Validation symbols may have different statistical properties
   - If the observation distribution shifts, the agent's learned policy may not apply

### How to Verify:

```python
# Add to multi-horizon profiling section (alphago_layering.py ~line 900):
if alpha_name == 'rl_ppo_mcts':
    print(f"  [DEBUG] RL mu distribution: min={np.min(mu_arr):.4f}, "
          f"max={np.max(mu_arr):.4f}, mean={np.mean(mu_arr):.4f}, "
          f"std={np.std(mu_arr):.4f}")
    print(f"  [DEBUG] RL mu non-zero: {np.sum(np.abs(mu_arr) > 0.001)}/{len(mu_arr)}")
```

### Solutions:

1. **Train longer**: Increase iterations to 20-30, steps to 200k-500k each
2. **Check RL outputs**: Verify the RL agent is actually producing non-zero predictions
3. **Simplify RL task**: Start with single-symbol training to verify RL can learn
4. **Use simpler baseline**: The traditional alphas (value, mean_reversion) show IC without any training

### Expected Behavior:

In a well-functioning RL system:
- RL agent should show IC > 0.05 if it learned useful patterns
- IC should be higher on training symbols than validation (but not 0.000 on validation)
- If IC = 0.000 on validation, the RL agent is not contributing to the ensemble

**Bottom line**: RL showing IC = 0.000 is **expected** if the agent hasn't converged yet. The traditional alphas (value, mean_reversion) don't require training and show non-zero IC immediately.

---

## Investigation Tests

### Test 1: Signal Inversions (Seasonality & Vol Premium)

**Hypothesis**: If seasonality IC = -0.014 to -0.046, inverting should produce IC = +0.014 to +0.046

**Command**:
```bash
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --invert-seasonality \
  --invert-vol-premium \
  > investigation_01_inversions.log 2>&1
```

**Expected Outcome**:
- Seasonality IC should flip from -0.046 → +0.046 at 15-bar
- Vol premium IC should flip from -0.062 → +0.062 at 15-bar
- If this works, confirms the signals were correct but had wrong sign

---

### Test 2: Ensemble Robustness (Disable Weak Alphas)

**Hypothesis**: Removing anti-predictive alphas improves overall performance

**Command**:
```bash
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --disable-alphas "seasonality,vol_premium" \
  > investigation_02_robust_ensemble.log 2>&1
```

**Expected Outcome**:
- Meta-learner assigns zero weight to disabled alphas
- Overall Sharpe should improve vs baseline
- Value and mean_reversion should receive higher allocations

---

### Test 3: Value Alpha Isolation

**Hypothesis**: Value alpha alone outperforms noisy ensemble

**Command**:
```bash
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --disable-alphas "rl_ppo_mcts,trend_follow,mean_reversion,carry,seasonality,vol_premium" \
  > investigation_03_value_only.log 2>&1
```

**Expected Outcome**:
- IC(value, 15-bar) = +0.055 should remain stable
- Sharpe driven entirely by value alpha
- Clean signal without ensemble noise

---

### Test 4: Top 3 Alphas (Value + Mean Reversion + Trend)

**Hypothesis**: Keep only alphas with positive IC

**Command**:
```bash
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --disable-alphas "rl_ppo_mcts,carry,seasonality,vol_premium" \
  > investigation_04_top3.log 2>&1
```

**Expected Outcome**:
- IC(value) = +0.055 at 15-bar (positive)
- IC(mean_reversion) = +0.031 at 5-bar (positive)
- IC(trend_follow) = -0.045 at 15-bar (negative, but close to zero at 1-bar and 5-bar)
- Meta-learner should learn to weight value and MR higher

---

### Test 5: Combined Inversions + Disabled Alphas

**Hypothesis**: Best configuration uses inverted signals and disables RL/carry

**Command**:
```bash
python alphago_layering.py \
  --iterations 8 \
  --steps-per-iter 80000 \
  --invert-seasonality \
  --invert-vol-premium \
  --disable-alphas "rl_ppo_mcts,carry" \
  > investigation_05_optimal.log 2>&1
```

**Expected Outcome**:
- Inverted seasonality: IC = +0.046 at 15-bar
- Inverted vol_premium: IC = +0.062 at 15-bar
- All 5 active alphas have positive IC
- Highest Sharpe of all configurations

---

## Value Timing Investigation (Separate Analysis)

**Hypothesis**: Value alpha works best with 15-bar holding period

**Evidence from multi-horizon profiling**:
- IC(value, 1-bar) = +0.018 (t=2.4)
- IC(value, 5-bar) = +0.033 (t=2.0)
- IC(value, 15-bar) = **+0.055** (t=1.9) ← **3x better than 1-bar!**

**Conclusion**: Value alpha's predictive power compounds over 2-3 weeks

**Implementation Note**:
Current system rebalances daily (1-bar frequency) even though value horizon = 63 bars.
To test actual 15-bar holding:

1. **Option A**: Modify L3 PortfolioConstructor to track position age and only rebalance after horizon elapsed
2. **Option B**: Use the IC at 15-bar as proxy (already done - shows 3x improvement)
3. **Option C**: Create separate backtester that holds value positions for exactly 15 bars

**Recommendation**: The IC analysis already proves value works better at 15-bar. Don't need to rerun full training just to confirm what multi-horizon profiling already showed.

---

## Execution Plan

```bash
# Run all investigations (takes ~10-12 hours total)
cd Trading

# Test 1: Inversions
python alphago_layering.py --iterations 8 --steps-per-iter 80000 \
  --invert-seasonality --invert-vol-premium \
  > investigation_01_inversions.log 2>&1 &

# Test 2: Robust ensemble (no bad alphas)
python alphago_layering.py --iterations 8 --steps-per-iter 80000 \
  --disable-alphas "seasonality,vol_premium" \
  > investigation_02_robust_ensemble.log 2>&1 &

# Test 3: Value only
python alphago_layering.py --iterations 8 --steps-per-iter 80000 \
  --disable-alphas "rl_ppo_mcts,trend_follow,mean_reversion,carry,seasonality,vol_premium" \
  > investigation_03_value_only.log 2>&1 &

# Wait for all to complete, then analyze
grep -A 15 "MULTI-HORIZON ALPHA PROFILING" investigation_*.log > summary_IC.txt
grep "Sharpe:" investigation_*.log > summary_sharpe.txt
```

---

## Expected Timeline

- Each test: ~2-2.5 hours (8 iterations × 80k steps × 50 symbols)
- Total if sequential: ~12-15 hours
- Total if parallel (3 at a time): ~4-5 hours

---

## Success Metrics

1. **Inversions work**: IC flips sign as predicted
2. **Ensemble improves**: Removing bad alphas increases Sharpe by >20%
3. **Value isolation**: Single alpha achieves Sharpe > 0.5
4. **RL diagnosis**: Confirm IC = 0.000 is due to insufficient training, not a bug
