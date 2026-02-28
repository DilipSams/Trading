# RL Alpha IC = 0.0000: Forensic Root-Cause Analysis & Improvement Plan

## Executive Summary

The RL alpha (`rl_ppo_mcts`) shows IC of exactly 0.0000 in walk-forward validation. This is **not** because the RL model lacks predictive power — it is because the walk-forward validation code path **never provides the observation vector** that the RL network requires for inference. The RL alpha's `generate()` method returns `mu=0.0` whenever `observation is None`, and the validation code never constructs or passes this observation. This is a **plumbing bug**, not a signal quality issue.

However, even after fixing the plumbing, there are deeper structural reasons why the RL alpha's IC may be weak. This document covers the full investigation: root cause → immediate fix → structural concerns → improvement recommendations.

---

## 1. The Smoking Gun: Observation Never Passed

### 1.1 The Two Code Paths

The RL alpha is called from two different places, and only one provides the observation:

**Path A — `evaluate_with_pipeline()` (alphago_layering.py ~line 315):**
```python
# TradingEnv constructs observation from the feature matrix
obs, _ = env.reset()          # <-- obs = env._obs() = windowed features
...
result = pipeline.step(
    observation=obs,            # ✅ PASSED — RL gets its observation
    closes=closes,
    volumes=volumes,
    bar_idx=bar_idx,
    ...
)
```
This flows through `pipeline.step() → alpha_factory.generate_all(observation=observation) → RLAlphaAdapter.generate(observation=obs)`. The RL alpha gets its observation and produces a real mu.

**Path B — `validate_alphas_walkforward()` (alphago_layering.py ~line 850):**
```python
sigs = pipeline.alpha_factory.generate_all(
    closes=closes_to_t,
    volumes=vols_to_t,
    bar_idx=t,
    # ❌ NO observation parameter
    # ❌ NO features parameter
)
```
The RL alpha's `generate()` method hits this guard:
```python
# alphago_architecture.py, RLAlphaAdapter.generate():
if self.net is None or observation is None:
    return AlphaSignal(
        mu=0.0, sigma=1.0, confidence=0.0,  # <-- ALWAYS zero
        horizon=self.horizon, alpha_name=self.name,
        timestamp=bar_idx
    )
```

**Result: Every bar in walk-forward validation, the RL alpha returns mu=0.0. IC of a constant is 0.0000.**

### 1.2 Why No Observation Exists in WF Validation

The observation vector is a 2,700-float tensor (60 bars × 45 features) produced by `TradingEnv._obs()`. This method:

1. Reads from the pre-computed feature matrix (`self.feat`) — which is built by `compute_indicators()` → `build_feature_matrix()` from OHLCV DataFrames
2. Windows the last 60 bars of this feature matrix
3. Appends 4 position-state features (exposure, unrealized PnL, holding time, direction)
4. Flattens to a 1D array

The walk-forward validation loop has **no TradingEnv instance**. It works directly with raw `prices_train` arrays (OHLCV columns). The feature matrix (`features_train`) exists in `SymbolDataset` but is never accessed or windowed.

### 1.3 The features_train Data IS Available

The `SymbolDataset` dataclass holds:
```python
class SymbolDataset:
    symbol: str
    features_train: np.ndarray   # ← Pre-computed feature matrix exists!
    features_val: np.ndarray
    features_test: np.ndarray
    prices_train: np.ndarray
    ...
```

The features are the output of `compute_indicators()` → `build_feature_matrix()`, containing all 45 technical indicators normalized and ready for the network. The validation code uses `d.prices_train` but ignores `d.features_train`.

---

## 2. The Fix: Construct Observations in WF Validation

### 2.1 Minimal Fix (Recommended for Phase 1)

Add observation construction to the signal generation loop in `validate_alphas_walkforward()`. The key change is to window the pre-computed feature matrix and flatten it, mimicking what `TradingEnv._obs()` does — but WITHOUT the position-state features (which don't exist outside a trading simulation):

```python
# In validate_alphas_walkforward(), BEFORE the signal generation loop:

# Access pre-computed features for observation construction
features_full = d.features_train  # shape: (n_obs, NUM_FEATURES)
window_size = cfg.window_size     # default: 60

# ...

for t in range(max(50, window_size), n_obs):  # Start after warmup
    closes_to_t = closes_full[:t + 1]
    vols_to_t = volumes_full[:t + 1] if volumes_full is not None else None

    # ---- Construct observation for RL alpha ----
    obs = None
    if features_full is not None and t >= window_size:
        # Window the feature matrix exactly as TradingEnv._obs() does
        obs_window = features_full[t - window_size:t].copy()
        # Position-state features (last 4 columns) are zero
        # because WF validation has no live position tracking.
        # This is intentional — we measure the RL's directional
        # belief from market state alone, not conditioned on position.
        obs = obs_window.flatten().astype(np.float32)
        # NaN guard (same as TradingEnv._obs)
        if not np.isfinite(obs).all():
            np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

    try:
        sigs = pipeline.alpha_factory.generate_all(
            closes=closes_to_t,
            volumes=vols_to_t,
            bar_idx=t,
            observation=obs,        # ← NOW PASSED
            features=features_full[:t + 1] if features_full is not None else None,
        )
        all_signals[t] = {
            name: sig.mu for name, sig in sigs.items()
            if sig.is_active
        }
    except Exception:
        all_signals[t] = {}
```

### 2.2 Impact of Zero Position-State Features

The last 4 features in the observation are position-state: `[has_position, position_pnl_pct, bars_in_trade, position_direction]`. In walk-forward validation, there's no simulated position, so these are always zero.

This means:
- The RL network sees a "no position" state at every bar
- The policy distribution will reflect "what to do from flat" — which is actually the correct measurement for directional IC (we want to know: does the network predict direction, not: does it know when to exit)
- During live evaluation, position-state features cause the network to behave differently (e.g., it may prefer to hold existing positions). This creates a **systematic difference** between WF validation IC and live IC, but the WF measurement is the purer directional signal

### 2.3 Verification Test

After applying the fix, run:
```bash
python alphago_layering.py --synthetic --verbose 2
```

Expected changes:
- RL alpha should now show non-zero mu values during WF validation
- IC should be non-zero (positive or negative — either tells us something)
- If IC is still ~0.0 after the fix, the issue is deeper (see Section 3)

---

## 3. Deeper Structural Concerns (Even After Plumbing Fix)

Even with the observation plumbing fixed, the RL alpha faces several structural challenges that may limit its IC:

### 3.1 The RL Was Trained to Maximize Reward, Not IC

The PPO objective is to maximize cumulative discounted reward (a function of PnL, drawdown, turnover, etc.). This is fundamentally different from maximizing IC (rank correlation between mu and forward returns).

**Why this matters:** A policy that maximizes reward might:
- Learn to "stay flat when uncertain" (high probability on FLAT action → mu_raw ≈ 0.0)
- Develop asymmetric preferences (e.g., prefer small longs to avoid borrow costs → systematic positive mu bias)
- Optimize for risk-adjusted sizing, not pure direction (the value head captures expected PnL given current portfolio state, which entangles direction with position management)

**Expected symptom:** The RL produces mu values that are more like "sizing recommendations from flat" than "directional forecasts." The mu_raw = dot(probs, [-1, -0.5, 0, 0.5, 1]) conversion assumes the policy probabilities reflect directional belief, but the policy was actually optimized for total return including costs, drawdown penalties, and Lagrangian constraints.

### 3.2 The mu Conversion Is Lossy

The conversion from policy logits to mu is:
```python
ACTION_RETURN_MAP = [-1.0, -0.5, 0.0, 0.5, 1.0]
mu_raw = dot(probs, ACTION_RETURN_MAP)  # Expected direction
mu = mu_raw * rl_logit_to_mu_scale      # Scale: default 0.10
```

**Problem 1 — Centering at zero:** The FLAT action (index 2) contributes 0.0 to mu_raw. If the policy puts significant probability on FLAT (which is common — staying flat avoids costs), mu_raw is pulled toward zero. A policy of [0.0, 0.0, 0.80, 0.10, 0.10] (80% flat, 10% half-long, 10% full-long) gives mu_raw = 0.15 — a weak positive signal that's often within noise.

**Problem 2 — Symmetric mapping:** The map treats FULL_SHORT and FULL_LONG as equal-magnitude opposite signals. But in equity markets, the distribution of returns is not symmetric (positive drift, heavier left tail). The RL policy may have learned asymmetric preferences that the symmetric mapping distorts.

**Problem 3 — Scale factor `rl_logit_to_mu_scale = 0.10`:** This is a fixed scaling. If the network has strong directional conviction (mu_raw = 0.5), the output mu = 0.05 (5 bps per bar). If the network is mildly directional (mu_raw = 0.1), mu = 0.01 (1 bps per bar). This may be too small relative to sigma, making the signal-to-noise ratio very poor in the ensemble.

### 3.3 Untrained Network in `--eval-only` Mode

If running with `--eval-only`, the network is randomly initialized:
```python
# alphago_layering.py ~line 1112:
if args.eval_only:
    tprint("Building untrained network for pipeline demonstration...", "info")
    net = build_network(cfg)
```

A randomly initialized network's policy is approximately uniform over 5 actions → mu_raw ≈ 0.0 for every bar. **If you only ran `--eval-only` or `--synthetic` without training, the RL alpha is inherently uninformative.** The fix in Section 2 alone won't help — you need actual training first.

### 3.4 Training Data vs. Validation Data Regime Mismatch

The RL is trained on `prices_train` through the TradingEnv. Walk-forward validation also uses `prices_train` but constructs folds within it. If the RL policy was overfit to specific patterns in the training data, the purged OOS folds within the same training set may show degraded performance. This is the standard overfitting concern, but it's amplified for RL because:

- RL sees the entire training trajectory during self-play (sequential exposure to all bars)
- The policy may memorize regime-specific patterns that don't hold OOS
- MCTS distillation could encode trajectory-specific knowledge

### 3.5 Entropy-Based Confidence May Be Miscalibrated

The RL alpha's confidence is computed as:
```python
confidence = 1.0 - (entropy / max_entropy)
```

A well-trained policy often converges to moderate entropy (exploration bonus in PPO prevents collapse to deterministic). If entropy is typically ~1.2 out of max ~1.61 (ln(5)), confidence ≈ 0.25. This is quite low and means the RL signal gets downweighted in the ensemble relative to rule-based alphas that may report confidence of 0.5-0.8. The RL's contribution to mu_hat is systematically suppressed.

---

## 4. Diagnostic Checks Checklist

After applying the plumbing fix (Section 2), run these checks to determine if additional structural fixes are needed:

### Check 1: Verify Observation Is Non-None
```python
# Add temporary debug logging in RLAlphaAdapter.generate():
if observation is not None:
    print(f"  [RL] bar={bar_idx} obs_shape={observation.shape} "
          f"obs_range=[{observation.min():.3f}, {observation.max():.3f}]")
else:
    print(f"  [RL] bar={bar_idx} observation=NONE → returning mu=0.0")
```

**Expected:** After fix, observation should be non-None for all bars ≥ window_size.

### Check 2: Verify mu Distribution
```python
# Collect RL mu values across all WF validation bars
rl_mus = [all_signals[t].get('rl_ppo_mcts', 0.0) for t in sorted(all_signals.keys())]
print(f"  RL mu stats: mean={np.mean(rl_mus):.6f} "
      f"std={np.std(rl_mus):.6f} "
      f"min={np.min(rl_mus):.6f} max={np.max(rl_mus):.6f} "
      f"pct_zero={np.mean(np.abs(rl_mus) < 1e-10)*100:.1f}%")
```

**Expected after fix:**
- `std > 0.001` (non-degenerate signal variation)
- `pct_zero < 50%` (most bars produce a non-zero mu)
- If `std < 0.001` or `pct_zero > 90%`: network is producing near-uniform policies → structural issue (3.1/3.3)

### Check 3: Verify Policy Entropy Distribution
```python
# In RLAlphaAdapter.generate(), log entropy:
entropy = -np.sum(probs * np.log(probs + 1e-10))
max_entropy = np.log(len(probs))  # ln(5) ≈ 1.609
# Collect across bars and report:
#   median entropy, min, max, pct where entropy > 0.95 * max_entropy
```

**Expected for a trained network:**
- Median entropy: 0.8–1.4 (not too low = mode collapse, not too high = no signal)
- If entropy > 0.95 × max for >80% of bars: policy is near-uniform → no directional signal learned

### Check 4: Verify Network Was Actually Trained
```python
# Check if network weights differ from initialization:
net_params = list(unwrap_net(net).parameters())
total_params = sum(p.numel() for p in net_params)
zero_params = sum((p.abs() < 1e-8).sum().item() for p in net_params)
print(f"  Network: {total_params} params, {zero_params} near-zero "
      f"({zero_params/total_params*100:.1f}%)")
# Also check weight magnitude:
weight_norms = [p.norm().item() for p in net_params]
print(f"  Weight norms: min={min(weight_norms):.4f} "
      f"max={max(weight_norms):.4f} mean={np.mean(weight_norms):.4f}")
```

**Expected for trained network:**
- near-zero% should be < 30%
- Weight norms should show meaningful spread (not all near initialization scale)
- If weights look random/small: training may not have converged

### Check 5: IC at Multiple Horizons (Cross-reference with multi_horizon_research.md)
After fix, run the multi-horizon profiling at H=1,5,10,15,20. The RL alpha (native horizon=5) should show:
- Peak IC at H=1-5 bars (if it learned short-term patterns)
- Near-zero IC at H=15-20 (if signal is truly short-horizon)
- If IC=0 at ALL horizons even after fix: the network has no directional alpha

### Check 6: Directional Accuracy (Sign Agreement)
```python
# For each bar where RL produces non-zero mu:
hit_rate = np.mean(np.sign(rl_mus) == np.sign(fwd_returns))
# where fwd_returns are the next-1-bar returns
print(f"  RL directional hit rate: {hit_rate*100:.1f}%")
```

**Expected:**
- > 51%: Some directional alpha exists
- ≈ 50%: No directional alpha (consistent with random)
- < 49%: Anti-correlated — the signal is inverted (could be useful if flipped)

---

## 5. Improvement Recommendations

### 5.1 Immediate Fix (P0 — Correctness)

**Fix the observation plumbing** per Section 2.1. This is the mandatory first step. Without it, all RL IC measurement is meaningless.

Specific code change in `validate_alphas_walkforward()` in `alphago_layering.py`:

```python
# BEFORE the signal generation loop (around line 836):
features_full = d.features_train  # Pre-computed feature matrix
window_size = cfg.window_size

# CHANGE the loop start (line 845):
for t in range(max(50, window_size), n_obs):
    closes_to_t = closes_full[:t + 1]
    vols_to_t = volumes_full[:t + 1] if volumes_full is not None else None

    # ---- Construct observation for RL alpha ----
    obs = None
    if features_full is not None and t >= window_size:
        obs_window = features_full[t - window_size:t].copy()
        # Position-state features (last 4 cols) stay zero — correct for
        # pure directional IC measurement (no position conditioning)
        obs = obs_window.flatten().astype(np.float32)
        if not np.isfinite(obs).all():
            np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

    try:
        sigs = pipeline.alpha_factory.generate_all(
            closes=closes_to_t,
            volumes=vols_to_t,
            bar_idx=t,
            observation=obs,         # ← NOW PASSED
            features=features_full[:t + 1] if features_full is not None else None,
        )
        all_signals[t] = {
            name: sig.mu for name, sig in sigs.items()
            if sig.is_active
        }
    except Exception:
        all_signals[t] = {}
```

### 5.2 mu Conversion Improvements (P1 — Signal Quality)

**5.2a — Use Value Head as mu Supplement:**

The value head estimates V(s) = expected cumulative future reward. The change in V(s) between bars is a direct estimate of expected return:

```python
# In RLAlphaAdapter.generate():
# Current: mu from policy direction only
mu_policy = float(np.dot(probs, self.ACTION_RETURN_MAP))

# Proposed: blend policy-based mu with value-head-based mu
# delta_v = V(s_t) - V(s_{t-1}) proxies expected 1-step return
if hasattr(self, '_prev_value') and self._prev_value is not None:
    # Value change as return signal (normalize by typical V magnitude)
    delta_v = value_est - self._prev_value
    mu_value = float(np.clip(delta_v * 0.05, -0.5, 0.5))
    # Blend: 60% policy direction + 40% value change
    mu_raw = 0.6 * mu_policy + 0.4 * mu_value
else:
    mu_raw = mu_policy

self._prev_value = value_est
mu = mu_raw * self.acfg.rl_logit_to_mu_scale
```

**Economic rationale:** The policy head answers "what should I do?" (action selection). The value head answers "how good is this state?" (state evaluation). For directional forecasting, the value head's state evaluation is often more informative because it directly estimates future reward, while the policy head entangles direction with sizing, costs, and risk management.

**5.2b — Use Softmax Temperature for Sharper Signals:**

Apply temperature scaling to sharpen the policy distribution before computing mu_raw:

```python
# Before computing mu_raw:
temperature = 0.5  # < 1.0 sharpens the distribution
logits_sharp = logits.cpu().numpy()[0] / temperature
probs_sharp = np.exp(logits_sharp - np.max(logits_sharp))
probs_sharp = probs_sharp / probs_sharp.sum()

mu_raw = float(np.dot(probs_sharp, self.ACTION_RETURN_MAP))
```

This amplifies the dominant action's contribution to mu_raw, producing stronger directional signals. The confidence calculation should still use the original (unsharpened) probs.

**5.2c — Asymmetric Action Map:**

Replace the symmetric `[-1, -0.5, 0, 0.5, 1]` map with an asymmetric one that accounts for equity market positive drift:

```python
# For equity markets (positive drift):
ACTION_RETURN_MAP_ASYMMETRIC = np.array([-1.2, -0.5, 0.0, 0.5, 0.8])
# Short signals need higher conviction (against drift)
# Long signals slightly discounted (with drift)
```

This is a minor tuning change that may improve IC for equity data.

### 5.3 Confidence Calibration (P1 — Ensemble Weight)

The entropy-based confidence systematically under-represents RL signal quality. Alternative approaches:

**5.3a — Value-Head Confidence:**
```python
# Higher |V(s)| = network is more certain about state value
# Normalize by running statistics
v_abs = abs(value_est)
if hasattr(self, '_value_running_stats'):
    v_mean, v_std, v_n = self._value_running_stats
    v_z = (v_abs - v_mean) / (v_std + 1e-6)
    confidence_value = float(np.clip(v_z / 3.0, 0.0, 1.0))
    # Update running stats
    v_n += 1
    v_mean += (v_abs - v_mean) / v_n
    v_std = np.sqrt(v_std**2 + (v_abs - v_mean)**2 / v_n)
    self._value_running_stats = (v_mean, v_std, v_n)
else:
    confidence_value = 0.3  # Default before warmup
    self._value_running_stats = (v_abs, 0.1, 1)

# Blend entropy-based and value-based confidence
confidence = 0.5 * confidence_entropy + 0.5 * confidence_value
```

**5.3b — Policy Kurtosis:**
```python
# When one action dominates, kurtosis of probs is high
# This is more sensitive than entropy for near-peaked distributions
kurtosis = np.sum((probs - np.mean(probs))**4) / (np.var(probs)**2 + 1e-10) / len(probs)
confidence_kurtosis = float(np.clip(kurtosis / 10.0, 0.0, 1.0))
```

### 5.4 RL Training Improvements (P2 — Longer-Term)

**5.4a — IC-Aware Reward Shaping:**

Add a term to the RL reward that explicitly rewards directional accuracy:

```python
# In TradingEnv._compute_reward() or as an auxiliary reward:
# direction_reward = sign(action_return_map[action]) * sign(next_bar_return) * scale
# This teaches the network that correct direction matters, not just total PnL
```

**Caution:** This changes the RL objective and may reduce its ability to manage risk. Use as a small auxiliary reward (10-20% of total), not the primary signal.

**5.4b — Separate Directional Head:**

Add a 4th head to the network specifically for directional forecasting:

```python
# In AlphaTradeNet.__init__():
self.direction_head = nn.Sequential(
    nn.Linear(h, h//2), nn.GELU(),
    nn.Linear(h//2, 1), nn.Tanh()  # Output in [-1, 1]
)

# Training target: sign(next_bar_return) or next_bar_return clipped
# This head is trained via auxiliary loss alongside PPO
```

The direction head's output replaces the policy-to-mu conversion entirely, giving a cleaner directional signal. The policy head continues to manage the RL optimization (position sizing, risk management).

**5.4c — Multi-Horizon RL Training:**

Currently, the RL reward is based on per-step returns (effectively 1-bar). Train with multi-step returns:

```python
# Use n-step returns in GAE computation:
# Instead of r_t + gamma * V(s_{t+1})
# Use r_t + gamma*r_{t+1} + ... + gamma^n * V(s_{t+n})
# With n=5 to match the RL alpha's declared horizon
```

This aligns the training objective with the alpha's declared horizon (5 bars), potentially improving IC at that timescale.

### 5.5 RL Scale Factor Tuning (P1 — Quick Win)

The `rl_logit_to_mu_scale = 0.10` may be too small. If the network produces mu_raw in [-0.5, 0.5] (typical range for a trained policy), the output mu is in [-0.05, 0.05], which is small relative to other alphas that can produce mu in [-0.3, 0.3].

**Diagnostic:** After the plumbing fix, compare the distribution of |mu| across alphas:
```
RL:     |mu| typically 0.00-0.05  (scale=0.10, raw in [-0.5, 0.5])
Trend:  |mu| typically 0.00-0.15  (clip ±0.3, raw_signal * 0.05)
MR:     |mu| typically 0.00-0.15  (clip ±0.3, z-score * 0.03)
```

If RL |mu| is systematically smaller, its contribution to the ensemble is proportionally weaker. Consider increasing `rl_logit_to_mu_scale` to 0.15–0.25 so the RL signal has comparable magnitude to other alphas.

**Warning:** Don't change the scale purely for magnitude parity — the scale should reflect the actual expected return per bar that the RL signal implies. If the network's mu_raw = 0.3 implies an expected 3 bps return, and that's genuine, then scale = 0.03 is correct. The scale should be calibrated against realized returns.

---

## 6. Additional Issues Found During Investigation

### 6.1 features_train Available but Unused in WF Validation

The `SymbolDataset.features_train` contains the full pre-computed 45-feature matrix. It should be passed alongside `prices_train` in the validation loop. Currently, only `prices_train` columns [open, high, low, close, volume] are used.

### 6.2 No MCTS During WF Validation

The MCTS-improved policy is not available during WF validation (no MCTS search is run). The RL alpha uses `self._mcts_policy = None`, so it falls back to the raw network policy. This is correct (MCTS is expensive and not needed for signal evaluation), but it means the WF IC measures the raw policy, not the MCTS-improved policy. In live evaluation, MCTS may or may not be run depending on configuration. This creates a **policy gap** between WF and live.

### 6.3 eval-only Mode Produces Random Signals

When running with `--eval-only`, the network is uninitialized. Even with the observation fix, a random network's policy will produce approximately uniform distributions over actions, giving mu_raw ≈ 0.0. This is expected behavior, not a bug — but it should be documented and flagged in output:

```python
# Add to WF validation output when eval-only:
if args.eval_only:
    tprint("WARNING: Network is untrained (--eval-only). "
           "RL alpha IC will be near-zero by construction.", "warn")
```

---

## 7. Priority-Ordered Action Plan

| Priority | Fix | Impact | Effort | Dependency |
|---|---|---|---|---|
| **P0** | Pass observation to generate_all() in WF validation (§2.1) | Unblocks all RL measurement | Small (10 lines) | None |
| **P0** | Add diagnostic checks (§4, Checks 1-3) | Verifies fix worked | Small | P0 fix |
| **P1** | Increase rl_logit_to_mu_scale if mu range is too narrow (§5.5) | Better ensemble contribution | Config change | P0 fix + Check 2 |
| **P1** | Value-head blended mu (§5.2a) | Richer directional signal | Medium (20 lines) | P0 fix |
| **P1** | Confidence calibration with value head (§5.3a) | Fair ensemble weighting | Medium (15 lines) | P0 fix |
| **P2** | Softmax temperature sharpening (§5.2b) | Stronger signals | Small (5 lines) | P0 fix |
| **P2** | Separate direction head (§5.4b) | Cleanest directional signal | Large (new head + training) | Retraining required |
| **P3** | IC-aware reward shaping (§5.4a) | Train for direction, not just PnL | Medium | Retraining required |
| **P3** | Multi-horizon RL training (§5.4c) | Align with declared horizon | Medium | Retraining required |

---

## 8. Expected Outcomes After Fixes

### After P0 (Observation Fix Only)
- RL IC should become non-zero
- Likely range: IC = -0.02 to +0.05 (weak but measurable)
- If the network was trained for multiple iterations with MCTS distillation, directional accuracy should be above random (>51%)
- If IC is still near zero: the network didn't learn useful directional patterns (proceed to P1/P2)

### After P0 + P1 (Observation + mu/Confidence Improvements)
- RL mu range should be comparable to other alphas (|mu| in [0.01, 0.15])
- IC should improve if underlying directional knowledge exists
- Confidence should be calibrated to not systematically under-weight RL in ensemble
- Expected IC range: -0.01 to +0.08

### After Full P0-P2 Implementation
- RL becomes a genuine contributor to the ensemble (not dead weight)
- Multi-horizon profiling should reveal the RL's natural timescale
- The direction head (P2) provides the cleanest measurement of whether the RL learned anything useful about return direction

### If IC Remains Near Zero After All Fixes
- The RL policy genuinely has no directional alpha
- This is a valid finding — it means the PPO+MCTS system learned to manage risk and sizing but not to forecast direction
- Recommendation: keep the RL alpha in the pipeline at low weight (it may still contribute useful volatility estimates via the risk head and regime-conditional behavior), but don't rely on it for directional alpha
- Focus alpha generation efforts on the rule-based alphas and the improvements from alpha_research.md

---

## 9. Summary

The RL alpha IC = 0.0000 is caused by a **plumbing bug** in `validate_alphas_walkforward()` that fails to pass the observation vector to `generate_all()`. The fix is 10 lines of code. However, structural concerns about the RL's directional signal quality mean that even after the fix, IC may be weak. A cascade of improvements (value-head blending, confidence calibration, scale tuning, and eventually a dedicated direction head) can progressively extract more directional alpha from the RL network. The most important next step is: **apply the plumbing fix, run the diagnostic checks, and measure actual IC before investing in deeper structural changes.**