# Multi-Horizon Alpha Profiling: Research & Implementation Plan

## Executive Summary

The system currently normalizes all alphas to a 1-bar common horizon and evaluates predictive power at that single timescale. This is fundamentally unfair to long-horizon alphas (Value at 63 bars, Trend at 21 bars) and wastes information from short-horizon alphas that may have predictive power at multiple timescales. The CLAUDE.md NEXT TASK directive specifies profiling at 1-bar, 5-bar, and 15-bar horizons. This document extends that to a full spectrum: **1, 5, 10, 15, 20 bars**, provides the theoretical foundation, maps each current alpha to its expected optimal horizon, and delivers a concrete implementation plan.

---

## 1. Why Multi-Horizon Matters: The SNR Problem

### 1.1 Signal-to-Noise Ratio and Horizon

Daily equity returns have an annualized signal-to-noise ratio (SNR) of approximately 1:16 (Sharpe ~0.5 for strong alphas on a broad market). At 1-bar resolution, the noise dominates overwhelmingly:

| Horizon (bars) | Noise Reduction Factor | Effective SNR Improvement | IC Measurement Quality |
|---|---|---|---|
| 1 | 1.0x (baseline) | — | Very noisy; needs ~1000+ observations for significance |
| 5 | √5 ≈ 2.24x | 2.24x | Meaningfully better; ~500 obs often sufficient |
| 10 | √10 ≈ 3.16x | 3.16x | Good; captures weekly-scale predictability |
| 15 | √15 ≈ 3.87x | 3.87x | Strong; reduces microstructure noise |
| 20 | √20 ≈ 4.47x | 4.47x | Near-monthly; captures medium-term trends |

The √N noise reduction comes from the central limit theorem applied to non-overlapping return windows. This is why the CLAUDE.md notes that "1-bar scoring is unfair to long-horizon alphas" and "averaging over 5 bars improves measurement by √5 ≈ 2.2×."

### 1.2 The Horizon Mismatch Problem

Each alpha has a *native* prediction horizon — the timescale at which it expects to be profitable. The current `_normalize_horizon()` converts all signals to per-bar units via linear scaling (mu) and √T scaling (sigma). While mathematically correct for unit conversion, this does NOT mean each alpha is best evaluated at 1-bar. Consider:

- **ValueAlpha** (native horizon=63): Predicts "cheap assets appreciate over months." Evaluating at 1-bar is like testing a 6-month weather forecast by checking if it rained today. The signal is real but the noise at 1-bar overwhelms it.
- **MeanReversionAlpha** (native horizon=5): Designed for weekly timescales. Should show peak IC at 3-7 bars, decay at longer horizons.
- **TrendAlpha** (native horizon=21): Momentum effects are strongest at 2-12 month horizons (Moskowitz, Ooi, Pedersen 2012). At 1-bar, it picks up mostly noise; at 20 bars, it should shine.

### 1.3 Academic Evidence for Multi-Horizon Effects

The literature is clear that alpha signals have horizon-dependent power:

- **Momentum** (Jegadeesh & Titman 1993, Asness et al. 2013): Strongest at 3-12 months. Very weak at 1-day. Reverses at 3-5 years. Testing only at 1-bar severely under-counts its true predictive power.
- **Mean Reversion** (Lo & MacKinlay 1990, Poterba & Summers 1988): Short-term reversal (1-5 days) is a distinct effect from long-term reversal (3-5 years). The Bollinger band z-score approach in MeanReversionAlpha should peak at 3-7 bars.
- **Value** (Fama & French 1992, Asness et al. 2013): Very long-horizon effect (months to years). At 1-bar, it is statistically indistinguishable from noise. At 20 bars, it may emerge. At 63+ bars (its native horizon), it should be strongest — but 63-bar non-overlapping windows require enormous data.
- **Volatility Premium** (Bakshi & Kapadia 2003): The variance risk premium is typically harvested at monthly horizons. Evaluation at 10-20 bars is most appropriate.
- **Carry** (Koijen et al. 2018): Carry returns accrue slowly. 20-bar evaluation is reasonable.
- **RL Alpha**: No a priori horizon — data-driven. Profiling across all horizons reveals what timescale the network actually learned.

---

## 2. Alpha-by-Alpha Horizon Predictions

Based on economic theory, the alpha_research.md findings, and the signal_attenuation_analysis.md results, here are the expected IC profiles:

### 2.1 Current 7 Alphas

| Alpha | Native Horizon | Expected Peak IC Horizon | Expected Weak Horizons | Rationale |
|---|---|---|---|---|
| **RL (PPO+MCTS)** | 5 bars | 1-5 bars | 15-20 bars | Trained on short-horizon rewards; learned patterns are likely short-lived |
| **TrendAlpha** | 21 bars | 10-20 bars | 1 bar | EMA crossover is a medium-horizon signal; 1-bar noise overwhelms |
| **MeanReversionAlpha** | 5 bars | 1-5 bars | 15-20 bars | Bollinger z-score reverts within days, not weeks |
| **ValueAlpha** | 63 bars | 15-20 bars (possibly still weak) | 1-5 bars | Fundamental undervaluation corrects slowly; may need 63+ bars |
| **CarryAlpha** | 21 bars | 10-20 bars | 1-5 bars | Carry accrues over time; short-horizon noise dominates |
| **SeasonalityAlpha** | 5 bars | 1-5 bars | 10-20 bars | Day-of-week effects are by definition 1-bar phenomena |
| **VolPremiumAlpha** | 21 bars | 10-20 bars | 1 bar | Vol premium accrues over weeks; daily measurement is noise |

### 2.2 Proposed New Alphas from alpha_research.md

For the Tier 1-2 alphas recommended in the research document:

| Alpha | Native Horizon | Expected Peak IC Horizon | Notes |
|---|---|---|---|
| **Amihud Illiquidity** | 21 bars | 10-20 bars | Illiquidity premium is medium-term (Amihud 2002) |
| **Short-Term Reversal** | 5 bars | 1-5 bars | Pure microstructure effect; decays rapidly beyond 5 bars |
| **Idiosyncratic Momentum** | 21 bars | 15-20 bars | Underreaction to firm-specific info corrects over weeks |
| **CalendarAlpha** | 5 bars | 1-5 bars | Intraweek anomalies; no reason for multi-week persistence |

### 2.3 What This Tells Us About Horizon Bucket Design

The alphas cluster into three natural groups:

1. **Short-horizon group (1-5 bars)**: RL, MeanReversion, Seasonality, ShortTermReversal, Calendar
2. **Medium-horizon group (10-15 bars)**: Trend, Carry, VolPremium, Amihud
3. **Long-horizon group (15-20+ bars)**: Value, IdiosyncraticMomentum, Trend (tail)

This suggests the current `horizon_bars: (1, 5, 21)` configuration is approximately right but could benefit from finer granularity. The proposed profiling horizons **(1, 5, 10, 15, 20)** map well to these clusters.

---

## 3. Methodology: Multi-Horizon IC Measurement

### 3.1 Non-Overlapping Forward Returns

For each horizon H ∈ {1, 5, 10, 15, 20}, compute the forward return:

```
r_fwd(t, H) = log(close[t+H] / close[t])
```

Use **non-overlapping** windows to avoid autocorrelation inflation of IC estimates. For H=5, evaluate at bars 0, 5, 10, 15, ... For H=20, evaluate at bars 0, 20, 40, 60, ...

This is critical: overlapping windows inflate IC by a factor proportional to √H due to shared information between adjacent measurements.

### 3.2 IC Computation

For each alpha i and horizon H, compute the Information Coefficient as the Spearman rank correlation between the alpha's mu signal and the subsequent H-bar return:

```
IC_i(H) = SpearmanCorr(mu_i[t], r_fwd[t, H])
```

Over the OOS test windows from purged walk-forward CV.

### 3.3 Statistical Significance with Multiple Testing

With 7 alphas × 5 horizons = 35 hypothesis tests, Holm-Bonferroni correction is mandatory. The CLAUDE.md specifies 21 tests (7 × 3 horizons); expanding to 5 horizons increases this to 35, requiring stricter individual p-value thresholds.

For the most significant alpha-horizon pair, the adjusted threshold is:
```
p_adjusted_1 = 0.05 / 35 ≈ 0.00143
```

This is strict but appropriate — it prevents declaring a spurious horizon effect as real.

### 3.4 Purge Gap Adjustment

The purge gap must be at least the maximum horizon being tested. With H_max = 20:

```
purge_gap = max(20, current_purge_gap)
```

The current system uses `purge_gap = max(21, acfg.meta_learner_retrain_every // 10)` which is already ≥ 21, so this is satisfied. The embargo remains at 5 bars.

### 3.5 Sample Size Requirements

For IC to be reliably estimated, we need:
- H=1: ~500+ non-overlapping observations (standard)
- H=5: ~100+ non-overlapping observations (500/5)
- H=10: ~50+ non-overlapping observations (minimum for reliability)
- H=15: ~33+ observations (marginal — flag in output)
- H=20: ~25+ observations (quite thin — flag in output)

For a typical 500-bar OOS window:
- H=1: 500 observations ✓
- H=5: 100 observations ✓
- H=10: 50 observations ✓ (borderline)
- H=15: 33 observations ⚠ (flag)
- H=20: 25 observations ⚠ (flag)

The diagnostics should report both the IC value and the number of non-overlapping observations, with warnings when n < 50.

---

## 4. Expected Outcomes and Decision Framework

### 4.1 Possible Findings

**Scenario A: Short-horizon alphas dominate (peak at 1-5 bars)**
- Implies: The system is correctly configured. Keep current horizon_blend_weights favoring short horizons (0.5, 0.3, 0.2).
- Action: Consider dropping or de-emphasizing Value/Carry which provide no edge.

**Scenario B: Medium-horizon alphas show strong IC (10-15 bars)**
- Implies: We're leaving money on the table by normalizing everything to 1-bar. The meta-learner sees attenuated signals.
- Action: Shift horizon_blend_weights toward medium (e.g., 0.3, 0.4, 0.3) and consider a 2-speed rebalancing approach (fast signals trade daily, slow signals trade weekly).

**Scenario C: Value/Carry show IC only at 15-20 bars**
- Implies: These alphas are real but completely invisible at 1-bar. The REJECT verdicts from walk-forward validation (signal_attenuation_analysis.md shows Value and Carry were REJECTED) may be false negatives due to horizon mismatch.
- Action: Create separate evaluation tracks. Don't score long-horizon alphas at 1-bar. Feed their signals with appropriate weighting into the horizon blender.

**Scenario D: Some alphas show multi-horizon power**
- Implies: Trend might be predictive at both 5 and 20 bars. This alpha should be cloned into multiple horizon variants.
- Action: Register horizon-specific variants (e.g., TrendAlpha_5bar, TrendAlpha_20bar) in L1, each with appropriate configuration.

### 4.2 How Results Feed Back Into the System

The profiling results are **diagnostic only** (per CLAUDE.md: "don't change trading logic until profiles are reviewed"). However, the following changes should be queued pending review:

1. **HorizonBlender weights**: Shift from hardcoded (0.5, 0.3, 0.2) to IC-proportional weights derived from profiling results.
2. **Alpha demotion policy**: Don't reject alphas based on 1-bar IC alone. Test at the alpha's native horizon ± the two adjacent profiling horizons.
3. **Rebalancing frequency**: If medium/long-horizon alphas show strong IC, consider a tiered rebalancing approach where slow alphas update positions weekly (every 5 bars) rather than daily.
4. **Ensemble feature expansion**: Add horizon-specific IC as features for the meta-learner, enabling it to learn which horizons are informative in which regimes.

---

## 5. Implementation: Code Changes to `alphago_layering.py`

### 5.1 Overview

Add a `profile_alphas_multi_horizon()` function that runs inside `validate_alphas_walkforward()` using the same purged walk-forward CV infrastructure but evaluates each alpha at multiple horizons.

### 5.2 Core Implementation

```python
# ============================================================================
# MULTI-HORIZON ALPHA PROFILING (Diagnostic Module)
# ============================================================================
# Location: Add to alphago_layering.py, called from validate_alphas_walkforward()
# Purpose: Grade each alpha at 1, 5, 10, 15, 20-bar horizons
# Output: IC matrix (n_alphas × n_horizons) with significance flags
# Constraint: Diagnostic only — does not change trading logic
# ============================================================================

PROFILE_HORIZONS = (1, 5, 10, 15, 20)
MIN_NON_OVERLAPPING_OBS = 25    # Absolute minimum for IC estimation
WARN_NON_OVERLAPPING_OBS = 50   # Below this, flag as low-confidence

def profile_alphas_multi_horizon(
    alpha_signals_by_bar: Dict[int, Dict[str, 'AlphaSignal']],
    closes: np.ndarray,
    test_start: int,
    test_end: int,
    alpha_names: List[str],
    horizons: Tuple[int, ...] = PROFILE_HORIZONS,
    verbose: int = 1,
) -> Dict[str, Dict[int, Dict]]:
    """
    Profile each alpha's IC at multiple forward-return horizons.

    Uses NON-OVERLAPPING windows for unbiased IC estimation.
    Returns IC, t-stat, n_obs, and significance flag per alpha per horizon.

    Args:
        alpha_signals_by_bar: {bar_idx: {alpha_name: AlphaSignal}}
            Pre-computed signals from the walk-forward test window.
        closes: Full close price array for the dataset.
        test_start: Start index of the OOS test window.
        test_end: End index of the OOS test window (exclusive).
        alpha_names: List of alpha names to profile.
        horizons: Tuple of horizon lengths (bars) to evaluate.
        verbose: Print detail level.

    Returns:
        {alpha_name: {horizon: {
            'ic': float,           # Spearman rank IC
            'ic_tstat': float,     # t-stat of IC
            'n_obs': int,          # Number of non-overlapping observations
            'p_value': float,      # Raw p-value (before multiple testing)
            'significant': bool,   # After Holm-Bonferroni correction
            'low_confidence': bool # True if n_obs < WARN threshold
        }}}
    """
    from scipy import stats as scipy_stats  # Only used here; graceful degrade

    results = {name: {} for name in alpha_names}

    for H in horizons:
        # Build non-overlapping evaluation points within [test_start, test_end)
        # Signal at bar t predicts return from t to t+H
        eval_bars = list(range(test_start, test_end - H, H))
        n_obs = len(eval_bars)

        if n_obs < MIN_NON_OVERLAPPING_OBS:
            # Not enough data for this horizon — skip with flag
            for name in alpha_names:
                results[name][H] = {
                    'ic': float('nan'),
                    'ic_tstat': 0.0,
                    'n_obs': n_obs,
                    'p_value': 1.0,
                    'significant': False,
                    'low_confidence': True,
                    'skip_reason': f'n_obs={n_obs} < min={MIN_NON_OVERLAPPING_OBS}',
                }
            continue

        # Compute forward H-bar log returns at each eval point
        fwd_returns = np.array([
            np.log(closes[t + H] / (closes[t] + 1e-12))
            for t in eval_bars
            if t + H < len(closes) and closes[t] > 0
        ])

        # Filter eval_bars to match (in case some were clipped)
        valid_bars = [
            t for t in eval_bars
            if t + H < len(closes) and closes[t] > 0
        ]
        n_valid = len(valid_bars)
        if n_valid < MIN_NON_OVERLAPPING_OBS:
            for name in alpha_names:
                results[name][H] = {
                    'ic': float('nan'), 'ic_tstat': 0.0,
                    'n_obs': n_valid, 'p_value': 1.0,
                    'significant': False, 'low_confidence': True,
                    'skip_reason': f'n_valid={n_valid} < min',
                }
            continue

        # For each alpha, collect mu predictions at eval points
        for name in alpha_names:
            mus = []
            rets = []
            for i, t in enumerate(valid_bars):
                if t in alpha_signals_by_bar:
                    sig = alpha_signals_by_bar[t].get(name)
                    if sig is not None and sig.is_active and not np.isnan(sig.mu):
                        mus.append(sig.mu)
                        rets.append(fwd_returns[i])

            n_alpha = len(mus)
            if n_alpha < MIN_NON_OVERLAPPING_OBS:
                results[name][H] = {
                    'ic': float('nan'), 'ic_tstat': 0.0,
                    'n_obs': n_alpha, 'p_value': 1.0,
                    'significant': False, 'low_confidence': True,
                    'skip_reason': f'active signals={n_alpha} < min',
                }
                continue

            # Spearman rank IC
            mu_arr = np.array(mus)
            ret_arr = np.array(rets)

            try:
                ic, p_raw = scipy_stats.spearmanr(mu_arr, ret_arr)
            except Exception:
                ic, p_raw = 0.0, 1.0

            if np.isnan(ic):
                ic, p_raw = 0.0, 1.0

            # t-stat: IC * sqrt(n - 2) / sqrt(1 - IC^2)
            if abs(ic) < 1.0 - 1e-10 and n_alpha > 2:
                ic_tstat = ic * np.sqrt(n_alpha - 2) / np.sqrt(1 - ic**2 + 1e-12)
            else:
                ic_tstat = 0.0

            results[name][H] = {
                'ic': float(ic),
                'ic_tstat': float(ic_tstat),
                'n_obs': n_alpha,
                'p_value': float(p_raw),
                'significant': False,  # Set after multiple testing
                'low_confidence': n_alpha < WARN_NON_OVERLAPPING_OBS,
            }

    # ---- Holm-Bonferroni multiple testing correction ----
    # Collect all (alpha, horizon, p_value) tuples
    all_tests = []
    for name in alpha_names:
        for H in horizons:
            entry = results[name].get(H, {})
            if not np.isnan(entry.get('ic', float('nan'))):
                all_tests.append((name, H, entry['p_value']))

    n_tests = len(all_tests)
    if n_tests > 0:
        # Sort by p-value ascending
        all_tests.sort(key=lambda x: x[2])
        for rank, (name, H, p) in enumerate(all_tests):
            # Holm-Bonferroni: compare p to alpha / (n_tests - rank)
            adjusted_threshold = 0.05 / (n_tests - rank)
            results[name][H]['significant'] = (p <= adjusted_threshold)
            results[name][H]['holm_bonf_threshold'] = adjusted_threshold
            results[name][H]['holm_bonf_rank'] = rank + 1

    # ---- Print results ----
    if verbose >= 1:
        _print_horizon_profile(results, alpha_names, horizons, n_tests)

    return results


def _print_horizon_profile(results, alpha_names, horizons, n_tests):
    """Pretty-print the multi-horizon IC profile matrix."""

    print(f"\n  {'='*78}")
    print(f"  MULTI-HORIZON ALPHA PROFILE  "
          f"({n_tests} tests, Holm-Bonferroni corrected)")
    print(f"  {'='*78}")
    print()

    # Header
    header = f"  {'Alpha':<22s}"
    for H in horizons:
        header += f" {'H=' + str(H):>10s}"
    header += f"  {'Best H':>7s} {'Peak IC':>8s}"
    print(header)

    divider = f"  {'-'*22}"
    for _ in horizons:
        divider += f" {'-'*10}"
    divider += f"  {'-'*7} {'-'*8}"
    print(divider)

    for name in alpha_names:
        row = f"  {name:<22s}"
        best_h = None
        best_ic = -999

        for H in horizons:
            entry = results[name].get(H, {})
            ic = entry.get('ic', float('nan'))

            if np.isnan(ic):
                cell = '   ---   '
            else:
                # Color coding (conceptual — terminal colors)
                sig = '**' if entry.get('significant') else '  '
                warn = '?' if entry.get('low_confidence') else ' '
                cell = f"{sig}{ic:+.3f}{warn}{sig}"

                if ic > best_ic:
                    best_ic = ic
                    best_h = H

            row += f" {cell:>10s}"

        if best_h is not None and best_ic > -999:
            row += f"  {best_h:>5d}b {best_ic:>+.3f}"
        else:
            row += f"  {'N/A':>7s} {'N/A':>8s}"

        print(row)

    print()
    print(f"  ** = significant after Holm-Bonferroni  "
          f"? = low confidence (n < {WARN_NON_OVERLAPPING_OBS})")
    print()

    # Summary: which alphas are horizon-mismatched?
    print(f"  HORIZON MISMATCH ANALYSIS:")
    for name in alpha_names:
        h1_ic = results[name].get(1, {}).get('ic', float('nan'))
        best_h = None
        best_ic = -999
        for H in horizons:
            ic = results[name].get(H, {}).get('ic', float('nan'))
            if not np.isnan(ic) and ic > best_ic:
                best_ic = ic
                best_h = H

        if best_h is not None and best_h > 1 and not np.isnan(h1_ic):
            improvement = best_ic / (abs(h1_ic) + 1e-10)
            if improvement > 1.5:
                print(f"    {name}: Best at H={best_h} "
                      f"(IC={best_ic:+.3f} vs H=1 IC={h1_ic:+.3f}, "
                      f"{improvement:.1f}x improvement)")
    print()
```

### 5.3 Integration Point in `validate_alphas_walkforward()`

```python
# After the existing per-alpha IC computation loop, add:

# ---- Multi-Horizon Profiling (diagnostic) ----
if verbose >= 1:
    tprint("Running multi-horizon alpha profiling...", "info")

# Collect alpha signals by bar from the test folds
# (signals were already generated bar-by-bar in the fold loop above)
horizon_profiles = profile_alphas_multi_horizon(
    alpha_signals_by_bar=alpha_signals_cache,  # {bar_idx: {name: AlphaSignal}}
    closes=closes_full,
    test_start=fold.test_start,
    test_end=fold.test_end,
    alpha_names=alpha_names,
    horizons=PROFILE_HORIZONS,
    verbose=verbose,
)

# Store in the validation report for downstream use
reports['_multi_horizon_profiles'] = horizon_profiles
```

### 5.4 Signal Caching Requirement

The existing `validate_alphas_walkforward()` generates signals bar-by-bar but doesn't cache them in a bar-indexed dict. A minor change is needed:

```python
# Inside the fold loop, after generating signals:
# ADD: Cache signals for multi-horizon profiling
if bar_idx not in alpha_signals_cache:
    alpha_signals_cache[bar_idx] = {}
alpha_signals_cache[bar_idx] = dict(signals)  # {name: AlphaSignal}
```

---

## 6. Horizon-Aware Ensemble Improvements (Post-Profiling)

These changes should NOT be made until profiling results are reviewed. They are documented here as the logical follow-on.

### 6.1 IC-Proportional Horizon Blend Weights

Replace the hardcoded `horizon_blend_weights: (0.5, 0.3, 0.2)` with empirically derived weights:

```python
def compute_horizon_blend_weights(
    profiles: Dict[str, Dict[int, Dict]],
    alpha_names: List[str],
    horizon_bars: Tuple[int, ...] = (1, 5, 21),
) -> np.ndarray:
    """
    Derive horizon blend weights from profiling IC results.

    For each horizon bucket, aggregate the IC of alphas assigned to that bucket.
    Weight proportional to aggregate IC, with shrinkage toward equal weight.
    """
    bucket_ics = {h: [] for h in horizon_bars}

    for name in alpha_names:
        for H, entry in profiles.get(name, {}).items():
            ic = entry.get('ic', 0.0)
            if np.isnan(ic) or not entry.get('significant', False):
                continue
            # Assign to nearest horizon bucket
            best_bucket = min(horizon_bars, key=lambda b: abs(b - H))
            bucket_ics[best_bucket].append(max(ic, 0.0))

    raw_weights = np.array([
        np.mean(bucket_ics[h]) if bucket_ics[h] else 0.0
        for h in horizon_bars
    ])

    # Shrinkage: 70% IC-proportional + 30% equal weight
    equal_w = np.ones(len(horizon_bars)) / len(horizon_bars)
    if raw_weights.sum() > 0:
        ic_w = raw_weights / (raw_weights.sum() + 1e-10)
        blended = 0.7 * ic_w + 0.3 * equal_w
    else:
        blended = equal_w

    return blended / (blended.sum() + 1e-10)
```

### 6.2 Horizon-Specific Alpha Demotion

Instead of rejecting alphas based solely on 1-bar IC, evaluate at the alpha's declared native horizon ± 1 bucket:

```python
def evaluate_alpha_at_native_horizon(
    profiles: Dict[int, Dict],
    native_horizon: int,
    horizons: Tuple[int, ...] = PROFILE_HORIZONS,
) -> str:
    """
    Evaluate alpha at its native horizon (±1 adjacent bucket).

    Returns 'PASS', 'MARGINAL', or 'REJECT' based on:
    - IC at native horizon (or nearest profiling horizon)
    - IC at the two adjacent horizons
    - Best of the three
    """
    # Find the 3 nearest profiling horizons to native_horizon
    sorted_h = sorted(horizons, key=lambda h: abs(h - native_horizon))
    test_horizons = sorted_h[:3]

    best_ic = -999
    best_sig = False
    for H in test_horizons:
        entry = profiles.get(H, {})
        ic = entry.get('ic', float('nan'))
        sig = entry.get('significant', False)
        if not np.isnan(ic) and ic > best_ic:
            best_ic = ic
            best_sig = sig

    if best_sig and best_ic > 0.02:
        return 'PASS'
    elif best_ic > 0.01:
        return 'MARGINAL'
    else:
        return 'REJECT'
```

### 6.3 Expanded HorizonBlender Buckets

If profiling reveals strong IC at intermediate horizons (10, 15), expand the horizon buckets:

```python
# Current config:
horizon_bars: Tuple[int, ...] = (1, 5, 21)
horizon_blend_weights: Tuple[float, ...] = (0.5, 0.3, 0.2)

# Proposed expansion (pending profiling results):
horizon_bars: Tuple[int, ...] = (1, 5, 10, 21)
horizon_blend_weights: Tuple[float, ...] = (0.35, 0.25, 0.25, 0.15)
```

This would require updating the HorizonBlender and ArchitectureConfig. The weights shown are illustrative — actual values come from profiling.

### 6.4 Tiered Rebalancing (Advanced)

If medium/long-horizon alphas show strong IC, a two-speed approach avoids unnecessary turnover:

- **Fast track** (rebalance every bar): RL, MR, Seasonality signals
- **Slow track** (rebalance every 5 bars): Trend, Carry, VolPremium, Value signals

This reduces turnover cost drag from slow alphas (which change slowly anyway) and is consistent with the finding from signal_attenuation_analysis.md that compound multiplicative constraints crush small signals — slow rebalancing only acts when the accumulated signal is large enough.

Implementation sketch:

```python
class TieredRebalancer:
    """
    Separates alpha signals into fast and slow rebalancing tracks.

    Fast track: signals with native horizon <= 5 bars, rebalance every bar.
    Slow track: signals with native horizon > 5 bars, rebalance every N bars.
    Final position = fast_weight * fast_position + slow_weight * slow_position.
    """
    def __init__(self, fast_rebal_freq: int = 1, slow_rebal_freq: int = 5,
                 fast_weight: float = 0.5, slow_weight: float = 0.5):
        self.fast_freq = fast_rebal_freq
        self.slow_freq = slow_rebal_freq
        self.fast_weight = fast_weight
        self.slow_weight = slow_weight
        self._slow_last_rebal = -999
        self._slow_target = 0.0

    def compute_target(self, fast_mu: float, slow_mu: float,
                       sigma: float, bar_idx: int) -> float:
        """
        Combine fast and slow signals into a single target exposure.
        """
        fast_target = fast_mu / (sigma**2 + 1e-6)

        if bar_idx - self._slow_last_rebal >= self.slow_freq:
            self._slow_target = slow_mu / (sigma**2 + 1e-6)
            self._slow_last_rebal = bar_idx

        return (self.fast_weight * fast_target +
                self.slow_weight * self._slow_target)
```

---

## 7. Interaction with Existing Architecture

### 7.1 Impact on L2 (Ensemble Layer)

The multi-horizon profiling directly informs three L2 components:

1. **MetaLearner**: Currently trains on 1-bar realized returns. Post-profiling, consider training on the optimal-horizon return for each alpha. This means the training target for the Ridge regression could be multi-horizon blended returns rather than pure 1-bar returns.

2. **HorizonBlender**: Current weights are static (0.5, 0.3, 0.2). Post-profiling, these become empirically derived and potentially regime-conditional (in trending regimes, upweight the 10-20 bar bucket; in mean-reverting regimes, upweight the 1-5 bar bucket).

3. **AlphaCorrelationManager**: Cross-horizon correlations are typically lower than same-horizon correlations. This means that running Trend at both H=5 and H=20 adds genuine diversification. The N_eff diagnostic should capture this.

### 7.2 Impact on L3 (Portfolio Construction)

The signal attenuation analysis identified that mu_hat ≈ 0.006 after ensemble averaging — too small to survive L3 constraints. Multi-horizon profiling addresses this indirectly:

- Removing horizon-mismatched evaluations stops dragging ensemble mu toward zero.
- Proper horizon weighting amplifies the strongest signals.
- The net effect: mu_hat should increase, potentially enough to survive L3+L4.

### 7.3 Impact on L4 (Execution/Kill Switches)

No direct impact on L4. However, if tiered rebalancing is adopted, the no-trade filter and turnover limits need horizon-awareness: slow-track trades should be evaluated against their own (wider) no-trade zone, not the fast-track zone.

### 7.4 Impact on Validation Engine

The `validate_alphas_walkforward()` function currently tests at 1-bar only. The multi-horizon profiling runs as a diagnostic alongside this. The verdict logic ("PASS/MARGINAL/REJECT") should eventually incorporate the multi-horizon results, but per CLAUDE.md this change waits until profiles are reviewed.

---

## 8. Anti-Overfitting Considerations

### 8.1 Multiple Testing Burden

Adding horizons increases the testing burden from 7 tests to 35. The Holm-Bonferroni correction handles this, but the practical effect is that marginal signals become harder to detect. This is by design — we'd rather miss a weak real signal than declare a spurious one significant.

### 8.2 Horizon Snooping Risk

If we select the "best" horizon for each alpha and then use that horizon for live trading, we've introduced selection bias. Mitigation:

- Report the full IC × horizon matrix, not just the best cell.
- Require that the optimal horizon matches the alpha's declared economic hypothesis. If TrendAlpha peaks at H=1 instead of H=10-20, that's suspicious and shouldn't be trusted.
- Cross-validate: the horizon profile from fold 1 should be similar to fold 2. If the optimal horizon shifts dramatically between folds, the finding is unstable.

### 8.3 Feature Count Discipline

The profiling itself doesn't add features to the meta-learner. But if we eventually add horizon-specific IC as meta-learner features, we need to check:

```
Current features: 7 alphas × 3 (mu, sigma, conf) + 4 regime = 25
With horizon IC: + 7 alphas × 5 horizons = +35 → total 60
For 126 training samples: sqrt(126) ≈ 11 features
```

This would violate the feature count discipline massively. Solution: use the horizon profile to select a SINGLE best horizon per alpha, and pass only that information (as a weight modifier or confidence scalar) rather than the full matrix.

---

## 9. Recommended Config Changes (ArchitectureConfig)

```python
# Add to ArchitectureConfig dataclass:

# Multi-Horizon Profiling
profile_horizons: Tuple[int, ...] = (1, 5, 10, 15, 20)
profile_min_obs: int = 25           # Minimum non-overlapping obs per horizon
profile_warn_obs: int = 50          # Warn if fewer than this
profile_significance_level: float = 0.05  # Pre-correction threshold
profile_multiple_testing: str = "holm_bonferroni"

# Horizon-aware alpha evaluation
evaluate_at_native_horizon: bool = False  # Enable after first profiling run
horizon_blend_weights_source: str = "static"  # "static" or "ic_proportional"
```

---

## 10. Execution Plan

### Phase 1: Diagnostic Profiling (Current Task)

1. Implement `profile_alphas_multi_horizon()` in `alphago_layering.py`
2. Add signal caching in `validate_alphas_walkforward()`
3. Run on synthetic data + real data (AAPL, MSFT, NVDA)
4. Review the IC × horizon matrix
5. **Do not change trading logic**

### Phase 2: Horizon-Informed Ensemble (Pending Review)

6. Update HorizonBlender weights based on profiling results
7. Implement horizon-aware alpha demotion
8. Consider expanding horizon_bars from (1, 5, 21) to (1, 5, 10, 21)

### Phase 3: Advanced Optimization (Future)

9. IC-proportional horizon blend weights
10. Tiered rebalancing (fast/slow tracks)
11. Regime-conditional horizon weighting (Hurst exponent gates)
12. Horizon-specific meta-learner training targets

---

## 11. Key Risks and Unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| Insufficient data for long horizons (H=15,20) | HIGH | Flag low-confidence estimates; require minimum 33 non-overlapping obs |
| Horizon snooping bias | MEDIUM | Holm-Bonferroni correction; require economic hypothesis alignment |
| Profiling on synthetic data may not transfer to real data | HIGH | Run on both; compare profiles; weight real-data results higher |
| Expanding horizon buckets increases complexity without benefit | LOW | Phase 2 changes gated behind review; start with diagnostics only |
| Overlapping return bias if implementation uses overlapping windows | HIGH | Non-overlapping windows are mandatory in the implementation |
| Feature count explosion if horizon IC fed to meta-learner | MEDIUM | Aggregate to single best-horizon-per-alpha, not full matrix |

---

## 12. Summary of Recommendations

1. **Implement the profiling function now** as a diagnostic module within the existing `validate_alphas_walkforward()` flow. Profile at horizons {1, 5, 10, 15, 20} using non-overlapping returns and Holm-Bonferroni correction across 35 tests.

2. **Expect Value and Carry to rehabilitate** at longer horizons. The REJECT verdicts from 1-bar evaluation are likely false negatives from horizon mismatch. This is the single most important finding this analysis will produce.

3. **Expect Trend to peak at H=10-20**, confirming the academic consensus on momentum timeframes. If confirmed, the current horizon_blend_weights are underweighting the medium-horizon bucket.

4. **Expect RL and MR to peak at H=1-5**, confirming their short-horizon design.

5. **Do not act on the results until the full IC matrix is reviewed.** The profiling is diagnostic-only per the CLAUDE.md directive. Phase 2 changes are queued but gated behind explicit approval.

6. **The 5 specific horizons chosen (1, 5, 10, 15, 20) are motivated by**: (a) the natural clustering of alpha families into short/medium/long groups, (b) the √N noise reduction at each step, (c) practical sample size constraints (H=20 requires 500+ bars in the test window for 25 non-overlapping observations), and (d) alignment with the common rebalancing frequencies used in institutional practice (daily, weekly, biweekly, monthly).