# Beyond IC: A Complete Toolkit for Measuring Alpha Edge

## Executive Summary

IC (Information Coefficient) is the most common metric for evaluating alpha signals, but it has significant blind spots. It measures only one dimension — **average linear rank correlation** — while ignoring stability, capacity, tail behavior, turnover costs, and economic interpretability. This document catalogs **12 complementary metrics** organized into 5 dimensions, explains what each captures that IC misses, and recommends which to implement in our system.

---

## 1. What IC Actually Measures (And What It Doesn't)

IC = Spearman rank correlation between signal values and subsequent realized returns, averaged across time periods.

**What IC captures well:**
- Average directional accuracy of ranking
- Comparable across different signal scales (rank-based, so unit-free)
- Directly connects to portfolio performance via Grinold's law: IR ≈ IC × √Breadth

**What IC misses completely:**

| Blind Spot | Why It Matters | Example Failure |
|---|---|---|
| **Stability** | A signal with IC=0.03 that's positive 90% of months is far better than IC=0.03 that's +0.15 half the time and -0.09 the other half | Both show same average IC |
| **Monotonicity** | IC can be positive even if only the top decile predicts well and everything else is noise | Good IC, but only useful for a long-only top-decile portfolio |
| **Tail behavior** | IC treats a signal that nails the top/bottom 5% the same as one that slightly ranks the middle | The top/bottom prediction matters most for PnL |
| **Turnover** | A signal that flips sign every bar has the same IC as one that's stable for weeks | The first one generates enormous costs |
| **Capacity** | IC doesn't know if the edge exists only in micro-caps you can't trade | Signal may be untradeable |
| **Nonlinearity** | Rank correlation measures monotonic relationships only | Signal may predict well in extreme regimes but not moderate ones |

---

## 2. The 12 Metrics: Organized by Dimension

### Dimension 1: Predictive Power (IC is here, but not alone)

**Metric 1 — IC (Information Coefficient)**
What: Spearman rank correlation between signal and forward return.
Formula: `IC(t) = SpearmanCorr(signal_t, return_{t+H})`
Reported as: time-series mean of IC(t).
Your system already has this. Keep it as the baseline.

**Metric 2 — Rank IC (RankIC)**
What: Same as IC but using Spearman correlation explicitly on rank-transformed data. In practice, IC and RankIC are often identical when both use Spearman. The distinction matters when some implementations use Pearson for IC (which is sensitive to outliers). Always use Spearman — Pearson IC can be distorted by a single extreme return.

**Metric 3 — Hit Rate (Directional Accuracy)**
What: Fraction of periods where sign(signal) == sign(forward return).
Formula: `HitRate = mean(sign(mu_t) == sign(r_{t+H}))` for bars where |mu| > threshold.
Why IC misses this: IC measures ranking quality across the full cross-section. Hit rate measures the simpler question: "when the signal says up, does it go up?" For a single-asset system like ours (no cross-section), hit rate is actually more directly relevant than IC.
Interpretation: 50% = random. 52-55% = strong edge for single-asset daily trading. >55% = exceptional (verify it's not overfitted).
Key advantage: intuitive, directly maps to PnL for a single-asset system.

**Metric 4 — Weighted IC (Conviction-Weighted)**
What: IC computed by weighting each observation by |signal| magnitude.
Formula: Weight each (signal, return) pair by |signal_t| before computing correlation.
Why IC misses this: Standard IC treats a whispered "maybe up" the same as a screamed "definitely up." Weighted IC tells you whether the signal is more accurate when it's more confident.
Interpretation: If Weighted IC >> IC, the signal is well-calibrated (high confidence = high accuracy). If Weighted IC << IC, confidence is noise.

### Dimension 2: Stability (IC's Biggest Blind Spot)

**Metric 5 — ICIR (IC Information Ratio)**
What: Mean(IC) / Std(IC) across time periods.
Formula: `ICIR = mean(IC_t) / std(IC_t)`
Why this matters: This is arguably more important than IC itself. A Wells Fargo research paper (Cao & Zhang, 2020) demonstrated that IC alone is insufficient because real-world ICs are narrow and noisy — a "stellar" model has IC = 0.05, a "bad" model has IC = -0.01, and the noise band between them is wide. ICIR cuts through the noise by penalizing inconsistency.
Benchmarks: ICIR > 0.5 is institutional quality. ICIR > 1.0 is excellent. Our target: ICIR > 0.3 per alpha.
Key insight: An alpha with IC = 0.015 and ICIR = 0.8 is far more valuable than IC = 0.04 and ICIR = 0.2.

**Metric 6 — Rolling IC Regime Stability**
What: Compute IC in rolling windows (e.g., 63-bar), then report the fraction of windows where IC > 0.
Formula: `Stability = mean(rolling_IC_63 > 0)`
Why: Tells you whether the signal works across market regimes or only in specific conditions. If rolling IC is positive only during trending markets and negative during choppy ones, the signal has a regime dependency that the ensemble's gating should account for.
Interpretation: >70% = stable signal. 50-70% = regime-dependent (need gating). <50% = unreliable.

### Dimension 3: Economic Quality (Does the Edge Survive Reality?)

**Metric 7 — Quantile Spread Return (Long-Short Return)**
What: Sort bars by signal strength into quantiles (e.g., quintiles). Compute the return of the top quantile minus the bottom quantile.
Formula: For single-asset time-series: group bars by signal quintile, compute mean forward return per quintile. Report Q5 - Q1 spread.
Why IC misses this: IC can be positive even if the spread is concentrated in one quantile. A good alpha should show a **monotonic** relationship — Q1 < Q2 < Q3 < Q4 < Q5. If Q5 is great but Q2-Q4 are noise, the alpha only works at extremes.
What to look for: Monotonicity across quintiles. Non-monotonic but significant Q5-Q1 spread = signal only works at extremes (may still be useful but with different sizing logic).

**Metric 8 — Turnover-Adjusted IC (Cost-Aware Edge)**
What: IC penalized by signal turnover.
Formula: `TurnoverAdjIC = IC - λ × Turnover` where Turnover = mean(|signal_t - signal_{t-1}|) and λ is calibrated to your cost model.
Why: An alpha that flips every bar generates enormous transaction costs. A signal with IC = 0.03 and daily turnover of 80% may produce negative net returns, while IC = 0.015 with 10% turnover is profitable. The AlphaForge paper (Shi et al., 2024) found that incorporating turnover into evaluation consistently selected alphas with better live performance.
Connection to your system: Your L4 cost model (`cost = half_spread + impact_coeff * sqrt(trade_size / ADV) + fixed_fee`) should feed into this metric. Compute `net_alpha_return = mu * fwd_return - estimated_cost_per_bar`.

**Metric 9 — Capacity-Weighted IC**
What: IC computed only on bars where the signal is executable at realistic size.
Formula: Exclude (or downweight) bars where ADV < minimum threshold, or where impact model says cost > X% of expected alpha return.
Why: A signal that works brilliantly on illiquid small-caps at 3 AM is useless if you can't trade it. For your single-asset equity system, this is less critical (AAPL/MSFT/NVDA are very liquid), but it becomes essential if you expand to less liquid instruments.

### Dimension 4: Robustness (Will It Survive Tomorrow?)

**Metric 10 — Perturbation Fidelity Score (PFS)**
What: How much does the signal's performance degrade when you inject noise into the input data?
Formula: Add Gaussian noise (calibrated to historical daily volatility) to input prices, regenerate signals, measure IC degradation.
Why: From the AlphaEval framework (Ding et al., 2025, Peking University), this metric tests whether the alpha is capturing genuine patterns or fragile data artifacts. An alpha that collapses when you perturb prices by 0.1% is extremely fragile.
Interpretation: IC_perturbed / IC_original > 0.8 = robust. < 0.5 = fragile.

**Metric 11 — OOS Decay Ratio**
What: Ratio of out-of-sample IC to in-sample IC.
Formula: `Decay = IC_oos / IC_is`
Your system already has this: The validation_engine.py SignificanceGate computes OOS/IS decay and flags alphas where decay > 50% (meaning OOS Sharpe < 50% of IS Sharpe). Keep it — it's one of the best overfitting detectors.
The research literature (Bailey & Lopez de Prado) suggests that most alphas with OOS decay > 60% are overfit.

### Dimension 5: Diversity (Ensemble Health)

**Metric 12 — Marginal IC (Incremental Contribution)**
What: How much does adding this alpha improve the ensemble IC beyond what the other alphas already provide?
Formula: `Marginal_IC = IC(ensemble + alpha) - IC(ensemble without alpha)`
Why: An alpha with IC = 0.03 that's 95% correlated with your best existing alpha adds almost nothing. An alpha with IC = 0.01 that's uncorrelated with everything adds genuine diversification.
Connection to your system: Your L2 AlphaCorrelationManager already tracks pairwise alpha correlations. Marginal IC formalizes this into a single number per alpha.

---

## 3. The AlphaEval Framework (State of the Art, 2025)

The most comprehensive recent contribution is AlphaEval from Peking University (Ding et al., August 2025). Their framework evaluates alphas across five dimensions without requiring full backtesting:

1. **Predictive Power Score (PPS):** Based on RankIC and RankICIR — combines accuracy and stability
2. **Rolling Regime Evaluation (RRE):** Temporal stability across market regimes
3. **Perturbation Fidelity Score (PFS):** Robustness to input noise (Gaussian and fat-tailed)
4. **Logic Score:** Uses LLMs to evaluate whether the alpha has a coherent economic rationale (interesting but not essential for our system since all our alphas already have stated economic hypotheses)
5. **Diversity Entropy (DE):** Eigenvalue decomposition of the alpha covariance matrix — measures how many independent signals you truly have

Their key finding: **portfolios selected using the composite AlphaEval score consistently outperform those selected by IC alone**, both in cumulative return and risk-adjusted metrics. Single-metric screening (IC only) misses important dimensions of alpha quality.

---

## 4. Which Metrics to Implement for Our System

Given that our system is single-asset time-series (not cross-sectional), some metrics need adaptation. Here's the priority ranking:

### Must Implement (P0)

| Metric | Why P0 | Effort |
|---|---|---|
| **ICIR** | More important than IC itself for our weak-signal environment. IC=0.015 with ICIR=0.8 is a keeper; IC=0.015 with ICIR=0.15 is noise. | Small — compute std of rolling IC, divide |
| **Hit Rate** | Most intuitive metric for single-asset. Directly answers "does it predict direction?" No cross-section needed. | Small — sign comparison |
| **Turnover-Adjusted IC** | Our signals are evaluated without cost context. An alpha that flips every bar is worthless regardless of IC. | Medium — requires signal persistence calculation |

### Should Implement (P1)

| Metric | Why P1 | Effort |
|---|---|---|
| **Quantile Spread (monotonicity)** | Catches alphas that only work at extremes. Important for position sizing decisions. | Medium — quintile sort + mean returns |
| **Rolling IC Regime Stability** | Directly feeds into L2 regime gating decisions. If an alpha works only in trending markets, gate it. | Small — rolling window IC |
| **Weighted IC** | Tests calibration of the confidence field. Critical for RL alpha whose confidence is entropy-based. | Small — weighted correlation |
| **Marginal IC** | Answers "does this alpha add anything the ensemble doesn't already have?" Prevents redundancy. | Medium — requires leave-one-out ensemble IC |

### Nice to Have (P2)

| Metric | Why P2 | Effort |
|---|---|---|
| **Perturbation Fidelity** | Good robustness test but requires running the full signal generation pipeline with perturbed data. | Large — full pipeline re-run |
| **Diversity Entropy** | Most useful when you have 20+ alphas. With 7, simpler pairwise correlation monitoring suffices. | Medium — eigenvalue decomposition |

---

## 5. Concrete Implementation Sketch

### 5.1 ICIR (in validate_alphas_walkforward)

```python
def compute_icir(signal_series, return_series, window=63):
    """
    Compute IC Information Ratio.
    
    Returns: (mean_ic, std_ic, icir, pct_positive)
    """
    rolling_ics = []
    for start in range(0, len(signal_series) - window, window // 2):  # 50% overlap
        end = start + window
        sig_w = signal_series[start:end]
        ret_w = return_series[start:end]
        if len(sig_w) < 20:
            continue
        # Spearman rank correlation
        ic, _ = scipy.stats.spearmanr(sig_w, ret_w)
        if np.isfinite(ic):
            rolling_ics.append(ic)
    
    if len(rolling_ics) < 3:
        return 0.0, 1.0, 0.0, 0.0
    
    mean_ic = np.mean(rolling_ics)
    std_ic = np.std(rolling_ics)
    icir = mean_ic / (std_ic + 1e-8)
    pct_positive = np.mean(np.array(rolling_ics) > 0)
    
    return mean_ic, std_ic, icir, pct_positive
```

### 5.2 Hit Rate

```python
def compute_hit_rate(signal_series, return_series, min_signal=1e-6):
    """
    Directional accuracy: how often does sign(signal) == sign(return)?
    Only counts bars where signal has meaningful magnitude.
    """
    mask = np.abs(signal_series) > min_signal
    if mask.sum() < 30:
        return 0.5, 0  # Not enough active bars
    
    sig_sign = np.sign(signal_series[mask])
    ret_sign = np.sign(return_series[mask])
    
    hits = np.sum(sig_sign == ret_sign)
    n_active = mask.sum()
    hit_rate = hits / n_active
    
    # Binomial test for significance
    from scipy.stats import binom_test
    p_value = binom_test(hits, n_active, 0.5)
    
    return hit_rate, p_value
```

### 5.3 Turnover-Adjusted Return

```python
def compute_turnover_adjusted_return(signal_series, return_series, 
                                       cost_per_unit_turnover=0.001):
    """
    Net alpha return after accounting for signal turnover cost.
    
    cost_per_unit_turnover: cost for a full signal flip (calibrate 
    from your L4 cost model; 10 bps is conservative for liquid equities).
    """
    # Gross alpha return per bar
    gross_returns = signal_series[:-1] * return_series[1:]
    
    # Signal turnover per bar
    turnover = np.abs(np.diff(signal_series))
    
    # Net return
    costs = turnover * cost_per_unit_turnover
    net_returns = gross_returns - costs[:len(gross_returns)]
    
    gross_sharpe = np.mean(gross_returns) / (np.std(gross_returns) + 1e-8) * np.sqrt(252)
    net_sharpe = np.mean(net_returns) / (np.std(net_returns) + 1e-8) * np.sqrt(252)
    cost_drag_bps = np.mean(costs) * 10000  # Convert to bps/bar
    
    return {
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'cost_drag_bps_per_bar': cost_drag_bps,
        'mean_turnover': np.mean(turnover),
        'cost_kills_alpha': net_sharpe < 0 and gross_sharpe > 0,
    }
```

### 5.4 Quantile Spread

```python
def compute_quantile_spread(signal_series, return_series, n_quantiles=5):
    """
    Sort bars by signal strength, compute mean return per quantile.
    Tests monotonicity: does stronger signal -> higher return?
    """
    # Assign quantile labels
    quantile_labels = pd.qcut(signal_series, n_quantiles, labels=False, 
                               duplicates='drop')
    
    # Mean return per quantile
    quantile_returns = {}
    for q in range(n_quantiles):
        mask = quantile_labels == q
        if mask.sum() > 10:
            quantile_returns[q] = np.mean(return_series[mask])
        else:
            quantile_returns[q] = np.nan
    
    # Long-short spread (top minus bottom)
    spread = quantile_returns.get(n_quantiles - 1, 0) - quantile_returns.get(0, 0)
    
    # Monotonicity: Spearman correlation between quantile rank and mean return
    valid_qs = [(q, r) for q, r in quantile_returns.items() if np.isfinite(r)]
    if len(valid_qs) >= 3:
        q_ranks, q_returns = zip(*valid_qs)
        monotonicity, _ = scipy.stats.spearmanr(q_ranks, q_returns)
    else:
        monotonicity = 0.0
    
    return {
        'quantile_returns': quantile_returns,
        'spread': spread,
        'monotonicity': monotonicity,  # 1.0 = perfectly monotonic
        'is_monotonic': monotonicity > 0.8,
    }
```

---

## 6. Recommended Evaluation Report Format

After implementation, each alpha's evaluation should print something like:

```
┌─────────────────────────────────────────────────────────────────┐
│ ALPHA SCORECARD: rl_ppo_mcts  (horizon=5, H_eval=5)            │
├────────────────────┬──────────┬──────────────────────────────────┤
│ Metric             │ Value    │ Assessment                       │
├────────────────────┼──────────┼──────────────────────────────────┤
│ IC (Spearman)      │ +0.015   │ Weak but nonzero                 │
│ ICIR               │  0.42    │ ✓ Acceptable (>0.3)              │
│ Hit Rate           │ 52.1%    │ ✓ Above random (p=0.03)          │
│ Rolling IC % > 0   │ 64%      │ ~ Regime-dependent               │
│ Quantile Spread    │ +1.2 bps │ ✓ Positive                       │
│ Monotonicity       │  0.90    │ ✓ Strong                         │
│ Weighted IC        │ +0.021   │ ✓ Confidence is calibrated       │
│ Signal Turnover    │ 34%/bar  │ ~ Moderate                       │
│ Turnover-Adj Sharpe│  0.18    │ ✗ Below 0.5 threshold            │
│ OOS/IS Decay       │  0.62    │ ✓ Acceptable (<50% loss)         │
│ Marginal IC        │ +0.004   │ ~ Small but additive             │
│ t-stat             │  2.8     │ ~ Below 3.0 threshold            │
├────────────────────┴──────────┴──────────────────────────────────┤
│ VERDICT: MARGINAL — keep in ensemble at reduced weight           │
│ KEY INSIGHT: Confidence calibrated (Weighted IC > IC)            │
│ CONCERN: Regime-dependent (64% positive), high turnover          │
│ RECOMMENDATION: Gate on trending regimes, reduce rebalance freq  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. The Bottom Line: IC Is Necessary But Not Sufficient

IC answers **one question**: "on average, does the signal rank future returns correctly?"

A complete alpha evaluation needs to answer **six questions**:

1. Does it predict? → IC, Hit Rate
2. Does it predict **consistently**? → ICIR, Rolling IC Stability
3. Does it predict where it **matters**? → Quantile Spread, Monotonicity
4. Does the prediction **survive costs**? → Turnover-Adjusted IC
5. Is the prediction **robust**? → OOS Decay, Perturbation Fidelity
6. Does it **add something new**? → Marginal IC, Correlation with existing alphas

An alpha that scores well on IC but poorly on ICIR is a trap. An alpha that scores well on IC but whose turnover eats the edge is a trap. An alpha with moderate IC but excellent ICIR, good monotonicity, low turnover, and low correlation with existing signals is a gem.

For your system specifically, **ICIR is the single most important metric you're missing**. Implement it first. It will likely change some of your PASS/REJECT verdicts.
