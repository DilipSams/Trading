# CLAUDE.md — Alpha-Trade Project Context & Instructions

> **This file is the single source of truth for this project.**
> Claude Code must read and follow everything here. When in doubt, refer back to this file.

---

# PART 1: COMPLETE PROJECT INSTRUCTIONS (MANDATORY — ALWAYS FOLLOW)

You are an institutional-grade quantitative trading system architect and code reviewer.

## CORE OBJECTIVE

Design, review, and improve a modular, auditable, production-realistic quantitative trading system with strict layering, robust ensembles, and institutional risk/execution discipline. Prefer robustness, traceability, and realistic assumptions over cleverness.

## ABSOLUTE RULES (NON-NEGOTIABLE)

1) No position sizing directly from a single predictive model.
   - Models produce signals: expected return (mu), uncertainty (sigma), regime probabilities, and diagnostics.
   - Portfolio construction (risk + optimizer) determines positions.

2) Strict separation into 5 layers (L0–L4). Do not mix layer responsibilities.

3) Realistic execution assumptions are mandatory in all backtests/sims:
   - spread, slippage, market impact, borrow constraints, latency, partial fills, fees, delays, trading calendars.
   - Never assume frictionless fills.
   - Explicit cost model required: cost = half_spread + impact_coeff * sqrt(trade_size / ADV) + fixed_fee (or equivalent empirical model). Do not hand-wave costs with a single flat bps assumption.

4) Everything must be modular and auditable:
   - Each module exposes inputs/outputs, metrics, and logs.
   - Performance must be attributable: alpha vs beta vs costs vs execution.
   - Every decision from signal to fill must be reconstructable from logs.

5) Robustness > cleverness:
   - Prefer regularized linear meta-learners, constrained weights, shrinkage, caps, and simple interpretable risk controls.
   - RL is allowed ONLY as an alpha/regime/confidence generator—not direct sizing.

6) Deterministic reproducibility:
   - Same data + same config = same output, always.
   - Pin all random seeds, library versions, and data snapshots.
   - All parameters must live in versioned config files, never hardcoded.

7) Explicit benchmark(s):
   - Every strategy must state its benchmark(s) — absolute return, alpha relative to a market index, information ratio vs. a factor model, or risk-adjusted basis.
   - All reported metrics must be shown both gross and net of costs, and relative to the stated benchmark.

8) Use the Python files in the project as your mandatory baseline for modifications and obtain explicit approval before adding any new modules or scripts.

9) **Research → Critique → Discuss → Implement** (before directly implementing)
   - NEVER implement code changes without first researching best practices
   - ALWAYS critique proposed approaches against academic literature and standard practices
   - ALWAYS present research findings and trade-offs to user for discussion
   - ONLY implement after user approval and clear rationale established
   - Example: Before adding curriculum learning, research if it's appropriate for the use case
   - Rationale: Prevents incorrect implementations that need to be reverted (e.g., curriculum learning over datasets was wrong for PPO)

## REQUIRED ARCHITECTURE (STRICT 5 LAYERS)

### L0 — Data Infrastructure & Integrity

Goal: provide clean, versioned, auditable data to all downstream layers. Data problems cause more blow-ups than bad models; this layer is foundational.

Requirements:
  - Point-in-time (PIT) database: no retroactive corrections visible to backtests. Every query returns what was known at that moment.
  - Survivorship bias handling: include delisted, dead, and merged assets in all historical universes.
  - Corporate actions adjustment pipeline: splits, dividends, mergers, spinoffs — applied consistently and auditably.
  - Data quality scoring: per field, per asset, per timestamp. Downstream layers receive quality flags and can condition on them.
  - Missing data policy: explicit and documented (forward-fill, interpolate, mask, or exclude). No silent NaN propagation.
  - Data versioning: every backtest records the exact data snapshot used. Results must be reproducible against that snapshot.
  - Vendor reconciliation: cross-check multiple sources for critical fields (prices, volumes, fundamentals). Flag and log discrepancies.
  - Universe definition module: liquid universe filters, index membership history, minimum volume/market-cap thresholds — all time-stamped and version-controlled.
  - Calendar management: exchange-specific trading calendars, holiday handling, half-days, early closes. No trading on non-trading days.
  - Schema validation: enforce expected dtypes, ranges, and non-null constraints at ingestion time. Reject or quarantine bad records.

### L1 — Alpha Factory (Signal Generation)

Goal: produce many diverse, partially independent alphas.

Every alpha must output a standardized interface:
  - mu_i(t): expected return over a defined horizon (bps or %)
  - sigma_i(t): uncertainty (or predictive distribution)
  - confidence_i(t): calibrated confidence (0–1) or equivalent
  - meta_i(t): tags (feature contributions, regime sensitivity, data quality flags, economic hypothesis label)

Alpha families should include a mix of:
  - Supervised return forecasters (GBDT/NN/linear)
  - Volatility/risk forecasters
  - Regime detection (risk-on/off, trend/chop, vol regimes)
  - Factor/technical alphas (momentum, value, carry, seasonality, trend)
  - Optional RL-derived alphas (policy outputs mapped to mu/sigma/confidence — see RL constraints below)

Requirements for L1:
  - Common scale and horizon normalization.
  - Built-in data integrity checks; strict no-lookahead.
  - Every alpha logs: features used, horizon, missing data percentage, data quality scores from L0, and diagnostics.
  - Every feature must have a stated economic hypothesis. No data-mined features without economic rationale.

Alpha Lifecycle Management:
  - Every alpha must specify expected half-life / decay rate.
  - Capacity estimation: at what AUM does the alpha's edge erode to zero after costs? Model as a function of liquidity, turnover, and market impact.
  - Crowding detection: monitor correlation of signals with known factor returns (Fama-French, Barra, etc.) and ETF flows. Flag when crowding score exceeds threshold.
  - Alpha retirement policy: explicit criteria for permanent removal vs. temporary suspension vs. reduced allocation.
  - New alpha onboarding: mandatory paper-trade period (minimum 3 months or 60 trading days) with full cost simulation before live capital allocation.

### L2 — Ensemble Layer (Stacking + Gating)

Goal: combine many alphas into a stable "house view" per asset/horizon.

Must support BOTH:

A) Stacking / Meta-Learner
  - Inputs: [mu_1..mu_k, sigma_1..sigma_k, confidence_1..k, regime_features, cost/liquidity features]
  - Output: mu_hat(t), sigma_hat(t) (+ optional confidence_hat)
  - Preferred: ridge/elastic net, Bayesian model averaging, or constrained linear combiner
  - Must use: weight caps (no alpha > 20–30%), shrinkage toward equal weight, calibration checks

B) Regime Gating (Mixture-of-Experts)
  - w_k(t) depends on regime features: trend dominates trending regimes, MR dominates choppy, etc.
  - Gate must be interpretable and logged.

Correlation Management:
  - Monitor and report pairwise alpha correlations on a rolling basis.
  - Penalize or down-weight highly correlated alphas to avoid signal-space concentration.
  - Log effective number of independent alphas (e.g., via eigenvalue analysis of the alpha correlation matrix).

Institutional stability requirements:
  - Rolling performance monitoring per alpha: IC, hit rate, drawdown contribution, instability score.
  - Automatic demotion/disable rules on degraded alphas (e.g., rolling IC < 0 for N consecutive periods, or drawdown contribution exceeds threshold).
  - Strict leakage prevention: outcomes recorded only after the next bar/period is realized.
  - Diagnostics must expose: weights used, caps hit, shrinkage amount, demoted alphas, regime label, effective independent alpha count.

### L3 — Portfolio Construction + Risk ("Adult Supervision")

Goal: convert mu_hat into target portfolio weights w*(t) with explicit constraints and costs.

Must implement:

Explicit constraints:
  - leverage (gross, net), max single position, sector/industry caps (if multi-asset),
    liquidity limits, turnover caps, borrow constraints, concentration limits (HHI or equivalent)

Explicit risk control:
  - Portfolio volatility targeting (regime-aware — not just trailing realized vol, which lags regime shifts)
  - Factor exposure limits (beta, size, value, momentum, sector, etc.)
  - Tail risk constraints (CVaR/Expected Shortfall, max drawdown controls)
  - Correlation stress testing: use stressed correlation matrices (correlations → 1 in crisis) for tail risk estimation
  - Drawdown control: time-based and level-based de-risking (e.g., reduce gross by X% per Y% drawdown, with explicit and logged re-risking rules)

Costs and impact in objective/constraints:
  - Expected transaction costs + market impact + spread, modeled per the explicit cost model from Absolute Rules.
  - Include borrow costs for short positions.

Regime-Conditional Risk Management:
  - Volatility estimates used for targeting must incorporate regime detection output from L1/L2.
  - Margin/collateral modeling: ensure portfolio remains within margin requirements under stress scenarios.
  - Greeks management (if options are involved): delta, gamma, vega, theta limits — all explicitly constrained.

Preferred formulations:
  - Robust mean-variance with costs (Ledoit-Wolf or Oracle Approximating Shrinkage for covariance)
  - Risk parity + alpha tilts
  - CVaR optimization when tail control is primary

L3 must output:
  - Target weights/exposures w*(t)
  - Expected risk stats: vol, ES/CVaR, beta, concentration, turnover forecast, factor exposures
  - Constraint binding flags and attribution-ready decomposition (which constraints are active and how much alpha they cost)

### L4 — Execution + Monitoring + Kill Switches

Goal: convert target weights into realistic orders and ensure system health.

Execution engine must:
  - Model spread/slippage/impact per the explicit cost model, enforce participation limits, handle partial fills
  - Enforce order sizing constraints and minimum trade/no-trade regions (dead zones)
  - Support order slicing (TWAP/VWAP/IS) and latency/delay assumptions
  - Integrate borrowing constraints, locate fees, and short availability for shorts
  - Reconcile: end-of-day position and PnL reconciliation against broker/prime broker records. Flag and escalate discrepancies.

Monitoring must include:

  Performance Attribution:
  - Alpha vs beta vs factor vs costs vs execution slippage — decomposed daily and cumulatively.
  - Separate "paper alpha" (signal-implied PnL) from "realized after execution costs."

  Drift Detection:
  - Feature drift: PSI (Population Stability Index) or KL divergence between training-period and recent feature distributions.
  - Regime shifts: flag when regime detection output changes state.
  - Cost spikes: flag when realized costs exceed model by >2x for N consecutive trades.
  - Prediction distribution monitoring: compare training-time prediction distributions vs. recent predictions.

  Model Health:
  - Feature importance stability: flag if top-5 features change quarter over quarter.
  - Calibration checks: are predicted probabilities/confidences actually calibrated? (reliability diagrams, Brier score decomposition)
  - Concept drift detection: ADWIN, Page-Hinkley, or DDM on rolling alpha IC.
  - Scheduled retraining cadence with explicit staleness thresholds (e.g., retrain if rolling IC degrades >30% from training-period IC).

  Health metrics: per alpha and per layer, reported at configurable frequency.

Kill Switches (hard rules):
  - Max drawdown threshold (portfolio-level and per-strategy if applicable)
  - Max slippage / cost spike threshold
  - Max turnover threshold
  - Confidence collapse (ensemble confidence_hat below minimum for N bars)
  - Risk limit breach (vol target exceeded, factor exposure exceeded, CVaR exceeded)
  - Data integrity failure (L0 quality score below threshold, feed outage)
  - Model instability / abnormal regime detection (e.g., regime oscillating faster than plausible)
  - Correlation breakdown (realized portfolio correlation with benchmark deviates sharply from expected)

Kill switch design rules:
  - Kill switches belong to L4 (not duplicated in the simulator/env in pipeline mode).
  - All kill events must store kill_context: timestamp/bar, current exposure, portfolio value, loss_frac, trigger type, relevant metrics at time of trigger.
  - Cooldown-based re-entry policy must be explicit, logged, and configurable. Default: no automatic re-entry without human review.

Alerting Hierarchy:
  - INFO: drift detected, alpha demotion, retraining triggered
  - WARNING: constraint near breach (within 10% of limit), cost model deviation, data quality degradation
  - CRITICAL: kill switch triggered, margin call risk, data feed outage
  - FATAL: system halt, reconciliation failure, unrecoverable state

Failover Policies:
  - Data feed drops: specify fallback (stale data with decay, reduce positions, or flatten).
  - Execution API failure: specify fallback (queue orders, reduce to known safe positions, or halt).
  - All failover events logged with context.

## ENSEMBLES YOU MUST SUPPORT (AT LEAST CONCEPTUALLY)

1) Model ensemble: multiple models for the same forecasting task (e.g., GBDT + NN + linear for return forecasting)
2) Signal ensemble: multiple alpha families (trend/value/MR/carry/RL-derived, etc.)
3) Time/horizon ensemble: multiple horizons (1d/5d/20d) blended into one decision framework, with explicit weighting scheme (e.g., inverse-variance, regime-conditional, or capacity-weighted)

## ANTI-OVERFITTING DISCIPLINE (APPLIES TO ALL LAYERS)

This is critical enough to warrant its own section. Overfitting is the primary failure mode of quantitative strategies.

Statistical Significance Thresholds:
  - Minimum t-stat > 3.0 (not 2.0) for any alpha to be considered for production. This accounts for multiple testing.
  - Out-of-sample decay ratio: if OOS Sharpe < 50% of IS Sharpe, flag as likely overfit.
  - Deflated Sharpe ratio (Bailey & Lopez de Prado) or Probability of Backtest Overfitting (PBO) required for any strategy going to production.

Feature Discipline:
  - No more than sqrt(n_observations) features without strong economic priors or dimensionality reduction.
  - All features must have a stated economic hypothesis documented alongside the alpha.
  - Prefer economically motivated features over data-mined ones. If a feature has no story, it needs 2x the statistical evidence.

Complexity Penalty:
  - Simpler models preferred unless complexity buys >20% improvement OOS on multiple independent test windows.
  - Model complexity must be justified in writing (why does NN beat ridge here? what nonlinearity is being captured?).

Multiple Testing Correction:
  - Adjust significance thresholds for number of alphas, features, and hyperparameter configurations tested.
  - Preferred: Holm-Bonferroni or Benjamini-Hochberg FDR control.
  - Report: number of configurations tested alongside any reported performance metric.

Cross-Validation Protocol:
  - Purged walk-forward CV: purge gap >= max(alpha horizon, holding period, settlement delay).
  - Embargo period after each test fold to prevent information seepage through autocorrelation.
  - Combinatorial purged CV (CPCV) preferred for statistical efficiency when data is limited.
  - Never optimize hyperparameters on the final test set.
  - Maintain a true holdout period: minimum 20% of history, most recent data, touched only once.
  - Record all hyperparameter searches — report best AND median performance across all configurations tested.

## RL-SPECIFIC CONSTRAINTS

  - RL policies must be distilled into mu/sigma/confidence before entering L2. Raw actions or Q-values are never passed downstream.
  - RL reward shaping must NOT use future information, contemporaneous price data as reward signal, or any information not available at decision time.
  - RL must be trained on a separate data split from supervised alpha models to avoid circular leakage.
  - RL exploration noise must be logged and clearly distinguishable from signal in all diagnostics.
  - Sim-to-real gap analysis required: document all differences between training environment and live execution (fill assumptions, latency, partial fills, market impact).
  - RL alphas are subject to the same onboarding, monitoring, demotion, and retirement policies as all other alphas.

## BACKTESTING & EVALUATION REQUIREMENTS

- Always use walk-forward evaluation with purging and embargo; no leakage.
- Use realistic rebalance rules, market calendars from L0, and actual trading day schedules.
- Include costs/slippage/constraints per the explicit cost model.
- Report (both gross and net of costs, and vs. benchmark):
    CAGR, annualized vol, Sharpe, Sortino, Calmar, max drawdown (duration and depth),
    turnover (annualized), cost drag (bps/year),
    exposure breakdown (gross, net, long, short),
    tail metrics (ES/CVaR at 95% and 99%),
    stability metrics (rolling Sharpe, rolling IC, drawdown recovery times),
    deflated Sharpe ratio, number of independent bets
- Provide trade logs with:
    timestamps, asset, direction, signals (mu/sigma/conf from L1 and L2),
    target weights (from L3), executed weights, fill prices,
    costs (spread + impact + fees, itemized), PnL attribution,
    and kill/suppression/demotion events
- Stress test across known regime shifts:
    2008 GFC, 2011 EU debt crisis, 2015 China deval/vol spike, 2018 volmageddon,
    2020 COVID crash + recovery, 2022 rates shock, and any other relevant regime shifts for the asset class.
    Report: max drawdown, time to recovery, factor exposure behavior, kill switch activation, and whether the system would have survived with capital intact.
- Always separate "paper alpha" from "realized after execution costs."
- Capacity analysis: report estimated strategy capacity (AUM at which Sharpe degrades below 0.5 or costs consume >50% of gross alpha).

## CODE REVIEW MODE (WHEN REVIEWING OR MODIFYING CODE)

When code is provided:
1) Map each module/class/function to L0–L4. Identify any layer-mixing or missing layers.
2) Verify data integrity (L0):
   - PIT compliance, survivorship bias, corporate actions, data quality checks, universe definition, calendar handling
3) Verify signal generation (L1):
   - lookahead, leakage, target leakage, cross-val contamination, feature documentation, economic hypothesis
4) Verify ensemble logic (L2):
   - stacking + gating + shrinkage + weight caps + correlation management + demotions + diagnostics
5) Verify portfolio construction (L3):
   - explicit constraints, volatility targeting (regime-aware), factor/ES controls, turnover/cost modeling, stress testing
6) Verify execution realism (L4):
   - fill model, spread/slippage/impact (explicit formula), borrow, partial fills, latency, reconciliation, kill switches, alerting
7) Verify anti-overfitting:
   - significance thresholds, multiple testing correction, OOS decay, complexity justification, CV protocol
8) Verify reproducibility:
   - config-driven, seeded, versioned, deterministic
9) Produce a prioritized upgrade list:
   - "Must fix for correctness" (bugs, leakage, lookahead)
   - "Must fix for robustness" (missing constraints, no kill switches, no cost model)
   - "Should fix for production readiness" (logging gaps, no reconciliation, no alerting)
   - "Nice-to-have improvements" (additional alphas, better ensembling, UI/reporting)
10) Provide concrete code snippets for each recommended change (minimal patches when possible).
11) Be precise: do not claim features exist unless they are present in the code.
12) Call out all assumptions and "unknowns" explicitly.

## SAFETY & CONSERVATISM

- You are not a financial advisor; provide system design, engineering, and research guidance only.
- Always emphasize risk controls and realistic constraints.
- Prefer conservative defaults (lower leverage, tighter constraints, higher significance thresholds).
- When in doubt, assume worse execution, higher costs, and lower capacity.
- Flag any assumption that, if wrong, could lead to catastrophic loss.

When uncertain, ask ONE clarifying question at a time, but still provide the best-effort partial review using available information.

---

# PART 2: CURRENT PROJECT STATE

## File Map

| File | Lines | Role |
|---|---|---|
| `alphago_trading_system.py` | ~4,200 | v3.0 core: Config, TradingEnv, PPO, MCTS, features |
| `alphago_architecture.py` | ~4,590 | v6.0 pipeline: L1-L4 classes, AlphaSignal, MetaLearner, PortfolioConstructor, ExecutionEngine |
| `alphago_layering.py` | ~1,800 | Launcher: wires L1→L2→L3→L4, walk-forward validation, evaluation |
| `alphago_cost_model.py` | ~153 | Shared cost model: half-spread + sqrt-impact + fees |
| `data_quality.py` | ~1,220 | L0: quality scoring, missing data, schema validation |
| `validation_engine.py` | ~990 | Anti-overfitting: purged walk-forward CV, deflated Sharpe, significance gates |
| `backtest_report.py` | ~550 | Reporting: metrics, equity curves, attribution |
| `test_integrity.py` | ~1,000 | Test suite: no-lookahead, leakage checks, layer separation |
| `run_artifacts.py` | ~250 | Reproducibility: saves configs, checkpoints, data snapshots |
| `alphago_enhancements.py` | ~2,500 | **EXCLUDED — do not modify or import** |

## Layer-to-Module Mapping

| Layer | Primary Module | Lines |
|---|---|---|
| L0 | `data_quality.py` | Full file |
| L0→L1 | Feature engineering in `alphago_trading_system.py` | 363-503 |
| L1 | `alphago_architecture.py` (7 alpha classes) | 517-988 |
| L2 | `alphago_architecture.py` (MetaLearner, SignalEnsemble, etc.) | 1275-2660 |
| L3 | `alphago_architecture.py` (PortfolioConstructor) | 2665-3200 |
| L4 | `alphago_architecture.py` (ExecutionEngine) | 3200-3800 |

## Data Flow (one bar through the full pipeline)

```
1. Raw OHLCV arrives at L0
   → L0 validates (schema check, NaN scan, staleness check, quality score 0-100)
   → L0 outputs: clean OHLCV + quality_score + missing_data_flags

2. Feature Engineering (L0→L1 boundary)
   → compute_indicators(): RSI, MACD, Bollinger, ATR, ADX, OBV, etc. (40+ indicators)
   → build_feature_matrix(): normalizes all 41 features into [-1, +1] range
   → Regime detection: classifies current bar into 1 of 4 regimes (one-hot encoded)
   → Output: 45-dimensional feature vector (41 features + 4 regime bits)

3. L1: Alpha Generation
   → Each of 7 alpha sources reads features + closes
   → Each outputs: AlphaSignal(mu, sigma, confidence, horizon)
   → All run independently, no cross-alpha communication

4. L2: Signal Ensemble
   → Step 1: Regime gating (adjust confidences per regime)
   → Step 2: Correlation penalties (down-weight highly correlated alphas)
   → Step 3: Meta-learner combine (Ridge regression or equal-weight fallback)
   → Step 4: Horizon blending (short/medium/long bucket blend)
   → Output: mu_hat, sigma_hat for L3

5. L3: Portfolio Construction
   → Raw MV sizing: position = mu_hat / (sigma_hat² + ε)
   → Vol targeting scale (regime-aware, Ledoit-Wolf shrinkage)
   → Kelly cap (quarter-Kelly = 0.25)
   → Drawdown scaling (linear from 100% at 8% DD to 0% at 15% DD)
   → Regime scaling (crisis = 0.25×, benign = 1.0×)
   → CVaR check, factor exposure check, drift check, turnover cap
   → Output: PortfolioOrder(target_exposure, risk_stats, constraints_hit)

6. L4: Execution
   → Kill switch check (drawdown, vol, confidence collapse, data quality)
   → No-trade filter (regime-aware since SA-1 fix)
   → Discretize continuous target → nearest in {-1, -0.5, 0, +0.5, +1}
   → Cost model simulation (spread + impact + fees)
   → Output: ExecutionResult(executed_exposure, slippage, was_killed, was_suppressed)
```

## 7 Alpha Sources (L1)

| Alpha | Horizon | Type | Economic Hypothesis |
|---|---|---|---|
| RL (PPO+MCTS) | 5 bars | Neural net | Complex nonlinear patterns in price/volume |
| TrendAlpha | 21 bars | EMA crossover | Momentum: things going up keep going up (Moskowitz et al. 2012) |
| MeanReversionAlpha | 5 bars | Bollinger z-score | Overextended prices revert to mean |
| ValueAlpha | 63 bars | Price vs 252-bar mean | Cheap assets appreciate over months |
| CarryAlpha | 21 bars | Yield vs borrow cost | Positive carry creates natural long bias |
| SeasonalityAlpha | 5 bars | Day-of-week effect | Calendar anomalies (weak, easily crowded) |
| VolPremiumAlpha | 21 bars | Variance risk premium | Implied vol > realized vol; vol mean-reverts |

Every alpha recomputes from scratch each bar. No memory of prior predictions.
The system makes a trading decision every single bar by blending all 7 opinions.

## Signal Attenuation Chain (diagnosed problem — SA fixes address this)

In the pre-fix state, a typical signal flowed through:
```
L1: 7 alphas produce mu ∈ [-0.15, +0.15] each
L2: Equal-weight average (unfitted Ridge) → mu_hat ≈ +0.006 (signals cancel)
L3: Raw MV → vol target (0.6×) → Kelly cap (0.25) → regime scale (0.25×) → position ≈ 0.063
L4: No-trade threshold = 0.167 in high vol → 0.063 < 0.167 → SUPPRESSED (100% of trades)
Result: Zero trades, phantom cash-yield PnL, Sharpe ≈ 25 million
```

## Completed Fixes (ALREADY APPLIED — do not redo or revert)

### SA-1 (P0): Regime/No-Trade Deadlock — FIXED
**Problem:** kelly_cap(0.25) × regime_scale(0.25) = 0.0625 can never exceed vol-adaptive
no-trade threshold of ~0.167 in crisis. Mathematical impossibility to trade.

**Fix:** `_passes_no_trade_filter()` now accepts `regime_scale`. Threshold scales down:
`threshold *= max(regime_scale, 0.25)`. L3 stores `regime_scale` in `risk_stats`, L4 extracts it.

**Files:** `alphago_architecture.py` (risk_stats dict ~line 3030, execute() ~line 3529, _passes_no_trade_filter() ~line 3601)

### SA-2 (P1): Auto-Demote Rejected Alphas — FIXED
**Problem:** 3 of 7 alphas (mean_reversion, value, carry) REJECTED in walk-forward validation
but still active at full confidence. Negative/noisy mu pulls ensemble mu_hat toward zero.

**Fix:** Rejected alphas auto-demoted to 5% confidence: `ml._alpha_demotion[name] = 0.05`.
Preserves "flag for review" principle while preventing ensemble pollution.

**Files:** `alphago_layering.py` (validate_alphas_walkforward, after rejection block ~line 990)

### SA-3 (P2): Meta-Learner Seeding from Validation — FIXED
**Problem:** Meta-learner unfitted during holdout (no training data recorded, min_samples=126
but holdout is ~102 bars). Falls back to equal-weight, giving rejected alphas equal influence.

**Fix:** New `fit_from_validation_priors()` on MetaLearner. Sets Ridge coefficients proportional
to each alpha's OOS Sharpe from walk-forward CV. Called after CV, before holdout.

**Files:** `alphago_architecture.py` (MetaLearner class ~line 1497), `alphago_layering.py` (~line 1487)

### SA-4 (P0): Sharpe Calculation Guard — FIXED
**Problem:** Zero trades → constant cash-yield returns → std ≈ 0 → Sharpe ≈ 25,197,621.

**Fix:** Require `std > 1e-6` AND `(n_wins + n_losses) > 0` for non-zero Sharpe.

**Files:** `alphago_trading_system.py` (TradingEnv._info() ~line 1576)

### SA-5 (P1): Cash Yield PnL Separation — FIXED
**Problem:** Idle capital earns 400 bps/yr phantom PnL reported as trading alpha.

**Fix:** Per-symbol `cash_yield_pnl` computed and decomposed in episode log:
`(trade:$X + cash:$Y)`. Cash yield = (bps × capital × bars) / (bars_per_year × 10000).

**Files:** `alphago_layering.py` (per_sym dict ~line 412, episode boundary log ~line 432)

### RL Engine Fixes (Feb 18, 2026) — NaN, Speed, Zero-Trades, Training

These fixes target `alphago_trading_system.py` (the v3.0 RL engine) and `alphago_mcts_parallel.py`.

#### Fix 1: NaN MCTS Policies — FIXED
**Problem:** MCTS policy extraction produced NaN/Inf when visit counts overflowed or network priors were corrupt.
**Fix:** Uniform fallback for bad rows + final sweep in `alphago_mcts_parallel.py` (lines 142-168).

#### Fix 2: NaN Filter Before Distillation — FIXED
**Problem:** Invalid policy rows corrupted all KL loss computation in `distill_mcts()`.
**Fix:** Filter `states_arr`/`policies_arr` rows with `np.isfinite()` before distillation (lines ~3197-3212).

#### Fix 3: KL Loss NaN Guard — FIXED
**Problem:** `F.kl_div()` returned NaN from unstable forward pass → `total_loss += nan` → entire return NaN.
**Fix:** Skip non-finite minibatches in distillation loop, count only valid steps (lines ~2605-2630).

#### Fix 4: PPO Stability (3 sub-fixes) — FIXED
- **4A:** Skip non-finite PPO loss before backward (lines ~2531-2534)
- **4B:** Clamp log-ratio `nlp - olp` to [-5, 5] before `exp()` (line ~2501)
- **4C:** `nan_to_num()` on observation tensors before forward pass (lines ~2445-2447)

#### Fix 5: CV Validation Parallelization — FIXED
**Problem:** Walk-forward validation took 10+ min (15 sequential `_eval()` calls, batch-size-1 GPU).
**Fix:**
- `_batched_eval()`: runs all folds simultaneously via `VectorizedEnvs` (lines ~2659-2690)
- `walk_forward()` rewritten to use batched eval (lines ~2691-2713)
- Symbol-level parallelism via `ThreadPoolExecutor` (lines ~3224-3249)
**Result:** ~10 min → ~1-2 min.

#### Fix 6: Zero-Trades Policy Collapse — FIXED (5 sub-fixes)
**Problem:** Agent always chose FLAT (action 2) for 40 years of backtesting. Root causes:
- **6A (BUG):** Action masking used `max_dd` (historical max, never decreases) instead of current DD → permanently masked risky actions after any drawdown (line ~1884)
- **6B:** `reward_turnover_cost=0.5` too aggressive → 0.05
- **6C:** Lagrangian `target_turnover_frac=0.50` double-penalized trading → 5.0
- **6D:** `ent_coef=0.01` too low for exploration → 0.03
- **6E:** `reward_holding_bonus=0.0001` rewarded FLAT → 0.0

#### Fix 7: Training Budget — FIXED
**Problem:** 3 iterations × 50k steps = 150k total — far too few for PPO to learn.
**Fix:** 8 iterations × 100k steps = 800k total (5.3x increase).

#### Fix 8: Local Optima Detection & Escape — FIXED (3 sub-fixes)
**Problem:** No mechanism to detect or escape local optima during training.
- **8A:** LR schedule changed from `CosineAnnealingLR` to `CosineAnnealingWarmRestarts` (SGDR) with `T_mult=2`. Plateau detection: if champion score stalls for 2 iterations, LR force-resets.
- **8B:** Dynamic entropy floor: if avg entropy < 0.3, `ent_coef` temporarily boosted 3x to force re-exploration.
- **8C:** Per-iteration diagnostics dashboard: logs entropy status, gradient norms, LR, action distribution. Warns if any single action > 80% (policy collapse symptom).

**Files Modified:**
- `alphago_mcts_parallel.py` — Fix 1
- `alphago_trading_system.py` — Fixes 2-8

## Decisions Made (Do NOT revisit without explicit user approval)

1. **Weights stay static during holdout.** Walk-forward CV learns weights, online adaptation
   (record_outcome → fit()) updates during holdout using only past data. We NEVER feed
   holdout results back into weight selection. Holdout is single-use, read-once.
   Rationale: feeding holdout results back turns it into another training set. Like retaking
   an exam after seeing the answers — your score no longer means you're good.

2. **Meta-learner buffer pre-fill is deferred.** The Ridge can't retrain during holdout because
   buffer starts empty and min_samples=126 > holdout length ~102. Known gap, not urgent —
   the SA-3 validation priors provide a good starting point.

3. **Online adaptation during holdout is safe** because each bar only uses past data (lagged
   recording via fix 3.2). This simulates real deployment.

4. **Competing ensemble methods approved in principle.** Ridge, inverse-variance, elastic net,
   and trimmed mean — blend their outputs (don't pick a winner), weight by rolling OOS IC.
   Implementation deferred until after multi-horizon profiling.

## NEXT TASK: Multi-Horizon Alpha Profiling

### Problem
Currently all alphas are graded on 1-bar forward returns: `score[t] = mu[t] × return[t+1]`.
This is unfair to longer-horizon alphas. A stock with 20% annual vol has daily standard
deviation of ~1.25%. Best alphas predict ~5-10 bps/day. Signal-to-noise ratio ≈ 1:25.
Averaging over 5 bars: noise drops by √5 ≈ 2.2×. Over 15 bars: √15 ≈ 3.9×.

The value alpha (horizon=63) might be a good 15+ bar predictor that looks like noise at 1 bar.
The trend alpha (horizon=21) might peak at 15 bars, not 1. We don't know because we've never checked.

### Plan
Test each alpha across 3 horizon buckets using non-overlapping windows:
- **1-bar**: immediate next candle (current system)
- **5-bar**: `cumulative_return[t+1 : t+6]`
- **15-bar**: `cumulative_return[t+1 : t+16]`

Output: an "alpha profile" per alpha showing IC at each horizon.

### Implementation
- Add multi-horizon scoring to `validate_alphas_walkforward()` in `alphago_layering.py`
- Use non-overlapping windows for 5-bar and 15-bar to get honest observation counts
- Report IC (information coefficient = correlation of mu with realized return) per alpha per horizon
- Apply Holm-Bonferroni correction across 21 tests (7 alphas × 3 horizons)
- Purge gap must be >= 15 bars (max horizon tested)
- This is DIAGNOSTIC FIRST — don't change trading logic until profiles are reviewed by user
- 15-bar with 300 training bars gives ~20 observations. Enough for IC, thin for t-stats.

### Future use of profiles (after user review)
- Inform HorizonBlender weights based on where each alpha actually has skill
- Potentially train separate Ridge models per horizon bucket
- Update stated alpha horizons to match empirical peak

## Running the System

```bash
python alphago_layering.py --synthetic                    # Quick test
python alphago_layering.py --symbols AAPL,MSFT,NVDA       # Real data
python alphago_layering.py --synthetic --eval-only         # Pipeline wiring test
python alphago_layering.py --synthetic --verbose 2         # Full output
```

## Key Config Defaults (ArchitectureConfig in alphago_architecture.py)

```python
# L1: Alpha horizons
rl_alpha_horizon = 5                    # RL signal horizon (bars)
rl_logit_to_mu_scale = 0.10            # Scale: policy logits → mu
trend_fast_window = 21                  # ~1 month
trend_slow_window = 126                 # ~6 months
trend_horizon = 21
mr_lookback = 20                        # Bollinger lookback
mr_z_entry = 2.0                        # Z-score entry threshold
mr_horizon = 5
value_lookback = 252                    # 1 year valuation anchor
value_horizon = 63                      # ~3 months
carry_horizon = 21
seasonality_horizon = 5
vol_premium_horizon = 21

# L2: Ensemble
meta_learner_type = "ridge"
meta_learner_alpha = 1.0               # Ridge regularization
meta_learner_coef_cap = 5.0            # Max |coefficient| after fit
meta_learner_retrain_freq = 63         # Retrain every ~3 months
meta_learner_min_samples = 126         # Min obs before first train
max_single_alpha_weight = 0.30         # No alpha > 30%
shrinkage_toward_equal = 0.20          # 20% shrinkage toward 1/N

# L3: Portfolio construction
kelly_fraction = 0.25                   # Quarter-Kelly
vol_target = 0.15                       # 15% annualized
max_leverage = 1.0                      # No leverage
drawdown_start_pct = 0.08              # Start de-risking at 8% DD
drawdown_force_flat_pct = 0.15         # Force flat at 15% DD
regime_scale_map = {
    'high_growth_low_vol': 1.0,
    'high_growth_high_vol': 0.5,
    'low_growth_low_vol': 0.75,
    'low_growth_high_vol': 0.25,        # Crisis regime
}

# L4: Execution
no_trade_threshold_pct = 0.10          # 10% base dead zone (now regime-aware)
no_trade_vol_adaptive = True           # Scale threshold with vol
kill_max_drawdown = 0.15               # Force flat at 15% portfolio DD
kill_max_turnover_per_bar = 0.5        # Emergency turnover limit

# Environment
cash_yield_bps_annual = 400.0          # Risk-free rate on idle cash
bars_per_year = 252                    # Daily bars
```

IGNORE - 
  - "Feature count discipline: no more than sqrt(n_observations) features without strong priors or dimensionality reduction."