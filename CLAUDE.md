# CLAUDE.md — Alpha-Trade Project Context & Instructions

> **This file is the single source of truth for this project.**
> Claude Code must read and follow everything here. When in doubt, refer back to this file.

---

# PART 1: COMPLETE PROJECT INSTRUCTIONS (MANDATORY — ALWAYS FOLLOW)

You are an institutional-grade quantitative trading system architect and code reviewer.

## CORE OBJECTIVE

Design, review, and improve a modular, auditable, production-realistic quantitative trading system with strict layering, robust ensembles, and institutional risk/execution discipline. Prefer robustness, traceability, and realistic assumptions over cleverness.

**Performance Target:** Generate at least **1.5× SPY's CAGR** (50% more than SPY on a compounded annual basis) across a full market cycle, net of all execution costs, spreads, slippage, and fees. For example, if SPY delivers 10% CAGR over the backtest period, the system must deliver ≥ 15% CAGR. Every design decision — stock selection, signal weights, position sizing, risk controls — should be evaluated against this benchmark first.

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

9) All code must be production-grade:
   - No prototype-quality, throwaway, or "good enough for now" code. Every change must be written as if it will run in a live trading environment with real capital.
   - Vectorized numpy/pandas over Python for-loops wherever possible. Profile before optimizing, but never leave known O(n²) or per-bar Python loops in hot paths.
   - Proper error handling at system boundaries (I/O, network, data ingestion). Internal code should fail fast and loud, not silently swallow errors.
   - Memory-conscious: chunk large array operations, avoid unnecessary copies, clean up intermediate arrays in tight loops.
   - Thread/process-safe: no global mutable state in parallel code paths. Use ProcessPoolExecutor for CPU-bound work, not threads (GIL).
   - No dead code, no commented-out blocks, no TODO/FIXME left behind. If something is removed, remove it completely.
   - Type-stable numeric code: use explicit dtypes (float64 for prices/returns), guard against division by zero, NaN propagation, and overflow.

9) **Research → Critique → Discuss → Implement** (before directly implementing)
   - NEVER implement code changes without first researching best practices
   - ALWAYS critique proposed approaches against academic literature and standard practices
   - ALWAYS present research findings and trade-offs to user for discussion
   - ONLY implement after user approval and clear rationale established
   - Example: Before adding curriculum learning, research if it's appropriate for the use case
   - Rationale: Prevents incorrect implementations that need to be reverted (e.g., curriculum learning over datasets was wrong for PPO)

10) **Mandatory Parallelization — ALWAYS SATURATE ALL 16 CPU CORES**
    - The machine has 16 CPU cores. Every loop or batch operation that is computationally
      non-trivial MUST be parallelized. Sequential code is a performance bug.
    - Use `concurrent.futures.ThreadPoolExecutor` for I/O-bound and NumPy-heavy tasks
      (NumPy releases the GIL → real CPU overlap). Use `ProcessPoolExecutor` for pure-Python
      CPU-bound tasks where pickling cost is justified (e.g. walk-forward CV folds).
    - Default worker count: `max_workers = max(1, min(N, 16))` where N = number of independent
      tasks. Never hard-cap below 16 unless there is an explicit memory constraint documented
      inline.
    - Any new loop over datasets, symbols, sectors, ETFs, configurations, folds, or simulation
      runs MUST use a thread/process pool unless the loop body has fewer than ~10 iterations
      AND each iteration is trivially fast (<1 ms). Document the exception inline if skipping.
    - Scoring / ranking loops (e.g. `rank_universe`): use `_score_symbol_chunk` pattern —
      module-level worker function + `ThreadPoolExecutor` chunked submit + `extend` collect +
      single final `sort`. Worker functions must be pure (no shared mutable state).
    - ETF / file loading loops: always parallel — I/O latency hides behind concurrency for free.
    - Ablation / config-sweep loops: always parallel up to 16 workers.
    - Vectorize inner loops before parallelizing outer ones. Example: `_ema()` must use
      `pandas.Series.ewm()` (vectorized), not a Python `for` loop, before the per-sector
      outer loop is submitted to a pool.
    - When adding any new loop, the code review checklist (CODE REVIEW MODE item 9) must
      include: "Is this loop parallelized? If not, why not?"

11) **Hypothesis-Driven Sectoral Research — PRICE/VOLUME ONLY, BACKTESTED FROM 1990**
    - We have Norgate daily OHLCV data going back to 1990. This is the ONLY permitted data source
      for all sectoral churn and stock selection research. No fundamental data, no sentiment, no
      macro indicators, no external feeds. Pure price and volume.
    - The system must proactively generate hypotheses about HOW to detect sectoral churn and
      WHICH stocks benefit — then validate every hypothesis through historical backtesting
      on the full 1990–present dataset.
    - A "hypothesis" is NOT valid until it has a backtest result. Anecdote is not evidence.
    - Every proposed signal, indicator, or selection rule must specify:
        a) The economic hypothesis (WHY should this work?)
        b) The detection method (WHAT price/volume pattern signals it?)
        c) The backtest result across the full history since 1990 (DOES it actually work?)
        d) Performance across distinct regime periods (does it work in ALL regimes, or just one?)
    - Sectoral churn research must cover AT MINIMUM these regime periods:
        1990–1999: Tech/Growth bull market
        2000–2002: Tech bust, value outperforms
        2003–2007: Commodities/Energy supercycle, Financials peak
        2008–2009: GFC — Energy/Financials collapse, Utilities/Staples defend
        2010–2019: QE-era, low-vol Growth/Tech dominance
        2020: COVID crash and V-recovery — Healthcare, Tech lead
        2021: Reopening — Energy, Industrials, Financials surge
        2022: Rates shock — Energy only positive sector; Growth/Tech collapse
        2023–2024: AI supercycle — Tech/Comm Services rebound
    - Research outputs must answer:
        "Which OHLCV-only signals, computable at time T with data ≤ T, predicted sector rotation
         BEFORE it was obvious to the consensus?"
    - Permitted signal families (all OHLCV-derivable):
        Price momentum at multiple timeframes (1m, 3m, 6m, 12m)
        Relative strength of sector ETF vs SPY (multi-timeframe)
        RRG quadrant transitions (RS-Ratio × RS-Momentum — already in v9.0)
        Sector breadth (% stocks above 50d/200d SMA)
        Volume expansion on up-days vs down-days (institutional accumulation)
        52-week high breakouts within a sector (breadth of breakouts, not just price)
        ATR expansion (volatility awakening before a move)
        Inter-sector relative performance divergence (widening gap = rotation starting)
        New 52-week high count acceleration within a sector
    - Research must always separate "sector detection" from "stock selection within sector":
        Tier A: Which sector is about to lead? (sector-level signal)
        Tier B: Which stocks WITHIN that sector benefit first? (stock-level signal)
    - For Tier B (individual stock selection in churning sector), examine:
        Stocks near 52-week highs WITHIN sector before sector moves
        Stocks with volume accumulation ratio > 60% over 63 bars WITHIN sector
        Stocks with highest RS vs. sector ETF (intra-sector relative strength leaders)
        Stocks with lowest drawdown from 52-week high (resilience = quiet accumulation)
    - All backtests must report: hit rate, average return in subsequent 1m/3m/6m,
      Sharpe of a signal-triggered selection vs. random sector stock selection,
      and false positive rate (how often does the signal fire but sector does NOT rotate?)

## REQUIRED ARCHITECTURE (STRICT 5 LAYERS)

### L0 — Data Infrastructure & Integrity

Goal: provide clean, versioned, auditable data to all downstream layers. Data problems cause more blow-ups than bad models; this layer is foundational.

Requirements:
  - Point-in-time (PIT) database: no retroactive corrections visible to backtests. Every query returns what was known at that moment.
  - Survivorship bias handling: include delisted, dead, and merged assets in all historical universes.
    At any past date T, the backtest universe must contain EXACTLY the stocks that existed and were tradeable on date T — including stocks that later went bankrupt, were acquired, or delisted. Conversely, stocks that are top performers TODAY but did not exist or were illiquid at date T must NOT appear in the universe at date T. Testing only on today's survivors inflates returns by ~2-4% CAGR and hides the real cost of picking losers that disappeared.
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
9) **Verify parallelization (Absolute Rule #10):**
   - Identify every loop over datasets, symbols, sectors, folds, or configs.
   - Flag any that are sequential and non-trivial as "performance bug."
   - Confirm worker count uses `max(1, min(N, 16))` pattern.
   - Confirm inner loops (e.g. EMA, indicator computation) are vectorized before outer loops
     are parallelized.
10) Produce a prioritized upgrade list:
    - "Must fix for correctness" (bugs, leakage, lookahead)
    - "Must fix for performance" (sequential loops that should be parallel — Absolute Rule #10)
    - "Must fix for robustness" (missing constraints, no kill switches, no cost model)
    - "Should fix for production readiness" (logging gaps, no reconciliation, no alerting)
    - "Nice-to-have improvements" (additional alphas, better ensembling, UI/reporting)
11) Provide concrete code snippets for each recommended change (minimal patches when possible).
12) Be precise: do not claim features exist unless they are present in the code.
13) Call out all assumptions and "unknowns" explicitly.

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

| File | Lines (approx) | Role |
|---|---|---|
| `alphago_trading_system.py` | ~5,100 | v3.0 core: Config, TradingEnv, PPO, MCTS, features, discover_midcap_symbols() |
| `alphago_architecture.py` | ~5,200 | v6–v9 pipeline: L1–L4 classes + SelectionConfig, StockSelector, SectorRotationDetector |
| `alphago_layering.py` | ~4,300 | Launcher: v7/v8/v9 wiring, walk-forward, evaluation, charts, sector output |
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
| Stock Selection | `alphago_architecture.py` (SelectionConfig, StockSelector, SectorRotationDetector) | ~472-1100 |

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

### Performance Optimizations (Feb 2026) — 7 Hot-Path Fixes

Target: reduce full backtest runtime from ~1h20m to ~30-40 min.

**Fix P1: Parallel `evaluate_with_pipeline()` across symbols**
- `ThreadPoolExecutor` with up to 8 workers; each thread draws a `deepcopy(pipeline)` from a `Queue`
- NumPy releases the GIL enabling real CPU overlap
- **File:** `alphago_layering.py`

**Fix P2: Pre-concatenate training closes (O(1) SMA slice)**
- Build `_v8_sma_combined = np.concatenate([train_tail, test_closes])` once per symbol
- `pipeline.step()` slices `[:train_len + bar_idx + 1]` instead of calling `np.concatenate` every bar
- **File:** `alphago_architecture.py` (`pipeline.step()`)

**Fix P3: `_dq_score` → `_sym_dq_score` bug**
- Dataset quality score is stable; moved outside `while not done:` loop, renamed to avoid NameError
- **File:** `alphago_layering.py` (~line 396)

**Fix P4: Parallel ablation study**
- Ablation study parallelized with `ThreadPoolExecutor`; each thread gets `deepcopy(pipeline)`
- Results printed in deterministic order after `as_completed`
- **File:** `alphago_layering.py`

**Fix P5: Vectorized log returns**
- `log_rets[1:][_valid] = np.log(closes[1:][_valid] / closes[:-1][_valid])` replaces Python for-loop
- **File:** `alphago_layering.py`

**Fix P6: Vectorized cumulative returns**
- `_cs = np.cumsum(log_rets)`, then slice-based 5-bar and 15-bar windows replace O(n×w) Python loops
- **File:** `alphago_layering.py`

**Fix P7: Pre-compute normalized weights**
- `_w_mom`, `_w_trend`, `_w_rs`, `_w_invvol`, `_w_vacc`, `_w_h52w` computed once before the symbol loop
- **File:** `alphago_architecture.py` (`rank_universe()`)

### Sector/Symbol Output Fixes (Feb 2026)

**Sector chart `_1d` suffix mismatch**
- `SECTOR_MAP` uses plain symbols (`AAPL`); Norgate keys are `AAPL_1d`
- Fixed with `_bare_to_key = {k.split('_')[0]: k for k in per_sym}` reverse lookup
- **File:** `alphago_layering.py` (CHARTS section)

**Sector text table added**
- Companion text table alongside hbar chart showing: sector, total P&L, stock count, win rate
- Visible in CI/log files where terminal charts may not render
- **File:** `alphago_layering.py`

**Score Quartile vs Returns table**
- Groups selected stocks by composite score quartile (Top 25%, Upper-Mid, Lower-Mid, Bottom 25%)
- Shows Avg P&L, Win%, Avg Sharpe per quartile — validates that score rank predicts actual returns
- **File:** `alphago_layering.py`

**Factor Correlation table**
- Pearson correlation between each scoring factor and actual trade P&L
- Factors: momentum, SMA alignment, RS vs SPY, volatility, volume accumulation, 52w high proximity
- **File:** `alphago_layering.py`

---

## v8.0 Stock Selection Engine (FULLY IMPLEMENTED)

### Overview
Pre-pipeline stock selection layer that filters a 206-stock large-cap universe down to the highest-conviction names before the L1–L4 pipeline runs. The selection decision is pure price/volume — no RL or ML predictions involved.

Activated with `--version v8` or `--version v9`.

### SelectionConfig (`alphago_architecture.py`, ~line 472)

| Field | Default | Description |
|---|---|---|
| `top_n` | 15 | Fixed cap (only used when `--no-adaptive-n` passed) |
| `adaptive_n` | False | **Auto-True in v8/v9**: trade all stocks passing quality threshold |
| `min_score_pct` | 0.50 | Quality gate: select stocks within 50% of top score's magnitude |
| `momentum_lookback` | 252 | 12-month momentum window |
| `momentum_skip` | 21 | Skip last month (avoids short-term reversal noise) |
| `min_bars` | 63 | Minimum 3 months of data required |
| `sma_alignment_required` | True | Drop stocks not in uptrend (P > SMA50 > SMA100 > SMA200) |
| `volatility_cap_percentile` | 95.0 | Exclude top 5% most volatile stocks |
| `sector_momentum_gate` | False | Enable Tier 3 sector boost |
| `w_sector_momentum` | 0.10 | Max additive boost from sector momentum |
| `w_momentum` | 0.50 | Composite score: 12-1m momentum weight |
| `w_trend` | 0.15 | Composite score: SMA alignment weight |
| `w_rs` | 0.30 | Composite score: relative strength vs SPY weight |
| `w_invvol` | 0.05 | Composite score: inverse volatility weight |
| `w_volume_acc` | 0.00 | v9.0 Tier 4: volume accumulation weight (0=off) |
| `w_high52w` | 0.00 | v9.0 Tier 5: 52-week high proximity weight (0=off) |
| `use_rrg_rotation` | False | v9.0: use RRG ETF quadrant scores for Tier 3 |
| `rrg_fast_period` | 50 | v9.0: RRG fast EMA (~10 trading weeks) |
| `rrg_slow_period` | 200 | v9.0: RRG slow EMA (~40 trading weeks) |
| `sector_breadth_gate` | False | v9.0: require >50% of sector stocks above 50d SMA |
| `min_dollar_volume` | 0.0 | v9.0: min avg daily dollar volume (mid-cap liquidity gate) |

### Composite Scoring (rank_universe)

All weights are pre-normalized to sum to 1.0 before the symbol loop (Fix P7).

```
score = w_momentum  * momentum_12_1m
      + w_trend     * (sma_score / 3.0)          # SMA score is 0-3
      + w_rs        * rs_vs_spy_6m
      + w_invvol    * (inv_vol / 100.0)
      + w_volume_acc * (vol_acc - 0.5) * 2.0     # centered: >50% up-vol = positive
      + w_high52w   * high52w_proximity           # 1.0 = at 52w high
```

**Tier 3 Sector Boost** (when `sector_momentum_gate=True`):
```
score += w_sector_momentum * relative_boost
```
where `relative_boost` is RRG quadrant score (v9) or normalized mean 6-month sector return (v8).

### Adaptive N (Tier 2)

Default behavior in v8/v9: select ALL stocks above quality threshold.
```
threshold = top_score - abs(top_score) * (1 - min_score_pct)
effective_n = count(stocks where score >= threshold)
```
- Weak market: threshold is high relative to the pool → fewer stocks qualify
- Bull market: many stocks above threshold → larger portfolio automatically
- Override with `--no-adaptive-n --top-n N` for fixed-size portfolios

### Rank-Based Position Sizing

Selected stocks get position sizes proportional to their composite score rank. Top-ranked stock gets the largest allocation. Implemented via `pipeline.use_v8_sizing=True` and `pipeline._v8_rank` dict passed to the execution layer.

### Selection Log

Every `selector.select()` call appends to `selector.selection_log`:
- `selected`: list of symbols chosen
- `rankings`: full sorted list of (symbol, score, components) for all universe stocks
- `scores`: dict symbol → score
- `sector_allocation`: sector → count of selected stocks
- `effective_n`: actual N selected (key for adaptive N monitoring)
- `sector_momentum`: sector → momentum score (or RRG score)
- `rrg_metadata`: full RRG quadrant data per sector (when `use_rrg_rotation=True`)
- `sector_breadth`: sector → fraction of stocks above 50d SMA

---

## v9.0 Full Stack (FULLY IMPLEMENTED)

v9.0 = v8.0 + RRG sector rotation + volume/breakout factors + sector breadth gate + mid-cap universe.

All v9.0 features auto-enable when `--version v9` is passed. Individual flags can still override.

### Auto-Enabled Features in v9

| Feature | Auto-Value | Overridable |
|---|---|---|
| `--sector-momentum-gate` | True | `--no-sector-momentum-gate` (if added) |
| `--rrg-rotation` | True | `--no-rrg-rotation` (if added) |
| `--sector-breadth-gate` | True | pass flag explicitly to override |
| `--volume-acc-weight` | 0.10 | `--volume-acc-weight 0.0` to disable |
| `--high52w-weight` | 0.05 | `--high52w-weight 0.0` to disable |
| `--include-midcap` | True | `--no-include-midcap` (if added) |

### SectorRotationDetector (RRG)

Relative Rotation Graph framework (de Kempenaer) for leading-edge sector detection.

**Quadrant classification** (using RS-Ratio and RS-Momentum vs. SPY):
```
RS-Momentum > 100
      │
 ┌────┴────┐    ┌────────────┐
 │IMPROVING│───▶│  LEADING   │
 │(EARLY!) │    │(RIDE WAVE) │
 └────┬────┘    └────┬───────┘
      │              │
 ┌────┴────┐    ┌────┴───────┐
 │ LAGGING │◀───│ WEAKENING  │
 │ (AVOID) │    │(START EXIT)│
 └─────────┘    └────────────┘
      │
RS-Momentum < 100
```

**Quadrant scores** (used as sector boost multiplier):
- Leading: 1.00 (confirmed outperformance)
- Improving: 0.85 (BEST ENTRY — sector turning before consensus)
- Weakening: 0.30 (momentum leaving)
- Lagging: 0.00 (no boost)

**ETF mapping**: XLK (tech), XLF (fins), XLV (health), XLI (indus), XLE (energy), XLY (cons disc), XLP (cons staples), XLB (materials), XLU (utilities), XLRE (real estate), XLC (telecom)

**Combined score** (when both RRG and multi-TF RS enabled):
```
sector_score = 0.60 × RRG_quadrant_score + 0.40 × multitf_rs_norm_score
```

### Tier 4: Volume Accumulation

Fraction of total dollar volume traded on up-days over 63 bars.
```
vol_acc = sum(volume on up-days) / sum(all volume)
# >60% = institutional accumulation (strong buy signal)
# Centered: (vol_acc - 0.5) × 2.0 maps [0,1] → [-1,+1]
# 50% up-volume = neutral (zero contribution to score)
```

### Tier 5: 52-Week High Proximity

```
high52w_proximity = max(0, 1 - (high52w - price) / high52w)
# 1.0 = stock AT 52-week high (confirmed breakout)
# 0.0 = stock far below its 52-week high (value trap risk)
```

### Mid-Cap Rising Stars Universe

`discover_midcap_symbols()` in `alphago_trading_system.py` — dynamically scans the Norgate
`US_Equities` database at runtime using objective OHLCV-only filters (no hand-curation):
- Avg daily dollar volume $20M–$2B/day — proxy for ~$1B+ market cap
- Avg daily share volume >= 300K/day — float proxy, rules out low-float / spike-only names
- Close price >= $10 — eliminates sub-institutional / penny stocks
- History >= 252 bars (1 year minimum for reliable signal computation)
- Not already in `DEFAULT_SYMBOLS` (no large-cap duplication)

Tune with `--min-dollar-volume`, `--max-dollar-volume`, `--min-avg-volume`.

Activated via `--include-midcap`. Tune bounds with `--min-dollar-volume` / `--max-dollar-volume`.

Universe in v9: 206 large-caps + ~70 mid-caps = ~276 stocks (de-duplicated).
The same `StockSelector` composite scoring naturally filters weak mid-caps — only genuine
momentum/breakout mid-caps float above the quality threshold.

Dollar volume liquidity gate: `--min-dollar-volume 20000000` ensures $20M/day minimum
(institutional participation threshold) for mid-cap names.

**Rising Stars log**: when a mid-cap stock is selected, it is flagged in terminal output:
```
Rising Stars selected: CAVA, BROS, HIMS
```

---

## Version Summary

| Feature | `--version v7` | `--version v8` | `--version v9` |
|---|---|---|---|
| Stock selection (pre-pipeline filter) | Off | On | On |
| Adaptive N (all qualifying stocks) | Off | **On by default** | On |
| Sector momentum gate | Off | Opt-in | **On** |
| RRG quadrant sector scoring | Off | Opt-in | **On** |
| Sector breadth gate (>50% above 50d SMA) | Off | Opt-in | **On** |
| Volume accumulation (Tier 4) | Off | Opt-in | **0.10 weight** |
| 52-week high proximity (Tier 5) | Off | Opt-in | **0.05 weight** |
| Mid-cap Rising Stars universe extension | Off | Opt-in | **On** |
| Rank-based position sizing | Off | On | On |
| Universe size | `--n-symbols` | 206 | 206 + ~70 mid-cap |

**Key behavioral rule**: In v8/v9, `adaptive_n` is always `True` by default. The system trades ALL stocks meeting the composite score quality threshold — NOT a fixed top-N. Use `--no-adaptive-n --top-n N` to override back to fixed-size selection.

---

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

5. **v8/v9 always uses adaptive N.** Fixed top-N (e.g., top-15) is NOT the intended production
   behavior. The research conclusion is "trade all stocks meeting the quality threshold" — the
   market regime determines the portfolio size, not an arbitrary cap.

6. **v9 mid-cap extension is on by default.** The StockSelector's composite scoring is the
   self-cleaning filter — weak mid-caps score below the threshold and are naturally excluded.
   No manual curating needed at runtime.

---

## Norgate Data & Sectoral Churn Research Mandate

### Data Available

**Source:** Norgate Premium — daily OHLCV (Open, High, Low, Close, Volume), adjusted for splits and dividends.
**History:** 1990–present (35+ years).
**Universe:** US equities — ~8,000+ symbols including delisted stocks (survivorship-bias free).
**Sector ETFs available:** XLK, XLF, XLV, XLI, XLE, XLY, XLP, XLB, XLU, XLRE, XLC, SPY.
**Location:** Norgate parquet files, accessed via `alphago_layering.py` data loading block.

**CRITICAL CONSTRAINT:** All research, hypotheses, and selection signals must be derivable
from OHLCV data alone. No fundamentals, no earnings, no macro, no sentiment. Pure price + volume.

---

### Research Mandate: Detect Sectoral Churn Before Consensus

The core unsolved question driving v9.0+ research:

> **"Using only historical OHLCV data available at time T, what signals predict that sector X
> is about to become the market leader — before it is obvious to the consensus?"**

This must be answered empirically via backtesting from 1990. No hypothetical arguments.
Every candidate signal must produce a testable backtest result.

---

### Known Regime Periods for Backtesting (MANDATORY coverage)

Every proposed signal must be validated across ALL of these:

| Period | Dominant Rotation | Key Signal Opportunity |
|---|---|---|
| 1991–1999 | Tech/Growth → everything | Growth momentum, SMA breakouts |
| 2000–2002 | Tech bust → Value/Staples | RS divergence, breadth collapse in Tech |
| 2003–2007 | Energy/Materials supercycle | Commodity price momentum, volume expansion |
| 2008–2009 | GFC — Financials collapse | Breadth deterioration, vol expansion leading |
| 2010–2019 | QE era — Tech/Growth dominance | Low-vol, high-RS, 52w high density in Tech |
| 2020 Q1 | COVID crash — All sectors down | Volume spike, breadth collapse as early warning |
| 2020 Q2–Q4 | V-recovery — Healthcare, Tech | 52w high breakouts in Health/Tech before SPY |
| 2021 | Reopening — Energy, Industrials, Fins | RS acceleration in cyclicals vs SPY |
| 2022 | Rates shock — Energy only positive | XLE RS-ratio moving to Leading while XLK Lagging |
| 2023–2024 | AI supercycle — Tech/Comm Services | Volume accumulation in Semis/Cloud before move |

---

### Signal Families to Hypothesis-Test (OHLCV-only, all computable)

#### Tier A: Sector-Level Rotation Detection

| Signal | Hypothesis | Detection Method |
|---|---|---|
| RRG Quadrant Transition | Improving sectors are the leading-edge entry — they are gaining RS before consensus | RS-Ratio crosses 100 while RS-Momentum already > 100 (already in v9.0) |
| Multi-TF RS Acceleration | 1m RS > 3m RS → sector outperforming SPY more recently than before = momentum building | `rs_accel = rs_1m - rs_3m > 0` (already in v9.0) |
| Sector Breadth Expansion | When % of sector stocks above 50d SMA rises from <40% to >60% in 4 weeks, institutional rotation is underway | `breadth_now / breadth_4w_ago > 1.5` |
| 52-Week High Density Surge | Count of sector stocks hitting 52w highs accelerating → institutional buying pressure broadening | `new_highs_this_week / sector_size` trending up over 4 weeks |
| Volume Expansion on Up-Days | Sector-level up-day volume fraction rising → institutional accumulation preceding the price move | Sector aggregate: `sum(vol on up-days) / sum(all vol)` over 21 bars |
| ATR Expansion + Upward Bias | Volatility awakening with upward direction → sector waking up | `ATR_21 / ATR_63 > 1.3` AND `close > close[21]` |
| Low-Vol → High-Vol Transition | A sector going from quiet to active with upward price = accumulation complete, markup beginning | Rolling 10d vol / rolling 63d vol crossing above 1.2 |
| Intra-Sector RS Dispersion Collapse | When stocks in a sector start moving together (lower pairwise dispersion), an institutional theme is consolidating | Pairwise correlation of sector member 20d returns rising |

#### Tier B: Stock-Level Selection Within Rotating Sector

| Signal | Hypothesis | Detection Method |
|---|---|---|
| Intra-Sector RS Leader | Stocks beating the sector ETF WITHIN the rotating sector capture the most upside | Stock 6m return > sector ETF 6m return |
| Resilience Under Pressure | Stocks with smallest drawdown from 52w high during sector weakness = quiet accumulation = next leader | `(52w_high - price) / 52w_high < 5%` while sector is Lagging or Weakening |
| Volume Accumulation Leader | Stocks with highest up-day volume fraction within the sector = most institutional buying | `vol_acc > 0.65` (>65% of volume on up-days over 63 bars) |
| 52-Week High Breakout Pioneer | First stocks in a sector to make new 52w highs = leading indicators for the sector move | `close > max(close[-252:])` within sector before sector ETF makes new high |
| Low Drawdown + High Momentum | Stocks that held up best in the down phase AND have highest 6m momentum entering the up phase | `drawdown_from_52w < 10%` AND `momentum_6m > sector_avg_momentum_6m` |

---

### Research Output Format

For every hypothesis tested, document:

```
HYPOTHESIS: [Name]
SIGNAL: [Exact formula, computable from OHLCV at time T only]
ECONOMIC RATIONALE: [Why should this work? What behavior does it detect?]

BACKTEST RESULTS (1990–2024):
  Hit rate:        X% of signals followed by sector outperformance in next 63 bars
  Avg excess return (vs SPY, 63-bar):  +X%
  Sharpe of signal-triggered portfolio: X.XX
  False positive rate:  X% (signal fires but sector does NOT rotate in next 63 bars)
  Worst regime:    [period where signal failed most]
  Best regime:     [period where signal worked best]

REGIME BREAKDOWN:
  2003–2007 (commodities): hit_rate=X%, avg_return=+X%
  2010–2019 (QE/Tech):     hit_rate=X%, avg_return=+X%
  2022 (rates shock):      hit_rate=X%, avg_return=+X%
  ... (all mandatory periods)

VERDICT: KEEP / DISCARD / CONDITIONAL (works only in regime X)
IMPLEMENTATION: [If KEEP: which SelectionConfig field / weight to add]
```

---

### Research Pipeline (How to Generate and Test Hypotheses)

1. **Propose** — state the hypothesis in the format above (signal + economic rationale)
2. **Backtest script** — write a standalone analysis script in `d:\Experiments\Trading\` that reads Norgate parquet files directly and measures the signal's predictive power from 1990
3. **Report** — produce the Research Output Format table above for every signal tested
4. **Discuss** — present findings to user before any implementation changes
5. **Implement** — if approved, wire into `SelectionConfig` / `StockSelector` / `SectorRotationDetector`

**Backtest scripts naming convention:** `sector_research_<signal_name>.py`
**Never modify core system files during research phase** — research scripts are standalone and read-only against Norgate data.

---

### Current Research Status

| Signal | Status | Verdict |
|---|---|---|
| RRG Quadrant (RS-Ratio × RS-Momentum) | Implemented in v9.0 | Implemented — needs 1990 backtest validation |
| Multi-TF RS Acceleration (1m vs 3m) | Implemented in v9.0 | Implemented — needs 1990 backtest validation |
| Sector Breadth (% above 50d SMA) | Implemented in v9.0 | Implemented — needs 1990 backtest validation |
| Volume Accumulation (up-day vol fraction) | Implemented in v9.0 | Implemented — needs 1990 backtest validation |
| 52-Week High Proximity | Implemented in v9.0 | Implemented — needs 1990 backtest validation |
| 52-Week High Density Surge (sector-level) | Not yet tested | HYPOTHESIS — needs backtest |
| ATR Expansion + Upward Bias | Not yet tested | HYPOTHESIS — needs backtest |
| Intra-Sector RS Dispersion Collapse | Not yet tested | HYPOTHESIS — needs backtest |
| Resilience Under Pressure (drawdown filter) | Not yet tested | HYPOTHESIS — needs backtest |
| 52-Week High Breakout Pioneer (stock-level) | Not yet tested | HYPOTHESIS — needs backtest |

**Priority next research:** Validate all v9.0 implemented signals against full 1990–2024 history.
Then test the remaining HYPOTHESIS-status signals.

---

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
# --- v7.0 baseline ---
python alphago_layering.py --synthetic                    # Quick test (synthetic data)
python alphago_layering.py --symbols AAPL,MSFT,NVDA       # Real data, 3 stocks
python alphago_layering.py --synthetic --eval-only         # Pipeline wiring test
python alphago_layering.py --synthetic --verbose 2         # Full output

# --- v8.0: Stock Selection Engine ---
python alphago_layering.py --version v8 --skip-ablation
# → 206-stock large-cap universe
# → Adaptive N: trades all stocks passing quality threshold
# → Rank-based position sizing

# v8 with sector momentum boost
python alphago_layering.py --version v8 --sector-momentum-gate --skip-ablation

# v8 with mid-cap extension
python alphago_layering.py --version v8 --include-midcap --skip-ablation

# v8 with tighter quality gate
python alphago_layering.py --version v8 --min-score-pct 0.75 --skip-ablation

# v8 override: fixed top-20 instead of adaptive N
python alphago_layering.py --version v8 --no-adaptive-n --top-n 20 --skip-ablation

# --- v9.0: Full Stack (all features on by default) ---
python alphago_layering.py --version v9 --skip-ablation
# → 206 large-cap + ~70 mid-cap universe
# → RRG sector rotation + breadth confirmation
# → Volume accumulation (10%) + 52w high proximity (5%) factors
# → Adaptive N: all qualifying stocks

# v9 with custom weights
python alphago_layering.py --version v9 --volume-acc-weight 0.15 --high52w-weight 0.10 --skip-ablation

# v9 with stricter quality threshold
python alphago_layering.py --version v9 --min-score-pct 0.70 --skip-ablation

# --- Comparison mode (v7 baseline, all 4 columns in comparison table) ---
# Run WITHOUT --version v8/v9; the comparison table always includes a v8.0 (Sizing) column
python alphago_layering.py --symbols AAPL,MSFT,NVDA,GOOGL
```

## Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--version` | v7 | Strategy version: v7, v8, v9 |
| `--adaptive-n` | False | Tier 2: select all stocks passing quality threshold |
| `--no-adaptive-n` | False | Override v8/v9 default: revert to fixed `--top-n` cap |
| `--top-n` | 15 | Fixed cap (only active when `--no-adaptive-n` passed) |
| `--min-score-pct` | 0.50 | Adaptive N threshold: stocks within 50% of top score |
| `--sector-momentum-gate` | False | Tier 3: sector momentum boost to composite score |
| `--sector-momentum-weight` | 0.10 | Tier 3: max additive boost magnitude |
| `--volume-acc-weight` | 0.0 | v9 Tier 4: institutional accumulation factor weight |
| `--high52w-weight` | 0.0 | v9 Tier 5: 52-week high proximity factor weight |
| `--rrg-rotation` | False | v9: use RRG ETF quadrant scores for sector rotation |
| `--rrg-fast` | 50 | v9 RRG: fast EMA period in bars (~10 weeks) |
| `--rrg-slow` | 200 | v9 RRG: slow EMA period in bars (~40 weeks) |
| `--sector-breadth-gate` | False | v9: require >50% of sector stocks above 50d SMA |
| `--include-midcap` | False | v9: add mid-cap Rising Stars to universe |
| `--min-dollar-volume` | 0.0 | v9: minimum avg daily dollar volume ($) |
| `--skip-ablation` | False | Skip ablation study (saves ~10 min in v8/v9) |
| `--verbose` | 0 | Verbosity: 0=quiet, 1=info, 2=debug |
| `--synthetic` | False | Use synthetic price data (fast, no Norgate needed) |

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


---

# PART 3: WAVERIDER STRATEGY SYSTEM

## Overview

**WaveRider T5 MS BearVol2x** is a cross-sectional momentum rotation strategy independent of the AlphaGo L0–L4 pipeline. It operates on a completely separate codebase within the same repo.

**Name breakdown:**
- **T5** = Top-5 portfolio size
- **MS** = Meme Score (6-factor speculative filtering)
- **BearVol2x** = Leverage overlay: vol-target to 20% annualized (max 2x), reduced to 0.5x when SPY < SMA200

**Verified backtest (1991–2026):** CAGR +33.0%, Sharpe 1.01, MaxDD -59.6%

## Architecture

```
universe_builder.py (point-in-time rankings, survivorship-bias free)
        ↓
  [data_cache/universe_prices_top150.parquet]
  [data_cache/universe_rankings_top150.parquet]
        ↓
waverider.py ← core strategy (source of truth for all signal computation)
  ├─ compute_signals()      → Carhart momentum composite + vol
  ├─ compute_meme_scores()  → 6-factor speculative filtering
  ├─ backtest()             → full replay: NAV + holdings log + leverage
  └─ current_portfolio()    → PortfolioSignal (live recommendation)
        ↓
  ┌─────┴────────┐
  ↓              ↓
waverider_live.py    waverider_signal_bot.py
(CLI terminal)       (Telegram notifications)
```

## WaveRider File Map

| File | Lines | Role |
|------|-------|------|
| `waverider.py` | ~754 | Core strategy: config, scoring, backtesting, signal generation |
| `waverider_live.py` | ~445 | CLI signal generator with dollar allocations and trade instructions |
| `waverider_signal_bot.py` | ~549 | Automated Telegram bot: daily summaries + rebalance alerts |
| `universe_builder.py` | ~348 | Point-in-time universe builder with survivorship-bias elimination |
| `setup_scheduler.ps1` | ~37 | Windows Task Scheduler setup (S4U logon for headless execution) |

## Key Dataclasses

**WaveRiderConfig** (18 params):
- `top_n=5` — portfolio size
- `exit_band_mult=2.5` — hysteresis exit band (top 5×2.5 = top-12)
- `rebalance_freq=21` — trading days between rebalances (monthly)
- `weight_12m/6m/3m = 0.40/0.35/0.25` — Carhart momentum blend
- `meme_exclude/max1/max2 = 70/50/30` — graduated meme score thresholds
- `target_vol=0.20` — vol-targeting (20% annualized)
- `target_leverage=2.0` — max leverage multiplier
- `bear_leverage=0.5` — reduced leverage when SPY < SMA200
- `bear_sma=200` — SMA window for bear regime
- `universe_top_n=100` — universe construction size

**PortfolioSignal** (live output):
- `date`, `holdings` (UIDs), `holdings_clean` (symbols)
- `weights`, `meme_scores`, `leverage`, `bear_regime`
- `realized_vol`, `nav_lev`, `nav_unlev`
- `buys`, `sells`, `candidates` (top-20 ranked alternatives)

**BacktestResult**:
- `nav_leveraged` / `nav_unlevered` — equity curves
- `leverage_series`, `holdings_log`, `filtered_log`, `trades_log`
- `rebalance_dates`, `meme_scores`

## Meme Score (6-Factor Speculative Filter)

Quantifies speculative/risky characteristics (0–115 raw points):

| Factor | Max Pts | Threshold Tiers |
|--------|---------|-----------------|
| Volatility (vol_63d) | 25 | >40% → 8, >60% → 15, >80% → 20, >100% → 25 |
| Parabolic 3m move | 25 | >50% → 8, >100% → 18, >200% → 25 |
| Price-SMA200 stretch | 20 | >1.3x → 8, >2.0x → 15, >3.0x → 20 |
| Momentum concentration (1m/12m) | 15 | >0.3 → 5, >0.5 → 10, >0.8 → 15 |
| Volatility acceleration (vol_21d/vol_126d) | 15 | >1.0x → 5, >1.5x → 10, >2.5x → 15 |
| Universe tenure | 15 | ≤2 months → 15, 3–5 months → 8 |

**Classification:** Clean Growth ≤30 | Gray Zone 31–50 | Meme-Adjacent 51–70 | Pure Meme >70 (excluded)

## BearVol2x Leverage Overlay

```
if SPY < SMA200:
    leverage = bear_leverage (0.5x)  # bear gate: reduce exposure
else:
    realized_vol = 21d rolling annualized vol of portfolio
    leverage = min(target_vol / realized_vol, target_leverage)
    # Example: 20% target / 15% realized = 1.33x
    # Example: 20% target / 8% realized  = 2.0x (capped)
```

## Universe Builder

`universe_builder.py` builds a point-in-time, survivorship-bias-free universe:
- Scans ALL Norgate US Equities (active + delisted, ~8,000+ symbols)
- At each month-end, ranks alive stocks by trailing 63-day average dollar volume
- Filters out: warrants, units, rights, preferred shares, ETFs, leveraged ETFs
- Outputs top-150 as cached parquets (first run ~60-90s, subsequent <1s)
- Cache location: `Trading/data_cache/` (gitignored)

**Norgate data paths:**
- Active equities: `D:\Experiments\norgate_data\US_Equities\`
- Delisted equities: `D:\Experiments\norgate_data\US_Equities_Delisted\`
- Economic data: `D:\Experiments\norgate_data\Economic\` (includes 3-month T-bill %3MTCM)

## Slippage Analysis (Feb 2026)

Measured close-to-open slippage across 1,157 historical trades:
- **Buy cost:** +9.3 bps mean (slight gap-up overnight)
- **Sell benefit:** -26.9 bps mean (stocks being sold tend to gap up — mean-reversion overnight effect)
- **Net annual impact:** -56 bps/year (FAVORABLE — net benefit from overnight effects)
- **Conclusion:** Using close prices for signals and executing at next-day open is slightly beneficial, not harmful

---

# PART 4: DEPLOYMENT & OPERATIONS

## Telegram Signal Bot

**File:** `waverider_signal_bot.py`

**Setup:**
```bash
python waverider_signal_bot.py --setup    # Interactive first-time setup
```
Walks through: BotFather bot creation → token entry → auto chat_id detection → .env save → test message

**Daily operation:**
```bash
python waverider_signal_bot.py                    # run + send via Telegram
python waverider_signal_bot.py --capital 250000   # custom capital
python waverider_signal_bot.py --dry-run          # print only, don't send
```

**Message formats:**
- **Hold days:** Portfolio table with entry dates, P&L, leverage, next rebalance estimate
- **Rebalance days:** Full trade instructions (BUY/SELL with share counts and dollar amounts)
- **Format:** HTML with emoji icons, monospace `<pre>` tables, `<b>` bold headers

**Configuration:** `Trading/.env` (gitignored via root `**/.env` rule)
```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
CAPITAL=100000
```

**Error handling:**
- Signal generation failure → error alert sent via Telegram + logged
- Telegram send failure → retry once, then log locally
- All exceptions logged to `Trading/signal_log.txt`

## Windows Task Scheduler

**File:** `setup_scheduler.ps1` (run as Administrator)

```powershell
powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1
```

**Configuration:**
- Task name: `WaveRider-DailySignal`
- Schedule: Mon–Fri at 4:30 PM (local time, ~30 min after market close for Norgate data update)
- Logon type: S4U (Service for User — runs even when computer is locked, no stored password)
- Settings: `AllowStartIfOnBatteries`, `StartWhenAvailable` (runs on next boot if missed)

**Limitations:**
- Will NOT run if computer is powered off (but `StartWhenAvailable` catches up on next boot)
- For guaranteed delivery, consider cloud VM (AWS/Azure) or GitHub Actions

## Security Practices

- `.env` files are gitignored via root `**/.env` pattern — never committed
- Telegram bot tokens and chat IDs stored only in `.env`
- `signal_log.txt` is gitignored (contains portfolio positions)
- No API keys or credentials in any tracked Python file

---

# PART 5: LESSONS LEARNED & COMMON BUGS

## Survivorship Bias — The Silent Backtest Killer

**The problem:** If you backtest a momentum strategy using only today's stock universe (e.g., current S&P 500 members), you are implicitly selecting winners. The stocks in today's index are there BECAUSE they survived and grew. Stocks that went bankrupt (Enron, Lehman), were acquired (Sun Microsystems, Compaq), or simply faded (BlackBerry, Sears) vanish from your universe — but a real trader in 2005 would have seen them, ranked them, and potentially held them.

**Two-sided bias:**
1. **Dead stocks disappear:** You never see the losers that would have dragged your returns down. A momentum strategy in 2000 would have ranked Enron, WorldCom, and Tyco highly — then lost 100% when they imploded. Backtesting on survivors hides this.
2. **Future stars are invisible:** Today's leaders (NVDA, APP, CRDO) were small-caps or didn't exist 10 years ago. A backtest on today's universe pretends they were always available, front-running their entire growth arc. A real trader couldn't have bought NVDA at $5 in a large-cap momentum strategy — it wasn't in the universe yet.

**Impact:** Survivorship bias inflates backtested CAGR by ~2-4% and understates max drawdown. Momentum strategies are especially vulnerable because they chase winners — the exact stocks that survive.

**Our solution:** `universe_builder.py` uses Norgate's full database (active + delisted, ~8,000+ symbols) to reconstruct the point-in-time universe at each month-end. At date T, only stocks alive and liquid on date T appear. This is why the WaveRider backtest (CAGR +33%, MaxDD -59.6%) is credible — it includes every stock that existed, including the ones that later vanished.

**Rule:** NEVER backtest on a fixed list of today's symbols. Always use point-in-time universe construction. Any new strategy or research script must use `universe_builder.build_universe_cache()` or equivalent time-stamped filtering.

## Python Scoping Bug Pattern

**NORGATE_DIR UnboundLocalError** (fixed Feb 2026, commit `d8b490c`):
- `from module import X` inside a function causes Python to treat `X` as local throughout the ENTIRE function
- If `X` is referenced BEFORE the import line within the same function, Python raises `UnboundLocalError`
- Fix: Remove redundant local imports when names are already imported at module level
- **Location:** `alphago_layering.py` line 5634 had `from alphago_trading_system import load_from_norgate, NORGATE_DIR` inside `main()`, but `NORGATE_DIR` was used at line 3710

## Windows/Shell Gotchas

- **Git Bash mangles schtasks arguments:** `/create`, `/tn` etc. are interpreted as Unix paths. Workaround: wrap in `powershell.exe -Command "schtasks ..."`
- **Windows cp1252 terminal can't render emoji:** Wrap `print(emoji_text)` in try/except UnicodeEncodeError with UTF-8 fallback
- **schtasks `/change` for "run whether logged on or not"** requires interactive password prompt. Use PowerShell `New-ScheduledTaskPrincipal -LogonType S4U` instead (no password needed)

## Telegram Bot Tips

- Use `parse_mode="HTML"` (not Markdown) for reliable formatting — Markdown V2 requires escaping many characters
- Always include a plain-text fallback: if HTML parse fails, retry without `parse_mode`
- `<pre>` blocks work well for monospace tables in Telegram
- Max message length: 4096 characters — split if needed

---

# PART 6: COMPLETE FILE MAP (UPDATED)

## Core Strategy Files

| File | Lines | Role |
|------|-------|------|
| `alphago_trading_system.py` | ~5,100 | v3.0 core: Config, TradingEnv, PPO, MCTS, features, discover_midcap_symbols() |
| `alphago_architecture.py` | ~5,200 | v6–v9 pipeline: L1–L4 classes + SelectionConfig, StockSelector, SectorRotationDetector |
| `alphago_layering.py` | ~4,300 | Launcher: v7/v8/v9 wiring, walk-forward, evaluation, charts, sector output |
| `alphago_cost_model.py` | ~153 | Shared cost model: half-spread + sqrt-impact + fees |
| `alphago_mcts_parallel.py` | — | Parallel MCTS implementation |
| `alphago_new_alphas.py` | — | Additional alpha factories |
| `alphago_stop_loss.py` | — | Stop-loss logic |
| `waverider.py` | ~754 | WaveRider core: config, scoring, backtesting, signal generation |
| `waverider_live.py` | ~445 | WaveRider CLI signal generator |
| `waverider_signal_bot.py` | ~549 | WaveRider Telegram bot (daily automated alerts) |
| `universe_builder.py` | ~348 | Point-in-time universe builder (survivorship-bias free) |

## Infrastructure & Validation

| File | Lines | Role |
|------|-------|------|
| `data_quality.py` | ~1,220 | L0: quality scoring, missing data, schema validation |
| `validation_engine.py` | ~990 | Anti-overfitting: purged walk-forward CV, deflated Sharpe |
| `backtest_report.py` | ~550 | Reporting: metrics, equity curves, attribution |
| `run_artifacts.py` | ~250 | Reproducibility: saves configs, checkpoints, data snapshots |
| `table_formatter.py` | — | Table formatting utilities |

## Diagnostic & Research Tools

| File | Role |
|------|------|
| `_compare_all_strategies.py` | Cross-strategy performance comparison |
| `_cross_check_strategy.py` | Strategy cross-validation checks |
| `_diagnose_momentum.py` | Why momentum misses steady compounders |
| `_forensic_analysis.py` | Concentrated momentum forensics (meme damage, rank traces) |
| `_show_holdings.py` | Yearly holdings dump for concentrated momentum |
| `_show_taw_holdings.py` | Tactical All-Weather monthly allocations viewer |
| `_show_waverider_holdings.py` | WaveRider holdings viewer |
| `_verify_standalone.py` | Standalone strategy verification |
| `strategy_research.py` | General strategy research |
| `sector_compare.py` | Sector performance comparison |
| `sector_v7v8.py` | v7 vs v8 sector analysis |

## LETF Research Series

| File | Lines | Description |
|------|-------|-------------|
| `letf_research_comprehensive.py` | ~349 | Comprehensive LETF analysis |
| `letf_research_v2.py` | ~212 | LETF research iteration 2 |
| `letf_research_v3.py` | ~76 | LETF research iteration 3 |
| `letf_research_v4.py` | ~263 | LETF research iteration 4 |
| `letf_research_v5.py` | ~409 | LETF research iteration 5 |
| `letf_research_v6.py` | ~464 | LETF research iteration 6 |
| `letf_research_v7.py` | ~356 | LETF research iteration 7 |
| `letf_rotation_backtest.py` | ~189 | LETF rotation backtest |

## Deployment

| File | Role |
|------|------|
| `setup_scheduler.ps1` | Windows Task Scheduler setup (S4U logon, Mon-Fri 4:30 PM) |
| `.env` | Telegram credentials (GITIGNORED — never committed) |
| `.gitignore` | Excludes: data_cache, __pycache__, logs, HTML charts, .env, bootstrap scripts |

## Excluded / Archived

| File | Status |
|------|--------|
| `alphago_enhancements.py` | **EXCLUDED — do not modify or import** |
| `_backup/` | Archived versions, old docs, investigation scripts |

---

# PART N: WAVERIDER LEARNINGS & VERIFIED DECISIONS

> These are confirmed decisions from live testing and auditing. Do not revert without explicit user approval and a re-run of the audit script.

## Git Policy
- **Branch**: ALL changes go to `dev/new-machine-setup`. NEVER commit or push to master.
- Python environment: `C:\Users\Administrator\miniconda3\envs\alphatrade\python.exe`

## Verified Performance Baseline (as of 2026-03-01)
Produced by deterministic audit — two consecutive runs give identical NAV (diff = 0.00e+00).

| Metric | Value |
|--------|-------|
| CAGR (leveraged 2x) | +30.62% |
| Sharpe (0% rf) | 0.9531 |
| Sortino (correct formula) | 1.3858 |
| Max Drawdown | -53.18% |
| Avg Leverage | 1.17x |
| Total Return | +1,190,781% |
| Data period | 35.1 years (1991-01-08 to 2026-02-27) |
| Rebalances | 422 (EOM) |

If numbers differ from these, run the audit script before assuming the code changed. Numbers can shift between sessions if the price cache is rebuilt with new data.

## Rebalancing: EOM beats 21-Day Fixed Cycle
- **Decision**: `rebalance_eom = True` (default in `WaveRiderConfig`)
- **Evidence**: Backtested over 35 years. EOM gives CAGR +30.6% vs +24.2% for 21-day (+6.4pp), Sharpe 0.95 vs 0.81, better Sortino and lower MaxDD. Year-by-year: EOM wins 21 of 36 years, 21-day wins 13.
- **Why**: Monthly rankings data is computed at month-end boundaries. Rebalancing at EOM reads the signal at the exact boundary it was built for, eliminating mid-cycle noise.
- **Do not revert**: Unless a full re-backtest shows otherwise.

## Correct Sortino Formula
- **Wrong** (old code): `down = daily_r[daily_r < 0]; down.std() * sqrt(252)` — uses only negative-return days, underestimates downside dev, inflates Sortino.
- **Correct** (current code): `sqrt(mean(min(r, 0)^2) * 252)` — floors ALL daily returns at 0 and averages over ALL trading days. Industry-standard denominator.
- Applied in both `compute_nav_metrics()` in `waverider.py` and inline in `waverider_live.py`.

## SPY Benchmark Alignment
- **Never use `.bfill()`** after `.ffill()` when aligning SPY to strategy dates. `.bfill()` fills a missing period-start price with a future price = lookahead bias.
- Current code: `spy_price.reindex(p_dates).ffill().dropna()`

## pct_change — Always Use fill_method=None
- All calls to `.pct_change()` must pass `fill_method=None` to suppress FutureWarning and prevent silent forward-fill of NaN gaps before differencing.
- Applies to: `compute_nav_metrics()`, daily_r in `waverider_live.py`, `winner_capture_rate`.

## YTD Base Date
- Use the **last trading day of the prior year** as the YTD base price, not the first trading day of the current year.
- First trading day of current year already has that day's return baked into the NAV — using it as base excludes the first day's return from YTD.

## Universe Integrity — ETF Exclusion
- ETFs must be excluded from **both** the price matrix AND the rankings table.
- The `UNIVERSE_EXCLUDE` list in `universe_builder.py` controls this. Excluded tickers: GDX, GDXJ, SIL, SILJ, EWY, EWZ, EWJ, EWC, EWT, EWA, EWG, FXI, KWEB, CQQQ, ASHR, ARKK, ARKG, ARKW, ARKF, ARKQ, XOP, OIH, KRE, KBE, SMH, IBB (and all broad-market ETFs).
- After any universe rebuild, verify: `len([s for s in rankings['uid'].unique() if s in ETF_PATTERNS]) == 0`
- ETFs in the universe historically inflated CAGR by picking high-momentum ETFs (ARKK 2020, SMH 2023) that are not tradeable as momentum stocks.

## Auditing Methodology — Before Claiming Any Performance Number
1. Run two consecutive backtests — NAV diff must be 0.00e+00 (deterministic).
2. Verify metric formulas: manual formula vs `compute_nav_metrics()` — diff must be < 1e-8.
3. Verify rebalance dates: every date must be the last trading day of its month.
4. Verify holdings: every month must have exactly `top_n` stocks, no duplicates.
5. **Never assert a performance number without running this check first.**
6. If numbers differ from a previous session, possible causes in order of likelihood:
   a. Price cache was rebuilt with new data (more days = slightly different returns).
   b. Config changed (e.g., `rebalance_eom` toggled).
   c. Universe changed (ETF exclusions added/removed).
   d. Actual bug introduced — check git diff.

## Performance Attribution — Why Numbers Change Between Sessions
- Adding new data (rebuilding cache with fresher prices): minor CAGR drift, expected.
- Switching rebalancing mode (21-day -> EOM): +6.4pp CAGR, major.
- Excluding ETFs from universe: removes high-momentum ETF selections — can reduce CAGR if ETFs were top picks in some years.
- Sortino fix: does NOT affect CAGR, Sharpe, or MaxDD — only changes the Sortino column.
- SPY bfill fix: negligible impact on strategy returns (only affects benchmark comparison display).

## cash_remainder — Cache Price Lookups
- `get_current_price()` was called 2-3x per stock in the cash remainder calculation. Fixed by building `_price_cache` dict once per portfolio print. No correctness impact, only efficiency.

## find_entry_date() — Behaviour
- Walks backwards through rebalance dates to find the start of the most recent **continuous** holding streak. Correct by design. If a stock was held, dropped for one month, then re-entered — shows the more recent entry date. This is intentional.

## Backtest Calls in waverider_live.py
- `current_portfolio()` internally calls `backtest()` once.
- `main()` then calls `backtest()` again for the performance tables.
- This means the full backtest runs **twice** on every waverider_live.py run. If speed is a concern, cache the result.
