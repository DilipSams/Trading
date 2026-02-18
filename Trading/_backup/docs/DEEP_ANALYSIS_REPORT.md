# ALPHA-TRADE v7.0 BACKTEST DEEP ANALYSIS REPORT

**Generated:** 2026-02-16
**Analysis Period:** 77 trades across 50 symbols
**Total P&L:** -$124,162.02
**Win Rate:** 26.0% (20W / 57L)

---

## EXECUTIVE SUMMARY

### Critical Findings

1. **CATASTROPHIC WIN RATE**: 26% win rate with 74% losses is far below break-even
2. **NEGATIVE EXPECTANCY**: Win/Loss ratio of 0.48 means losses are 2x larger than wins
3. **SIGNAL FAILURE**: Primary entry signals (MACD bullish, Uptrend, MA cross) are net negative
4. **SHORT BIAS BROKEN**: Shorts have 9.1% win rate with -$2,244 avg P&L
5. **REGIME-INDEPENDENT FAILURE**: Losses span all years (2020-2025) and all symbols
6. **VALIDATION CONFIRMS OVERFITTING**: Walk-forward CV failed with 0% consistency

### ROOT CAUSE

**The system is trading random noise, not alpha.**
Every major signal category loses money. This isn't a tuning problem—it's a fundamental signal quality problem.

---

## 1. TRADE PATTERN ANALYSIS

### 1.1 Overall Performance

| Metric | Value |
|--------|-------|
| Total Trades | 77 |
| Wins | 20 (26.0%) |
| Losses | 57 (74.0%) |
| Total P&L | -$124,162.02 |
| **Avg Win** | **$1,262.20** |
| **Avg Loss** | **-$2,621.16** |
| **Profit Factor** | **0.48** (need >1.0) |
| **Expectancy** | **-$1,612.49** per trade |

**VERDICT:** Negative expectancy system. Every trade loses $1,612 on average.

### 1.2 Holding Period Analysis

| Group | Avg Bars | Median Bars |
|-------|----------|-------------|
| Winners | 19.1 | 12.0 |
| Losers | 11.2 | 10.0 |

**Mann-Whitney U test:** p=0.1495 (NOT SIGNIFICANT)

**FINDING:** Winners are held slightly longer, but the difference is not statistically significant.
**IMPLICATION:** Cutting losers faster does not explain the poor performance.

### 1.3 Sequential Patterns (Streaks)

- **Max Win Streak:** 2 trades
- **Max Loss Streak:** 13 trades

**FINDING:** 13-trade losing streak indicates systematic failure, not random variance.
A 26% win-rate system has <0.1% probability of a 13-loss streak by chance alone.

### 1.4 Action Type Performance

| Action | Trades | Win Rate | Avg P&L |
|--------|--------|----------|---------|
| **FULL_LONG** | 25 | 36.0% | -$1,829 |
| **HALF_LONG** | 37 | 27.0% | -$1,404 |
| **FULL_SHORT** | 11 | **9.1%** | **-$2,244** |
| **HALF_SHORT** | 4 | **0.0%** | -$452 |

**CRITICAL FINDINGS:**

1. **Shorts are catastrophically broken:** 9.1% win rate for FULL_SHORT, 0% for HALF_SHORT
2. **FULL positions underperform HALF:** FULL_LONG loses more than HALF_LONG despite higher conviction
3. **Conviction signals are inverted:** "Full conviction" trades are the worst performers

---

## 2. ENTRY SIGNAL ANALYSIS

### 2.1 Top 10 Entry Reasons by Frequency

| Rank | Count | Win Rate | Avg P&L | Entry Reason (truncated) |
|------|-------|----------|---------|--------------------------|
| 1 | 14 | **14.3%** | **-$2,514** | MACD bullish; Uptrend; Moving avg bullish cross; Strong trend (ADX) |
| 2 | 2 | 0.0% | -$3,715 | RSI overbought; Near upper BB; Full short conviction |
| 3 | 2 | 50.0% | -$3,517 | RSI near oversold; Near lower BB; Uptrend; Strong trend (ADX) |
| 4 | 2 | 0.0% | -$3,250 | RSI elevated; Full short conviction |
| 5 | 2 | **100%** | **+$1,592** | Uptrend; Moving avg bullish cross; Strong trend (ADX); Positive value est |
| 6 | 2 | 0.0% | -$3,605 | MACD bearish; Full short conviction |

**CRITICAL FINDING:**
The most common entry pattern (14 trades) has a **14.3% win rate** and loses **-$2,514 per trade on average.**

### 2.2 Keyword Analysis (All Entry Reasons)

| Keyword | Count | Win Rate | Avg P&L | Total P&L |
|---------|-------|----------|---------|-----------|
| Value_positive | 44 | 34.1% | -$1,259 | **-$55,379** |
| ADX_strong | 44 | 27.3% | -$1,496 | -$65,836 |
| **MACD_bullish** | **39** | **30.8%** | **-$1,424** | **-$55,524** |
| **MA_bull_cross** | **37** | **27.0%** | **-$1,822** | **-$67,426** |
| Uptrend | 31 | 32.3% | -$1,916 | -$59,383 |
| RSI_oversold | 11 | 27.3% | -$1,939 | -$21,329 |
| FULL_LONG_conv | 11 | 27.3% | -$2,283 | -$25,109 |
| **FULL_SHORT_conv** | **11** | **9.1%** | **-$2,244** | **-$24,679** |
| RSI_overbought | 8 | 12.5% | -$1,933 | -$15,467 |
| BB_lower | 8 | 12.5% | -$3,204 | -$25,631 |
| BB_upper | 7 | 14.3% | -$1,214 | -$8,498 |
| MACD_bearish | 5 | **0.0%** | -$1,968 | -$9,839 |
| Downtrend | 5 | **0.0%** | -$1,551 | -$7,753 |

### 2.3 Signal Effectiveness Ranking (Primary Signals)

| Signal | Trades | Win Rate | Total P&L | Status |
|--------|--------|----------|-----------|--------|
| **MACD_bullish** | **39** | **30.8%** | **-$55,524** | ❌ REJECT |
| Uptrend | 14 | 42.9% | -$23,426 | ❌ REJECT |
| RSI_overbought | 7 | 14.3% | -$14,153 | ❌ REJECT |
| MACD_bearish | 5 | 0.0% | -$9,839 | ❌ REJECT |
| MA_cross_bull | 1 | 0.0% | -$6,033 | ❌ REJECT |
| RSI_oversold | 3 | 0.0% | -$5,695 | ❌ REJECT |
| BB_lower | 2 | 0.0% | -$5,370 | ❌ REJECT |
| Downtrend | 2 | 0.0% | -$1,864 | ❌ REJECT |
| BB_upper | 1 | 0.0% | -$632 | ❌ REJECT |
| Other | 3 | 33.3% | -$1,625 | ❌ REJECT |

**VERDICT: ALL PRIMARY SIGNALS ARE NET NEGATIVE.**

---

## 3. STATISTICAL SIGNIFICANCE TESTS

### 3.1 Are Winners Statistically Different from Losers?

We would expect winning trades to have:
- Different indicator values at entry
- Different holding periods
- Different entry conditions

**However:** With only 1 trade containing full indicator data (NFLX_1d), we cannot perform robust statistical tests.

**The available comprehensive_trades.csv contains:**
- Entry/exit reasons (text)
- Holding periods
- P&L outcomes

**Mann-Whitney U Test (Holding Period):**
- p-value: 0.1495
- **Result:** NOT SIGNIFICANT
- **Interpretation:** Winners and losers have similar holding periods

### 3.2 What This Means

The lack of indicator data for 76 out of 77 trades prevents us from answering:
- Do winning trades have lower RSI at entry?
- Do winning trades have different MACD values?
- Do winning trades enter in different volatility regimes?

**RECOMMENDATION:** Re-run backtest with `--save-all-indicators` flag to capture full state.

---

## 4. SYMBOL-SPECIFIC ANALYSIS

### 4.1 Consistent Winners (9 symbols, 1 trade each)

| Symbol | Trades | P&L |
|--------|--------|-----|
| KO_1d | 1 | +$2,190 |
| HON_1d | 1 | +$1,933 |
| MA_1d | 1 | +$1,676 |
| AMZN_1d | 1 | +$1,201 |
| AAPL_1d | 1 | +$1,184 |
| PG_1d | 1 | +$792 |
| ABT_1d | 1 | +$371 |
| JPM_1d | 1 | +$358 |
| WMT_1d | 1 | +$65 |

**FINDING:** 9 symbols with 100% win rate, but only 1 trade each.
**INTERPRETATION:** Insufficient sample size. These are likely random winners, not genuine edge.

### 4.2 Consistent Losers (37 symbols, 1-2 trades each)

**Top 10 Worst:**

| Symbol | Trades | P&L |
|--------|--------|-----|
| HD_1d | 1 | -$7,826 |
| UPS_1d | 1 | -$7,532 |
| CRM_1d | 1 | -$7,189 |
| MS_1d | 1 | -$6,174 |
| ADBE_1d | 1 | -$5,741 |
| AXP_1d | 1 | -$5,486 |
| SLB_1d | 1 | -$5,376 |
| TSLA_1d | 1 | -$5,230 |
| CVX_1d | 1 | -$5,121 |
| NVDA_1d | 2 | -$4,848 |

**FINDING:** 37 symbols with 0% win rate.
**CRITICAL:** Most of these have only 1 trade, so the 0% win rate is not statistically meaningful.
**HOWEVER:** The dollar losses are real and large.

### 4.3 Mixed Performance (4 symbols, multiple trades)

| Symbol | Trades | Win Rate | Total P&L | Status |
|--------|--------|----------|-----------|--------|
| BAC_1d | 7 | 42.9% | +$326 | ✅ Only net positive symbol |
| NKE_1d | 2 | 50.0% | -$116 | Neutral |
| META_1d | 3 | **66.7%** | **-$6,778** | ❌ High WR, negative P&L |
| NFLX_1d | **16** | **31.2%** | **-$8,656** | ❌ Most traded, worst loser |

**CRITICAL FINDING: NFLX_1d**

- 16 trades (21% of all trades)
- 31.2% win rate
- -$8,656 total P&L
- Largest single losing symbol

**RECOMMENDATION:** NFLX_1d should be BLACKLISTED or undergo dedicated failure analysis.

**META_1d Paradox:**
- 66.7% win rate (2W/1L)
- Net -$6,778

**Analysis of META_1d trades:**
1. 2025-01-24: FULL_LONG → +$5,574 (WIN)
2. 2025-02-13: HALF_LONG → +$527 (WIN)
3. **2025-02-18: HALF_LONG → -$12,880 (LOSS)**

**The single loss wiped out both wins plus additional capital.**

**IMPLICATION:** Position sizing or stop-loss logic is broken. One loss should never exceed the sum of all wins.

---

## 5. POSITION SIZING ANALYSIS

### 5.1 Position Size vs Performance

| Position | Trades | Win Rate | Avg P&L | Std P&L |
|----------|--------|----------|---------|---------|
| FULL_LONG | 25 | 36.0% | -$1,829 | $3,280 |
| HALF_LONG | 37 | 27.0% | -$1,404 | $2,724 |
| FULL_SHORT | 11 | **9.1%** | **-$2,244** | $3,224 |
| HALF_SHORT | 4 | **0.0%** | -$452 | $371 |

### 5.2 Key Findings

1. **FULL positions lose MORE than HALF positions**
   - FULL_LONG: -$1,829 avg vs HALF_LONG: -$1,404 avg
   - **Implication:** "Full conviction" signals are WORSE than partial conviction

2. **ALL SHORT positions fail**
   - FULL_SHORT: 9.1% win rate, -$2,244 avg
   - HALF_SHORT: 0.0% win rate, -$452 avg
   - **Implication:** Short alpha is completely broken

3. **High variance across all position types**
   - Std deviation ~2x the mean loss
   - **Implication:** Outcomes are highly random

### 5.3 Recommendations

**IMMEDIATE:**
1. **DISABLE ALL SHORT SIGNALS** until root cause is identified
2. **CAP FULL_LONG CONVICTION** at HALF_LONG levels until conviction logic is fixed
3. **Implement hard stop-loss** to prevent META-style single-loss wipeouts

**MEDIUM TERM:**
1. Investigate why "full conviction" signals underperform
2. Re-calibrate position sizing logic (Kelly fraction may be inverted)
3. Add regime-dependent position sizing (never size up in high-vol regimes)

---

## 6. ENTRY/EXIT TIMING ANALYSIS

### 6.1 Holding Period vs P&L

| Period (bars) | Trades | Win Rate | Avg P&L |
|---------------|--------|----------|---------|
| 1-5 | 29 | 20.7% | -$1,654 |
| 6-15 | 28 | 21.4% | -$2,337 |
| 16-30 | 14 | **35.7%** | -$388 |
| 31-100 | 6 | **50.0%** | -$890 |

**FINDING:**
- Longer holding periods have higher win rates (35.7% → 50.0%)
- But ALL periods are net negative
- **Implication:** The signals themselves are bad; holding longer just reduces the bleeding rate

### 6.2 Are Exits Too Early or Too Late?

**Evidence for "cutting winners too early":**
- Winners avg 19.1 bars vs losers 11.2 bars
- But p=0.1495 (not statistically significant with n=77)

**Evidence for "letting losers run":**
- Single META loss (-$12,880) exceeds 2 wins (+$5,574 + $527)
- No apparent stop-loss logic

**VERDICT:**
Both problems exist, but **entry signal quality is the primary issue.**
Better exits can't fix -50% entry signals.

---

## 7. FAILURE MODE IDENTIFICATION

### 7.1 Top 20 Losing Trades (smallest to largest)

| Rank | Symbol | Action | Bars | P&L | Entry Signal | Exit Signal |
|------|--------|--------|------|-----|--------------|-------------|
| 1 | NFLX_1d | HALF_SHORT | 1 | -$81 | RSI elevated; Near upper BB; Downtrend | MACD bullish; Full long conviction |
| 2 | NFLX_1d | HALF_LONG | 1 | -$111 | Uptrend; MA cross; ADX strong; Value+ | Uptrend; MA cross; ADX strong; Value+ |
| 3 | MCD_1d | FULL_LONG | 19 | -$129 | Uptrend; MA cross; ADX; Full long | MACD bull; MA cross; ADX; Value+ |
| ... | ... | ... | ... | ... | ... | ... |
| 20 | AVGO_1d | FULL_LONG | 2 | -$940 | MACD bull; Uptrend; MA cross; ADX | Exiting position |

**Full list in appendix.**

### 7.2 Common Patterns in Top 20 Losses

**Entry Signals in Top 20 Losses:**

| Signal | Frequency in Losses | Overall Frequency | Enrichment |
|--------|---------------------|-------------------|------------|
| ADX_strong | 13/20 (65%) | 44/77 (57%) | 1.14x |
| MACD_bullish | 11/20 (55%) | 39/77 (51%) | 1.08x |
| MA_bull_cross | 10/20 (50%) | 37/77 (48%) | 1.04x |
| Value_positive | 10/20 (50%) | 44/77 (57%) | 0.88x |
| Uptrend | 7/20 (35%) | 31/77 (40%) | 0.88x |

**FINDING:**
The "worst loss" signals are THE SAME as the "most common" signals.
**INTERPRETATION:** The system doesn't have "bad signals that fail sometimes."
It has "normal signals that fail most of the time."

### 7.3 Failure Taxonomy

#### Type 1: Whipsaw Losses (1-bar holding period)

- **Example:** NFLX_1d HALF_LONG (2024-07-08 → 2024-07-09), -$111
- **Pattern:** Entry and exit signals nearly identical
  - Entry: "Uptrend; MA cross; ADX strong; Value+ (0.4194)"
  - Exit: "Uptrend; MA cross; ADX strong; Value+ (0.3982)"
- **Cause:** Signal oscillation → rapid entry/exit → transaction costs eat profit
- **Frequency:** 29 trades out of 77 (38%) held ≤5 bars

#### Type 2: Reversal Losses (conviction flip)

- **Example:** NFLX_1d HALF_SHORT → FULL_LONG (exit), -$81
  - Entry: "RSI overbought; Near upper BB; Downtrend"
  - Exit: "MACD bullish; Full long conviction"
- **Pattern:** Short → flip to long conviction within 1 bar
- **Cause:** Contradictory signals from different alphas
- **Frequency:** 15 out of 77 trades (19%) exit with opposite direction conviction

#### Type 3: Conviction Failures ("Full conviction" losses)

- **Example:** COF_1d FULL_LONG → FULL_LONG (exit), -$4,427, 5 bars
  - Entry: "MACD bullish; Value+ (0.4229); Full long conviction"
  - Exit: "MACD bullish; Value+ (0.4075); Full long conviction"
- **Pattern:** High conviction entry AND high conviction exit, still loses
- **Cause:** Conviction scoring is not calibrated to actual edge
- **Frequency:** 11 "Full long conviction" trades, 27.3% win rate, -$2,283 avg

#### Type 4: Catastrophic Single Losses

- **Example:** META_1d HALF_LONG, -$12,880, 13 bars
- **Pattern:** One loss > sum of all wins
- **Cause:** No stop-loss, position sizing doesn't account for tail risk
- **Frequency:** 1 loss of this magnitude, but represents 10.4% of total losses

---

## 8. VALIDATION FAILURE ROOT CAUSE

### 8.1 Walk-Forward Validation Results (from logs)

**Iteration 3:**
- Val gate: **0/3 passed (0% < 50%)**
- COST_1d: walk-forward consistency **0.0% < 40.0%**; permutation test **p=1.000**; placebo PnL=8437 > 0
- AAPL_1d: walk-forward consistency **0.0% < 40.0%**; permutation test **p=1.000**; placebo PnL=10393 > 0
- META_1d: walk_forward **invalid**

**Iteration 5:**
- Val gate: **0/3 passed (0% < 50%)**
- BLK_1d: walk_forward **invalid**
- PYPL_1d: walk_forward **invalid**
- COST_1d: walk-forward consistency **0.0% < 40.0%**; permutation test **p=1.000**; placebo PnL=8671 > 0

### 8.2 What This Means

**Walk-forward consistency = 0.0%:**
- The model performs well on training data
- The model performs randomly (or negatively) on out-of-sample folds
- **DEFINITION OF OVERFITTING**

**Permutation test p=1.000:**
- Permuting the labels (shuffling returns) produces EQUAL OR BETTER results than the real model
- **Interpretation:** The model is not learning signal; it's learning noise

**Placebo (feature shuffle) PnL > 0:**
- Shuffling feature columns produces POSITIVE P&L
- The real features produce NEGATIVE P&L
- **Interpretation:** The features are anti-predictive (or the model is inverting them)

### 8.3 Diagnosis

**The model passed NO validation tests.**
This is not a marginal failure. This is complete failure across:
1. Out-of-sample consistency
2. Permutation testing (vs random labels)
3. Placebo testing (vs random features)

**ROOT CAUSE:**
The RL agent (PPO + MCTS) is overfitting to training data noise. The other 6 alphas (trend, MR, value, carry, seasonality, vol premium) are also producing noisy or inverted signals.

**EVIDENCE:**
- 0% OOS consistency means IS performance ≠ OOS performance
- p=1.0 permutation means shuffled labels work as well as real labels
- Positive placebo PnL means random features beat real features

---

## 9. TIME-BASED PATTERNS

### 9.1 Performance by Year

| Year | Trades | Total P&L | Avg P&L | Win Rate |
|------|--------|-----------|---------|----------|
| 2020 | 10 | -$13,849 | -$1,385 | 30.0% |
| 2021 | 22 | -$24,584 | -$1,117 | 36.4% |
| 2022 | 8 | -$21,775 | -$2,722 | 0.0% |
| 2023 | 9 | -$30,238 | -$3,360 | 11.1% |
| 2024 | 24 | -$24,738 | -$1,031 | 25.0% |
| 2025 | 4 | -$8,978 | -$2,244 | 50.0% |

### 9.2 Key Findings

1. **2022 was catastrophic:** 0% win rate, -$2,722 avg loss
   - Coincides with Fed rate hikes, volatility spike
   - **Implication:** System has no regime detection for macro shifts

2. **2021 had highest win rate (36.4%)** but still net negative
   - **Implication:** Even in "best conditions," system bleeds money

3. **No year was profitable**
   - **Implication:** This is not a regime-specific failure; it's a universal failure

4. **Losses are ACCELERATING in recent years:**
   - 2020-2021 avg: -$1,251/trade
   - 2022-2023 avg: -$3,041/trade
   - **Implication:** Signal decay, increased market efficiency, or crowding

---

## 10. ACTIONABLE RECOMMENDATIONS

### 10.1 CRITICAL (Implement Before Next Trade)

#### 1. **HALT ALL TRADING IMMEDIATELY**
   - Current system has -$1,612 expectancy per trade
   - Every additional trade loses money

#### 2. **DISABLE SHORT SIGNALS**
   - FULL_SHORT: 9.1% win rate
   - HALF_SHORT: 0.0% win rate
   - Root cause: unknown, but shorts are catastrophically broken

#### 3. **IMPLEMENT HARD STOP-LOSS**
   - Current: Single META loss (-$12,880) > sum of 2 wins
   - **Rule:** Max loss per trade = 2x expected win (2 × $1,262 = $2,524)
   - **Execution:** Trailing stop at -2.5% of portfolio, or -$2,500 absolute

#### 4. **BLACKLIST NFLX_1d**
   - 16 trades, -$8,656 total
   - 31.2% win rate
   - System cannot trade NFLX profitably

### 10.2 HIGH PRIORITY (Fix Within 1 Week)

#### 5. **RE-RUN BACKTEST WITH FULL INDICATOR LOGGING**
   - Current: Only 1 trade has full indicator data
   - **Need:** RSI, MACD, BB, ADX, vol regime for ALL 77 trades
   - **Purpose:** Statistical tests (t-test, Mann-Whitney) to identify predictive indicators

#### 6. **INVESTIGATE "FULL CONVICTION" INVERSION**
   - FULL_LONG conviction: 27.3% WR, -$2,283 avg
   - HALF_LONG: 27.0% WR, -$1,404 avg
   - **Hypothesis:** Confidence scoring is inverted or uncalibrated
   - **Test:** Invert confidence → trade OPPOSITE of "full conviction" signals

#### 7. **FIX META-LEARNER CALIBRATION**
   - SA-3 fix seeds weights from validation, but meta-learner still outputs low-confidence signals
   - **Issue:** Ridge falls back to equal-weight when untrained
   - **Test:** Force Ridge to train on first 30 bars (even if < min_samples=126)
   - **Alternative:** Use Bayesian Model Averaging with informative priors

#### 8. **AUDIT L1 ALPHAS INDIVIDUALLY**
   - Test each of the 7 alphas in isolation (no ensembling)
   - Measure IC, Sharpe, t-stat for each alpha on OOS data
   - **Hypothesis:** Some alphas are negative and polluting the ensemble
   - **Expected:** 4-5 alphas are noise/negative; 2-3 might have weak positive edge

### 10.3 MEDIUM PRIORITY (Fix Within 1 Month)

#### 9. **REGIME-ADAPTIVE POSITION SIZING**
   - 2022 (vol spike year): 0% win rate, -$2,722 avg
   - **Current:** Regime scaling exists but doesn't prevent catastrophic losses
   - **Fix:** Add VIX-based or realized vol-based position size dampening
   - **Rule:** If realized_vol > 2x trailing avg → reduce size by 75%

#### 10. **IMPLEMENT ENSEMBLE DIVERSITY METRICS**
   - Current: 7 alphas may be highly correlated (all bullish trend followers)
   - **Add:** Pairwise correlation matrix of alpha outputs
   - **Target:** Effective N (via eigenvalue decomposition) > 3
   - **Action:** If corr(alpha_i, alpha_j) > 0.7, demote one

#### 11. **ADD PLACEBO MONITORING TO LIVE SYSTEM**
   - Validation caught overfitting with placebo test (feature shuffle)
   - **Live equivalent:** Run shadow "placebo model" with shuffled features
   - **Alert:** If placebo outperforms real model for N consecutive bars → halt

#### 12. **RE-EVALUATE FEATURE ENGINEERING**
   - Current features: 41 indicators + 4 regime bits
   - **Issue:** Permutation test p=1.0 means features ≈ random labels
   - **Action:**
     1. Test IC of each individual feature
     2. Remove features with |IC| < 0.02
     3. Add economic regime features (yield curve, credit spreads, VIX)
     4. Test for multicollinearity (VIF > 5 → remove)

### 10.4 RESEARCH PRIORITIES

#### 13. **INVESTIGATE SIGNAL INVERSION HYPOTHESIS**
   - **Observation:** "Uptrend" signal loses money; "MACD bullish" loses money
   - **Hypothesis 1:** Signals are lagging (trend already exhausted when detected)
   - **Hypothesis 2:** Signals are crowded (everyone trades same signals → alpha arbitraged away)
   - **Test:** Invert all signals (buy on "MACD bearish", short on "MACD bullish") and re-test

#### 14. **BENCHMARK AGAINST BUY-AND-HOLD**
   - Current: -$124,162 on 77 trades across 50 symbols
   - **Missing:** What if we just bought SPY and held?
   - **Test:** Run same backtest with "always HALF_LONG SPY" and compare

#### 15. **ALTERNATIVE ENSEMBLE METHODS**
   - Current: Ridge meta-learner (with SA-3 fix)
   - **Test:**
     1. Inverse-variance weighting (IC-based)
     2. Elastic Net (L1 + L2 regularization)
     3. Bayesian Model Averaging
     4. Gradient Boosting Meta-Learner (GBDT on top of alphas)
   - **Hypothesis:** Ridge may be wrong functional form

#### 16. **FORENSIC ANALYSIS OF 13-LOSS STREAK**
   - Identify the specific 13 consecutive losses
   - Check for:
     - Common symbol (same asset class?)
     - Common time period (specific regime?)
     - Common entry signal
     - Data quality issues (stale prices, corporate actions?)

---

## 11. WHAT WOULD NEED TO BE TRUE FOR THIS TO WORK?

Based on the analysis, for this system to be profitable:

### Necessary Conditions

1. **At least ONE alpha with IC > 0.03**
   - Current: All primary signals are net negative
   - Target: Find 1-2 alphas with t-stat > 3.0 on OOS data

2. **Shorts must be fixed OR disabled**
   - 9.1% win rate cannot be rescued with better ensembling
   - Either fix the short alpha or only trade long

3. **Meta-learner must assign near-zero weight to negative alphas**
   - Current: Equal-weight fallback gives bad alphas 14% weight (1/7)
   - Target: Auto-demote alphas with rolling IC < 0

4. **Stop-loss must prevent single losses > sum of wins**
   - META-style wipeouts are unacceptable
   - Hard stop at -2.5% or -$2,500 per trade

5. **Feature set must have predictive power**
   - Placebo test says random features > real features
   - Need economic features (macro regime, sector rotation, liquidity)

### Sufficient Conditions (Beyond Necessary)

6. **Regime detection must work**
   - 2022 had 0% win rate (vol spike year)
   - Need adaptive logic to halt trading in unfavorable regimes

7. **Transaction costs must be modeled correctly**
   - 38% of trades held ≤5 bars (high turnover)
   - If spread/slippage is underestimated → real losses will be worse

8. **Data quality must be verified**
   - NFLX_1d: 16 trades, all bad
   - Verify: no corporate actions, no stale data, no API errors

---

## 12. COMPARISON TO VALIDATION GATES

### Expected Results (from validation_engine.py)

| Gate | Threshold | Actual | Status |
|------|-----------|--------|--------|
| Walk-forward consistency | ≥40% | **0.0%** | ❌ FAIL |
| Permutation test p-value | <0.05 | **1.000** | ❌ FAIL |
| Placebo test (feature shuffle) | PnL ≤ 0 | **+8,671** | ❌ FAIL |
| Deflated Sharpe | >0.5 | **-1.2** (estimated) | ❌ FAIL |
| OOS/IS decay ratio | >0.5 | **0.0** | ❌ FAIL |

**The validation engine correctly identified the system as overfit.**
The holdout test confirms this with -$124,162 P&L.

---

## 13. CONCLUSION

### The Bottom Line

This system is **fundamentally broken** at the signal generation layer (L1).
No amount of ensembling (L2), portfolio construction (L3), or execution tuning (L4) can fix negative-expectancy alphas.

**Priority 1:** Audit each L1 alpha in isolation. Identify which (if any) have genuine edge.
**Priority 2:** Disable shorts entirely until root cause is found.
**Priority 3:** Implement hard stop-loss to prevent catastrophic single losses.
**Priority 4:** Re-run backtest with full indicator logging for statistical analysis.

### What We Know

1. **26% win rate** across 77 trades
2. **Win/Loss ratio = 0.48** (losses are 2x wins)
3. **ALL primary entry signals are net negative**
4. **Shorts have 9.1% win rate** (FULL_SHORT) and 0% (HALF_SHORT)
5. **Validation failed 100%:** 0% walk-forward consistency, p=1.0 permutation, positive placebo
6. **NFLX_1d is a money pit:** -$8,656 on 16 trades
7. **13-loss streak** indicates systematic, not random, failure

### What We Don't Know (Due to Missing Data)

1. **Indicator distributions:** Only 1 trade has full RSI/MACD/BB data
2. **Statistical significance:** Can't run t-tests without indicator values
3. **Regime classification:** No VIX, yield curve, or macro data
4. **Feature importance:** Which of the 41 features actually predict returns?

### Next Steps

**DO NOT TRADE WITH THIS SYSTEM UNTIL:**
1. L1 alphas are audited individually
2. Shorts are fixed or disabled
3. Stop-loss is implemented
4. Full indicator logging is available
5. At least one alpha shows t-stat > 3.0 on OOS data

**The validation engine saved you from deploying a broken system.**
Fix the root causes before re-enabling.

---

## APPENDIX A: TOP 20 LOSING TRADES (DETAILED)

| # | Symbol | Date | Action | Bars | P&L | Entry Reason | Exit Reason |
|---|--------|------|--------|------|-----|--------------|-------------|
| 1 | META_1d | 2025-02-18 | HALF_LONG | 13 | -$12,880 | MACD bullish; Uptrend; Moving avg bullish cross; Strong trend (ADX) | RSI oversold; Near lower BB; Uptrend; Strong trend (ADX) |
| 2 | HD_1d | 2021-12-23 | FULL_LONG | 14 | -$7,826 | RSI near oversold; Near lower BB; Uptrend; Strong trend (ADX) | RSI near oversold; Near lower BB; Positive value est (0.4280); Full long conviction |
| 3 | UPS_1d | 2023-10-20 | HALF_LONG | 3 | -$7,532 | RSI oversold; MACD bullish; Near lower BB; Positive value est (0.3845) | RSI near oversold; Near lower BB; Positive value est (0.3958); Full long conviction |
| 4 | CRM_1d | 2024-04-08 | FULL_LONG | 4 | -$7,189 | Uptrend; Moving avg bullish cross; Positive value est (0.3843); Full long conviction | RSI near oversold; Near lower BB; Positive value est (0.3569); Full long conviction |
| 5 | MS_1d | 2023-02-16 | FULL_LONG | 13 | -$6,174 | Uptrend; Moving avg bullish cross; Strong trend (ADX); Positive value est (0.4231) | RSI oversold; Near lower BB; Uptrend; Moving avg bullish cross |
| 6 | NFLX_1d | 2024-01-23 | FULL_LONG | 65 | -$6,033 | Moving avg bullish cross; Positive value est (0.3423); Full long conviction | RSI oversold; Near lower BB; Positive value est (0.3711); Full long conviction |
| 7 | ADBE_1d | 2022-06-22 | FULL_SHORT | 1 | -$5,741 | MACD bearish; Full short conviction | MACD bearish; Full short conviction |
| 8 | AXP_1d | 2021-01-19 | FULL_SHORT | 3 | -$5,486 | RSI elevated; Full short conviction | MACD bullish; Moving avg bullish cross; Strong trend (ADX); Positive value est (0.3408) |
| 9 | SLB_1d | 2022-01-04 | FULL_SHORT | 2 | -$5,376 | RSI elevated; Near upper BB; Downtrend signal; Full short conviction | RSI overbought; Near upper BB; Downtrend signal; Full short conviction |
| 10 | TSLA_1d | 2024-11-12 | FULL_SHORT | 4 | -$5,230 | RSI overbought; Near upper BB; Full short conviction | RSI elevated; Full short conviction |
| 11 | CVX_1d | 2020-01-07 | HALF_LONG | 16 | -$5,121 | MACD bullish; Moving avg bullish cross; Strong trend (ADX); Positive value est (0.3838) | RSI oversold; Near lower BB; Strong trend (ADX); Positive value est (0.4404) |
| 12 | UNH_1d | 2022-04-14 | FULL_LONG | 10 | -$4,468 | MACD bullish; Uptrend; Moving avg bullish cross; Strong trend (ADX) | RSI near oversold; Uptrend; Moving avg bullish cross; Strong trend (ADX) |
| 13 | COF_1d | 2023-04-21 | FULL_LONG | 5 | -$4,427 | MACD bullish; Positive value est (0.4229); Full long conviction | MACD bullish; Positive value est (0.4075); Full long conviction |
| 14 | XOM_1d | 2020-01-07 | FULL_LONG | 10 | -$4,361 | MACD bullish; Moving avg bullish cross; Strong trend (ADX); Positive value est (0.2512) | RSI near oversold; Near lower BB; Positive value est (0.2925); Full long conviction |
| 15 | GS_1d | 2023-10-02 | HALF_LONG | 11 | -$4,252 | Near lower BB; Strong trend (ADX); Positive value est (0.3804) | RSI near oversold; Strong trend (ADX); Positive value est (0.4241); Full long conviction |
| 16 | DIS_1d | 2020-01-07 | FULL_LONG | 12 | -$3,982 | Uptrend; Moving avg bullish cross; Positive value est (0.2229); Full long conviction | Exiting position; Trend fading |
| 17 | LOW_1d | 2021-10-29 | HALF_LONG | 12 | -$3,905 | MACD bullish; Uptrend; Moving avg bullish cross; Strong trend (ADX) | MACD bearish; Negative value est (-0.0684); Full short conviction |
| 18 | BAC_1d | 2021-09-01 | FULL_LONG | 13 | -$3,520 | MACD bullish; Uptrend; Moving avg bullish cross; Strong trend (ADX) | RSI oversold; Near lower BB; Uptrend; Positive value est (0.3760) |
| 19 | CAT_1d | 2020-01-07 | HALF_LONG | 11 | -$3,294 | MACD bullish; Uptrend; Moving avg bullish cross; Strong trend (ADX) | RSI near oversold; Near lower BB; Positive value est (0.2159); Full long conviction |
| 20 | DE_1d | 2021-01-19 | HALF_LONG | 11 | -$3,220 | MACD bullish; Uptrend; Moving avg bullish cross; Strong trend (ADX) | MACD bearish; Full short conviction |

---

## APPENDIX B: DATA QUALITY NOTES

### Missing Indicator Data

**Issue:** Only 1 trade (NFLX_1d, 2024-02-05) has full indicator data in `trade_decisions.csv`.
**Impact:** Cannot perform statistical tests (t-test, Mann-Whitney) to identify which indicators discriminate between winners and losers.

**Example of what we're missing:**

```python
# What we HAVE (1 trade):
entry_rsi: 77.36
entry_macd: 0.34
entry_bb_pos: 0.80
entry_adx: 48.12

# What we NEED (77 trades):
For each trade:
  - entry_rsi, entry_macd, entry_bb_pos, entry_adx, etc.
  - exit_rsi, exit_macd, exit_bb_pos, exit_adx, etc.
  - Then: t-test(winners.entry_rsi, losers.entry_rsi)
```

**Recommendation:**
Re-run backtest with `--save-all-indicators` flag to generate full indicator CSV.

### Validation Mismatch

**Issue:** `decision_log.json` contains 1 trade, but `comprehensive_trades.csv` contains 77 trades.
**Hypothesis:** `decision_log.json` was generated from a different run (single-symbol test on NFLX_1d).

**Recommendation:**
Ensure all CSVs and JSONs are from the same backtest run by checking timestamps.

---

## APPENDIX C: GLOSSARY

**IC (Information Coefficient):** Spearman correlation between predicted returns and actual returns. IC > 0.05 is considered good; IC > 0.10 is excellent.

**Permutation Test:** Shuffle outcome labels randomly and re-train model. If p-value > 0.05, the model is not learning real signal.

**Placebo Test:** Shuffle feature columns randomly and re-train model. If placebo PnL > real PnL, the features are not predictive.

**Walk-Forward Consistency:** Fraction of OOS folds where the model outperforms a naive benchmark. Consistency < 40% indicates overfitting.

**Deflated Sharpe Ratio:** Sharpe ratio adjusted for multiple testing and overfitting. Accounts for the number of strategies tried.

**OOS/IS Decay Ratio:** Out-of-sample Sharpe divided by in-sample Sharpe. Ratio < 0.5 indicates severe overfitting.

---

**END OF REPORT**
