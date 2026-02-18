# Comprehensive Backtest Analysis Report
## Alpha-Trade v7.0 - 50 Symbol Full Test
**Date**: 2026-02-16
**Test Parameters**: 5 iterations, 40,000 steps/iteration, 50 large-cap stocks

---

## EXECUTIVE SUMMARY

### ⚠️ CRITICAL FINDING: SEVERE OVERFITTING DETECTED

The system shows **massive overfitting** to training data with complete failure on out-of-sample evaluation:

- **Training Performance**: +$655k (positive across all iterations)
- **Holdout Performance**: +$24k (only 4/5 symbols profitable)
- **Full Evaluation**: **-$124k** (20W/57L, 26% win rate)

**Recommendation**: **DO NOT DEPLOY** - System requires fundamental redesign

---

## DETAILED RESULTS

### Training Iterations (Train/Val Sets)

| Iteration | P&L | Win Rate | Sharpe | Score | Validation |
|-----------|-----|----------|--------|-------|------------|
| 1 | +$654,956 | 33.2% | 1.46 | +11.73 | ❌ FAILED |
| 2 | +$653,183 | 34.7% | 1.58 | +11.83 | ✅ CHAMPION |
| 3 | +$566,303 | 36.8% | 3.87 | +11.03 | ❌ FAILED |
| 4 | +$523,488 | 32.8% | 1.31 | +9.44 | ❌ FAILED |
| 5 | +$591,652 | 38.7% | 3.63 | +11.32 | ❌ FAILED |

**Champion Model**: Iteration 2 (+$653k, 34.7% WR, 1.58 Sharpe)

### Final Evaluation (All 50 Symbols - OOS)

```
❌ Total P&L:       -$124,162.02
❌ Win Rate:        26.0% (20W / 57L)
❌ Avg Win:         $1,262.20
❌ Avg Loss:        -$2,621.16
❌ Reward/Risk:     0.48 (CRITICAL: need >2.0)
   Avg Hold:        13.2 bars
   Total Trades:    77
```

---

## ROOT CAUSE ANALYSIS

### 1. ⚠️ VALIDATION FAILURES (CRITICAL)

**All 5 iterations failed validation gates:**

```
Iteration 2 (Champion):
  - COST_1d: walk-forward consistency 0.0% < 40.0%
  - AAPL_1d: walk-forward consistency 0.0% < 40.0%
  - META_1d: walk_forward invalid

  Permutation test: p=1.000 (complete failure)
  Placebo test: feature shuffle OUTPERFORMED real model

  Result: 0/3 symbols passed (0% < 50% threshold)
```

**What this means**:
- Model has **zero predictive power** on held-out data
- Performance is **random noise**, not signal
- Feature shuffle beating real model = features are noise

### 2. 🚫 FEATURE DISCIPLINE VIOLATIONS

```
WARNING: 31 features > sqrt(252) = 15.9 bars
WARNING: 40 features > sqrt(504) = 22.4 bars
WARNING: 40 features > sqrt(756) = 27.5 bars
WARNING: 38 features > sqrt(1008) = 31.7 bars
WARNING: 38 features > sqrt(1260) = 35.5 bars
```

**Rule of Thumb Violation**: features should be < sqrt(observations)

With 49 features and typical observation windows of 250-1000 bars:
- 49 >> sqrt(250) = 15.8 ❌
- Need either: **reduce features to <16** OR **increase window to >2,400 bars**

### 3. 📉 TERRIBLE RISK/REWARD PROFILE

**Current**: Win 26% × $1,262 = $328 expected per winning trade
**Current**: Lose 74% × $2,621 = $1,940 expected per losing trade
**Net Expected Value per trade**: **-$1,612** ❌

**Required for profitability** (at 26% win rate):
- Need Reward/Risk > 2.85 (currently 0.48)
- Avg Win needs to be **$7,460** (currently $1,262)
- OR Win Rate needs to be >72% (currently 26%)

### 4. 🔴 WORST LOSING TRADES ANALYSIS

| Symbol | Entry | Exit | P&L | Bars | Reason |
|--------|-------|------|-----|------|---------|
| META | 2025-02-18 | 2025-03-07 | -$12,880 | 13 | MACD bullish trend following |
| HD | 2021-12-23 | 2022-01-13 | -$7,826 | 14 | RSI oversold + BB signal |
| UPS | 2023-10-20 | 2023-10-25 | -$7,532 | 3 | RSI oversold mean reversion |
| CRM | 2024-04-08 | 2024-04-12 | -$7,189 | 4 | Trend following |
| MS | 2023-02-16 | 2023-03-08 | -$6,174 | 13 | Moving avg cross |

**Pattern**: All losses followed **textbook entry signals** that failed

### 5. 📊 SYMBOL-LEVEL PERFORMANCE

**Only 10 Profitable Symbols** (out of 50):

| Symbol | Trades | Win% | P&L |
|--------|--------|------|-----|
| KO | 1 | 100% | +$2,190 |
| HON | 1 | 100% | +$1,933 |
| MA | 1 | 100% | +$1,676 |
| AMZN | 1 | 100% | +$1,201 |
| AAPL | 1 | 100% | +$1,184 |
| PG | 1 | 100% | +$792 |
| ABT | 1 | 100% | +$371 |
| JPM | 1 | 100% | +$358 |
| BAC | 7 | 43% | +$326 |
| WMT | 1 | 100% | +$65 |

**Worst 10 Losing Symbols**:

| Symbol | Trades | Win% | P&L |
|--------|--------|------|-----|
| NFLX | 16 | 31% | -$8,656 |
| HD | 1 | 0% | -$7,826 |
| UPS | 1 | 0% | -$7,532 |
| CRM | 1 | 0% | -$7,189 |
| META | 3 | 67% | -$6,778 |
| MS | 1 | 0% | -$6,174 |
| ADBE | 1 | 0% | -$5,741 |
| AXP | 1 | 0% | -$5,486 |
| SLB | 1 | 0% | -$5,376 |
| TSLA | 1 | 0% | -$5,230 |

**Key Observation**: 80% of symbols (40/50) are unprofitable

### 6. 🎯 ENTRY REASON ANALYSIS

**Most Common Entry Signals**:
- Strong trend (ADX): 44 trades (57%)
- MACD bullish: 39 trades (51%)
- Moving avg bullish cross: 37 trades (48%)
- Uptrend: 31 trades (40%)

**Problem**: Classic trend-following signals are **failing in all market conditions**

### 7. 📈 TRADE DURATION ISSUES

**Very Short Holding Periods**:
- Many 1-4 bar trades (likely over-trading)
- Average hold: 13.2 bars (~2.5 weeks)
- Shortest losing trade: 1 bar (-$5,741 on ADBE)

**Interpretation**:
- System entering too early (whipsawed out)
- OR exiting too early (not letting winners run)
- High transaction costs from frequent trading

---

## WHAT WENT WRONG

### Theoretical Issues

1. **Alpha Saturation**: 10 alphas may have redundant signals
   - Trend + Hurst Regime + Short-Term Reversal likely correlated
   - Calendar + Volatility Premium may overlap
   - Amihud Liquidity + Value may be similar

2. **Curse of Dimensionality**: 49 features >> 15-35 optimal (per sqrt rule)
   - ML model fitting noise, not signal
   - Needs PCA or feature selection

3. **Regime Blindness**: System trained on one market regime, tested on another
   - Training: 2015-2023 (mostly bull market)
   - Testing: 2024-2026 (different regime)

4. **Look-Ahead Bias** (possible): Some alphas may use future data
   - Hurst exponent calculation
   - Calendar effects with forward-looking dates

### Practical Issues

1. **Parameter Over-Optimization**:
   - 5 iterations trying to find "champion" = p-hacking
   - Champion (iter 2) failed validation but used anyway

2. **Validation Gate Failure Ignored**:
   - System correctly detected overfitting
   - But "champion" model was deployed anyway

3. **Risk Management Failure**:
   - Max drawdown gates not enforced
   - No per-symbol loss limits
   - Position sizing appears incorrect (half vs full positions inconsistent)

---

## COMPARISON WITH BASELINE

### Previous Quick Test (1 iteration, 1k steps)
- P&L: -$18k
- Win Rate: 32%
- Trades: 50

### Current Full Test (5 iterations, 40k steps)
- P&L: -$124k (**6.9x worse**)
- Win Rate: 26% (**worse**)
- Trades: 77

**Conclusion**: More training iterations made performance **worse**, confirming overfitting

---

## CRITICAL IMPROVEMENTS NEEDED

### 🔴 URGENT (Must Fix Before Any Deployment)

#### 1. **Feature Reduction**
```
Current: 49 features
Target: <16 features (per sqrt rule)
```

**Actions**:
- Run PCA or feature importance analysis
- Keep only top 15 features with IC > 0.02
- Remove redundant alphas (keep 4-5 best)

**Candidate Features to Remove**:
- Hurst regime (redundant with trend)
- Short-term reversal (conflicts with trend)
- Calendar effects (weak predictive power)
- Some technical indicators (RSI, MACD redundant)

#### 2. **Fix Validation Logic**
```
Current: Validation fails but model deployed anyway
Target: NEVER deploy if validation fails
```

**Actions**:
- Enforce walk-forward consistency > 60%
- Enforce permutation test p < 0.05
- Enforce placebo test: real > shuffle by 2x
- If ANY validation fails → reject model

#### 3. **Fix Risk/Reward Profile**
```
Current: RR=0.48, WR=26% → -$1,612 per trade expected
Target: RR>2.0, WR>35% → positive expectancy
```

**Actions**:
- Wider stops (reduce whipsaw losses)
- Let winners run longer (increase avg win)
- More selective entry (reduce false signals)
- Add conviction filtering (only trade high-confidence setups)

#### 4. **Regime Detection**
```
Current: Blind to market regime
Target: Adapt to current regime
```

**Actions**:
- Add macro regime classifier (bull/bear/sideways)
- Enable/disable alphas per regime
  - Bull: Enable trend, disable mean reversion
  - Bear: Enable shorts, disable long-bias alphas
  - Sideways: Enable mean reversion, disable trend

### 🟡 HIGH PRIORITY

#### 5. **Alpha Consolidation**
Keep only **4-5 best alphas**:
1. **Trend** (15-bar horizon, inverted) - IC: 0.059
2. **Mean Reversion** (5-bar horizon) - IC: 0.028
3. **Value** (long-term) - IC: 0.026
4. **RL/MCTS** (adaptive) - IC: unknown
5. **Vol Premium** (optional) - IC: 0.015

Remove:
- Calendar (IC too low)
- Carry (IC too low)
- Amihud Liquidity (new, unproven)
- Hurst Regime (redundant with trend)
- Short-Term Reversal (conflicts with trend)

#### 6. **Position Sizing Fix**
```
Current: Inconsistent HALF vs FULL positions
Target: Kelly Criterion with proper volatility scaling
```

**Actions**:
- Implement strict Kelly sizing (currently 0.25 factor)
- Scale by realized volatility
- Cap at 2x leverage (currently set but not enforced?)
- Never go full conviction without IC > 0.05

#### 7. **Transaction Cost Realism**
```
Current: May be underestimating costs
Target: Conservative cost assumptions
```

**Actions**:
- Increase slippage model (1-5 bps realistic for large caps)
- Add market impact model (size-dependent)
- Model borrowing costs for shorts
- Include overnight financing costs

### 🟢 MEDIUM PRIORITY

#### 8. **Training Strategy**
- Use **rolling window** instead of expanding window
- Retrain every quarter, not continuously
- Use fixed validation sets (don't optimize on validation)

#### 9. **Meta-Labeling**
- Add confidence predictor on top of alphas
- Only trade when confidence > 60%
- May reduce trades but improve win rate

#### 10. **Ensemble Approach**
- Instead of 10 alphas → L1 ensemble
- Train separate models per market regime
- Weight alphas by recent performance

---

## RECOMMENDED NEXT STEPS

### Phase 1: Diagnostic (1-2 days)

1. **Feature Importance Analysis**
   ```bash
   python analyze_feature_importance.py --method shap
   ```
   - Identify which features are actually predictive
   - Remove features with SHAP value < 0.001

2. **IC Profiling Per Alpha**
   ```bash
   python quick_ic_test.py --invert-trend --per-alpha
   ```
   - Measure each alpha's IC independently
   - Remove alphas with IC < 0.015

3. **Regime Analysis**
   ```bash
   python analyze_regime_performance.py
   ```
   - Break down performance by VIX regime, trend regime, vol regime
   - Identify which regimes system works in

### Phase 2: Simplification (2-3 days)

1. **Reduce to 4 Core Alphas**
   - Keep: Trend, Mean Reversion, Value, RL
   - Remove: Calendar, Carry, Liquidity, Hurst, Reversal, Vol Premium

2. **Feature Reduction to <16**
   - Run PCA
   - Keep top 15 principal components OR top 15 features by IC

3. **Re-run Validation**
   ```bash
   python alphago_layering.py --invert-trend --iterations 1 --steps-per-iter 50000
   ```
   - Should pass validation gates
   - Should have positive OOS performance

### Phase 3: Risk Management (1-2 days)

1. **Implement Stricter Position Sizing**
   - Kelly fraction: 0.1 (was 0.25)
   - Max single position: 5% (was ~20%)
   - Max drawdown kill switch: 10% (was 15%)

2. **Add Regime Filters**
   - VIX > 30: Reduce position sizes by 50%
   - Trend strength < 20: Disable trend alpha
   - Vol regime > 1.2: Reduce leverage

3. **Improve Stop Loss Logic**
   - ATR-based stops (2× ATR)
   - Time-based stops (exit if no profit after 20 bars)
   - Volatility-adjusted stops

### Phase 4: Re-Test (1 day)

```bash
python alphago_layering.py --invert-trend --iterations 3 --steps-per-iter 40000
```

**Success Criteria**:
- ✅ Validation gates: >50% pass rate
- ✅ Walk-forward consistency: >60%
- ✅ OOS Win Rate: >35%
- ✅ OOS Reward/Risk: >1.5
- ✅ OOS P&L: >$0 (positive)

If ANY criterion fails → back to Phase 2

---

## ESTIMATED TIMELINE

**Fast Track** (if drop everything else):
- Phase 1: 2 days
- Phase 2: 3 days
- Phase 3: 2 days
- Phase 4: 1 day
- **Total: 8 days**

**Realistic** (normal pace):
- Phase 1: 3-4 days
- Phase 2: 4-5 days
- Phase 3: 2-3 days
- Phase 4: 1-2 days
- **Total: 10-14 days (2 weeks)**

---

## ADDITIONAL INVESTIGATIONS

### 1. Check for Data Snooping

**Hypothesis**: System might be using future data unknowingly

**Test**:
```python
python test_lookahead_bias.py --check-all-alphas
```

Look for:
- Hurst calculation using full history
- Calendar effects using next year's data
- Feature engineering looking ahead

### 2. Test on Different Universe

**Hypothesis**: System overfit to these specific 50 symbols

**Test**:
- Run on different 50 symbols (mid-cap or international)
- Run on crypto or commodities
- If performance similar → good (generalizes)
- If performance different → bad (overfit to universe)

### 3. Monte Carlo Robustness

**Hypothesis**: Results might be luck (single path dependency)

**Test**:
```python
python monte_carlo_test.py --runs 1000 --randomize-order
```

- Randomize symbol order
- Randomize entry timing (±1-2 bars)
- Generate distribution of outcomes
- If 95% confidence interval includes $0 → not robust

---

## CONCLUSION

### Current Status: ❌ NOT READY FOR DEPLOYMENT

**Summary**:
- System shows severe overfitting to training data
- Validation correctly detected failure but was ignored
- Out-of-sample performance is **negative** and **not statistically significant**
- Too many features (49) relative to observations (sqrt rule violated)
- Risk/reward profile is catastrophically bad (0.48 vs needed 2.0)

**Root Causes**:
1. Feature explosion (49 features, 10 alphas)
2. Overfitting to training regime
3. Poor risk management
4. Validation gates correctly failed but bypassed

**Path Forward**:
- Reduce to 4-5 core alphas
- Cut features to <16
- Fix risk/reward (target RR>2.0, WR>35%)
- Never deploy if validation fails
- Re-test with strict success criteria

**Estimated Fix Time**: 2 weeks

---

## FILES GENERATED

1. `comprehensive_trades.csv` - All 77 trades with full details
2. `symbol_summary.csv` - Per-symbol P&L breakdown
3. `full_test_output_20260216_165433.log` - Complete backtest log
4. `C:\Users\dilip\alpha_trade_v2_results\decision_log.json` - Detailed trade decisions
5. `C:\Users\dilip\alpha_trade_v2_results\trade_decisions.csv` - Trade CSV with indicators
6. `C:\Users\dilip\alpha_trade_v2_results\step_decisions.csv` - Bar-by-bar decisions (43,171 rows)

---

**Report Generated**: 2026-02-16 18:30
**Analyst**: Claude Sonnet 4.5
**Next Review**: After Phase 2 simplification complete
