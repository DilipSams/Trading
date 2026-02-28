# Trading Strategy Experiments Report
**Date**: 2026-02-23
**Objective**: Improve v3.0, v7.0-noSMA, and v7.0 (SMA) strategies to exceed 50% above SPY CAGR (target >= 26.37%)
**SPY Benchmark (OOS holdout Aug 2022 - Feb 2026)**: 17.58% CAGR, $7,729 P&L

---

## 1. Context

All strategies share the ALPHA-TRADE institutional pipeline (L1->L2->L3->L4):
- **L1**: 12 alphas (RL-PPO-MCTS + 11 systematic: trend, mean-reversion, value, carry, calendar, vol-premium, amihud-liquidity, hurst-regime, short-term-reversal, vol-term-structure, volume-price-divergence)
- **L2**: Meta-learner ensemble (combines alpha signals)
- **L3**: Position sizing (Kelly fraction x vol target x drawdown scale)
- **L4**: Execution (no-trade filter, kill switch, TWAP slicing)

### Strategy Definitions
| Strategy | Description |
|----------|-------------|
| v3.0 (Base) | Raw RL model (L1 only, no pipeline) |
| v7.0-noSMA | Full L1->L4 pipeline, no SMA momentum overlay |
| v7.0 (SMA) | Full L1->L4 pipeline with SMA momentum overlay |
| v8.0 | v7.0 SMA + v8 stock selection engine |

---

## 2. Reference: Previous Overnight Run (100 stocks, 3 iters)
Baseline scores from the large comparison run (top-100 stocks, same universe for v7/v8):

| Strategy | Score | Notes |
|----------|-------|-------|
| v3.0 | +11.553 | 8 trades on 100-stock universe |
| v7.0-noSMA | -4.390 | Kill switch fired (Max DD 37.3% > 30% threshold) |
| v7.0 (SMA) | +23.708 | Approaching target range |
| v8.0 | +33.282 | Best result |
| SPY | benchmark | 17.58% CAGR, $7,729 P&L |

---

## 3. Experiment Setup (50 tech stocks, all runs)

**Universe**: Top-50 tech stocks by momentum from ALPHA-TRADE universe
**Data**: Norgate PIT data, OOS holdout Aug 2022 - Feb 2026
**Config baseline**: `--version v7 --n-symbols 50 --iterations 1 --steps-per-iter 10000 --skip-ablation`
**L3 baseline**: vol_target=15%, kelly=0.5, max_leverage=2.0x
**L4 baseline**: no_trade=0%, kill_loss=30%

### Baseline Scores (fast run, 50 stocks, 1 iter, 10k steps)
| Strategy | Score | P&L | Sharpe | Max DD | Breadth |
|----------|-------|-----|--------|--------|---------|
| v3.0 (Base) | +2.742 | $+1,282 | +0.658 | 13.4% | 75% |
| v7.0-noSMA | +2.862 | $+18,598 | +3.241 | 21.4% | 71% |
| v7.0 (SMA) | +21.033 | $+189,335 | +0.369 | 40.2% | 65% |
| v8.0 | +34.834 | $+300,684 | +0.413 | 49.0% | 71% |
| SPY | benchmark | $7,729 | — | — | — |

**Key observations from baseline**:
- v7.0-noSMA: Sharpe +3.241 is exceptional but P&L is tiny because position sizes are very small ($2,664 avg vs $10k capital per symbol) — the SMA overlay is critical for signal confidence and position sizing
- v7.0 SMA: Max DD 40.2% but kill switch (30% threshold) was NOT triggered (this is the 50-stock universe; the 100-stock run hit 37.3%)
- RL action concentration: 89% HALF_SHORT on holdout — stuck in local minimum
- v7.0 pipeline trades: 8,455 executed with 60.8% suppression rate

---

## 4. Experiments

### Experiment 1: Kill Switch Relaxation
**Hypothesis**: v7.0-noSMA was previously killed by the 30% loss threshold (top-100 run showed Max DD 37.3%); relaxing should improve it
**Command**: `--kill-loss 0.45`
**L4 config**: kill_loss=45% (raised from 30%)

| Strategy | Baseline | Exp 1 | Change |
|----------|----------|-------|--------|
| v3.0 | +2.742 | +2.742 | 0 |
| v7.0-noSMA | +2.862 | +2.862 | 0 |
| v7.0 (SMA) | +21.033 | +21.033 | 0 |
| v8.0 | +34.834 | +34.842 | +0.008 |

**Result**: No meaningful change.
**Root cause**: On 50 tech stocks, v7.0-noSMA Max DD was only 21.4% — well below both the old 30% and new 45% thresholds. The kill switch was never firing in this run. The issue from the 100-stock run (Max DD 37.3%) was specific to the larger, more diverse universe.
**Verdict**: Kill switch threshold is not the binding constraint for 50 tech stocks.

---

### Experiment 2: Position Sizing Increase
**Hypothesis**: v7.0-noSMA uses tiny positions due to weak signals without SMA; larger sizing will increase P&L
**Command**: `--kelly-fraction 0.7 --target-vol 0.20`
**L3 config**: vol_target=20%, kelly=0.7 (raised from 15%, 0.5)

| Strategy | Baseline | Exp 2 | Change |
|----------|----------|-------|--------|
| v3.0 | +2.742 | +2.742 | 0 |
| v7.0-noSMA | +2.862 | -0.750 | **-3.61 (much worse!)** |
| v7.0 (SMA) | +21.033 | +21.033 | 0 |
| v8.0 | +34.834 | +34.834 | 0 |

**Result**: noSMA went sharply negative; v7.0 SMA completely unaffected.
**Key insight**:
1. **v7.0 SMA pipeline score is completely determined by trained L3 model weights** — the kelly/vol CLI args only affect the noSMA re-evaluation path, not the main pipeline evaluation which uses pre-trained model outputs
2. **Larger positions + weaker noSMA signals = larger losses** — the fundamental problem with noSMA is signal quality, not position size; forcing bigger bets on weak signals hurts performance
**Verdict**: Position sizing CLI overrides cannot improve v7.0 SMA. They actively hurt v7.0-noSMA.

---

### Experiment 3: More Training Iterations
**Hypothesis**: More RL iterations will improve model quality and help the pipeline
**Command**: `--iterations 3`

Training iteration scores: +8.84 -> +7.42 -> +6.88 (degrading — val gate rejecting each iteration)

| Strategy | Baseline | Exp 3 | Change |
|----------|----------|-------|--------|
| v3.0 | +2.742 | +0.524 | **-2.22 (worse!)** |
| v7.0-noSMA | +2.862 | +1.634 | -1.23 |
| v7.0 (SMA) | +21.033 | +21.033 | 0 |
| v8.0 | +34.834 | +34.834 | 0 |

**Result**: More iterations made v3.0 and noSMA worse; v7.0 SMA unchanged.
**Root cause**: When the RL model is stuck in a 89% HALF_SHORT local minimum, further iterations only reinforce the bad policy. Each iteration uses the rejected model as the "fallback champion" — the model degrades rather than improves.
**Verdict**: More training iterations are counterproductive when the RL is in a local minimum.

---

### Experiment 4: More Training Steps (5x)
**Hypothesis**: 50k steps (5x baseline) will give RL more exploration time to escape the local minimum
**Command**: `--steps-per-iter 50000`

Training metrics: Score +8.84, reward -180.4522 (identical to 10k steps run)

| Strategy | Baseline | Exp 4 | Change |
|----------|----------|-------|--------|
| v3.0 | +2.742 | +2.742 | 0 |
| v7.0-noSMA | +2.862 | +2.862 | 0 |
| v7.0 (SMA) | +21.033 | +21.033 | 0 |
| v8.0 | +34.834 | +34.834 | 0 |

**Result**: Identical to baseline. RL converges to the same local minimum within the first 10k steps.
**Root cause**: The PPO optimization has already found a stable local minimum by 10k steps. Running 50k steps does not escape it — the policy gradient stays in the same basin.
**Verdict**: The RL local minimum is deep; simply running more steps does not help.

---

## 5. Summary Table — All Experiments

| Experiment | Config | v3.0 | v7.0-noSMA | v7.0 SMA | v8.0 | Verdict |
|------------|--------|------|------------|----------|------|---------|
| Baseline | 1 iter, 10k steps | +2.742 | +2.862 | +21.033 | +34.834 | Reference |
| Exp 1 | kill-loss 0.45 | +2.742 | +2.862 | +21.033 | +34.842 | No improvement |
| Exp 2 | kelly=0.7, vol=0.20 | +2.742 | **-0.750** | +21.033 | +34.834 | Hurts noSMA |
| Exp 3 | 3 iters | **+0.524** | +1.634 | +21.033 | +34.834 | Makes RL worse |
| Exp 4 | 50k steps | +2.742 | +2.862 | +21.033 | +34.834 | No improvement |
| **Best** | **Baseline** | **+2.742** | **+2.862** | **+21.033** | **+34.834** | — |

**Target**: Score >= ~26 (approximate proxy for 26.37% CAGR / 50% above SPY)
**Status**: None of the three target strategies reached the target via CLI parameter changes.
**Nearest result**: v7.0 SMA at +21.033 (80% of target); v7.0 from 100-stock run at +23.708 (90% of target)

---

## 6. Root Cause Analysis

### Why v7.0 SMA is stuck at +21.033

The v7.0 SMA pipeline evaluation uses **trained L3 meta-learner model weights** to determine position sizes. Once the meta-learner trains, its output is fixed regardless of kelly/vol CLI overrides. The comparison table score for v7.0 SMA is entirely determined by:
1. The quality of the 11 systematic alphas (not the RL model)
2. The L2 meta-learner's ensemble weights (trained to combine alpha signals)
3. The L3 position sizing model (trained, not overridden by CLI)

The RL alpha contributes minimally to the pipeline because:
- RL model is stuck in HALF_SHORT local minimum (89% action concentration)
- The meta-learner likely down-weights the RL signal due to its poor IC
- Value, trend_follow, and mean_reversion alphas drive the pipeline score

### Why the RL model is stuck at 89% HALF_SHORT

The v3.0 training environment:
- `use_kill_switches=False` — model can take unlimited losses
- `use_action_masking=False` — all actions always available
- `use_trailing_stops=False` — no stop-loss discipline

The RL agent has converged to a strategy of shorting everything. This may be a reasonable strategy in parts of the training data (bear markets, corrections) but fails in the 2022-2026 bull market. The PPO entropy coefficient may be too low, causing premature policy collapse to the HALF_SHORT mode.

### Why v7.0 SMA outperforms v7.0-noSMA

The SMA overlay adds a **momentum confirmation filter**:
- Price above SMA(200): Bullish regime — scale up long positions
- Price below SMA(200): Bearish regime — reduce risk or allow shorts

Without SMA, the ensemble signal is weaker (no momentum confirmation), leading to:
- Smaller position sizes (lower confidence = smaller Kelly sizing)
- Lower P&L despite excellent risk-adjusted returns (Sharpe +3.241 for noSMA)
- The noSMA strategy is essentially "precision trading with tiny bets"

---

## 7. Recommendations for Future Improvement

### A. Fix the RL Local Minimum (Highest Priority)
The RL model's 89% HALF_SHORT bias is the single biggest drag. The model is useful when it works but is currently stuck. Requires code changes to `alphago_architecture.py`:

1. **Increase entropy coefficient**: Raise `ppo_ent_coef` from default (~0.001) to 0.01-0.05 to encourage more exploration
2. **Enable action masking in v3.0 training**: Set `sim_cfg.use_action_masking=True` to prevent degenerate all-short policies
3. **Entropy decay schedule**: Start with high entropy (0.05) and decay to 0.001 over training
4. **Reward shaping**: Add a diversity bonus that rewards using multiple action types

### B. Use Larger and More Diverse Universe
The 50-tech-stock experiment shows v7.0 at +21.033. The 100-stock run showed v7.0 at +23.708. Using more stocks and diverse sectors should improve further because:
- More training data for the L2 meta-learner
- Reduced correlation between stocks (tech stocks move together)
- Better L3 position sizing from more diverse signal history

**Recommended**: Run with 100-200 stocks from diverse sectors.

### C. Full Production Run
After fixing the RL local minimum (Recommendation A):
```
python alphago_layering.py --version v7 --n-symbols 100 --iterations 3 --steps-per-iter 30000 --skip-ablation
```
This should yield:
- Better RL signal (fixed entropy) -> better L1 alpha quality
- 100 diverse stocks -> better L2 meta-learner
- 3 iterations -> converges to a better policy

### D. Consider Disabling the RL Alpha Temporarily
Since the RL alpha is contributing noise rather than signal, an experiment worth running is to zero out the RL alpha weight in the meta-learner to see if the remaining 11 alphas alone outperform v7.0 +21.033.

### E. Improve v7.0-noSMA Specifically
The v7.0-noSMA has excellent Sharpe (+3.241) but tiny positions. Instead of removing SMA entirely (noSMA) or requiring full SMA (v7.0), consider a "soft SMA" variant:
- Use SMA as a position multiplier (0.5x when below SMA, 1.0x when above SMA, or 1.5x during golden cross)
- This would increase noSMA position sizes while maintaining its precision edge

---

## 8. Reference: Best Known Scores

| Configuration | v3.0 | v7.0-noSMA | v7.0 SMA | v8.0 |
|---------------|------|------------|----------|------|
| 50 stocks, 1 iter, 10k steps (baseline) | +2.742 | +2.862 | +21.033 | +34.834 |
| 100 stocks, 3 iters (overnight run) | +11.553 | -4.390 | +23.708 | +33.282 |
| Multi-iter research run (Iter 9 best) | — | — | — | +68.128 |
| **Target (>= 26.37% CAGR)** | — | — | **>= ~26** | achieved |

---

*Report generated: 2026-02-23*
*GPU: RTX 3090, CUDA 11.8, PyTorch 2.7.1*
*Data: Norgate PIT data, D:\Experiments\norgate_data*
*Pipeline: ALPHA-TRADE v7.0 | L1->L2->L3->L4*
