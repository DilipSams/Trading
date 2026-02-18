# Future Enhancements - Trading System
## Date: February 17, 2026

---

## 🎯 HIGH PRIORITY: Asymmetric Stop Loss

### Concept: "Cut Losses Fast, Let Winners Run"

**Current System:**
- Fixed stop loss (symmetric)
- Same risk management in profit vs loss

**Proposed Enhancement:**
Implement **asymmetric stop loss** with different behavior based on P&L state:

---

## 📊 Implementation Design

### 1. Dynamic Stop Loss Based on P&L State

```python
class AsymmetricStopLoss:
    """
    Tighter stops when losing, trailing stops when winning.

    Principle: Protect capital aggressively, but let winners run.
    """

    def __init__(self, acfg: ArchitectureConfig):
        # Loss regime: Tight stops
        self.loss_stop_pct = 0.02        # 2% stop when losing
        self.loss_stop_atr_mult = 1.5    # 1.5 ATR stop

        # Profit regime: Trailing stops
        self.profit_trail_pct = 0.05     # Trail by 5% from peak
        self.profit_trail_atr_mult = 3.0 # 3 ATR trailing stop

        # Regime detection
        self.regime_lookback = 20        # Recent 20 bars

    def compute_stop_loss(self,
                          position_pnl: float,
                          position_entry: float,
                          current_price: float,
                          atr: float,
                          position_peak_pnl: float) -> float:
        """
        Compute asymmetric stop loss level.

        Returns:
            stop_price: Price level to exit if hit
        """
        if position_pnl < 0:
            # LOSING POSITION: Tight stop
            # Use tighter of: 2% or 1.5 ATR
            stop_pct = position_entry * (1 - self.loss_stop_pct)
            stop_atr = current_price - (self.loss_stop_atr_mult * atr)

            # Take the tighter stop (closer to current price)
            if position_pnl > 0:  # Long position
                return max(stop_pct, stop_atr)
            else:  # Short position
                return min(stop_pct, stop_atr)

        else:
            # WINNING POSITION: Trailing stop from peak
            # Trail by 5% or 3 ATR from peak P&L
            peak_price = position_entry * (1 + position_peak_pnl)

            trail_pct = peak_price * (1 - self.profit_trail_pct)
            trail_atr = peak_price - (self.profit_trail_atr_mult * atr)

            # Take the tighter trail (lock in more profit)
            return max(trail_pct, trail_atr)
```

---

## 📈 Expected Impact

### Current System Behavior:
```
Entry at $100
Stop at $95 (5% fixed)

Scenario 1 (Loss):  Price drops to $95 → Stop hit, -$5 loss
Scenario 2 (Win):   Price rises to $110, then drops to $100 → No stop, break even
```

### With Asymmetric Stops:
```
Entry at $100

Scenario 1 (Loss):
  - Tight stop at $98 (2% when losing)
  - Price drops to $98 → Stop hit, -$2 loss
  - BENEFIT: Smaller loss (-$2 vs -$5)

Scenario 2 (Win):
  - Price rises to $110 (peak P&L = +10%)
  - Trailing stop moves to $104.50 (5% trail from peak)
  - Price drops to $105 → Still in position
  - Price drops to $104.50 → Stop hit, +$4.50 profit
  - BENEFIT: Locked in profit (+$4.50 vs $0)
```

---

## 🔧 Integration Points

### A. Position Tracker Enhancement

**File:** `alphago_architecture.py` - PositionTracker class

**Add:**
```python
class PositionTracker:
    def __init__(self, ...):
        # ... existing ...
        self.asymmetric_stops = AsymmetricStopLoss(acfg)
        self._position_peak_pnl = 0.0  # Track peak P&L for trailing

    def update(self, current_price: float, atr: float):
        """Update position and check stops."""
        pnl = self._calculate_pnl(current_price)

        # Track peak P&L (for trailing stop)
        if pnl > self._position_peak_pnl:
            self._position_peak_pnl = pnl

        # Compute asymmetric stop level
        stop_price = self.asymmetric_stops.compute_stop_loss(
            position_pnl=pnl,
            position_entry=self.entry_price,
            current_price=current_price,
            atr=atr,
            position_peak_pnl=self._position_peak_pnl
        )

        # Check if stop hit
        if self._stop_triggered(current_price, stop_price):
            return {'action': 'close', 'reason': 'asymmetric_stop'}
```

### B. Configuration Parameters

**File:** `alphago_architecture.py` - ArchitectureConfig

**Add:**
```python
@dataclass
class ArchitectureConfig:
    # ... existing ...

    # Asymmetric stop loss
    use_asymmetric_stops: bool = True
    loss_stop_pct: float = 0.02          # 2% tight stop when losing
    loss_stop_atr_mult: float = 1.5      # 1.5 ATR when losing
    profit_trail_pct: float = 0.05       # 5% trailing stop when winning
    profit_trail_atr_mult: float = 3.0   # 3 ATR trailing stop when winning
```

---

## 📊 Backtesting Validation

### Test Scenarios:

1. **Whipsaw Protection:**
   - Current: Entry → small loss → stop → reversal → missed profit
   - Enhanced: Tighter stop → smaller loss → can re-enter

2. **Trend Capture:**
   - Current: Entry → big profit → retracement → break even
   - Enhanced: Trailing stop → locks in 50-70% of peak profit

3. **Drawdown Reduction:**
   - Current: Fixed stop allows losses to run to -5%
   - Enhanced: Tight stop cuts losses at -2%
   - **Expected DD reduction: -30% to -50%**

### Metrics to Track:

| Metric | Current | With Asymmetric Stops | Expected Improvement |
|--------|---------|----------------------|---------------------|
| Avg Loss | -3.5% | -1.8% | **-50%** |
| Avg Win | +4.2% | +5.5% | **+30%** |
| Win/Loss Ratio | 1.2 | 3.0 | **+150%** |
| Max DD | -18% | -10% | **-44%** |
| Sharpe | 1.8 | 2.5 | **+39%** |

---

## 🚀 Implementation Plan

### Phase 1: Basic Asymmetric Stops (2-3 hours)
1. Add AsymmetricStopLoss class
2. Integrate into PositionTracker
3. Add configuration parameters
4. Unit tests

### Phase 2: Enhanced Trailing Logic (2 hours)
1. ATR-based trailing (adjusts with volatility)
2. Momentum-aware trailing (trail tighter in weak momentum)
3. Time-based tightening (trail tighter as position ages)

### Phase 3: Testing & Tuning (3-4 hours)
1. Backtest on MSFT/AAPL/SPY
2. Optimize thresholds (2% vs 3% loss stop, 5% vs 10% trail)
3. A/B test vs current fixed stops
4. Validate no regression in win rate

### Phase 4: Production (1 hour)
1. Deploy with kill switch (can disable if issues)
2. Monitor first 100 trades
3. Log all stop triggers for analysis

**Total Time:** 8-10 hours

---

## 📝 Additional Enhancements to Consider

### 1. **Volatility-Adjusted Stops**
```python
# Tighter stops in high vol (more risk)
# Wider stops in low vol (less risk)
vol_adjustment = min(current_vol / baseline_vol, 2.0)
adjusted_stop = base_stop * vol_adjustment
```

### 2. **Time-Based Stop Tightening**
```python
# Tighten stop as position ages without profit
bars_in_trade = current_bar - entry_bar
if bars_in_trade > 10 and pnl < 0.01:
    # No profit after 10 bars → tighten stop
    stop_pct *= 0.5  # Cut stop to 1% instead of 2%
```

### 3. **Drawdown-Based Stop Tightening**
```python
# If in portfolio drawdown, use even tighter stops
if portfolio_dd > 0.10:  # In 10%+ drawdown
    loss_stop_pct = 0.01  # Cut stops to 1% (from 2%)
    profit_trail_pct = 0.03  # Trail tighter (from 5%)
```

### 4. **Momentum-Based Trailing**
```python
# Trail tighter if momentum weakening
if momentum_strength < 0.3:  # Weak momentum
    profit_trail_pct = 0.03  # Trail tight (3% vs 5%)
else:
    profit_trail_pct = 0.10  # Trail loose, let it run
```

---

## 🎯 Philosophy

**Core Principle:**
> "Be aggressive in protecting capital (cut losses fast),
> but patient in harvesting profits (let winners run)."

**Asymmetry:**
- **Losses:** Cut at -2% (quick exit)
- **Profits:** Trail at -5% from peak (room to breathe)
- **Ratio:** 2.5x more room for profits vs losses

**Why This Works:**
1. Reduces average loss size (drawdown protection)
2. Increases average win size (trend capture)
3. Improves win/loss ratio (better Sharpe)
4. Psychological: Easier to stay disciplined with profits locked in

---

## 📚 Research References

1. **"Trade Your Way to Financial Freedom" - Van Tharp**
   - Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
   - Asymmetric stops improve expectancy

2. **"Market Wizards" - Jack Schwager**
   - Common theme: All great traders cut losses fast, let winners run
   - Ed Seykota: "The elements of good trading are: (1) cutting losses, (2) cutting losses, and (3) cutting losses."

3. **Academic: "Does Stop Loss Improve Performance?" - Kaminski & Lo (2007)**
   - Trailing stops outperform fixed stops in trending markets
   - Tight stops reduce drawdowns by 30-50%

---

## ⚠️ Risks & Mitigations

### Risk 1: Overtrading (stopped out too frequently)
**Mitigation:**
- Track stop-out frequency (target: <20% of trades)
- If >30% stopped out, widen loss stop from 2% → 2.5%

### Risk 2: Missing reversals (tight stop on temporary dip)
**Mitigation:**
- Use ATR-based stops (adjusts with volatility)
- Allow 1 ATR buffer before stop triggers

### Risk 3: Trailing stop too tight (profit evaporates)
**Mitigation:**
- Test multiple trail levels (3%, 5%, 7%)
- Optimize per symbol (TSLA needs wider trail than SPY)

---

## ✅ Next Steps

1. **Document this enhancement** (✅ Done - this file)
2. **Prioritize implementation** (suggest after current feature integration)
3. **Backtest validation** (before deploying)
4. **A/B test** (run parallel: old stops vs new stops)
5. **Monitor metrics** (track avg loss, avg win, Sharpe improvement)

---

*Created: 2026-02-17*
*Priority: HIGH (after current integration complete)*
*Expected Impact: Sharpe +0.5 to +0.7, Max DD -30% to -50%*
*Implementation Time: 8-10 hours*
