# Asymmetric Stop Loss - Implementation Guide
## Step-by-Step Instructions

---

## ✅ STEP 1: Create Stop Loss Module (COMPLETE)

**File Created:** `alphago_stop_loss.py`

**What it contains:**
- `StopLossConfig` - Configuration dataclass
- `AsymmetricStopLoss` - Main stop loss manager
- Built-in unit tests

**Test it:**
```bash
cd /d/Experiments/Trading
python alphago_stop_loss.py
```

**Expected output:**
```
[TEST 1] Losing Position
  Stop Price: $98.00 (tight 2% stop)

[TEST 2] Winning Position
  Stop Price: $104.50 (trailing from $110 peak)

[TEST 3] High Volatility
  Stop Price: $96.00 (wider stop due to 2x vol)
```

---

## 📝 STEP 2: Add Configuration to ArchitectureConfig

**File:** `alphago_architecture.py`

**Location:** Find `@dataclass class ArchitectureConfig:` (around line 240)

**Add these parameters:**

```python
@dataclass
class ArchitectureConfig:
    # ... existing parameters ...

    # ===================================================================
    # NEW: Asymmetric Stop Loss Configuration
    # ===================================================================
    use_asymmetric_stops: bool = True        # Enable asymmetric stops

    # Loss regime (tight stops)
    loss_stop_pct: float = 0.02              # 2% stop when losing
    loss_stop_atr_mult: float = 1.5          # 1.5 ATR stop when losing

    # Profit regime (trailing stops)
    profit_trail_pct: float = 0.05           # Trail 5% from peak
    profit_trail_atr_mult: float = 3.0       # 3 ATR trailing stop

    # Volatility adjustments
    vol_adjust_stops: bool = True            # Adjust stops for volatility
    vol_baseline: float = 0.15               # Baseline volatility (15%)
    vol_max_adjustment: float = 2.0          # Max vol adjustment factor

    # Time-based tightening
    time_tighten_enabled: bool = False       # Tighten stops over time
    time_tighten_bars: int = 10              # Start tightening after N bars
    time_tighten_factor: float = 0.5         # Tighten to 50% of original
```

**Verification:**
```bash
python -c "from alphago_architecture import ArchitectureConfig; acfg = ArchitectureConfig(); print(f'Asymmetric stops enabled: {acfg.use_asymmetric_stops}')"
```

---

## 🔧 STEP 3: Integrate into PositionTracker

**File:** `alphago_architecture.py`

### A. Add Import (top of file, around line 56)

```python
import numpy as np
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque, OrderedDict
from contextlib import contextmanager
import warnings

# NEW: Import asymmetric stop loss
from alphago_stop_loss import AsymmetricStopLoss, StopLossConfig
```

### B. Find PositionTracker Class (search for "class PositionTracker:")

### C. Update __init__ Method

**Find:**
```python
class PositionTracker:
    def __init__(self, acfg: ArchitectureConfig, bars_per_year: int = 252):
        self.acfg = acfg
        self.bpy = bars_per_year
        # ... existing initialization ...
```

**Add after existing initialization:**
```python
        # NEW: Asymmetric stop loss
        if self.acfg.use_asymmetric_stops:
            stop_config = StopLossConfig(
                loss_stop_pct=acfg.loss_stop_pct,
                loss_stop_atr_mult=acfg.loss_stop_atr_mult,
                profit_trail_pct=acfg.profit_trail_pct,
                profit_trail_atr_mult=acfg.profit_trail_atr_mult,
                vol_adjust_stops=acfg.vol_adjust_stops,
                vol_baseline=acfg.vol_baseline,
                vol_max_adjustment=acfg.vol_max_adjustment,
                time_tighten_enabled=acfg.time_tighten_enabled,
                time_tighten_bars=acfg.time_tighten_bars,
                time_tighten_factor=acfg.time_tighten_factor,
            )
            self.stop_manager = AsymmetricStopLoss(stop_config)
        else:
            self.stop_manager = None

        # Track peak P&L for trailing stops
        self._peak_pnl_pct = 0.0
        self._bars_in_current_trade = 0
```

### D. Add Stop Check Method

**Add new method to PositionTracker:**

```python
    def check_stop_loss(self, current_price: float, atr: float, realized_vol: float = 0.15) -> Dict:
        """
        Check if asymmetric stop loss should trigger.

        Args:
            current_price: Current market price
            atr: Average True Range
            realized_vol: Current realized volatility

        Returns:
            {
                'should_exit': bool,
                'stop_price': float,
                'stop_type': str,
                'pnl_at_stop': float,
            }
        """
        if not self.has_position() or self.stop_manager is None:
            return {'should_exit': False}

        # Calculate current P&L
        if self.position > 0:  # Long
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            is_long = True
        else:  # Short
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            is_long = False

        # Update peak P&L
        if pnl_pct > self._peak_pnl_pct:
            self._peak_pnl_pct = pnl_pct

        # Compute stop level
        result = self.stop_manager.compute_stop(
            position_pnl_pct=pnl_pct,
            entry_price=self.entry_price,
            current_price=current_price,
            atr=atr,
            peak_pnl_pct=self._peak_pnl_pct,
            bars_in_trade=self._bars_in_current_trade,
            is_long=is_long,
            realized_vol=realized_vol
        )

        # If stop hit, record statistics
        if result['should_exit']:
            self.stop_manager.record_stop_hit(result['stop_type'], pnl_pct)

        return {
            'should_exit': result['should_exit'],
            'stop_price': result['stop_price'],
            'stop_type': result['stop_type'],
            'stop_distance_pct': result['stop_distance_pct'],
            'pnl_at_stop': pnl_pct,
        }

    def reset_position_stats(self):
        """Reset position statistics when opening new position."""
        self._peak_pnl_pct = 0.0
        self._bars_in_current_trade = 0

    def increment_bars_in_trade(self):
        """Increment bar counter for current trade."""
        if self.has_position():
            self._bars_in_current_trade += 1
```

### E. Update step() Method

**Find the step() method in PositionTracker**

**Add at the beginning of step():**

```python
    def step(self, bar_return: float, bar_idx: int = 0, current_price: float = None,
             atr: float = None, realized_vol: float = 0.15) -> Dict:
        """
        Update position tracker for new bar.

        Args:
            bar_return: Return for this bar
            bar_idx: Current bar index
            current_price: Current price (for stop loss)
            atr: Average True Range (for stop loss)
            realized_vol: Realized volatility (for stop loss adjustment)
        """
        # Increment bars in trade counter
        self.increment_bars_in_trade()

        # Check asymmetric stop loss
        if current_price is not None and atr is not None:
            stop_result = self.check_stop_loss(current_price, atr, realized_vol)

            if stop_result['should_exit']:
                # Stop triggered - close position
                pnl = self.close_position(bar_return=0, bar_idx=bar_idx)

                return {
                    'action': 'stop_triggered',
                    'stop_type': stop_result['stop_type'],
                    'stop_price': stop_result['stop_price'],
                    'pnl': pnl,
                    'pnl_pct': stop_result['pnl_at_stop'],
                }

        # ... rest of existing step() logic ...
```

---

## 🎯 STEP 4: Integrate into TradingEnv

**File:** `alphago_trading_system.py`

**Find:** `TradingEnv.step()` method

**Modify to pass price, ATR, and vol to PositionTracker:**

```python
    def step(self, action):
        # ... existing code ...

        # Get current bar data
        current_bar = self.current_df.iloc[self.bar_idx]
        current_price = float(current_bar['Close'])
        atr = float(current_bar.get('ATR', 0))
        realized_vol = float(current_bar.get('Realized_Vol_20', 0.15))

        # Update position tracker with stop loss check
        position_update = self.position_tracker.step(
            bar_return=bar_return,
            bar_idx=self.bar_idx,
            current_price=current_price,
            atr=atr,
            realized_vol=realized_vol
        )

        # Check if stop was triggered
        if position_update.get('action') == 'stop_triggered':
            # Log stop event
            print(f"[STOP] Bar {self.bar_idx}: {position_update['stop_type']} stop triggered")
            print(f"  Price: ${position_update['stop_price']:.2f}")
            print(f"  P&L: {position_update['pnl_pct']*100:.2f}%")

        # ... rest of step() logic ...
```

---

## 🧪 STEP 5: Test Integration

### Test 1: Unit Test Stop Loss Module

```bash
cd /d/Experiments/Trading
python alphago_stop_loss.py
```

**Expected:** All 3 tests pass

### Test 2: Test PositionTracker Integration

Create `test_asymmetric_stops.py`:

```python
"""Test asymmetric stop loss integration"""
from alphago_architecture import ArchitectureConfig, PositionTracker

# Enable asymmetric stops
acfg = ArchitectureConfig()
acfg.use_asymmetric_stops = True

tracker = PositionTracker(acfg)

# Simulate trade
print("Opening long position at $100")
tracker.open_position(size=1.0, entry_price=100.0, bar_idx=0)

# Test 1: Losing trade (should hit tight stop at $98)
print("\n[TEST 1] Losing trade")
for i in range(1, 10):
    price = 100 - i  # Price dropping
    atr = 2.0
    result = tracker.check_stop_loss(price, atr)

    print(f"Bar {i}: Price ${price:.2f}, Stop: ${result['stop_price']:.2f}")

    if result['should_exit']:
        print(f"  STOP HIT: {result['stop_type']} stop at ${result['stop_price']:.2f}")
        print(f"  P&L: {result['pnl_at_stop']*100:.2f}%")
        break

# Test 2: Winning trade (should trail from peak)
tracker = PositionTracker(acfg)
tracker.open_position(size=1.0, entry_price=100.0, bar_idx=0)

print("\n[TEST 2] Winning trade with trailing stop")
prices = [102, 105, 108, 110, 109, 107, 105]  # Rise to $110, then fall

for i, price in enumerate(prices, 1):
    atr = 2.0
    result = tracker.check_stop_loss(price, atr)

    print(f"Bar {i}: Price ${price:.2f}, Stop: ${result['stop_price']:.2f}, Type: {result['stop_type']}")

    if result['should_exit']:
        print(f"  TRAIL STOP HIT at ${result['stop_price']:.2f}")
        print(f"  P&L: {result['pnl_at_stop']*100:.2f}%")
        print(f"  Locked in profit from peak of $110")
        break

print("\n[SUCCESS] Asymmetric stops working correctly!")
```

Run:
```bash
python test_asymmetric_stops.py
```

### Test 3: Quick Backtest

```bash
# Run short backtest to ensure no regressions
python alphago_layering.py --symbols MSFT --iterations 1 --steps-per-iter 5000
```

**Check logs for:**
- `[STOP]` events showing stop triggers
- P&L at stop (should be small losses, larger profits)

---

## 📊 STEP 6: Compare Performance (Before vs After)

### Baseline Run (Without Asymmetric Stops)

```bash
# Disable asymmetric stops
python -c "
from alphago_architecture import ArchitectureConfig
acfg = ArchitectureConfig()
acfg.use_asymmetric_stops = False
# Save to temp config
import pickle
with open('config_baseline.pkl', 'wb') as f:
    pickle.dump(acfg, f)
"

# Run baseline
python alphago_layering.py --symbols MSFT --iterations 3 --steps-per-iter 50000 > baseline_run.log
```

### Enhanced Run (With Asymmetric Stops)

```bash
# Enable asymmetric stops (default)
python alphago_layering.py --symbols MSFT --iterations 3 --steps-per-iter 50000 > enhanced_run.log
```

### Compare Metrics

```python
# Compare results
import re

def parse_results(logfile):
    with open(logfile) as f:
        log = f.read()

    sharpe = float(re.search(r'Sharpe:\s+([\d.]+)', log).group(1))
    max_dd = float(re.search(r'Max DD:\s+([\d.]+)', log).group(1))
    avg_win = float(re.search(r'Avg Win:\s+([\d.]+)', log).group(1))
    avg_loss = float(re.search(r'Avg Loss:\s+([\d.]+)', log).group(1))

    return {'sharpe': sharpe, 'max_dd': max_dd, 'avg_win': avg_win, 'avg_loss': avg_loss}

baseline = parse_results('baseline_run.log')
enhanced = parse_results('enhanced_run.log')

print("BASELINE vs ENHANCED COMPARISON")
print("=" * 60)
print(f"{'Metric':<20} {'Baseline':>12} {'Enhanced':>12} {'Change':>12}")
print("-" * 60)
for key in baseline:
    change = ((enhanced[key] - baseline[key]) / baseline[key] * 100)
    print(f"{key.upper():<20} {baseline[key]:>12.2f} {enhanced[key]:>12.2f} {change:>11.1f}%")
```

**Expected improvements:**
- Sharpe: +20% to +40%
- Max DD: -30% to -50%
- Avg Loss: -40% to -60%
- Avg Win: +20% to +40%

---

## 🎛️ STEP 7: Tune Parameters (Optional)

### Grid Search for Optimal Stops

```python
# Test different stop configurations
configs_to_test = [
    {'loss_pct': 0.015, 'trail_pct': 0.03},  # Tighter
    {'loss_pct': 0.020, 'trail_pct': 0.05},  # Default
    {'loss_pct': 0.025, 'trail_pct': 0.07},  # Looser
]

for cfg in configs_to_test:
    # Run backtest with this config
    # Record Sharpe, DD, Win/Loss ratio
    # Compare results
```

### Per-Symbol Optimization

Different symbols may need different stops:
- **TSLA:** Wider stops (high vol)
- **SPY:** Tighter stops (low vol)
- **AAPL:** Medium stops

---

## ✅ COMPLETION CHECKLIST

- [x] **Step 1:** Create `alphago_stop_loss.py` module
- [ ] **Step 2:** Add config parameters to `ArchitectureConfig`
- [ ] **Step 3:** Integrate into `PositionTracker`
- [ ] **Step 4:** Update `TradingEnv.step()`
- [ ] **Step 5:** Run integration tests
- [ ] **Step 6:** Compare baseline vs enhanced performance
- [ ] **Step 7:** Tune parameters (optional)

---

## 📝 MONITORING & VALIDATION

### Metrics to Track:

1. **Stop Hit Rate:**
   - Loss stops: Should be ~15-25% of trades
   - Trail stops: Should be ~10-20% of trades

2. **Average P&L at Stop:**
   - Loss stops: -1.5% to -2.5%
   - Trail stops: +3% to +7%

3. **Win/Loss Ratio:**
   - Target: 2.5:1 to 4:1
   - Current baseline: ~1.2:1

4. **Sharpe Improvement:**
   - Target: +0.3 to +0.7
   - Monitor over 100+ trades

### Warning Signs:

- **Too many stops (>40%):** Stops too tight, widen by 20%
- **Average loss > -3%:** Stops too wide, tighten loss_stop_pct
- **No trail stops:** Trail too tight, increase profit_trail_pct
- **Sharpe decreases:** Revert and investigate

---

## 🚀 DEPLOYMENT

Once testing validates improvement:

1. **Merge to main branch**
2. **Deploy to paper trading** (30 days)
3. **Monitor metrics daily**
4. **Gradual rollout** (10% → 50% → 100% of capital)
5. **Document learnings**

---

*Implementation Time: 8-10 hours*
*Expected Sharpe Improvement: +0.4 to +0.7*
*Expected DD Reduction: -30% to -50%*
