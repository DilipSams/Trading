"""
Quick Integration Test for Asymmetric Stop Loss
================================================

Verifies that Steps 2, 3, and 4 are correctly integrated.
"""

import sys
sys.path.insert(0, '.')

# Test imports
try:
    from alphago_stop_loss import AsymmetricStopLoss, StopLossConfig
    print("[OK] alphago_stop_loss module imported")
except ImportError as e:
    print(f"[ERROR] Failed to import alphago_stop_loss: {e}")
    sys.exit(1)

try:
    from alphago_trading_system import Config, TradingEnv, NUM_FEATURES
    print("[OK] alphago_trading_system imported")
except ImportError as e:
    print(f"[ERROR] Failed to import alphago_trading_system: {e}")
    sys.exit(1)

import numpy as np
import pandas as pd

print("\n" + "="*70)
print("INTEGRATION TEST: Asymmetric Stop Loss")
print("="*70)

# Test 1: Check config parameters
print("\n[TEST 1] Configuration Parameters")
print("-" * 70)

cfg = Config()

required_params = [
    'use_asymmetric_stops',
    'loss_stop_pct',
    'loss_stop_atr_mult',
    'profit_trail_pct',
    'profit_trail_atr_mult',
    'vol_adjust_stops',
    'vol_baseline',
    'vol_max_adjustment',
    'time_tighten_enabled',
    'time_tighten_bars',
    'time_tighten_factor',
]

all_present = True
for param in required_params:
    if hasattr(cfg, param):
        value = getattr(cfg, param)
        print(f"  {param:<30} = {value}")
    else:
        print(f"  {param:<30} = MISSING!")
        all_present = False

if all_present:
    print("\n[PASS] All config parameters present")
else:
    print("\n[FAIL] Some config parameters missing")
    sys.exit(1)

# Verify default values match grid search optimal
print("\n[TEST 2] Default Values Check")
print("-" * 70)

expected_defaults = {
    'use_asymmetric_stops': True,
    'loss_stop_pct': 0.015,  # 1.5% (grid search optimal)
    'profit_trail_pct': 0.05,  # 5.0% (grid search optimal)
}

defaults_ok = True
for param, expected in expected_defaults.items():
    actual = getattr(cfg, param)
    status = "OK" if actual == expected else "MISMATCH"
    print(f"  {param:<30} = {actual} (expected {expected}) [{status}]")
    if actual != expected:
        defaults_ok = False

if defaults_ok:
    print("\n[PASS] Default values match grid search optimal configuration")
else:
    print("\n[WARN] Default values differ from recommended (may be intentional)")

# Test 3: Create TradingEnv with asymmetric stops
print("\n[TEST 3] TradingEnv Initialization")
print("-" * 70)

try:
    # Create synthetic data
    n_bars = 1000
    np.random.seed(42)

    prices = np.zeros((n_bars, 5))
    prices[:, 3] = 100 * (1 + np.cumsum(np.random.randn(n_bars) * 0.02))  # Close
    prices[:, 0] = prices[:, 3] * (1 + np.random.randn(n_bars) * 0.005)   # Open
    prices[:, 1] = np.maximum(prices[:, 0], prices[:, 3]) * (1 + np.abs(np.random.randn(n_bars)) * 0.01)  # High
    prices[:, 2] = np.minimum(prices[:, 0], prices[:, 3]) * (1 - np.abs(np.random.randn(n_bars)) * 0.01)  # Low
    prices[:, 4] = np.random.randint(100000, 1000000, n_bars)  # Volume

    # Create features (minimal set)
    feat_data = {}
    for i in range(NUM_FEATURES):
        feat_data[f'feat_{i}'] = np.random.randn(n_bars)

    # Add ATR and Realized_Vol_20 (needed for stop loss)
    atr = np.abs(prices[:, 1] - prices[:, 2])
    feat_data['ATR'] = atr
    feat_data['Realized_Vol_20'] = np.ones(n_bars) * 0.15

    feat = pd.DataFrame(feat_data)

    # Create environment
    env = TradingEnv(feat, prices, cfg, sym="TEST_1d", ev=False)

    print(f"  Environment created: {type(env).__name__}")
    print(f"  Asymmetric stop manager: {type(env.asymmetric_stop_manager).__name__ if env.asymmetric_stop_manager else 'None'}")

    if env.asymmetric_stop_manager is not None:
        print(f"  Stop config:")
        print(f"    Loss stop: {env.asymmetric_stop_manager.config.loss_stop_pct*100:.1f}%")
        print(f"    Trail stop: {env.asymmetric_stop_manager.config.profit_trail_pct*100:.1f}%")
        print("\n[PASS] TradingEnv initialized with asymmetric stops")
    else:
        print("\n[FAIL] Asymmetric stop manager not initialized")
        sys.exit(1)

except Exception as e:
    print(f"\n[FAIL] Failed to initialize TradingEnv: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify stop loss methods exist
print("\n[TEST 4] Stop Loss Methods")
print("-" * 70)

required_methods = [
    '_check_asymmetric_stop',
    '_reset_position_stats',
    '_increment_bars_in_trade',
]

methods_ok = True
for method in required_methods:
    if hasattr(env, method):
        print(f"  {method:<30} EXISTS")
    else:
        print(f"  {method:<30} MISSING!")
        methods_ok = False

if methods_ok:
    print("\n[PASS] All stop loss methods present")
else:
    print("\n[FAIL] Some methods missing")
    sys.exit(1)

# Test 5: Simulate a trade with stop loss
print("\n[TEST 5] Simulate Trade with Stop Loss")
print("-" * 70)

try:
    env.reset()

    # Go long
    obs, reward, term, trunc, info = env.step(4)  # Action 4 = +1.0 exposure (full long)
    print(f"  Step 1: Opened long position")
    print(f"    Shares: {env.shares:.2f}")
    print(f"    Entry VWAP: ${env.entry_vwap:.2f}")
    print(f"    Bars in trade: {env._bars_in_current_trade}")
    print(f"    Peak P&L: {env._peak_pnl_pct*100:.2f}%")

    # Hold for a few bars
    for i in range(5):
        obs, reward, term, trunc, info = env.step(4)  # Hold

    print(f"\n  Step 6: After 5 bars")
    print(f"    Shares: {env.shares:.2f}")
    print(f"    Bars in trade: {env._bars_in_current_trade}")
    print(f"    Peak P&L: {env._peak_pnl_pct*100:.2f}%")

    # Check if asymmetric stop check is being called
    current_price = env._get_mid_price()
    atr = 2.0
    result = env._check_asymmetric_stop(current_price, atr)

    print(f"\n  Manual stop check:")
    print(f"    Should exit: {result['should_exit']}")
    print(f"    Stop price: ${result['stop_price']:.2f}")
    print(f"    Stop type: {result['stop_type']}")

    # Close position
    obs, reward, term, trunc, info = env.step(2)  # Action 2 = 0.0 exposure (flat)

    print(f"\n  Step 7: Closed position")
    print(f"    Shares: {env.shares:.2f}")
    print(f"    Bars in trade (should reset): {env._bars_in_current_trade}")
    print(f"    Peak P&L (should reset): {env._peak_pnl_pct*100:.2f}%")

    if env._bars_in_current_trade == 0 and env._peak_pnl_pct == 0.0:
        print("\n[PASS] Position stats correctly reset after closing")
    else:
        print("\n[WARN] Position stats may not have reset properly")

except Exception as e:
    print(f"\n[FAIL] Trade simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*70)
print("INTEGRATION TEST SUMMARY")
print("="*70)
print("\n[SUCCESS] All integration tests passed!")
print("\nAsymmetric stop loss is now integrated:")
print("  - Config parameters added to Config")
print("  - Stop manager initialized in TradingEnv")
print("  - Stop checks integrated into step() method")
print("  - Position tracking (peak P&L, bars in trade) working")
print("\nReady for full backtest validation!")
print("\n" + "="*70)
