#!/usr/bin/env python
"""Test RL Observation Fix - Verify observations are constructed correctly"""

import numpy as np
import pandas as pd
from alphago_trading_system import (
    Config, SyntheticMarketGenerator, compute_indicators, build_feature_matrix, NUM_FEATURES
)
from alphago_architecture import build_default_pipeline, ArchitectureConfig

print("="*80)
print("  TESTING RL OBSERVATION FIX")
print("="*80)
print()

# Generate synthetic data
print("1. Generating synthetic data...")
cfg = Config(starting_capital=100000, n_iterations=1, total_timesteps_per_iter=1000)
gen = SyntheticMarketGenerator(cfg)
df = gen.generate(n=2000, p0=100.0)
print(f"   Generated {len(df)} bars")
print(f"   Columns: {list(df.columns)}")
print()

# Build features
print("2. Building feature matrix...")
try:
    df_with_indicators = compute_indicators(df)
    features = build_feature_matrix(df_with_indicators)
    print(f"   Features shape: {features.shape}")
    print(f"   NUM_FEATURES: {NUM_FEATURES}")
    print(f"   Features finite: {np.isfinite(features).all()}")
    print()
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test observation construction
print("3. Testing observation construction...")
window_size = 60
t = 100  # Test at bar 100
try:
    obs_window = features[t - window_size:t].copy()
    obs_window[:, -4:] = 0.0  # Zero out position state
    observation = obs_window.flatten().astype(np.float32)

    print(f"   Window shape: {obs_window.shape}")
    print(f"   Expected: ({window_size}, {NUM_FEATURES})")
    print(f"   Observation length: {len(observation)}")
    print(f"   Expected: {window_size * NUM_FEATURES}")
    print(f"   Observation finite: {np.isfinite(observation).all()}")
    print(f"   Observation range: [{observation.min():.3f}, {observation.max():.3f}]")
    print()
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test RL alpha generation
print("4. Testing RL alpha with observation...")
acfg = ArchitectureConfig()
pipeline = build_default_pipeline(acfg, bars_per_year=252, net=None)

closes = df['Close'].values
volumes = df.get('Volume', np.ones(len(df))).values

# Find RL alpha
rl_alpha = None
for name in pipeline.alpha_factory.alpha_names:
    alpha = pipeline.alpha_factory._alphas.get(name)
    from alphago_architecture import RLAlphaAdapter
    if isinstance(alpha, RLAlphaAdapter):
        rl_alpha = alpha
        print(f"   Found RL alpha: {name}")
        print(f"   Network: {type(alpha.net)}")
        break

if rl_alpha is None:
    print("   ERROR: RL alpha not found!")
    exit(1)

# Test signal generation WITHOUT observation
print("\n5. Testing signal WITHOUT observation...")
signal_no_obs = rl_alpha.generate(closes[:t+1], volumes[:t+1], None, t, observation=None)
print(f"   mu: {signal_no_obs.mu}")
print(f"   sigma: {signal_no_obs.sigma}")
print(f"   confidence: {signal_no_obs.confidence}")

# Test signal generation WITH observation (but no network)
print("\n6. Testing signal WITH observation (no network)...")
signal_with_obs = rl_alpha.generate(closes[:t+1], volumes[:t+1], None, t, observation=observation)
print(f"   mu: {signal_with_obs.mu}")
print(f"   sigma: {signal_with_obs.sigma}")
print(f"   confidence: {signal_with_obs.confidence}")

print("\n" + "="*80)
if signal_with_obs.mu == 0.0 and signal_with_obs.confidence == 0.0:
    print("  STATUS: RL alpha returns ZERO (expected - no trained network)")
    print("  OBSERVATION FIX: ✅ Working (no errors, but needs trained network)")
else:
    print("  STATUS: Unexpected result!")
    print("  OBSERVATION FIX: ❓ Check above output")
print("="*80)
