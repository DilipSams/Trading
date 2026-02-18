#!/usr/bin/env python3
"""
Quick test to verify the bug fixes allow trades to execute.
Tests with MSFT only for 200 bars.
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from alphago_layering import build_default_pipeline
from alphago_architecture import ArchitectureConfig
import yfinance as yf

def main():
    print("=" * 80)
    print("  QUICK TRADE TEST - Verifying Bug Fixes")
    print("=" * 80)

    # Load config with bug fixes
    cfg = ArchitectureConfig()
    print(f"\n[OK] Bug fixes applied:")
    print(f"   - no_trade_threshold: {cfg.no_trade_threshold_pct} (was 0.005)")
    print(f"   - trend_slow_window: {cfg.trend_slow_window} (was 126)")
    print(f"   - value_lookback: {cfg.value_lookback} (was 252)")
    print(f"   - calendar_warmup: {cfg.calendar_warmup} (was 252)")

    # Download MSFT data only
    print(f"\n[DATA] Loading MSFT data...")
    df = yf.download('MSFT', start='2020-01-01', end='2024-12-31', progress=False)
    print(f"   Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")


    # Build pipeline
    print(f"\n[PIPELINE] Building pipeline with 12 alphas...")
    pipeline = build_default_pipeline(
        cfg,
        invert_trend=True,  # Known optimization
        invert_vol_premium=False,
        net=None  # No RL for quick test
    )

    print(f"   Pipeline initialized with {len(pipeline.alpha_factory.alpha_names)} alphas")

    # Run 200 bars
    print(f"\n[RUN] Running backtest for 200 bars...")

    closes = df['Close'].values
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values

    num_bars = min(200, len(df))
    trades_executed = 0
    positions = []
    alpha_signals = []
    ensemble_signals = []
    suppressions = 0

    for bar_idx in range(num_bars):
        # Step pipeline
        result = pipeline.step(
            bar_idx=bar_idx,
            closes=closes[:bar_idx+1],
            opens=opens[:bar_idx+1],
            highs=highs[:bar_idx+1],
            lows=lows[:bar_idx+1],
            volumes=volumes[:bar_idx+1],
        )

        # Track signals
        alpha_signals.append(result.combined_mu)
        ensemble_signals.append(result.target_exposure)
        positions.append(result.execution.executed_exposure)

        # Count trades (position changes)
        if bar_idx > 0:
            if abs(positions[-1] - positions[-2]) > 0.001:  # Position changed
                trades_executed += 1

        # Count suppressions
        if result.execution.was_suppressed:
            suppressions += 1

        # Log first 80 bars
        if bar_idx in [10, 30, 50, 70, 100, 150, 199]:
            active_alphas = sum(1 for s in result.signals.values() if s.is_active)
            print(f"   Bar {bar_idx:3d}: {active_alphas:2d}/12 alphas active, "
                  f"mu_hat={result.combined_mu:+.4f}, "
                  f"target={result.target_exposure:+.3f}, "
                  f"position={positions[-1]:+.3f}")

    # Results
    print(f"\n" + "=" * 80)
    print(f"  RESULTS")
    print(f"=" * 80)

    non_zero_positions = sum(1 for p in positions if abs(p) > 0.001)
    position_changes = sum(1 for i in range(1, len(positions)) if abs(positions[i] - positions[i-1]) > 0.001)

    avg_signal = np.mean([abs(s) for s in alpha_signals])
    max_position = max([abs(p) for p in positions])

    print(f"\n[ACTIVITY] Trading Activity:")
    print(f"   Bars processed:        {num_bars}")
    print(f"   Position changes:      {position_changes}")
    print(f"   Non-zero positions:    {non_zero_positions} ({non_zero_positions/num_bars*100:.1f}%)")
    print(f"   Suppressions:          {suppressions} ({suppressions/num_bars*100:.1f}%)")

    print(f"\n[SIGNALS] Signal Quality:")
    print(f"   Avg |mu_hat|:          {avg_signal:.4f}")
    print(f"   Max |position|:        {max_position:.3f}")
    print(f"   Non-zero signals:      {sum(1 for s in alpha_signals if abs(s) > 0.001)}/{num_bars}")

    # Verdict
    print(f"\n" + "=" * 80)
    if position_changes > 0:
        print(f"  [SUCCESS] Trades are executing!")
        print(f"     {position_changes} position changes detected over {num_bars} bars")
        print(f"     Bug fixes are working correctly.")
    else:
        print(f"  [FAILURE] Still no trades")
        print(f"     Possible issues:")
        print(f"     - All signals still canceling out (avg |mu_hat| = {avg_signal:.4f})")
        print(f"     - Suppressions too high ({suppressions}/{num_bars})")
        print(f"     - RLAlpha not active (requires trained network)")
    print(f"=" * 80)

    return position_changes > 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
