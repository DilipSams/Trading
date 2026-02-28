#!/usr/bin/env python3
"""
Test Priority Filters - Simpler approach than complex AMT features

Priority 1: ATR Regime Filter (avoid choppy markets)
Priority 2: Better Trade Management (trail stops, protect winners)
Priority 3: ATR Normalization (volatility-adjusted signals)
"""

import numpy as np
import pandas as pd
import yfinance as yf

print("="*80)
print("PRIORITY FILTERS VALIDATION")
print("="*80)

# Download data
print("\nDownloading SPY 2-year data...")
df = yf.download('SPY', period='2y', interval='1d', progress=False)

# Compute indicators
print("Computing indicators...")

# MACD
ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
df['MACD'] = ema12 - ema26
df['Signal'] = df['MACD'].ewm(span=9).mean()

# ATR
h_l = df['High'] - df['Low']
h_c = abs(df['High'] - df['Close'].shift())
l_c = abs(df['Low'] - df['Close'].shift())
tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
df['ATR'] = tr.rolling(14).mean()

# Average ATR for regime filter
df['ATR_avg'] = df['ATR'].rolling(50).mean()

df = df.dropna().copy()

# Generate MACD crossover signals
df['cross_up'] = ((df['MACD'] > df['Signal']) &
                  (df['MACD'].shift(1) <= df['Signal'].shift(1)))
df['cross_down'] = ((df['MACD'] < df['Signal']) &
                    (df['MACD'].shift(1) >= df['Signal'].shift(1)))

print(f"Valid bars: {len(df)}")
print(f"Total MACD crossovers: {df['cross_up'].sum() + df['cross_down'].sum()}")

# ============================================================================
# BASELINE: Original Strategy (no filters)
# ============================================================================

print("\n" + "="*80)
print("BASELINE: MACD Crossover (No Filters)")
print("="*80)

baseline_trades = []
position = 0  # 0 = flat, 1 = long, -1 = short
entry_price = 0
entry_idx = 0

for i in range(len(df)):
    row = df.iloc[i]

    # Entry
    if position == 0:
        if row['cross_up'].item():
            position = 1
            entry_price = row['Close'].item()
            entry_idx = i
        elif row['cross_down'].item():
            position = -1
            entry_price = row['Close'].item()
            entry_idx = i

    # Exit on opposite signal
    elif position == 1 and row['cross_down'].item().item():
        exit_price = row['Close'].item()
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        baseline_trades.append({
            'entry_idx': entry_idx,
            'exit_idx': i,
            'side': 'LONG',
            'entry': entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': i - entry_idx
        })
        position = 0

    elif position == -1 and row['cross_up'].item().item():
        exit_price = row['Close'].item()
        pnl_pct = (entry_price - exit_price) / entry_price * 100
        baseline_trades.append({
            'entry_idx': entry_idx,
            'exit_idx': i,
            'side': 'SHORT',
            'entry': entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': i - entry_idx
        })
        position = 0

baseline_df = pd.DataFrame(baseline_trades)
if len(baseline_df) > 0:
    bl_wr = (baseline_df['pnl_pct'] > 0).sum() / len(baseline_df) * 100
    bl_avg_win = baseline_df[baseline_df['pnl_pct'] > 0]['pnl_pct'].mean() if (baseline_df['pnl_pct'] > 0).any() else 0
    bl_avg_loss = baseline_df[baseline_df['pnl_pct'] < 0]['pnl_pct'].mean() if (baseline_df['pnl_pct'] < 0).any() else 0
    bl_total_pnl = baseline_df['pnl_pct'].sum()

    print(f"\nTrades: {len(baseline_df)}")
    print(f"Win Rate: {bl_wr:.1f}%")
    print(f"Avg Win: {bl_avg_win:.2f}%")
    print(f"Avg Loss: {bl_avg_loss:.2f}%")
    print(f"Total P&L: {bl_total_pnl:+.2f}%")
    print(f"Reward/Risk: {abs(bl_avg_win/bl_avg_loss):.2f}" if bl_avg_loss != 0 else "N/A")
else:
    print("\nNo trades!")
    bl_wr = 0
    bl_total_pnl = 0

# ============================================================================
# PRIORITY 1: ATR Regime Filter
# ============================================================================

print("\n" + "="*80)
print("PRIORITY 1: ATR Regime Filter")
print("="*80)
print("Filter: Only trade when ATR > 0.5x average (avoid choppy markets)")

p1_trades = []
position = 0
entry_price = 0
entry_idx = 0

for i in range(len(df)):
    row = df.iloc[i]

    # Priority 1 Filter: Check ATR regime
    atr_ok = row['ATR'].item() > 0.5 * row['ATR_avg'].item()

    if position == 0 and atr_ok:  # Only enter if ATR filter passes
        if row['cross_up'].item():
            position = 1
            entry_price = row['Close'].item()
            entry_idx = i
        elif row['cross_down'].item():
            position = -1
            entry_price = row['Close'].item()
            entry_idx = i

    elif position == 1 and row['cross_down'].item():
        exit_price = row['Close'].item()
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        p1_trades.append({
            'entry_idx': entry_idx,
            'exit_idx': i,
            'side': 'LONG',
            'entry': entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': i - entry_idx
        })
        position = 0

    elif position == -1 and row['cross_up'].item():
        exit_price = row['Close'].item()
        pnl_pct = (entry_price - exit_price) / entry_price * 100
        p1_trades.append({
            'entry_idx': entry_idx,
            'exit_idx': i,
            'side': 'SHORT',
            'entry': entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': i - entry_idx
        })
        position = 0

p1_df = pd.DataFrame(p1_trades)
if len(p1_df) > 0:
    p1_wr = (p1_df['pnl_pct'] > 0).sum() / len(p1_df) * 100
    p1_avg_win = p1_df[p1_df['pnl_pct'] > 0]['pnl_pct'].mean() if (p1_df['pnl_pct'] > 0).any() else 0
    p1_avg_loss = p1_df[p1_df['pnl_pct'] < 0]['pnl_pct'].mean() if (p1_df['pnl_pct'] < 0).any() else 0
    p1_total_pnl = p1_df['pnl_pct'].sum()

    print(f"\nTrades: {len(p1_df)} (vs {len(baseline_df)} baseline)")
    print(f"Win Rate: {p1_wr:.1f}% (vs {bl_wr:.1f}% baseline)")
    print(f"Avg Win: {p1_avg_win:.2f}%")
    print(f"Avg Loss: {p1_avg_loss:.2f}%")
    print(f"Total P&L: {p1_total_pnl:+.2f}% (vs {bl_total_pnl:+.2f}% baseline)")

    wr_improvement = p1_wr - bl_wr
    pnl_improvement = p1_total_pnl - bl_total_pnl

    print(f"\nIMPROVEMENT:")
    print(f"  Win Rate: {wr_improvement:+.1f} pp")
    print(f"  Total P&L: {pnl_improvement:+.2f}%")

    if p1_wr > bl_wr and p1_total_pnl > bl_total_pnl:
        print("\n[PASS] Filter improves both WR and P&L")
        rec1 = "IMPLEMENT"
    elif p1_wr > bl_wr or p1_total_pnl > bl_total_pnl:
        print("\n[WEAK] Filter improves one metric")
        rec1 = "CONSIDER"
    else:
        print("\n[FAIL] Filter does not improve performance")
        rec1 = "SKIP"
else:
    print("\nNo trades after filter!")
    rec1 = "SKIP"
    p1_wr = 0
    p1_total_pnl = 0

# ============================================================================
# PRIORITY 2: Better Trade Management (Trailing Stop)
# ============================================================================

print("\n" + "="*80)
print("PRIORITY 2: Better Trade Management")
print("="*80)
print("Strategy: Trail stop at recent swing low (5-bar), protect 5%+ winners")

p2_trades = []
position = 0
entry_price = 0
entry_idx = 0
trailing_stop = 0

for i in range(5, len(df)):  # Start at bar 5 for swing calculation
    row = df.iloc[i]

    # Calculate swing low (5-bar low)
    swing_low = df['Low'].iloc[i-5:i].min()

    if position == 0:
        if row['cross_up'].item():
            position = 1
            entry_price = row['Close'].item()
            entry_idx = i
            trailing_stop = swing_low  # Initial stop
        elif row['cross_down'].item():
            position = -1
            entry_price = row['Close'].item()
            entry_idx = i
            # For shorts, use swing high
            trailing_stop = df['High'].iloc[i-5:i].max()

    elif position == 1:
        # Update trailing stop
        new_stop = max(trailing_stop, swing_low)
        trailing_stop = new_stop

        # Calculate unrealized P&L
        unrealized_pnl_pct = (row['Close'] - entry_price) / entry_price * 100

        # Exit conditions
        hit_stop = row['Low'].item() <= trailing_stop
        opposite_signal = row['cross_down']
        protect_winner = unrealized_pnl_pct > 5.0  # If up 5%+, tighten stop

        if hit_stop:
            exit_price = trailing_stop
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            p2_trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'side': 'LONG',
                'entry': entry_price,
                'exit': exit_price,
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_idx,
                'exit_reason': 'trailing_stop'
            })
            position = 0
        elif opposite_signal:
            exit_price = row['Close'].item()
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            p2_trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'side': 'LONG',
                'entry': entry_price,
                'exit': exit_price,
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_idx,
                'exit_reason': 'signal'
            })
            position = 0

    elif position == -1:
        # Similar logic for shorts (inverted)
        new_stop = min(trailing_stop, df['High'].iloc[i-5:i].max())
        trailing_stop = new_stop

        unrealized_pnl_pct = (entry_price - row['Close']) / entry_price * 100

        hit_stop = row['High'].item() >= trailing_stop
        opposite_signal = row['cross_up']

        if hit_stop:
            exit_price = trailing_stop
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            p2_trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'side': 'SHORT',
                'entry': entry_price,
                'exit': exit_price,
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_idx,
                'exit_reason': 'trailing_stop'
            })
            position = 0
        elif opposite_signal:
            exit_price = row['Close'].item()
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            p2_trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'side': 'SHORT',
                'entry': entry_price,
                'exit': exit_price,
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_idx,
                'exit_reason': 'signal'
            })
            position = 0

p2_df = pd.DataFrame(p2_trades)
if len(p2_df) > 0:
    p2_wr = (p2_df['pnl_pct'] > 0).sum() / len(p2_df) * 100
    p2_avg_win = p2_df[p2_df['pnl_pct'] > 0]['pnl_pct'].mean() if (p2_df['pnl_pct'] > 0).any() else 0
    p2_avg_loss = p2_df[p2_df['pnl_pct'] < 0]['pnl_pct'].mean() if (p2_df['pnl_pct'] < 0).any() else 0
    p2_total_pnl = p2_df['pnl_pct'].sum()

    stopped_out = (p2_df['exit_reason'] == 'trailing_stop').sum()
    signal_exits = (p2_df['exit_reason'] == 'signal').sum()

    print(f"\nTrades: {len(p2_df)} (vs {len(baseline_df)} baseline)")
    print(f"  Stopped out: {stopped_out}")
    print(f"  Signal exits: {signal_exits}")
    print(f"Win Rate: {p2_wr:.1f}% (vs {bl_wr:.1f}% baseline)")
    print(f"Avg Win: {p2_avg_win:.2f}%")
    print(f"Avg Loss: {p2_avg_loss:.2f}%")
    print(f"Total P&L: {p2_total_pnl:+.2f}% (vs {bl_total_pnl:+.2f}% baseline)")

    wr_improvement = p2_wr - bl_wr
    pnl_improvement = p2_total_pnl - bl_total_pnl

    print(f"\nIMPROVEMENT:")
    print(f"  Win Rate: {wr_improvement:+.1f} pp")
    print(f"  Total P&L: {pnl_improvement:+.2f}%")

    # Check if stops helped protect winners
    protected_winners = p2_df[(p2_df['pnl_pct'] > 3) & (p2_df['exit_reason'] == 'trailing_stop')]
    print(f"  Protected winners (>3% saved by stop): {len(protected_winners)}")

    if p2_wr > bl_wr and abs(p2_avg_loss) < abs(baseline_df[baseline_df['pnl_pct'] < 0]['pnl_pct'].mean()):
        print("\n[PASS] Trailing stops improve WR and reduce avg loss")
        rec2 = "IMPLEMENT"
    elif p2_total_pnl > bl_total_pnl:
        print("\n[WEAK] Trailing stops improve P&L")
        rec2 = "CONSIDER"
    else:
        print("\n[FAIL] Trailing stops do not improve performance")
        rec2 = "SKIP"
else:
    print("\nNo trades!")
    rec2 = "SKIP"
    p2_wr = 0
    p2_total_pnl = 0

# ============================================================================
# COMBINED: Priority 1 + Priority 2
# ============================================================================

print("\n" + "="*80)
print("COMBINED: ATR Filter + Trailing Stops")
print("="*80)

combined_trades = []
position = 0
entry_price = 0
entry_idx = 0
trailing_stop = 0

for i in range(5, len(df)):
    row = df.iloc[i]

    # Priority 1: ATR filter
    atr_ok = row['ATR'].item() > 0.5 * row['ATR_avg'].item()

    # Swing calculation
    swing_low = df['Low'].iloc[i-5:i].min()
    swing_high = df['High'].iloc[i-5:i].max()

    if position == 0 and atr_ok:  # Only enter if ATR OK
        if row['cross_up'].item():
            position = 1
            entry_price = row['Close'].item()
            entry_idx = i
            trailing_stop = swing_low
        elif row['cross_down'].item():
            position = -1
            entry_price = row['Close'].item()
            entry_idx = i
            trailing_stop = swing_high

    elif position == 1:
        trailing_stop = max(trailing_stop, swing_low)

        hit_stop = row['Low'].item() <= trailing_stop
        opposite_signal = row['cross_down']

        if hit_stop:
            exit_price = trailing_stop
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            combined_trades.append({
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_idx,
                'side': 'LONG',
                'exit_reason': 'stop'
            })
            position = 0
        elif opposite_signal:
            exit_price = row['Close'].item()
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            combined_trades.append({
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_idx,
                'side': 'LONG',
                'exit_reason': 'signal'
            })
            position = 0

    elif position == -1:
        trailing_stop = min(trailing_stop, swing_high)

        hit_stop = row['High'].item() >= trailing_stop
        opposite_signal = row['cross_up']

        if hit_stop:
            exit_price = trailing_stop
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            combined_trades.append({
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_idx,
                'side': 'SHORT',
                'exit_reason': 'stop'
            })
            position = 0
        elif opposite_signal:
            exit_price = row['Close'].item()
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            combined_trades.append({
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_idx,
                'side': 'SHORT',
                'exit_reason': 'signal'
            })
            position = 0

comb_df = pd.DataFrame(combined_trades)
if len(comb_df) > 0:
    comb_wr = (comb_df['pnl_pct'] > 0).sum() / len(comb_df) * 100
    comb_total_pnl = comb_df['pnl_pct'].sum()

    print(f"\nTrades: {len(comb_df)} (vs {len(baseline_df)} baseline)")
    print(f"Win Rate: {comb_wr:.1f}% (vs {bl_wr:.1f}% baseline)")
    print(f"Total P&L: {comb_total_pnl:+.2f}% (vs {bl_total_pnl:+.2f}% baseline)")

    wr_improvement = comb_wr - bl_wr
    pnl_improvement = comb_total_pnl - bl_total_pnl

    print(f"\nTOTAL IMPROVEMENT:")
    print(f"  Win Rate: {wr_improvement:+.1f} pp")
    print(f"  Total P&L: {pnl_improvement:+.2f}%")

    if comb_wr > bl_wr and comb_total_pnl > bl_total_pnl:
        print("\n[PASS] Combined filters significantly improve performance")
        rec_comb = "IMPLEMENT"
    else:
        print("\n[FAIL] Combined filters do not improve enough")
        rec_comb = "SKIP"
else:
    print("\nNo trades after filters!")
    rec_comb = "SKIP"

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL RECOMMENDATIONS")
print("="*80)

print(f"\nBASELINE (no filters):")
print(f"  Trades: {len(baseline_df)}, WR: {bl_wr:.1f}%, P&L: {bl_total_pnl:+.2f}%")

print(f"\nPriority 1 (ATR Filter):        {rec1}")
if len(p1_df) > 0:
    print(f"  Trades: {len(p1_df)}, WR: {p1_wr:.1f}%, P&L: {p1_total_pnl:+.2f}%")

print(f"\nPriority 2 (Trailing Stops):    {rec2}")
if len(p2_df) > 0:
    print(f"  Trades: {len(p2_df)}, WR: {p2_wr:.1f}%, P&L: {p2_total_pnl:+.2f}%")

print(f"\nCombined (P1 + P2):             {rec_comb}")
if len(comb_df) > 0:
    print(f"  Trades: {len(comb_df)}, WR: {comb_wr:.1f}%, P&L: {comb_total_pnl:+.2f}%")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if rec_comb == "IMPLEMENT":
    print("\nImplement Priority 1 + Priority 2 filters")
    print("Expected impact: Significantly improved win rate and P&L")
elif rec1 == "IMPLEMENT" or rec2 == "IMPLEMENT":
    print("\nImplement filters that passed individually")
    if rec1 == "IMPLEMENT":
        print("  - ATR regime filter")
    if rec2 == "IMPLEMENT":
        print("  - Trailing stops")
else:
    print("\nNone of the priority filters improved performance")
    print("Need to revisit strategy fundamentals")

print("="*80)
