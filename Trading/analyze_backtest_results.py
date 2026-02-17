#!/usr/bin/env python3
"""
DEEP ANALYSIS SCRIPT FOR ALPHA-TRADE BACKTEST RESULTS
Performs comprehensive statistical analysis to identify root causes and actionable fixes.
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA FILES")
print("=" * 80)

# Load comprehensive trades
trades_df = pd.read_csv(r'C:\Users\dilip\alpha_trade_v2_results\comprehensive_trades.csv')
print(f"[+] Loaded {len(trades_df)} trades from comprehensive_trades.csv")

# Load symbol summary
symbol_df = pd.read_csv(r'C:\Users\dilip\alpha_trade_v2_results\symbol_summary.csv')
print(f"[+] Loaded {len(symbol_df)} symbols from symbol_summary.csv")

# Load decision log
with open(r'C:\Users\dilip\alpha_trade_v2_results\decision_log.json', 'r') as f:
    decision_log = json.load(f)
print(f"[+] Loaded decision_log.json")

# Load trade decisions (with indicators)
trade_decisions_df = pd.read_csv(r'C:\Users\dilip\alpha_trade_v2_results\trade_decisions.csv')
print(f"[+] Loaded {len(trade_decisions_df)} trade decisions with indicators")

print()

# ============================================================================
# 1. TRADE PATTERN ANALYSIS
# ============================================================================

print("=" * 80)
print("1. TRADE PATTERN ANALYSIS")
print("=" * 80)

# Clean P&L values
trades_df['trade_pnl_clean'] = pd.to_numeric(trades_df['trade_pnl'], errors='coerce')

# Classify winners and losers
winners = trades_df[trades_df['result'] == 'WIN'].copy()
losers = trades_df[trades_df['result'] == 'LOSS'].copy()

print(f"\n[BASIC STATS]")
print(f"   Total Trades: {len(trades_df)}")
print(f"   Wins: {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%)")
print(f"   Losses: {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%)")
print(f"   Total P&L: ${trades_df['trade_pnl_clean'].sum():,.2f}")
print(f"   Avg Win: ${winners['trade_pnl_clean'].mean():,.2f}")
print(f"   Avg Loss: ${losers['trade_pnl_clean'].mean():,.2f}")
print(f"   Win/Loss Ratio: {abs(winners['trade_pnl_clean'].mean() / losers['trade_pnl_clean'].mean()):.2f}")

# Holding period analysis
print(f"\n[HOLDING PERIOD]")
print(f"   Winners avg: {winners['holding_bars'].mean():.1f} bars (median: {winners['holding_bars'].median():.0f})")
print(f"   Losers avg: {losers['holding_bars'].mean():.1f} bars (median: {losers['holding_bars'].median():.0f})")
_, p_holding = stats.mannwhitneyu(winners['holding_bars'], losers['holding_bars'], alternative='two-sided')
sig = 'SIGNIFICANT' if p_holding < 0.05 else 'NOT SIGNIFICANT'
print(f"   Mann-Whitney U test p-value: {p_holding:.4f} [{sig}]")

# Action type analysis
print(f"\n[ACTION TYPE ANALYSIS]")
for action in trades_df['entry_action'].unique():
    action_trades = trades_df[trades_df['entry_action'] == action]
    action_wins = len(action_trades[action_trades['result'] == 'WIN'])
    win_rate = action_wins / len(action_trades) * 100 if len(action_trades) > 0 else 0
    avg_pnl = action_trades['trade_pnl_clean'].mean()
    print(f"   {action:15s}: {len(action_trades):3d} trades | Win Rate: {win_rate:5.1f}% | Avg P&L: ${avg_pnl:10,.2f}")

# Entry reason analysis
print(f"\n[NOTE] TOP 10 ENTRY REASONS (by frequency):")
entry_reasons = trades_df['entry_reason'].value_counts().head(10)
for reason, count in entry_reasons.items():
    reason_trades = trades_df[trades_df['entry_reason'] == reason]
    win_rate = len(reason_trades[reason_trades['result'] == 'WIN']) / len(reason_trades) * 100
    avg_pnl = reason_trades['trade_pnl_clean'].mean()
    reason_short = reason[:60] + "..." if len(reason) > 60 else reason
    print(f"   {count:2d}x | WR:{win_rate:5.1f}% | P&L:${avg_pnl:9,.0f} | {reason_short}")

# Extract keywords from entry reasons
print(f"\n[SEARCH] ENTRY REASON KEYWORD ANALYSIS:")
def extract_keywords(reason_str):
    """Extract meaningful keywords from entry reason"""
    keywords = []
    if 'MACD bullish' in reason_str: keywords.append('MACD_bullish')
    if 'MACD bearish' in reason_str: keywords.append('MACD_bearish')
    if 'RSI overbought' in reason_str or 'RSI elevated' in reason_str: keywords.append('RSI_overbought')
    if 'RSI oversold' in reason_str or 'RSI near oversold' in reason_str: keywords.append('RSI_oversold')
    if 'Near upper BB' in reason_str: keywords.append('BB_upper')
    if 'Near lower BB' in reason_str: keywords.append('BB_lower')
    if 'Strong trend (ADX)' in reason_str: keywords.append('ADX_strong')
    if 'Uptrend' in reason_str: keywords.append('Uptrend')
    if 'Downtrend' in reason_str: keywords.append('Downtrend')
    if 'Moving avg bullish cross' in reason_str: keywords.append('MA_bull_cross')
    if 'Positive value est' in reason_str: keywords.append('Value_positive')
    if 'Negative value est' in reason_str: keywords.append('Value_negative')
    if 'Full long conviction' in reason_str: keywords.append('FULL_LONG_conv')
    if 'Full short conviction' in reason_str: keywords.append('FULL_SHORT_conv')
    return keywords

trades_df['entry_keywords'] = trades_df['entry_reason'].apply(extract_keywords)
keyword_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0})
for idx, row in trades_df.iterrows():
    for kw in row['entry_keywords']:
        keyword_stats[kw]['trades'] += 1
        if row['result'] == 'WIN':
            keyword_stats[kw]['wins'] += 1
        keyword_stats[kw]['pnl'] += row['trade_pnl_clean']

keyword_df = pd.DataFrame([
    {
        'keyword': kw,
        'count': stats['trades'],
        'win_rate': stats['wins'] / stats['trades'] * 100,
        'avg_pnl': stats['pnl'] / stats['trades'],
        'total_pnl': stats['pnl']
    }
    for kw, stats in keyword_stats.items()
]).sort_values('count', ascending=False)

print(keyword_df.to_string(index=False))

# Time-based patterns
print(f"\n[DATE] TIME-BASED PATTERNS:")
trades_df['entry_year'] = pd.to_datetime(trades_df['entry_time']).dt.year
trades_df['entry_month'] = pd.to_datetime(trades_df['entry_time']).dt.month
yearly_stats = trades_df.groupby('entry_year').agg({
    'trade_pnl_clean': ['count', 'sum', 'mean'],
    'result': lambda x: (x == 'WIN').sum() / len(x) * 100
}).round(2)
print("\nBy Year:")
print(yearly_stats)

# Sequential patterns (streaks)
print(f"\n[RANDOM] SEQUENTIAL PATTERNS (Win/Loss Streaks):")
results = trades_df.sort_values('entry_time')['result'].tolist()
current_streak = 1
max_win_streak = 0
max_loss_streak = 0
current_type = results[0]
for i in range(1, len(results)):
    if results[i] == current_type:
        current_streak += 1
    else:
        if current_type == 'WIN':
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
        current_streak = 1
        current_type = results[i]
if current_type == 'WIN':
    max_win_streak = max(max_win_streak, current_streak)
else:
    max_loss_streak = max(max_loss_streak, current_streak)

print(f"   Max win streak: {max_win_streak}")
print(f"   Max loss streak: {max_loss_streak}")

# ============================================================================
# 2. INDICATOR ANALYSIS (if we have the data)
# ============================================================================

print("\n" + "=" * 80)
print("2. INDICATOR ANALYSIS")
print("=" * 80)

if len(trade_decisions_df) > 1:  # If we have multiple trades with indicators
    print("\n[WARN]  Limited indicator data available. Analyzing single trade:")
    print(trade_decisions_df.to_string())
else:
    print("\n[WARN]  Only one trade with full indicator data available.")
    print("     Falling back to entry_reason text analysis (see section 1).")

# ============================================================================
# 3. SYMBOL-SPECIFIC ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("3. SYMBOL-SPECIFIC ANALYSIS")
print("=" * 80)

# Clean symbol summary
symbol_df['win_rate_float'] = symbol_df['win_rate'].str.replace('%', '').astype(float)
symbol_df['total_pnl_clean'] = symbol_df['total_pnl'].str.replace('$', '').str.replace(',', '').astype(float)

# Classify symbols
consistent_winners = symbol_df[(symbol_df['win_rate_float'] == 100) & (symbol_df['losses'] == 0)]
consistent_losers = symbol_df[(symbol_df['win_rate_float'] == 0) & (symbol_df['wins'] == 0)]
mixed = symbol_df[(symbol_df['wins'] > 0) & (symbol_df['losses'] > 0)]

print(f"\n[WIN] CONSISTENT WINNERS ({len(consistent_winners)} symbols):")
for idx, row in consistent_winners.sort_values('total_pnl_clean', ascending=False).head(10).iterrows():
    print(f"   {row['symbol']:15s}: {row['trades']:2.0f} trades | P&L: ${row['total_pnl_clean']:10,.2f}")

print(f"\n[LOSS] CONSISTENT LOSERS ({len(consistent_losers)} symbols):")
for idx, row in consistent_losers.sort_values('total_pnl_clean').head(10).iterrows():
    print(f"   {row['symbol']:15s}: {row['trades']:2.0f} trades | P&L: ${row['total_pnl_clean']:10,.2f}")

print(f"\n[TARGET] MIXED PERFORMANCE ({len(mixed)} symbols):")
for idx, row in mixed.sort_values('total_pnl_clean', ascending=False).head(5).iterrows():
    print(f"   {row['symbol']:15s}: {row['trades']:2.0f} trades | WR: {row['win_rate_float']:5.1f}% | P&L: ${row['total_pnl_clean']:10,.2f}")

# ============================================================================
# 4. POSITION SIZING ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("4. POSITION SIZING ANALYSIS")
print("=" * 80)

position_size_stats = trades_df.groupby('entry_action').agg({
    'trade_pnl_clean': ['count', 'mean', 'std'],
    'result': lambda x: (x == 'WIN').sum() / len(x) * 100
})
position_size_stats.columns = ['Count', 'Avg_PnL', 'Std_PnL', 'Win_Rate']
print("\n[SIZE] Position Size Performance:")
print(position_size_stats.round(2))

# FULL vs HALF comparison for LONGs
full_long = trades_df[trades_df['entry_action'] == 'FULL_LONG']
half_long = trades_df[trades_df['entry_action'] == 'HALF_LONG']
print(f"\n[SEARCH] FULL_LONG vs HALF_LONG:")
print(f"   FULL_LONG: {len(full_long)} trades | WR: {len(full_long[full_long['result']=='WIN'])/len(full_long)*100:.1f}% | Avg: ${full_long['trade_pnl_clean'].mean():,.2f}")
print(f"   HALF_LONG: {len(half_long)} trades | WR: {len(half_long[half_long['result']=='WIN'])/len(half_long)*100:.1f}% | Avg: ${half_long['trade_pnl_clean'].mean():,.2f}")

# FULL vs HALF comparison for SHORTs
full_short = trades_df[trades_df['entry_action'] == 'FULL_SHORT']
half_short = trades_df[trades_df['entry_action'] == 'HALF_SHORT']
if len(full_short) > 0:
    print(f"   FULL_SHORT: {len(full_short)} trades | WR: {len(full_short[full_short['result']=='WIN'])/len(full_short)*100:.1f}% | Avg: ${full_short['trade_pnl_clean'].mean():,.2f}")
if len(half_short) > 0:
    print(f"   HALF_SHORT: {len(half_short)} trades | WR: {len(half_short[half_short['result']=='WIN'])/len(half_short)*100:.1f}% | Avg: ${half_short['trade_pnl_clean'].mean():,.2f}")

# ============================================================================
# 5. ENTRY/EXIT TIMING ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("5. ENTRY/EXIT TIMING ANALYSIS")
print("=" * 80)

# Group by simplified entry reason
def simplify_entry_reason(reason):
    """Extract primary signal"""
    if 'MACD bullish' in reason: return 'MACD_bullish'
    if 'MACD bearish' in reason: return 'MACD_bearish'
    if 'RSI overbought' in reason or 'RSI elevated' in reason: return 'RSI_overbought'
    if 'RSI oversold' in reason: return 'RSI_oversold'
    if 'Uptrend' in reason: return 'Uptrend'
    if 'Downtrend' in reason: return 'Downtrend'
    if 'Moving avg bullish cross' in reason: return 'MA_cross_bull'
    if 'Near lower BB' in reason: return 'BB_lower'
    if 'Near upper BB' in reason: return 'BB_upper'
    return 'Other'

trades_df['primary_signal'] = trades_df['entry_reason'].apply(simplify_entry_reason)
signal_performance = trades_df.groupby('primary_signal').agg({
    'trade_pnl_clean': ['count', 'mean', 'sum'],
    'result': lambda x: (x == 'WIN').sum() / len(x) * 100
}).round(2)
signal_performance.columns = ['Count', 'Avg_PnL', 'Total_PnL', 'Win_Rate']
signal_performance = signal_performance.sort_values('Total_PnL', ascending=False)

print("\n[TARGET] ENTRY SIGNAL EFFECTIVENESS:")
print(signal_performance)

# Holding period vs P&L
print(f"\n[TIME]  HOLDING PERIOD VS P&L:")
for period_range in [(1,5), (6,15), (16,30), (31,100)]:
    subset = trades_df[(trades_df['holding_bars'] >= period_range[0]) &
                       (trades_df['holding_bars'] <= period_range[1])]
    if len(subset) > 0:
        win_rate = len(subset[subset['result'] == 'WIN']) / len(subset) * 100
        avg_pnl = subset['trade_pnl_clean'].mean()
        print(f"   {period_range[0]:3d}-{period_range[1]:3d} bars: {len(subset):3d} trades | WR: {win_rate:5.1f}% | Avg: ${avg_pnl:10,.2f}")

# ============================================================================
# 6. FAILURE MODE IDENTIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("6. FAILURE MODE IDENTIFICATION - TOP 20 LOSING TRADES")
print("=" * 80)

top_losses = losers.nlargest(20, 'trade_pnl_clean', keep='first')[['symbol', 'entry_time', 'exit_time',
                                                                     'entry_action', 'holding_bars',
                                                                     'trade_pnl_clean', 'entry_reason', 'exit_reason']]

for idx, (i, row) in enumerate(top_losses.iterrows(), 1):
    print(f"\n#{idx}. {row['symbol']} | {row['entry_action']} | ${row['trade_pnl_clean']:,.2f} | {row['holding_bars']} bars")
    print(f"    Entry ({row['entry_time']}): {row['entry_reason'][:80]}")
    print(f"    Exit  ({row['exit_time']}): {row['exit_reason'][:80]}")

print("\n[SEARCH] COMMON FAILURE PATTERNS:")
# Analyze entry reasons for top losses
top_loss_keywords = []
for reason in top_losses['entry_reason']:
    top_loss_keywords.extend(extract_keywords(reason))
failure_keyword_counts = Counter(top_loss_keywords)
print("\nMost common entry signals in top 20 losses:")
for kw, count in failure_keyword_counts.most_common(10):
    # Compare to overall frequency
    overall_freq = keyword_stats[kw]['trades']
    loss_freq = count / 20 * 100
    overall_rate = overall_freq / len(trades_df) * 100
    print(f"   {kw:20s}: {count:2d}/20 ({loss_freq:5.1f}%) vs overall {overall_freq:2d}/{len(trades_df)} ({overall_rate:5.1f}%)")

# ============================================================================
# 7. VALIDATION FAILURE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("7. VALIDATION FAILURE ROOT CAUSE")
print("=" * 80)

print("\n[INFO] From decision_log.json:")
print(f"   Total symbols analyzed: {decision_log.get('total_symbols_analyzed', 'N/A')}")
print(f"   Total trades: {decision_log.get('total_trades', 'N/A')}")
print(f"   Total wins: {decision_log.get('total_wins', 'N/A')}")
print(f"   Total losses: {decision_log.get('total_losses', 'N/A')}")
print(f"   Total P&L: ${decision_log.get('total_pnl', 'N/A'):,.2f}")

if 'symbol_summaries' in decision_log and len(decision_log['symbol_summaries']) > 0:
    print(f"\n   Symbol summaries available: {len(decision_log['symbol_summaries'])}")
    for sym_summary in decision_log['symbol_summaries']:
        print(f"   - {sym_summary['symbol']}: {sym_summary['trades']} trades, Win rate: {sym_summary['win_rate']*100:.1f}%")

# ============================================================================
# GENERATE REPORT SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING MARKDOWN REPORT...")
print("=" * 80)

# This will be saved to the markdown report
report_data = {
    'total_trades': len(trades_df),
    'win_rate': len(winners) / len(trades_df) * 100,
    'total_pnl': trades_df['trade_pnl_clean'].sum(),
    'avg_win': winners['trade_pnl_clean'].mean(),
    'avg_loss': losers['trade_pnl_clean'].mean(),
    'keyword_df': keyword_df,
    'signal_performance': signal_performance,
    'consistent_winners': consistent_winners,
    'consistent_losers': consistent_losers,
    'top_losses': top_losses,
    'failure_keywords': failure_keyword_counts,
    'position_size_stats': position_size_stats
}

print(f"\n[+] Analysis complete. Data ready for report generation.")
print(f"[+] Save to: d:\\Experiments\\Trading\\DEEP_ANALYSIS_REPORT.md")
