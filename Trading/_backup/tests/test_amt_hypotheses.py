#!/usr/bin/env python3
"""
Hypothesis Validation Tests for AMT Features

Tests three hypotheses BEFORE implementing:
1. Volume Confirmation: Do high-volume signals outperform?
2. Acceptance/Rejection: Do accepted breakouts work better than rejected?
3. ATR Normalization: Does normalizing signals improve IC?

If hypotheses fail Don't implement
If hypotheses pass Implement with confidence
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import existing system components
import sys
sys.path.append(str(Path(__file__).parent))

from alphago_trading_system import download_data, compute_indicators
from alphago_architecture import ArchitectureConfig

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_forward_returns(df, horizon=5):
 """Compute forward returns for IC calculation"""
 df['forward_return'] = df['Close'].pct_change(horizon).shift(-horizon)
 return df

def compute_ic(predictions, actuals):
 """Information Coefficient (Spearman correlation)"""
 valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
 if valid_mask.sum() < 10:
 return 0.0
 return np.corrcoef(predictions[valid_mask], actuals[valid_mask])[0, 1]

def compute_win_rate(df, signal_col, forward_return_col='forward_return'):
 """Win rate for directional signals"""
 signals = df[signal_col].values
 returns = df[forward_return_col].values

 # Long signals
 long_mask = signals > 0
 long_wins = 0
 if long_mask.sum() > 0:
 long_wins = ((signals > 0) & (returns > 0)).sum()
 long_wr = long_wins / long_mask.sum() * 100
 else:
 long_wr = 0

 # Short signals
 short_mask = signals < 0
 short_wins = 0
 if short_mask.sum() > 0:
 short_wins = ((signals < 0) & (returns < 0)).sum()
 short_wr = short_wins / short_mask.sum() * 100
 else:
 short_wr = 0

 # Combined
 total_signals = long_mask.sum() + short_mask.sum()
 if total_signals > 0:
 total_wins = long_wins + short_wins
 total_wr = total_wins / total_signals * 100
 else:
 total_wr = 0

 return {
 'total_signals': total_signals,
 'win_rate': total_wr,
 'long_wr': long_wr,
 'short_wr': short_wr
 }

# ============================================================================
# HYPOTHESIS 1: VOLUME CONFIRMATION
# ============================================================================

def test_volume_hypothesis(df, volume_threshold=1.5):
 """
 Hypothesis: High-volume signals outperform low-volume signals

 Expected:
 - High-volume MACD: 50-55% WR
 - Low-volume MACD: 35-40% WR
 - Difference: >10 percentage points
 """
 print("\n" + "="*80)
 print("HYPOTHESIS 1: VOLUME CONFIRMATION")
 print("="*80)

 # Compute volume ratio
 df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

 # Generate MACD signals
 macd = df['MACD']
 signal_line = df['MACD_signal']

 df['macd_bullish'] = ((macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))).astype(int)
 df['macd_bearish'] = ((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))).astype(int)
 df['macd_signal'] = df['macd_bullish'] - df['macd_bearish']

 # Split by volume
 high_vol_mask = df['volume_ratio'] > volume_threshold
 low_vol_mask = df['volume_ratio'] <= volume_threshold

 df['macd_high_vol'] = df['macd_signal'].where(high_vol_mask, 0)
 df['macd_low_vol'] = df['macd_signal'].where(low_vol_mask, 0)

 # Compute forward returns
 df = compute_forward_returns(df, horizon=5)

 # Calculate win rates
 all_signals = compute_win_rate(df, 'macd_signal')
 high_vol = compute_win_rate(df, 'macd_high_vol')
 low_vol = compute_win_rate(df, 'macd_low_vol')

 print(f"\n[*] MACD Signal Performance:")
 print(f" All MACD signals: {all_signals['total_signals']:>4} trades, WR = {all_signals['win_rate']:>5.1f}%")
 print(f" High-volume MACD: {high_vol['total_signals']:>4} trades, WR = {high_vol['win_rate']:>5.1f}%")
 print(f" Low-volume MACD: {low_vol['total_signals']:>4} trades, WR = {low_vol['win_rate']:>5.1f}%")

 improvement = high_vol['win_rate'] - low_vol['win_rate']
 print(f"\n Improvement: {improvement:+.1f} percentage points")

 # Statistical test
 if high_vol['total_signals'] > 20 and low_vol['total_signals'] > 20:
 # Compute IC for both
 high_vol_ic = compute_ic(
 df['macd_high_vol'].values,
 df['forward_return'].values
 )
 low_vol_ic = compute_ic(
 df['macd_low_vol'].values,
 df['forward_return'].values
 )

 print(f"\n IC (Information Coefficient):")
 print(f" High-volume: {high_vol_ic:>6.3f}")
 print(f" Low-volume: {low_vol_ic:>6.3f}")
 print(f" Improvement: {high_vol_ic - low_vol_ic:+.3f}")

 # Verdict
 print(f"\n{'='*80}")
 if improvement > 10 and high_vol['win_rate'] > 50:
 print("[OK] HYPOTHESIS CONFIRMED")
 print(" High-volume signals significantly outperform low-volume")
 print(" RECOMMENDATION: IMPLEMENT volume confirmation")
 elif improvement > 5:
 print("[!] HYPOTHESIS PARTIALLY CONFIRMED")
 print(" Modest improvement, but may be worth implementing")
 print(" RECOMMENDATION: CONSIDER implementing with caution")
 else:
 print("[NO] HYPOTHESIS REJECTED")
 print(" Volume does not provide meaningful edge")
 print(" RECOMMENDATION: DO NOT implement volume confirmation")
 print(f"{'='*80}")

 return {
 'passed': improvement > 10 and high_vol['win_rate'] > 50,
 'improvement': improvement,
 'high_vol_wr': high_vol['win_rate'],
 'low_vol_wr': low_vol['win_rate']
 }

# ============================================================================
# HYPOTHESIS 2: ACCEPTANCE/REJECTION
# ============================================================================

def test_acceptance_rejection_hypothesis(df, atr_threshold=0.3):
 """
 Hypothesis: Accepted breakouts outperform, rejected breakouts can be faded

 Expected:
 - Accepted breakouts: 60%+ WR (continuation)
 - Rejected breakouts (faded): 60%+ WR (reversal)
 - Non-confirmed breakouts: ~45% WR
 """
 print("\n" + "="*80)
 print("HYPOTHESIS 2: ACCEPTANCE/REJECTION LOGIC")
 print("="*80)

 # Find breakouts (20-bar high breaks)
 df['resistance'] = df['High'].rolling(20).max().shift(1)
 df['breakout'] = (df['High'] > df['resistance']).astype(int)

 # Classify acceptance/rejection
 df['close_vs_resistance'] = df['Close'] - df['resistance']
 df['atr_normalized_distance'] = df['close_vs_resistance'] / df['ATR']

 # Accepted: Closed well above resistance
 df['accepted'] = (df['breakout'] == 1) & (df['atr_normalized_distance'] > atr_threshold)

 # Rejected: Broke high but closed below resistance
 df['rejected'] = (df['breakout'] == 1) & (df['atr_normalized_distance'] < -atr_threshold)

 # Neutral: Broke but closed near resistance (ambiguous)
 df['neutral_breakout'] = (df['breakout'] == 1) & ~df['accepted'] & ~df['rejected']

 # Create signals
 df['acceptance_signal'] = df['accepted'].astype(int) # Long
 df['rejection_signal'] = -df['rejected'].astype(int) # Short (fade)
 df['combined_signal'] = df['acceptance_signal'] + df['rejection_signal']

 # Compute forward returns
 df = compute_forward_returns(df, horizon=10) # Longer horizon for breakouts

 # Calculate win rates
 all_breakouts = df['breakout'].sum()
 accepted_wr = compute_win_rate(df, 'acceptance_signal')
 rejected_wr = compute_win_rate(df, 'rejection_signal')
 combined_wr = compute_win_rate(df, 'combined_signal')

 print(f"\n Breakout Classification:")
 print(f" Total breakouts: {all_breakouts:>4}")
 print(f" Accepted (continued): {df['accepted'].sum():>4} ({df['accepted'].sum()/all_breakouts*100:.1f}%)")
 print(f" Rejected (failed): {df['rejected'].sum():>4} ({df['rejected'].sum()/all_breakouts*100:.1f}%)")
 print(f" Neutral (ambiguous): {df['neutral_breakout'].sum():>4} ({df['neutral_breakout'].sum()/all_breakouts*100:.1f}%)")

 print(f"\n Signal Performance:")
 print(f" Accepted breakouts (LONG): {accepted_wr['total_signals']:>4} trades, WR = {accepted_wr['win_rate']:>5.1f}%")
 print(f" Rejected breakouts (FADE): {rejected_wr['total_signals']:>4} trades, WR = {rejected_wr['win_rate']:>5.1f}%")
 print(f" Combined strategy: {combined_wr['total_signals']:>4} trades, WR = {combined_wr['win_rate']:>5.1f}%")

 # IC analysis
 if combined_wr['total_signals'] > 20:
 ic = compute_ic(
 df['combined_signal'].values,
 df['forward_return'].values
 )
 print(f"\n IC (Information Coefficient): {ic:.3f}")

 # Verdict
 print(f"\n{'='*80}")
 if accepted_wr['win_rate'] > 55 and rejected_wr['win_rate'] > 55:
 print("[OK] HYPOTHESIS CONFIRMED")
 print(" Both acceptance and rejection patterns show strong edge")
 print(" RECOMMENDATION: IMPLEMENT acceptance/rejection alpha")
 elif combined_wr['win_rate'] > 52:
 print("[!] HYPOTHESIS PARTIALLY CONFIRMED")
 print(" Combined strategy shows modest edge")
 print(" RECOMMENDATION: CONSIDER implementing")
 else:
 print("[NO] HYPOTHESIS REJECTED")
 print(" Acceptance/rejection does not provide meaningful edge")
 print(" RECOMMENDATION: DO NOT implement")
 print(f"{'='*80}")

 return {
 'passed': accepted_wr['win_rate'] > 55 and rejected_wr['win_rate'] > 55,
 'accepted_wr': accepted_wr['win_rate'],
 'rejected_wr': rejected_wr['win_rate'],
 'combined_wr': combined_wr['win_rate']
 }

# ============================================================================
# HYPOTHESIS 3: ATR NORMALIZATION
# ============================================================================

def test_atr_normalization_hypothesis(df):
 """
 Hypothesis: ATR-normalized signals have better IC than raw signals

 Expected:
 - Raw MACD IC: ~0.01-0.03
 - Normalized MACD IC: +0.01 to +0.02 improvement
 """
 print("\n" + "="*80)
 print("HYPOTHESIS 3: ATR NORMALIZATION")
 print("="*80)

 # Raw MACD signal
 macd = df['MACD'].values
 signal_line = df['MACD_signal'].values
 df['macd_raw'] = macd - signal_line

 # ATR-normalized MACD
 df['macd_normalized'] = df['macd_raw'] / df['ATR']

 # Compute forward returns
 df = compute_forward_returns(df, horizon=5)

 # Calculate ICs
 ic_raw = compute_ic(df['macd_raw'].values, df['forward_return'].values)
 ic_normalized = compute_ic(df['macd_normalized'].values, df['forward_return'].values)

 improvement = ic_normalized - ic_raw

 print(f"\n Information Coefficient Comparison:")
 print(f" Raw MACD IC: {ic_raw:>7.4f}")
 print(f" Normalized MACD IC: {ic_normalized:>7.4f}")
 print(f" Improvement: {improvement:>+7.4f}")

 # Test with RSI as well
 if 'RSI' in df.columns:
 df['rsi_signal'] = (df['RSI'] - 50) / 50 # Raw: -1 to +1
 df['rsi_normalized'] = (df['RSI'] - 50) / df['ATR'] # ATR normalized

 ic_rsi_raw = compute_ic(df['rsi_signal'].values, df['forward_return'].values)
 ic_rsi_norm = compute_ic(df['rsi_normalized'].values, df['forward_return'].values)

 print(f"\n Raw RSI IC: {ic_rsi_raw:>7.4f}")
 print(f" Normalized RSI IC: {ic_rsi_norm:>7.4f}")
 print(f" Improvement: {ic_rsi_norm - ic_rsi_raw:>+7.4f}")

 # Percentile analysis
 print(f"\n Signal Strength Distribution:")
 print(f" Raw MACD percentiles:")
 print(f" 10th: {np.nanpercentile(df['macd_raw'], 10):>8.3f}")
 print(f" 50th: {np.nanpercentile(df['macd_raw'], 50):>8.3f}")
 print(f" 90th: {np.nanpercentile(df['macd_raw'], 90):>8.3f}")

 print(f"\n Normalized MACD percentiles:")
 print(f" 10th: {np.nanpercentile(df['macd_normalized'], 10):>8.3f}")
 print(f" 50th: {np.nanpercentile(df['macd_normalized'], 50):>8.3f}")
 print(f" 90th: {np.nanpercentile(df['macd_normalized'], 90):>8.3f}")

 # Verdict
 print(f"\n{'='*80}")
 if improvement > 0.01:
 print("[OK] HYPOTHESIS CONFIRMED")
 print(" ATR normalization improves signal quality")
 print(" RECOMMENDATION: IMPLEMENT ATR normalization")
 elif improvement > 0.005:
 print("[!] HYPOTHESIS PARTIALLY CONFIRMED")
 print(" Modest improvement from normalization")
 print(" RECOMMENDATION: CONSIDER implementing")
 else:
 print("[NO] HYPOTHESIS REJECTED")
 print(" ATR normalization does not improve IC")
 print(" RECOMMENDATION: DO NOT implement")
 print(f"{'='*80}")

 return {
 'passed': improvement > 0.01,
 'ic_raw': ic_raw,
 'ic_normalized': ic_normalized,
 'improvement': improvement
 }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_tests(symbols=['SPY', 'QQQ'], period='2y'):
 """Run all hypothesis tests on multiple symbols"""

 print("\n" + "="*80)
 print("AMT HYPOTHESIS VALIDATION FRAMEWORK")
 print("="*80)
 print(f"\nTesting on: {', '.join(symbols)}")
 print(f"Period: {period}")
 print(f"\nThis will validate 3 hypotheses BEFORE implementing features:")
 print(" 1. Volume Confirmation")
 print(" 2. Acceptance/Rejection Logic")
 print(" 3. ATR Normalization")

 results = {
 'volume': [],
 'acceptance': [],
 'atr': []
 }

 for symbol in symbols:
 print("\n" + "="*80)
 print(f"Testing Symbol: {symbol}")
 print("="*80)

 # Download data
 print(f"\n[*] Downloading {symbol} data...")
 df = yf.download(symbol, period=period, progress=False)

 if df.empty:
 print(f"[!] Failed to download {symbol}, skipping...")
 continue

 print(f"[+] Downloaded {len(df)} bars")

 # Compute indicators
 print(f"[*] Computing technical indicators...")
 cfg = ArchitectureConfig()

 # Calculate basic indicators
 df['Returns'] = df['Close'].pct_change()

 # MACD
 exp1 = df['Close'].ewm(span=12, adjust=False).mean()
 exp2 = df['Close'].ewm(span=26, adjust=False).mean()
 df['MACD'] = exp1 - exp2
 df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

 # RSI
 delta = df['Close'].diff()
 gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
 loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
 rs = gain / loss
 df['RSI'] = 100 - (100 / (1 + rs))

 # ATR
 high_low = df['High'] - df['Low']
 high_close = np.abs(df['High'] - df['Close'].shift())
 low_close = np.abs(df['Low'] - df['Close'].shift())
 ranges = pd.concat([high_low, high_close, low_close], axis=1)
 true_range = np.max(ranges, axis=1)
 df['ATR'] = true_range.rolling(14).mean()

 # Drop NaN rows
 df = df.dropna()

 print(f"[+] Indicators computed, {len(df)} valid bars")

 # Run tests
 print("\n" + "-"*80)
 vol_result = test_volume_hypothesis(df)
 results['volume'].append(vol_result)

 print("\n" + "-"*80)
 acc_result = test_acceptance_rejection_hypothesis(df)
 results['acceptance'].append(acc_result)

 print("\n" + "-"*80)
 atr_result = test_atr_normalization_hypothesis(df)
 results['atr'].append(atr_result)

 # Aggregate results
 print("\n" + "="*80)
 print("AGGREGATE RESULTS ACROSS ALL SYMBOLS")
 print("="*80)

 # Volume
 print(f"\n[1] VOLUME CONFIRMATION:")
 vol_passed = sum(r['passed'] for r in results['volume'])
 avg_improvement = np.mean([r['improvement'] for r in results['volume']])
 avg_high_vol_wr = np.mean([r['high_vol_wr'] for r in results['volume']])
 avg_low_vol_wr = np.mean([r['low_vol_wr'] for r in results['volume']])

 print(f" Passed: {vol_passed}/{len(symbols)} symbols")
 print(f" Avg High-vol WR: {avg_high_vol_wr:.1f}%")
 print(f" Avg Low-vol WR: {avg_low_vol_wr:.1f}%")
 print(f" Avg Improvement: {avg_improvement:+.1f} pp")

 if vol_passed >= len(symbols) * 0.6:
 print(f" [OK] IMPLEMENT: Works on majority of symbols")
 else:
 print(f" [NO] SKIP: Inconsistent across symbols")

 # Acceptance/Rejection
 print(f"\n[2] ACCEPTANCE/REJECTION:")
 acc_passed = sum(r['passed'] for r in results['acceptance'])
 avg_accepted_wr = np.mean([r['accepted_wr'] for r in results['acceptance']])
 avg_rejected_wr = np.mean([r['rejected_wr'] for r in results['acceptance']])
 avg_combined_wr = np.mean([r['combined_wr'] for r in results['acceptance']])

 print(f" Passed: {acc_passed}/{len(symbols)} symbols")
 print(f" Avg Accepted WR: {avg_accepted_wr:.1f}%")
 print(f" Avg Rejected WR: {avg_rejected_wr:.1f}%")
 print(f" Avg Combined WR: {avg_combined_wr:.1f}%")

 if acc_passed >= len(symbols) * 0.6:
 print(f" [OK] IMPLEMENT: Works on majority of symbols")
 else:
 print(f" [NO] SKIP: Inconsistent across symbols")

 # ATR Normalization
 print(f"\n[3] ATR NORMALIZATION:")
 atr_passed = sum(r['passed'] for r in results['atr'])
 avg_ic_improvement = np.mean([r['improvement'] for r in results['atr']])
 avg_ic_raw = np.mean([r['ic_raw'] for r in results['atr']])
 avg_ic_norm = np.mean([r['ic_normalized'] for r in results['atr']])

 print(f" Passed: {atr_passed}/{len(symbols)} symbols")
 print(f" Avg Raw IC: {avg_ic_raw:.4f}")
 print(f" Avg Normalized IC: {avg_ic_norm:.4f}")
 print(f" Avg Improvement: {avg_ic_improvement:+.4f}")

 if atr_passed >= len(symbols) * 0.6:
 print(f" [OK] IMPLEMENT: Works on majority of symbols")
 else:
 print(f" [NO] SKIP: Inconsistent across symbols")

 # Final recommendation
 print("\n" + "="*80)
 print("FINAL RECOMMENDATIONS")
 print("="*80)

 print(f"\n[OK] IMPLEMENT:")
 if vol_passed >= len(symbols) * 0.6:
 print(f" - Volume Confirmation ({avg_improvement:+.1f} pp WR improvement)")
 if acc_passed >= len(symbols) * 0.6:
 print(f" - Acceptance/Rejection Alpha ({avg_combined_wr:.1f}% WR)")
 if atr_passed >= len(symbols) * 0.6:
 print(f" - ATR Normalization ({avg_ic_improvement:+.4f} IC improvement)")

 print(f"\n[NO] SKIP:")
 if vol_passed < len(symbols) * 0.6:
 print(f" - Volume Confirmation (inconsistent)")
 if acc_passed < len(symbols) * 0.6:
 print(f" - Acceptance/Rejection (inconsistent)")
 if atr_passed < len(symbols) * 0.6:
 print(f" - ATR Normalization (inconsistent)")

 print("\n" + "="*80)

if __name__ == '__main__':
 # Test on major indices and high-volume stocks
 test_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']

 print("\n" + "="*80)
 print("Starting Hypothesis Validation Tests...")
 print("="*80)

 run_all_tests(symbols=test_symbols, period='2y')

 print("\n" + "="*80)
 print("Validation Complete!")
 print("="*80)
