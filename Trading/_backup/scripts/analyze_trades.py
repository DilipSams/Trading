#!/usr/bin/env python3
"""
Comprehensive Trade Analysis Script
Extracts all trades from backtest results and generates detailed CSV reports
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
import argparse

def extract_trades_from_decision_log(log_path):
    """Extract trade data from decision_log.json"""
    with open(log_path, 'r') as f:
        data = json.load(f)

    trades = []
    for symbol, symbol_data in data.items():
        for trade in symbol_data.get('trades', []):
            trade['symbol'] = symbol
            trades.append(trade)

    return trades

def extract_trade_entries_from_events(events_path):
    """Extract trade_entries from events.jsonl files"""
    trade_entries = []

    if not Path(events_path).exists():
        return trade_entries

    with open(events_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            event = json.loads(line)

            # Look for trade_entries in eval_summary
            if event.get('_type') == 'eval_summary':
                continue  # Summary doesn't have individual entries

            # Check if this event contains trade_entries
            if 'trade_entries' in event:
                for entry in event['trade_entries']:
                    entry['symbol'] = event.get('symbol', 'UNKNOWN')
                    trade_entries.append(entry)

    return trade_entries

def analyze_latest_run(results_dir='C:\\Users\\dilip\\alpha_trade_v2_results'):
    """Analyze the latest backtest run"""
    results_path = Path(results_dir)

    print("\n" + "="*100)
    print("COMPREHENSIVE TRADE ANALYSIS")
    print("="*100)

    # Load decision log
    decision_log_path = results_path / 'decision_log.json'
    trade_csv_path = results_path / 'trade_decisions.csv'

    if not decision_log_path.exists():
        print(f"\n[!] ERROR: Could not find {decision_log_path}")
        return None

    print(f"\n[*] Loading trades from: {decision_log_path}")

    # Load trades from decision log
    with open(decision_log_path, 'r') as f:
        decision_data = json.load(f)

    # Extract all trades
    all_trades = []
    symbol_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0})

    # Get trades from top-level trades array
    trades_list = decision_data.get('trades', [])

    for trade in trades_list:
        symbol = trade.get('symbol', 'UNKNOWN')

        trade_data = {
            'symbol': symbol,
            'entry_time': trade.get('entry_time'),
            'exit_time': trade.get('exit_time'),
            'entry_price': trade.get('entry_price'),
            'exit_price': trade.get('exit_price'),
            'entry_action': trade.get('entry_action'),
            'exit_action': trade.get('exit_action'),
            'holding_bars': trade.get('holding_bars'),
            'trade_pnl': trade.get('trade_pnl'),
            'result': trade.get('result'),
            'entry_reason': trade.get('entry_reason'),
            'exit_reason': trade.get('exit_reason'),
        }
        all_trades.append(trade_data)

        # Update symbol statistics
        pnl = trade.get('trade_pnl', 0)
        symbol_stats[symbol]['trades'] += 1
        symbol_stats[symbol]['total_pnl'] += pnl

        if pnl > 0:
            symbol_stats[symbol]['wins'] += 1
        else:
            symbol_stats[symbol]['losses'] += 1

    # Calculate win rates
    for symbol in symbol_stats:
        stats = symbol_stats[symbol]
        stats['win_rate'] = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0

    # Sort by P&L
    all_trades_sorted = sorted(all_trades, key=lambda x: abs(x['trade_pnl'] or 0), reverse=True)

    # Generate comprehensive CSV
    output_csv = results_path / 'comprehensive_trades.csv'
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'entry_action', 'exit_action', 'holding_bars', 'trade_pnl', 'result',
            'entry_reason', 'exit_reason'
        ])
        writer.writeheader()
        writer.writerows(all_trades)

    print(f"[+] Exported {len(all_trades)} trades to: {output_csv}")

    # Generate symbol summary CSV
    summary_csv = results_path / 'symbol_summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'symbol', 'trades', 'wins', 'losses', 'win_rate', 'total_pnl'
        ])
        writer.writeheader()

        for symbol in sorted(symbol_stats.keys(), key=lambda x: symbol_stats[x]['total_pnl'], reverse=True):
            stats = symbol_stats[symbol]
            writer.writerow({
                'symbol': symbol,
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': f"{stats['win_rate']:.1f}%",
                'total_pnl': f"${stats['total_pnl']:,.2f}",
            })

    print(f"[+] Exported symbol summary to: {summary_csv}")

    # Display summary
    print("\n" + "="*100)
    print("AGGREGATE SUMMARY")
    print("="*100)

    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t['trade_pnl'] > 0)
    total_losses = sum(1 for t in all_trades if t['trade_pnl'] <= 0)
    total_pnl = sum(t['trade_pnl'] for t in all_trades)
    avg_win = sum(t['trade_pnl'] for t in all_trades if t['trade_pnl'] > 0) / total_wins if total_wins > 0 else 0
    avg_loss = sum(t['trade_pnl'] for t in all_trades if t['trade_pnl'] <= 0) / total_losses if total_losses > 0 else 0
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    avg_hold = sum(t['holding_bars'] for t in all_trades) / total_trades if total_trades > 0 else 0

    print(f"\n  Total Trades:     {total_trades}")
    print(f"  Winners:          {total_wins} ({win_rate:.1f}%)")
    print(f"  Losers:           {total_losses} ({100-win_rate:.1f}%)")
    print(f"  Total P&L:        ${total_pnl:,.2f}")
    print(f"  Avg Win:          ${avg_win:,.2f}")
    print(f"  Avg Loss:         ${avg_loss:,.2f}")
    print(f"  Reward/Risk:      {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}")
    print(f"  Avg Hold:         {avg_hold:.1f} bars")

    # Top winners
    print("\n" + "="*100)
    print("TOP 10 WINNING TRADES")
    print("="*100)
    print(f"  {'Symbol':<15} {'Entry':<12} {'Exit':<12} {'P&L':>15} {'Hold':>8} {'Entry Reason':<50}")
    print("  " + "-"*98)

    for trade in all_trades_sorted[:10]:
        if trade['trade_pnl'] > 0:
            print(f"  {trade['symbol']:<15} {trade['entry_time']:<12} {trade['exit_time']:<12} "
                  f"${trade['trade_pnl']:>13,.2f} {trade['holding_bars']:>6}b {trade['entry_reason'][:48]:<50}")

    # Top losers
    print("\n" + "="*100)
    print("TOP 10 LOSING TRADES")
    print("="*100)
    print(f"  {'Symbol':<15} {'Entry':<12} {'Exit':<12} {'P&L':>15} {'Hold':>8} {'Entry Reason':<50}")
    print("  " + "-"*98)

    losing_trades = [t for t in all_trades if t['trade_pnl'] <= 0]
    losing_trades_sorted = sorted(losing_trades, key=lambda x: x['trade_pnl'])

    for trade in losing_trades_sorted[:10]:
        print(f"  {trade['symbol']:<15} {trade['entry_time']:<12} {trade['exit_time']:<12} "
              f"${trade['trade_pnl']:>13,.2f} {trade['holding_bars']:>6}b {trade['entry_reason'][:48]:<50}")

    # Symbol performance
    print("\n" + "="*100)
    print("SYMBOL PERFORMANCE (Top 10 by P&L)")
    print("="*100)
    print(f"  {'Symbol':<15} {'Trades':>8} {'Win%':>8} {'P&L':>15}")
    print("  " + "-"*50)

    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
    for symbol, stats in sorted_symbols[:10]:
        print(f"  {symbol:<15} {stats['trades']:>8} {stats['win_rate']:>7.1f}% ${stats['total_pnl']:>13,.2f}")

    print("\n" + "="*100)
    print("\n[+] Analysis complete!")
    print(f"   [*] Comprehensive trades: {output_csv}")
    print(f"   [*] Symbol summary: {summary_csv}")
    print("="*100 + "\n")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'symbol_stats': symbol_stats,
    }

def analyze_run_output(run_dir):
    """Analyze a specific run_output directory"""
    run_path = Path(run_dir)

    if not run_path.exists():
        print(f"[!] ERROR: Directory not found: {run_dir}")
        return None

    events_file = run_path / 'events.jsonl'

    if not events_file.exists():
        print(f"[!] ERROR: No events.jsonl found in {run_dir}")
        return None

    print(f"\n[*] Analyzing run: {run_path.name}")

    trade_entries = extract_trade_entries_from_events(events_file)

    if trade_entries:
        print(f"   Found {len(trade_entries)} trade entries")

        # Export trade entries
        output_csv = run_path / 'trade_entries_detailed.csv'
        with open(output_csv, 'w', newline='') as f:
            if trade_entries:
                writer = csv.DictWriter(f, fieldnames=trade_entries[0].keys())
                writer.writeheader()
                writer.writerows(trade_entries)

        print(f"   [+] Exported to: {output_csv}")
    else:
        print(f"   ⚠️  No trade_entries found in events")

    return trade_entries

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze trading backtest results')
    parser.add_argument('--run-output', help='Specific run_output directory to analyze')
    parser.add_argument('--results-dir', default='C:\\Users\\dilip\\alpha_trade_v2_results',
                       help='Results directory path')

    args = parser.parse_args()

    if args.run_output:
        analyze_run_output(args.run_output)
    else:
        analyze_latest_run(args.results_dir)
