"""
Norgate Data Sample Script
===========================
Downloads and displays sample data from Norgate Data.
Requires: Norgate Data Updater running + active subscription.

Usage:
    python norgate_data_sample.py
    python norgate_data_sample.py --symbols TSLA,AAPL,MSFT
    python norgate_data_sample.py --watchlist "S&P 500"
    python norgate_data_sample.py --list-watchlists
    python norgate_data_sample.py --list-databases
"""

import argparse
import sys
from datetime import datetime

try:
    import norgatedata
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install norgatedata pandas numpy")
    sys.exit(1)


def check_norgate_connection():
    """Verify Norgate Data Updater is running and accessible."""
    try:
        # Try a simple call to see if the service is available
        status = norgatedata.status()
        print(f"Norgate Data Status: {status}")
        return True
    except Exception as e:
        print(f"Norgate Data connection check: {e}")
        # Try fetching a known symbol as fallback check
        try:
            _ = norgatedata.security_name("AAPL")
            print("Norgate Data: Connected (security lookup OK)")
            return True
        except Exception as e2:
            print(f"ERROR: Cannot connect to Norgate Data: {e2}")
            print("Make sure:")
            print("  1. Norgate Data Updater is installed and running")
            print("  2. You have an active Norgate subscription")
            print("  3. Data has been downloaded at least once via the Updater")
            return False


def list_watchlists():
    """Show all available watchlists."""
    try:
        wl = norgatedata.watchlists()
        print(f"\n{'='*60}")
        print(f"  AVAILABLE WATCHLISTS ({len(wl)})")
        print(f"{'='*60}")
        for w in sorted(wl):
            try:
                syms = norgatedata.watchlist_symbols(w)
                print(f"  {w:<40} ({len(syms)} symbols)")
            except Exception:
                print(f"  {w:<40} (unable to read)")
    except Exception as e:
        print(f"Error listing watchlists: {e}")


def list_databases():
    """Show all available databases."""
    try:
        dbs = norgatedata.databases()
        print(f"\n{'='*60}")
        print(f"  AVAILABLE DATABASES ({len(dbs)})")
        print(f"{'='*60}")
        for db in sorted(dbs):
            try:
                syms = norgatedata.database_symbols(db)
                print(f"  {db:<40} ({len(syms)} symbols)")
            except Exception:
                print(f"  {db:<40} (unable to read)")
    except Exception as e:
        print(f"Error listing databases: {e}")


def get_symbol_metadata(symbol):
    """Fetch metadata for a single symbol."""
    info = {}
    try:
        info["name"] = norgatedata.security_name(symbol)
    except Exception:
        info["name"] = "N/A"
    try:
        info["exchange"] = norgatedata.exchange_name(symbol)
    except Exception:
        info["exchange"] = "N/A"
    try:
        info["currency"] = norgatedata.currency(symbol)
    except Exception:
        info["currency"] = "N/A"
    try:
        info["first_date"] = norgatedata.first_quoted_date(symbol)
    except Exception:
        info["first_date"] = "N/A"
    try:
        info["last_date"] = norgatedata.last_quoted_date(symbol)
    except Exception:
        info["last_date"] = "N/A"
    try:
        info["asset_id"] = norgatedata.assetid(symbol)
    except Exception:
        info["asset_id"] = "N/A"
    return info


def fetch_price_data(symbol, start_date=None, end_date=None, interval="D",
                     adjustment="totalreturn"):
    """
    Fetch OHLCV price data for a symbol.

    Args:
        symbol: Ticker symbol (e.g., "TSLA")
        start_date: Start date (str "YYYY-MM-DD" or None for all)
        end_date: End date (str "YYYY-MM-DD" or None for latest)
        interval: "D" (daily), "W" (weekly), "M" (monthly)
        adjustment: "none", "capital", "totalreturn"

    Returns:
        pandas DataFrame with OHLCV data
    """
    adj_map = {
        "none": norgatedata.StockPriceAdjustmentType.NONE,
        "capital": norgatedata.StockPriceAdjustmentType.CAPITAL,
        "totalreturn": norgatedata.StockPriceAdjustmentType.TOTALRETURN,
    }
    adj_setting = adj_map.get(adjustment.lower(),
                              norgatedata.StockPriceAdjustmentType.TOTALRETURN)

    kwargs = {
        "symbol": symbol,
        "stock_price_adjustment_setting": adj_setting,
        "padding_setting": norgatedata.PaddingType.NONE,
        "timeseriesformat": "pandas-dataframe",
        "interval": interval,
    }

    if start_date:
        kwargs["start_date"] = pd.Timestamp(start_date)
    if end_date:
        kwargs["end_date"] = pd.Timestamp(end_date)

    df = norgatedata.price_timeseries(**kwargs)
    return df


def display_sample(symbol, df, metadata):
    """Pretty-print a sample of the data."""
    print(f"\n{'='*80}")
    print(f"  {symbol} — {metadata.get('name', 'N/A')}")
    print(f"{'='*80}")
    print(f"  Exchange:    {metadata.get('exchange', 'N/A')}")
    print(f"  Currency:    {metadata.get('currency', 'N/A')}")
    print(f"  Asset ID:    {metadata.get('asset_id', 'N/A')}")
    print(f"  First Date:  {metadata.get('first_date', 'N/A')}")
    print(f"  Last Date:   {metadata.get('last_date', 'N/A')}")
    print(f"  Total Bars:  {len(df)}")
    print(f"  Columns:     {list(df.columns)}")
    print(f"  Date Range:  {df.index[0]} to {df.index[-1]}" if len(df) > 0 else "  (empty)")

    if len(df) > 0:
        print(f"\n  --- First 5 rows ---")
        print(df.head().to_string(max_cols=10))

        print(f"\n  --- Last 5 rows ---")
        print(df.tail().to_string(max_cols=10))

        print(f"\n  --- Summary Statistics ---")
        stats = df.describe()
        # Show stats for key columns only
        key_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in stats.columns]
        if key_cols:
            print(stats[key_cols].round(2).to_string())

        # Show some quick metrics
        if "Close" in df.columns and len(df) > 1:
            returns = df["Close"].pct_change().dropna()
            print(f"\n  --- Quick Metrics ---")
            print(f"  Daily Return (mean):  {returns.mean()*100:.4f}%")
            print(f"  Daily Return (std):   {returns.std()*100:.4f}%")
            print(f"  Annual Sharpe (est):  {returns.mean()/returns.std()*np.sqrt(252):.2f}")
            print(f"  Max Drawdown:         {((df['Close']/df['Close'].cummax()) - 1).min()*100:.1f}%")
            print(f"  Total Return:         {(df['Close'].iloc[-1]/df['Close'].iloc[0] - 1)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Norgate Data Sample Viewer")
    parser.add_argument("--symbols", type=str, default="TSLA,AAPL,MSFT",
                        help="Comma-separated symbols (default: TSLA,AAPL,MSFT)")
    parser.add_argument("--watchlist", type=str, default="",
                        help="Norgate watchlist name to load symbols from")
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end", type=str, default="",
                        help="End date YYYY-MM-DD (default: latest)")
    parser.add_argument("--interval", type=str, default="D",
                        choices=["D", "W", "M"],
                        help="D=daily, W=weekly, M=monthly (default: D)")
    parser.add_argument("--adjustment", type=str, default="totalreturn",
                        choices=["none", "capital", "totalreturn"],
                        help="Price adjustment type (default: totalreturn)")
    parser.add_argument("--list-watchlists", action="store_true",
                        help="List all available watchlists")
    parser.add_argument("--list-databases", action="store_true",
                        help="List all available databases")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save data to CSV files in data_cache/norgate/")
    args = parser.parse_args()

    print("Norgate Data Sample Script")
    print("-" * 40)

    # Check connection
    if not check_norgate_connection():
        sys.exit(1)

    # List mode
    if args.list_watchlists:
        list_watchlists()
        return

    if args.list_databases:
        list_databases()
        return

    # Get symbols
    if args.watchlist:
        try:
            symbols = norgatedata.watchlist_symbols(args.watchlist)
            print(f"\nWatchlist '{args.watchlist}': {len(symbols)} symbols")
            print(f"  First 10: {symbols[:10]}")
            # Only sample first 5 for display
            symbols = symbols[:5]
        except Exception as e:
            print(f"Error loading watchlist '{args.watchlist}': {e}")
            return
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(f"\nFetching data for: {symbols}")
    print(f"Date range: {args.start} to {args.end or 'latest'}")
    print(f"Interval: {args.interval} | Adjustment: {args.adjustment}")

    # Fetch and display each symbol
    all_data = {}
    for sym in symbols:
        try:
            metadata = get_symbol_metadata(sym)
            df = fetch_price_data(
                sym,
                start_date=args.start,
                end_date=args.end or None,
                interval=args.interval,
                adjustment=args.adjustment,
            )
            all_data[sym] = df
            display_sample(sym, df, metadata)

            if args.save_csv and len(df) > 0:
                import os
                save_dir = os.path.join("data_cache", "norgate")
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"{sym}_{args.interval}.csv")
                df.to_csv(path)
                print(f"\n  Saved to: {path}")

        except Exception as e:
            print(f"\n  ERROR fetching {sym}: {e}")

    # Cross-symbol comparison
    if len(all_data) > 1:
        print(f"\n{'='*80}")
        print(f"  CROSS-SYMBOL COMPARISON")
        print(f"{'='*80}")
        rows = []
        for sym, df in all_data.items():
            if "Close" in df.columns and len(df) > 1:
                rets = df["Close"].pct_change().dropna()
                rows.append({
                    "Symbol": sym,
                    "Bars": len(df),
                    "First": str(df.index[0].date()) if hasattr(df.index[0], 'date') else str(df.index[0]),
                    "Last": str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1]),
                    "Last Close": f"${df['Close'].iloc[-1]:.2f}",
                    "Tot Return": f"{(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:.1f}%",
                    "Sharpe": f"{rets.mean()/rets.std()*np.sqrt(252):.2f}",
                    "Max DD": f"{((df['Close']/df['Close'].cummax())-1).min()*100:.1f}%",
                    "Avg Vol": f"{df['Volume'].mean()/1e6:.1f}M" if "Volume" in df.columns else "N/A",
                })
        if rows:
            comp = pd.DataFrame(rows)
            print(comp.to_string(index=False))

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
