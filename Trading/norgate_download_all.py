"""
Norgate Data — Full Download to Parquet
========================================
Downloads ALL available data from Norgate and saves as Parquet files.

Parquet format:
  - Binary columnar with built-in checksums (no corruption)
  - Preserves exact dtypes (float64, datetime64, etc.)
  - Compressed (~3-5x smaller than CSV)
  - Loads instantly: pd.read_parquet("file.parquet")

Directory structure:
  D:\\Experiments\\norgate_data\\
    US_Equities/           <- 13,754 active stocks
      AAPL.parquet
      MSFT.parquet
      ...
    US_Equities_Delisted/  <- 1,362 delisted (survivorship-bias-free)
    US_Indices/            <- 1,614 indices
    Economic/              <- 157 macro indicators
    Forex_Spot/            <- 57 currency pairs
    Cash_Commodities/      <- 14 commodities
    Continuous_Futures/    <- futures
    World_Indices/         <- 31 global indices
    _catalog.parquet       <- master catalog of all symbols + metadata

Usage:
    python norgate_download_all.py                    # Download everything
    python norgate_download_all.py --databases "US Equities"  # Specific database
    python norgate_download_all.py --resume            # Skip already downloaded
    python norgate_download_all.py --update            # Re-download existing (refresh)
    python norgate_download_all.py --verify            # Verify integrity of saved files

Loading data later:
    import pandas as pd
    df = pd.read_parquet(r"D:\\Experiments\\norgate_data\\US_Equities\\AAPL.parquet")
    catalog = pd.read_parquet(r"D:\\Experiments\\norgate_data\\_catalog.parquet")
"""

import os
import sys
import time
import signal
import argparse
from datetime import datetime
from pathlib import Path

try:
    import norgatedata
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Missing: {e}")
    print("pip install norgatedata pandas numpy pyarrow")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(r"D:\Experiments\norgate_data")
CATALOG_FILE = BASE_DIR / "_catalog.parquet"
LOG_FILE = BASE_DIR / "_download_log.txt"

MAX_RETRIES = 3
RETRY_BACKOFF = 0.5      # seconds, doubles each attempt
MAX_CONSECUTIVE_ERRORS = 10  # abort database if this many symbols fail in a row

ALL_DATABASES = [
    "US Equities",
    "US Equities Delisted",
    "US Indices",
    "Economic",
    "Forex Spot",
    "Cash Commodities",
    "Continuous Futures",
    "World Indices",
]

DB_FOLDER_MAP = {
    "US Equities": "US_Equities",
    "US Equities Delisted": "US_Equities_Delisted",
    "US Indices": "US_Indices",
    "Economic": "Economic",
    "Forex Spot": "Forex_Spot",
    "Cash Commodities": "Cash_Commodities",
    "Continuous Futures": "Continuous_Futures",
    "World Indices": "World_Indices",
}


# ============================================================================
# Helpers
# ============================================================================
def format_size(nbytes):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def format_time(seconds):
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"


def sanitize_filename(symbol):
    """Make symbol safe for filesystem."""
    return (symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))


def log(msg, also_print=True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    if also_print:
        print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============================================================================
# Progress tracker
# ============================================================================
class ProgressTracker:
    """Tracks overall download progress across all databases."""

    def __init__(self):
        self.total_symbols = 0
        self.completed_symbols = 0
        self.downloaded = 0
        self.skipped = 0
        self.errors = 0
        self.empty = 0
        self.total_bytes = 0
        self.total_bars = 0
        self.start_time = time.time()
        self.db_name = ""
        self.db_total = 0
        self.db_done = 0

    def set_total(self, total):
        self.total_symbols = total
        self.start_time = time.time()  # reset timer when we know the total

    def start_database(self, db_name, db_count):
        self.db_name = db_name
        self.db_total = db_count
        self.db_done = 0

    def update(self, status, file_size=0, bars=0):
        self.completed_symbols += 1
        self.db_done += 1
        self.total_bars += bars
        if status == "downloaded":
            self.downloaded += 1
            self.total_bytes += file_size
        elif status == "skipped":
            self.skipped += 1
        elif status == "empty":
            self.empty += 1
        else:
            self.errors += 1
        self._print()

    def _print(self):
        elapsed = time.time() - self.start_time
        rate = self.completed_symbols / max(elapsed, 0.01)
        remaining = max(0, self.total_symbols - self.completed_symbols)
        eta = remaining / max(rate, 0.01)

        overall_pct = self.completed_symbols / max(self.total_symbols, 1) * 100
        bar_width = 25
        filled = int(bar_width * self.completed_symbols / max(self.total_symbols, 1))
        bar = "=" * filled + ">" * (1 if filled < bar_width else 0) + "." * max(0, bar_width - filled - 1)
        db_pct = self.db_done / max(self.db_total, 1) * 100

        line = (
            f"\r  [{bar}] {overall_pct:5.1f}%  "
            f"OVERALL: {self.completed_symbols}/{self.total_symbols}  "
            f"{self.db_name}: {self.db_done}/{self.db_total} ({db_pct:.0f}%)  "
            f"| dl:{self.downloaded} skip:{self.skipped} err:{self.errors}  "
            f"| {format_size(self.total_bytes)}  "
            f"| ETA: {format_time(eta)}  "
        )
        print(line, end="", flush=True)

    def final_summary(self):
        elapsed = time.time() - self.start_time
        print()
        print()
        log(f"{'='*65}")
        log(f"  DOWNLOAD COMPLETE")
        log(f"{'='*65}")
        log(f"  Symbols processed: {self.completed_symbols:,}")
        log(f"  Downloaded:        {self.downloaded:,}")
        log(f"  Cached (skipped):  {self.skipped:,}")
        log(f"  Empty (no data):   {self.empty:,}")
        log(f"  Errors:            {self.errors:,}")
        log(f"  Total bars:        {self.total_bars:,}")
        log(f"  Disk written:      {format_size(self.total_bytes)}")
        log(f"  Time elapsed:      {format_time(elapsed)}")
        if self.downloaded > 0:
            log(f"  Avg speed:         {self.downloaded / max(elapsed, 0.01):.1f} symbols/sec")
        log(f"{'='*65}")


progress = ProgressTracker()

# Graceful shutdown flag
_shutdown_requested = False


def _handle_sigint(signum, frame):
    """Handle Ctrl+C — finish current symbol, save catalog, then exit."""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n\n  Force quit. Catalog NOT saved.")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n  >>> Ctrl+C received. Finishing current symbol, then saving catalog...")
    print("  >>> Press Ctrl+C again to force quit.\n")


# ============================================================================
# Norgate fetch + atomic Parquet save
# ============================================================================
def fetch_symbol_data(symbol, database):
    """Fetch all available data for a symbol. Returns (df, metadata)."""
    metadata = {
        "symbol": symbol, "database": database,
        "name": "", "exchange": "", "currency": "",
        "first_date": None, "last_date": None,
        "bars": 0, "status": "ok", "error": "",
    }

    # Metadata (non-critical — skip date lookups since we get them from the DataFrame)
    try: metadata["name"] = norgatedata.security_name(symbol)
    except Exception: pass
    try: metadata["exchange"] = norgatedata.exchange_name(symbol)
    except Exception: pass
    try: metadata["currency"] = norgatedata.currency(symbol)
    except Exception: pass

    # Adjustment type
    if database in ("Economic", "Forex Spot", "Cash Commodities", "US Indices", "World Indices"):
        adj = norgatedata.StockPriceAdjustmentType.NONE
    else:
        adj = norgatedata.StockPriceAdjustmentType.TOTALRETURN

    # Fetch price data
    try:
        df = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=adj,
            padding_setting=norgatedata.PaddingType.NONE,
            timeseriesformat="pandas-dataframe",
            interval="D",
        )
        if df is None or len(df) == 0:
            metadata["status"] = "empty"
            return None, metadata

        metadata["bars"] = len(df)
        metadata["first_date"] = pd.Timestamp(df.index[0])
        metadata["last_date"] = pd.Timestamp(df.index[-1])
        return df, metadata

    except Exception as e:
        metadata["status"] = "error"
        metadata["error"] = str(e)[:200]
        return None, metadata


def save_parquet_atomic(df, filepath):
    """
    Save DataFrame to Parquet ATOMICALLY to prevent corruption.

    1. Write to .tmp file
    2. Verify .tmp is readable with correct row count
    3. Rename .tmp -> final (atomic on same filesystem)

    If crash happens during write, only .tmp is corrupt — final file is safe.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    expected_rows = len(df)
    # .with_suffix(".parquet.tmp") crashes on Python < 3.12 (two dots in suffix)
    tmp_path = filepath.parent / (filepath.name + ".tmp")

    try:
        # Write to temp file
        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(
            table, tmp_path,
            compression="zstd",
            compression_level=3,
            write_statistics=True,
            data_page_size=1024 * 1024,
            write_page_index=True,
        )

        # Verify: metadata-only check (instant, no data read)
        # ZSTD has built-in checksums — if pyarrow can open the file, data is intact
        pf = pq.ParquetFile(tmp_path)
        actual_rows = pf.metadata.num_rows
        if actual_rows != expected_rows:
            raise IOError(f"Wrote {expected_rows} rows, metadata says {actual_rows}")
        del pf

        # Atomic replace: os.replace() is atomic on same volume (Windows + POSIX)
        # Unlike unlink+rename, no window where both files are missing
        os.replace(str(tmp_path), str(filepath))

        return filepath.stat().st_size

    except Exception:
        # Clean up tmp on ANY failure (disk full, verification fail, etc.)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def verify_parquet(filepath):
    """Verify a parquet file is readable and not corrupted."""
    try:
        df = pd.read_parquet(filepath)
        if len(df) == 0:
            return False, "empty"
        if df.isnull().all().any():
            return True, f"warning: all-NaN columns: {list(df.columns[df.isnull().all()])}"
        return True, f"ok ({len(df)} rows, {len(df.columns)} cols)"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Database download (sequential with retry)
# ============================================================================
def download_database(database, resume=False):
    """Download all symbols in a database to parquet files."""
    folder_name = DB_FOLDER_MAP.get(database, database.replace(" ", "_"))
    db_dir = BASE_DIR / folder_name
    db_dir.mkdir(parents=True, exist_ok=True)

    try:
        symbols = norgatedata.database_symbols(database)
    except Exception as e:
        log(f"  ERROR: Cannot get symbols for '{database}': {e}")
        return []

    if not symbols:
        log(f"  No symbols found in '{database}'")
        return []

    log(f"  {database}: {len(symbols)} symbols -> {db_dir}")

    catalog_entries = []
    t0 = time.time()
    db_downloaded = 0
    db_skipped = 0
    db_errors = 0
    db_empty = 0
    consecutive_errors = 0

    progress.start_database(DB_FOLDER_MAP.get(database, database), len(symbols))

    for symbol in symbols:
        if _shutdown_requested:
            log(f"  Stopped early (Ctrl+C) after {progress.db_done}/{len(symbols)} symbols")
            break

        # Abort if Norgate service appears to be down
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            log(f"\n  ABORTING: {consecutive_errors} consecutive errors — "
                f"Norgate service may be down. Run with --resume to retry later.")
            break

        safe_name = sanitize_filename(symbol)
        filepath = db_dir / f"{safe_name}.parquet"

        # Resume: skip if file exists and is readable
        if resume and filepath.exists() and filepath.stat().st_size > 0:
            try:
                # Use pyarrow metadata — instant, reads zero data rows
                pf = pq.ParquetFile(filepath)
                n_rows = pf.metadata.num_rows

                # Read just first and last row for date range
                first_date = None
                last_date = None
                if n_rows > 0:
                    first_row = pf.read_row_group(0, columns=["Date"]).to_pandas()
                    if len(first_row) > 0:
                        first_date = pd.Timestamp(first_row.iloc[0, 0])
                    last_group = pf.metadata.num_row_groups - 1
                    last_row = pf.read_row_group(last_group, columns=["Date"]).to_pandas()
                    if len(last_row) > 0:
                        last_date = pd.Timestamp(last_row.iloc[-1, 0])

                meta = {
                    "symbol": symbol, "database": database, "name": "", "exchange": "",
                    "currency": "",
                    "first_date": first_date, "last_date": last_date,
                    "bars": n_rows, "status": "cached", "error": "",
                }
                catalog_entries.append(meta)
                db_skipped += 1
                consecutive_errors = 0  # file is valid, service was fine when it was saved
                progress.update("skipped", bars=n_rows)
                continue
            except Exception:
                pass  # Re-download if can't read

        # Download with retry
        success = False
        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                df, meta = fetch_symbol_data(symbol, database)

                if df is not None and len(df) > 0:
                    file_size = save_parquet_atomic(df, filepath)
                    db_downloaded += 1
                    consecutive_errors = 0  # success resets counter
                    progress.update("downloaded", file_size=file_size, bars=len(df))
                    catalog_entries.append(meta)
                    success = True
                    break
                elif meta["status"] == "empty":
                    db_empty += 1
                    consecutive_errors = 0  # empty is a valid response
                    progress.update("empty")
                    catalog_entries.append(meta)
                    success = True
                    break
                else:
                    last_error = meta.get("error", "unknown")

            except Exception as e:
                last_error = str(e)[:200]

            # Backoff before retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

        if not success:
            meta = {
                "symbol": symbol, "database": database, "name": "", "exchange": "",
                "currency": "", "first_date": None, "last_date": None,
                "bars": 0, "status": "error",
                "error": f"Failed after {MAX_RETRIES} attempts: {last_error}",
            }
            catalog_entries.append(meta)
            db_errors += 1
            consecutive_errors += 1
            progress.update("error")

    # Clean up any leftover .tmp files (both naming patterns)
    for tmp_file in list(db_dir.glob("*.parquet.tmp")) + list(db_dir.glob("*.tmp")):
        try:
            tmp_file.unlink()
        except Exception:
            pass

    elapsed = time.time() - t0
    print()  # newline after progress bar
    log(f"  Done: {db_downloaded} downloaded, {db_skipped} cached, {db_empty} empty, "
        f"{db_errors} errors in {format_time(elapsed)}")

    return catalog_entries


# ============================================================================
# Catalog + reporting
# ============================================================================
def build_catalog(all_entries):
    """Build and save the master catalog parquet."""
    if not all_entries:
        log("No entries for catalog")
        return

    catalog = pd.DataFrame(all_entries)
    catalog = catalog.sort_values(["database", "symbol"]).reset_index(drop=True)

    # Atomic write: tmp -> verify -> replace (same pattern as symbol files)
    tmp_catalog = CATALOG_FILE.parent / (CATALOG_FILE.name + ".tmp")
    try:
        catalog.to_parquet(tmp_catalog, compression="zstd", index=False)
        # Verify (metadata-only — instant)
        pf = pq.ParquetFile(tmp_catalog)
        if pf.metadata.num_rows != len(catalog):
            raise IOError(f"Catalog verify failed: {len(catalog)} vs {pf.metadata.num_rows}")
        del pf
        os.replace(str(tmp_catalog), str(CATALOG_FILE))
    except Exception as e:
        log(f"  WARNING: Catalog save failed: {e}")
        try:
            tmp_catalog.unlink(missing_ok=True)
        except Exception:
            pass
        # Fallback: try direct write (better than no catalog)
        try:
            catalog.to_parquet(CATALOG_FILE, compression="zstd", index=False)
        except Exception:
            log(f"  ERROR: Could not save catalog at all")

    log(f"\nCatalog saved: {CATALOG_FILE}")
    log(f"  Total symbols: {len(catalog)}")

    for db in catalog["database"].unique():
        subset = catalog[catalog["database"] == db]
        ok = (subset["status"].isin(["ok", "cached"])).sum()
        err = (subset["status"] == "error").sum()
        empty_count = (subset["status"] == "empty").sum()
        total_bars = subset["bars"].sum()
        log(f"  {db:<30} {ok:>6} ok  {empty_count:>4} empty  {err:>4} err  "
            f"{total_bars:>10,} total bars")

    # Per-symbol detail report
    downloaded = catalog[catalog["status"].isin(["ok", "cached"])].copy()
    if len(downloaded) > 0:
        report_file = BASE_DIR / "_symbol_report.txt"
        lines = []
        header = f"{'Symbol':<25} {'Database':<25} {'First Date':<12} {'Last Date':<12} {'Bars':>7}"
        sep = "-" * len(header)

        log(f"\n{sep}")
        log(f"  DOWNLOADED SYMBOLS ({len(downloaded):,})")
        log(f"{sep}")
        log(header)
        log(sep)
        lines.extend([sep, f"  DOWNLOADED SYMBOLS ({len(downloaded):,})", sep, header, sep])

        for db in sorted(downloaded["database"].unique()):
            db_syms = downloaded[downloaded["database"] == db].sort_values("symbol")
            for _, row in db_syms.iterrows():
                first = str(row["first_date"].date()) if pd.notna(row["first_date"]) else "N/A"
                last = str(row["last_date"].date()) if pd.notna(row["last_date"]) else "N/A"
                line = f"{row['symbol']:<25} {row['database']:<25} {first:<12} {last:<12} {row['bars']:>7,}"
                lines.append(line)
                if len(downloaded) <= 200:
                    log(line)

            if len(downloaded) > 200:
                log(f"  ... {db}: {len(db_syms)} symbols (see {report_file.name} for full list)")

        lines.append(sep)
        log(sep)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Norgate Download Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for rline in lines:
                f.write(rline + "\n")
        log(f"\n  Full symbol report saved: {report_file}")


def verify_all():
    """Verify integrity of all saved parquet files."""
    log("Verifying all parquet files...")
    total = 0
    ok = 0
    bad = 0

    for db_folder in sorted(BASE_DIR.iterdir()):
        if not db_folder.is_dir() or db_folder.name.startswith("_"):
            continue
        files = list(db_folder.glob("*.parquet"))
        for f in files:
            total += 1
            valid, msg = verify_parquet(f)
            if valid:
                ok += 1
            else:
                bad += 1
                log(f"  CORRUPT: {f.relative_to(BASE_DIR)} - {msg}")

    log(f"\nVerification complete: {ok}/{total} OK, {bad} corrupt")
    if bad == 0:
        log("All files intact.")


def print_summary():
    """Print summary of what's stored on disk."""
    if not BASE_DIR.exists():
        print("No data directory found.")
        return

    print(f"\n{'='*70}")
    print(f"  NORGATE DATA ARCHIVE - {BASE_DIR}")
    print(f"{'='*70}")

    total_files = 0
    total_size = 0

    for db_folder in sorted(BASE_DIR.iterdir()):
        if not db_folder.is_dir() or db_folder.name.startswith("_"):
            continue
        files = list(db_folder.glob("*.parquet"))
        size = sum(f.stat().st_size for f in files)
        total_files += len(files)
        total_size += size
        print(f"  {db_folder.name:<30} {len(files):>6} files  {size/1e6:>8.1f} MB")

    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<30} {total_files:>6} files  {total_size/1e6:>8.1f} MB")

    if CATALOG_FILE.exists():
        cat = pd.read_parquet(CATALOG_FILE)
        print(f"\n  Catalog: {len(cat)} entries, {cat['bars'].sum():,} total bars")
        print(f"  Date range: {cat['first_date'].min()} to {cat['last_date'].max()}")

    print(f"{'='*70}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download all Norgate data to Parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python norgate_download_all.py                          # Download everything
  python norgate_download_all.py --databases "US Equities"  # Just stocks
  python norgate_download_all.py --databases "US Equities,US Indices"
  python norgate_download_all.py --resume                 # Skip existing files
  python norgate_download_all.py --verify                 # Check file integrity
  python norgate_download_all.py --summary                # Show what's on disk

Loading data:
  import pandas as pd
  df = pd.read_parquet(r"D:\\Experiments\\norgate_data\\US_Equities\\AAPL.parquet")
  catalog = pd.read_parquet(r"D:\\Experiments\\norgate_data\\_catalog.parquet")
        """,
    )
    parser.add_argument("--databases", type=str, default="",
                        help="Comma-separated database names (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip symbols already downloaded")
    parser.add_argument("--update", action="store_true",
                        help="Re-download all (refresh data)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify integrity of all saved files")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary of stored data")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify_all()
        return

    # Check Norgate connection
    log("Norgate Data - Full Download to Parquet")
    log(f"Output directory: {BASE_DIR}")
    try:
        norgatedata.status()
        log("Norgate Data Updater: Connected")
    except Exception:
        try:
            norgatedata.security_name("AAPL")
            log("Norgate Data: Connected (security lookup OK)")
        except Exception as e:
            log(f"ERROR: Cannot connect to Norgate Data: {e}")
            log("Make sure Norgate Data Updater is running.")
            sys.exit(1)

    # Select databases
    if args.databases:
        databases = [d.strip() for d in args.databases.split(",") if d.strip()]
    else:
        databases = ALL_DATABASES

    resume = args.resume and not args.update
    if resume:
        log("Resume mode: skipping already downloaded symbols")

    # Ctrl+C handler
    signal.signal(signal.SIGINT, _handle_sigint)
    log("Press Ctrl+C to stop gracefully (saves progress). Ctrl+C twice to force quit.")

    # Count total symbols for progress
    valid_databases = [db for db in databases if db in DB_FOLDER_MAP]
    total_symbols = 0
    for db in valid_databases:
        try:
            syms = norgatedata.database_symbols(db)
            total_symbols += len(syms) if syms else 0
        except Exception:
            pass
    progress.set_total(total_symbols)
    log(f"Total symbols to process: {total_symbols:,} across {len(valid_databases)} databases")
    log(f"Retries per symbol: {MAX_RETRIES} | Atomic writes: ON")

    # Download each database
    all_catalog = []

    for db in valid_databases:
        if _shutdown_requested:
            break

        log(f"\n{'-'*60}")
        log(f"Database: {db}")
        log(f"{'-'*60}")

        entries = download_database(db, resume=resume)
        all_catalog.extend(entries)

    # Build catalog (always, even on Ctrl+C)
    log(f"\n{'-'*60}")
    log("Building master catalog...")
    build_catalog(all_catalog)

    progress.final_summary()
    print_summary()

    if _shutdown_requested:
        log("Download was interrupted. Run with --resume to continue where you left off.")


if __name__ == "__main__":
    main()
