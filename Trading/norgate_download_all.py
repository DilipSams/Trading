"""
Norgate Data — Full Download to Parquet  (Parallel Edition)
============================================================
Downloads ALL available data from Norgate and saves as Parquet files.
Uses ProcessPoolExecutor for parallel downloads (default: 16 workers).

Parquet format:
  - Binary columnar with built-in checksums (no corruption)
  - Preserves exact dtypes (float64, datetime64, etc.)
  - Compressed (~3-5x smaller than CSV)
  - Loads instantly: pd.read_parquet("file.parquet")

Directory structure:
  D:\\Experiments\\norgate_data\\
    US_Equities/           <- active stocks
      AAPL.parquet
      MSFT.parquet
      ...
    US_Equities_Delisted/  <- delisted (survivorship-bias-free)
    US_Indices/            <- indices
    Economic/              <- macro indicators
    Forex_Spot/            <- currency pairs
    Cash_Commodities/      <- commodities
    Continuous_Futures/    <- futures
    World_Indices/         <- global indices
    _catalog.parquet       <- master catalog of all symbols + metadata

Usage:
    python norgate_download_all.py                           # Download everything (16 workers)
    python norgate_download_all.py --workers 8               # Custom worker count
    python norgate_download_all.py --databases "US Equities" # Specific database
    python norgate_download_all.py --resume                  # Skip already downloaded
    python norgate_download_all.py --update                  # Re-download existing (refresh)
    python norgate_download_all.py --verify                  # Verify integrity of saved files

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
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
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

DEFAULT_WORKERS = 16
MAX_RETRIES = 3
RETRY_BACKOFF = 0.5              # seconds, doubles each attempt
MAX_ERRORS_BEFORE_ABORT = 50     # abort database after this many errors without a success

# Databases that use no price adjustment (indices, economic, forex, commodities)
NO_ADJUST_DBS = frozenset({"Economic", "Forex Spot", "Cash Commodities",
                            "US Indices", "World Indices"})

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


_log_lock = threading.Lock()


def log(msg, also_print=True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    if also_print:
        print(line)
    with _log_lock:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ============================================================================
# Atomic Parquet save  (used by worker processes)
# ============================================================================
def save_parquet_atomic(df, filepath):
    """
    Save DataFrame to Parquet ATOMICALLY to prevent corruption.

    1. Write to .tmp file
    2. Verify .tmp is readable with correct row count
    3. Rename .tmp -> final (atomic on same filesystem)
    """
    filepath = Path(filepath)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    expected_rows = len(df)
    tmp_path = filepath.parent / (filepath.name + ".tmp")

    try:
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
        pf = pq.ParquetFile(tmp_path)
        actual_rows = pf.metadata.num_rows
        if actual_rows != expected_rows:
            raise IOError(f"Wrote {expected_rows} rows, metadata says {actual_rows}")
        del pf

        # Atomic replace
        os.replace(str(tmp_path), str(filepath))
        return filepath.stat().st_size

    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


# ============================================================================
# Worker function  (runs in subprocess — must be at module level for pickling)
# ============================================================================
def _download_one(args):
    """
    Download one symbol and save to parquet.  Runs in a worker process.

    Args: tuple of (symbol, database, db_dir_str, adj_str, max_retries, retry_backoff)
    Returns: dict with download result
    """
    symbol, database, db_dir_str, adj_str, max_retries, retry_backoff = args

    # Each subprocess imports norgatedata independently → own requests.Session
    import norgatedata as _ng

    result = {
        "symbol": symbol, "database": database,
        "name": "", "exchange": "", "currency": "",
        "first_date": None, "last_date": None,
        "bars": 0, "status": "error", "error": "",
        "file_size": 0,
    }

    db_dir = Path(db_dir_str)
    safe_name = sanitize_filename(symbol)
    filepath = db_dir / f"{safe_name}.parquet"

    # Determine adjustment type
    if adj_str == "NONE":
        adj = _ng.StockPriceAdjustmentType.NONE
    else:
        adj = _ng.StockPriceAdjustmentType.TOTALRETURN

    # Retry loop
    last_error = ""
    for attempt in range(max_retries):
        try:
            df = _ng.price_timeseries(
                symbol,
                stock_price_adjustment_setting=adj,
                padding_setting=_ng.PaddingType.NONE,
                timeseriesformat="pandas-dataframe",
                interval="D",
            )

            if df is None or len(df) == 0:
                result["status"] = "empty"
                return result

            # Save atomically
            file_size = save_parquet_atomic(df, filepath)

            result["status"] = "ok"
            result["bars"] = len(df)
            result["file_size"] = file_size
            result["first_date"] = str(df.index[0])
            result["last_date"] = str(df.index[-1])
            return result

        except Exception as e:
            last_error = str(e)[:200]
            if attempt < max_retries - 1:
                time.sleep(retry_backoff * (2 ** attempt))

    result["error"] = f"Failed after {max_retries} attempts: {last_error}"
    return result


# ============================================================================
# Progress tracker  (main process only, but thread-safe defensively)
# ============================================================================
class ProgressTracker:
    """Tracks overall download progress across all databases."""

    def __init__(self):
        self._lock = threading.Lock()
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
        self.start_time = time.time()

    def start_database(self, db_name, db_count):
        with self._lock:
            self.db_name = db_name
            self.db_total = db_count
            self.db_done = 0

    def update(self, status, file_size=0, bars=0):
        with self._lock:
            self.completed_symbols += 1
            self.db_done += 1
            self.total_bars += bars
            if status in ("ok", "downloaded"):
                self.downloaded += 1
                self.total_bytes += file_size
            elif status in ("skipped", "cached"):
                self.skipped += 1
            elif status == "empty":
                self.empty += 1
            else:
                self.errors += 1
            self._print_unlocked()

    def bulk_skip(self, count, bars):
        """Fast-path for resume: skip many symbols at once."""
        with self._lock:
            self.completed_symbols += count
            self.db_done += count
            self.skipped += count
            self.total_bars += bars
            self._print_unlocked()

    def _print_unlocked(self):
        elapsed = time.time() - self.start_time
        rate = self.completed_symbols / max(elapsed, 0.01)
        remaining = max(0, self.total_symbols - self.completed_symbols)
        eta = remaining / max(rate, 0.01)

        overall_pct = self.completed_symbols / max(self.total_symbols, 1) * 100
        bar_width = 30
        filled = int(bar_width * self.completed_symbols / max(self.total_symbols, 1))
        bar = "=" * filled + ">" * (1 if filled < bar_width else 0) + "." * max(0, bar_width - filled - 1)
        db_pct = self.db_done / max(self.db_total, 1) * 100

        line = (
            f"\r  [{bar}] {overall_pct:5.1f}%  "
            f"ALL: {self.completed_symbols}/{self.total_symbols}  "
            f"{self.db_name}: {self.db_done}/{self.db_total} ({db_pct:.0f}%)  "
            f"| dl:{self.downloaded} skip:{self.skipped} err:{self.errors}  "
            f"| {format_size(self.total_bytes)}  "
            f"| {rate:.1f}/s  "
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


# Graceful shutdown flag
_shutdown_requested = False


def _handle_sigint(signum, frame):
    """Handle Ctrl+C — finish in-flight workers, save catalog, then exit."""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n\n  Force quit. Catalog NOT saved.")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n  >>> Ctrl+C received. Finishing in-flight workers, then saving catalog...")
    print("  >>> Press Ctrl+C again to force quit.\n")


# ============================================================================
# Parquet verification
# ============================================================================
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
# Load existing catalog for fast resume
# ============================================================================
def load_old_catalog():
    """Load existing catalog into a lookup dict for fast resume."""
    if not CATALOG_FILE.exists():
        return {}
    try:
        cat = pd.read_parquet(CATALOG_FILE)
        lookup = {}
        for _, row in cat.iterrows():
            key = (row["symbol"], row["database"])
            lookup[key] = {
                "symbol": row["symbol"],
                "database": row["database"],
                "name": row.get("name", ""),
                "exchange": row.get("exchange", ""),
                "currency": row.get("currency", ""),
                "first_date": row.get("first_date", None),
                "last_date": row.get("last_date", None),
                "bars": row.get("bars", 0),
                "status": "cached",
                "error": "",
            }
        return lookup
    except Exception:
        return {}


# ============================================================================
# Parallel database download
# ============================================================================
def download_database(database, resume, workers, old_catalog, progress):
    """Download all symbols in a database using parallel workers."""
    global _shutdown_requested

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

    # Adjustment type string (picklable)
    adj_str = "NONE" if database in NO_ADJUST_DBS else "TOTALRETURN"

    # Start tracking this database BEFORE resume scan so progress bar
    # shows the correct db_name during bulk_skip
    progress.start_database(DB_FOLDER_MAP.get(database, database), len(symbols))

    # ----- Fast resume: pre-scan directory -----
    cached_entries = []
    to_download = symbols

    if resume:
        existing_files = {f.stem for f in db_dir.glob("*.parquet")}
        to_download = []
        cached_bars = 0

        for sym in symbols:
            safe = sanitize_filename(sym)
            if safe in existing_files:
                # Reuse catalog entry if available, otherwise minimal entry
                entry = old_catalog.get((sym, database))
                if entry is None:
                    # File exists but not in old catalog — read bar count from metadata
                    filepath = db_dir / f"{safe}.parquet"
                    try:
                        pf = pq.ParquetFile(filepath)
                        n_rows = pf.metadata.num_rows
                        del pf
                    except Exception:
                        # Can't read — re-download
                        to_download.append(sym)
                        continue
                    entry = {
                        "symbol": sym, "database": database,
                        "name": "", "exchange": "", "currency": "",
                        "first_date": None, "last_date": None,
                        "bars": n_rows, "status": "cached", "error": "",
                    }
                cached_entries.append(entry)
                bars_val = entry.get("bars", 0)
                cached_bars += int(bars_val) if bars_val == bars_val else 0  # guard NaN
            else:
                to_download.append(sym)

        # Bulk-update progress for all cached symbols at once
        if cached_entries:
            progress.bulk_skip(len(cached_entries), cached_bars)

    log(f"  {database}: {len(symbols)} total, {len(cached_entries)} cached, "
        f"{len(to_download)} to download -> {db_dir}")

    if not to_download:
        print()  # newline after progress
        log(f"  Done: all {len(cached_entries)} cached, 0 to download")
        return cached_entries

    # ----- Build task list -----
    tasks = [
        (sym, database, str(db_dir), adj_str, MAX_RETRIES, RETRY_BACKOFF)
        for sym in to_download
    ]

    # ----- Parallel download -----
    download_entries = []
    error_streak = 0
    db_downloaded = 0
    db_errors = 0
    db_empty = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_sym = {}
        for task in tasks:
            if _shutdown_requested:
                break
            f = executor.submit(_download_one, task)
            future_to_sym[f] = task[0]

        # Collect results as they complete
        try:
            for future in as_completed(future_to_sym):
                if _shutdown_requested:
                    for f in future_to_sym:
                        f.cancel()
                    break

                sym = future_to_sym[future]
                try:
                    result = future.result()
                except BrokenExecutor as e:
                    # Worker process crashed (segfault / OOM) — pool is dead
                    log(f"\n  FATAL: Worker pool crashed: {e}")
                    log(f"  Run with --resume to retry remaining symbols.")
                    for f in future_to_sym:
                        f.cancel()
                    result = {
                        "symbol": sym, "database": database,
                        "name": "", "exchange": "", "currency": "",
                        "first_date": None, "last_date": None,
                        "bars": 0, "status": "error",
                        "error": f"Pool crashed: {str(e)[:200]}",
                        "file_size": 0,
                    }
                    download_entries.append(result)
                    progress.update("error")
                    break
                except Exception as e:
                    result = {
                        "symbol": sym, "database": database,
                        "name": "", "exchange": "", "currency": "",
                        "first_date": None, "last_date": None,
                        "bars": 0, "status": "error",
                        "error": f"Worker exception: {str(e)[:200]}",
                        "file_size": 0,
                    }

                download_entries.append(result)

                # Update progress
                status = result["status"]
                progress.update(status,
                                file_size=result.get("file_size", 0),
                                bars=result.get("bars", 0))

                if status == "ok":
                    db_downloaded += 1
                    error_streak = 0
                elif status == "empty":
                    db_empty += 1
                    error_streak = 0
                else:
                    db_errors += 1
                    error_streak += 1

                # Service-down detection: too many errors without a success
                if error_streak >= MAX_ERRORS_BEFORE_ABORT:
                    log(f"\n  ABORTING {database}: {error_streak} errors without a success. "
                        f"Norgate service may be down. Run with --resume to retry later.")
                    for f in future_to_sym:
                        f.cancel()
                    break
        except KeyboardInterrupt:
            # Backup Ctrl+C catch if signal handler doesn't fire fast enough
            _shutdown_requested = True
            for f in future_to_sym:
                f.cancel()
            log(f"\n  Interrupted. Saving progress...")

    # Clean up leftover .tmp files
    for tmp_file in list(db_dir.glob("*.parquet.tmp")) + list(db_dir.glob("*.tmp")):
        try:
            tmp_file.unlink()
        except Exception:
            pass

    elapsed = time.time() - t0
    print()  # newline after progress bar
    log(f"  Done: {db_downloaded} downloaded, {len(cached_entries)} cached, "
        f"{db_empty} empty, {db_errors} errors in {format_time(elapsed)}")

    return cached_entries + download_entries


# ============================================================================
# Catalog + reporting
# ============================================================================
def build_catalog(all_entries):
    """Build and save the master catalog parquet."""
    if not all_entries:
        log("No entries for catalog")
        return

    catalog = pd.DataFrame(all_entries)

    # Normalize date columns
    for col in ("first_date", "last_date"):
        if col in catalog.columns:
            catalog[col] = pd.to_datetime(catalog[col], errors="coerce")

    # Drop file_size if present (internal tracking, not catalog data)
    catalog = catalog.drop(columns=["file_size"], errors="ignore")

    catalog = catalog.sort_values(["database", "symbol"]).reset_index(drop=True)

    # Atomic write
    tmp_catalog = CATALOG_FILE.parent / (CATALOG_FILE.name + ".tmp")
    try:
        catalog.to_parquet(tmp_catalog, compression="zstd", index=False)
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
        description="Download all Norgate data to Parquet files (parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python norgate_download_all.py                             # Download everything (16 workers)
  python norgate_download_all.py --workers 8                 # Custom worker count
  python norgate_download_all.py --databases "US Equities"   # Just stocks
  python norgate_download_all.py --databases "US Equities,US Indices"
  python norgate_download_all.py --resume                    # Skip existing files
  python norgate_download_all.py --verify                    # Check file integrity
  python norgate_download_all.py --summary                   # Show what's on disk

Loading data:
  import pandas as pd
  df = pd.read_parquet(r"D:\\Experiments\\norgate_data\\US_Equities\\AAPL.parquet")
  catalog = pd.read_parquet(r"D:\\Experiments\\norgate_data\\_catalog.parquet")
        """,
    )
    parser.add_argument("--databases", type=str, default="",
                        help="Comma-separated database names (default: all)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download workers (default: {DEFAULT_WORKERS})")
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

    # Clamp workers
    workers = max(1, min(args.workers, 64))

    # Check Norgate connection
    log("Norgate Data - Parallel Download to Parquet")
    log(f"Output directory: {BASE_DIR}")
    log(f"Workers: {workers}")
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

    progress = ProgressTracker()
    progress.set_total(total_symbols)
    log(f"Total symbols to process: {total_symbols:,} across {len(valid_databases)} databases")
    log(f"Workers: {workers} | Retries per symbol: {MAX_RETRIES} | Atomic writes: ON")

    # Load existing catalog for fast resume
    old_catalog = load_old_catalog() if resume else {}
    if old_catalog:
        log(f"Loaded existing catalog: {len(old_catalog)} entries for fast resume")

    # Download each database
    all_catalog = []

    try:
        for db in valid_databases:
            if _shutdown_requested:
                break

            log(f"\n{'-'*60}")
            log(f"Database: {db}")
            log(f"{'-'*60}")

            try:
                entries = download_database(db, resume=resume, workers=workers,
                                            old_catalog=old_catalog, progress=progress)
                all_catalog.extend(entries)
            except Exception as e:
                log(f"\n  ERROR processing {db}: {e}")
                log(f"  Continuing to next database. Run with --resume to retry.")
    finally:
        # Build catalog ALWAYS — even on crash, Ctrl+C, or unexpected errors
        log(f"\n{'-'*60}")
        log("Building master catalog...")
        build_catalog(all_catalog)

        progress.final_summary()
        print_summary()

        if _shutdown_requested:
            log("Download was interrupted. Run with --resume to continue where you left off.")


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows (spawn-based multiprocessing)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
