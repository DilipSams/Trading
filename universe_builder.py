"""
Point-in-time dynamic stock universe builder.

Eliminates survivorship bias by ranking ALL stocks (active + delisted)
by trailing dollar volume at each monthly rebalance date, using only
information available on that date.

Cache-based: first run takes ~60-90s to process 34k files, subsequent
runs load from parquet cache in <1s.

Parallelism: ThreadPoolExecutor saturates all I/O cores for file loading.
Falls back to directory scan when _catalog.parquet is absent.

Performance: rolling mean is computed per-stock during load (parallel),
producing a precomputed monthly matrix. Step 3 rankings then become a
vectorized matrix slice + nlargest — no inner Python loop over stocks.
"""

import os
import re
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional

# Default paths
NORGATE_ROOT = os.environ.get("NORGATE_ROOT", r"C:\ProgramData\NorgateData")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

# Saturate I/O workers, capped at 16
_IO_WORKERS = max(1, min(16, os.cpu_count() or 4))

# ETFs / indices to exclude from the stock universe (they'd dominate by turnover)
UNIVERSE_EXCLUDE = {
    "SPY", "QQQ", "IVV", "VOO", "DIA", "IWM", "VTI", "VTV", "VUG", "VIG",
    "VXUS", "BND", "AGG", "TLT", "GLD", "SLV", "USO", "UNG", "HYG", "LQD",
    "EEM", "EFA", "VWO", "VEA", "IEMG", "IEFA", "SHY", "IEF", "TIP",
    # Sector ETFs
    "XLK", "XLF", "XLV", "XLI", "XLE", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC",
    # Leveraged ETFs
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SDS", "QLD", "SSO",
    "TECL", "ERX", "FAS", "FAZ", "LABU", "LABD", "SOXL", "SOXS",
    # Commodity / thematic ETFs that slipped through
    "GDX", "GDXJ", "SIL", "SILJ",                      # gold/silver miners
    "EWY", "EWZ", "EWJ", "EWC", "EWT", "EWA", "EWG",   # country ETFs
    "FXI", "KWEB", "CQQQ", "ASHR",                      # China ETFs
    "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ",             # ARK ETFs
    "XOP", "OIH", "KRE", "KBE", "SMH", "IBB",           # sector ETFs (missed)
}

# Compiled exclude pattern for non-equity instruments
_EXCLUDE_RE = re.compile(
    r"\.WS[AB]?$"   # warrants
    r"|\.U[N]?$"    # units
    r"|\.R[T]?$"    # rights
    r"|\.P[AB]?$"   # preferred (some)
)


def _build_catalog_from_dirs(norgate_root: str) -> pd.DataFrame:
    """
    Fallback: scan US_Equities / US_Equities_Delisted directories and
    build a catalog DataFrame by reading parquet file metadata.

    Called when _catalog.parquet is missing (fresh Norgate install).
    Uses ThreadPoolExecutor to parallelize the I/O-bound scan.
    """
    eq_dir = os.path.join(norgate_root, "US_Equities")
    de_dir = os.path.join(norgate_root, "US_Equities_Delisted")

    tasks: List[Tuple[str, str]] = []
    for d, db in [(eq_dir, "US Equities"), (de_dir, "US Equities Delisted")]:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.endswith(".parquet"):
                    tasks.append((os.path.join(d, fname), db))

    if not tasks:
        raise FileNotFoundError(
            f"No parquet files found in {eq_dir!r} or {de_dir!r}. "
            "Check the NORGATE_ROOT environment variable."
        )

    print(f"    Scanning {len(tasks)} parquet files ({_IO_WORKERS} workers)...", end="\r")
    t0 = time.time()

    def _probe(args: Tuple[str, str]) -> Optional[dict]:
        fpath, database = args
        sym = os.path.basename(fpath)[:-8]  # strip .parquet
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(fpath)
            n_rows = pf.metadata.num_rows
            if n_rows < 126:
                return None
            # Read just the index to get first/last date
            idx = pd.to_datetime(pf.read(columns=[]).to_pandas().index)
            if len(idx) == 0:
                return None
            return {
                "symbol": sym,
                "database": database,
                "first_date": idx.min(),
                "last_date": idx.max(),
                "bars": n_rows,
                "filepath": fpath,
            }
        except Exception:
            return None

    records: List[dict] = []
    n_tasks = len(tasks)
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as ex:
        for i, rec in enumerate(ex.map(_probe, tasks), 1):
            if i % 2000 == 0:
                print(f"    Scanned {i}/{n_tasks} files...", end="\r")
            if rec is not None:
                records.append(rec)

    elapsed = time.time() - t0
    print(f"    Scanned {n_tasks} files in {elapsed:.1f}s, "
          f"{len(records)} valid stocks               ")

    if not records:
        return pd.DataFrame()

    cat = pd.DataFrame(records)

    # Derive base_symbol and uid
    base_symbols: List[str] = []
    uids: List[str] = []
    for _, row in cat.iterrows():
        sym, db = row["symbol"], row["database"]
        if db == "US Equities":
            base_symbols.append(sym)
            uids.append(sym)
        else:
            m = re.match(r"^(.+)-(\d{6})$", sym)
            base_symbols.append(m.group(1) if m else sym)
            uids.append(sym)

    cat["base_symbol"] = base_symbols
    cat["uid"] = uids
    return cat


def _parse_catalog(norgate_root: str) -> pd.DataFrame:
    """
    Load Norgate catalog and return filtered DataFrame with columns:
      symbol, database, first_date, last_date, bars, filepath, base_symbol, uid

    Falls back to _build_catalog_from_dirs() when _catalog.parquet is absent.
    """
    cat_path = os.path.join(norgate_root, "_catalog.parquet")
    eq_dir = os.path.join(norgate_root, "US_Equities")
    de_dir = os.path.join(norgate_root, "US_Equities_Delisted")

    if os.path.exists(cat_path):
        cat = pd.read_parquet(cat_path)

        # Filter to equities only
        eq_mask = cat["database"].isin(["US Equities", "US Equities Delisted"])
        cat = cat[eq_mask].copy()

        # Need minimum history for 63-day rolling average
        cat = cat[cat["bars"] >= 126].copy()

        # Build filepath and extract base_symbol / uid
        filepaths: List[str] = []
        base_symbols: List[str] = []
        uids: List[str] = []

        for _, row in cat.iterrows():
            sym = row["symbol"]
            db = row["database"]
            if db == "US Equities":
                filepaths.append(os.path.join(eq_dir, f"{sym}.parquet"))
                base_symbols.append(sym)
                uids.append(sym)
            else:
                filepaths.append(os.path.join(de_dir, f"{sym}.parquet"))
                m = re.match(r"^(.+)-(\d{6})$", sym)
                base_symbols.append(m.group(1) if m else sym)
                uids.append(sym)

        cat["filepath"] = filepaths
        cat["base_symbol"] = base_symbols
        cat["uid"] = uids
    else:
        print("    _catalog.parquet not found — scanning directories (one-time)...")
        cat = _build_catalog_from_dirs(norgate_root)
        if cat.empty:
            return cat

    # Filter out warrants, units, rights, preferred
    keep = cat["base_symbol"].apply(lambda s: not _EXCLUDE_RE.search(s))
    cat = cat[keep].copy()

    # Exclude ETFs/indices by base_symbol
    cat = cat[~cat["base_symbol"].isin(UNIVERSE_EXCLUDE)].copy()

    # Exclude single-character symbols (often index artifacts)
    cat = cat[cat["base_symbol"].str.len() >= 2].copy()

    cat = cat.reset_index(drop=True)
    return cat


def _preload_and_resample_turnover(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Load all Turnover series in parallel, compute trailing 63-day rolling
    mean per stock, and resample to month-ends.

    Returns a wide DataFrame: index=month-end dates, columns=uid,
    values=trailing 63-day average turnover as of each month-end.

    This replaces the old _preload_turnover_parallel() and eliminates
    the per-stock inner loop in _compute_monthly_rankings() entirely.
    """
    uids = catalog["uid"].tolist()
    fpaths = catalog["filepath"].tolist()
    n = len(uids)

    print(f"    Loading & rolling {n} Turnover series ({_IO_WORKERS} workers)...", end="\r")
    t0 = time.time()

    def _load_and_resample(args: Tuple[str, str]) -> Tuple[str, Optional[pd.Series]]:
        uid, fpath = args
        if not os.path.exists(fpath):
            return uid, None
        try:
            df = pd.read_parquet(fpath, columns=["Turnover"])
            df.index = pd.to_datetime(df.index)
            ts = df["Turnover"].sort_index()
            if len(ts) < 21:
                return uid, None
            # 63-day trailing mean (min 21 days = 1 month of data)
            # then take the last value of each calendar month
            ts_roll = ts.rolling(63, min_periods=21).mean()
            ts_me = ts_roll.resample("ME").last()
            ts_me = ts_me[ts_me > 0].dropna()
            if ts_me.empty:
                return uid, None
            return uid, ts_me
        except Exception:
            return uid, None

    monthly_series: dict = {}
    n_loaded = 0
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as ex:
        for i, (uid, ts_me) in enumerate(ex.map(_load_and_resample, zip(uids, fpaths)), 1):
            if i % 2000 == 0:
                print(f"    Loaded {i}/{n} Turnover series...", end="\r")
            if ts_me is not None:
                monthly_series[uid] = ts_me
                n_loaded += 1

    elapsed = time.time() - t0
    print(f"    Loaded {n_loaded}/{n} Turnover series in {elapsed:.1f}s       ")

    # Build the monthly wide matrix: index=ME dates, columns=uid
    # pd.DataFrame aligns series on their indices automatically (NaN where absent)
    print(f"    Building monthly turnover matrix ({n_loaded} stocks)...", end="\r")
    t1 = time.time()
    monthly_matrix = pd.DataFrame(monthly_series).sort_index()
    print(f"    Matrix: {monthly_matrix.shape[0]} months × {monthly_matrix.shape[1]} stocks "
          f"in {time.time() - t1:.1f}s")

    return monthly_matrix


def _compute_monthly_rankings(
    catalog: pd.DataFrame,
    top_n: int = 150,
    start_date: str = "1990-01-01",
    monthly_matrix: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    For each month-end, rank all alive stocks by trailing 63-day average
    Turnover. Return top-N rankings.

    When monthly_matrix is provided (precomputed by _preload_and_resample_turnover),
    each month is a single vectorized matrix slice + nlargest — no inner loop.
    """
    today = pd.Timestamp.now().normalize()
    month_ends = pd.date_range(start_date, today, freq="ME")
    if len(month_ends) == 0:
        return pd.DataFrame()

    # Pre-index catalog arrays for fast numpy alive-stock lookup
    cat_first = catalog["first_date"].values.astype("datetime64[ns]")
    cat_last = catalog["last_date"].values.astype("datetime64[ns]")
    cat_uids = catalog["uid"].values
    cat_bases = catalog["base_symbol"].values
    uid_to_base = dict(zip(cat_uids, cat_bases))

    all_records: List[dict] = []
    total_months = len(month_ends)
    t0 = time.time()

    if monthly_matrix is not None:
        # ---------------------------------------------------------------
        # Fast path: vectorized per-month slice of precomputed matrix
        # ---------------------------------------------------------------
        # Reindex to our exact month-end grid (no forward-fill so we don't
        # carry stale values forward for delisted stocks)
        mat = monthly_matrix.reindex(month_ends)
        mat_uids = np.array(mat.columns)

        # Pre-build numpy arrays for matrix lookup speed
        mat_values = mat.values  # shape (n_months, n_stocks)

        # Build uid -> column index for fast alive-uid filtering
        uid_to_col = {uid: i for i, uid in enumerate(mat.columns)}

        for mi, me in enumerate(month_ends):
            if mi % 48 == 0:
                elapsed = time.time() - t0
                pct = mi / total_months * 100
                print(
                    f"    Building rankings: {mi}/{total_months} months ({pct:.0f}%), "
                    f"{elapsed:.1f}s elapsed",
                    end="\r",
                )

            me_np = np.datetime64(me)
            lookback_start = me_np - np.timedelta64(90, "D")

            # Alive stocks at this month-end
            alive_mask = (cat_first <= me_np) & (cat_last >= lookback_start)
            alive_uids = cat_uids[alive_mask]

            # Get column indices for alive stocks that exist in matrix
            alive_cols = [uid_to_col[u] for u in alive_uids if u in uid_to_col]
            if not alive_cols:
                continue

            # Slice the precomputed row for this month
            row_vals = mat_values[mi, alive_cols]
            alive_cols_arr = np.array(alive_cols)

            # Filter out NaN and zero
            valid = np.isfinite(row_vals) & (row_vals > 0)
            if not valid.any():
                continue

            row_vals = row_vals[valid]
            alive_cols_arr = alive_cols_arr[valid]

            # Rank: argsort descending, take top-N
            if len(row_vals) > top_n:
                top_idx = np.argpartition(row_vals, -top_n)[-top_n:]
                top_idx = top_idx[np.argsort(row_vals[top_idx])[::-1]]
            else:
                top_idx = np.argsort(row_vals)[::-1]

            for rank, ti in enumerate(top_idx, 1):
                uid = mat_uids[alive_cols_arr[ti]]
                all_records.append(
                    {
                        "date": me,
                        "rank": rank,
                        "uid": uid,
                        "base_symbol": uid_to_base.get(uid, uid),
                        "avg_turnover_63d": float(row_vals[ti]),
                    }
                )

    else:
        # ---------------------------------------------------------------
        # Slow fallback: lazy per-stock I/O inside the monthly loop
        # ---------------------------------------------------------------
        turnover_cache: dict = {}
        n_stocks = len(catalog)
        cat_paths = catalog["filepath"].values
        _n_errs = [0]

        for mi, me in enumerate(month_ends):
            if mi % 24 == 0:
                elapsed = time.time() - t0
                pct = mi / total_months * 100
                print(
                    f"    Building rankings: {mi}/{total_months} months ({pct:.0f}%), "
                    f"{elapsed:.0f}s elapsed, cache={len(turnover_cache)}",
                    end="\r",
                )

            me_np = np.datetime64(me)
            lookback_start = me_np - np.timedelta64(90, "D")
            alive_mask = (cat_first <= me_np) & (cat_last >= lookback_start)
            alive_idx = np.where(alive_mask)[0]

            turnover_vals: List[float] = []
            alive_uids_month: List[str] = []
            alive_bases_month: List[str] = []

            for idx in alive_idx:
                uid = cat_uids[idx]
                base = cat_bases[idx]
                if uid not in turnover_cache:
                    fpath = cat_paths[idx]
                    if os.path.exists(fpath):
                        try:
                            df = pd.read_parquet(fpath, columns=["Turnover"])
                            df.index = pd.to_datetime(df.index)
                            turnover_cache[uid] = df["Turnover"]
                        except Exception as e:
                            if _n_errs[0] < 5:
                                print(f"\n    WARNING: {uid}: {e}")
                            _n_errs[0] += 1
                            turnover_cache[uid] = pd.Series(dtype=float)
                    else:
                        turnover_cache[uid] = pd.Series(dtype=float)

                ts = turnover_cache.get(uid)
                if ts is None or len(ts) == 0:
                    continue
                mask = ts.index <= me
                recent = ts[mask].iloc[-63:] if mask.any() else pd.Series(dtype=float)
                if len(recent) >= 21:
                    avg_to = recent.mean()
                    if avg_to > 0 and not np.isnan(avg_to):
                        turnover_vals.append(avg_to)
                        alive_uids_month.append(uid)
                        alive_bases_month.append(base)

            if turnover_vals:
                arr = np.array(turnover_vals)
                top_idx = np.argsort(arr)[::-1][:top_n]
                for rank, ti in enumerate(top_idx, 1):
                    all_records.append(
                        {
                            "date": me,
                            "rank": rank,
                            "uid": alive_uids_month[ti],
                            "base_symbol": alive_bases_month[ti],
                            "avg_turnover_63d": turnover_vals[ti],
                        }
                    )

            if mi % 12 == 0 and mi > 0:
                prune_cutoff = me_np - np.timedelta64(180, "D")
                for ci in range(n_stocks):
                    uid = cat_uids[ci]
                    if uid in turnover_cache and cat_last[ci] < prune_cutoff:
                        del turnover_cache[uid]

    elapsed = time.time() - t0
    print(
        f"    Building rankings: {total_months}/{total_months} months (100%), "
        f"{elapsed:.1f}s total, {len(all_records)} rank records          "
    )
    return pd.DataFrame(all_records)


def _build_price_matrix(
    rankings: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load adjusted Close prices for all UIDs that ever appeared in rankings.
    Returns wide DataFrame: index=Date, columns=uid.

    Uses ThreadPoolExecutor for parallel I/O across all workers.
    """
    unique_uids = rankings["uid"].unique()
    n = len(unique_uids)
    print(f"    Loading prices for {n} unique symbols ({_IO_WORKERS} workers)...")

    uid_to_path = dict(zip(catalog["uid"], catalog["filepath"]))

    def _load_price(uid: str) -> Tuple[str, Optional[pd.Series]]:
        fpath = uid_to_path.get(uid)
        if not fpath or not os.path.exists(fpath):
            return uid, None
        try:
            df = pd.read_parquet(fpath, columns=["Close"])
            df.index = pd.to_datetime(df.index)
            return uid, df["Close"]
        except Exception:
            return uid, None

    price_dict: dict = {}
    n_errs = 0
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as ex:
        for uid, series in ex.map(_load_price, unique_uids):
            if series is not None:
                price_dict[uid] = series
            else:
                n_errs += 1

    if n_errs:
        print(f"    WARNING: {n_errs} price files failed to load")
    print(f"    Loaded {len(price_dict)} price series")

    prices = pd.DataFrame(price_dict).sort_index()
    return prices


def build_universe_cache(
    norgate_root: str = NORGATE_ROOT,
    cache_dir: str = CACHE_DIR,
    top_n: int = 250,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build or load the cached dynamic universe.

    Returns:
      rankings: DataFrame [date, rank, uid, base_symbol, avg_turnover_63d]
      prices:   DataFrame [Date index, uid columns] of adjusted Close prices
    """
    os.makedirs(cache_dir, exist_ok=True)

    rank_path = os.path.join(cache_dir, f"universe_rankings_top{top_n}.parquet")
    price_path = os.path.join(cache_dir, f"universe_prices_top{top_n}.parquet")

    # --- Cache freshness check ---
    if not force_rebuild and os.path.exists(rank_path) and os.path.exists(price_path):
        cat_path = os.path.join(norgate_root, "_catalog.parquet")
        cache_mtime = os.path.getmtime(rank_path)
        stale = False

        if os.path.exists(cat_path):
            stale = cache_mtime < os.path.getmtime(cat_path)
        else:
            eq_dir = os.path.join(norgate_root, "US_Equities")
            try:
                newest_data = max(
                    os.path.getmtime(os.path.join(eq_dir, f))
                    for f in os.listdir(eq_dir)
                    if f.endswith(".parquet")
                )
                stale = cache_mtime < newest_data
            except (FileNotFoundError, ValueError):
                stale = True

        if not stale:
            print("  Loading cached universe...")
            t0 = time.time()
            rankings = pd.read_parquet(rank_path)
            prices = pd.read_parquet(price_path)
            print(
                f"  Universe cache loaded in {time.time()-t0:.1f}s "
                f"({len(rankings)} rankings, {prices.shape[1]} symbols)"
            )
            return rankings, prices
        else:
            print("  Cache stale (data updated), rebuilding...")

    # --- Build from scratch ---
    print("  Building point-in-time universe...")
    t_start = time.time()

    print("  Step 1/4: Parsing catalog...")
    catalog = _parse_catalog(norgate_root)
    if catalog.empty:
        raise RuntimeError(
            "No valid stocks found — check NORGATE_ROOT path and data files."
        )
    print(f"    {len(catalog)} candidate stocks (active + delisted)")

    print(f"  Step 2/4: Loading & resampling Turnover series ({_IO_WORKERS} workers)...")
    monthly_matrix = _preload_and_resample_turnover(catalog)

    print("  Step 3/4: Computing monthly rankings (vectorized)...")
    rankings = _compute_monthly_rankings(
        catalog, top_n=top_n, monthly_matrix=monthly_matrix
    )
    if rankings.empty:
        raise RuntimeError("No rankings computed — check data range or catalog.")

    print(f"  Step 4/4: Building price matrix ({_IO_WORKERS} workers)...")
    prices = _build_price_matrix(rankings, catalog)

    # Save cache
    rankings.to_parquet(rank_path, index=False)
    prices.to_parquet(price_path)
    print(f"  Universe built in {time.time()-t_start:.0f}s, cached to {cache_dir}")

    return rankings, prices


def get_universe_at_date(
    rankings: pd.DataFrame,
    target_date: pd.Timestamp,
    top_n: int = 100,
) -> List[str]:
    """
    Return the top_n UIDs as of the most recent month-end <= target_date.
    """
    valid = rankings[rankings["date"] <= target_date]
    if valid.empty:
        return []
    latest_date = valid["date"].max()
    month_data = valid[(valid["date"] == latest_date) & (valid["rank"] <= top_n)]
    return list(month_data["uid"].values)


def uid_to_ticker(uid: str) -> str:
    """Convert uid back to a display-friendly ticker."""
    m = re.match(r"^(.+)-(\d{6})$", uid)
    if m:
        return m.group(1)
    return uid


def get_full_universe_history(
    rankings: pd.DataFrame,
    top_n: int = 150,
) -> dict:
    """
    Return dict mapping each rebalance date to its list of UIDs.

    Returns:
        dict[pd.Timestamp -> List[str]]
    """
    filtered = rankings[rankings["rank"] <= top_n]
    result = {}
    for dt in sorted(filtered["date"].unique()):
        mask = filtered["date"] == dt
        result[dt] = list(filtered.loc[mask, "uid"].values)
    return result


def get_universe_stats(rankings: pd.DataFrame, top_n: int = 150) -> dict:
    """
    Compute universe membership statistics.

    Returns:
        dict with keys: n_unique, n_entered_per_year, n_exited_per_year,
        turnover_rate, first_date, last_date, n_rebalances.
    """
    filtered = rankings[rankings["rank"] <= top_n]
    dates = sorted(filtered["date"].unique())
    if len(dates) < 2:
        return {}

    date_sets = {}
    for dt in dates:
        mask = filtered["date"] == dt
        date_sets[dt] = set(filtered.loc[mask, "uid"].values)

    all_uids: set = set()
    entries_per_year: dict = {}
    exits_per_year: dict = {}
    prev_set: set = set()
    for dt in dates:
        curr_set = date_sets[dt]
        entered = curr_set - prev_set
        exited = prev_set - curr_set
        yr = dt.year
        entries_per_year[yr] = entries_per_year.get(yr, 0) + len(entered)
        exits_per_year[yr] = exits_per_year.get(yr, 0) + len(exited)
        all_uids.update(curr_set)
        prev_set = curr_set

    years = sorted(set(entries_per_year.keys()) | set(exits_per_year.keys()))
    avg_entries = np.mean([entries_per_year.get(y, 0) for y in years]) if years else 0
    avg_exits = np.mean([exits_per_year.get(y, 0) for y in years]) if years else 0

    return {
        "n_unique": len(all_uids),
        "n_rebalances": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "avg_entries_per_year": avg_entries,
        "avg_exits_per_year": avg_exits,
        "turnover_rate": (avg_entries + avg_exits) / (2 * top_n),
        "entries_per_year": entries_per_year,
        "exits_per_year": exits_per_year,
    }


if __name__ == "__main__":
    # Standalone test: build cache and print summary
    rankings, prices = build_universe_cache(top_n=250, force_rebuild=True)

    print(f"\nRankings: {len(rankings)} records")
    print(f"Date range: {rankings['date'].min()} to {rankings['date'].max()}")
    print(f"Unique symbols: {rankings['uid'].nunique()}")
    print(f"Price matrix: {prices.shape}")

    for check_date in ["2000-01-31", "2008-01-31", "2020-01-31"]:
        dt = pd.Timestamp(check_date)
        top10 = get_universe_at_date(rankings, dt, top_n=10)
        tickers = [uid_to_ticker(u) for u in top10]
        print(f"\nTop 10 by dollar volume as of {check_date}:")
        print(f"  {', '.join(tickers)}")
