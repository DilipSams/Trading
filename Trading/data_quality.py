"""
data_quality.py -- Data Infrastructure: Quality Assessment

Provides dataset identity logging, coverage/missingness analysis,
sanity checks (OHLC integrity, stale data, extreme returns), and
a composite quality score (0-100) with PASS/WARN/FAIL verdict.

Usage:
    from data_quality import (
        DataLoadMeta, analyze_ohlcv, print_l0_summary,
        print_quality_table, hash_df,
    )

    # -- Single dataset --
    meta = DataLoadMeta(
        source="yfinance", symbols=["AAPL"], timeframe="1D",
        requested_start="2020-01-01", requested_end="2025-01-01",
        loaded_start=str(df["timestamp"].min()),
        loaded_end=str(df["timestamp"].max()),
        timezone="UTC", calendar="NYSE", adjusted=True,
        fields=["open","high","low","close","volume"],
        rows_total=len(df),
    )
    meta.dataset_hash = hash_df(df.set_index("timestamp"), cols=meta.fields)
    q = analyze_ohlcv(df, ts_col="timestamp", timeframe="1D")
    print_l0_summary(meta, q)

    # -- Multi-dataset (used by alphago_layering.py) --
    results = [(meta, q), ...]
    print_quality_table(results)

References:
    - Lopez de Prado (2018) on data integrity for backtesting
    - CRSP/Compustat point-in-time standards
"""

from __future__ import annotations
import sys
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import hashlib
import numpy as np
import pandas as pd


class MissingDataPolicy(str, Enum):
    """
    Configurable missing-data handling policy for DataIngestionGate.

    Spec requirement: "explicit and documented (forward-fill, interpolate,
    mask, or exclude). No silent NaN propagation."

    Each policy applies to non-close fields (close is always REJECT):
        FORWARD_FILL  - ffill then bfill (default for production)
        ROLLING_MEDIAN - fill with rolling 20-bar median (legacy default)
        INTERPOLATE   - linear interpolation within gaps
        MASK          - keep NaN but flag in metadata (downstream must handle)
        EXCLUDE       - drop entire bar if any field is missing
    """
    FORWARD_FILL = "forward_fill"
    ROLLING_MEDIAN = "rolling_median"
    INTERPOLATE = "interpolate"
    MASK = "mask"
    EXCLUDE = "exclude"


@dataclass
class DataLoadMeta:
    """Identity card for a loaded dataset -- printed at run start."""
    source: str
    symbols: List[str]
    timeframe: str
    requested_start: str
    requested_end: str
    loaded_start: str
    loaded_end: str
    timezone: str
    calendar: str
    adjusted: bool
    fields: List[str]
    rows_total: int
    notes: Optional[str] = None
    dataset_hash: Optional[str] = None


@dataclass
class DataQualityReport:
    """Quantitative quality assessment of OHLCV data."""
    expected_bars: Optional[int]
    received_bars: int
    missing_bar_ratio: Optional[float]
    duplicates: int
    out_of_order: int
    missing_by_col: Dict[str, float]
    ohlc_violations: int
    non_positive_prices: int
    stale_close_runs_max: int
    extreme_return_count: int
    score_0_100: float
    verdict: str  # "PASS" / "WARN" / "FAIL"


def hash_df(
    df: pd.DataFrame,
    cols: List[str],
    n: int = 5000,
) -> str:
    """Cheap reproducibility hash: first/last N rows of selected columns."""
    if df.empty:
        return "empty"
    sample = pd.concat([df.head(n), df.tail(n)], axis=0)
    avail_cols = [c for c in cols if c in sample.columns]
    if not avail_cols:
        return "no_matching_cols"
    payload = sample[avail_cols].to_csv(index=True).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:16]


def analyze_ohlcv(
    df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    symbol_col: Optional[str] = None,
    cols: Tuple[str, ...] = ("open", "high", "low", "close", "volume"),
    timeframe: str = "1D",
    expected_bars: Optional[int] = None,
    extreme_ret_threshold: float = 0.30,
    stale_run_threshold: int = 10,
) -> DataQualityReport:
    """
    Run all quality checks on an OHLCV DataFrame.

    Returns DataQualityReport with a 0-100 score and verdict.
    """
    if df is None or len(df) == 0:
        return DataQualityReport(
            expected_bars=expected_bars, received_bars=0,
            missing_bar_ratio=None, duplicates=0, out_of_order=0,
            missing_by_col={c: 1.0 for c in cols},
            ohlc_violations=0, non_positive_prices=0,
            stale_close_runs_max=0, extreme_return_count=0,
            score_0_100=0.0, verdict="FAIL",
        )

    d = df.copy()

    # -- Timestamp handling --
    dup, ooo = 0, 0
    if ts_col in d.columns:
        d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
        if symbol_col and symbol_col in d.columns:
            d = d.sort_values([symbol_col, ts_col])
            dup = int(d.duplicated([symbol_col, ts_col]).sum())
            ooo = int((d.groupby(symbol_col)[ts_col].diff().dt.total_seconds() < 0).sum())
        else:
            d = d.sort_values(ts_col)
            dup = int(d.duplicated([ts_col]).sum())
            ooo = int((d[ts_col].diff().dt.total_seconds() < 0).sum())

    # -- Missingness per column --
    missing_by_col = {}
    for c in cols:
        if c in d.columns:
            missing_by_col[c] = float(d[c].isna().mean())
        else:
            missing_by_col[c] = 1.0

    # -- Price sanity --
    non_positive_prices = 0
    for pc in ("close", "open"):
        if pc in d.columns:
            non_positive_prices += int((d[pc] <= 0).sum())

    # -- OHLC integrity --
    ohlc_violations = 0
    if all(c in d.columns for c in ("open", "high", "low", "close")):
        hi = d["high"]
        lo = d["low"]
        mx = d[["open", "close"]].max(axis=1)
        mn = d[["open", "close"]].min(axis=1)
        ohlc_violations = int(((hi < mx) | (lo > mn) | (hi < lo)).sum())

    # -- Stale close runs --
    stale_max = 0
    if "close" in d.columns:
        x = d["close"].to_numpy()
        if len(x) > 1:
            run = 1
            for i in range(1, len(x)):
                if np.isfinite(x[i]) and np.isfinite(x[i - 1]) and x[i] == x[i - 1]:
                    run += 1
                    stale_max = max(stale_max, run)
                else:
                    run = 1

    # -- Extreme returns --
    extreme_cnt = 0
    if "close" in d.columns:
        c = pd.to_numeric(d["close"], errors="coerce")
        r = c.pct_change()
        extreme_cnt = int((r.abs() > extreme_ret_threshold).sum())

    # -- Bar coverage --
    received = int(len(d))
    missing_bar_ratio = None
    if expected_bars is not None and expected_bars > 0:
        missing_bar_ratio = max(float(expected_bars - received) / float(expected_bars), 0.0)

    # -- Composite score (simple penalty model) --
    score = 100.0
    score -= 40.0 * (missing_bar_ratio if missing_bar_ratio is not None else 0.0)
    score -= 30.0 * float(np.mean(list(missing_by_col.values())))
    score -= 0.01 * dup
    score -= 0.02 * ooo
    score -= 0.02 * ohlc_violations
    score -= 0.02 * non_positive_prices
    if stale_max >= stale_run_threshold:
        score -= min(10.0, (stale_max - stale_run_threshold) * 0.5)
    score -= min(10.0, extreme_cnt * 0.05)
    score = float(max(min(score, 100.0), 0.0))

    # -- Verdict: based on score AND critical flags --
    # Use violation rates (not absolute counts) so the logic scales with dataset size.
    n = max(received, 1)
    ohlc_viol_rate = ohlc_violations / n

    verdict = "PASS"
    if score < 85.0 or (missing_bar_ratio is not None and missing_bar_ratio > 0.01):
        verdict = "WARN"
    if ohlc_viol_rate > 0.005:  # > 0.5% of bars have OHLC issues -> WARN
        verdict = "WARN"
    if score < 70.0 or non_positive_prices > 0:
        verdict = "FAIL"
    if ohlc_viol_rate > 0.05:   # > 5% of bars have OHLC issues -> FAIL
        verdict = "FAIL"

    return DataQualityReport(
        expected_bars=expected_bars,
        received_bars=received,
        missing_bar_ratio=missing_bar_ratio,
        duplicates=dup,
        out_of_order=ooo,
        missing_by_col=missing_by_col,
        ohlc_violations=ohlc_violations,
        non_positive_prices=non_positive_prices,
        stale_close_runs_max=stale_max,
        extreme_return_count=extreme_cnt,
        score_0_100=score,
        verdict=verdict,
    )


def print_l0_summary(meta: DataLoadMeta, q: DataQualityReport) -> None:
    """Pretty-print the data load summary in tabular format."""
    W = 68  # table width
    SEP = "+" + "-" * (W - 2) + "+"
    HDR = "+" + "=" * (W - 2) + "+"

    def row(label: str, value: str) -> str:
        return f"| {label:<20s} | {value:<{W - 26}s} |"

    # Verdict color hint (works in terminals that support ANSI)
    v = q.verdict
    if v == "PASS":
        badge = "[OK] PASS"
    elif v == "WARN":
        badge = "[!] WARN"
    else:
        badge = "[X] FAIL"

    print(f"\n{HDR}")
    print(f"|{'DATA QUALITY REPORT':^{W - 2}s}|")
    print(HDR)

    # -- Dataset Identity --
    print(SEP)
    print(f"|{'DATASET IDENTITY':^{W - 2}s}|")
    print(SEP)
    print(row("Source", meta.source))
    print(row("Timeframe", meta.timeframe))
    print(row("Symbols", f"{len(meta.symbols)} (e.g. {', '.join(meta.symbols[:5])})"))
    print(row("Requested range", f"{meta.requested_start} -> {meta.requested_end}"))
    print(row("Loaded range", f"{meta.loaded_start} -> {meta.loaded_end}"))
    print(row("Timezone / Cal", f"{meta.timezone} | {meta.calendar}"))
    print(row("Adjusted", str(meta.adjusted)))
    print(row("Fields", ", ".join(meta.fields)))
    print(row("Rows total", f"{meta.rows_total:,}"))
    if meta.dataset_hash:
        print(row("Dataset hash", meta.dataset_hash))
    if meta.notes:
        print(row("Notes", meta.notes))

    # -- Quality Checks --
    print(SEP)
    print(f"|{'QUALITY CHECKS':^{W - 2}s}|")
    print(SEP)

    # Coverage
    exp_str = f"{q.expected_bars:,}" if q.expected_bars is not None else "N/A"
    print(row("Bars received", f"{q.received_bars:,}"))
    print(row("Bars expected", exp_str))
    if q.missing_bar_ratio is not None:
        print(row("Missing bar ratio", f"{q.missing_bar_ratio:.3%}"))
    print(row("Duplicates", str(q.duplicates)))
    print(row("Out-of-order", str(q.out_of_order)))

    # Missingness per column
    miss_str = "  ".join(f"{k}={v:.2%}" for k, v in q.missing_by_col.items())
    print(row("Missing by col", miss_str))

    # Sanity
    n = max(q.received_bars, 1)
    viol_rate = q.ohlc_violations / n
    print(row("OHLC violations", f"{q.ohlc_violations} ({viol_rate:.3%} of bars)"))
    print(row("Non-positive prices", str(q.non_positive_prices)))
    print(row("Stale close max run", str(q.stale_close_runs_max)))
    print(row("Extreme returns", str(q.extreme_return_count)))

    # Score
    print(SEP)
    print(f"| {'SCORE':<20s} | {q.score_0_100:5.1f} / 100    =>    {badge:<{W - 44}s} |")
    print(HDR)
    print()


def print_quality_table(
    results: "list[tuple[DataLoadMeta, DataQualityReport]]",
) -> None:
    """
    Consolidated quality table for multiple datasets.

    Called by alphago_layering.py after running analyze_ohlcv on every
    (symbol, timeframe) pair.  Prints one compact row per dataset plus
    aggregate stats at the bottom.

    Parameters
    ----------
    results : list of (DataLoadMeta, DataQualityReport) tuples
    """
    if not results:
        print("  No datasets to report.\n")
        return

    W = 120
    HDR = "+" + "=" * (W - 2) + "+"
    SEP = "+" + "-" * (W - 2) + "+"

    print(f"\n{HDR}")
    print(f"|{'L0 DATA QUALITY SUMMARY':^{W - 2}s}|")
    print(HDR)

    # Column header
    hdr_fmt = (
        f"| {'Symbol':<14s} {'TF':>4s} {'Bars':>7s} "
        f"{'Start':>12s} {'End':>12s} "
        f"{'Miss%':>6s} {'Dup':>5s} {'OHLC':>5s} "
        f"{'Stale':>5s} {'ExtRet':>6s} {'Score':>6s} {'Verdict':>8s} |"
    )
    print(hdr_fmt)
    print(SEP)

    n_pass = n_warn = n_fail = 0
    scores = []
    failed_syms = []

    for meta, q in results:
        sym_label = meta.symbols[0] if meta.symbols else "?"
        tf_label = meta.timeframe or "?"

        # Format dates: show just YYYY-MM-DD, trim timestamps
        start_str = str(meta.loaded_start)[:10] if meta.loaded_start != "N/A" else "N/A"
        end_str = str(meta.loaded_end)[:10] if meta.loaded_end != "N/A" else "N/A"

        # FIX: Show 0.0% instead of N/A when missing_bar_ratio is None
        # (None means expected_bars wasn't provided, so we assume no gaps)
        miss_pct = (
            f"{q.missing_bar_ratio:.1%}" if q.missing_bar_ratio is not None else "0.0%"
        )

        badge = q.verdict
        if badge == "PASS":
            n_pass += 1
        elif badge == "WARN":
            n_warn += 1
        else:
            n_fail += 1
            failed_syms.append(sym_label)

        scores.append(q.score_0_100)

        row = (
            f"| {sym_label:<14s} {tf_label:>4s} {q.received_bars:>7,d} "
            f"{start_str:>12s} {end_str:>12s} "
            f"{miss_pct:>6s} {q.duplicates:>5d} {q.ohlc_violations:>5d} "
            f"{q.stale_close_runs_max:>5d} {q.extreme_return_count:>6d} "
            f"{q.score_0_100:>5.1f} {badge:>8s} |"
        )
        print(row)

    # Aggregate
    print(SEP)
    avg_score = float(np.mean(scores)) if scores else 0.0
    total = len(results)
    summary = (
        f"|  {total} datasets  |  Avg Score: {avg_score:.1f}  |  "
        f"PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}"
    )
    summary += " " * max(0, W - 2 - len(summary) + 1) + "|"
    print(summary)

    if failed_syms:
        fail_str = f"|  FAILED: {', '.join(failed_syms[:15])}"
        if len(failed_syms) > 15:
            fail_str += f" (+{len(failed_syms) - 15} more)"
        fail_str += " " * max(0, W - 2 - len(fail_str) + 1) + "|"
        print(fail_str)

    print(f"{HDR}")
    sys.stdout.flush()


# ============================================================================
# WS2C: EXCHANGE TRADING CALENDAR
# ============================================================================

class TradingCalendar:
    """
    Exchange-specific trading calendar.

    Loads from JSON config or uses built-in NYSE defaults.
    Validates: no trading on non-trading days.

    Usage:
        cal = TradingCalendar("NYSE")
        assert cal.is_trading_day("2024-01-15")
        days = cal.trading_days_between("2024-01-01", "2024-12-31")
    """

    # Default NYSE holidays (2020-2026 subset)
    _NYSE_HOLIDAYS = {
        # Format: "YYYY-MM-DD"
        # New Year's Day, MLK Day, Presidents Day, Good Friday,
        # Memorial Day, Juneteenth, Independence Day, Labor Day,
        # Thanksgiving, Christmas
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
        "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
        "2024-11-28", "2024-12-25",
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
        "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
        "2025-11-27", "2025-12-25",
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
        "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
        "2026-11-26", "2026-12-25",
    }

    def __init__(self, exchange: str = "NYSE", holidays_path: str = None):
        self.exchange = exchange
        self._holidays: set = set()

        if holidays_path and os.path.exists(holidays_path):
            import json
            with open(holidays_path) as f:
                data = json.load(f)
            self._holidays = set(data.get("holidays", []))
        else:
            self._holidays = set(self._NYSE_HOLIDAYS)

    def is_trading_day(self, date_str: str) -> bool:
        """Check if a given date is a trading day."""
        dt = pd.Timestamp(date_str)
        # Weekend check
        if dt.weekday() >= 5:
            return False
        # Holiday check
        return dt.strftime("%Y-%m-%d") not in self._holidays

    def trading_days_between(self, start: str, end: str) -> List[str]:
        """Return list of trading day date strings between start and end."""
        dates = pd.date_range(start, end, freq='B')  # Business days
        return [
            d.strftime("%Y-%m-%d") for d in dates
            if d.strftime("%Y-%m-%d") not in self._holidays
        ]

    def next_trading_day(self, date_str: str) -> str:
        """Return the next trading day after date_str."""
        dt = pd.Timestamp(date_str) + pd.Timedelta(days=1)
        while not self.is_trading_day(dt.strftime("%Y-%m-%d")):
            dt += pd.Timedelta(days=1)
        return dt.strftime("%Y-%m-%d")

    def validate_timestamps(self, timestamps: pd.Series) -> Dict:
        """
        Validate that timestamps correspond to trading days.
        Returns dict with counts of violations.
        """
        ts = pd.to_datetime(timestamps, errors='coerce')
        weekend_bars = int((ts.dt.weekday >= 5).sum())
        holiday_bars = 0
        for t in ts.dropna():
            if t.strftime("%Y-%m-%d") in self._holidays:
                holiday_bars += 1
        return {
            'total_bars': len(ts),
            'weekend_bars': weekend_bars,
            'holiday_bars': holiday_bars,
            'non_trading_bars': weekend_bars + holiday_bars,
        }

    def expected_bars(self, start: str, end: str) -> int:
        """Expected number of trading days between start and end."""
        return len(self.trading_days_between(start, end))


# ============================================================================
# WS2D: DATA INGESTION GATE
# ============================================================================

class DataIngestionGate:
    """
    Validates data at ingestion. Rejects or quarantines bad records
    BEFORE they enter the pipeline.

    Policy (configurable via MissingDataPolicy):
      - Non-positive prices -> REJECT entire bar (always)
      - OHLC violation -> QUARANTINE (flag, but include) (always)
      - Missing close -> REJECT bar (always)
      - Missing volume -> Handled per missing_data_policy:
            FORWARD_FILL:   ffill then bfill
            ROLLING_MEDIAN: fill with rolling 20-bar median (legacy default)
            INTERPOLATE:    linear interpolation
            MASK:           keep NaN, flag in metadata
            EXCLUDE:        drop entire bar
      - Extreme return > threshold -> QUARANTINE with manual review flag

    Usage:
        gate = DataIngestionGate(policy=MissingDataPolicy.FORWARD_FILL)
        clean_df, report = gate.ingest(raw_df, calendar=TradingCalendar())
    """

    def __init__(self, extreme_return_threshold: float = 0.50,
                 policy: MissingDataPolicy = MissingDataPolicy.ROLLING_MEDIAN):
        self.extreme_threshold = extreme_return_threshold
        self.policy = policy

    def ingest(self, df: pd.DataFrame,
               calendar: TradingCalendar = None,
               ts_col: str = "timestamp") -> tuple:
        """
        Validate and clean data.

        CRITICAL: preserves original column names (e.g. 'Close' from yfinance).
        All checks use case-insensitive column lookup internally.

        Missing-data handling follows self.policy (MissingDataPolicy enum).

        Returns: (clean_df, rejection_report_dict)
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(), {'rejected': 0, 'quarantined': 0, 'reason': 'empty'}

        d = df.copy()
        rejected_indices = []
        quarantined_indices = []
        fill_count = 0

        # Build case-insensitive column map: lowercase -> original name
        # This lets us find 'Close', 'close', 'CLOSE' etc. without renaming
        col_map = {c.lower(): c for c in d.columns}

        def _col(name):
            """Resolve a lowercase name to the actual column name in df."""
            return col_map.get(name.lower())

        # 1. Reject non-positive prices
        for pc in ("close", "open"):
            actual = _col(pc)
            if actual and actual in d.columns:
                bad = d[actual] <= 0
                rejected_indices.extend(d.index[bad].tolist())

        # 2. Reject missing close (always, regardless of policy)
        close_col = _col("close")
        if close_col and close_col in d.columns:
            bad = d[close_col].isna()
            rejected_indices.extend(d.index[bad].tolist())

        # 3. Handle missing volume per configured policy
        vol_col = _col("volume")
        if vol_col and vol_col in d.columns:
            mask = d[vol_col].isna() | (d[vol_col] <= 0)
            if mask.any():
                fill_count = int(mask.sum())
                if self.policy == MissingDataPolicy.ROLLING_MEDIAN:
                    med = d[vol_col].rolling(20, min_periods=1).median()
                    d.loc[mask, vol_col] = med[mask]
                elif self.policy == MissingDataPolicy.FORWARD_FILL:
                    d[vol_col] = d[vol_col].replace(0, np.nan)
                    d[vol_col] = d[vol_col].ffill().bfill()
                elif self.policy == MissingDataPolicy.INTERPOLATE:
                    d.loc[mask, vol_col] = np.nan
                    d[vol_col] = d[vol_col].interpolate(method='linear').bfill()
                elif self.policy == MissingDataPolicy.MASK:
                    # Keep NaN â€” downstream must handle. Flag only.
                    d.loc[mask, vol_col] = np.nan
                elif self.policy == MissingDataPolicy.EXCLUDE:
                    rejected_indices.extend(d.index[mask].tolist())

        # 4. Quarantine OHLC violations
        o, h, l, c = _col("open"), _col("high"), _col("low"), _col("close")
        if all(x and x in d.columns for x in (o, h, l, c)):
            hi = d[h]
            lo = d[l]
            mx = d[[o, c]].max(axis=1)
            mn = d[[o, c]].min(axis=1)
            ohlc_bad = (hi < mx) | (lo > mn) | (hi < lo)
            quarantined_indices.extend(d.index[ohlc_bad].tolist())

        # 5. Quarantine extreme returns
        if close_col and close_col in d.columns:
            rets = d[close_col].pct_change().abs()
            extreme = rets > self.extreme_threshold
            quarantined_indices.extend(d.index[extreme].tolist())

        # 6. Calendar validation (if provided)
        # Handles both column-based timestamps and DatetimeIndex (yfinance)
        calendar_violations = 0
        if calendar is not None:
            ts_actual = _col(ts_col) or _col("date") or _col("timestamp")
            if ts_actual and ts_actual in d.columns:
                cal_report = calendar.validate_timestamps(d[ts_actual])
                calendar_violations = cal_report['non_trading_bars']
            elif hasattr(d.index, 'to_series'):
                # DatetimeIndex (e.g., yfinance delivers DatetimeIndex)
                try:
                    cal_report = calendar.validate_timestamps(d.index.to_series())
                    calendar_violations = cal_report['non_trading_bars']
                except Exception:
                    pass

        # Remove rejected rows
        rejected_set = set(rejected_indices)
        clean = d.loc[~d.index.isin(rejected_set)]

        # Flag quarantined rows (keep them but mark)
        quarantined_set = set(quarantined_indices) - rejected_set
        if quarantined_set:
            clean = clean.copy()
            clean['_quarantined'] = clean.index.isin(quarantined_set)

        report = {
            'input_rows': len(df),
            'output_rows': len(clean),
            'rejected': len(rejected_set),
            'quarantined': len(quarantined_set),
            'volume_filled': fill_count,
            'calendar_violations': calendar_violations,
            'missing_data_policy': self.policy.value,
        }

        return clean, report


# ============================================================================
# WS2A: POINT-IN-TIME DATA STORE (File-Based)
# ============================================================================

class PointInTimeStore:
    """
    File-based point-in-time data store.

    Structure:
        data_pit/
          snapshots/
            2024-01-15_v001/   # Immutable snapshot
              AAPL_1d.parquet
              manifest.json    # {hash, timestamp, source, version}
          universe/
            members.csv

    Query: "Give me AAPL closes as-of 2024-01-15"
      -> Finds the most recent snapshot before that date
      -> Returns only rows with timestamp <= 2024-01-15
      -> No retroactive corrections visible

    Usage:
        pit = PointInTimeStore("data_pit")
        snap_id = pit.create_snapshot({"AAPL": df_aapl}, source="yfinance", version="v1")
        df = pit.query_as_of("AAPL", "2024-01-15")
    """

    def __init__(self, root_dir: str = "data_pit"):
        self.root_dir = root_dir
        self._snapshots_dir = os.path.join(root_dir, "snapshots")
        os.makedirs(self._snapshots_dir, exist_ok=True)

    def create_snapshot(self, data: Dict[str, pd.DataFrame],
                        source: str = "unknown",
                        version: str = "v001") -> str:
        """
        Create an immutable snapshot. Returns snapshot_id.

        Each snapshot is a directory containing one parquet file per symbol
        and a manifest.json with metadata.
        """
        import json
        import time

        snap_id = f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{version}"
        snap_dir = os.path.join(self._snapshots_dir, snap_id)
        os.makedirs(snap_dir, exist_ok=True)

        manifest = {
            'snapshot_id': snap_id,
            'source': source,
            'version': version,
            'timestamp': pd.Timestamp.now().isoformat(),
            'symbols': list(data.keys()),
            'row_counts': {},
        }

        for sym, df in data.items():
            try:
                path = os.path.join(snap_dir, f"{sym}.parquet")
                df.to_parquet(path)
            except Exception:
                # Fallback to CSV if parquet engine not available
                path = os.path.join(snap_dir, f"{sym}.csv")
                df.to_csv(path, index=False)
            manifest['row_counts'][sym] = len(df)

        # Write manifest
        with open(os.path.join(snap_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        return snap_id

    def query_as_of(self, symbol: str, as_of_date: str,
                    fields: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Return data as it was known on as_of_date.

        Finds the most recent snapshot before as_of_date,
        then filters rows to only those with timestamp <= as_of_date.
        """
        snapshots = self._list_snapshots()
        if not snapshots:
            return None

        # Find most recent snapshot before as_of_date
        target = pd.Timestamp(as_of_date)
        best_snap = None
        for snap_id in sorted(snapshots, reverse=True):
            snap_ts = self._snapshot_timestamp(snap_id)
            if snap_ts is not None and snap_ts <= target:
                best_snap = snap_id
                break

        if best_snap is None and snapshots:
            best_snap = sorted(snapshots)[-1]  # Use latest available

        if best_snap is None:
            return None

        # Load symbol data from snapshot
        parquet_path = os.path.join(self._snapshots_dir, best_snap, f"{symbol}.parquet")
        csv_path = os.path.join(self._snapshots_dir, best_snap, f"{symbol}.csv")
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                return None
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            return None

        # Filter to as-of date
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            target_tz = target.tz_localize('UTC') if target.tz is None else target.tz_convert('UTC')
            df = df[df['timestamp'] <= target_tz]

        if fields:
            avail = [f for f in fields if f in df.columns]
            df = df[avail]

        return df

    def _list_snapshots(self) -> List[str]:
        """List all snapshot IDs."""
        if not os.path.exists(self._snapshots_dir):
            return []
        return [d for d in os.listdir(self._snapshots_dir)
                if os.path.isdir(os.path.join(self._snapshots_dir, d))]

    def _snapshot_timestamp(self, snap_id: str) -> Optional[pd.Timestamp]:
        """Extract timestamp from snapshot manifest."""
        import json
        manifest_path = os.path.join(self._snapshots_dir, snap_id, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                data = json.load(f)
            return pd.Timestamp(data.get('timestamp', snap_id[:8]))
        # Fallback: parse from directory name
        try:
            return pd.Timestamp(snap_id[:8])
        except Exception:
            return None

    def diff_snapshots(self, snap_a: str, snap_b: str) -> Dict:
        """Report what changed between two snapshots (for audit)."""
        import json

        def _load_manifest(snap_id):
            path = os.path.join(self._snapshots_dir, snap_id, "manifest.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
            return {}

        ma = _load_manifest(snap_a)
        mb = _load_manifest(snap_b)

        syms_a = set(ma.get('symbols', []))
        syms_b = set(mb.get('symbols', []))

        return {
            'added_symbols': list(syms_b - syms_a),
            'removed_symbols': list(syms_a - syms_b),
            'common_symbols': list(syms_a & syms_b),
            'row_count_changes': {
                s: mb.get('row_counts', {}).get(s, 0) - ma.get('row_counts', {}).get(s, 0)
                for s in syms_a & syms_b
            },
        }


# ============================================================================
# WS2B: SURVIVORSHIP BIAS MODULE
# ============================================================================

class UniverseManager:
    """
    Time-stamped universe membership tracking.

    Tracks: index additions, removals, delistings, mergers, spinoffs.
    Query: "What was in the S&P 500 on 2020-03-15?"

    For backtesting: the universe at each rebalance date must be
    reconstructed from historical membership, not current constituents.

    Usage:
        um = UniverseManager()
        um.add_event("AAPL", "2020-01-01", "add")
        um.add_event("GE", "2021-06-15", "remove")
        universe = um.universe_as_of("2021-01-01")  # includes GE
    """

    VALID_EVENTS = ('add', 'remove', 'delist', 'merge_into', 'spinoff_from')

    def __init__(self, membership_path: str = None):
        self._events: List[Dict] = []

        if membership_path and os.path.exists(membership_path):
            df = pd.read_csv(membership_path)
            for _, row in df.iterrows():
                self._events.append({
                    'date': str(row.get('date', '')),
                    'symbol': str(row.get('symbol', '')),
                    'event': str(row.get('event', 'add')),
                })

    def add_event(self, symbol: str, date: str, event: str):
        """Record a universe membership event."""
        assert event in self.VALID_EVENTS, f"Invalid event: {event}"
        self._events.append({
            'date': date, 'symbol': symbol, 'event': event,
        })

    def universe_as_of(self, date: str,
                       min_mcap: float = None,
                       min_adv: float = None) -> List[str]:
        """Symbols that were active members as of date."""
        target = pd.Timestamp(date)
        active = set()

        # Sort events by date
        sorted_events = sorted(self._events, key=lambda e: e['date'])

        for ev in sorted_events:
            if pd.Timestamp(ev['date']) > target:
                break
            sym = ev['symbol']
            if ev['event'] in ('add', 'spinoff_from'):
                active.add(sym)
            elif ev['event'] in ('remove', 'delist', 'merge_into'):
                active.discard(sym)

        return sorted(list(active))

    def is_survivor(self, symbol: str, start: str, end: str) -> bool:
        """Did this symbol survive the entire backtest period?"""
        for ev in self._events:
            if (ev['symbol'] == symbol
                    and ev['event'] in ('delist', 'merge_into', 'remove')
                    and start <= ev['date'] <= end):
                return False
        return True

    def delisted_between(self, start: str, end: str) -> List[Dict]:
        """All delisting events in the period (for inclusion in backtest)."""
        return [
            ev for ev in self._events
            if ev['event'] == 'delist' and start <= ev['date'] <= end
        ]


# ============================================================================
# WS2E: CORPORATE ACTIONS ENGINE
# ============================================================================

class CorporateActionsEngine:
    """
    Manages corporate action adjustments for consistent backtesting.

    Handles: stock splits (forward/reverse), cash dividends, mergers.
    Produces BOTH "total return" and "price return" series explicitly.
    All adjustments are logged and auditable.

    Usage:
        ca = CorporateActionsEngine()
        ca.add_split("AAPL", "2020-08-31", ratio=4.0)
        ca.add_dividend("AAPL", "2024-02-09", amount=0.24)
        adj = ca.adjust_prices("AAPL", raw_prices, dates, return_type="total")
    """

    def __init__(self):
        # {asset_id: [{'type': ..., 'ex_date': ..., ...}]}
        self._actions: Dict[str, List[Dict]] = {}
        self._adjustment_log: List[Dict] = []

    def add_split(self, asset_id: str, ex_date: str, ratio: float):
        """Register a stock split. ratio=4.0 means 4-for-1 forward split."""
        self._actions.setdefault(asset_id, []).append({
            'type': 'split', 'ex_date': ex_date, 'ratio': ratio,
        })
        self._actions[asset_id].sort(key=lambda a: a['ex_date'])

    def add_dividend(self, asset_id: str, ex_date: str, amount: float):
        """Register a cash dividend (per share)."""
        self._actions.setdefault(asset_id, []).append({
            'type': 'dividend', 'ex_date': ex_date, 'amount': amount,
        })
        self._actions[asset_id].sort(key=lambda a: a['ex_date'])

    def add_merger(self, asset_id: str, effective_date: str,
                   successor_id: str, conversion_ratio: float = 1.0):
        """Register a merger/acquisition event."""
        self._actions.setdefault(asset_id, []).append({
            'type': 'merger', 'ex_date': effective_date,
            'successor': successor_id, 'ratio': conversion_ratio,
        })

    def adjust_prices(self, asset_id: str,
                      prices: np.ndarray,
                      dates: List[str],
                      return_type: str = "total",
                      as_of: str = "9999-12-31") -> np.ndarray:
        """
        Adjust historical prices for corporate actions.

        Args:
            prices: Raw unadjusted price array
            dates: Corresponding date strings (ISO format)
            return_type: "total" (includes dividends) or "price" (splits only)
            as_of: Only apply actions known by this date (PIT-safe)

        Returns:
            Adjusted price array (same length as input)
        """
        if asset_id not in self._actions or len(prices) == 0:
            return prices.copy()

        adjusted = np.array(prices, dtype=np.float64)
        actions = [a for a in self._actions[asset_id] if a['ex_date'] <= as_of]

        for action in actions:
            if action['type'] == 'split' and action.get('ratio', 1.0) != 0:
                ratio = action['ratio']
                for i, d in enumerate(dates):
                    if d < action['ex_date']:
                        adjusted[i] /= ratio
                self._adjustment_log.append({
                    'asset': asset_id, 'type': 'split',
                    'ex_date': action['ex_date'], 'ratio': ratio,
                })

            elif action['type'] == 'dividend' and return_type == 'total':
                amt = action.get('amount', 0.0)
                for i, d in enumerate(dates):
                    if d >= action['ex_date']:
                        adjusted[i] += amt
                self._adjustment_log.append({
                    'asset': asset_id, 'type': 'dividend',
                    'ex_date': action['ex_date'], 'amount': amt,
                })

        return adjusted

    def adjust_volume(self, asset_id: str,
                      volumes: np.ndarray,
                      dates: List[str],
                      as_of: str = "9999-12-31") -> np.ndarray:
        """Adjust historical volume for splits (inverse of price adj)."""
        if asset_id not in self._actions:
            return volumes.copy()

        adjusted = np.array(volumes, dtype=np.float64)
        splits = [a for a in self._actions[asset_id]
                  if a['type'] == 'split' and a['ex_date'] <= as_of]

        for action in splits:
            ratio = action.get('ratio', 1.0)
            if ratio != 0:
                for i, d in enumerate(dates):
                    if d < action['ex_date']:
                        adjusted[i] *= ratio
        return adjusted

    def get_actions(self, asset_id: str, start: str = "",
                    end: str = "9999-12-31") -> List[Dict]:
        return [a for a in self._actions.get(asset_id, [])
                if start <= a['ex_date'] <= end]

    @property
    def adjustment_log(self) -> List[Dict]:
        return list(self._adjustment_log)


# ============================================================================
# WS2F: SCHEMA VALIDATOR (Ingestion-time enforcement)
# ============================================================================

class SchemaValidator:
    """
    Enforces expected dtypes, ranges, and non-null constraints at ingestion.
    Rejects or quarantines bad records BEFORE they enter the pipeline.
    Prevents silent NaN propagation.

    Usage:
        sv = SchemaValidator()
        sv.add_field("close", dtype="float", nullable=False, min_value=0.0)
        sv.add_field("volume", dtype="int", nullable=False, min_value=0)
        report = sv.validate_record({'close': 150.0, 'volume': 1000000})
    """

    def __init__(self):
        self._schema: Dict[str, Dict] = {}
        self._violation_log: List[Dict] = []
        self._records_validated: int = 0
        self._records_rejected: int = 0

    def add_field(self, name: str, dtype: str = "float",
                  nullable: bool = False,
                  min_value: float = None, max_value: float = None):
        """Register expected schema for a field."""
        self._schema[name] = {
            'dtype': dtype, 'nullable': nullable,
            'min_value': min_value, 'max_value': max_value,
        }

    def validate_record(self, record: Dict, record_id: str = ""
                        ) -> Dict[str, Any]:
        """
        Validate a single record against schema.

        Returns: {'valid': bool, 'violations': [...], 'quarantined_fields': [...]}
        """
        self._records_validated += 1
        violations = []
        quarantined = []

        for name, schema in self._schema.items():
            value = record.get(name)

            # Null check
            if value is None or (isinstance(value, float) and np.isnan(value)):
                if not schema['nullable']:
                    violations.append(f"{name}: null not allowed")
                    quarantined.append(name)
                continue

            # Type check
            if schema['dtype'] == 'float':
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    violations.append(f"{name}: cannot cast to float")
                    quarantined.append(name)
                    continue
            elif schema['dtype'] == 'int':
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    violations.append(f"{name}: cannot cast to int")
                    quarantined.append(name)
                    continue

            # Range checks
            if schema['min_value'] is not None and value < schema['min_value']:
                violations.append(f"{name}: {value} < min {schema['min_value']}")
                quarantined.append(name)
            if schema['max_value'] is not None and value > schema['max_value']:
                violations.append(f"{name}: {value} > max {schema['max_value']}")
                quarantined.append(name)

        valid = len(violations) == 0
        if not valid:
            self._records_rejected += 1
            self._violation_log.append({
                'record_id': record_id, 'violations': violations,
            })

        return {
            'valid': valid,
            'violations': violations,
            'quarantined_fields': quarantined,
        }

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate an entire DataFrame row-by-row."""
        results = []
        for i, row in enumerate(df.to_dict('records')):
            results.append(self.validate_record(row, str(i)))
        n_valid = sum(1 for r in results if r['valid'])
        return {
            'total': len(df), 'valid': n_valid,
            'rejected': len(df) - n_valid,
            'rejection_rate': (len(df) - n_valid) / max(len(df), 1),
        }

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            'n_fields': len(self._schema),
            'records_validated': self._records_validated,
            'records_rejected': self._records_rejected,
            'rejection_rate': self._records_rejected / max(self._records_validated, 1),
        }


# ============================================================================
# WS2G: VENDOR RECONCILIATION
# ============================================================================

class VendorReconciler:
    """
    Cross-check multiple data sources for critical fields.
    Flags and logs discrepancies when vendors disagree beyond tolerance.

    Usage:
        vr = VendorReconciler(tolerance_pct=0.01)
        vr.add_observation('AAPL', 'close', '2024-01-15', 'vendor_a', 185.50)
        vr.add_observation('AAPL', 'close', '2024-01-15', 'vendor_b', 185.55)
        discreps = vr.check_discrepancies()
    """

    def __init__(self, tolerance_pct: float = 0.01):
        self.tolerance_pct = tolerance_pct
        # {(asset, field, date): {vendor: value}}
        self._observations: Dict[tuple, Dict[str, float]] = {}
        self._discrepancy_log: List[Dict] = []

    def add_observation(self, asset_id: str, field_name: str,
                        date_str: str, vendor: str, value: float):
        key = (asset_id, field_name, date_str)
        if key not in self._observations:
            self._observations[key] = {}
        self._observations[key][vendor] = value

    def check_discrepancies(self, asset_id: str = None,
                            field_name: str = None) -> List[Dict]:
        """Check all observations for discrepancies beyond tolerance."""
        discrepancies = []
        for key, vendors in self._observations.items():
            a, f, d = key
            if asset_id and a != asset_id:
                continue
            if field_name and f != field_name:
                continue
            if len(vendors) < 2:
                continue

            values = list(vendors.values())
            mean_val = np.mean(values)
            max_diff = max(values) - min(values)
            diff_pct = max_diff / (abs(mean_val) + 1e-12)

            if diff_pct > self.tolerance_pct:
                record = {
                    'asset': a, 'field': f, 'date': d,
                    'vendors': dict(vendors),
                    'diff_pct': float(diff_pct),
                    'tolerance': self.tolerance_pct,
                    'flagged': True,
                }
                discrepancies.append(record)
                self._discrepancy_log.append(record)

        return discrepancies

    @property
    def n_discrepancies(self) -> int:
        return len(self._discrepancy_log)

    @property
    def discrepancy_log(self) -> List[Dict]:
        return list(self._discrepancy_log)