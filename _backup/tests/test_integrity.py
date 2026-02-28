"""
tests/test_integrity.py -- Core invariant tests for the trading system.

Covers:
  5a) Cost model invariants (alphago_cost_model.py)
  5b) L0 Data quality invariants (data_quality.py)
  5c) WS1: Anti-Overfitting Framework (validation_engine.py)
  5d) WS2: L0 Data Infrastructure (data_quality.py new classes)
  5e) WS3: L3 Risk Controls (alphago_architecture.py new classes)
  5f) WS4: Evaluation & Reporting (backtest_report.py)
  5g) WS5: L2/L4 Hardening (alphago_architecture.py new classes)
  5h) WS6: Reproducibility (validation_engine.py)

Run with: python -m pytest tests/test_integrity.py -v
"""

import sys
import os
import math
import tempfile
import json
import shutil

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphago_cost_model import estimate_cost_dollars, fill_price_from_cost, CostBreakdown


# ============================================================================
# 5a) Cost model invariants
# ============================================================================

class TestCostModelMonotonicity:
    """Larger trades must cost more (all else equal)."""

    def test_cost_increases_with_notional(self):
        cb_small = estimate_cost_dollars(
            mid=100, trade_notional=10_000, adv_dollars=10_000_000,
            spread_bps=1.0, impact_coeff=0.1, fees_pct=0.0,
        )
        cb_large = estimate_cost_dollars(
            mid=100, trade_notional=100_000, adv_dollars=10_000_000,
            spread_bps=1.0, impact_coeff=0.1, fees_pct=0.0,
        )
        assert cb_large.total_cost > cb_small.total_cost, (
            f"Cost should increase with trade size: {cb_large.total_cost} <= {cb_small.total_cost}"
        )

    def test_adv_reduces_impact(self):
        """Higher ADV = lower market impact for same trade size."""
        cb_illiquid = estimate_cost_dollars(
            mid=100, trade_notional=100_000, adv_dollars=1_000_000,
            spread_bps=1.0, impact_coeff=0.1, fees_pct=0.0,
        )
        cb_liquid = estimate_cost_dollars(
            mid=100, trade_notional=100_000, adv_dollars=100_000_000,
            spread_bps=1.0, impact_coeff=0.1, fees_pct=0.0,
        )
        assert cb_liquid.total_cost < cb_illiquid.total_cost, (
            f"Liquid ADV should have lower cost: {cb_liquid.total_cost} >= {cb_illiquid.total_cost}"
        )

    def test_zero_trade_zero_cost(self):
        """No trade = no cost."""
        cb = estimate_cost_dollars(
            mid=100, trade_notional=0.0, adv_dollars=10_000_000,
            spread_bps=1.0, impact_coeff=0.1, fees_pct=0.001,
        )
        assert cb.total_cost == 0.0

    def test_cost_always_non_negative(self):
        """Cost must never be negative."""
        for notional in [0, 100, 10_000, 1_000_000]:
            cb = estimate_cost_dollars(
                mid=50, trade_notional=notional, adv_dollars=5_000_000,
                spread_bps=2.0, impact_coeff=0.15, fees_pct=0.001,
            )
            assert cb.total_cost >= 0.0
            assert cb.half_spread_cost >= 0.0
            assert cb.impact_cost >= 0.0
            assert cb.fees_cost >= 0.0

    def test_cost_components_sum_to_total(self):
        cb = estimate_cost_dollars(
            mid=100, trade_notional=50_000, adv_dollars=10_000_000,
            spread_bps=1.5, impact_coeff=0.12, fees_pct=0.0005,
        )
        assert abs(cb.half_spread_cost + cb.impact_cost + cb.fees_cost - cb.total_cost) < 1e-10


class TestFillPriceConsistency:
    """Fill price must move adversely (buy higher, sell lower)."""

    def test_buy_fill_above_mid(self):
        fill, cb = fill_price_from_cost(
            mid=100.0, side="buy", trade_notional=50_000,
            adv_dollars=10_000_000, spread_bps=1.0, impact_coeff=0.1,
        )
        assert fill > 100.0, f"Buy fill should be above mid: {fill}"

    def test_sell_fill_below_mid(self):
        fill, cb = fill_price_from_cost(
            mid=100.0, side="sell", trade_notional=50_000,
            adv_dollars=10_000_000, spread_bps=1.0, impact_coeff=0.1,
        )
        assert fill < 100.0, f"Sell fill should be below mid: {fill}"

    def test_symmetry(self):
        """Buy slippage and sell slippage should be symmetric."""
        fill_buy, _ = fill_price_from_cost(
            mid=100.0, side="buy", trade_notional=50_000,
            adv_dollars=10_000_000, spread_bps=1.0, impact_coeff=0.1,
        )
        fill_sell, _ = fill_price_from_cost(
            mid=100.0, side="sell", trade_notional=50_000,
            adv_dollars=10_000_000, spread_bps=1.0, impact_coeff=0.1,
        )
        slip_buy = fill_buy - 100.0
        slip_sell = 100.0 - fill_sell
        assert abs(slip_buy - slip_sell) < 1e-10


# ============================================================================
# 5b) L0 Data Quality invariants
# ============================================================================

class TestL0DataQuality:
    """L0 data quality module should catch common data problems."""

    def test_clean_data_passes(self):
        from data_quality import analyze_ohlcv
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(252) * 0.5)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices + np.random.randn(252) * 0.1,
            "high": prices + abs(np.random.randn(252) * 0.5),
            "low": prices - abs(np.random.randn(252) * 0.5),
            "close": prices,
            "volume": np.random.randint(1_000_000, 10_000_000, 252),
        })
        # Ensure OHLC integrity
        df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
        df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01

        q = analyze_ohlcv(df, ts_col="timestamp")
        assert q.verdict == "PASS", f"Clean data should PASS, got {q.verdict} (score={q.score_0_100})"
        assert q.ohlc_violations == 0
        assert q.non_positive_prices == 0

    def test_negative_prices_fail(self):
        from data_quality import analyze_ohlcv
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="B"),
            "open": [100]*10,
            "high": [101]*10,
            "low": [99]*10,
            "close": [100, -5, 100, 100, 100, 100, 100, 100, 100, 100],
            "volume": [1_000_000]*10,
        })
        q = analyze_ohlcv(df, ts_col="timestamp")
        assert q.non_positive_prices > 0
        assert q.verdict == "FAIL"

    def test_empty_df_fails(self):
        from data_quality import analyze_ohlcv
        q = analyze_ohlcv(pd.DataFrame())
        assert q.verdict == "FAIL"
        assert q.received_bars == 0


# ============================================================================
# 5c) WS1: Anti-Overfitting Framework (validation_engine.py)
# ============================================================================

class TestPurgedWalkForwardCV:
    """PurgedWalkForwardCV must prevent train/test leakage."""

    def test_folds_generated(self):
        from validation_engine import PurgedWalkForwardCV
        cv = PurgedWalkForwardCV(n_obs=1000, purge_gap=21, embargo=5, n_splits=5)
        folds = cv.generate_folds()
        assert len(folds) > 0, "Must generate at least one fold"

    def test_no_leakage_in_any_fold(self):
        """Train must never overlap with test after purge + embargo."""
        from validation_engine import PurgedWalkForwardCV
        cv = PurgedWalkForwardCV(n_obs=2000, purge_gap=21, embargo=5, n_splits=5)
        folds = cv.generate_folds()
        for f in folds:
            # Test must start after train end + purge + embargo
            assert f.test_start > f.train_end, (
                f"Fold {f.fold_id}: test_start={f.test_start} <= train_end={f.train_end}"
            )
            assert f.test_start >= f.embargo_end, (
                f"Fold {f.fold_id}: test_start={f.test_start} < embargo_end={f.embargo_end}"
            )
            # Purge gap is at least what was requested
            assert f.purge_end >= f.train_end + cv.purge_gap

    def test_holdout_never_in_folds(self):
        """Final holdout region is never used in any fold."""
        from validation_engine import PurgedWalkForwardCV
        cv = PurgedWalkForwardCV(n_obs=1000, purge_gap=21, embargo=5, n_splits=5, holdout_pct=0.20)
        folds = cv.generate_folds()
        holdout_start, _ = cv.get_holdout_indices()
        for f in folds:
            assert f.test_end <= holdout_start, (
                f"Fold {f.fold_id}: test_end={f.test_end} exceeds holdout_start={holdout_start}"
            )
            assert f.train_end < holdout_start

    def test_expanding_window(self):
        """Training windows must be expanding (non-shrinking)."""
        from validation_engine import PurgedWalkForwardCV
        cv = PurgedWalkForwardCV(n_obs=2000, purge_gap=21, embargo=5, n_splits=5)
        folds = cv.generate_folds()
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                assert folds[i].train_end >= folds[i-1].train_end, "Windows must expand"

    def test_summary_reports(self):
        from validation_engine import PurgedWalkForwardCV
        cv = PurgedWalkForwardCV(n_obs=500, purge_gap=10, embargo=3, n_splits=3)
        s = cv.summary()
        assert 'n_folds' in s
        assert s['purge_gap'] == 10
        assert s['holdout_size'] > 0


class TestSignificanceGate:
    """SignificanceGate must filter overfit alphas."""

    def test_strong_alpha_passes(self):
        """Alpha with genuine signal should PASS."""
        from validation_engine import SignificanceGate
        gate = SignificanceGate()
        np.random.seed(42)
        # Strong daily signal: 40bps/day mean, 1% daily vol -> annualized Sharpe ~ 6.3
        # Over 2 years OOS, t-stat ~ 6.3 * sqrt(2) ~ 8.9
        is_rets = np.random.randn(504) * 0.01 + 0.004
        oos_rets = np.random.randn(504) * 0.01 + 0.004
        report = gate.evaluate("good_alpha", is_rets, oos_rets, n_configs_tested=1)
        assert report.verdict in ("PASS", "MARGINAL"), (
            f"Strong alpha should not REJECT: {report.verdict}, t={report.t_stat:.2f}"
        )

    def test_noise_alpha_rejected(self):
        """Pure noise alpha should be REJECTED."""
        from validation_engine import SignificanceGate
        gate = SignificanceGate()
        np.random.seed(99)
        is_rets = np.random.randn(252) * 0.01   # Zero mean
        oos_rets = np.random.randn(252) * 0.01
        report = gate.evaluate("noise_alpha", is_rets, oos_rets, n_configs_tested=50)
        assert report.verdict == "REJECT", (
            f"Noise should REJECT: {report.verdict}, t={report.t_stat:.2f}"
        )

    def test_overfit_detected_via_decay(self):
        """Alpha with great IS but bad OOS should flag high decay."""
        from validation_engine import SignificanceGate
        gate = SignificanceGate()
        np.random.seed(42)
        is_rets = np.random.randn(252) * 0.01 + 0.003  # Great IS
        oos_rets = np.random.randn(252) * 0.01 - 0.0005  # Bad OOS
        report = gate.evaluate("overfit_alpha", is_rets, oos_rets)
        # OOS/IS decay should be poor
        assert report.oos_is_decay_ratio < 0.5 or report.verdict != "PASS"

    def test_report_fields_populated(self):
        from validation_engine import SignificanceGate
        gate = SignificanceGate()
        np.random.seed(42)
        report = gate.evaluate(
            "test", np.random.randn(100) * 0.01, np.random.randn(100) * 0.01
        )
        assert report.alpha_name == "test"
        assert isinstance(report.t_stat, float)
        assert isinstance(report.deflated_sharpe, float)
        assert isinstance(report.pbo, float)
        assert report.verdict in ("PASS", "MARGINAL", "REJECT")


class TestMultipleTestingTracker:
    """Holm-Bonferroni correction must make significance harder, not easier."""

    def test_adjusted_p_values_larger(self):
        from validation_engine import MultipleTestingTracker
        mtt = MultipleTestingTracker()
        mtt.register_test("alpha_1", 0.01)
        mtt.register_test("alpha_2", 0.03)
        mtt.register_test("alpha_3", 0.04)

        adj = mtt.adjusted_p_values()
        # Adjusted p-values must be >= raw p-values
        for t in mtt._tests:
            assert adj[t['name']] >= t['p_value'], (
                f"{t['name']}: adjusted {adj[t['name']]} < raw {t['p_value']}"
            )

    def test_more_tests_stricter(self):
        """Adding more tests should only make thresholds stricter."""
        from validation_engine import MultipleTestingTracker
        mtt_few = MultipleTestingTracker()
        mtt_few.register_test("alpha_1", 0.04)
        adj_few = mtt_few.adjusted_p_values()

        mtt_many = MultipleTestingTracker()
        mtt_many.register_test("alpha_1", 0.04)
        for i in range(20):
            mtt_many.register_test(f"noise_{i}", 0.5 + i * 0.01)
        adj_many = mtt_many.adjusted_p_values()

        assert adj_many["alpha_1"] >= adj_few["alpha_1"], (
            "More tests should not make p-value smaller"
        )

    def test_total_tests_tracked(self):
        from validation_engine import MultipleTestingTracker
        mtt = MultipleTestingTracker()
        for i in range(10):
            mtt.register_test(f"t_{i}", 0.01 * (i + 1))
        assert mtt.total_tests == 10

    def test_summary_structure(self):
        from validation_engine import MultipleTestingTracker
        mtt = MultipleTestingTracker()
        mtt.register_test("a", 0.01)
        mtt.register_test("b", 0.10)
        s = mtt.summary()
        assert 'total_tests' in s
        assert s['total_tests'] == 2
        assert 'tests' in s
        assert len(s['tests']) == 2


# ============================================================================
# 5d) WS2: L0 Data Infrastructure (data_quality.py new classes)
# ============================================================================

class TestTradingCalendar:
    """TradingCalendar must correctly identify trading days."""

    def test_weekday_is_trading(self):
        from data_quality import TradingCalendar
        cal = TradingCalendar("NYSE")
        # 2024-01-16 is a Tuesday (not a holiday)
        assert cal.is_trading_day("2024-01-16") is True

    def test_weekend_not_trading(self):
        from data_quality import TradingCalendar
        cal = TradingCalendar("NYSE")
        # Saturday
        assert cal.is_trading_day("2024-01-13") is False
        # Sunday
        assert cal.is_trading_day("2024-01-14") is False

    def test_holiday_not_trading(self):
        from data_quality import TradingCalendar
        cal = TradingCalendar("NYSE")
        # Christmas 2024
        assert cal.is_trading_day("2024-12-25") is False
        # MLK Day 2024
        assert cal.is_trading_day("2024-01-15") is False

    def test_trading_days_between(self):
        from data_quality import TradingCalendar
        cal = TradingCalendar("NYSE")
        days = cal.trading_days_between("2024-01-15", "2024-01-19")
        # Jan 15 = MLK holiday, 16-19 = Tue-Fri (4 days)
        assert "2024-01-15" not in days, "MLK Day should be excluded"
        assert "2024-01-16" in days
        assert len(days) == 4

    def test_next_trading_day_skips_weekend(self):
        from data_quality import TradingCalendar
        cal = TradingCalendar("NYSE")
        # Friday -> next Monday (Jan 15 is MLK, so next = Jan 16)
        nxt = cal.next_trading_day("2024-01-12")  # Friday
        assert nxt == "2024-01-16", f"Expected 2024-01-16 (skip weekend + MLK), got {nxt}"

    def test_expected_bars(self):
        from data_quality import TradingCalendar
        cal = TradingCalendar("NYSE")
        n = cal.expected_bars("2024-01-02", "2024-01-05")  # Tue-Fri
        assert n == 4

    def test_validate_timestamps(self):
        from data_quality import TradingCalendar
        cal = TradingCalendar("NYSE")
        ts = pd.Series(["2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16"])
        report = cal.validate_timestamps(ts)
        assert report['weekend_bars'] == 2  # Sat + Sun
        assert report['holiday_bars'] == 1  # MLK


class TestDataIngestionGate:
    """DataIngestionGate must reject/quarantine bad data."""

    def _make_clean_df(self, n=50):
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        np.random.seed(42)
        p = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": p + 0.1,
            "close": p,
            "volume": np.random.randint(1_000_000, 5_000_000, n),
        })
        df["high"] = df[["open", "close"]].max(axis=1) + 0.5
        df["low"] = df[["open", "close"]].min(axis=1) - 0.5
        return df

    def test_clean_data_passes_through(self):
        from data_quality import DataIngestionGate
        gate = DataIngestionGate()
        df = self._make_clean_df()
        clean, report = gate.ingest(df)
        assert report['rejected'] == 0
        assert report['output_rows'] == len(df)

    def test_negative_price_rejected(self):
        from data_quality import DataIngestionGate
        gate = DataIngestionGate()
        df = self._make_clean_df()
        df.loc[5, "close"] = -10.0
        clean, report = gate.ingest(df)
        assert report['rejected'] >= 1
        assert report['output_rows'] < len(df)

    def test_missing_close_rejected(self):
        from data_quality import DataIngestionGate
        gate = DataIngestionGate()
        df = self._make_clean_df()
        df.loc[3, "close"] = np.nan
        clean, report = gate.ingest(df)
        assert report['rejected'] >= 1

    def test_ohlc_violation_quarantined(self):
        from data_quality import DataIngestionGate
        gate = DataIngestionGate()
        df = self._make_clean_df()
        # Make high < close (OHLC violation)
        df.loc[10, "high"] = df.loc[10, "close"] - 5.0
        clean, report = gate.ingest(df)
        assert report['quarantined'] >= 1
        # Quarantined rows are kept but flagged
        assert report['output_rows'] == len(df)  # No rejections, just quarantine

    def test_extreme_return_quarantined(self):
        from data_quality import DataIngestionGate
        gate = DataIngestionGate(extreme_return_threshold=0.10)
        df = self._make_clean_df()
        # Inject 50% jump
        df.loc[20, "close"] = df.loc[19, "close"] * 1.60
        df.loc[20, "high"] = df.loc[20, "close"] + 1.0
        clean, report = gate.ingest(df)
        assert report['quarantined'] >= 1

    def test_empty_df_handled(self):
        from data_quality import DataIngestionGate
        gate = DataIngestionGate()
        clean, report = gate.ingest(pd.DataFrame())
        assert len(clean) == 0
        assert report['reason'] == 'empty'

    def test_missing_volume_filled(self):
        from data_quality import DataIngestionGate
        gate = DataIngestionGate()
        df = self._make_clean_df()
        df.loc[5, "volume"] = np.nan
        df.loc[6, "volume"] = 0
        clean, report = gate.ingest(df)
        assert report['volume_filled'] >= 1

    def test_preserves_capitalized_columns(self):
        """REGRESSION: yfinance delivers capitalized columns (Close, Open, etc).
        The gate must NOT lowercase them -- downstream prepare_datasets depends
        on the original column names.  KeyError: 'Close' crash otherwise."""
        from data_quality import DataIngestionGate
        gate = DataIngestionGate()
        n = 30
        np.random.seed(42)
        p = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "Open": p + 0.1,
            "High": p + abs(np.random.randn(n) * 0.5) + 0.5,
            "Low": p - abs(np.random.randn(n) * 0.5) - 0.5,
            "Close": p,
            "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
        clean, report = gate.ingest(df)
        # Column names MUST remain capitalized
        assert "Close" in clean.columns, f"Columns lowered: {list(clean.columns)}"
        assert "Open" in clean.columns
        assert "Volume" in clean.columns
        assert "close" not in clean.columns, "Gate must not lowercase column names"


class TestPointInTimeStore:
    """PointInTimeStore must enforce point-in-time semantics."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp(prefix="pit_test_")

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_create_and_query_snapshot(self):
        from data_quality import PointInTimeStore
        pit = PointInTimeStore(self._tmpdir)

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "close": [100.0, 101.0, 102.0],
        })
        snap_id = pit.create_snapshot({"AAPL": df}, source="test", version="v001")
        assert snap_id is not None

        result = pit.query_as_of("AAPL", "2024-01-02")
        assert result is not None
        assert len(result) <= 2  # Only rows up to Jan 2

    def test_query_missing_symbol_returns_none(self):
        from data_quality import PointInTimeStore
        pit = PointInTimeStore(self._tmpdir)
        df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "close": [100]})
        pit.create_snapshot({"AAPL": df})
        result = pit.query_as_of("MISSING_SYM", "2024-01-01")
        assert result is None

    def test_diff_snapshots(self):
        from data_quality import PointInTimeStore
        import time
        pit = PointInTimeStore(self._tmpdir)
        df1 = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "close": [100]})
        df2 = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]), "close": [100, 101]})
        snap_a = pit.create_snapshot({"AAPL": df1}, version="v001")
        time.sleep(0.1)
        snap_b = pit.create_snapshot({"AAPL": df2, "MSFT": df1}, version="v002")

        diff = pit.diff_snapshots(snap_a, snap_b)
        assert "MSFT" in diff['added_symbols']
        assert "AAPL" in diff['common_symbols']


class TestUniverseManager:
    """UniverseManager must reconstruct historical universes."""

    def test_add_and_query(self):
        from data_quality import UniverseManager
        um = UniverseManager()
        um.add_event("AAPL", "2020-01-01", "add")
        um.add_event("MSFT", "2020-01-01", "add")
        um.add_event("GE", "2020-01-01", "add")

        u = um.universe_as_of("2020-06-01")
        assert "AAPL" in u
        assert "MSFT" in u
        assert "GE" in u
        assert len(u) == 3

    def test_removal_respected(self):
        from data_quality import UniverseManager
        um = UniverseManager()
        um.add_event("AAPL", "2020-01-01", "add")
        um.add_event("GE", "2020-01-01", "add")
        um.add_event("GE", "2021-06-15", "remove")

        # Before removal
        u1 = um.universe_as_of("2021-01-01")
        assert "GE" in u1

        # After removal
        u2 = um.universe_as_of("2021-07-01")
        assert "GE" not in u2
        assert "AAPL" in u2

    def test_delist_tracked(self):
        from data_quality import UniverseManager
        um = UniverseManager()
        um.add_event("LMND", "2020-01-01", "add")
        um.add_event("LMND", "2022-03-15", "delist")

        # Survivorship check
        assert um.is_survivor("LMND", "2020-01-01", "2023-01-01") is False
        assert um.is_survivor("LMND", "2020-01-01", "2021-12-31") is True

    def test_delisted_between(self):
        from data_quality import UniverseManager
        um = UniverseManager()
        um.add_event("A", "2020-01-01", "add")
        um.add_event("A", "2021-06-01", "delist")
        um.add_event("B", "2020-01-01", "add")
        um.add_event("B", "2022-03-01", "delist")

        events = um.delisted_between("2021-01-01", "2021-12-31")
        assert len(events) == 1
        assert events[0]['symbol'] == "A"

    def test_merge_removes_from_universe(self):
        from data_quality import UniverseManager
        um = UniverseManager()
        um.add_event("TWX", "2020-01-01", "add")
        um.add_event("TWX", "2021-05-15", "merge_into")
        u = um.universe_as_of("2021-06-01")
        assert "TWX" not in u


# ============================================================================
# 5e) WS3: L3 Risk Controls (alphago_architecture.py)
# ============================================================================

class TestTailRiskManager:
    """TailRiskManager CVaR must be negative (losses) and scale with vol."""

    def test_cvar_negative_for_losses(self):
        from alphago_architecture import TailRiskManager
        trm = TailRiskManager(lookback=100)
        np.random.seed(42)
        for _ in range(200):
            trm.update(np.random.randn() * 0.02 - 0.001)  # Slight negative drift
        cvar = trm.cvar_historical()
        assert cvar < 0, f"CVaR should be negative for lossy returns: {cvar}"

    def test_cvar_99_worse_than_95(self):
        """99% CVaR should be more negative (worse) than 95%."""
        from alphago_architecture import TailRiskManager
        trm = TailRiskManager(lookback=200)
        np.random.seed(42)
        for _ in range(300):
            trm.update(np.random.randn() * 0.02)
        cvar_95 = trm.cvar_historical()
        cvar_99 = trm.cvar_99()
        assert cvar_99 <= cvar_95, f"CVaR 99 ({cvar_99}) should be <= CVaR 95 ({cvar_95})"

    def test_should_constrain_when_breached(self):
        from alphago_architecture import TailRiskManager
        trm = TailRiskManager(lookback=50)
        np.random.seed(42)
        # Feed very volatile returns
        for _ in range(100):
            trm.update(np.random.randn() * 0.05)
        breached, max_safe = trm.should_constrain(
            current_exposure=1.0, cvar_limit=-0.01
        )
        # With 5% daily vol, CVaR should breach -1% limit
        assert breached is True
        assert 0 < max_safe <= 1.0

    def test_calm_market_no_constraint(self):
        from alphago_architecture import TailRiskManager
        trm = TailRiskManager(lookback=50)
        for _ in range(100):
            trm.update(0.001)  # Constant small positive return
        breached, max_safe = trm.should_constrain(
            current_exposure=0.5, cvar_limit=-0.10
        )
        # Very calm market shouldn't breach wide limit
        assert breached is False


class TestFactorExposureMonitor:
    """FactorExposureMonitor must track and limit factor exposures."""

    def test_high_beta_detected(self):
        from alphago_architecture import FactorExposureMonitor
        fem = FactorExposureMonitor(lookback=30)
        np.random.seed(42)
        for _ in range(60):
            bench = np.random.randn() * 0.01
            port = bench * 2.0 + np.random.randn() * 0.002  # Beta ~2
            fem.update(port, bench)
        within, exposures = fem.is_within_limits(limit=0.5)
        # Beta should be around 2, which exceeds 0.5 limit
        assert within is False, f"Beta ~2 should exceed limit=0.5: {exposures}"

    def test_low_beta_within_limits(self):
        from alphago_architecture import FactorExposureMonitor
        fem = FactorExposureMonitor(lookback=30)
        np.random.seed(42)
        for _ in range(60):
            bench = np.random.randn() * 0.01
            port = bench * 0.1 + np.random.randn() * 0.005  # Low beta
            fem.update(port, bench)
        within, exposures = fem.is_within_limits(limit=2.0)
        assert within is True


# ============================================================================
# 5f) WS4: Evaluation & Reporting (backtest_report.py)
# ============================================================================

class TestBacktestReport:
    """BacktestReport must enforce mandatory fields."""

    def test_validate_requires_fields(self):
        from backtest_report import BacktestReport
        report = BacktestReport()
        try:
            report.validate()
            assert False, "Should have raised assertion (no benchmark/name)"
        except AssertionError:
            pass  # Expected

    def test_populated_report_validates(self):
        from backtest_report import BacktestReport
        report = BacktestReport(
            strategy_name="Test Strategy",
            benchmark="cash",
            n_configs_tested=5,
        )
        report.validate()  # Should not raise

    def test_to_dict_roundtrip(self):
        from backtest_report import BacktestReport
        report = BacktestReport(
            strategy_name="Test",
            benchmark="SPY",
            sharpe_net=1.5,
            n_configs_tested=1,
        )
        d = report.to_dict()
        assert d['strategy_name'] == "Test"
        assert d['sharpe_net'] == 1.5


class TestComputeBacktestReport:
    """compute_backtest_report must produce correct metrics."""

    def test_positive_returns_positive_sharpe(self):
        from backtest_report import compute_backtest_report
        np.random.seed(42)
        rets = np.random.randn(252) * 0.01 + 0.001  # Positive drift
        report = compute_backtest_report(
            daily_returns_gross=rets,
            daily_returns_net=rets * 0.95,
            strategy_name="Test",
            benchmark="cash",
        )
        assert report.sharpe_gross > 0
        assert report.sharpe_net > 0
        assert report.sharpe_net <= report.sharpe_gross  # Net <= Gross

    def test_net_worse_than_gross(self):
        """Net metrics should be worse than gross (costs drag)."""
        from backtest_report import compute_backtest_report
        np.random.seed(42)
        gross = np.random.randn(252) * 0.01 + 0.001
        net = gross - 0.0005  # 5bps daily cost
        report = compute_backtest_report(
            daily_returns_gross=gross, daily_returns_net=net,
            strategy_name="Test", benchmark="cash",
        )
        assert report.cagr_net < report.cagr_gross

    def test_drawdown_non_negative(self):
        from backtest_report import compute_backtest_report
        np.random.seed(42)
        rets = np.random.randn(252) * 0.01
        report = compute_backtest_report(
            daily_returns_gross=rets, daily_returns_net=rets,
            strategy_name="Test", benchmark="cash",
        )
        assert report.max_dd_depth >= 0

    def test_cvar_negative(self):
        from backtest_report import compute_backtest_report
        np.random.seed(42)
        rets = np.random.randn(252) * 0.02  # Some losses
        report = compute_backtest_report(
            daily_returns_gross=rets, daily_returns_net=rets,
            strategy_name="Test", benchmark="cash",
        )
        assert report.cvar_95 < 0
        assert report.cvar_99 <= report.cvar_95


class TestStressTester:
    """StressTester must compute valid stress metrics."""

    def test_stress_test_returns_structure(self):
        from backtest_report import StressTester
        st = StressTester()
        np.random.seed(42)
        rets = np.random.randn(252) * 0.02
        result = st.stress_test_returns(rets, crisis_name="test_crisis")
        assert result['available'] is True
        assert 'max_dd' in result
        assert 'worst_day' in result
        assert 'cvar_95' in result
        assert result['n_days'] == 252

    def test_empty_returns(self):
        from backtest_report import StressTester
        st = StressTester()
        result = st.stress_test_returns(np.array([]))
        assert result['available'] is False

    def test_crash_scenario_high_dd(self):
        """Simulated crash should have high max DD."""
        from backtest_report import StressTester
        st = StressTester()
        # 20 days of -3% daily returns
        crash = np.full(20, -0.03)
        result = st.stress_test_returns(crash, crisis_name="sim_crash")
        assert result['max_dd'] > 0.30  # Should be > 30% DD

    def test_full_stress_report(self):
        from backtest_report import StressTester
        st = StressTester()
        np.random.seed(42)
        report = st.full_stress_report(np.random.randn(500) * 0.01)
        assert 'full_sample' in report


class TestCapacityEstimator:
    """CapacityEstimator must produce sensible capacity estimates."""

    def test_higher_alpha_higher_capacity(self):
        from backtest_report import CapacityEstimator
        ce = CapacityEstimator()
        low = ce.estimate(gross_alpha_bps=50, annual_turnover=12,
                          avg_adv_dollars=50_000_000)
        high = ce.estimate(gross_alpha_bps=200, annual_turnover=12,
                           avg_adv_dollars=50_000_000)
        assert high['capacity_at_sharpe_05'] >= low['capacity_at_sharpe_05']

    def test_zero_alpha_zero_capacity(self):
        from backtest_report import CapacityEstimator
        ce = CapacityEstimator()
        result = ce.estimate(gross_alpha_bps=0, annual_turnover=12,
                             avg_adv_dollars=50_000_000)
        assert result['capacity_at_sharpe_05'] == 0.0

    def test_cost_curve_generated(self):
        from backtest_report import CapacityEstimator
        ce = CapacityEstimator()
        result = ce.estimate(gross_alpha_bps=100, annual_turnover=12,
                             avg_adv_dollars=50_000_000)
        assert len(result['cost_curve']) > 0
        # AUM levels should be increasing
        aums = [p['aum'] for p in result['cost_curve']]
        assert all(aums[i] <= aums[i+1] for i in range(len(aums)-1))


# ============================================================================
# 5g) WS5: L2/L4 Hardening (alphago_architecture.py)
# ============================================================================

class TestAlphaCorrelationManager:
    """AlphaCorrelationManager must detect correlated alphas."""

    def test_identical_signals_n_eff_one(self):
        """Perfectly correlated signals -> N_eff ~ 1."""
        from alphago_architecture import AlphaCorrelationManager, AlphaSignal
        acm = AlphaCorrelationManager(["a", "b", "c"], window=20)
        for _ in range(30):
            val = np.random.randn() * 0.01
            signals = {
                "a": AlphaSignal(mu=val, sigma=0.01, confidence=0.5),
                "b": AlphaSignal(mu=val, sigma=0.01, confidence=0.5),
                "c": AlphaSignal(mu=val, sigma=0.01, confidence=0.5),
            }
            acm.update(signals)
        n_eff = acm.effective_n_independent()
        assert n_eff < 1.5, f"Identical signals should have N_eff ~1, got {n_eff}"

    def test_independent_signals_high_n_eff(self):
        """Independent signals -> N_eff ~ n_alphas."""
        from alphago_architecture import AlphaCorrelationManager, AlphaSignal
        np.random.seed(42)
        acm = AlphaCorrelationManager(["a", "b", "c"], window=50)
        for _ in range(60):
            signals = {
                "a": AlphaSignal(mu=np.random.randn() * 0.01, sigma=0.01, confidence=0.5),
                "b": AlphaSignal(mu=np.random.randn() * 0.01, sigma=0.01, confidence=0.5),
                "c": AlphaSignal(mu=np.random.randn() * 0.01, sigma=0.01, confidence=0.5),
            }
            acm.update(signals)
        n_eff = acm.effective_n_independent()
        assert n_eff > 2.0, f"Independent signals should have N_eff ~3, got {n_eff}"

    def test_correlation_penalty_reduces_correlated(self):
        from alphago_architecture import AlphaCorrelationManager, AlphaSignal
        np.random.seed(42)
        acm = AlphaCorrelationManager(["a", "b"], window=30)
        for _ in range(40):
            val = np.random.randn() * 0.01
            signals = {
                "a": AlphaSignal(mu=val, sigma=0.01, confidence=0.5),
                "b": AlphaSignal(mu=val + np.random.randn() * 0.0001, sigma=0.01, confidence=0.5),
            }
            acm.update(signals)
        penalties = acm.correlation_penalty_weights(max_corr=0.7)
        # Both signals are highly correlated, penalties should reduce them
        for v in penalties.values():
            assert 0 < v <= 1.0


class TestAlertManager:
    """AlertManager must correctly categorize and log alerts."""

    def test_alert_logged(self):
        from alphago_architecture import AlertManager
        am = AlertManager()
        am.alert("INFO", "test", "Hello")
        assert am.counts["INFO"] == 1

    def test_all_levels(self):
        from alphago_architecture import AlertManager
        am = AlertManager()
        for level in ("INFO", "WARNING", "CRITICAL", "FATAL"):
            am.alert(level, "test", f"msg_{level}")
        assert am.counts["INFO"] == 1
        assert am.counts["CRITICAL"] == 1
        assert len(am.log) == 4

    def test_context_stored(self):
        from alphago_architecture import AlertManager
        am = AlertManager()
        am.alert("WARNING", "L4", "Vol breach", context={"vol": 0.35})
        assert am.log[-1]['context']['vol'] == 0.35

    def test_recent_alerts_filtered(self):
        from alphago_architecture import AlertManager
        am = AlertManager()
        am.alert("INFO", "src", "msg1")
        am.alert("CRITICAL", "src", "msg2")
        am.alert("INFO", "src", "msg3")
        critical = [e for e in am.log if e['level'] == "CRITICAL"]
        assert len(critical) == 1
        assert critical[0]['level'] == "CRITICAL"


# ============================================================================
# 5h) WS6: Reproducibility (validation_engine.py)
# ============================================================================

class TestReproducibilityManager:
    """ReproducibilityManager must set seeds and record fingerprint."""

    def test_seed_produces_same_results(self):
        from validation_engine import ReproducibilityManager
        ReproducibilityManager.set_global_seed(123)
        a = np.random.randn(10).tolist()
        ReproducibilityManager.set_global_seed(123)
        b = np.random.randn(10).tolist()
        assert a == b, "Same seed must produce same random sequence"

    def test_different_seeds_differ(self):
        from validation_engine import ReproducibilityManager
        ReproducibilityManager.set_global_seed(1)
        a = np.random.randn(10).tolist()
        ReproducibilityManager.set_global_seed(2)
        b = np.random.randn(10).tolist()
        assert a != b

    def test_fingerprint_has_required_fields(self):
        from validation_engine import ReproducibilityManager
        fp = ReproducibilityManager.environment_fingerprint(seed=42)
        assert 'python' in fp
        assert 'numpy' in fp
        assert 'seed' in fp
        assert fp['seed'] == 42
        assert 'platform' in fp


class TestVersionedConfig:
    """VersionedConfig must hash and round-trip configs."""

    def test_hash_deterministic(self):
        from validation_engine import VersionedConfig
        cfg = {"target_vol": 0.15, "max_leverage": 2.0}
        vc1 = VersionedConfig(cfg)
        vc2 = VersionedConfig(cfg)
        assert vc1.config_hash() == vc2.config_hash()

    def test_different_config_different_hash(self):
        from validation_engine import VersionedConfig
        vc1 = VersionedConfig({"target_vol": 0.15})
        vc2 = VersionedConfig({"target_vol": 0.20})
        assert vc1.config_hash() != vc2.config_hash()

    def test_json_roundtrip(self):
        from validation_engine import VersionedConfig
        cfg = {"target_vol": 0.15, "max_leverage": 2.0, "kelly": 0.25}
        vc = VersionedConfig(cfg, version="v7.0")
        original_hash = vc.config_hash()

        tmpf = os.path.join(tempfile.mkdtemp(), "cfg.json")
        vc.to_json(tmpf)
        assert os.path.exists(tmpf)

        vc2 = VersionedConfig.from_json(tmpf)
        assert vc2.version == "v7.0"
        # The loaded config is a dict wrapper, hash should match
        assert vc2.config_hash() == original_hash

        # Cleanup
        shutil.rmtree(os.path.dirname(tmpf), ignore_errors=True)


# ============================================================================
# Run directly
# ============================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])