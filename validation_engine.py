"""
validation_engine.py -- Anti-Overfitting Framework & Reproducibility
================================================================================

Implements Workstream 1 (Anti-Overfitting) and Workstream 6 (Reproducibility)
from the v7.0 roadmap.

Components:
    PurgedWalkForwardCV      -- WS1A: Walk-forward CV with purge + embargo
    CombinatorialPurgedCV    -- WS1A+: CPCV for research evaluation + PBO
    SignificanceGate          -- WS1B: Statistical significance gates
    MultipleTestingTracker    -- WS1C: Holm-Bonferroni multiple testing correction
    ReproducibilityManager    -- WS6B: Global seed management + fingerprinting
    VersionedConfig           -- WS6C: Config hash-based versioning
    RunArtifactBundle         -- WS6D: Immutable run audit trail

References:
    - Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
    - Bailey et al (2017) "Probability of Backtest Overfitting"
    - Lopez de Prado (2018) "Advances in Financial Machine Learning" Ch. 7-12

Usage:
    from validation_engine import (
        PurgedWalkForwardCV, CombinatorialPurgedCV,
        SignificanceGate, MultipleTestingTracker,
        ReproducibilityManager, VersionedConfig, RunArtifactBundle,
    )
"""

from __future__ import annotations
import sys
import os
import math
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================================
# WS1A: Purged Walk-Forward Cross-Validation Engine
# ============================================================================

@dataclass
class CVFold:
    """
    One fold in a purged walk-forward cross-validation.

    Timeline:  [train_start ... train_end] | PURGE | EMBARGO | [test_start ... test_end]

    The purge gap removes samples between train and test that could leak
    information through overlapping labels or autocorrelation.
    The embargo period adds additional buffer after the purge.
    """
    fold_id: int
    train_start: int
    train_end: int
    purge_end: int        # train_end + purge_gap
    embargo_end: int      # purge_end + embargo
    test_start: int       # embargo_end
    test_end: int


class PurgedWalkForwardCV:
    """
    Walk-forward CV with purge gap and embargo period.

    Args:
        n_obs:        Total number of observations
        purge_gap:    >= max(alpha_horizon, holding_period) bars
        embargo:      Additional buffer for autocorrelation bleed
        n_splits:     Number of expanding-window folds
        holdout_pct:  Final holdout fraction (touched once)

    The holdout region (last holdout_pct of data) is NEVER used in
    any fold -- it is reserved for the final, one-time-only evaluation.

    Reference: Lopez de Prado (2018) Ch. 7 "Cross-Validation in Finance"
    """

    def __init__(self, n_obs: int, purge_gap: int = 21,
                 embargo: int = 5, n_splits: int = 5,
                 holdout_pct: float = 0.20):
        assert n_obs > 0, "n_obs must be positive"
        assert 0.0 < holdout_pct < 1.0, "holdout_pct must be in (0, 1)"
        assert n_splits >= 2, "Need at least 2 splits"

        self.n_obs = n_obs
        self.purge_gap = purge_gap
        self.embargo = embargo
        self.n_splits = n_splits
        self.holdout_pct = holdout_pct
        self.holdout_start = int(n_obs * (1 - holdout_pct))

    def generate_folds(self) -> List[CVFold]:
        """
        Generate expanding-window folds with purge and embargo.

        Each fold:
          - Train: [0 ... fold_train_end]  (expanding)
          - Purge: (fold_train_end ... fold_train_end + purge_gap]
          - Embargo: (purge_end ... purge_end + embargo]
          - Test:  (embargo_end ... next_fold_start]

        All within [0, holdout_start).
        """
        usable = self.holdout_start
        min_train = max(60, self.purge_gap * 3)  # Minimum training size

        # Divide the usable region into n_splits test windows
        test_size = max(1, (usable - min_train) // self.n_splits)

        folds = []
        for i in range(self.n_splits):
            train_end = min_train + i * test_size
            purge_end = train_end + self.purge_gap
            embargo_end = purge_end + self.embargo
            test_start = embargo_end
            test_end = min(train_end + (i + 1) * test_size, usable)

            if test_start >= test_end:
                continue  # Skip degenerate fold

            if test_end > usable:
                test_end = usable

            fold = CVFold(
                fold_id=i,
                train_start=0,
                train_end=train_end,
                purge_end=purge_end,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end,
            )
            if self.validate_no_leakage(fold):
                folds.append(fold)

        return folds

    def validate_no_leakage(self, fold: CVFold) -> bool:
        """Assert: no overlap between train and test after purging."""
        return fold.test_start > fold.embargo_end - 1 and fold.test_start > fold.train_end

    def get_holdout_indices(self) -> Tuple[int, int]:
        """Return (start, end) of the final holdout period."""
        return self.holdout_start, self.n_obs

    def summary(self) -> Dict[str, Any]:
        folds = self.generate_folds()
        return {
            'n_obs': self.n_obs,
            'n_folds': len(folds),
            'purge_gap': self.purge_gap,
            'embargo': self.embargo,
            'holdout_start': self.holdout_start,
            'holdout_size': self.n_obs - self.holdout_start,
            'folds': [asdict(f) for f in folds],
        }


class HoldoutGuard:
    """
    Enforces single-use holdout evaluation.

    The holdout set must be touched EXACTLY ONCE. This guard tracks
    whether the holdout has been consumed and raises on repeat access.

    Usage:
        guard = HoldoutGuard(holdout_start=800, holdout_end=1000)

        # During walk-forward CV:
        guard.validate_no_holdout_overlap(train_start=0, train_end=600,
                                          test_start=650, test_end=750)

        # Final one-time evaluation:
        holdout_data = guard.consume_holdout()  # Returns (800, 1000)
        # ... evaluate ...
        guard.record_holdout_result(sharpe=1.2, n_obs=200)

        # Attempting again raises:
        guard.consume_holdout()  # -> HoldoutViolationError
    """

    def __init__(self, holdout_start: int, holdout_end: int):
        assert holdout_start < holdout_end, "Invalid holdout range"
        self.holdout_start = holdout_start
        self.holdout_end = holdout_end
        self._consumed = False
        self._result: Optional[Dict] = None
        self._access_log: List[Dict] = []

    def validate_no_holdout_overlap(self, train_start: int, train_end: int,
                                     test_start: int, test_end: int) -> bool:
        """Verify a CV fold does not overlap with the holdout region."""
        if test_end > self.holdout_start or train_end > self.holdout_start:
            raise HoldoutViolationError(
                f"CV fold [{train_start}:{train_end}] / [{test_start}:{test_end}] "
                f"overlaps holdout [{self.holdout_start}:{self.holdout_end}]."
            )
        return True

    def consume_holdout(self) -> Tuple[int, int]:
        """
        Return holdout indices for one-time evaluation.
        Raises HoldoutViolationError on repeat access.
        """
        if self._consumed:
            raise HoldoutViolationError(
                "Holdout already consumed! The holdout set must be touched "
                "EXACTLY ONCE. This is a critical anti-overfitting safeguard. "
                f"Previous result: {self._result}"
            )
        self._consumed = True
        self._access_log.append({
            'action': 'consumed',
            'timestamp': datetime.utcnow().isoformat(),
        })
        return self.holdout_start, self.holdout_end

    def record_holdout_result(self, **metrics):
        """Store the holdout evaluation result for audit trail."""
        self._result = {
            'holdout_range': (self.holdout_start, self.holdout_end),
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self._access_log.append({
            'action': 'result_recorded',
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat(),
        })

    @property
    def is_consumed(self) -> bool:
        return self._consumed

    @property
    def result(self) -> Optional[Dict]:
        return self._result

    @property
    def audit_log(self) -> List[Dict]:
        return list(self._access_log)


class HoldoutViolationError(Exception):
    """Raised when holdout set is accessed more than once or overlapped."""
    pass


# ============================================================================
# WS1B: Statistical Significance Gates
# ============================================================================

@dataclass
class AlphaSignificanceReport:
    """Result of statistical significance evaluation for one alpha."""
    alpha_name: str
    t_stat: float                   # Must be > 3.0
    sharpe_is: float                # In-sample Sharpe
    sharpe_oos: float               # Out-of-sample Sharpe
    deflated_sharpe: float          # Bailey & Lopez de Prado
    oos_is_decay_ratio: float       # OOS Sharpe / IS Sharpe (flag if < 0.5)
    pbo: float                      # Probability of Backtest Overfitting
    n_configs_tested: int           # For multiple testing correction
    adjusted_p_value: float         # After Holm-Bonferroni correction
    n_obs_is: int
    n_obs_oos: int
    verdict: str                    # "PASS" / "MARGINAL" / "REJECT"


class SignificanceGate:
    """
    Gate keeper: no alpha reaches L2 without passing significance tests.

    Rules:
      - t_stat > 3.0 (accounts for multiple testing)
      - OOS/IS decay < 0.5 -> flag as likely overfit
      - deflated_sharpe computed per Bailey & Lopez de Prado (2014)
      - All configs tested must be reported alongside best result

    Reference: Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
    """
    MIN_T_STAT = 3.0
    MAX_OOS_DECAY = 0.50

    def evaluate(self, alpha_name: str,
                 is_returns: np.ndarray,
                 oos_returns: np.ndarray,
                 n_configs_tested: int = 1) -> AlphaSignificanceReport:
        """
        Evaluate an alpha's statistical significance.

        Args:
            alpha_name:       Identifier
            is_returns:       In-sample returns array (daily)
            oos_returns:      Out-of-sample returns array (daily)
            n_configs_tested: Total hyperparameter configurations tested
        """
        is_returns = np.asarray(is_returns, dtype=np.float64)
        oos_returns = np.asarray(oos_returns, dtype=np.float64)

        # IS metrics
        is_mu = float(np.mean(is_returns)) * 252
        is_std = float(np.std(is_returns)) * np.sqrt(252)
        sharpe_is = is_mu / (is_std + 1e-10)
        n_is = len(is_returns)

        # OOS metrics
        oos_mu = float(np.mean(oos_returns)) * 252
        oos_std = float(np.std(oos_returns)) * np.sqrt(252)
        sharpe_oos = oos_mu / (oos_std + 1e-10)
        n_oos = len(oos_returns)

        # t-statistic (annualized Sharpe * sqrt(years))
        n_years_oos = max(n_oos / 252, 0.1)
        t_stat = sharpe_oos * math.sqrt(n_years_oos)

        # OOS/IS decay ratio
        decay_ratio = sharpe_oos / (sharpe_is + 1e-10) if abs(sharpe_is) > 1e-10 else 0.0

        # Deflated Sharpe Ratio
        skew = float(self._skewness(oos_returns))
        kurt = float(self._kurtosis(oos_returns))
        dsr = self.deflated_sharpe(
            sharpe_oos, n_oos, skew, kurt, max(n_configs_tested, 1)
        )

        # PBO estimate (simplified: fraction of OOS folds with negative Sharpe)
        pbo = self._estimate_pbo(is_returns, oos_returns)

        # p-value from t-stat
        from scipy.stats import norm
        raw_p = 1.0 - norm.cdf(abs(t_stat))
        # Adjust for multiple testing (simple Bonferroni as baseline)
        adjusted_p = min(1.0, raw_p * max(n_configs_tested, 1))

        # Verdict
        if t_stat > self.MIN_T_STAT and decay_ratio >= self.MAX_OOS_DECAY and dsr > 0.5:
            verdict = "PASS"
        elif t_stat > 2.0 and decay_ratio >= 0.3:
            verdict = "MARGINAL"
        else:
            verdict = "REJECT"

        return AlphaSignificanceReport(
            alpha_name=alpha_name,
            t_stat=t_stat,
            sharpe_is=sharpe_is,
            sharpe_oos=sharpe_oos,
            deflated_sharpe=dsr,
            oos_is_decay_ratio=decay_ratio,
            pbo=pbo,
            n_configs_tested=n_configs_tested,
            adjusted_p_value=adjusted_p,
            n_obs_is=n_is,
            n_obs_oos=n_oos,
            verdict=verdict,
        )

    def deflated_sharpe(self, sharpe: float, n_obs: int,
                        skew: float, kurtosis: float,
                        n_trials: int) -> float:
        """
        Bailey & Lopez de Prado (2014) deflated Sharpe ratio.
        Adjusts for multiple testing, non-normality, and short samples.
        """
        if n_obs < 10 or n_trials < 1:
            return 0.0

        from scipy.stats import norm

        # Standard error of Sharpe ratio under non-normality
        sr_var = (1 + 0.5 * sharpe**2 - skew * sharpe
                  + (kurtosis - 3) / 4.0 * sharpe**2) / max(n_obs - 1, 1)
        sr_std = math.sqrt(max(sr_var, 1e-12))

        # Expected max Sharpe under null (Euler-Mascheroni constant)
        euler_gamma = 0.5772156649
        if n_trials > 1:
            z1 = norm.ppf(1.0 - 1.0 / n_trials)
            z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
            e_max_sr = sr_std * ((1 - euler_gamma) * z1 + euler_gamma * z2)
        else:
            e_max_sr = 0.0

        # Probability that observed Sharpe exceeds the expected max
        dsr = float(norm.cdf((sharpe - e_max_sr) / (sr_std + 1e-10)))
        return dsr

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        n = len(x)
        if n < 3:
            return 0.0
        mu = np.mean(x)
        s = np.std(x, ddof=1)
        if s < 1e-12:
            return 0.0
        return float(np.mean(((x - mu) / s) ** 3))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        n = len(x)
        if n < 4:
            return 3.0
        mu = np.mean(x)
        s = np.std(x, ddof=1)
        if s < 1e-12:
            return 3.0
        return float(np.mean(((x - mu) / s) ** 4))

    @staticmethod
    def _estimate_pbo(is_rets: np.ndarray, oos_rets: np.ndarray,
                      n_subsplits: int = 5) -> float:
        """
        Simplified PBO estimate using subsample analysis.
        PBO ~= fraction of subsample trials where OOS Sharpe is negative.
        """
        n = len(oos_rets)
        if n < n_subsplits * 10:
            return 0.5  # Insufficient data

        chunk = n // n_subsplits
        negative_count = 0
        for i in range(n_subsplits):
            sub = oos_rets[i * chunk:(i + 1) * chunk]
            if len(sub) > 5:
                sub_sharpe = np.mean(sub) / (np.std(sub) + 1e-10) * np.sqrt(252)
                if sub_sharpe < 0:
                    negative_count += 1

        return negative_count / n_subsplits


# ============================================================================
# WS1C: Multiple Testing Correction
# ============================================================================

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Instead of sequential walk-forward splits, CPCV generates ALL possible
    N-choose-K train/test partitions of time-ordered groups, with purge
    and embargo between each train and test segment.

    This provides more statistically efficient estimates of OOS performance
    and enables direct computation of the Probability of Backtest Overfitting
    (PBO) by measuring the fraction of OOS paths with negative Sharpe.

    Reference: Bailey, Borwein, Lopez de Prado & Zhu (2017)
               "The Probability of Backtest Overfitting"

    Args:
        n_obs:      Total number of observations
        n_groups:   Number of equal-sized time groups to divide data into
        n_test:     Number of groups to hold out as test in each combination
        purge_gap:  Bars to purge between train and test boundaries
        embargo:    Additional embargo bars after purge

    Usage:
        cpcv = CombinatorialPurgedCV(n_obs=1000, n_groups=6, n_test=2, purge_gap=21)
        for fold in cpcv.generate_folds():
            train_idx, test_idx = fold['train_indices'], fold['test_indices']
            # fit and evaluate
        pbo = cpcv.estimate_pbo(oos_sharpes)
    """

    def __init__(self, n_obs: int, n_groups: int = 6, n_test: int = 2,
                 purge_gap: int = 21, embargo: int = 5):
        assert n_obs > 0
        assert 2 <= n_test < n_groups
        self.n_obs = n_obs
        self.n_groups = n_groups
        self.n_test = n_test
        self.purge_gap = purge_gap
        self.embargo = embargo

        # Divide observations into groups
        self.group_size = n_obs // n_groups
        self.group_boundaries = [
            (i * self.group_size, min((i + 1) * self.group_size, n_obs))
            for i in range(n_groups)
        ]

    def generate_folds(self) -> List[Dict[str, Any]]:
        """
        Generate all C(n_groups, n_test) train/test partitions.

        Each fold specifies which groups are test and which are train,
        with purge+embargo applied at every boundary between train and
        test segments.
        """
        from itertools import combinations

        folds = []
        for fold_id, test_groups in enumerate(
            combinations(range(self.n_groups), self.n_test)
        ):
            test_set = set(test_groups)
            train_groups = [g for g in range(self.n_groups) if g not in test_set]

            # Build raw index sets
            train_indices = set()
            test_indices = set()

            for g in train_groups:
                start, end = self.group_boundaries[g]
                train_indices.update(range(start, end))

            for g in test_groups:
                start, end = self.group_boundaries[g]
                test_indices.update(range(start, end))

            # Apply purge + embargo at train/test boundaries
            purge_indices = set()
            for tg in test_groups:
                t_start, t_end = self.group_boundaries[tg]
                # Purge before test segment
                purge_start = max(0, t_start - self.purge_gap - self.embargo)
                purge_end = t_start
                purge_indices.update(range(purge_start, purge_end))
                # Embargo after test segment
                emb_start = t_end
                emb_end = min(self.n_obs, t_end + self.embargo)
                purge_indices.update(range(emb_start, emb_end))

            train_indices -= purge_indices
            train_final = sorted(train_indices)
            test_final = sorted(test_indices)

            if len(train_final) > 0 and len(test_final) > 0:
                folds.append({
                    'fold_id': fold_id,
                    'test_groups': list(test_groups),
                    'train_groups': train_groups,
                    'train_indices': train_final,
                    'test_indices': test_final,
                    'n_train': len(train_final),
                    'n_test': len(test_final),
                    'n_purged': len(purge_indices & train_indices),
                })

        return folds

    def n_combinations(self) -> int:
        """Total number of fold combinations."""
        from math import comb
        return comb(self.n_groups, self.n_test)

    def estimate_pbo(self, oos_sharpes: List[float]) -> float:
        """
        Probability of Backtest Overfitting.

        PBO = fraction of OOS paths where Sharpe is <= 0.
        If PBO > 0.5, the strategy is more likely overfit than not.
        """
        if not oos_sharpes:
            return 1.0
        n_negative = sum(1 for s in oos_sharpes if s <= 0)
        return n_negative / len(oos_sharpes)

    def summary(self) -> Dict[str, Any]:
        folds = self.generate_folds()
        return {
            'n_obs': self.n_obs,
            'n_groups': self.n_groups,
            'n_test': self.n_test,
            'n_combinations': self.n_combinations(),
            'n_valid_folds': len(folds),
            'purge_gap': self.purge_gap,
            'embargo': self.embargo,
            'group_size': self.group_size,
        }


class MultipleTestingTracker:
    """
    Global registry of all hypotheses tested.

    Tracks every alpha, hyperparameter config, and feature combination tested.
    Reports the full count alongside any performance metric.
    Applies Holm-Bonferroni correction for family-wise error rate control.

    Reference: Holm (1979) "A simple sequentially rejective multiple test procedure"
    """

    def __init__(self, method: str = "holm_bonferroni"):
        self._tests: List[Dict] = []
        self.method = method

    def register_test(self, name: str, p_value: float,
                      context: Dict = None):
        """Register one hypothesis test."""
        self._tests.append({
            'name': name,
            'p_value': float(p_value),
            'context': context or {},
            'timestamp': datetime.utcnow().isoformat(),
        })

    def adjusted_p_values(self) -> Dict[str, float]:
        """Apply Holm-Bonferroni correction to all registered tests."""
        if not self._tests:
            return {}

        sorted_tests = sorted(self._tests, key=lambda t: t['p_value'])
        m = len(sorted_tests)
        adjusted = {}
        cumulative_max = 0.0
        for i, test in enumerate(sorted_tests):
            # Holm step-down: p_adj = p * (m - i)
            adj = min(1.0, test['p_value'] * (m - i))
            # Enforce monotonicity (adjusted p must be non-decreasing)
            cumulative_max = max(cumulative_max, adj)
            adjusted[test['name']] = cumulative_max

        return adjusted

    @property
    def total_tests(self) -> int:
        return len(self._tests)

    def summary(self) -> Dict[str, Any]:
        adjusted = self.adjusted_p_values()
        n_sig_raw = sum(1 for t in self._tests if t['p_value'] < 0.05)
        n_sig_adj = sum(1 for v in adjusted.values() if v < 0.05)
        return {
            'total_tests': len(self._tests),
            'method': self.method,
            'significant_at_005_raw': n_sig_raw,
            'significant_at_005_adjusted': n_sig_adj,
            'tests': [
                {
                    'name': t['name'],
                    'p_raw': t['p_value'],
                    'p_adjusted': adjusted.get(t['name'], t['p_value']),
                }
                for t in self._tests
            ],
        }


# ============================================================================
# WS6B: Global Seed Management & Reproducibility
# ============================================================================

class ReproducibilityManager:
    """
    Pin all random seeds and record environment for reproducibility.

    Usage:
        ReproducibilityManager.set_global_seed(42)
        fingerprint = ReproducibilityManager.environment_fingerprint()
    """

    @staticmethod
    def set_global_seed(seed: int = 42):
        """Set all random seeds for deterministic execution."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def environment_fingerprint(seed: int = 42) -> Dict[str, Any]:
        """Record library versions, system info, and code commit for reproducibility."""
        fp = {
            'python': sys.version,
            'numpy': np.__version__,
            'seed': seed,
            'platform': platform.platform(),
            'timestamp': datetime.utcnow().isoformat(),
        }
        if HAS_TORCH:
            fp['torch'] = torch.__version__
            fp['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                fp['cuda_device'] = torch.cuda.get_device_name(0)
        if HAS_PANDAS:
            fp['pandas'] = pd.__version__

        # WS6 fix: Capture git commit hash for code traceability.
        # Gracefully degrades if not in a git repo or git not installed.
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode('ascii').strip()
            fp['git_commit'] = git_hash
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired, OSError):
            fp['git_commit'] = 'unavailable'

        try:
            git_dirty = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode('ascii').strip()
            fp['git_dirty'] = len(git_dirty) > 0
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired, OSError):
            fp['git_dirty'] = None

        return fp


# ============================================================================
# WS6C: Config Versioning
# ============================================================================

class VersionedConfig:
    """
    Wraps any config dataclass with hash-based versioning.

    The config_hash deterministically fingerprints all parameter values,
    allowing exact reproducibility checks: if two runs have different
    hashes, they used different configs.

    Usage:
        vc = VersionedConfig(acfg)
        h = vc.config_hash()
        vc.to_json("configs/run_001.json")
        vc2 = VersionedConfig.from_json("configs/run_001.json")
        assert vc2.config_hash() == h
    """

    def __init__(self, config, version: str = "v7.0"):
        self.config = config
        self.version = version

    def config_hash(self) -> str:
        """Deterministic hash of all parameter values."""
        if hasattr(self.config, '__dataclass_fields__'):
            content = json.dumps(asdict(self.config), sort_keys=True, default=str)
        elif isinstance(self.config, dict):
            content = json.dumps(self.config, sort_keys=True, default=str)
        else:
            content = str(vars(self.config))
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_json(self, path: str):
        """Save config to versioned JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        payload = {
            'version': self.version,
            'config_hash': self.config_hash(),
            'timestamp': datetime.utcnow().isoformat(),
        }
        if hasattr(self.config, '__dataclass_fields__'):
            payload['config'] = {k: _safe_val(v) for k, v in asdict(self.config).items()}
        elif isinstance(self.config, dict):
            payload['config'] = {k: _safe_val(v) for k, v in self.config.items()}
        else:
            payload['config'] = {k: _safe_val(v) for k, v in vars(self.config).items()}
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> 'VersionedConfig':
        """Load config from JSON. Returns VersionedConfig wrapping a dict."""
        with open(path, 'r') as f:
            payload = json.load(f)
        vc = cls(config=payload.get('config', {}),
                 version=payload.get('version', 'unknown'))
        return vc

    def to_yaml(self, path: str):
        """Save config to versioned YAML file (uses JSON-subset for portability)."""
        # YAML-like format without requiring pyyaml dependency
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        lines = [
            f"# VersionedConfig v{self.version}",
            f"# Hash: {self.config_hash()}",
            f"# Generated: {datetime.utcnow().isoformat()}",
            "",
        ]
        if hasattr(self.config, '__dataclass_fields__'):
            cfg_dict = {k: _safe_val(v) for k, v in asdict(self.config).items()}
        elif isinstance(self.config, dict):
            cfg_dict = self.config
        else:
            cfg_dict = {k: _safe_val(v) for k, v in vars(self.config).items()}

        for k, v in sorted(cfg_dict.items()):
            lines.append(f"{k}: {json.dumps(v, default=str)}")

        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    @staticmethod
    def generate_requirements_txt(path: str = "requirements.txt"):
        """
        Generate a pinned requirements.txt from the current environment.
        Captures exact library versions used in the run.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        lines = [
            "# Auto-generated pinned requirements for reproducibility",
            f"# Generated: {datetime.utcnow().isoformat()}",
            f"numpy=={np.__version__}",
        ]
        try:
            import torch
            lines.append(f"torch=={torch.__version__}")
        except ImportError:
            pass
        try:
            import pandas
            lines.append(f"pandas=={pandas.__version__}")
        except ImportError:
            pass
        try:
            import scipy
            lines.append(f"scipy=={scipy.__version__}")
        except ImportError:
            pass

        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return path


def _safe_val(v):
    """Convert non-JSON-serializable values."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ============================================================================
# WS6D: Immutable Run Artifact Bundle
# ============================================================================

class RunArtifactBundle:
    """
    Captures everything needed to reconstruct a run for audit.

    Bundles: config hash, data snapshot ID, feature schema hash,
    model weights fingerprint, ensemble weights, optimizer solution,
    and key decisions. Saved as immutable JSON.

    Every run produces exactly one artifact bundle. If two bundles have
    the same content hash, the runs were identical.

    Usage:
        bundle = RunArtifactBundle(run_id="run_001")
        bundle.set_config(acfg)
        bundle.set_data_snapshot("snap_20240115_v001")
        bundle.set_model_fingerprint(model)
        bundle.set_ensemble_weights({'trend': 0.3, 'mr': 0.4, 'rl': 0.3})
        bundle.add_decision(bar=100, action='buy', reason='mu=0.02')
        bundle.save("artifacts/run_001.json")
    """

    def __init__(self, run_id: str = ""):
        self.run_id = run_id or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        self._config_hash: str = ""
        self._data_snapshot_id: str = ""
        self._feature_schema_hash: str = ""
        self._model_fingerprint: str = ""
        self._ensemble_weights: Dict[str, float] = {}
        self._optimizer_solution: Dict[str, Any] = {}
        self._decisions: List[Dict] = []
        self._environment: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        self._timestamp = datetime.utcnow().isoformat()

    def set_config(self, config):
        """Record config hash."""
        vc = VersionedConfig(config)
        self._config_hash = vc.config_hash()

    def set_data_snapshot(self, snapshot_id: str):
        self._data_snapshot_id = snapshot_id

    def set_feature_schema(self, feature_names: List[str]):
        """Hash the feature schema for drift detection."""
        content = json.dumps(sorted(feature_names))
        self._feature_schema_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

    def set_model_fingerprint(self, model_weights=None):
        """Hash model weights for reproducibility verification."""
        if model_weights is not None:
            if isinstance(model_weights, np.ndarray):
                content = model_weights.tobytes()
            else:
                content = str(model_weights).encode()
            self._model_fingerprint = hashlib.sha256(content).hexdigest()[:12]

    def set_ensemble_weights(self, weights: Dict[str, float]):
        self._ensemble_weights = dict(weights)

    def set_environment(self, env_fingerprint: Dict[str, Any]):
        self._environment = dict(env_fingerprint)

    def set_metrics(self, metrics: Dict[str, Any]):
        self._metrics = {k: _safe_val(v) for k, v in metrics.items()}

    def add_decision(self, **kwargs):
        """Record a key decision for audit trail."""
        self._decisions.append({k: _safe_val(v) for k, v in kwargs.items()})

    def content_hash(self) -> str:
        """Deterministic hash of entire bundle content."""
        payload = {
            'config_hash': self._config_hash,
            'data_snapshot': self._data_snapshot_id,
            'feature_schema': self._feature_schema_hash,
            'model_fingerprint': self._model_fingerprint,
            'ensemble_weights': self._ensemble_weights,
        }
        content = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save(self, path: str):
        """Save immutable artifact bundle."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        payload = {
            'run_id': self.run_id,
            'timestamp': self._timestamp,
            'content_hash': self.content_hash(),
            'config_hash': self._config_hash,
            'data_snapshot_id': self._data_snapshot_id,
            'feature_schema_hash': self._feature_schema_hash,
            'model_fingerprint': self._model_fingerprint,
            'ensemble_weights': self._ensemble_weights,
            'optimizer_solution': self._optimizer_solution,
            'environment': self._environment,
            'metrics': self._metrics,
            'n_decisions': len(self._decisions),
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> 'RunArtifactBundle':
        """Load an artifact bundle."""
        with open(path, 'r') as f:
            data = json.load(f)
        bundle = cls(run_id=data.get('run_id', ''))
        bundle._config_hash = data.get('config_hash', '')
        bundle._data_snapshot_id = data.get('data_snapshot_id', '')
        bundle._feature_schema_hash = data.get('feature_schema_hash', '')
        bundle._model_fingerprint = data.get('model_fingerprint', '')
        bundle._ensemble_weights = data.get('ensemble_weights', {})
        bundle._environment = data.get('environment', {})
        bundle._metrics = data.get('metrics', {})
        return bundle

    def summary(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'content_hash': self.content_hash(),
            'config_hash': self._config_hash,
            'data_snapshot': self._data_snapshot_id,
            'n_decisions': len(self._decisions),
        }