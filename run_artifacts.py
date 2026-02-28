"""
run_artifacts.py -- Solo-grade run logging: manifest + JSONL events + trades.csv

Provides reproducibility and forensic audit capability without requiring
a full OMS or database.  Drop-in for any pipeline run.

Usage:
    writer = RunWriter(out_dir="runs/2026-02-13_001")
    writer.write_manifest(config_dict)
    writer.log_step({"_type": "l2_ensemble", "mu_hat": 0.01, ...})
    writer.log_trade({"t": 42, "side": "buy", "trade_notional": 5000, ...})
    writer.close()
"""

from __future__ import annotations
import csv
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


TRADE_FIELDS = [
    "t", "side", "trade_notional", "fill", "mid",
    "slippage_bps", "fees", "impact", "half_spread",
]


class RunWriter:
    """Append-mode JSONL event log + CSV trade log + manifest."""

    def __init__(
        self,
        out_dir: str = "run_output",
        run_id: Optional[str] = None,
        flush_every: int = 200,
        trade_fields: Optional[list] = None,
    ):
        self.out_dir = out_dir
        self.run_id = run_id or f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self._flush_every = flush_every
        os.makedirs(out_dir, exist_ok=True)

        # Manifest
        self.manifest_path = os.path.join(out_dir, "manifest.json")

        # JSONL event stream (kept open for speed)
        self.steps_path = os.path.join(out_dir, "events.jsonl")
        self._fh = open(self.steps_path, "a", encoding="utf-8")
        self._n = 0

        # Trades CSV
        self.trades_path = os.path.join(out_dir, "trades.csv")
        _fields = trade_fields or TRADE_FIELDS
        self._trades_fh = open(self.trades_path, "w", newline="", encoding="utf-8")
        self._trades_writer = csv.DictWriter(self._trades_fh, fieldnames=_fields)
        self._trades_writer.writeheader()

    # -- Manifest --
    def write_manifest(self, config: Dict[str, Any]) -> None:
        """Write run config / metadata as a JSON manifest."""
        manifest = {
            "run_id": self.run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": _safe_serialize(config),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

    # -- JSONL events --
    def log_step(self, event: Dict[str, Any]) -> None:
        """Append one event (dict) to the JSONL log."""
        event = dict(event)
        event["run_id"] = self.run_id
        self._fh.write(json.dumps(event, default=str) + "\n")
        self._n += 1
        if self._n % self._flush_every == 0:
            self._fh.flush()

    # -- Trade log --
    def log_trade(self, row: Dict[str, Any]) -> None:
        """Append one trade row to trades.csv."""
        self._trades_writer.writerow(row)

    # -- Lifecycle --
    def close(self) -> None:
        """Flush and close all file handles."""
        try:
            self._fh.flush()
        finally:
            self._fh.close()
        try:
            self._trades_fh.flush()
        finally:
            self._trades_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _safe_serialize(obj: Any) -> Any:
    """Recursively convert dataclasses / non-serializable types."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _safe_serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ============================================================================
# CONSOLIDATED TRADE LOG (Spec: full per-trade record)
# ============================================================================

@dataclass
class TradeRecord:
    """
    Single consolidated trade record meeting spec requirements.

    Joins: timestamps, asset, direction, L1 signals, L2 ensemble output,
    L3 target weights, L4 executed weights, fill prices, itemized costs,
    PnL attribution, and kill/suppression/demotion events.
    """
    bar_idx: int
    timestamp: str = ""
    asset: str = ""

    # Direction
    direction: str = ""       # "long", "short", "flat", "cover", "sell"
    trade_size: float = 0.0   # Signed position change

    # L1 signals (per-alpha)
    l1_signals: Dict[str, Dict] = field(default_factory=dict)
    # e.g. {'trend': {'mu': 0.01, 'sigma': 0.05, 'conf': 0.6}, ...}

    # L2 ensemble output
    l2_mu_hat: float = 0.0
    l2_sigma_hat: float = 0.0
    l2_diagnostics: Dict = field(default_factory=dict)

    # L3 target
    l3_target_exposure: float = 0.0
    l3_constraints_hit: List[str] = field(default_factory=list)

    # L4 execution
    l4_executed_exposure: float = 0.0
    l4_was_suppressed: bool = False
    l4_was_killed: bool = False
    l4_kill_type: str = ""

    # Fill and costs (itemized)
    fill_price: float = 0.0
    cost_spread: float = 0.0
    cost_impact: float = 0.0
    cost_fees: float = 0.0
    cost_borrow: float = 0.0
    cost_total: float = 0.0

    # PnL attribution
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Events
    demotion_events: List[str] = field(default_factory=list)
    regime: str = ""


class ConsolidatedTradeLog:
    """
    Maintains a queryable log of all trades with full provenance.

    Satisfies spec requirement:
        "trade logs with timestamps, asset, direction, signals (mu/sigma/conf
         from L1 and L2), target weights (from L3), executed weights, fill
         prices, costs (spread + impact + fees, itemized), PnL attribution,
         and kill/suppression/demotion events"

    Usage:
        trade_log = ConsolidatedTradeLog()
        trade_log.record(TradeRecord(bar_idx=100, ...))
        trade_log.to_csv("trades.csv")
    """

    def __init__(self, max_records: int = 100_000):
        self._records: List[TradeRecord] = []
        self._max_records = max_records

    def record(self, trade: TradeRecord):
        """Add a trade record to the log."""
        if len(self._records) < self._max_records:
            self._records.append(trade)

    def record_from_pipeline_result(self, result, asset: str = "",
                                     regime: str = ""):
        """
        Build and record a TradeRecord from a PipelineResult.

        This is the primary integration point: called after each pipeline.step().
        """
        audit = getattr(result, 'audit', {}) or {}
        l1_signals = {}
        for name, detail in audit.get('alpha_signals', {}).items():
            l1_signals[name] = {
                'mu': detail.get('mu', 0.0),
                'sigma': detail.get('sigma', 0.0),
                'conf': detail.get('confidence', 0.0),
            }

        # Determine direction
        target = getattr(result, 'target_exposure', 0.0)
        if target > 0.01:
            direction = "long"
        elif target < -0.01:
            direction = "short"
        else:
            direction = "flat"

        # Kill/suppression info
        exec_res = getattr(result, 'execution_result', None)
        was_killed = getattr(exec_res, 'was_killed', False) if exec_res else False
        was_suppressed = getattr(exec_res, 'was_suppressed', False) if exec_res else False
        kill_type = audit.get('kill_type', '')

        trade = TradeRecord(
            bar_idx=getattr(result, 'bar_idx', 0),
            asset=asset,
            direction=direction,
            l1_signals=l1_signals,
            l2_mu_hat=audit.get('mu_hat', 0.0),
            l2_sigma_hat=audit.get('sigma_hat', 0.0),
            l2_diagnostics=audit.get('l2_diagnostics', {}),
            l3_target_exposure=target,
            l3_constraints_hit=audit.get('constraints_hit', []),
            l4_executed_exposure=getattr(result, 'executed_exposure',
                                         getattr(result, 'discrete_action', 0)),
            l4_was_suppressed=was_suppressed,
            l4_was_killed=was_killed,
            l4_kill_type=kill_type,
            regime=regime,
        )
        self.record(trade)
        return trade

    def to_csv(self, filepath: str):
        """Export trade log to CSV file."""
        import csv
        if not self._records:
            return

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        fieldnames = [
            'bar_idx', 'timestamp', 'asset', 'direction', 'trade_size',
            'l2_mu_hat', 'l2_sigma_hat',
            'l3_target_exposure', 'l3_constraints_hit',
            'l4_executed_exposure', 'l4_was_suppressed', 'l4_was_killed',
            'l4_kill_type',
            'fill_price', 'cost_spread', 'cost_impact', 'cost_fees',
            'cost_borrow', 'cost_total',
            'gross_pnl', 'net_pnl', 'regime',
        ]
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for rec in self._records:
                row = {fn: getattr(rec, fn, '') for fn in fieldnames}
                row['l3_constraints_hit'] = ','.join(rec.l3_constraints_hit)
                writer.writerow(row)

    def to_jsonl(self, filepath: str):
        """Export trade log as JSON Lines."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            for rec in self._records:
                f.write(json.dumps(asdict(rec), default=str) + '\n')

    @property
    def records(self) -> List[TradeRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)