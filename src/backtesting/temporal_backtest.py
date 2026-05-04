"""
Temporal backtest harness — rolling-window evaluation.

Runs rolling-window backtests across the 49 Elliptic time steps to validate
that the detection layer holds up out-of-time. This is the primary validation
method since we have no real-time data.

Default windows (from default.yaml):
    1: train [1-25],  test [26-30]
    2: train [1-30],  test [31-35]
    3: train [1-35],  test [36-40]
    4: train [1-40],  test [41-49]
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.backtesting.metrics import FraudMetrics, compute_metrics
from src.backtesting.cost_weighted import CostBreakdown, compute_cost
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def label(self) -> str:
        return f"train[{self.train_start}-{self.train_end}]_test[{self.test_start}-{self.test_end}]"


@dataclass
class WindowResult:
    window: BacktestWindow
    metrics: FraudMetrics
    cost_breakdown: Optional[CostBreakdown] = None
    n_train: int = 0
    n_test: int = 0
    notes: str = ""


@dataclass
class BacktestReport:
    windows: List[WindowResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "windows": [
                {
                    "window": asdict(w.window),
                    "metrics": w.metrics.to_dict(),
                    "cost": asdict(w.cost_breakdown) if w.cost_breakdown else None,
                    "n_train": w.n_train,
                    "n_test": w.n_test,
                    "notes": w.notes,
                }
                for w in self.windows
            ],
            "summary": self.summary,
        }


def parse_windows(config_windows: List[Dict]) -> List[BacktestWindow]:
    out = []
    for entry in config_windows:
        tr = entry["train"]; te = entry["test"]
        out.append(BacktestWindow(
            train_start=int(tr[0]), train_end=int(tr[1]),
            test_start=int(te[0]), test_end=int(te[1]),
        ))
    return out


# train_score_fn: (train_ids, test_ids) -> (y_true_test, y_scores_test)
TrainScoreFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class TemporalBacktester:
    """Run rolling-window backtests using a caller-supplied train+score function."""

    def __init__(
        self,
        time_steps: Dict[str, int],
        cost_fn: float = 100.0,
        cost_fp: float = 1.0,
    ):
        self.time_steps = time_steps
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def _ids_in_range(self, lo: int, hi: int) -> np.ndarray:
        return np.array([tx for tx, t in self.time_steps.items() if lo <= t <= hi])

    def run(
        self,
        windows: List[BacktestWindow],
        train_score_fn: TrainScoreFn,
    ) -> BacktestReport:
        report = BacktestReport()
        all_pr = []
        for i, w in enumerate(windows):
            train_ids = self._ids_in_range(w.train_start, w.train_end)
            test_ids = self._ids_in_range(w.test_start, w.test_end)
            logger.info("backtest_window_start", idx=i, window=w.label(),
                        n_train=len(train_ids), n_test=len(test_ids))
            try:
                y_true, y_scores = train_score_fn(train_ids, test_ids)
            except Exception as e:
                logger.warning("backtest_window_failed", idx=i, error=str(e))
                report.windows.append(WindowResult(
                    window=w,
                    metrics=FraudMetrics(0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.0),
                    n_train=len(train_ids), n_test=len(test_ids),
                    notes=f"FAILED: {e}",
                ))
                continue

            y_true = np.asarray(y_true).astype(int).ravel()
            y_scores = np.asarray(y_scores).astype(float).ravel()
            labeled = y_true != -1
            y_true = y_true[labeled]
            y_scores = y_scores[labeled]
            metrics = compute_metrics(y_true, y_scores)
            y_pred = (y_scores >= metrics.optimal_threshold).astype(int)
            cost = compute_cost(y_true, y_pred, self.cost_fn, self.cost_fp)
            report.windows.append(WindowResult(
                window=w, metrics=metrics, cost_breakdown=cost,
                n_train=len(train_ids), n_test=len(test_ids),
            ))
            all_pr.append(metrics.pr_auc)
            logger.info("backtest_window_done", idx=i,
                        pr_auc=metrics.pr_auc, cost=cost.total_cost)

        if all_pr:
            report.summary = {
                "n_windows": len(report.windows),
                "mean_pr_auc": float(np.mean(all_pr)),
                "std_pr_auc": float(np.std(all_pr)),
                "min_pr_auc": float(np.min(all_pr)),
                "max_pr_auc": float(np.max(all_pr)),
            }
        else:
            report.summary = {"n_windows": 0}
        return report
