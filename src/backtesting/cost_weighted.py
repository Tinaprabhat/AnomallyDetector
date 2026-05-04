"""
Cost-weighted evaluation.

In fraud detection, false negatives (missed fraud) typically cost much more
than false positives (analyst review time). Cost-weighted metrics translate
predictions into expected dollar/operational cost.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class CostBreakdown:
    n_tp: int
    n_fp: int
    n_tn: int
    n_fn: int
    cost_fn: float
    cost_fp: float
    total_cost: float
    cost_per_transaction: float


def compute_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0,
) -> CostBreakdown:
    """Compute cost breakdown from binary predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = fn * cost_fn + fp * cost_fp
    n = len(y_true)
    per_tx = total / n if n > 0 else 0.0
    return CostBreakdown(
        n_tp=tp, n_fp=fp, n_tn=tn, n_fn=fn,
        cost_fn=cost_fn, cost_fp=cost_fp,
        total_cost=total, cost_per_transaction=per_tx,
    )


def find_min_cost_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0,
) -> float:
    """Find threshold minimizing total cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5; best_cost = float("inf")
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        cost = compute_cost(y_true, y_pred, cost_fn, cost_fp).total_cost
        if cost < best_cost:
            best_cost = cost; best_t = float(t)
    return best_t
