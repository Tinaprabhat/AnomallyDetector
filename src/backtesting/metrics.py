"""
Production-grade metrics for fraud detection.

Critical metrics for imbalanced data (~2% illicit):
- PR-AUC (Average Precision) — primary
- Precision@k — analyst review capacity
- MCC (Matthews Correlation Coefficient) — robust to imbalance
- F1, ROC-AUC — for completeness
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass
class FraudMetrics:
    pr_auc: float
    roc_auc: float
    precision_at_5pct: float
    precision_at_10pct: float
    f1_at_optimal_threshold: float
    mcc_at_optimal_threshold: float
    optimal_threshold: float
    n_samples: int
    n_positives: int
    base_rate: float

    def to_dict(self) -> Dict:
        return asdict(self)


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(y_scores):
        return 0.0
    order = np.argsort(-y_scores)
    return float(y_true[order[:k]].sum() / k)


def precision_at_pct(y_true: np.ndarray, y_scores: np.ndarray, pct: float) -> float:
    k = max(1, int(len(y_scores) * pct))
    return precision_at_k(y_true, y_scores, k)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1",
) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5; best = -1.0
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "mcc":
            try:
                score = matthews_corrcoef(y_true, y_pred)
            except Exception:
                score = -1.0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if score > best:
            best = score; best_t = float(t)
    return best_t


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> FraudMetrics:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_scores = np.asarray(y_scores).astype(float).ravel()

    # Drop unlabeled samples (-1) — they have no ground truth for binary scoring
    labeled = y_true != -1
    y_true = y_true[labeled]
    y_scores = y_scores[labeled]

    n = len(y_true)
    n_pos = int(y_true.sum())
    base_rate = n_pos / n if n > 0 else 0.0

    if n == 0 or n_pos == 0 or n_pos == n:
        return FraudMetrics(0, 0, 0, 0, 0, 0, 0.5, n, n_pos, base_rate)

    pr_auc = float(average_precision_score(y_true, y_scores))
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc_auc = 0.0
    p_at_5 = precision_at_pct(y_true, y_scores, 0.05)
    p_at_10 = precision_at_pct(y_true, y_scores, 0.10)
    optimal_t = find_optimal_threshold(y_true, y_scores, metric="f1")
    y_pred = (y_scores >= optimal_t).astype(int)
    f1_opt = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        mcc_opt = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc_opt = 0.0

    return FraudMetrics(
        pr_auc=pr_auc, roc_auc=roc_auc,
        precision_at_5pct=p_at_5, precision_at_10pct=p_at_10,
        f1_at_optimal_threshold=f1_opt, mcc_at_optimal_threshold=mcc_opt,
        optimal_threshold=optimal_t, n_samples=n, n_positives=n_pos,
        base_rate=base_rate,
    )
