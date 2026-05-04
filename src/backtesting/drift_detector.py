"""
Drift detection — flags significant performance drops or distribution shifts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import ks_2samp

from src.backtesting.metrics import FraudMetrics


@dataclass
class DriftSignal:
    window_idx: int
    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    is_significant: bool
    message: str


def detect_metric_drift(
    metrics_per_window: List[FraudMetrics],
    metric_name: str = "pr_auc",
    drop_threshold: float = 0.10,
) -> List[DriftSignal]:
    """Compare each window to the first (baseline). Flag drops > threshold."""
    if len(metrics_per_window) < 2:
        return []
    baseline = getattr(metrics_per_window[0], metric_name)
    signals = []
    for i, m in enumerate(metrics_per_window[1:], start=1):
        cur = getattr(m, metric_name)
        delta = cur - baseline
        is_sig = abs(delta) >= drop_threshold and delta < 0
        signals.append(DriftSignal(
            window_idx=i, metric_name=metric_name,
            baseline_value=baseline, current_value=cur,
            delta=delta, is_significant=is_sig,
            message=(f"{metric_name} dropped by {abs(delta):.3f}"
                     if is_sig else f"{metric_name} stable"),
        ))
    return signals


def detect_distribution_drift(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    p_threshold: float = 0.01,
) -> Tuple[float, float, bool]:
    """KS test on score distributions. Returns (statistic, p_value, is_significant)."""
    if len(baseline_scores) < 2 or len(current_scores) < 2:
        return 0.0, 1.0, False
    stat, p = ks_2samp(baseline_scores, current_scores)
    return float(stat), float(p), bool(p < p_threshold)
