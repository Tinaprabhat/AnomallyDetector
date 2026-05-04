"""
Isolation Forest detector.

Critical fix from v1: use `decision_function` for continuous scores,
NOT `predict` which returns binary {-1, +1}. v1 stored binary preds as
"anomaly_score" and computed ROC AUC on them — mathematically broken.

This module returns proper continuous anomaly scores in [0, 1].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IFConfig:
    n_estimators: int = 100
    contamination: float = 0.05
    random_state: int = 42


class IsolationForestDetector:
    """IF wrapped to return continuous [0, 1] scores."""

    def __init__(self, config: Optional[IFConfig] = None):
        self.config = config or IFConfig()
        self.model: Optional[IsolationForest] = None
        self._score_min: Optional[float] = None
        self._score_max: Optional[float] = None
        self._is_fit = False

    def fit(self, X: np.ndarray, X_val: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        norm_X = X_val if X_val is not None else X
        raw = -self.model.decision_function(norm_X)  # higher = more anomalous
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        self._is_fit = True
        logger.info("isolation_forest_fit_complete", n_samples=len(X),
                    score_min=self._score_min, score_max=self._score_max)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Continuous anomaly score in [0, 1]. Higher = more anomalous."""
        if not self._is_fit or self.model is None:
            raise RuntimeError("IsolationForestDetector not fit. Call .fit() first.")
        raw = -self.model.decision_function(X)
        rng = max(self._score_max - self._score_min, 1e-9)
        return np.clip((raw - self._score_min) / rng, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)
