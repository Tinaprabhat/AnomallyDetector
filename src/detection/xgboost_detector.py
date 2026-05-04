"""
XGBoost detector — primary supervised tabular fraud detector.

Per locked design:
- scale_pos_weight to handle ~2% illicit imbalance
- Platt scaling for probability calibration
- Tabular features only (GraphSAGE handles graph)

Falls back gracefully if xgboost isn't installed (so unit tests can still run
on the rest of the pipeline).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class XGBoostConfig:
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    scale_pos_weight: float = 50.0
    random_state: int = 42


class XGBoostDetector:
    """XGBoost with Platt-scaled calibrated probabilities."""

    def __init__(self, config: Optional[XGBoostConfig] = None):
        if not _HAS_XGB:
            raise ImportError("XGBoostDetector requires xgboost. pip install xgboost")
        self.config = config or XGBoostConfig()
        self.model: Optional[CalibratedClassifierCV] = None
        self._is_fit = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostDetector":
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        spw = self.config.scale_pos_weight
        if spw <= 1.0:
            spw = max(n_neg / n_pos, 1.0)

        base = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            scale_pos_weight=spw,
            random_state=self.config.random_state,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        )

        if X_val is not None and y_val is not None:
            base.fit(X_train, y_train)
            self.model = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
            self.model.fit(X_val, y_val)
        else:
            self.model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
            self.model.fit(X_train, y_train)

        self._is_fit = True
        logger.info("xgboost_fit_complete", n_train=len(y_train),
                    n_pos=n_pos, n_neg=n_neg, scale_pos_weight=float(spw))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit or self.model is None:
            raise RuntimeError("XGBoostDetector not fit. Call .fit() first.")
        proba = self.model.predict_proba(X)
        if proba.shape[1] < 2:
            return np.zeros(len(X))
        return proba[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
