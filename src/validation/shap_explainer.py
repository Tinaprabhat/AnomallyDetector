"""
SHAP explainer — top-K feature attributions for the XGBoost detector.

Falls back to feature_importance-based pseudo-attributions if shap is not
available. Used as part of statistical validation evidence + LLM context.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SHAPExplanation:
    top_features: List[Tuple[str, float]]


class SHAPExplainer:
    """Compute SHAP values for XGBoost predictions."""

    def __init__(self, feature_names: Optional[List[str]] = None, top_k: int = 3):
        self.feature_names = feature_names
        self.top_k = top_k
        self.explainer = None
        self.fallback_importances: Optional[np.ndarray] = None

    def fit(self, model, X_background: Optional[np.ndarray] = None) -> "SHAPExplainer":
        underlying = model
        if hasattr(model, "calibrated_classifiers_"):
            underlying = model.calibrated_classifiers_[0].estimator

        if _HAS_SHAP:
            try:
                self.explainer = shap.TreeExplainer(underlying)
                logger.info("shap_explainer_fit", explainer="TreeExplainer")
                return self
            except Exception as e:
                logger.warning("shap_treeexplainer_failed", error=str(e))

        if hasattr(underlying, "feature_importances_"):
            self.fallback_importances = underlying.feature_importances_
            logger.info("shap_fallback_using_feature_importances")
        return self

    def explain_one(self, x: np.ndarray) -> SHAPExplanation:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(x)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                row = np.asarray(shap_vals)[0]
            except Exception as e:
                logger.warning("shap_explain_failed", error=str(e))
                row = np.zeros(x.shape[1])
        elif self.fallback_importances is not None:
            row = self.fallback_importances * np.abs(x[0])
        else:
            row = np.zeros(x.shape[1])

        n_feat = len(row)
        if self.feature_names is None:
            names = [f"feat_{i}" for i in range(n_feat)]
        else:
            names = list(self.feature_names)
            if len(names) < n_feat:
                names += [f"feat_{i}" for i in range(len(names), n_feat)]

        abs_vals = np.abs(row)
        order = np.argsort(-abs_vals)[: self.top_k]
        top = [(names[i], float(row[i])) for i in order]
        return SHAPExplanation(top_features=top)
