"""
Mahalanobis distance validator.

For each flagged transaction, compute the Mahalanobis distance from the licit
cluster centroid. Returns distance + chi-squared p-value.
Provides per-instance statistical evidence — independent of any single model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import chi2

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MahalanobisStats:
    distance_squared: float
    p_value: float


class MahalanobisValidator:
    """Compute Mahalanobis distance from a fitted reference distribution."""

    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
        self.mean_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None
        self.dim_: Optional[int] = None
        self._is_fit = False

    def fit(self, X_licit: np.ndarray) -> "MahalanobisValidator":
        """Fit on licit (normal) transactions."""
        if X_licit.shape[0] < 2:
            raise ValueError("Need at least 2 samples to compute covariance")
        self.mean_ = X_licit.mean(axis=0)
        cov = np.cov(X_licit, rowvar=False)
        cov_reg = cov + self.regularization * np.eye(cov.shape[0])
        self.cov_inv_ = np.linalg.pinv(cov_reg)
        self.dim_ = X_licit.shape[1]
        self._is_fit = True
        logger.info("mahalanobis_fit_complete", n_samples=X_licit.shape[0], dim=self.dim_)
        return self

    def compute(self, X: np.ndarray) -> np.ndarray:
        """Return D² for each row of X."""
        if not self._is_fit:
            raise RuntimeError("MahalanobisValidator not fit.")
        diff = X - self.mean_
        d_sq = np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff)
        return d_sq

    def p_value(self, X: np.ndarray) -> np.ndarray:
        """p-value under chi-squared with df = dim."""
        d_sq = self.compute(X)
        return 1.0 - chi2.cdf(d_sq, df=self.dim_)

    def validate_one(self, x: np.ndarray) -> MahalanobisStats:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        d_sq = float(self.compute(x)[0])
        p = float(1.0 - chi2.cdf(d_sq, df=self.dim_))
        return MahalanobisStats(distance_squared=d_sq, p_value=p)
