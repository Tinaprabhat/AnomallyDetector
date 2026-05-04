"""Tabular feature pipeline — StandardScaler + optional PCA."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class TabularPipeline:
    """Holds fitted scaler + optional PCA for reuse on val/test data."""
    scaler: StandardScaler
    pca: Optional[PCA] = None

    @property
    def feature_dim(self) -> int:
        if self.pca is not None:
            return self.pca.n_components_
        return self.scaler.n_features_in_


def fit_tabular_pipeline(
    X_train: pd.DataFrame | np.ndarray,
    apply_pca: bool = False,
    pca_components: int = 30,
    random_state: int = 42,
) -> TabularPipeline:
    """Fit StandardScaler (and optional PCA) on training data only."""
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    scaler = StandardScaler()
    scaler.fit(X)
    pca = None
    if apply_pca:
        pca = PCA(n_components=pca_components, random_state=random_state)
        pca.fit(scaler.transform(X))
    return TabularPipeline(scaler=scaler, pca=pca)


def transform_tabular(
    pipeline: TabularPipeline,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Apply fitted pipeline to new data."""
    arr = X.values if isinstance(X, pd.DataFrame) else X
    arr = pipeline.scaler.transform(arr)
    if pipeline.pca is not None:
        arr = pipeline.pca.transform(arr)
    return arr
