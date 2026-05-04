"""
Temporal splitter for Elliptic Bitcoin Dataset.

Critical: Elliptic has 49 time steps. Random splits would leak future into past.
Strict temporal boundaries — train: 1-30, val: 31-40, test: 41-49.
This is THE most important correctness fix from v1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.ingestion.elliptic_loader import EllipticData
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalSplit:
    """Container for a temporal split — train/val/test transaction IDs."""
    train_tx_ids: np.ndarray
    val_tx_ids: np.ndarray
    test_tx_ids: np.ndarray
    train_range: Tuple[int, int]
    val_range: Tuple[int, int]
    test_range: Tuple[int, int]

    def summary(self) -> dict:
        return {
            "n_train": len(self.train_tx_ids),
            "n_val": len(self.val_tx_ids),
            "n_test": len(self.test_tx_ids),
            "train_range": self.train_range,
            "val_range": self.val_range,
            "test_range": self.test_range,
        }


class TemporalSplitter:
    """Split transactions by time step. Default: 1-30 / 31-40 / 41-49."""

    def __init__(
        self,
        train_range: Tuple[int, int] = (1, 30),
        val_range: Tuple[int, int] = (31, 40),
        test_range: Tuple[int, int] = (41, 49),
        labeled_only: bool = True,
    ):
        if train_range[1] >= val_range[0]:
            raise ValueError(f"train_range {train_range} overlaps val_range {val_range}")
        if val_range[1] >= test_range[0]:
            raise ValueError(f"val_range {val_range} overlaps test_range {test_range}")
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        self.labeled_only = labeled_only

    def split(self, data: EllipticData) -> TemporalSplit:
        time_steps = data.time_steps
        if self.labeled_only:
            labeled_tx = data.classes.loc[data.classes["label"] != -1, "txId"].astype(str)
            mask_labeled = time_steps.index.isin(labeled_tx.values)
            time_steps = time_steps[mask_labeled]

        train_mask = time_steps.between(*self.train_range)
        val_mask = time_steps.between(*self.val_range)
        test_mask = time_steps.between(*self.test_range)

        split = TemporalSplit(
            train_tx_ids=time_steps[train_mask].index.values,
            val_tx_ids=time_steps[val_mask].index.values,
            test_tx_ids=time_steps[test_mask].index.values,
            train_range=self.train_range,
            val_range=self.val_range,
            test_range=self.test_range,
        )
        logger.info("temporal_split_created", **split.summary())
        return split

    @staticmethod
    def get_features_and_labels(
        data: EllipticData,
        tx_ids: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        valid = pd.Index(tx_ids).intersection(data.features.index)
        feats = data.features.loc[valid].drop(columns=["time_step"])
        labels_map = data.classes.set_index("txId")["label"].to_dict()
        labels = np.array([labels_map.get(t, -1) for t in valid])
        return feats, labels
