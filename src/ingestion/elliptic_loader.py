"""
Elliptic Bitcoin Dataset Loader.

Salvaged and cleaned from v1's Source/dataset.py:
- Robust label normalization (handles "1"/"2", "licit"/"illicit", "unknown")
- Returns clean DataFrames + NetworkX graph
- Type-safe column handling
- Includes load_synthetic() for unit tests when real data isn't available
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EllipticData:
    """Container for the loaded Elliptic dataset."""
    features: pd.DataFrame      # indexed by txId; cols: time_step + 165 features
    classes: pd.DataFrame       # cols: txId, label (0=licit, 1=illicit, -1=unknown)
    edges: pd.DataFrame         # cols: txId1, txId2
    graph: nx.Graph             # transaction graph
    time_steps: pd.Series       # txId -> time_step

    def labeled_mask(self) -> pd.Series:
        """Boolean mask for transactions with known labels (label != -1)."""
        return self.classes["label"] != -1

    def labeled_subset(self) -> pd.DataFrame:
        """Return only labeled transactions."""
        return self.classes[self.labeled_mask()]


class EllipticLoader:
    """Loads the Elliptic Bitcoin dataset from CSV files."""

    LABEL_MAP = {
        "unknown": -1, "licit": 0, "illicit": 1,
        "1": 0, "2": 1, 1: 0, 2: 1,
    }

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            logger.warning("dataset_root_missing", path=str(self.root))

    def _normalize_label(self, value) -> int:
        """Map any incoming label representation to {-1, 0, 1}."""
        s = str(value).strip()
        if s in self.LABEL_MAP:
            return int(self.LABEL_MAP[s])
        if s.lstrip("-").isdigit():
            v = int(s)
            return v if v in (-1, 0, 1) else -1
        return -1

    def load(self) -> EllipticData:
        """Load all three CSVs and build the graph."""
        features_path = self.root / "elliptic_txs_features.csv"
        classes_path = self.root / "elliptic_txs_classes.csv"
        edges_path = self.root / "elliptic_txs_edgelist.csv"

        features = pd.read_csv(features_path, header=None)
        n_features = features.shape[1] - 2
        features.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(n_features)]
        features["txId"] = features["txId"].astype(str)
        features = features.set_index("txId")

        classes = pd.read_csv(classes_path, header=None)
        if not str(classes.iloc[0, 0]).lstrip("-").isdigit():
            classes = classes.iloc[1:].reset_index(drop=True)
        classes.columns = ["txId", "label"]
        classes["txId"] = classes["txId"].astype(str)
        classes["label"] = classes["label"].apply(self._normalize_label).astype(int)

        edges = pd.read_csv(edges_path, header=None)
        if not str(edges.iloc[0, 0]).lstrip("-").isdigit():
            edges = edges.iloc[1:].reset_index(drop=True)
        edges.columns = ["txId1", "txId2"]
        edges["txId1"] = edges["txId1"].astype(str)
        edges["txId2"] = edges["txId2"].astype(str)

        graph = nx.from_pandas_edgelist(edges, source="txId1", target="txId2")
        time_steps = features["time_step"].astype(int)

        logger.info(
            "elliptic_loaded",
            n_features=n_features,
            n_transactions=len(features),
            n_classes=len(classes),
            n_edges=len(edges),
        )

        return EllipticData(
            features=features,
            classes=classes,
            edges=edges,
            graph=graph,
            time_steps=time_steps,
        )

    def load_synthetic(self, n_transactions: int = 1000, seed: int = 42) -> EllipticData:
        """Generate a synthetic dataset with the same shape as Elliptic.
        Used for unit tests when real data isn't available."""
        rng = np.random.default_rng(seed)
        n_features = 165

        tx_ids = [f"tx_{i:06d}" for i in range(n_transactions)]
        time_steps = rng.integers(1, 50, size=n_transactions)
        feat_array = rng.standard_normal((n_transactions, n_features))

        features = pd.DataFrame(
            np.column_stack([time_steps, feat_array]),
            columns=["time_step"] + [f"feat_{i}" for i in range(n_features)],
            index=pd.Index(tx_ids, name="txId"),
        )
        features["time_step"] = features["time_step"].astype(int)

        labels = rng.choice([-1, 0, 1], size=n_transactions, p=[0.78, 0.20, 0.02])
        classes = pd.DataFrame({"txId": tx_ids, "label": labels})

        n_edges = n_transactions * 2
        src = rng.choice(tx_ids, size=n_edges)
        dst = rng.choice(tx_ids, size=n_edges)
        edges = pd.DataFrame({"txId1": src, "txId2": dst})
        edges = edges[edges["txId1"] != edges["txId2"]].reset_index(drop=True)

        graph = nx.from_pandas_edgelist(edges, source="txId1", target="txId2")
        time_step_series = features["time_step"]

        return EllipticData(
            features=features,
            classes=classes,
            edges=edges,
            graph=graph,
            time_steps=time_step_series,
        )
