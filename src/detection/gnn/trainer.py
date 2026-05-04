"""
GNN Trainer — salvaged + refactored from v1's trainer_gnn.py.

Improvements:
- Temporal masks (not random splits)
- Class weights for imbalance
- Cleaner alignment logic
- Returns embeddings AND probabilities
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx
    import networkx as nx
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from src.detection.gnn.graphsage import GraphSAGE
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GNNConfig:
    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    random_state: int = 42


class GNNTrainer:
    """Wraps GraphSAGE: trains, predicts, extracts embeddings."""

    def __init__(self, config: Optional[GNNConfig] = None):
        if not _HAS_PYG:
            raise ImportError(
                "GNN training requires torch + torch_geometric. "
                "Install: pip install torch torch-geometric"
            )
        self.config = config or GNNConfig()
        self.model: Optional[GraphSAGE] = None
        self.data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_fit = False
        self.node_id_to_idx: dict = {}

    @staticmethod
    def build_pyg_data(graph, features_df, labels_series):
        common = [n for n in graph.nodes() if n in features_df.index]
        sub = graph.subgraph(common).copy()
        node_list = list(sub.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        feats = np.vstack([features_df.loc[n].values.astype(np.float32) for n in node_list])
        x = torch.tensor(feats, dtype=torch.float32)
        H = nx.relabel_nodes(sub, node_to_idx, copy=True)
        pyg = from_networkx(H)
        pyg.x = x
        labels = np.array([int(labels_series.get(n, -1)) for n in node_list], dtype=np.int64)
        pyg.y = torch.tensor(labels, dtype=torch.long)
        pyg.node_ids = node_list
        return pyg, node_to_idx

    def fit(self, graph, features_df, labels_series, train_ids, val_ids):
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        self.data, self.node_id_to_idx = self.build_pyg_data(graph, features_df, labels_series)
        n_nodes = self.data.x.shape[0]
        train_set = set(map(str, train_ids))
        val_set = set(map(str, val_ids))

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        for i, n in enumerate(self.data.node_ids):
            n_str = str(n)
            label = int(self.data.y[i].item())
            if label == -1:
                continue
            if n_str in train_set:
                train_mask[i] = True
            elif n_str in val_set:
                val_mask[i] = True

        train_labels = self.data.y[train_mask].cpu().numpy()
        n_pos = max(int((train_labels == 1).sum()), 1)
        n_neg = max(int((train_labels == 0).sum()), 1)
        weight_pos = (n_pos + n_neg) / (2.0 * n_pos)
        weight_neg = (n_pos + n_neg) / (2.0 * n_neg)
        class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(self.device)

        self.model = GraphSAGE(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.config.hidden_channels,
            out_channels=2,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        opt = optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                         weight_decay=self.config.weight_decay)
        crit = nn.CrossEntropyLoss(weight=class_weights)
        self.data = self.data.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            opt.zero_grad()
            logits, _ = self.model(self.data.x, self.data.edge_index)
            loss = crit(logits[train_mask], self.data.y[train_mask])
            loss.backward()
            opt.step()
            if epoch % 5 == 0 or epoch == 1:
                self.model.eval()
                with torch.no_grad():
                    val_logits, _ = self.model(self.data.x, self.data.edge_index)
                    val_pred = val_logits[val_mask].argmax(dim=1)
                    val_acc = (val_pred == self.data.y[val_mask]).float().mean().item()
                logger.info("gnn_epoch", epoch=epoch, train_loss=float(loss.item()),
                            val_acc=val_acc)
        self._is_fit = True
        return self

    def predict_proba(self, tx_ids: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("GNNTrainer not fit. Call .fit() first.")
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        result = np.zeros(len(tx_ids))
        for i, tx in enumerate(tx_ids):
            idx = self.node_id_to_idx.get(str(tx))
            if idx is not None:
                result[i] = float(probs[idx])
        return result

    def get_embeddings(self, tx_ids: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("GNNTrainer not fit. Call .fit() first.")
        self.model.eval()
        with torch.no_grad():
            _, embs = self.model(self.data.x, self.data.edge_index)
            embs_np = embs.cpu().numpy()
        emb_dim = embs_np.shape[1]
        result = np.zeros((len(tx_ids), emb_dim))
        for i, tx in enumerate(tx_ids):
            idx = self.node_id_to_idx.get(str(tx))
            if idx is not None:
                result[i] = embs_np[idx]
        return result
