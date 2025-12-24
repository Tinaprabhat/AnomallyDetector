"""
Source/gnn_model.py

Graph neural network utilities for anomaly detection on transaction graphs.
- Converts networkx graph + feature matrix + labels -> torch_geometric.data.Data
- Implements a simple GraphSAGE model (2 layers)
- Training, evaluation, embedding extraction helpers
"""

import os
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

# Try importing PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.utils import from_networkx
except Exception as e:
    raise ImportError(
        "PyTorch Geometric (torch_geometric) is required for gnn_model.py. "
        "Install it following instructions at https://pytorch-geometric.readthedocs.io/ "
        "or run: pip install torch-geometric -f https://data.pyg.org/whl/torch-<torch-version>+<cuda>.html"
    ) from e

import networkx as nx
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support


# -------------------------
# Utility: Build PyG Data
# -------------------------
def build_pyg_data(
    G: nx.Graph,
    features_df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    node_id_col: Optional[str] = None,
    include_edge_index: bool = True,
) -> Data:
    """
    Convert a NetworkX graph + feature DataFrame (+ optional labels) into a PyG Data object.
    - G: networkx graph (nodes should be same keys as features_df.index or contain an id attribute)
    - features_df: pd.DataFrame indexed by node id; each row -> feature vector
    - labels: pd.Series indexed by node id OR dict node->label
    Returns: torch_geometric.data.Data with x, edge_index, y (if labels provided), node_id map as metadata
    """
    # Ensure features_df index are strings/ints matching G nodes
    node_list = list(G.nodes())
    # Create mapping from node -> index
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    # Build feature matrix aligned to G.nodes() order
    feats = []
    for n in node_list:
        if n in features_df.index:
            feats.append(features_df.loc[n].values.astype(np.float32))
        else:
            # fallback: zeros if node missing in features_df
            feats.append(np.zeros(features_df.shape[1], dtype=np.float32))
    x = torch.tensor(np.vstack(feats), dtype=torch.float32)

    # Use from_networkx to get edge_index; ensure node ordering consistent
    H = nx.relabel_nodes(G, node_to_idx, copy=True)
    pyg_data = from_networkx(H)  # will include edge_index
    if not hasattr(pyg_data, "edge_index") and include_edge_index:
        # fallback: build manually
        edge_index = torch.tensor(list(H.edges)).t().contiguous()
        pyg_data.edge_index = edge_index

    pyg_data.x = x

    # Attach labels if given
    if labels is not None:
        if isinstance(labels, pd.Series):
            y_list = []
            for n in node_list:
                y_list.append(labels.get(n, -1))  # -1 if unknown
            pyg_data.y = torch.tensor(y_list, dtype=torch.long)
        elif isinstance(labels, dict):
            y_list = [labels.get(n, -1) for n in node_list]
            pyg_data.y = torch.tensor(y_list, dtype=torch.long)
        else:
            raise ValueError("labels must be a pd.Series or dict mapping node -> label")

    # Save mapping
    pyg_data.node_id = node_list
    pyg_data.node_to_idx = node_to_idx
    return pyg_data


# -------------------------
# GraphSAGE Model
# -------------------------
class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super(GraphSAGE, self).__init__()
        assert num_layers >= 2, "num_layers should be at least 2"
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        # x: [N, F], edge_index: [2, E]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin(x)
        return out, x  # return logits and final embeddings


# -------------------------
# Training & Evaluation
# -------------------------
def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer,
    criterion,
    device: torch.device,
    train_mask: Optional[torch.Tensor] = None,
):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    logits, _ = model(data.x, data.edge_index)
    if train_mask is None:
        mask = torch.ones(len(logits), dtype=torch.bool, device=device)
    else:
        mask = train_mask.to(device)
    loss = criterion(logits[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(
    model: nn.Module,
    data: Data,
    device: torch.device,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Evaluate a GNN model on given data.

    Args:
        model (nn.Module): Trained GNN model.
        data (Data): PyG Data object.
        device (torch.device): Device to run evaluation on (CPU or GPU).
        mask (Optional[torch.Tensor]): Boolean mask to select nodes.

    Returns:
        Dict[str, float]: Dictionary with metrics: auc, acc, precision, recall, f1.
    """
    model.eval()
    model = model.to(device)
    data = data.to(device)

    with torch.no_grad():
        logits, embeddings = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # probability for class 1
        preds = logits.argmax(dim=1).cpu().numpy()

    # Ground truth
    y_true = data.y.cpu().numpy() if hasattr(data, "y") else None
    if y_true is None:
        return {"auc": float("nan"), "acc": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    # Apply mask
    if mask is None:
        mask_arr = np.ones_like(y_true, dtype=bool)
    else:
        mask_arr = mask.cpu().numpy()

    y_true_masked = y_true[mask_arr]
    probs_masked = probs[mask_arr]
    preds_masked = preds[mask_arr]

    # Compute metrics safely
    try:
        auc = roc_auc_score(y_true_masked, probs_masked)
    except Exception:
        auc = float("nan")
    acc = accuracy_score(y_true_masked, preds_masked)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_masked, preds_masked, average="binary", zero_division=0
    )

    return {"auc": auc, "acc": acc, "precision": precision, "recall": recall, "f1": f1}


# -------------------------
# High-level Trainer
# -------------------------
def train_gnn(
    data: Data,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    hidden_channels: int = 128,
    out_channels: int = 2,
    device: Optional[str] = None,
    val_split_ratio: float = 0.2,
    save_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a GraphSAGE on the provided PyG Data object.
    - data.y must exist and contain integer labels (0/1)
    - splits are created by shuffling nodes (simple approach)
    Returns model and training history dict.
    """
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    n_nodes = data.x.shape[0]
    # Build simple random train/val split (mask tensors)
    perm = torch.randperm(n_nodes)
    val_count = int(n_nodes * val_split_ratio)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    model = GraphSAGE(in_channels=data.num_features, hidden_channels=hidden_channels, out_channels=out_channels)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_auc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion, device, train_mask)
        metrics = evaluate(model, data, device, val_mask)
        history["train_loss"].append(loss)
        history["val_auc"].append(metrics["auc"])
        history["val_acc"].append(metrics["acc"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={loss:.4f} | val_auc={metrics['auc']:.4f} | val_acc={metrics['acc']:.4f}")

    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model, history


# -------------------------
# Embedding extraction & inference
# -------------------------
def get_node_embeddings(model: nn.Module, data: Data, device: Optional[str] = None) -> np.ndarray:
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        _, embeddings = model(data.x, data.edge_index)
    return embeddings.cpu().numpy()


def predict_proba(model: nn.Module, data: Data, device: Optional[str] = None) -> np.ndarray:
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


# -------------------------
# Save / Load helpers
# -------------------------
def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, device: Optional[str] = None):
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# -------------------------
# Example usage (commented)
# -------------------------
# from Source.gnn_model import build_pyg_data, train_gnn, get_node_embeddings
# ds = EllipticDataset("Data/elliptic_bitcoin_dataset")  # your dataset loader
# G = ds.load()
# features = ...  # pd.DataFrame indexed by node id
# labels = ...    # pd.Series or dict node->label (0/1)
# data = build_pyg_data(G, features, labels)
# model, history = train_gnn(data, epochs=30, save_path="Outputs/graphsage.pt")
# embs = get_node_embeddings(model, data)
