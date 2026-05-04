"""
GraphSAGE GNN — salvaged from v1, refactored.

Improvements over v1:
- Class weights for imbalanced data (handled in trainer.py)
- Returns logits AND embeddings cleanly
- Clean separation of model definition vs training
- Graceful import fallback for environments without torch_geometric
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


if _HAS_TORCH and _HAS_PYG:

    class GraphSAGE(nn.Module):
        """2-layer GraphSAGE for node classification + embedding extraction."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 128,
            out_channels: int = 2,
            num_layers: int = 2,
            dropout: float = 0.5,
        ):
            super().__init__()
            assert num_layers >= 2
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.lin = nn.Linear(hidden_channels, out_channels)
            self.dropout = dropout
            self.hidden_channels = hidden_channels

        def forward(self, x, edge_index):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            logits = self.lin(x)
            return logits, x   # (logits, embeddings)
else:
    class GraphSAGE:  # stub
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GraphSAGE requires torch + torch_geometric. "
                "Install: pip install torch torch-geometric"
            )
