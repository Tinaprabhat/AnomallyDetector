"""
Graph features computed from the transaction graph.

NOTE: Per locked design — these are NOT fed to XGBoost. GraphSAGE handles
graph structure end-to-end. These features are used for: statistical validation,
RAG query construction, and human review display.
"""
from __future__ import annotations
from typing import Iterable

import networkx as nx
import pandas as pd


def compute_basic_graph_features(graph: nx.Graph, tx_ids: Iterable[str]) -> pd.DataFrame:
    """Returns DataFrame indexed by txId with cols: degree, clustering_coef."""
    tx_list = [str(t) for t in tx_ids]
    cluster = nx.clustering(graph, nodes=[t for t in tx_list if t in graph])
    rows = []
    for t in tx_list:
        if t in graph:
            rows.append({
                "txId": t,
                "degree": graph.degree(t),
                "clustering_coef": cluster.get(t, 0.0),
            })
        else:
            rows.append({"txId": t, "degree": 0, "clustering_coef": 0.0})
    return pd.DataFrame(rows).set_index("txId")


def compute_pagerank(graph: nx.Graph, alpha: float = 0.85) -> pd.Series:
    if graph.number_of_nodes() == 0:
        return pd.Series(dtype=float)
    pr = nx.pagerank(graph, alpha=alpha)
    return pd.Series(pr, name="pagerank")


def compute_extended_graph_features(graph: nx.Graph, tx_ids: Iterable[str]) -> pd.DataFrame:
    """Extended features: degree, clustering, pagerank, triangles."""
    basic = compute_basic_graph_features(graph, tx_ids)
    pr = compute_pagerank(graph)
    triangles = {}
    for t in basic.index:
        if t in graph:
            try:
                triangles[t] = nx.triangles(graph, t)
            except Exception:
                triangles[t] = 0
        else:
            triangles[t] = 0
    basic["pagerank"] = basic.index.map(lambda x: pr.get(x, 0.0))
    basic["triangles"] = basic.index.map(lambda x: triangles.get(x, 0))
    return basic
