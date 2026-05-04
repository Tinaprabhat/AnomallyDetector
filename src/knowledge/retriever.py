"""
Multi-collection retriever — queries all 4 collections in parallel.

Per locked design:
- typology_library: top 3, weight 1.5×
- case_history:     top 3, weight 1.0×
- regulatory:       top 2, weight 1.0×
- personal_notes:   top 2, weight 1.2×

Query construction is HYBRID (per locked decision): text description from
SHAP features + statistical signals.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.knowledge.chromadb_client import ChromaDBClient, RetrievedDoc
from src.utils.config import load_default_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    docs: List[RetrievedDoc]
    max_similarity_per_collection: Dict[str, float]
    rag_match_score: float    # max similarity overall — used for HIGH-tier check
    query_text: str


def build_query_text(
    transaction_id: str,
    ensemble_score: float,
    agreement_count: int,
    top_features: List[tuple],
    statistical_evidence: Dict,
) -> str:
    """
    Build a text description of a flagged transaction for RAG retrieval.

    Hybrid query: this text gets embedded; for case_history we also use the
    GNN embedding (handled in retrieve()).
    """
    feat_strs = [f"{name}={value:.3f}" for name, value in (top_features or [])]
    feat_part = f"top features: {'; '.join(feat_strs)}" if feat_strs else ""

    stat_parts = []
    if "mahalanobis_distance" in statistical_evidence:
        stat_parts.append(f"mahalanobis={statistical_evidence['mahalanobis_distance']:.3f}")
    if "ensemble_percentile" in statistical_evidence:
        stat_parts.append(f"percentile={statistical_evidence['ensemble_percentile']:.1f}")

    query = (
        f"flagged Bitcoin transaction {transaction_id} "
        f"ensemble_score={ensemble_score:.3f} agreement={agreement_count}/4 "
        f"{feat_part} {' '.join(stat_parts)}"
    ).strip()
    return query


class MultiCollectionRetriever:
    """Retrieves from all 4 RAG collections, returns reranked combined list."""

    def __init__(self, client: ChromaDBClient, rag_config: Optional[Dict] = None):
        self.client = client
        if rag_config is None:
            rag_config = load_default_config()["rag"]
        self.rag_config = rag_config
        self.collections_cfg = rag_config["collections"]

    def retrieve(
        self,
        query_text: str,
        gnn_embedding: Optional[np.ndarray] = None,
    ) -> RetrievalResult:
        """
        Retrieve from all 4 collections.

        gnn_embedding: optional GNN embedding used as additional query signal
                       for case_history (hybrid query).
        """
        all_docs: List[RetrievedDoc] = []
        max_sim_per_coll: Dict[str, float] = {}

        # typology_library — text query
        for coll_name in ["typology_library", "case_history", "regulatory", "personal_notes"]:
            cfg = self.collections_cfg.get(coll_name, {"top_k": 3, "weight": 1.0})
            try:
                docs = self.client.query(
                    collection_name=coll_name,
                    query_text=query_text,
                    n_results=cfg["top_k"],
                )
            except Exception as e:
                logger.warning("retrieval_failed", collection=coll_name, error=str(e))
                docs = []

            # Bonus: if case_history and we have GNN embedding, also query by embedding
            if coll_name == "case_history" and gnn_embedding is not None:
                try:
                    extra = self.client.query(
                        collection_name=coll_name,
                        query_embedding=gnn_embedding,
                        n_results=cfg["top_k"],
                    )
                    # Merge by id, keep higher similarity
                    seen = {d.id: d for d in docs}
                    for d in extra:
                        if d.id not in seen or d.similarity > seen[d.id].similarity:
                            seen[d.id] = d
                    docs = list(seen.values())
                except Exception as e:
                    logger.warning("hybrid_retrieval_failed", error=str(e))

            all_docs.extend(docs)
            max_sim_per_coll[coll_name] = max(
                (d.similarity for d in docs), default=0.0
            )

        rag_match_score = max(max_sim_per_coll.values(), default=0.0)

        return RetrievalResult(
            docs=all_docs,
            max_similarity_per_collection=max_sim_per_coll,
            rag_match_score=rag_match_score,
            query_text=query_text,
        )
