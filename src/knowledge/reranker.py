"""
Reranker — applies collection-specific weights and prunes results.

Steps:
1. Apply collection weights (typology 1.5×, personal_notes 1.2×, others 1.0×)
2. Drop docs below min_similarity_threshold
3. Deduplicate (by id)
4. Take top N
"""
from __future__ import annotations
from typing import Dict, List, Optional

from src.knowledge.chromadb_client import RetrievedDoc
from src.utils.config import load_default_config


def rerank(
    docs: List[RetrievedDoc],
    rag_config: Optional[Dict] = None,
) -> List[RetrievedDoc]:
    """Rerank with collection weights and threshold."""
    if rag_config is None:
        rag_config = load_default_config()["rag"]

    collections_cfg = rag_config.get("collections", {})
    min_sim = float(rag_config.get("min_similarity_threshold", 0.4))
    max_final = int(rag_config.get("max_final_docs", 7))

    weighted = []
    for d in docs:
        cfg = collections_cfg.get(d.collection, {})
        weight = float(cfg.get("weight", 1.0))
        weighted_sim = d.similarity * weight
        weighted.append((weighted_sim, d))

    # Drop docs below threshold (use raw similarity for the threshold, not weighted)
    weighted = [(s, d) for s, d in weighted if d.similarity >= min_sim]

    # Sort by weighted score descending
    weighted.sort(key=lambda t: -t[0])

    # Deduplicate by id
    seen_ids = set()
    final = []
    for s, d in weighted:
        if d.id in seen_ids:
            continue
        seen_ids.add(d.id)
        final.append(d)
        if len(final) >= max_final:
            break

    return final
