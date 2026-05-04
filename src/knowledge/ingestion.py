"""
Knowledge base ingestion — orchestrates seeding of all 4 collections.

Run once at project setup (or any time you want to rebuild the KB):
    python -m src.knowledge.ingestion
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

from src.knowledge.chromadb_client import ChromaDBClient
from src.knowledge.collections.typology_library import seed_typology_library
from src.knowledge.collections.regulatory import seed_regulatory
from src.knowledge.collections.case_history import bootstrap_from_elliptic
from src.utils.config import load_default_config, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ingest_all(
    illicit_tx_ids: Optional[List[str]] = None,
    persist_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Seed all 4 RAG collections.

    typology_library: SEED_TYPOLOGIES (curated)
    regulatory: SEED_REGULATIONS (curated)
    case_history: bootstrap from illicit_tx_ids (optional)
    personal_notes: starts empty (grows from confirmed cases)

    Returns dict of collection_name -> n_added.
    """
    cfg = load_default_config()
    if persist_dir is None:
        persist_dir = resolve_path(cfg["paths"]["chromadb"])

    client = ChromaDBClient(
        persist_directory=persist_dir,
        embedding_model_name=cfg["rag"]["embedding_model"],
    )

    counts = {}
    counts["typology_library"] = seed_typology_library(client)
    counts["regulatory"] = seed_regulatory(client)
    if illicit_tx_ids:
        counts["case_history"] = bootstrap_from_elliptic(client, illicit_tx_ids)
    else:
        counts["case_history"] = 0
    counts["personal_notes"] = 0   # starts empty by design

    logger.info("knowledge_base_ingested", counts=counts)
    return counts


if __name__ == "__main__":
    print(ingest_all())
