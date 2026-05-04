"""GET /knowledge_base — inspect RAG collections."""
from __future__ import annotations
from fastapi import APIRouter

from src.knowledge.chromadb_client import ChromaDBClient
from src.utils.config import load_default_config, resolve_path

router = APIRouter()


@router.get("/knowledge_base")
def list_collections():
    """Return counts for each RAG collection."""
    cfg = load_default_config()
    client = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))
    return {
        "typology_library": client.count("typology_library"),
        "case_history": client.count("case_history"),
        "regulatory": client.count("regulatory"),
        "personal_notes": client.count("personal_notes"),
    }


@router.get("/knowledge_base/{collection}/search")
def search_collection(collection: str, q: str, k: int = 3):
    """Search a specific RAG collection."""
    cfg = load_default_config()
    client = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))
    docs = client.query(collection_name=collection, query_text=q, n_results=k)
    return {
        "collection": collection,
        "query": q,
        "results": [
            {"id": d.id, "similarity": d.similarity, "text": d.text, "metadata": d.metadata}
            for d in docs
        ],
    }
