"""
ChromaDB client wrapper.

Uniform interface for the 4 RAG collections. Uses sentence-transformers
(all-MiniLM-L6-v2, 384-dim) for embeddings — CPU-friendly.
Falls back to a hash-based stub if optional deps are missing,
so unit tests can still run.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import chromadb
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedDoc:
    id: str
    text: str
    metadata: Dict
    similarity: float
    collection: str


class _StubEmbedder:
    """Deterministic hash-based stub used when sentence-transformers isn't installed."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            n = np.linalg.norm(v) + 1e-9
            out[i] = v / n
        return out


class _InMemoryStubCollection:
    """Minimal Chroma-like collection for environments without chromadb."""

    def __init__(self, name: str, embedder):
        self.name = name
        self.embedder = embedder
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metas: List[Dict] = []
        self._embs: List[np.ndarray] = []

    def add(self, ids, documents, metadatas=None, embeddings=None):
        if isinstance(ids, str):
            ids = [ids]; documents = [documents]
            metadatas = [metadatas]; embeddings = [embeddings] if embeddings else None

        if embeddings is None:
            embs = self.embedder.encode(documents)
        else:
            embs = embeddings

        for i, doc, meta, emb in zip(
            ids, documents, metadatas or [{}] * len(ids), embs
        ):
            self._ids.append(i); self._texts.append(doc)
            self._metas.append(meta or {}); self._embs.append(np.asarray(emb))

    def query(self, query_texts=None, query_embeddings=None, n_results=3):
        if query_texts is not None:
            q_emb = self.embedder.encode(query_texts)
        else:
            q_emb = np.asarray(query_embeddings)
        q_emb = q_emb if q_emb.ndim == 2 else q_emb.reshape(1, -1)

        if not self._embs:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        embs = np.vstack(self._embs)
        denom = ((np.linalg.norm(embs, axis=1) + 1e-9)
                 * (np.linalg.norm(q_emb[0]) + 1e-9))
        sims = (embs @ q_emb[0]) / denom
        order = np.argsort(-sims)[:n_results]
        return {
            "ids": [[self._ids[i] for i in order]],
            "documents": [[self._texts[i] for i in order]],
            "metadatas": [[self._metas[i] for i in order]],
            "distances": [[float(1.0 - sims[i]) for i in order]],
        }

    def count(self):
        return len(self._ids)


class ChromaDBClient:
    """Wrapper around ChromaDB with all-MiniLM-L6-v2 embeddings."""

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self.embedder = self._init_embedder()
        self.client = self._init_client()
        self._collections: Dict[str, object] = {}

    def _init_embedder(self):
        if _HAS_SBERT:
            try:
                logger.info("loading_embedder", model=self.embedding_model_name)
                return SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                logger.warning("embedder_load_failed_using_stub", error=str(e))
        return _StubEmbedder(dim=384)

    def _init_client(self):
        if _HAS_CHROMA:
            try:
                return chromadb.PersistentClient(path=str(self.persist_directory))
            except Exception as e:
                logger.warning("chromadb_init_failed_using_stub", error=str(e))
        return None  # signals stub mode

    def get_or_create_collection(self, name: str):
        if name in self._collections:
            return self._collections[name]
        if self.client is not None:
            coll = self.client.get_or_create_collection(name=name)
        else:
            coll = _InMemoryStubCollection(name=name, embedder=self.embedder)
        self._collections[name] = coll
        return coll

    def embed_text(self, text):
        if isinstance(text, str):
            text = [text]
        return np.asarray(self.embedder.encode(text))

    def add(self, collection_name: str, doc_id: str, text: str, metadata: Dict) -> None:
        coll = self.get_or_create_collection(collection_name)
        safe_meta = metadata if metadata else {"_placeholder": "true"}
        if self.client is not None:
            emb = self.embed_text(text)[0].tolist()
            coll.add(ids=[doc_id], embeddings=[emb], documents=[text], metadatas=[safe_meta])
        else:
            coll.add(ids=[doc_id], documents=[text], metadatas=[safe_meta])

    def add_batch(
        self,
        collection_name: str,
        doc_ids: List[str],
        texts: List[str],
        metadatas: List[Dict],
    ) -> None:
        """Add multiple documents in a single encode + insert call."""
        if not doc_ids:
            return
        coll = self.get_or_create_collection(collection_name)
        safe_metas = [m if m else {"_placeholder": "true"} for m in metadatas]
        if self.client is not None:
            embs = self.embedder.encode(texts, show_progress_bar=False)
            coll.add(
                ids=doc_ids,
                embeddings=[e.tolist() for e in embs],
                documents=texts,
                metadatas=safe_metas,
            )
        else:
            coll.add(ids=doc_ids, documents=texts, metadatas=safe_metas)

    def query(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        n_results: int = 3,
    ) -> List[RetrievedDoc]:
        coll = self.get_or_create_collection(collection_name)
        if query_text is None and query_embedding is None:
            raise ValueError("Provide query_text or query_embedding")

        if self.client is not None:
            if query_embedding is not None:
                emb = np.asarray(query_embedding).flatten().tolist()
                res = coll.query(query_embeddings=[emb], n_results=n_results)
            else:
                emb = self.embed_text(query_text)[0].tolist()
                res = coll.query(query_embeddings=[emb], n_results=n_results)
        else:
            if query_embedding is not None:
                res = coll.query(query_embeddings=query_embedding, n_results=n_results)
            else:
                res = coll.query(query_texts=[query_text], n_results=n_results)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out = []
        for i, d, m, dist in zip(ids, docs, metas, dists):
            sim = max(0.0, 1.0 - float(dist))
            out.append(RetrievedDoc(
                id=i, text=d, metadata=m or {}, similarity=sim,
                collection=collection_name,
            ))
        return out

    def count(self, collection_name: str) -> int:
        coll = self.get_or_create_collection(collection_name)
        try:
            return int(coll.count())
        except Exception:
            return 0
