"""Unit tests for the RAG layer — ChromaDB client, retriever, reranker."""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

import pytest

from src.knowledge.chromadb_client import ChromaDBClient, RetrievedDoc
from src.knowledge.collections.typology_library import seed_typology_library, SEED_TYPOLOGIES
from src.knowledge.collections.regulatory import seed_regulatory, SEED_REGULATIONS
from src.knowledge.retriever import MultiCollectionRetriever, build_query_text
from src.knowledge.reranker import rerank


@pytest.fixture
def tmp_chroma():
    """Temporary ChromaDB instance, cleaned up after the test."""
    path = Path(tempfile.mkdtemp(prefix="chroma_test_"))
    client = ChromaDBClient(persist_directory=path)
    yield client
    shutil.rmtree(path, ignore_errors=True)


class TestChromaDBClient:

    def test_add_and_query(self, tmp_chroma):
        tmp_chroma.add(
            collection_name="test_coll",
            doc_id="doc1",
            text="bitcoin transaction layering pattern",
            metadata={"source": "test"},
        )
        docs = tmp_chroma.query(
            collection_name="test_coll",
            query_text="layering",
            n_results=3,
        )
        assert len(docs) >= 1
        assert docs[0].id == "doc1"
        assert docs[0].similarity >= 0.0

    def test_count(self, tmp_chroma):
        for i in range(5):
            tmp_chroma.add("col", f"d{i}", f"document {i}", {})
        assert tmp_chroma.count("col") == 5


class TestSeeding:

    def test_seed_typology_library(self, tmp_chroma):
        n = seed_typology_library(tmp_chroma)
        assert n == len(SEED_TYPOLOGIES) > 0
        assert tmp_chroma.count("typology_library") >= n

    def test_seed_regulatory(self, tmp_chroma):
        n = seed_regulatory(tmp_chroma)
        assert n == len(SEED_REGULATIONS) > 0


class TestQueryConstruction:

    def test_build_query_text_includes_features(self):
        q = build_query_text(
            transaction_id="tx_123",
            ensemble_score=0.91,
            agreement_count=4,
            top_features=[("feat_47", 0.5), ("feat_12", 0.3)],
            statistical_evidence={"mahalanobis_distance": 4.2, "ensemble_percentile": 99.0},
        )
        assert "tx_123" in q
        assert "0.91" in q or "0.910" in q
        assert "feat_47" in q

    def test_build_query_text_handles_missing_features(self):
        q = build_query_text("tx_1", 0.5, 2, [], {})
        assert "tx_1" in q


class TestMultiCollectionRetriever:

    def test_retrieves_from_seeded_collections(self, tmp_chroma):
        seed_typology_library(tmp_chroma)
        seed_regulatory(tmp_chroma)
        retriever = MultiCollectionRetriever(tmp_chroma)
        result = retriever.retrieve(
            query_text="rapid layering across multiple addresses",
        )
        assert len(result.docs) > 0
        # rag_match_score should reflect highest similarity
        assert 0.0 <= result.rag_match_score <= 1.0

    def test_max_similarity_per_collection_keys(self, tmp_chroma):
        seed_typology_library(tmp_chroma)
        retriever = MultiCollectionRetriever(tmp_chroma)
        result = retriever.retrieve(query_text="peel chain")
        assert "typology_library" in result.max_similarity_per_collection


class TestReranker:

    def test_drops_low_similarity(self):
        docs = [
            RetrievedDoc(id="a", text="x", metadata={}, similarity=0.9, collection="typology_library"),
            RetrievedDoc(id="b", text="x", metadata={}, similarity=0.2, collection="case_history"),
        ]
        result = rerank(docs)
        ids = [d.id for d in result]
        assert "a" in ids
        assert "b" not in ids   # below 0.4 threshold

    def test_dedup_by_id(self):
        docs = [
            RetrievedDoc(id="a", text="x", metadata={}, similarity=0.9, collection="typology_library"),
            RetrievedDoc(id="a", text="x", metadata={}, similarity=0.85, collection="typology_library"),
        ]
        result = rerank(docs)
        assert len(result) == 1

    def test_orders_by_weighted_similarity(self):
        docs = [
            # case_history with sim 0.7, weight 1.0 → weighted 0.7
            RetrievedDoc(id="c", text="x", metadata={}, similarity=0.7, collection="case_history"),
            # typology_library with sim 0.6, weight 1.5 → weighted 0.9
            RetrievedDoc(id="t", text="x", metadata={}, similarity=0.6, collection="typology_library"),
        ]
        result = rerank(docs)
        # typology should rank first due to weight
        assert result[0].id == "t"
