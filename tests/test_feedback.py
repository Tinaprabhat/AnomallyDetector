"""Unit tests for the feedback layer — review queue, curator, analyst actions."""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

import pytest

from src.feedback.analyst_actions import AnalystAction, AnalystOrchestrator
from src.feedback.rag_curator import FalsePositiveLog, RAGCurator
from src.feedback.review_queue import ReviewItem, ReviewQueue
from src.knowledge.chromadb_client import ChromaDBClient


@pytest.fixture
def tmp_workspace():
    path = Path(tempfile.mkdtemp(prefix="feedback_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


class TestReviewQueue:

    def test_enqueue_and_get(self, tmp_workspace):
        queue = ReviewQueue(tmp_workspace / "review.db")
        queue.enqueue(ReviewItem(
            transaction_id="tx_001", tier="LOW",
            detection_payload={"score": 0.7},
        ))
        item = queue.get("tx_001")
        assert item is not None
        assert item.tier == "LOW"

    def test_list_pending_only(self, tmp_workspace):
        queue = ReviewQueue(tmp_workspace / "review.db")
        for i in range(3):
            queue.enqueue(ReviewItem(
                transaction_id=f"tx_{i}", tier="LOW", detection_payload={},
            ))
        queue.resolve("tx_0", "confirmed", typology_assigned="peel_chain")
        pending = queue.list_pending()
        assert len(pending) == 2
        assert all(p.status == "pending" for p in pending)

    def test_resolve(self, tmp_workspace):
        queue = ReviewQueue(tmp_workspace / "review.db")
        queue.enqueue(ReviewItem(
            transaction_id="tx_x", tier="LOW", detection_payload={},
        ))
        ok = queue.resolve("tx_x", "confirmed", "good catch", "peel_chain")
        assert ok is True
        item = queue.get("tx_x")
        assert item.status == "confirmed"
        assert item.typology_assigned == "peel_chain"

    def test_resolve_invalid_status_raises(self, tmp_workspace):
        queue = ReviewQueue(tmp_workspace / "review.db")
        with pytest.raises(ValueError):
            queue.resolve("tx_x", "invalid")

    def test_stats(self, tmp_workspace):
        queue = ReviewQueue(tmp_workspace / "review.db")
        for i in range(4):
            queue.enqueue(ReviewItem(transaction_id=f"tx_{i}", tier="LOW",
                                     detection_payload={}))
        queue.resolve("tx_0", "confirmed")
        queue.resolve("tx_1", "rejected")
        stats = queue.stats()
        assert stats.get("pending") == 2
        assert stats.get("confirmed") == 1
        assert stats.get("rejected") == 1


class TestFalsePositiveLog:

    def test_log_rejection(self, tmp_workspace):
        log = FalsePositiveLog(tmp_workspace / "fp.db")
        log.log_rejection("tx_fp", "DEX arbitrage", {"score": 0.7})
        # Just verify it doesn't crash; deeper checks would inspect SQLite directly


class TestRAGCurator:

    def test_on_confirmed_adds_to_case_history(self, tmp_workspace):
        chroma = ChromaDBClient(tmp_workspace / "chroma")
        fp_log = FalsePositiveLog(tmp_workspace / "fp.db")
        curator = RAGCurator(chroma_client=chroma, fp_log=fp_log, llm_provider=None)

        result = curator.on_confirmed(
            transaction_id="tx_conf",
            tier="LOW",
            detection_payload={"ml_evidence": {"ensemble_score": 0.8}},
            analyst_notes="confirmed peel chain",
            typology_assigned="peel_chain",
        )
        assert result["case_history_added"] is True
        # personal_note not added because no LLM provider
        assert result["personal_note_added"] is False

    def test_on_rejected_does_not_add_to_rag(self, tmp_workspace):
        chroma = ChromaDBClient(tmp_workspace / "chroma")
        fp_log = FalsePositiveLog(tmp_workspace / "fp.db")
        curator = RAGCurator(chroma_client=chroma, fp_log=fp_log, llm_provider=None)

        before = chroma.count("case_history")
        curator.on_rejected("tx_rej", "false positive", {"score": 0.3})
        after = chroma.count("case_history")
        assert after == before, "Rejected cases must NOT enter RAG"


class TestAnalystOrchestrator:

    def test_apply_confirmed(self, tmp_workspace):
        queue = ReviewQueue(tmp_workspace / "review.db")
        chroma = ChromaDBClient(tmp_workspace / "chroma")
        fp_log = FalsePositiveLog(tmp_workspace / "fp.db")
        curator = RAGCurator(chroma_client=chroma, fp_log=fp_log, llm_provider=None)
        orch = AnalystOrchestrator(queue=queue, curator=curator)

        queue.enqueue(ReviewItem(
            transaction_id="tx_orch", tier="LOW",
            detection_payload={"ml_evidence": {"ensemble_score": 0.6}},
        ))
        result = orch.apply(AnalystAction(
            transaction_id="tx_orch",
            decision="confirmed",
            typology_assigned="layering",
            notes="multi-hop pattern",
        ))
        assert result["ok"] is True
        assert queue.get("tx_orch").status == "confirmed"

    def test_apply_unknown_tx(self, tmp_workspace):
        queue = ReviewQueue(tmp_workspace / "review.db")
        chroma = ChromaDBClient(tmp_workspace / "chroma")
        fp_log = FalsePositiveLog(tmp_workspace / "fp.db")
        curator = RAGCurator(chroma_client=chroma, fp_log=fp_log, llm_provider=None)
        orch = AnalystOrchestrator(queue=queue, curator=curator)

        result = orch.apply(AnalystAction(
            transaction_id="tx_unknown",
            decision="confirmed", typology_assigned="x",
        ))
        assert result["ok"] is False
