"""POST /feedback — submit analyst confirmation/rejection."""
from __future__ import annotations
from fastapi import APIRouter

from src.api.schemas import FeedbackInput
from src.feedback.analyst_actions import AnalystAction, AnalystOrchestrator
from src.feedback.rag_curator import FalsePositiveLog, RAGCurator
from src.feedback.review_queue import ReviewQueue
from src.knowledge.chromadb_client import ChromaDBClient
from src.utils.config import load_default_config, resolve_path

router = APIRouter()


def _build_orchestrator() -> AnalystOrchestrator:
    cfg = load_default_config()
    queue = ReviewQueue(db_path=resolve_path(cfg["paths"]["audit_db"]))
    chroma = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))
    fp_log = FalsePositiveLog(db_path=resolve_path("artifacts/false_positives.db"))
    curator = RAGCurator(chroma_client=chroma, fp_log=fp_log, llm_provider=None)
    return AnalystOrchestrator(queue=queue, curator=curator)


@router.post("/feedback")
def submit_feedback(payload: FeedbackInput):
    orchestrator = _build_orchestrator()
    action = AnalystAction(
        transaction_id=payload.transaction_id,
        decision=payload.decision,
        typology_assigned=payload.typology_assigned,
        notes=payload.notes,
    )
    return orchestrator.apply(action)
