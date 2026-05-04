"""POST /detect — main detection endpoint, wired to demo pipeline."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    AuditTrail, DetectionResult, MLEvidence, StatisticalEvidence,
    TransactionInput, TypologyMatch,
)
from src.feedback.review_queue import ReviewQueue
from src.knowledge.chromadb_client import ChromaDBClient
from src.pipeline import detect_transaction, load_artifacts
from src.utils.config import load_default_config, resolve_path

router = APIRouter()

_cfg = load_default_config()


def _get_chroma() -> ChromaDBClient:
    return ChromaDBClient(persist_directory=resolve_path(_cfg["paths"]["chromadb"]))


def _get_queue() -> ReviewQueue:
    return ReviewQueue(db_path=resolve_path(_cfg["paths"]["audit_db"]))


@router.post("/detect", response_model=DetectionResult)
def detect(payload: TransactionInput) -> DetectionResult:
    if len(payload.raw_features) == 0:
        raise HTTPException(400, "raw_features cannot be empty")
    if not (1 <= payload.time_step <= 49):
        raise HTTPException(400, "time_step must be in [1, 49]")

    artifacts = load_artifacts()
    if artifacts is None:
        raise HTTPException(
            503,
            "Pipeline not yet trained. Run: python -m scripts.bootstrap",
        )

    result = detect_transaction(
        transaction_id=payload.transaction_id,
        raw_features=list(payload.raw_features),
        time_step=payload.time_step,
        artifacts=artifacts,
        chroma_client=_get_chroma(),
        review_queue=_get_queue(),
        reasoning_router=None,        # template-only by default
    )

    return DetectionResult(
        transaction_id=result.transaction_id,
        verdict=result.verdict,
        confidence_tier=result.confidence_tier,
        ml_evidence=MLEvidence(**result.ml_evidence),
        statistical_evidence=StatisticalEvidence(**result.statistical_evidence),
        typology_match=TypologyMatch(**result.typology_match),
        rag_citations=result.rag_citations,
        narrative_explanation=result.narrative_explanation,
        recommended_action=result.recommended_action,
        audit_trail=AuditTrail(
            tier=result.confidence_tier,
            timestamp=result.audit_trail.get("timestamp", ""),
            llm_provider_used=None,
        ),
    )
