"""
Tier 1 — Template Explainer (no LLM used).

For HIGH-confidence cases where the ML ensemble unanimously agrees and there's
a strong RAG match. Just fills in a template. <10ms latency.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from src.knowledge.chromadb_client import RetrievedDoc
from src.reasoning.schemas import ExplanationOutput


def explain_with_template(
    transaction_id: str,
    ensemble_score: float,
    agreement_count: int,
    top_features: List[tuple],
    rag_docs: List[RetrievedDoc],
) -> ExplanationOutput:
    """Build a deterministic template explanation."""
    # Pick best typology match from rag_docs (filtered to typology_library)
    typo_docs = [d for d in rag_docs if d.collection == "typology_library"]
    typology_name = "no_match"
    typology_conf = 0.0
    if typo_docs:
        best = max(typo_docs, key=lambda d: d.similarity)
        typology_name = str(best.metadata.get("name") or "no_match")
        typology_conf = float(best.similarity)

    feat_str = ", ".join(f"{n}={v:.3f}" for n, v in (top_features[:3] or []))

    narrative = (
        f"Transaction {transaction_id} flagged with HIGH confidence. "
        f"All {agreement_count} detectors agreed. Ensemble score {ensemble_score:.3f}. "
        f"Pattern matches '{typology_name}' typology (similarity {typology_conf:.2f}). "
        f"Top contributing features: {feat_str if feat_str else 'n/a'}. "
        f"Recommended action: auto-report to compliance."
    )

    return ExplanationOutput(
        typology_match=typology_name,
        typology_confidence=typology_conf,
        supporting_evidence=[
            f"{agreement_count}/4 detectors agreed",
            f"Ensemble score {ensemble_score:.3f}",
            f"RAG typology match {typology_conf:.2f}",
        ],
        contradicting_evidence=[],
        narrative_explanation=narrative,
        recommended_action="auto_report",
        rag_citations=[d.id for d in rag_docs],
    )
