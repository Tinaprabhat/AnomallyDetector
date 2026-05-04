"""
Tier 2 — SLM Explainer (Ollama qwen2.5:1.5b).

For MEDIUM-confidence cases. Local CPU. ~2-5s latency.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from src.knowledge.chromadb_client import RetrievedDoc
from src.reasoning._context_formatter import format_rag_context
from src.reasoning.llm_abstraction import LLMProvider, parse_json_response
from src.reasoning.schemas import ExplanationOutput
from src.utils.config import load_prompts_config


def _coerce_explanation(payload: Optional[Dict], rag_docs: List[RetrievedDoc]) -> ExplanationOutput:
    """Coerce raw LLM JSON into a valid ExplanationOutput, with safe defaults."""
    if not payload:
        return ExplanationOutput(
            typology_match="no_match",
            typology_confidence=0.0,
            supporting_evidence=[],
            contradicting_evidence=[],
            narrative_explanation="LLM did not return valid JSON. Recommend analyst review.",
            recommended_action="analyst_review",
            rag_citations=[d.id for d in rag_docs],
        )

    typology = payload.get("typology_match")
    if typology is not None:
        typology = str(typology)

    conf = payload.get("typology_confidence", 0.0)
    try:
        conf = float(conf)
        conf = min(max(conf, 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0

    action = payload.get("recommended_action", "analyst_review")
    if action not in ("auto_report", "analyst_review", "monitor"):
        action = "analyst_review"

    return ExplanationOutput(
        typology_match=typology,
        typology_confidence=conf,
        supporting_evidence=list(payload.get("supporting_evidence") or []),
        contradicting_evidence=list(payload.get("contradicting_evidence") or []),
        narrative_explanation=str(payload.get("narrative_explanation") or ""),
        recommended_action=action,
        rag_citations=list(payload.get("rag_citations") or [d.id for d in rag_docs]),
    )


def explain_with_slm(
    provider: LLMProvider,
    transaction_id: str,
    individual_scores: Dict[str, float],
    ensemble_score: float,
    agreement_count: int,
    top_features: List[tuple],
    statistical_evidence: Dict,
    rag_docs: List[RetrievedDoc],
) -> ExplanationOutput:
    """Tier 2 explanation using a local SLM."""
    prompts = load_prompts_config()
    sections = format_rag_context(rag_docs)

    user_prompt = prompts["tier_2_user_prompt"].format(
        transaction_id=transaction_id,
        ensemble_score=float(ensemble_score),
        agreement_count=int(agreement_count),
        top_features=", ".join(f"{n}={v:.3f}" for n, v in (top_features or [])) or "n/a",
        mahalanobis=float(statistical_evidence.get("mahalanobis_distance", 0.0)),
        p_value=float(statistical_evidence.get("mahalanobis_p_value", 1.0)),
        percentile=float(statistical_evidence.get("ensemble_percentile", 0.0)),
        typology_options=sections["typology_options"],
        case_history=sections["case_history"],
        regulations=sections["regulations"],
        personal_notes=sections["personal_notes"],
    )

    resp = provider.complete(
        system=prompts["system_prompt"],
        user=user_prompt,
        max_tokens=1024,
        temperature=0.1,
    )
    payload = parse_json_response(resp.text)
    return _coerce_explanation(payload, rag_docs)
