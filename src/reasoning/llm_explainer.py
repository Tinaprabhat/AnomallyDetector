"""
Tier 3 — LLM Explainer (Mistral primary + fallback chain).

For LOW and AMBIGUOUS cases. ~1-3s latency via API.
"""
from __future__ import annotations
from typing import Dict, List

from src.knowledge.chromadb_client import RetrievedDoc
from src.reasoning._context_formatter import format_rag_context
from src.reasoning.llm_abstraction import LLMProvider, parse_json_response
from src.reasoning.schemas import ExplanationOutput
from src.reasoning.slm_explainer import _coerce_explanation
from src.utils.config import load_prompts_config


def explain_with_llm(
    provider: LLMProvider,
    transaction_id: str,
    individual_scores: Dict[str, float],
    ensemble_score: float,
    agreement_count: int,
    top_features: List[tuple],
    statistical_evidence: Dict,
    rag_docs: List[RetrievedDoc],
) -> ExplanationOutput:
    """Tier 3 explanation using a more capable LLM."""
    prompts = load_prompts_config()
    sections = format_rag_context(rag_docs)

    user_prompt = prompts["tier_3_user_prompt"].format(
        transaction_id=transaction_id,
        p_xgb=float(individual_scores.get("xgboost", 0.0)),
        p_gnn=float(individual_scores.get("graphsage", 0.0)),
        s_iso=float(individual_scores.get("isolation_forest", 0.0)),
        e_ae=float(individual_scores.get("autoencoder", 0.0)),
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
