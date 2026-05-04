"""
Reasoning Router — top-level dispatcher for Tier 1/2/3.

Given a routing decision + RAG docs + ML evidence, calls the appropriate
explainer and returns a structured ExplanationOutput.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from src.knowledge.chromadb_client import RetrievedDoc
from src.reasoning.llm_abstraction import FallbackChain, LLMProvider
from src.reasoning.llm_explainer import explain_with_llm
from src.reasoning.schemas import ExplanationOutput
from src.reasoning.slm_explainer import explain_with_slm
from src.reasoning.template_explainer import explain_with_template
from src.validation.confidence_router import RoutingDecision, Tier
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ReasoningRouter:
    """Dispatches to the right tier based on RoutingDecision."""

    def __init__(
        self,
        slm_provider: Optional[LLMProvider] = None,
        llm_provider_chain: Optional[FallbackChain] = None,
    ):
        self.slm_provider = slm_provider
        self.llm_provider_chain = llm_provider_chain

    def explain(
        self,
        routing_decision: RoutingDecision,
        transaction_id: str,
        individual_scores: Dict[str, float],
        ensemble_score: float,
        agreement_count: int,
        top_features: List[tuple],
        statistical_evidence: Dict,
        rag_docs: List[RetrievedDoc],
    ) -> ExplanationOutput:
        """Dispatch to the correct tier."""
        tier = routing_decision.tier

        if tier == Tier.PASS:
            return ExplanationOutput(
                typology_match=None,
                typology_confidence=0.0,
                supporting_evidence=[],
                contradicting_evidence=[],
                narrative_explanation="Transaction classified as normal — no flag.",
                recommended_action="monitor",
                rag_citations=[],
            )

        if tier == Tier.HIGH:
            return explain_with_template(
                transaction_id=transaction_id,
                ensemble_score=ensemble_score,
                agreement_count=agreement_count,
                top_features=top_features,
                rag_docs=rag_docs,
            )

        if tier == Tier.MEDIUM:
            if self.slm_provider is None:
                logger.warning("no_slm_provider_falling_back_to_template")
                return explain_with_template(
                    transaction_id, ensemble_score, agreement_count, top_features, rag_docs,
                )
            return explain_with_slm(
                provider=self.slm_provider,
                transaction_id=transaction_id,
                individual_scores=individual_scores,
                ensemble_score=ensemble_score,
                agreement_count=agreement_count,
                top_features=top_features,
                statistical_evidence=statistical_evidence,
                rag_docs=rag_docs,
            )

        # LOW or AMBIGUOUS → Tier 3
        if self.llm_provider_chain is None:
            logger.warning("no_llm_chain_falling_back_to_slm_or_template")
            if self.slm_provider:
                return explain_with_slm(
                    provider=self.slm_provider,
                    transaction_id=transaction_id,
                    individual_scores=individual_scores,
                    ensemble_score=ensemble_score,
                    agreement_count=agreement_count,
                    top_features=top_features,
                    statistical_evidence=statistical_evidence,
                    rag_docs=rag_docs,
                )
            return explain_with_template(
                transaction_id, ensemble_score, agreement_count, top_features, rag_docs,
            )

        return explain_with_llm(
            provider=self.llm_provider_chain,
            transaction_id=transaction_id,
            individual_scores=individual_scores,
            ensemble_score=ensemble_score,
            agreement_count=agreement_count,
            top_features=top_features,
            statistical_evidence=statistical_evidence,
            rag_docs=rag_docs,
        )
