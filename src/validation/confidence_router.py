"""
Confidence Router — applies routing rules to ensemble outputs.

Tiers (per locked spec):
- PASS:      ≤1 detector flagged → no further work
- HIGH:      4 detectors agree + RAG match ≥ 0.8 → Template (no LLM)
- MEDIUM:    ≥3 agree → Tier 2 SLM
- LOW:       ≥2 agree → Tier 3 LLM
- AMBIGUOUS: high score spread → Tier 3 + human flag
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from src.utils.config import load_routing_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Tier(str, Enum):
    PASS = "PASS"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class RoutingDecision:
    tier: Tier
    use_llm: bool
    llm_provider: Optional[str]
    flag_for_human: bool
    rationale: str


class ConfidenceRouter:
    """Apply routing rules to ensemble outputs."""

    def __init__(self, routing_config: Optional[Dict] = None):
        self.config = routing_config or load_routing_config()
        self.tiers_cfg = self.config["tiers"]

    def route(
        self,
        agreement_count: int,
        score_spread: float,
        rag_match_score: float = 0.0,
    ) -> RoutingDecision:
        # PASS first
        pass_max = self.tiers_cfg["PASS"]["rules"].get("agreement_count_max", 1)
        if agreement_count <= pass_max:
            return RoutingDecision(
                tier=Tier.PASS, use_llm=False, llm_provider=None,
                flag_for_human=False,
                rationale=f"Agreement count {agreement_count} ≤ {pass_max} → PASS",
            )

        # AMBIGUOUS — score spread is high (overrides HIGH/MED/LOW)
        ambig_min = self.tiers_cfg["AMBIGUOUS"]["rules"].get("score_disagreement_min", 0.5)
        if score_spread >= ambig_min:
            return RoutingDecision(
                tier=Tier.AMBIGUOUS, use_llm=True,
                llm_provider=self.tiers_cfg["AMBIGUOUS"].get("llm_provider", "mistral_primary"),
                flag_for_human=True,
                rationale=f"Score spread {score_spread:.3f} ≥ {ambig_min} → AMBIGUOUS",
            )

        # HIGH check
        high = self.tiers_cfg["HIGH"]["rules"]
        if (agreement_count >= high.get("agreement_count_min", 4)
                and rag_match_score >= high.get("rag_match_score_min", 0.8)):
            return RoutingDecision(
                tier=Tier.HIGH, use_llm=False, llm_provider=None,
                flag_for_human=False,
                rationale=f"All {agreement_count} agree + RAG {rag_match_score:.2f} → HIGH",
            )

        # MEDIUM
        med = self.tiers_cfg["MEDIUM"]["rules"]
        if agreement_count >= med.get("agreement_count_min", 3):
            return RoutingDecision(
                tier=Tier.MEDIUM, use_llm=True,
                llm_provider=self.tiers_cfg["MEDIUM"].get("llm_provider", "ollama_slm"),
                flag_for_human=False,
                rationale=f"Agreement {agreement_count} ≥ 3 → MEDIUM (Tier 2)",
            )

        # LOW
        low = self.tiers_cfg["LOW"]["rules"]
        if agreement_count >= low.get("agreement_count_min", 2):
            return RoutingDecision(
                tier=Tier.LOW, use_llm=True,
                llm_provider=self.tiers_cfg["LOW"].get("llm_provider", "mistral_primary"),
                flag_for_human=False,
                rationale=f"Agreement {agreement_count} ≥ 2 → LOW (Tier 3)",
            )

        return RoutingDecision(
            tier=Tier.PASS, use_llm=False, llm_provider=None,
            flag_for_human=False, rationale="No tier matched — defaulting to PASS",
        )
