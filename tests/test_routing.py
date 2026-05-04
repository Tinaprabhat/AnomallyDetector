"""Unit tests for the confidence router — verifies all 5 tiers."""
from __future__ import annotations
import pytest

from src.validation.confidence_router import ConfidenceRouter, Tier


class TestConfidenceRouter:

    def setup_method(self):
        self.router = ConfidenceRouter()

    def test_pass_tier(self):
        """≤1 detector flagged → PASS."""
        decision = self.router.route(agreement_count=0, score_spread=0.1)
        assert decision.tier == Tier.PASS
        assert decision.use_llm is False
        assert decision.flag_for_human is False

        decision = self.router.route(agreement_count=1, score_spread=0.1)
        assert decision.tier == Tier.PASS

    def test_high_tier_requires_full_agreement_and_rag(self):
        """4 agree + RAG match >= 0.8 → HIGH."""
        decision = self.router.route(
            agreement_count=4, score_spread=0.1, rag_match_score=0.85,
        )
        assert decision.tier == Tier.HIGH
        assert decision.use_llm is False

    def test_high_tier_falls_back_without_strong_rag(self):
        """4 agree + weak RAG match → MEDIUM (not HIGH)."""
        decision = self.router.route(
            agreement_count=4, score_spread=0.1, rag_match_score=0.4,
        )
        assert decision.tier in (Tier.MEDIUM, Tier.LOW)

    def test_medium_tier(self):
        """3 agree → MEDIUM."""
        decision = self.router.route(agreement_count=3, score_spread=0.2,
                                     rag_match_score=0.6)
        assert decision.tier == Tier.MEDIUM
        assert decision.use_llm is True

    def test_low_tier(self):
        """2 agree → LOW."""
        decision = self.router.route(agreement_count=2, score_spread=0.2)
        assert decision.tier == Tier.LOW
        assert decision.use_llm is True

    def test_ambiguous_tier_overrides(self):
        """High score spread → AMBIGUOUS, regardless of agreement."""
        decision = self.router.route(agreement_count=2, score_spread=0.7)
        assert decision.tier == Tier.AMBIGUOUS
        assert decision.flag_for_human is True

    def test_ambiguous_with_high_agreement(self):
        """Even with 4 agreeing, large spread = AMBIGUOUS — but agreement_count + spread
        with 4 detectors above threshold typically have low spread, so spread overrides."""
        decision = self.router.route(
            agreement_count=4, score_spread=0.6, rag_match_score=0.9,
        )
        assert decision.tier == Tier.AMBIGUOUS

    def test_pass_overrides_ambiguous(self):
        """Even if spread is high, PASS check fires first if agreement ≤ 1."""
        decision = self.router.route(agreement_count=0, score_spread=0.9)
        assert decision.tier == Tier.PASS

    def test_decision_includes_rationale(self):
        decision = self.router.route(agreement_count=3, score_spread=0.2)
        assert decision.rationale and len(decision.rationale) > 0
