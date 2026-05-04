"""Unit tests for the evaluation harness — eval cases, rubrics, runner."""
from __future__ import annotations
import pytest

from src.evaluation.eval_cases import EVAL_CASES, get_case, get_cases_by_tier
from src.evaluation.llm_rubric import (
    KNOWN_TYPOLOGIES, VALID_ACTIONS, aggregate_scorecards, score_explanation,
)
from src.evaluation.routing_rubric import evaluate_routing
from src.reasoning.schemas import ExplanationOutput


class TestEvalCases:

    def test_total_count(self):
        # Per spec: 50 cases (10 per tier × 5 tiers)
        assert len(EVAL_CASES) == 50

    def test_each_tier_has_10_cases(self):
        for tier in ["PASS", "HIGH", "MEDIUM", "LOW", "AMBIGUOUS"]:
            cases = get_cases_by_tier(tier)
            assert len(cases) == 10, f"Expected 10 {tier} cases, got {len(cases)}"

    def test_get_case_by_id(self):
        case = get_case("EVAL-PASS-00")
        assert case is not None
        assert case.expected_tier == "PASS"

    def test_individual_scores_in_range(self):
        for case in EVAL_CASES:
            for k, v in case.individual_scores.items():
                assert 0.0 <= v <= 1.0, f"{case.id}: {k}={v} out of range"


class TestRoutingRubric:

    def test_routing_evaluation_runs(self):
        result = evaluate_routing()
        assert result.total == 50
        # Routing rules are deterministic — accuracy should be high
        assert result.accuracy >= 0.7, (
            f"Routing accuracy too low: {result.accuracy:.2%}. "
            f"Mismatches: {result.mismatches[:3]}"
        )

    def test_per_tier_accuracy_present(self):
        result = evaluate_routing()
        for tier in ["PASS", "HIGH", "MEDIUM", "LOW", "AMBIGUOUS"]:
            assert tier in result.per_tier_accuracy

    def test_confusion_matrix_complete(self):
        result = evaluate_routing()
        # Every expected tier should appear as a key in confusion
        for case in EVAL_CASES:
            assert case.expected_tier in result.confusion


class TestLLMRubric:

    def test_score_valid_explanation(self):
        case = get_case("EVAL-HIGH-00")
        out = ExplanationOutput(
            typology_match="peel_chain",
            typology_confidence=0.85,
            supporting_evidence=["evidence"],
            contradicting_evidence=[],
            narrative_explanation="text",
            recommended_action="auto_report",
            rag_citations=["TYPO-001"],
        )
        card = score_explanation(out, case)
        assert card.schema_valid
        assert card.action_valid
        assert card.typology_in_library
        assert card.typology_confidence_valid
        assert card.has_citations is True

    def test_score_invalid_typology_caught(self):
        case = get_case("EVAL-HIGH-00")
        out = ExplanationOutput(
            typology_match="completely_made_up_typology",
            typology_confidence=0.5,
            narrative_explanation="x",
            recommended_action="auto_report",
        )
        card = score_explanation(out, case)
        assert card.typology_in_library is False

    def test_aggregate_scorecards(self):
        cases = [get_case(f"EVAL-PASS-{i:02d}") for i in range(3)]
        cards = []
        for c in cases:
            out = ExplanationOutput(
                typology_match="no_match", typology_confidence=0.0,
                narrative_explanation="x", recommended_action="monitor",
            )
            cards.append(score_explanation(out, c))
        agg = aggregate_scorecards(cards)
        assert agg["n"] == 3
        assert "schema_valid_rate" in agg
        assert "strict_pass_rate" in agg

    def test_known_typologies_includes_seeds(self):
        assert "peel_chain" in KNOWN_TYPOLOGIES
        assert "no_match" in KNOWN_TYPOLOGIES

    def test_valid_actions_set(self):
        assert VALID_ACTIONS == {"auto_report", "analyst_review", "monitor"}
