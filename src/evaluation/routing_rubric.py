"""
Routing rubric — evaluate the ConfidenceRouter against the 50-case eval set.

For each case, computes agreement_count from individual_scores using the
configured thresholds, then checks whether the router assigns the expected tier.

Output:
- per-tier accuracy
- confusion matrix
- list of mismatched cases
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from src.evaluation.eval_cases import EVAL_CASES, EvalCase
from src.utils.config import load_routing_config
from src.validation.confidence_router import ConfidenceRouter


@dataclass
class RoutingEvalResult:
    total: int
    correct: int
    accuracy: float
    per_tier_accuracy: Dict[str, float]
    confusion: Dict[str, Dict[str, int]]   # expected -> {predicted -> count}
    mismatches: List[Dict]


def evaluate_routing(
    cases: List[EvalCase] = None,
    threshold: float = 0.5,
) -> RoutingEvalResult:
    """Run all eval cases through the router; score against expected_tier."""
    cases = cases or EVAL_CASES
    router = ConfidenceRouter(load_routing_config())

    correct = 0
    confusion: Dict[str, Dict[str, int]] = {}
    per_tier_correct: Dict[str, int] = {}
    per_tier_total: Dict[str, int] = {}
    mismatches: List[Dict] = []

    for case in cases:
        # Compute agreement_count using uniform threshold
        agreement = sum(
            1 for v in case.individual_scores.values() if v >= threshold
        )
        decision = router.route(
            agreement_count=agreement,
            score_spread=case.score_spread,
            rag_match_score=case.rag_match_score,
        )
        predicted = decision.tier.value
        expected = case.expected_tier

        per_tier_total[expected] = per_tier_total.get(expected, 0) + 1
        if predicted == expected:
            correct += 1
            per_tier_correct[expected] = per_tier_correct.get(expected, 0) + 1
        else:
            mismatches.append({
                "case_id": case.id,
                "expected": expected,
                "predicted": predicted,
                "agreement_count": agreement,
                "score_spread": case.score_spread,
                "rag_match_score": case.rag_match_score,
                "rationale": decision.rationale,
            })

        confusion.setdefault(expected, {}).setdefault(predicted, 0)
        confusion[expected][predicted] += 1

    per_tier_acc = {
        t: per_tier_correct.get(t, 0) / max(per_tier_total[t], 1)
        for t in per_tier_total
    }

    return RoutingEvalResult(
        total=len(cases),
        correct=correct,
        accuracy=correct / max(len(cases), 1),
        per_tier_accuracy=per_tier_acc,
        confusion=confusion,
        mismatches=mismatches,
    )
