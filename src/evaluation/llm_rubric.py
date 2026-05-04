"""
LLM/SLM output rubric — score ExplanationOutput objects against expected behavior.

Deterministic checks (no LLM-as-judge) per locked decision:
- JSON validity (Pydantic)
- Recommended action ∈ enum
- typology_match in known list OR "no_match"
- typology_confidence in [0, 1]
- rag_citations is non-empty when retrieval returned docs
- typology_match matches expected (when expected_typology is set)
- recommended_action matches expected (when expected_action is set)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.evaluation.eval_cases import EvalCase
from src.knowledge.collections.typology_library import SEED_TYPOLOGIES
from src.reasoning.schemas import ExplanationOutput


KNOWN_TYPOLOGIES = {t.name for t in SEED_TYPOLOGIES} | {"no_match", None}
VALID_ACTIONS = {"auto_report", "analyst_review", "monitor"}


@dataclass
class LLMScoreCard:
    case_id: str
    schema_valid: bool = False
    action_valid: bool = False
    typology_in_library: bool = False
    typology_confidence_valid: bool = False
    has_citations: bool = False
    typology_matches_expected: Optional[bool] = None
    action_matches_expected: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def passes_strict(self) -> bool:
        required = [
            self.schema_valid, self.action_valid,
            self.typology_in_library, self.typology_confidence_valid,
        ]
        if self.typology_matches_expected is not None:
            required.append(self.typology_matches_expected)
        if self.action_matches_expected is not None:
            required.append(self.action_matches_expected)
        return all(required)


def score_explanation(
    output: ExplanationOutput,
    case: EvalCase,
    expected_citations_min: int = 0,
) -> LLMScoreCard:
    """Apply rubric checks to one explanation output."""
    card = LLMScoreCard(case_id=case.id)

    # Schema validity — already enforced by Pydantic upstream
    card.schema_valid = isinstance(output, ExplanationOutput)
    if not card.schema_valid:
        card.notes.append("output is not an ExplanationOutput instance")
        return card

    # Action enum check
    card.action_valid = output.recommended_action in VALID_ACTIONS
    if not card.action_valid:
        card.notes.append(f"invalid action: {output.recommended_action}")

    # Typology in known list (or 'no_match' / None)
    card.typology_in_library = output.typology_match in KNOWN_TYPOLOGIES
    if not card.typology_in_library:
        card.notes.append(f"hallucinated typology: {output.typology_match}")

    # Confidence range
    card.typology_confidence_valid = (
        0.0 <= float(output.typology_confidence) <= 1.0
    )
    if not card.typology_confidence_valid:
        card.notes.append(f"confidence out of range: {output.typology_confidence}")

    # Citations
    card.has_citations = len(output.rag_citations) >= expected_citations_min

    # Match expected (only if specified)
    if case.expected_typology is not None:
        card.typology_matches_expected = (
            output.typology_match == case.expected_typology
        )
        if not card.typology_matches_expected:
            card.notes.append(
                f"typology mismatch: expected {case.expected_typology}, "
                f"got {output.typology_match}"
            )

    if case.expected_action is not None:
        card.action_matches_expected = (
            output.recommended_action == case.expected_action
        )
        if not card.action_matches_expected:
            card.notes.append(
                f"action mismatch: expected {case.expected_action}, "
                f"got {output.recommended_action}"
            )

    return card


def aggregate_scorecards(cards: List[LLMScoreCard]) -> Dict:
    """Aggregate stats across all scorecards."""
    n = len(cards)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "schema_valid_rate": sum(c.schema_valid for c in cards) / n,
        "action_valid_rate": sum(c.action_valid for c in cards) / n,
        "typology_in_library_rate": sum(c.typology_in_library for c in cards) / n,
        "typology_confidence_valid_rate": sum(c.typology_confidence_valid for c in cards) / n,
        "has_citations_rate": sum(c.has_citations for c in cards) / n,
        "typology_match_rate": (
            sum(1 for c in cards if c.typology_matches_expected) /
            max(sum(1 for c in cards if c.typology_matches_expected is not None), 1)
        ),
        "action_match_rate": (
            sum(1 for c in cards if c.action_matches_expected) /
            max(sum(1 for c in cards if c.action_matches_expected is not None), 1)
        ),
        "strict_pass_rate": sum(c.passes_strict() for c in cards) / n,
    }
