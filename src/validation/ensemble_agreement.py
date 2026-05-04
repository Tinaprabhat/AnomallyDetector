"""
Ensemble agreement validator.

Computes agreement count + score spread for a single transaction. Spread is
used by the AMBIGUOUS tier check.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class AgreementStats:
    agreement_count: int           # 0..4
    max_score: float
    min_score: float
    spread: float                  # max - min
    individual_scores: Dict[str, float]


def compute_agreement_stats(
    individual_scores: Dict[str, float],
    thresholds: Dict[str, float],
) -> AgreementStats:
    """Per-transaction agreement stats."""
    flagged_count = sum(
        1 for k, v in individual_scores.items()
        if v >= thresholds.get(k, 0.5)
    )
    vals = list(individual_scores.values())
    if not vals:
        return AgreementStats(0, 0.0, 0.0, 0.0, {})
    max_v = max(vals); min_v = min(vals)
    return AgreementStats(
        agreement_count=flagged_count,
        max_score=max_v,
        min_score=min_v,
        spread=max_v - min_v,
        individual_scores=individual_scores,
    )
