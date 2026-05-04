"""
Evaluation case set.

50 hand-crafted scenarios covering different routing tiers and typologies.
Each case has expected ranges so we can score the pipeline objectively.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EvalCase:
    """A single evaluation scenario with expected behavior."""
    id: str
    description: str
    individual_scores: Dict[str, float]      # raw [0,1] per model
    score_spread: float                      # max-min of scores (for AMBIGUOUS check)
    expected_tier: str                       # PASS|HIGH|MEDIUM|LOW|AMBIGUOUS
    expected_typology: Optional[str] = None  # known typology if applicable
    expected_action: Optional[str] = None    # auto_report|analyst_review|monitor
    rag_match_score: float = 0.0
    notes: str = ""


# Distribution: 10 PASS, 10 HIGH, 10 MEDIUM, 10 LOW, 10 AMBIGUOUS
EVAL_CASES: List[EvalCase] = []

# 10 PASS cases — clearly normal
for i in range(10):
    EVAL_CASES.append(EvalCase(
        id=f"EVAL-PASS-{i:02d}",
        description=f"Normal transaction with low scores across all models",
        individual_scores={
            "xgboost": 0.05 + i * 0.01,
            "graphsage": 0.10 + i * 0.005,
            "isolation_forest": 0.08,
            "autoencoder": 0.12,
        },
        score_spread=0.07,
        expected_tier="PASS",
        expected_action="monitor",
    ))

# 10 HIGH cases — strong unanimous agreement + RAG match
for i in range(10):
    EVAL_CASES.append(EvalCase(
        id=f"EVAL-HIGH-{i:02d}",
        description="All 4 detectors strongly agree, strong typology match",
        individual_scores={
            "xgboost": 0.92 + i * 0.005,
            "graphsage": 0.89,
            "isolation_forest": 0.87,
            "autoencoder": 0.85,
        },
        score_spread=0.07,
        expected_tier="HIGH",
        expected_typology="peel_chain" if i < 5 else "layering",
        expected_action="auto_report",
        rag_match_score=0.85,
    ))

# 10 MEDIUM cases — 3-of-4 agree
for i in range(10):
    EVAL_CASES.append(EvalCase(
        id=f"EVAL-MEDIUM-{i:02d}",
        description="3 of 4 detectors agree",
        individual_scores={
            "xgboost": 0.75,
            "graphsage": 0.68,
            "isolation_forest": 0.71,
            "autoencoder": 0.30,                     # AE disagrees
        },
        score_spread=0.45,
        expected_tier="MEDIUM",
        expected_typology="smurfing" if i < 5 else "mule_account",
        expected_action="analyst_review",
        rag_match_score=0.6,
    ))

# 10 LOW cases — 2-of-4 agree
for i in range(10):
    EVAL_CASES.append(EvalCase(
        id=f"EVAL-LOW-{i:02d}",
        description="Only 2 detectors agree",
        individual_scores={
            "xgboost": 0.65,
            "graphsage": 0.30,
            "isolation_forest": 0.62,
            "autoencoder": 0.28,
        },
        score_spread=0.37,
        expected_tier="LOW",
        expected_typology="mixing" if i < 5 else None,
        expected_action="analyst_review",
        rag_match_score=0.45,
    ))

# 10 AMBIGUOUS cases — high score disagreement
for i in range(10):
    EVAL_CASES.append(EvalCase(
        id=f"EVAL-AMBIG-{i:02d}",
        description="Detectors strongly disagree (spread > 0.5)",
        individual_scores={
            "xgboost": 0.95,
            "graphsage": 0.20,
            "isolation_forest": 0.85,
            "autoencoder": 0.18,
        },
        score_spread=0.77,
        expected_tier="AMBIGUOUS",
        expected_action="analyst_review",
        rag_match_score=0.3,
    ))


def get_cases_by_tier(tier: str) -> List[EvalCase]:
    return [c for c in EVAL_CASES if c.expected_tier == tier]


def get_case(case_id: str) -> Optional[EvalCase]:
    for c in EVAL_CASES:
        if c.id == case_id:
            return c
    return None
