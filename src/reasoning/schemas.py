"""
Pydantic schemas for LLM/SLM structured output.

Per Level 2 autonomy: LLM classifies typology and writes narrative,
but does not invent typologies or make detection decisions.

Falls back to dataclass-based validation if pydantic is unavailable in
the environment. In production (with pydantic installed), validation is strict;
the fallback provides reasonable defaults so unit tests can still run.
"""
from __future__ import annotations
from typing import List, Optional

try:
    from pydantic import BaseModel, Field, field_validator
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False


_VALID_ACTIONS = {"auto_report", "analyst_review", "monitor"}


if _HAS_PYDANTIC:

    class ExplanationOutput(BaseModel):
        """Output of an LLM/SLM explainer call."""
        typology_match: Optional[str] = Field(
            None, description="Typology name from typology_library, or 'no_match'."
        )
        typology_confidence: float = Field(
            ge=0.0, le=1.0, description="Confidence in the typology match."
        )
        supporting_evidence: List[str] = Field(default_factory=list)
        contradicting_evidence: List[str] = Field(default_factory=list)
        narrative_explanation: str = Field(...)
        recommended_action: str = Field(...)
        rag_citations: List[str] = Field(default_factory=list)

        @field_validator("recommended_action")
        @classmethod
        def _check_action(cls, v):
            if v not in _VALID_ACTIONS:
                raise ValueError(f"recommended_action must be one of {_VALID_ACTIONS}")
            return v

        def model_dump(self, **kwargs):
            return super().model_dump(**kwargs)

    class SelfReflectionOutput(BaseModel):
        """LLM self-reflection output after analyst confirmation."""
        topic: str
        key_lesson: str
        indicators_to_watch: List[str] = Field(default_factory=list)
        related_typologies: List[str] = Field(default_factory=list)

        def model_dump(self, **kwargs):
            return super().model_dump(**kwargs)

else:
    # Fallback: dataclass with manual validation
    from dataclasses import dataclass, field, asdict

    @dataclass
    class ExplanationOutput:
        narrative_explanation: str
        recommended_action: str
        typology_match: Optional[str] = None
        typology_confidence: float = 0.0
        supporting_evidence: List[str] = field(default_factory=list)
        contradicting_evidence: List[str] = field(default_factory=list)
        rag_citations: List[str] = field(default_factory=list)

        def __post_init__(self):
            if not (0.0 <= self.typology_confidence <= 1.0):
                raise ValueError(
                    f"typology_confidence must be in [0,1], got {self.typology_confidence}"
                )
            if self.recommended_action not in _VALID_ACTIONS:
                raise ValueError(
                    f"recommended_action must be one of {_VALID_ACTIONS}, "
                    f"got {self.recommended_action}"
                )

        def model_dump(self, **kwargs):
            return asdict(self)

    @dataclass
    class SelfReflectionOutput:
        topic: str
        key_lesson: str
        indicators_to_watch: List[str] = field(default_factory=list)
        related_typologies: List[str] = field(default_factory=list)

        def model_dump(self, **kwargs):
            return asdict(self)
