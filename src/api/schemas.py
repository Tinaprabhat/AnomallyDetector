"""Pydantic schemas for the FastAPI service layer.

Falls back to dataclasses if pydantic isn't installed (so tests can run
in lighter environments).
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False


if _HAS_PYDANTIC:

    class TransactionInput(BaseModel):
        transaction_id: str
        raw_features: List[float] = Field(..., description="The 165 Elliptic features")
        time_step: int

    class MLEvidence(BaseModel):
        p_xgb: float = 0.0
        p_gnn: float = 0.0
        s_iso: float = 0.0
        e_ae: float = 0.0
        ensemble_score: float = 0.0
        agreement_count: int = 0

    class StatisticalEvidence(BaseModel):
        mahalanobis_distance: float = 0.0
        mahalanobis_p_value: float = 1.0
        ensemble_percentile: float = 0.0
        shap_top_features: List[Any] = Field(default_factory=list)

    class TypologyMatch(BaseModel):
        name: Optional[str] = None
        confidence: float = 0.0
        source_collection: str = ""

    class AuditTrail(BaseModel):
        timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
        model_versions: Dict[str, str] = Field(default_factory=dict)
        latency_ms: Dict[str, float] = Field(default_factory=dict)
        llm_provider_used: Optional[str] = None
        tier: str = "PASS"

    class DetectionResult(BaseModel):
        transaction_id: str
        verdict: str
        confidence_tier: str
        ml_evidence: MLEvidence
        statistical_evidence: StatisticalEvidence
        typology_match: TypologyMatch
        rag_citations: List[str] = Field(default_factory=list)
        narrative_explanation: str = ""
        recommended_action: str = "monitor"
        audit_trail: AuditTrail = Field(default_factory=AuditTrail)

    class FeedbackInput(BaseModel):
        transaction_id: str
        decision: str
        typology_assigned: str = ""
        notes: str = ""

    class BacktestRequest(BaseModel):
        cost_fn: float = 100.0
        cost_fp: float = 1.0
        use_default_windows: bool = True

else:
    from dataclasses import dataclass, field, asdict

    @dataclass
    class TransactionInput:
        transaction_id: str
        raw_features: List[float]
        time_step: int

    @dataclass
    class MLEvidence:
        p_xgb: float = 0.0
        p_gnn: float = 0.0
        s_iso: float = 0.0
        e_ae: float = 0.0
        ensemble_score: float = 0.0
        agreement_count: int = 0

    @dataclass
    class StatisticalEvidence:
        mahalanobis_distance: float = 0.0
        mahalanobis_p_value: float = 1.0
        ensemble_percentile: float = 0.0
        shap_top_features: List[Any] = field(default_factory=list)

    @dataclass
    class TypologyMatch:
        name: Optional[str] = None
        confidence: float = 0.0
        source_collection: str = ""

    @dataclass
    class AuditTrail:
        timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
        model_versions: Dict[str, str] = field(default_factory=dict)
        latency_ms: Dict[str, float] = field(default_factory=dict)
        llm_provider_used: Optional[str] = None
        tier: str = "PASS"

    @dataclass
    class DetectionResult:
        transaction_id: str
        verdict: str
        confidence_tier: str
        ml_evidence: MLEvidence
        statistical_evidence: StatisticalEvidence
        typology_match: TypologyMatch
        rag_citations: List[str] = field(default_factory=list)
        narrative_explanation: str = ""
        recommended_action: str = "monitor"
        audit_trail: AuditTrail = field(default_factory=AuditTrail)

    @dataclass
    class FeedbackInput:
        transaction_id: str
        decision: str
        typology_assigned: str = ""
        notes: str = ""

    @dataclass
    class BacktestRequest:
        cost_fn: float = 100.0
        cost_fp: float = 1.0
        use_default_windows: bool = True
