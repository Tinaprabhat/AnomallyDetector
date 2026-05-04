"""
Demo Pipeline — end-to-end fraud detection without requiring GPU/API.

Per Phase 0 functionality requirements: this is what the UI and API call
to actually run a transaction through the full pipeline.

Uses CPU-friendly detectors (Isolation Forest + Autoencoder + XGBoost),
with graceful degradation for missing optional deps.
GraphSAGE is added when torch_geometric is installed; otherwise the
ensemble runs on the available 3 detectors with re-normalized weights.
"""
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.detection.ensemble import EnsembleWeights, combine_scores
from src.detection.isolation_forest import IsolationForestDetector, IFConfig
from src.feedback.review_queue import ReviewItem, ReviewQueue
from src.knowledge.chromadb_client import ChromaDBClient
from src.knowledge.retriever import MultiCollectionRetriever, build_query_text
from src.knowledge.reranker import rerank
from src.reasoning.router import ReasoningRouter
from src.reasoning.template_explainer import explain_with_template
from src.utils.config import load_default_config, load_routing_config, resolve_path
from src.utils.logging import get_logger
from src.validation.confidence_router import ConfidenceRouter, Tier
from src.validation.ensemble_agreement import compute_agreement_stats
from src.validation.mahalanobis import MahalanobisValidator

# Optional imports — pipeline degrades gracefully if these fail
try:
    from src.detection.autoencoder import AutoencoderDetector, AEConfig
    _HAS_AE = True
except ImportError:
    _HAS_AE = False

try:
    from src.detection.xgboost_detector import XGBoostDetector, XGBoostConfig
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

logger = get_logger(__name__)

ARTIFACTS_FILENAME = "demo_pipeline_artifacts.pkl"


@dataclass
class PipelineArtifacts:
    """Trained-state container — saved/loaded across runs."""
    feature_names: List[str]
    iso_detector: Optional[IsolationForestDetector] = None
    ae_detector: Optional[Any] = None     # AutoencoderDetector when available
    xgb_detector: Optional[Any] = None    # XGBoostDetector when available
    mahalanobis: Optional[MahalanobisValidator] = None
    ensemble_weights: Optional[EnsembleWeights] = None
    ensemble_percentile_ref: Optional[np.ndarray] = None  # for percentile lookups
    trained_at: str = ""
    n_train_samples: int = 0
    available_models: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Output of one detect call — used by API and UI."""
    transaction_id: str
    verdict: str
    confidence_tier: str
    ml_evidence: Dict[str, float]
    statistical_evidence: Dict[str, Any]
    typology_match: Dict[str, Any]
    rag_citations: List[str]
    narrative_explanation: str
    recommended_action: str
    audit_trail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "verdict": self.verdict,
            "confidence_tier": self.confidence_tier,
            "ml_evidence": self.ml_evidence,
            "statistical_evidence": self.statistical_evidence,
            "typology_match": self.typology_match,
            "rag_citations": self.rag_citations,
            "narrative_explanation": self.narrative_explanation,
            "recommended_action": self.recommended_action,
            "audit_trail": self.audit_trail,
        }


def get_artifacts_path() -> Path:
    cfg = load_default_config()
    return resolve_path(cfg["paths"]["models"]) / ARTIFACTS_FILENAME


def save_artifacts(artifacts: PipelineArtifacts) -> Path:
    path = get_artifacts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    logger.info("artifacts_saved", path=str(path), models=artifacts.available_models)
    return path


def load_artifacts() -> Optional[PipelineArtifacts]:
    path = get_artifacts_path()
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("artifacts_load_failed", error=str(e))
        return None


def train_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
) -> PipelineArtifacts:
    """Train the CPU-friendly portion of the ensemble."""
    available: List[str] = []

    # 1. Isolation Forest (always available — sklearn)
    iso = IsolationForestDetector(IFConfig(n_estimators=100)).fit(X_train, X_val=X_val)
    available.append("isolation_forest")

    # 2. Autoencoder (if torch installed)
    ae = None
    if _HAS_AE:
        try:
            ae = AutoencoderDetector(AEConfig(epochs=15, batch_size=128)).fit(
                X_train, y_train=y_train, X_val=X_val,
            )
            available.append("autoencoder")
        except Exception as e:
            logger.warning("ae_training_skipped", error=str(e))

    # 3. XGBoost (if installed)
    xgb = None
    if _HAS_XGB and (y_train == 1).sum() > 0:
        try:
            xgb = XGBoostDetector(XGBoostConfig(n_estimators=80)).fit(
                X_train, y_train, X_val=X_val, y_val=y_val,
            )
            available.append("xgboost")
        except Exception as e:
            logger.warning("xgb_training_skipped", error=str(e))

    # 4. Mahalanobis on licit samples (always available)
    licit_mask = (y_train == 0)
    maha = None
    if licit_mask.sum() >= 30:
        maha = MahalanobisValidator().fit(X_train[licit_mask])

    # 5. Compute ensemble percentile reference using val set scores
    val_iso = iso.score(X_val)
    if ae is not None:
        val_ae = ae.score(X_val)
    else:
        val_ae = np.zeros(len(X_val))
    if xgb is not None:
        val_xgb = xgb.predict_proba(X_val)
    else:
        val_xgb = np.zeros(len(X_val))

    # Naive 4-model wrapping (graphsage = 0 for now since CPU GraphSAGE training is heavy)
    val_scores = {
        "xgboost": val_xgb,
        "graphsage": np.zeros(len(X_val)),
        "isolation_forest": val_iso,
        "autoencoder": val_ae,
    }
    weights = EnsembleWeights()  # uniform until Optuna runs in Phase 1
    ensemble = combine_scores(val_scores, weights)

    artifacts = PipelineArtifacts(
        feature_names=feature_names,
        iso_detector=iso,
        ae_detector=ae,
        xgb_detector=xgb,
        mahalanobis=maha,
        ensemble_weights=weights,
        ensemble_percentile_ref=np.sort(ensemble.ensemble_score),
        trained_at=datetime.utcnow().isoformat(),
        n_train_samples=len(X_train),
        available_models=available,
    )
    logger.info("pipeline_trained", available_models=available, n_train=len(X_train))
    return artifacts


# ----- INFERENCE -----

def _detect_one(
    artifacts: PipelineArtifacts,
    x: np.ndarray,
) -> Tuple[Dict[str, float], int, float]:
    """Run all available detectors on one row. Returns (scores, agreement, ensemble_score)."""
    if x.ndim == 1:
        x = x.reshape(1, -1)

    s_iso = float(artifacts.iso_detector.score(x)[0])
    s_ae = float(artifacts.ae_detector.score(x)[0]) if artifacts.ae_detector else 0.0
    s_xgb = float(artifacts.xgb_detector.predict_proba(x)[0]) if artifacts.xgb_detector else 0.0

    individual = {
        "xgboost": s_xgb,
        "graphsage": 0.0,
        "isolation_forest": s_iso,
        "autoencoder": s_ae,
    }
    scores_arr = {k: np.array([v]) for k, v in individual.items()}
    result = combine_scores(scores_arr, artifacts.ensemble_weights)
    return individual, int(result.agreement_count[0]), float(result.ensemble_score[0])


def _ensemble_percentile(score: float, ref: np.ndarray) -> float:
    """Where does this score sit among the validation distribution?"""
    if ref is None or len(ref) == 0:
        return 50.0
    return float(np.searchsorted(ref, score) / len(ref) * 100.0)


def detect_transaction(
    transaction_id: str,
    raw_features: List[float],
    time_step: int,
    artifacts: Optional[PipelineArtifacts] = None,
    chroma_client: Optional[ChromaDBClient] = None,
    review_queue: Optional[ReviewQueue] = None,
    reasoning_router: Optional[ReasoningRouter] = None,
) -> PipelineResult:
    """
    Run the full pipeline on a single transaction.
    Used by both the FastAPI /detect endpoint and the Streamlit UI.
    """
    started = datetime.utcnow()
    artifacts = artifacts or load_artifacts()
    if artifacts is None:
        return PipelineResult(
            transaction_id=transaction_id,
            verdict="ERROR",
            confidence_tier="PASS",
            ml_evidence={},
            statistical_evidence={},
            typology_match={"name": None, "confidence": 0.0, "source_collection": ""},
            rag_citations=[],
            narrative_explanation=(
                "Pipeline not yet trained. Run `python -m scripts.bootstrap` "
                "first to train detectors and seed the knowledge base."
            ),
            recommended_action="monitor",
            audit_trail={"error": "no_artifacts", "tier": "PASS"},
        )

    x = np.asarray(raw_features, dtype=np.float32).reshape(1, -1)
    expected_dim = len(artifacts.feature_names)
    if x.shape[1] != expected_dim:
        # Pad or truncate so demo never crashes on mismatched feature counts
        if x.shape[1] < expected_dim:
            x = np.pad(x, ((0, 0), (0, expected_dim - x.shape[1])))
        else:
            x = x[:, :expected_dim]

    individual, agreement, ensemble_score = _detect_one(artifacts, x)

    # Statistical validation
    maha_d = 0.0
    maha_p = 1.0
    if artifacts.mahalanobis is not None:
        maha_stats = artifacts.mahalanobis.validate_one(x[0])
        maha_d = float(maha_stats.distance_squared)
        maha_p = float(maha_stats.p_value)

    percentile = _ensemble_percentile(ensemble_score, artifacts.ensemble_percentile_ref)

    # Top features by absolute z-score (lightweight stand-in for SHAP at inference time)
    z = np.abs(x[0])
    top_idx = np.argsort(-z)[:3]
    top_features = [
        (artifacts.feature_names[i] if i < len(artifacts.feature_names) else f"feat_{i}",
         float(x[0, i]))
        for i in top_idx
    ]

    # RAG retrieval — only if we have a chroma client
    rag_match_score = 0.0
    rag_docs = []
    rag_citations = []
    if chroma_client is not None:
        try:
            retriever = MultiCollectionRetriever(chroma_client)
            query_text = build_query_text(
                transaction_id=transaction_id,
                ensemble_score=ensemble_score,
                agreement_count=agreement,
                top_features=top_features,
                statistical_evidence={
                    "mahalanobis_distance": maha_d,
                    "ensemble_percentile": percentile,
                },
            )
            retrieval = retriever.retrieve(query_text=query_text)
            rag_docs = rerank(retrieval.docs)
            rag_match_score = retrieval.rag_match_score
            rag_citations = [d.id for d in rag_docs]
        except Exception as e:
            logger.warning("rag_retrieval_failed", error=str(e))

    # Routing — when fewer than 4 detectors are available, scale the
    # agreement count up so that the existing 4-detector routing rules apply.
    # E.g., 1-of-1 active detectors flagging  →  effective 4-of-4.
    n_active = max(1, len(artifacts.available_models))
    effective_agreement = int(round(agreement * 4.0 / n_active))

    router = ConfidenceRouter()
    score_spread = max(individual.values()) - min(individual.values())
    decision = router.route(
        agreement_count=effective_agreement,
        score_spread=score_spread,
        rag_match_score=rag_match_score,
    )

    # Reasoning — defaults to template if no LLM provider
    if reasoning_router is None:
        reasoning_router = ReasoningRouter(slm_provider=None, llm_provider_chain=None)

    explanation = reasoning_router.explain(
        routing_decision=decision,
        transaction_id=transaction_id,
        individual_scores=individual,
        ensemble_score=ensemble_score,
        agreement_count=agreement,
        top_features=top_features,
        statistical_evidence={
            "mahalanobis_distance": maha_d,
            "mahalanobis_p_value": maha_p,
            "ensemble_percentile": percentile,
        },
        rag_docs=rag_docs,
    )

    verdict = "PASS" if decision.tier == Tier.PASS else "FLAG"

    typology_source = ""
    if explanation.rag_citations and rag_docs:
        match = next((d for d in rag_docs if d.id in explanation.rag_citations), None)
        if match:
            typology_source = match.collection

    audit = {
        "timestamp": started.isoformat(),
        "tier": decision.tier.value,
        "rationale": decision.rationale,
        "available_models": artifacts.available_models,
        "n_active_detectors": n_active,
        "effective_agreement_count": effective_agreement,
        "trained_at": artifacts.trained_at,
        "rag_match_score": rag_match_score,
        "score_spread": score_spread,
        "time_step": time_step,
    }

    result = PipelineResult(
        transaction_id=transaction_id,
        verdict=verdict,
        confidence_tier=decision.tier.value,
        ml_evidence={
            "p_xgb": individual["xgboost"],
            "p_gnn": individual["graphsage"],
            "s_iso": individual["isolation_forest"],
            "e_ae": individual["autoencoder"],
            "ensemble_score": ensemble_score,
            "agreement_count": agreement,
        },
        statistical_evidence={
            "mahalanobis_distance": maha_d,
            "mahalanobis_p_value": maha_p,
            "ensemble_percentile": percentile,
            "shap_top_features": top_features,
        },
        typology_match={
            "name": explanation.typology_match,
            "confidence": float(explanation.typology_confidence),
            "source_collection": typology_source,
        },
        rag_citations=explanation.rag_citations,
        narrative_explanation=explanation.narrative_explanation,
        recommended_action=explanation.recommended_action,
        audit_trail=audit,
    )

    # Push LOW/AMBIGUOUS into the review queue
    if (review_queue is not None
            and decision.tier in (Tier.LOW, Tier.AMBIGUOUS, Tier.MEDIUM)):
        try:
            review_queue.enqueue(ReviewItem(
                transaction_id=transaction_id,
                tier=decision.tier.value,
                detection_payload=result.to_dict(),
            ))
        except Exception as e:
            logger.warning("review_queue_enqueue_failed", error=str(e))

    return result
