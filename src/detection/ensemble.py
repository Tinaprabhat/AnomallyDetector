"""
Ensemble combiner — combines 4 model scores with Optuna-tuned weights.

Outputs:
1. Weighted continuous score: s_ensemble = Σ wᵢ · sᵢ
2. Agreement count: number of detectors that flagged (per-model thresholds)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score

try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleWeights:
    """Optuna-tuned weights and per-model thresholds."""
    w_xgboost: float = 0.25
    w_graphsage: float = 0.25
    w_isolation_forest: float = 0.25
    w_autoencoder: float = 0.25
    t_xgboost: float = 0.5
    t_graphsage: float = 0.5
    t_isolation_forest: float = 0.5
    t_autoencoder: float = 0.5

    def normalize_weights(self) -> "EnsembleWeights":
        s = (self.w_xgboost + self.w_graphsage
             + self.w_isolation_forest + self.w_autoencoder)
        if s <= 0:
            return EnsembleWeights()
        return EnsembleWeights(
            w_xgboost=self.w_xgboost / s,
            w_graphsage=self.w_graphsage / s,
            w_isolation_forest=self.w_isolation_forest / s,
            w_autoencoder=self.w_autoencoder / s,
            t_xgboost=self.t_xgboost,
            t_graphsage=self.t_graphsage,
            t_isolation_forest=self.t_isolation_forest,
            t_autoencoder=self.t_autoencoder,
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EnsembleResult:
    ensemble_score: np.ndarray
    agreement_count: np.ndarray
    individual_scores: Dict[str, np.ndarray]


def combine_scores(
    scores: Dict[str, np.ndarray],
    weights: EnsembleWeights,
) -> EnsembleResult:
    """Combine per-model scores into ensemble score + agreement count."""
    required = {"xgboost", "graphsage", "isolation_forest", "autoencoder"}
    missing = required - set(scores.keys())
    if missing:
        raise ValueError(f"Missing scores for models: {missing}")

    n = len(next(iter(scores.values())))
    for k, v in scores.items():
        if len(v) != n:
            raise ValueError(f"Score length mismatch for {k}")

    w = weights.normalize_weights()
    ensemble_score = (
        w.w_xgboost * scores["xgboost"]
        + w.w_graphsage * scores["graphsage"]
        + w.w_isolation_forest * scores["isolation_forest"]
        + w.w_autoencoder * scores["autoencoder"]
    )
    flagged = (
        (scores["xgboost"] >= weights.t_xgboost).astype(int)
        + (scores["graphsage"] >= weights.t_graphsage).astype(int)
        + (scores["isolation_forest"] >= weights.t_isolation_forest).astype(int)
        + (scores["autoencoder"] >= weights.t_autoencoder).astype(int)
    )
    return EnsembleResult(
        ensemble_score=ensemble_score,
        agreement_count=flagged,
        individual_scores=scores,
    )


def tune_ensemble_weights(
    val_scores: Dict[str, np.ndarray],
    val_labels: np.ndarray,
    n_trials: int = 50,
    metric: str = "pr_auc",
    seed: int = 42,
) -> EnsembleWeights:
    """Use Optuna to tune ensemble weights + thresholds for max PR-AUC."""
    if not _HAS_OPTUNA:
        logger.warning("optuna_unavailable_using_uniform_weights")
        return EnsembleWeights()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        w = EnsembleWeights(
            w_xgboost=trial.suggest_float("w_xgboost", 0.05, 1.0),
            w_graphsage=trial.suggest_float("w_graphsage", 0.05, 1.0),
            w_isolation_forest=trial.suggest_float("w_isolation_forest", 0.05, 1.0),
            w_autoencoder=trial.suggest_float("w_autoencoder", 0.05, 1.0),
            t_xgboost=trial.suggest_float("t_xgboost", 0.1, 0.9),
            t_graphsage=trial.suggest_float("t_graphsage", 0.1, 0.9),
            t_isolation_forest=trial.suggest_float("t_isolation_forest", 0.1, 0.9),
            t_autoencoder=trial.suggest_float("t_autoencoder", 0.1, 0.9),
        )
        result = combine_scores(val_scores, w)
        try:
            return float(average_precision_score(val_labels, result.ensemble_score))
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    weights = EnsembleWeights(
        w_xgboost=best["w_xgboost"],
        w_graphsage=best["w_graphsage"],
        w_isolation_forest=best["w_isolation_forest"],
        w_autoencoder=best["w_autoencoder"],
        t_xgboost=best["t_xgboost"],
        t_graphsage=best["t_graphsage"],
        t_isolation_forest=best["t_isolation_forest"],
        t_autoencoder=best["t_autoencoder"],
    ).normalize_weights()
    logger.info("ensemble_weights_tuned",
                best_score=study.best_value, weights=weights.to_dict())
    return weights
