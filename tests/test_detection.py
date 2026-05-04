"""Unit tests for the 4-model detection ensemble."""
from __future__ import annotations
import numpy as np
import pytest

from src.detection.ensemble import (
    EnsembleWeights, combine_scores, tune_ensemble_weights,
)
from src.detection.isolation_forest import IsolationForestDetector, IFConfig
from src.detection.xgboost_detector import XGBoostDetector, XGBoostConfig


class TestXGBoostDetector:
    def test_fit_predict_basic(self, random_features, random_labels):
        det = XGBoostDetector(XGBoostConfig(n_estimators=20, scale_pos_weight=10))
        det.fit(random_features, random_labels)
        proba = det.predict_proba(random_features)
        assert len(proba) == len(random_features)
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_unfit_raises(self, random_features):
        det = XGBoostDetector()
        with pytest.raises(RuntimeError):
            det.predict_proba(random_features)

    def test_predict_returns_binary(self, random_features, random_labels):
        det = XGBoostDetector(XGBoostConfig(n_estimators=20))
        det.fit(random_features, random_labels)
        preds = det.predict(random_features)
        assert set(np.unique(preds)).issubset({0, 1})


class TestIsolationForestDetector:
    """Critical: validates the v1 fix — continuous scores, not binary."""

    def test_score_returns_continuous(self, random_features):
        """v1 BUG: returned binary {0,1}. v2 must return continuous [0,1]."""
        det = IsolationForestDetector(IFConfig(n_estimators=30))
        det.fit(random_features)
        scores = det.score(random_features)
        unique = np.unique(scores)
        # Should NOT be just 2 values like v1
        assert len(unique) > 2, "Scores must be continuous, not binary"
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_unfit_raises(self, random_features):
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            det.score(random_features)

    def test_normalization_consistent(self, random_features):
        """Re-scoring same data twice should give same scores."""
        det = IsolationForestDetector(IFConfig(n_estimators=30))
        det.fit(random_features)
        s1 = det.score(random_features)
        s2 = det.score(random_features)
        np.testing.assert_array_almost_equal(s1, s2)


class TestEnsembleCombiner:

    def _scores(self, n: int = 10) -> dict:
        rng = np.random.default_rng(42)
        return {
            "xgboost": rng.uniform(0, 1, n),
            "graphsage": rng.uniform(0, 1, n),
            "isolation_forest": rng.uniform(0, 1, n),
            "autoencoder": rng.uniform(0, 1, n),
        }

    def test_combine_with_uniform_weights(self):
        scores = self._scores()
        w = EnsembleWeights()
        result = combine_scores(scores, w)
        assert len(result.ensemble_score) == 10
        assert result.ensemble_score.min() >= 0.0
        assert result.ensemble_score.max() <= 1.0

    def test_agreement_count_in_valid_range(self):
        scores = self._scores()
        w = EnsembleWeights()
        result = combine_scores(scores, w)
        assert result.agreement_count.min() >= 0
        assert result.agreement_count.max() <= 4

    def test_all_zeros_yield_zero_agreement(self):
        n = 5
        scores = {k: np.zeros(n) for k in
                  ["xgboost", "graphsage", "isolation_forest", "autoencoder"]}
        result = combine_scores(scores, EnsembleWeights())
        assert (result.agreement_count == 0).all()

    def test_all_ones_yield_full_agreement(self):
        n = 5
        scores = {k: np.ones(n) for k in
                  ["xgboost", "graphsage", "isolation_forest", "autoencoder"]}
        result = combine_scores(scores, EnsembleWeights())
        assert (result.agreement_count == 4).all()

    def test_missing_model_raises(self):
        scores = {"xgboost": np.zeros(5), "graphsage": np.zeros(5)}
        with pytest.raises(ValueError):
            combine_scores(scores, EnsembleWeights())

    def test_mismatched_lengths_raises(self):
        scores = {
            "xgboost": np.zeros(5), "graphsage": np.zeros(3),
            "isolation_forest": np.zeros(5), "autoencoder": np.zeros(5),
        }
        with pytest.raises(ValueError):
            combine_scores(scores, EnsembleWeights())

    def test_weights_normalize_to_one(self):
        w = EnsembleWeights(
            w_xgboost=2.0, w_graphsage=3.0,
            w_isolation_forest=1.0, w_autoencoder=4.0,
        )
        n = w.normalize_weights()
        s = n.w_xgboost + n.w_graphsage + n.w_isolation_forest + n.w_autoencoder
        assert abs(s - 1.0) < 1e-6

    def test_optuna_tuning_returns_valid_weights(self):
        rng = np.random.default_rng(0)
        n = 200
        scores = {
            "xgboost": rng.uniform(0, 1, n),
            "graphsage": rng.uniform(0, 1, n),
            "isolation_forest": rng.uniform(0, 1, n),
            "autoencoder": rng.uniform(0, 1, n),
        }
        labels = (rng.uniform(0, 1, n) > 0.95).astype(int)
        weights = tune_ensemble_weights(scores, labels, n_trials=5)
        assert isinstance(weights, EnsembleWeights)
        # weights normalized to ~1
        s = (weights.w_xgboost + weights.w_graphsage
             + weights.w_isolation_forest + weights.w_autoencoder)
        assert abs(s - 1.0) < 1e-3
