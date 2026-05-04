"""Unit tests for the statistical validation layer."""
from __future__ import annotations
import numpy as np
import pytest

from src.validation.ensemble_agreement import compute_agreement_stats
from src.validation.mahalanobis import MahalanobisValidator


class TestMahalanobisValidator:

    def test_fit_and_compute(self, random_features):
        validator = MahalanobisValidator()
        # Fit on first 100 rows as "licit" reference
        validator.fit(random_features[:100])
        d_sq = validator.compute(random_features[100:])
        assert len(d_sq) == 100
        assert (d_sq >= 0).all()

    def test_unfit_raises(self, random_features):
        validator = MahalanobisValidator()
        with pytest.raises(RuntimeError):
            validator.compute(random_features)

    def test_p_values_in_range(self, random_features):
        validator = MahalanobisValidator()
        validator.fit(random_features[:100])
        p = validator.p_value(random_features[100:])
        assert (p >= 0.0).all() and (p <= 1.0).all()

    def test_outlier_has_high_distance(self):
        # Generate Gaussian normal data
        rng = np.random.default_rng(42)
        normal = rng.standard_normal((200, 5))
        validator = MahalanobisValidator()
        validator.fit(normal)
        # Inject a far outlier
        outlier = np.array([[100.0, 100.0, 100.0, 100.0, 100.0]])
        d_normal = validator.compute(normal[:5]).mean()
        d_outlier = validator.compute(outlier)[0]
        assert d_outlier > d_normal * 10  # outlier should be much farther

    def test_validate_one(self):
        rng = np.random.default_rng(42)
        normal = rng.standard_normal((100, 5))
        validator = MahalanobisValidator()
        validator.fit(normal)
        stats = validator.validate_one(normal[0])
        assert stats.distance_squared >= 0.0
        assert 0.0 <= stats.p_value <= 1.0


class TestEnsembleAgreement:

    def test_all_below_threshold_zero_agreement(self):
        scores = {"xgboost": 0.1, "graphsage": 0.2,
                  "isolation_forest": 0.15, "autoencoder": 0.05}
        thresholds = {k: 0.5 for k in scores}
        stats = compute_agreement_stats(scores, thresholds)
        assert stats.agreement_count == 0

    def test_all_above_threshold_full_agreement(self):
        scores = {"xgboost": 0.9, "graphsage": 0.8,
                  "isolation_forest": 0.85, "autoencoder": 0.95}
        thresholds = {k: 0.5 for k in scores}
        stats = compute_agreement_stats(scores, thresholds)
        assert stats.agreement_count == 4

    def test_split_agreement(self):
        scores = {"xgboost": 0.9, "graphsage": 0.2,
                  "isolation_forest": 0.85, "autoencoder": 0.1}
        thresholds = {k: 0.5 for k in scores}
        stats = compute_agreement_stats(scores, thresholds)
        assert stats.agreement_count == 2

    def test_spread_computed_correctly(self):
        scores = {"xgboost": 0.9, "graphsage": 0.1,
                  "isolation_forest": 0.5, "autoencoder": 0.4}
        thresholds = {k: 0.5 for k in scores}
        stats = compute_agreement_stats(scores, thresholds)
        assert abs(stats.spread - 0.8) < 1e-9

    def test_empty_scores(self):
        stats = compute_agreement_stats({}, {})
        assert stats.agreement_count == 0
