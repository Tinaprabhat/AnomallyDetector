"""Unit tests for the backtesting harness and metrics."""
from __future__ import annotations
import numpy as np
import pytest

from src.backtesting.cost_weighted import compute_cost, find_min_cost_threshold
from src.backtesting.drift_detector import detect_metric_drift, detect_distribution_drift
from src.backtesting.metrics import (
    FraudMetrics, compute_metrics, precision_at_k, precision_at_pct,
    find_optimal_threshold,
)
from src.backtesting.temporal_backtest import (
    BacktestWindow, TemporalBacktester, parse_windows,
)


class TestMetrics:

    def test_compute_metrics_basic(self):
        rng = np.random.default_rng(0)
        n = 1000
        labels = (rng.uniform(0, 1, n) > 0.95).astype(int)
        # Slightly correlated scores
        scores = rng.uniform(0, 1, n) * 0.5 + labels * 0.5
        m = compute_metrics(labels, scores)
        assert isinstance(m, FraudMetrics)
        assert m.pr_auc > 0.0
        assert 0.0 <= m.roc_auc <= 1.0
        assert m.n_samples == n

    def test_precision_at_k(self):
        labels = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        # Top 3: indices 0, 1, 2 → labels 1,0,1 → precision = 2/3
        p = precision_at_k(labels, scores, k=3)
        assert abs(p - 2/3) < 1e-9

    def test_precision_at_pct(self):
        labels = np.zeros(100); labels[:5] = 1
        scores = np.linspace(1, 0, 100)
        p = precision_at_pct(labels, scores, 0.05)
        assert p == 1.0

    def test_handles_all_zero_labels(self):
        labels = np.zeros(50, dtype=int)
        scores = np.random.RandomState(0).uniform(0, 1, 50)
        m = compute_metrics(labels, scores)
        assert m.n_positives == 0
        # Should not crash; returns zeros
        assert m.pr_auc == 0.0

    def test_find_optimal_threshold(self):
        rng = np.random.default_rng(1)
        n = 500
        labels = (rng.uniform(0, 1, n) > 0.9).astype(int)
        scores = rng.uniform(0, 1, n) * 0.3 + labels * 0.7
        t = find_optimal_threshold(labels, scores, metric="f1")
        assert 0.01 <= t <= 0.99


class TestCostWeighted:

    def test_cost_breakdown_correct(self):
        labels = np.array([1, 1, 0, 0, 0])
        preds = np.array([1, 0, 1, 0, 0])
        # TP=1, FN=1, FP=1, TN=2
        breakdown = compute_cost(labels, preds, cost_fn=100, cost_fp=1)
        assert breakdown.n_tp == 1
        assert breakdown.n_fn == 1
        assert breakdown.n_fp == 1
        assert breakdown.n_tn == 2
        assert breakdown.total_cost == 100 + 1   # 1 FN * 100 + 1 FP * 1

    def test_min_cost_threshold(self):
        rng = np.random.default_rng(2)
        n = 200
        labels = (rng.uniform(0, 1, n) > 0.9).astype(int)
        scores = rng.uniform(0, 1, n) * 0.4 + labels * 0.6
        t = find_min_cost_threshold(labels, scores, cost_fn=50, cost_fp=1)
        assert 0.01 <= t <= 0.99


class TestDriftDetection:

    def test_no_drift_when_stable(self):
        metrics = [
            FraudMetrics(pr_auc=0.8, roc_auc=0.85, precision_at_5pct=0.9,
                         precision_at_10pct=0.8, f1_at_optimal_threshold=0.7,
                         mcc_at_optimal_threshold=0.6, optimal_threshold=0.5,
                         n_samples=100, n_positives=10, base_rate=0.1)
            for _ in range(3)
        ]
        signals = detect_metric_drift(metrics, drop_threshold=0.10)
        assert all(not s.is_significant for s in signals)

    def test_drift_detected_on_drop(self):
        metrics = [
            FraudMetrics(pr_auc=0.8, roc_auc=0.85, precision_at_5pct=0.9,
                         precision_at_10pct=0.8, f1_at_optimal_threshold=0.7,
                         mcc_at_optimal_threshold=0.6, optimal_threshold=0.5,
                         n_samples=100, n_positives=10, base_rate=0.1),
            FraudMetrics(pr_auc=0.5, roc_auc=0.85, precision_at_5pct=0.9,
                         precision_at_10pct=0.8, f1_at_optimal_threshold=0.7,
                         mcc_at_optimal_threshold=0.6, optimal_threshold=0.5,
                         n_samples=100, n_positives=10, base_rate=0.1),
        ]
        signals = detect_metric_drift(metrics, drop_threshold=0.10)
        assert any(s.is_significant for s in signals)

    def test_distribution_drift(self):
        rng = np.random.default_rng(3)
        baseline = rng.normal(0, 1, 500)
        shifted = rng.normal(2, 1, 500)
        stat, p, sig = detect_distribution_drift(baseline, shifted)
        assert sig is True


class TestTemporalBacktester:

    def test_parse_windows(self):
        cfg = [
            {"train": [1, 25], "test": [26, 30]},
            {"train": [1, 30], "test": [31, 35]},
        ]
        windows = parse_windows(cfg)
        assert len(windows) == 2
        assert windows[0].train_start == 1
        assert windows[0].test_end == 30

    def test_run_with_dummy_train_score_fn(self):
        time_steps = {f"tx_{i}": (i % 49) + 1 for i in range(200)}
        bt = TemporalBacktester(time_steps=time_steps, cost_fn=100, cost_fp=1)

        rng = np.random.default_rng(4)

        def dummy_train_score(train_ids, test_ids):
            n = len(test_ids)
            labels = (rng.uniform(0, 1, n) > 0.9).astype(int)
            scores = rng.uniform(0, 1, n)
            return labels, scores

        windows = [
            BacktestWindow(1, 30, 31, 40),
            BacktestWindow(1, 40, 41, 49),
        ]
        report = bt.run(windows, dummy_train_score)
        assert len(report.windows) == 2
        assert "n_windows" in report.summary
