"""Unit tests for the FastAPI service layer."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestAPIRoot:

    def test_root_returns_metadata(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "AnomalyDetector v2"
        assert "endpoints" in data

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestDetectEndpoint:

    def test_detect_valid_input(self, client):
        payload = {
            "transaction_id": "tx_test_001",
            "raw_features": [0.0] * 165,
            "time_step": 42,
        }
        r = client.post("/detect", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["transaction_id"] == "tx_test_001"
        assert data["confidence_tier"] in ["PASS", "HIGH", "MEDIUM", "LOW", "AMBIGUOUS"]

    def test_detect_empty_features_rejected(self, client):
        payload = {"transaction_id": "tx", "raw_features": [], "time_step": 1}
        r = client.post("/detect", json=payload)
        assert r.status_code == 400

    def test_detect_invalid_time_step_rejected(self, client):
        payload = {"transaction_id": "tx", "raw_features": [0.0], "time_step": 99}
        r = client.post("/detect", json=payload)
        assert r.status_code == 400


class TestKnowledgeBaseEndpoint:

    def test_list_collections(self, client):
        r = client.get("/knowledge_base")
        assert r.status_code == 200
        data = r.json()
        for k in ["typology_library", "case_history", "regulatory", "personal_notes"]:
            assert k in data


class TestBacktestEndpoint:

    def test_backtest_returns_status(self, client):
        r = client.post("/backtest", json={"cost_fn": 100.0, "cost_fp": 1.0})
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
