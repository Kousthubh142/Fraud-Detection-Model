"""
tests/test_api.py

FastAPI endpoint tests using TestClient.
Model is mocked — no MLflow server or GPU needed.

Run with:  pytest tests/test_api.py -v
"""

import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

# ── Mock the predictor BEFORE importing the app ───────────────────────────────
# This prevents the real model from loading during tests

mock_predictor = MagicMock()
mock_predictor.model_version = "1"
mock_predictor.model_stage   = "Production"
mock_predictor.threshold     = 0.5
mock_predictor.model         = MagicMock()

SAMPLE_RESULT = {
    "fraud_probability": 0.9231,
    "is_fraud":          True,
    "risk_level":        "HIGH",
    "threshold_used":    0.5,
    "top_features": [
        {"feature": "V14", "impact": -0.842, "value": -5.123},
        {"feature": "V17", "impact": -0.631, "value": -3.456},
        {"feature": "V12", "impact": -0.419, "value": -2.789},
        {"feature": "V10", "impact": -0.312, "value": -1.234},
        {"feature": "log_amount", "impact": 0.201, "value": 4.567},
    ],
    "latency_ms":    12.4,
    "model_version": "1",
}

LEGIT_RESULT = {**SAMPLE_RESULT,
                "fraud_probability": 0.02,
                "is_fraud": False,
                "risk_level": "LOW"}

mock_predictor.predict.return_value       = SAMPLE_RESULT
mock_predictor.predict_batch.return_value = [SAMPLE_RESULT, LEGIT_RESULT]

with patch("api.predictor.predictor", mock_predictor):
    with patch("api.main.predictor", mock_predictor):
        from fastapi.testclient import TestClient
        from api.main import app

client = TestClient(app)


# ── Sample transaction payload ────────────────────────────────────────────────

SAMPLE_TXN = {
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,  "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624,  "V7": 0.2396,  "V8": 0.0987,
    "V9": 0.3638,  "V10": 0.0908, "V11": -0.5516,"V12": -0.6178,
    "V13": -0.9914,"V14": -0.3112,"V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183,"V22": 0.2778, "V23": -0.1105,"V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891,"V27": 0.1336, "V28": -0.0211,
    "Amount": 149.62,
    "Time": 0.0,
}


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:

    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_returns_model_version(self):
        response = client.get("/health")
        assert "model_version" in response.json()

    def test_returns_model_stage(self):
        response = client.get("/health")
        assert response.json()["model_stage"] == "Production"


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:

    def test_returns_200(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        assert response.status_code == 200

    def test_returns_fraud_probability(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        data     = response.json()
        assert "fraud_probability" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_returns_is_fraud_bool(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        data     = response.json()
        assert "is_fraud" in data
        assert isinstance(data["is_fraud"], bool)

    def test_returns_risk_level(self):
        response  = client.post("/predict", json=SAMPLE_TXN)
        risk      = response.json()["risk_level"]
        assert risk in ["LOW", "MEDIUM", "HIGH"]

    def test_returns_top_features(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        features = response.json()["top_features"]
        assert len(features) == 5
        for f in features:
            assert "feature" in f
            assert "impact"  in f
            assert "value"   in f

    def test_returns_latency_ms(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        assert "latency_ms" in response.json()
        assert response.json()["latency_ms"] > 0

    def test_returns_model_version(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        assert "model_version" in response.json()

    def test_returns_threshold_used(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        assert "threshold_used" in response.json()

    def test_missing_field_returns_422(self):
        bad_payload = {k: v for k, v in SAMPLE_TXN.items() if k != "Amount"}
        response    = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_negative_amount_returns_422(self):
        bad_payload = {**SAMPLE_TXN, "Amount": -50.0}
        response    = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_high_probability_flagged_as_fraud(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        data     = response.json()
        if data["fraud_probability"] >= 0.5:
            assert data["is_fraud"] is True

    def test_risk_level_high_for_high_probability(self):
        response = client.post("/predict", json=SAMPLE_TXN)
        data     = response.json()
        assert data["risk_level"] == "HIGH"
        assert data["fraud_probability"] == 0.9231


# ── /predict/batch ────────────────────────────────────────────────────────────

class TestPredictBatch:

    def test_returns_200(self):
        response = client.post("/predict/batch",
                               json={"transactions": [SAMPLE_TXN, SAMPLE_TXN]})
        assert response.status_code == 200

    def test_returns_correct_count(self):
        response = client.post("/predict/batch",
                               json={"transactions": [SAMPLE_TXN, SAMPLE_TXN]})
        data = response.json()
        assert data["total"] == 2

    def test_returns_fraud_count(self):
        response = client.post("/predict/batch",
                               json={"transactions": [SAMPLE_TXN, SAMPLE_TXN]})
        assert "fraud_count" in response.json()

    def test_returns_batch_latency(self):
        response = client.post("/predict/batch",
                               json={"transactions": [SAMPLE_TXN, SAMPLE_TXN]})
        assert response.json()["latency_ms"] > 0

    def test_empty_batch_returns_422(self):
        response = client.post("/predict/batch", json={"transactions": []})
        assert response.status_code == 422


# ── /model/info ───────────────────────────────────────────────────────────────

class TestModelInfo:

    def test_returns_200(self):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_returns_framework(self):
        response = client.get("/model/info")
        assert response.json()["framework"] == "XGBoost"

    def test_returns_threshold(self):
        response = client.get("/model/info")
        assert "threshold" in response.json()


# ── / root ────────────────────────────────────────────────────────────────────

class TestRoot:

    def test_returns_200(self):
        assert client.get("/").status_code == 200

    def test_returns_docs_url(self):
        response = client.get("/")
        assert "docs" in response.json()