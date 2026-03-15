"""
api/predictor.py

Model loading and inference logic.
Separated from main.py so it can be tested independently.
"""

import time
from pathlib import Path

import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import yaml
from mlflow.tracking import MlflowClient


# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


cfg = load_config()

# Resolve tracking URI
_uri = cfg["mlflow"]["tracking_uri"]
if not _uri.startswith("http"):
    _project_root = Path(__file__).resolve().parent.parent
    _uri = str(_project_root / _uri)
mlflow.set_tracking_uri(_uri)


# ── Feature names (must match features.py output order) ──────────────────────

BASE_FEATURES = [f"V{i}" for i in range(1, 29)]
DERIVED_FEATURES = [
    "hour", "is_night", "log_amount", "amount_bin",
    "v14_x_amount", "v17_x_amount", "v12_x_amount", "v10_x_amount"
]
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


# ── Predictor class ───────────────────────────────────────────────────────────

class FraudPredictor:
    """
    Wraps the MLflow Production model with feature engineering
    and SHAP explainability.

    Loaded once at API startup and reused across all requests.
    """

    def __init__(self):
        self.model         = None
        self.explainer     = None
        self.model_version = "unknown"
        self.model_stage   = "Production"
        self.threshold     = cfg["model"]["threshold"]

    def load(self):
        """Load Production model from MLflow registry."""
        model_name = cfg["mlflow"]["registered_model_name"]
        uri        = f"models:/{model_name}/Production"

        print(f"Loading model from: {uri}")
        self.model = mlflow.xgboost.load_model(uri)

        # Get version metadata
        client   = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            self.model_version = str(versions[0].version)

        # Build SHAP explainer
        print("Building SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print(f"Model v{self.model_version} loaded and ready.")

    def _engineer_features(self, raw: dict) -> np.ndarray:
        """
        Apply the same feature engineering as features.py.
        Must stay in sync with FraudFeatureEngineer.transform().
        """
        # Extract V1-V28
        v_features = [raw[f"V{i}"] for i in range(1, 29)]

        # Derived features
        hour       = (raw["Time"] % 86400) / 3600
        is_night   = int(hour < 6 or hour > 22)
        log_amount = np.log1p(raw["Amount"])

        if raw["Amount"] <= 1:
            amount_bin = 0
        elif raw["Amount"] <= 10:
            amount_bin = 1
        elif raw["Amount"] <= 100:
            amount_bin = 2
        elif raw["Amount"] <= 500:
            amount_bin = 3
        else:
            amount_bin = 4

        v14_x_amount = raw["V14"] * log_amount
        v17_x_amount = raw["V17"] * log_amount
        v12_x_amount = raw["V12"] * log_amount
        v10_x_amount = raw["V10"] * log_amount

        features = v_features + [
            hour, is_night, log_amount, amount_bin,
            v14_x_amount, v17_x_amount, v12_x_amount, v10_x_amount
        ]
        return np.array(features, dtype=np.float32).reshape(1, -1)

    def predict(self, raw: dict) -> dict:
        """
        Score a single transaction.

        Args:
            raw: dict with keys V1-V28, Amount, Time

        Returns:
            dict with fraud_probability, is_fraud, risk_level,
                 top_features (SHAP), latency_ms, model_version
        """
        start    = time.time()
        features = self._engineer_features(raw)

        # Fraud probability
        proba    = float(self.model.predict_proba(features)[0][1])
        is_fraud = proba >= self.threshold

        # Risk level
        if proba >= 0.7:
            risk_level = "HIGH"
        elif proba >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # SHAP explanation
        shap_vals = self.explainer.shap_values(features)[0]
        top_features = sorted(
            [
                {
                    "feature": ALL_FEATURES[i],
                    "impact":  round(float(shap_vals[i]), 4),
                    "value":   round(float(features[0][i]), 4),
                }
                for i in range(len(ALL_FEATURES))
            ],
            key=lambda x: abs(x["impact"]),
            reverse=True,
        )[:5]

        latency_ms = round((time.time() - start) * 1000, 2)

        return {
            "fraud_probability": round(proba, 4),
            "is_fraud":          is_fraud,
            "risk_level":        risk_level,
            "threshold_used":    self.threshold,
            "top_features":      top_features,
            "latency_ms":        latency_ms,
            "model_version":     self.model_version,
        }

    def predict_batch(self, transactions: list[dict]) -> list[dict]:
        """Score multiple transactions."""
        return [self.predict(txn) for txn in transactions]


# Singleton — created once, shared across all requests
predictor = FraudPredictor()