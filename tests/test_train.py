"""
tests/test_train.py

Run with:  pytest tests/test_train.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification

from train import compute_metrics, DEFAULT_PARAMS, train


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def imbalanced_data():
    """Synthetic imbalanced binary dataset (mirrors real fraud ratio)."""
    X, y = make_classification(
        n_samples=2000,
        n_features=36,
        n_informative=20,
        weights=[0.95, 0.05],
        flip_y=0,
        random_state=42,
    )
    # 80/20 split
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


@pytest.fixture
def trained_model(imbalanced_data):
    """A quickly trained XGBoost model for testing downstream functions."""
    X_train, y_train, _, _ = imbalanced_data
    params = {**DEFAULT_PARAMS, "n_estimators": 50}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_returns_all_required_keys(self, trained_model, imbalanced_data):
        _, _, X_test, y_test = imbalanced_data
        metrics = compute_metrics(trained_model, X_test, y_test)
        required = [
            "auc_roc", "auc_pr", "precision", "recall", "f1",
            "true_positives", "false_positives",
            "false_negatives", "true_negatives", "fraud_caught_pct",
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

    def test_auc_scores_in_valid_range(self, trained_model, imbalanced_data):
        _, _, X_test, y_test = imbalanced_data
        metrics = compute_metrics(trained_model, X_test, y_test)
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert 0.0 <= metrics["auc_pr"]  <= 1.0

    def test_auc_roc_above_random(self, trained_model, imbalanced_data):
        """Any trained model should beat random (0.5) on AUC-ROC."""
        _, _, X_test, y_test = imbalanced_data
        metrics = compute_metrics(trained_model, X_test, y_test)
        assert metrics["auc_roc"] > 0.5, "Model should beat random baseline"

    def test_confusion_matrix_counts_sum_to_test_size(self, trained_model, imbalanced_data):
        _, _, X_test, y_test = imbalanced_data
        metrics = compute_metrics(trained_model, X_test, y_test)
        total = (
            metrics["true_positives"]  + metrics["false_positives"] +
            metrics["false_negatives"] + metrics["true_negatives"]
        )
        assert total == len(y_test)

    def test_precision_recall_between_0_and_1(self, trained_model, imbalanced_data):
        _, _, X_test, y_test = imbalanced_data
        metrics = compute_metrics(trained_model, X_test, y_test)
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"]    <= 1.0

    def test_fraud_caught_pct_matches_recall(self, trained_model, imbalanced_data):
        _, _, X_test, y_test = imbalanced_data
        metrics = compute_metrics(trained_model, X_test, y_test)
        expected = round(metrics["recall"] * 100, 2)
        assert abs(metrics["fraud_caught_pct"] - expected) < 0.1

    def test_custom_threshold_affects_results(self, trained_model, imbalanced_data):
        _, _, X_test, y_test = imbalanced_data
        m_low  = compute_metrics(trained_model, X_test, y_test, threshold=0.2)
        m_high = compute_metrics(trained_model, X_test, y_test, threshold=0.8)
        # Lower threshold → higher recall, lower precision
        assert m_low["recall"]    >= m_high["recall"]
        assert m_low["precision"] <= m_high["precision"]


# ── Model output sanity checks ────────────────────────────────────────────────

class TestModelOutputs:

    def test_predict_proba_sums_to_one(self, trained_model, imbalanced_data):
        _, _, X_test, _ = imbalanced_data
        probas = trained_model.predict_proba(X_test)
        assert probas.shape[1] == 2
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_fraud_proba_in_0_1(self, trained_model, imbalanced_data):
        _, _, X_test, _ = imbalanced_data
        fraud_proba = trained_model.predict_proba(X_test)[:, 1]
        assert (fraud_proba >= 0).all() and (fraud_proba <= 1).all()

    def test_model_has_feature_importances(self, trained_model):
        importances = trained_model.feature_importances_
        assert len(importances) > 0
        assert (importances >= 0).all()

    def test_model_output_shape(self, trained_model, imbalanced_data):
        _, _, X_test, _ = imbalanced_data
        preds = trained_model.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(preds).issubset({0, 1})


# ── Quick integration test (no Optuna, fast) ──────────────────────────────────

class TestTrainIntegration:

    def test_train_runs_without_error(self, tmp_path, monkeypatch):
        """
        Smoke test: train() should complete without raising,
        using fast default params and mocked data paths.
        """
        import numpy as np

        # Create minimal fake processed data in tmp dir
        X = np.random.randn(500, 36).astype(np.float32)
        y_sm = np.array([0]*460 + [1]*40)    # ~8% fraud (post-SMOTE)
        y_test_arr = np.array([0]*90 + [1]*10)

        (tmp_path / "data" / "processed").mkdir(parents=True)
        np.save(tmp_path / "data/processed/X_train_sm.npy",    X)
        np.save(tmp_path / "data/processed/y_train_sm.npy",    y_sm)
        np.save(tmp_path / "data/processed/X_test_scaled.npy", X[:100])
        np.save(tmp_path / "data/processed/y_test.npy",        y_test_arr)
        import pandas as pd
        feat_names = [f"V{i}" for i in range(36)]
        pd.Series(feat_names).to_csv(
            tmp_path / "data/processed/feature_names.csv", index=False
        )

        # Redirect DATA_DIR and MODEL_DIR to tmp paths
        import train as train_module
        monkeypatch.setattr(train_module, "DATA_DIR",  tmp_path / "data/processed")
        monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "models")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow_test.db")
        (tmp_path / "models").mkdir()

        model, metrics = train_module.train(n_trials=2, use_optuna=False)

        assert model is not None
        assert "auc_roc" in metrics
        assert "auc_pr"  in metrics
        assert metrics["auc_roc"] > 0.0