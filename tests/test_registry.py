"""
tests/test_registry.py

Run with:  pytest tests/test_registry.py -v
All tests use mocks — no live MLflow server needed.
"""

import sys
import numpy as np
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")


# ── Mock helpers ──────────────────────────────────────────────────────────────

def make_mock_version(version="1", stage="Production", run_id="abc123"):
    v               = MagicMock()
    v.version       = version
    v.current_stage = stage
    v.run_id        = run_id
    return v


def make_mock_run(auc_pr=0.85, auc_roc=0.97):
    run                  = MagicMock()
    run.data.metrics     = {"auc_pr": auc_pr, "auc_roc": auc_roc}
    return run


# ── list_versions ─────────────────────────────────────────────────────────────

class TestListVersions:

    @patch("registry.client")
    def test_prints_without_error(self, mock_client, capsys):
        mock_client.get_latest_versions.return_value = [
            make_mock_version("1", "Production", "run1"),
            make_mock_version("2", "Staging",    "run2"),
        ]
        mock_client.get_run.return_value = make_mock_run(auc_pr=0.85)

        from registry import list_versions
        list_versions("fraud_detector")

        captured = capsys.readouterr()
        assert "Production" in captured.out
        assert "Staging"    in captured.out

    @patch("registry.client")
    def test_handles_no_versions(self, mock_client, capsys):
        mock_client.get_latest_versions.return_value = []

        from registry import list_versions
        list_versions("fraud_detector")

        captured = capsys.readouterr()
        assert "No versions found" in captured.out


# ── promote_to_production ─────────────────────────────────────────────────────

class TestPromoteToProduction:

    @patch("registry.client")
    def test_calls_transition(self, mock_client):
        mock_client.get_model_version.return_value = make_mock_version("3")
        mock_client.get_run.return_value           = make_mock_run()

        from registry import promote_to_production
        promote_to_production("3", "fraud_detector")

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="fraud_detector",
            version="3",
            stage="Production",
            archive_existing_versions=True,
        )

    @patch("registry.client")
    def test_archives_existing_versions(self, mock_client):
        mock_client.get_model_version.return_value = make_mock_version("4")
        mock_client.get_run.return_value           = make_mock_run()

        from registry import promote_to_production
        promote_to_production("4", "fraud_detector")

        call_kwargs = mock_client.transition_model_version_stage.call_args.kwargs
        assert call_kwargs["archive_existing_versions"] is True

    @patch("registry.client")
    def test_promotes_correct_stage(self, mock_client):
        mock_client.get_model_version.return_value = make_mock_version("2")
        mock_client.get_run.return_value           = make_mock_run()

        from registry import promote_to_production
        promote_to_production("2", "fraud_detector")

        call_kwargs = mock_client.transition_model_version_stage.call_args.kwargs
        assert call_kwargs["stage"] == "Production"


# ── compare_staging_vs_production ─────────────────────────────────────────────

class TestCompare:

    @patch("registry.mlflow.xgboost.load_model")
    @patch("registry.client")
    def test_returns_results_for_both_stages(self, mock_client, mock_load):
        mock_client.get_latest_versions.side_effect = lambda name, stages: (
            [make_mock_version("1", stages[0])]
        )

        n       = 200
        y_test  = np.array([1]*10 + [0]*190)
        proba   = np.random.default_rng(0).uniform(0, 1, n)
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.column_stack([1-proba, proba])
        mock_load.return_value = mock_model

        from registry import compare_staging_vs_production
        results = compare_staging_vs_production(np.random.randn(n, 36), y_test)

        assert "Staging"    in results
        assert "Production" in results

    @patch("registry.mlflow.xgboost.load_model")
    @patch("registry.client")
    def test_auc_scores_in_valid_range(self, mock_client, mock_load):
        mock_client.get_latest_versions.side_effect = lambda name, stages: (
            [make_mock_version("1", stages[0])]
        )

        n      = 200
        y_test = np.array([1]*20 + [0]*180)
        proba  = np.random.default_rng(1).uniform(0, 1, n)
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.column_stack([1-proba, proba])
        mock_load.return_value = mock_model

        from registry import compare_staging_vs_production
        results = compare_staging_vs_production(np.random.randn(n, 36), y_test)

        for stage in results:
            assert 0 <= results[stage]["auc_roc"] <= 1
            assert 0 <= results[stage]["auc_pr"]  <= 1


# ── get_best_run ──────────────────────────────────────────────────────────────

class TestGetBestRun:

    @patch("registry.mlflow.search_runs")
    def test_returns_top_run(self, mock_search):
        mock_search.return_value = pd.DataFrame([
            {"run_id": "run1", "metrics.auc_pr": 0.88,
             "tags.mlflow.runName": "xgboost_tuned"},
            {"run_id": "run2", "metrics.auc_pr": 0.75,
             "tags.mlflow.runName": "baseline"},
        ])

        from registry import get_best_run
        best = get_best_run("fraud-detection", metric="auc_pr")
        assert best["run_id"] == "run1"

    @patch("registry.mlflow.search_runs")
    def test_handles_empty_results(self, mock_search):
        mock_search.return_value = pd.DataFrame()

        from registry import get_best_run
        result = get_best_run("fraud-detection")
        assert result is None

    @patch("registry.mlflow.search_runs")
    def test_first_row_is_best(self, mock_search):
        mock_search.return_value = pd.DataFrame([
            {"run_id": "run_a", "metrics.auc_pr": 0.91,
             "tags.mlflow.runName": "best"},
            {"run_id": "run_b", "metrics.auc_pr": 0.80,
             "tags.mlflow.runName": "worse"},
        ])

        from registry import get_best_run
        best = get_best_run()
        assert best["metrics.auc_pr"] == 0.91


# ── load_production_model ─────────────────────────────────────────────────────

class TestLoadProductionModel:

    @patch("registry.mlflow.xgboost.load_model")
    def test_loads_from_correct_uri(self, mock_load):
        mock_load.return_value = MagicMock()

        from registry import load_production_model
        load_production_model("fraud_detector")

        mock_load.assert_called_once_with("models:/fraud_detector/Production")

    @patch("registry.mlflow.xgboost.load_model")
    def test_returns_model_object(self, mock_load):
        expected       = MagicMock()
        mock_load.return_value = expected

        from registry import load_production_model
        result = load_production_model("fraud_detector")
        assert result is expected