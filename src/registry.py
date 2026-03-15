"""
src/registry.py

MLflow Model Registry utilities.
Handles promoting, loading, comparing, and listing model versions.

Usage:
    python src/registry.py --list
    python src/registry.py --promote --version 1
    python src/registry.py --compare
    python src/registry.py --best
"""

import argparse
import sys
from pathlib import Path
import os
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


# ── Load config ───────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


cfg        = load_config()
MODEL_NAME = cfg["mlflow"]["registered_model_name"]

# Resolve tracking URI — env var takes priority, then config, then safe temp fallback
_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not _tracking_uri:
    _raw = cfg["mlflow"]["tracking_uri"]
    if _raw.startswith("http") or _raw.startswith("sqlite"):
        _tracking_uri = _raw
    elif _raw.startswith("/"):
        # absolute path (e.g. /app/mlruns) — only valid inside Docker, not in CI
        _tracking_uri = "sqlite:///tmp/mlflow_registry.db"
    else:
        # relative path — anchor to project root
        _project_root = Path(__file__).resolve().parent.parent
        _tracking_uri = str(_project_root / _raw)

mlflow.set_tracking_uri(_tracking_uri)

# Lazily initialized — avoids connecting at import time (important for tests)
client = None


def _client():
    """Return the module-level MlflowClient, creating it on first call."""
    global client
    if client is None:
        client = MlflowClient()
    return client


# ── Registry operations ───────────────────────────────────────────────────────

def list_versions(model_name=MODEL_NAME):
    """Print all registered versions with their stage and metrics."""
    print(f"\nRegistered versions of '{model_name}':")
    print(f"{'Ver':<6} {'Stage':<15} {'Run ID':<36} {'AUC-PR':<10}")
    print("-" * 72)

    versions = _client().get_latest_versions(model_name)
    if not versions:
        print("  No versions found.")
        return

    for v in versions:
        run    = _client().get_run(v.run_id)
        auc_pr = run.data.metrics.get("auc_pr", "—")
        print(f"{v.version:<6} {v.current_stage:<15} {v.run_id:<36} {auc_pr:<10}")


def promote_to_production(version, model_name=MODEL_NAME):
    """
    Promote a specific model version to Production.
    Automatically archives the current Production version.
    """
    mv     = _client().get_model_version(model_name, version)
    run    = _client().get_run(mv.run_id)
    auc_pr = run.data.metrics.get("auc_pr", "unknown")

    print(f"\nPromoting version {version} to Production")
    print(f"  AUC-PR : {auc_pr}")
    print(f"  Run ID : {mv.run_id}")

    _client().transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"\nVersion {version} is now in Production.")
    print("Previous Production version archived automatically.")


def archive_version(version, model_name=MODEL_NAME):
    """Move a version to Archived stage."""
    _client().transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Archived",
    )
    print(f"Version {version} archived.")


def load_production_model(model_name=MODEL_NAME):
    """
    Load the current Production model.
    This is exactly what FastAPI calls at startup.
    """
    uri   = f"models:/{model_name}/Production"
    model = mlflow.xgboost.load_model(uri)
    print(f"Loaded Production model from: {uri}")
    return model


def get_production_run_id(model_name=MODEL_NAME):
    """Returns the run_id that produced the current Production model."""
    versions = _client().get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"No Production version found for '{model_name}'")
    return versions[0].run_id


def compare_staging_vs_production(X_test, y_test, model_name=MODEL_NAME):
    """
    Load both Staging and Production models and compare on test set.
    Helps you decide whether to promote a new version.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    results = {}
    for stage in ["Staging", "Production"]:
        versions = _client().get_latest_versions(model_name, stages=[stage])
        if not versions:
            print(f"  No {stage} model found — skipping.")
            continue

        v       = versions[0]
        model   = mlflow.xgboost.load_model(f"models:/{model_name}/{stage}")
        y_proba = model.predict_proba(X_test)[:, 1]

        results[stage] = {
            "version": v.version,
            "run_id":  v.run_id,
            "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
            "auc_pr":  round(average_precision_score(y_test, y_proba), 4),
        }

    if not results:
        print("No models found to compare.")
        return results

    print(f"\n{'Metric':<12} {'Staging':<12} {'Production':<12} {'Winner'}")
    print("-" * 50)
    for metric in ["auc_roc", "auc_pr"]:
        s = results.get("Staging",    {}).get(metric, "—")
        p = results.get("Production", {}).get(metric, "—")
        winner = ""
        if isinstance(s, float) and isinstance(p, float):
            winner = "Staging" if s > p else "Production"
        print(f"{metric:<12} {str(s):<12} {str(p):<12} {winner}")

    return results


def get_best_run(experiment_name=None, metric="auc_pr"):
    """
    Returns the run with the highest value of metric
    across all runs in the experiment.
    """
    if experiment_name is None:
        experiment_name = cfg["mlflow"]["experiment_name"]

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} DESC"],
    )

    if runs.empty:
        print("No runs found.")
        return None

    best = runs.iloc[0]
    print(f"\nBest run by {metric}:")
    print(f"  Run ID   : {best['run_id']}")
    print(f"  {metric:<8} : {best.get(f'metrics.{metric}', '—')}")
    print(f"  Run name : {best.get('tags.mlflow.runName', '—')}")
    return best


def add_model_description(version, description, model_name=MODEL_NAME):
    """Add a human-readable description to a model version."""
    _client().update_model_version(
        name=model_name,
        version=version,
        description=description,
    )
    print(f"Description added to version {version}.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Model Registry utilities")
    parser.add_argument("--list",    action="store_true",
                        help="List all registered model versions")
    parser.add_argument("--promote", action="store_true",
                        help="Promote a version to Production")
    parser.add_argument("--version", type=str, default=None,
                        help="Model version number to promote")
    parser.add_argument("--compare", action="store_true",
                        help="Compare Staging vs Production on test set")
    parser.add_argument("--best",    action="store_true",
                        help="Show best run by AUC-PR")
    args = parser.parse_args()

    if args.list:
        list_versions()

    elif args.promote:
        if not args.version:
            print("Error: --version required with --promote")
            sys.exit(1)
        promote_to_production(args.version)

    elif args.compare:
        DATA_DIR = Path(cfg["data"]["processed_dir"])
        X_test   = np.load(DATA_DIR / "X_test_scaled.npy")
        y_test   = np.load(DATA_DIR / "y_test.npy")
        compare_staging_vs_production(X_test, y_test)

    elif args.best:
        get_best_run()

    else:
        parser.print_help()