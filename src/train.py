"""
src/train.py

Full training pipeline: loads processed data, tunes XGBoost with Optuna,
logs everything to MLflow, saves final model.

Usage:
    python src/train.py
    python src/train.py --trials 100 --no-optuna
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import yaml
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Config ────────────────────────────────────────────────────────────────────

_cfg_path = Path(__file__).parent.parent / "config.yaml"
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

DATA_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 100,
    "eval_metric":      "aucpr",
    "random_state":     42,
    "verbosity":        0,
    "n_jobs":           -1,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Load the processed numpy arrays saved by features.py (Week 2)."""
    X_train = np.load(DATA_DIR / "X_train_sm.npy")   # SMOTE-balanced
    y_train = np.load(DATA_DIR / "y_train_sm.npy")
    X_test  = np.load(DATA_DIR / "X_test_scaled.npy") # raw test split
    y_test  = np.load(DATA_DIR / "y_test.npy")
    feature_names = (
        pd.read_csv(DATA_DIR / "feature_names.csv").iloc[:, 0].tolist()
    )
    print(f"Train: {X_train.shape} | Fraud: {y_train.mean()*100:.2f}%")
    print(f"Test:  {X_test.shape}  | Fraud: {y_test.mean()*100:.4f}%")
    return X_train, y_train, X_test, y_test, feature_names


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test, threshold=0.5):
    """Returns a dict of all evaluation metrics."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc_roc":        round(roc_auc_score(y_test, y_proba), 4),
        "auc_pr":         round(average_precision_score(y_test, y_proba), 4),
        "precision":      round(tp / (tp + fp + 1e-9), 4),
        "recall":         round(tp / (tp + fn + 1e-9), 4),
        "f1":             round(2*tp / (2*tp + fp + fn + 1e-9), 4),
        "true_positives":  int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives":  int(tn),
        "fraud_caught_pct": round(tp / (tp + fn + 1e-9) * 100, 2),
    }


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_train, y_train):
    """Returns an Optuna objective function closed over training data."""

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 50, 200),
            "eval_metric": "aucpr",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="average_precision", n_jobs=-1
        )
        return scores.mean()

    return objective


# ── Main training run ─────────────────────────────────────────────────────────

def train(n_trials=50, use_optuna=True):
    """
    Full training pipeline with MLflow logging.

    Args:
        n_trials:   number of Optuna hyperparameter search trials
        use_optuna: if False, uses DEFAULT_PARAMS (faster, for debugging)
    """
    _uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(_uri)
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name=f"xgboost_optuna_{n_trials}trials"):

        # 1. Load data
        print("\n── Loading data ──────────────────────────")
        X_train, y_train, X_test, y_test, feature_names = load_data()
        mlflow.log_param("train_size",       X_train.shape[0])
        mlflow.log_param("test_size",        X_test.shape[0])
        mlflow.log_param("n_features",       X_train.shape[1])
        mlflow.log_param("train_fraud_pct",  round(y_train.mean() * 100, 2))
        mlflow.log_param("test_fraud_pct",   round(y_test.mean() * 100, 4))

        # 2. Hyperparameter search
        if use_optuna:
            print(f"\n── Optuna search ({n_trials} trials) ─────────────")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                make_objective(X_train, y_train),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            best_params = study.best_params
            best_params.update({
                "eval_metric": "aucpr",
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
            })
            print(f"  Best CV AUC-PR: {study.best_value:.4f}")
            mlflow.log_metric("optuna_best_cv_auc_pr", round(study.best_value, 4))
        else:
            print("\n── Using default params (Optuna skipped) ─")
            best_params = DEFAULT_PARAMS

        mlflow.log_params({k: v for k, v in best_params.items()
                           if k not in ("eval_metric", "verbosity")})

        # 3. Train final model
        print("\n── Training final model ──────────────────")
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        # 4. Evaluate
        print("\n── Evaluating on test set ────────────────")
        metrics = compute_metrics(model, X_test, y_test)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        print(f"  AUC-ROC :        {metrics['auc_roc']}")
        print(f"  AUC-PR  :        {metrics['auc_pr']}   ← primary metric")
        print(f"  Precision:       {metrics['precision']}")
        print(f"  Recall:          {metrics['recall']}")
        print(f"  Fraud caught:    {metrics['fraud_caught_pct']}%")
        print(f"  False positives: {metrics['false_positives']}")
        print(f"  Fraud missed:    {metrics['false_negatives']}")

        # 5. Save model + log to MLflow
        print("\n── Saving model ──────────────────────────")
        model_path = MODEL_DIR / "xgb_fraud_v1.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraud_detector",
        )
        mlflow.log_artifact(str(model_path))

        # Save metrics + params as JSON for downstream use
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(MODEL_DIR / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        mlflow.log_artifact(str(MODEL_DIR / "metrics.json"))
        mlflow.log_artifact(str(MODEL_DIR / "best_params.json"))

        run_id = mlflow.active_run().info.run_id
        print(f"\n  Model saved to:   {model_path}")
        print(f"  MLflow run ID:    {run_id}")
        print(f"\n── Done ──────────────────────────────────")
        print(f"  AUC-PR: {metrics['auc_pr']} | AUC-ROC: {metrics['auc_roc']}")

        return model, metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--trials",     type=int,  default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--no-optuna",  action="store_true",
                        help="Skip Optuna, use default params")
    args = parser.parse_args()

    train(n_trials=args.trials, use_optuna=not args.no_optuna)