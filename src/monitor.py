"""
Data drift monitoring using Evidently AI.
Compares training distribution (reference) vs test/current data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


FEATURE_NAMES = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "hour","is_night","log_amount","amount_bin",
    "v14_x_amount","v17_x_amount","v12_x_amount","v10_x_amount",
]


def load_feature_names(path: str = "data/processed/feature_names.csv") -> list[str]:
    try:
        df = pd.read_csv(path)
        return df.iloc[:, 0].tolist()
    except FileNotFoundError:
        return FEATURE_NAMES


def run_drift_report(threshold: float = 0.5) -> None:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    feature_names = load_feature_names(os.path.join(base, "data/processed/feature_names.csv"))

    reference = np.load(os.path.join(base, "data/processed/X_train_sm.npy"))
    current = np.load(os.path.join(base, "data/processed/X_test_scaled.npy"))

    # Align columns — reference may have more rows but same feature count
    n_features = len(feature_names)
    reference_df = pd.DataFrame(reference[:, :n_features], columns=feature_names)
    current_df = pd.DataFrame(current[:, :n_features], columns=feature_names)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    output_dir = os.path.join(base, "reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "drift_report.html")
    report.save_html(output_path)
    print(f"Drift report saved to {output_path}")

    result = report.as_dict()
    drift_metrics = result["metrics"][0]["result"]
    drift_share = drift_metrics.get("share_of_drifted_columns", 0.0)
    n_drifted = drift_metrics.get("number_of_drifted_columns", 0)
    n_total = drift_metrics.get("number_of_columns", n_features)

    print(f"Drifted features: {n_drifted}/{n_total} ({drift_share:.1%})")

    if drift_share > threshold:
        print(f"ALERT: Drift share {drift_share:.1%} exceeds threshold {threshold:.1%}")
        sys.exit(1)
    else:
        print(f"Drift within acceptable range (threshold={threshold:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data drift report")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Drift share threshold to trigger alert (default: 0.5)")
    args = parser.parse_args()
    run_drift_report(threshold=args.threshold)
