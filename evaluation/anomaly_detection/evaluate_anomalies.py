"""
RootIQ - Anomaly Detection Evaluation Script
Evaluates Isolation Forest against Statistical Baseline on telemetry dataset.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.anomaly_detection.statistical_baseline import StatisticalBaselineDetector
from src.feature_engineering.metric_features import MetricFeatureEngineer
from evaluation.metrics import calculate_anomaly_metrics

def run_evaluation():
    print("=== Running Anomaly Detection Evaluation ===")
    metrics_path = "data/raw/metrics/telemetry_metrics.csv"
    gt_path = "data/evaluation/labelled_incidents/ground_truth_incidents.json"

    if not os.path.exists(metrics_path) or not os.path.exists(gt_path):
        print("Data files not found. Run python data_generator.py first.")
        return

    df = pd.read_csv(metrics_path, parse_dates=["timestamp"])
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    # Label ground-truth abnormal rows
    df["ground_truth"] = 0
    for inc in ground_truth:
        start = pd.to_datetime(inc["start_time"], utc=True)
        end = pd.to_datetime(inc["end_time"], utc=True)
        aff_services = inc["affected_services"]
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end) & (df["service"].isin(aff_services))
        df.loc[mask, "ground_truth"] = 1

    # Feature Engineering
    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df)

    # Train & Predict Isolation Forest
    iso_forest = IsolationForestDetector(contamination=0.15)
    iso_forest.fit(fe_df)
    iso_preds = iso_forest.predict(fe_df)

    # Save trained model to models/anomaly_detection/isolation_forest.pkl
    model_save_dir = "models/anomaly_detection"
    os.makedirs(model_save_dir, exist_ok=True)
    iso_forest.save_model(os.path.join(model_save_dir, "isolation_forest.pkl"))
    print(f"Saved trained Isolation Forest model to {model_save_dir}/isolation_forest.pkl")

    # Baseline Model
    baseline = StatisticalBaselineDetector(z_threshold=2.5)
    baseline.fit(df)
    base_preds = baseline.predict(df)

    # Calculate metrics
    iso_results = calculate_anomaly_metrics(
        df["ground_truth"].tolist(),
        iso_preds["is_anomaly"].tolist(),
        iso_preds["anomaly_score"].tolist()
    )

    base_results = calculate_anomaly_metrics(
        df["ground_truth"].tolist(),
        base_preds["baseline_anomaly"].tolist(),
        base_preds["baseline_z_max"].tolist()
    )

    comparison = {
        "IsolationForest": iso_results,
        "StatisticalBaseline_ZScore": base_results
    }

    out_dir = "outputs/evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "anomaly_evaluation.json"), "w") as f:
        json.dump(comparison, f, indent=4)

    print("\n--- Model Evaluation Results ---")
    print(f"Isolation Forest -> Precision: {iso_results['precision']}, Recall: {iso_results['recall']}, F1: {iso_results['f1_score']}, ROC-AUC: {iso_results['roc_auc']}")
    print(f"Z-Score Baseline -> Precision: {base_results['precision']}, Recall: {base_results['recall']}, F1: {base_results['f1_score']}, ROC-AUC: {base_results['roc_auc']}")
    print(f"Results exported to {out_dir}/anomaly_evaluation.json")

if __name__ == "__main__":
    run_evaluation()