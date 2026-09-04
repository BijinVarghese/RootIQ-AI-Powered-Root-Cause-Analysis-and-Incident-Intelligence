"""
RootIQ - Incident Correlation Evaluation Script
Measures alert grouping efficiency and alert noise reduction.
"""

import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.feature_engineering.metric_features import MetricFeatureEngineer
from src.incident_correlation.incident_correlator import IncidentCorrelator
from evaluation.metrics import calculate_correlation_metrics

def run_evaluation():
    print("=== Running Incident Correlation Evaluation ===")
    metrics_path = "data/raw/metrics/telemetry_metrics.csv"
    df = pd.read_csv(metrics_path, parse_dates=["timestamp"])

    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df)

    iso = IsolationForestDetector(contamination=0.15)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    raw_anomalies_count = int(preds["is_anomaly"].sum())

    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(preds)
    incident_count = len(incidents)

    metrics = calculate_correlation_metrics(raw_anomalies_count, incident_count)

    out_dir = "outputs/evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "incident_correlation_evaluation.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n--- Incident Correlation Summary ---")
    print(f"Total Raw Alerts: {metrics['total_raw_alerts']}")
    print(f"Correlated Incidents: {metrics['correlated_incidents']}")
    print(f"Alert Compression Ratio: {metrics['alert_compression_pct']}%")
    print(f"Results exported to {out_dir}/incident_correlation_evaluation.json")

if __name__ == "__main__":
    run_evaluation()