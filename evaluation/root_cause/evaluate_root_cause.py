"""
RootIQ - Root Cause Evaluation Script
Evaluates multi-source evidence root-cause ranking (Top-1, Top-3, MRR)
against ground-truth incident scenarios.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.service_graph.dependency_graph import ServiceDependencyGraph
from src.service_graph.graph_analysis import ServiceGraphAnalyzer
from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.feature_engineering.metric_features import MetricFeatureEngineer
from src.incident_correlation.incident_correlator import IncidentCorrelator
from src.root_cause.evidence_engine import EvidenceEngine
from src.root_cause.root_cause_scorer import RootCauseScorer
from src.root_cause.root_cause_ranker import RootCauseRanker
from evaluation.metrics import calculate_root_cause_metrics

def run_evaluation():
    print("=== Running Root Cause Analysis Evaluation ===")
    metrics_path = "data/raw/metrics/telemetry_metrics.csv"
    logs_path = "data/raw/logs/telemetry_logs.csv"
    gt_path = "data/evaluation/labelled_incidents/ground_truth_incidents.json"
    topo_path = "data/raw/service_dependencies.json"

    df_metrics = pd.read_csv(metrics_path, parse_dates=["timestamp"])
    df_logs = pd.read_csv(logs_path, parse_dates=["timestamp"])
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    # Load Graph
    sdg = ServiceDependencyGraph()
    sdg.load_from_json(topo_path)
    analyzer = ServiceGraphAnalyzer(sdg)

    # Preprocess & Detect Anomalies
    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_metrics)
    
    detector = IsolationForestDetector(contamination=0.15)
    detector.fit(fe_df)
    anom_df = detector.predict(fe_df)

    # Merge log error counts onto metric windows
    df_logs["is_err"] = df_logs["level"].isin(["ERROR", "CRITICAL", "FATAL"]).astype(int)
    log_errs = df_logs.groupby(["timestamp", "service"])["is_err"].sum().reset_index().rename(columns={"is_err": "error_count"})
    anom_df = anom_df.merge(log_errs, on=["timestamp", "service"], how="left").fillna(0.0)

    # Correlate into incidents
    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(anom_df, sdg.graph)
    print(f"Detected {len(incidents)} correlated incidents.")

    # Evaluate Root Cause per incident
    evidence_engine = EvidenceEngine()
    scorer = RootCauseScorer()
    ranker = RootCauseRanker(scorer)

    eval_records = []
    normal_df = anom_df[anom_df["is_anomaly"] == 0]

    for gt in ground_truth:
        true_rc = gt["true_root_cause"]
        gt_start = pd.to_datetime(gt["start_time"], utc=True)
        gt_end = pd.to_datetime(gt["end_time"], utc=True)

        # Find overlapping detected incident
        matched_inc = None
        for inc in incidents:
            inc_start = pd.to_datetime(inc["start_time"], utc=True)
            inc_end = pd.to_datetime(inc["end_time"], utc=True)
            if not (inc_end < gt_start or inc_start > gt_end):
                matched_inc = inc
                break

        if matched_inc:
            evidence = evidence_engine.extract_candidate_evidence(
                matched_inc["raw_alert_df"],
                normal_df,
                analyzer
            )
            ranked = ranker.rank_candidates(evidence)
            eval_res = ranker.evaluate_ranking(ranked, true_rc)
            eval_records.append(eval_res)
            print(f"Scenario: '{gt['scenario']}' -> True Root Cause: {true_rc} | Top-1 Predicted: {eval_res['predicted_root_cause']} | Rank: {eval_res['true_rank']}")

    # Save trained root cause weights
    rc_model_dir = "models/root_cause"
    os.makedirs(rc_model_dir, exist_ok=True)
    joblib.dump({"weights": scorer.weights, "ranker": ranker}, os.path.join(rc_model_dir, "root_cause_model.pkl"))

    metrics_res = calculate_root_cause_metrics(eval_records)
    out_dir = "outputs/evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "root_cause_evaluation.json"), "w") as f:
        json.dump({"metrics": metrics_res, "details": eval_records}, f, indent=4)

    print("\n--- Root Cause Evaluation Summary ---")
    print(f"Top-1 Accuracy: {metrics_res['top_1_accuracy'] * 100:.1f}%")
    print(f"Top-3 Accuracy: {metrics_res['top_3_accuracy'] * 100:.1f}%")
    print(f"Mean Reciprocal Rank (MRR): {metrics_res['mean_reciprocal_rank']:.4f}")
    print(f"Results exported to {out_dir}/root_cause_evaluation.json")

if __name__ == "__main__":
    run_evaluation()