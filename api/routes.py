"""
RootIQ - FastAPI Backend Routes
Provides RESTful endpoints for telemetry ingestion, anomaly detection,
incident correlation, root-cause ranking, and model evaluation.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.service_graph.dependency_graph import ServiceDependencyGraph
from src.service_graph.graph_analysis import ServiceGraphAnalyzer
from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.feature_engineering.metric_features import MetricFeatureEngineer
from src.incident_correlation.incident_correlator import IncidentCorrelator
from src.root_cause.evidence_engine import EvidenceEngine
from src.root_cause.root_cause_scorer import RootCauseScorer
from src.root_cause.root_cause_ranker import RootCauseRanker
from src.explanation.llm_explainer import IncidentExplainer
from src.utils.helpers import get_system_metrics

app = FastAPI(
    title="RootIQ API",
    description="AI-Powered Root Cause Analysis & Incident Intelligence Engine",
    version="1.0.0"
)

DATA_METRICS = "data/raw/metrics/telemetry_metrics.csv"
DATA_LOGS = "data/raw/logs/telemetry_logs.csv"
DATA_TOPO = "data/raw/service_dependencies.json"

@app.get("/health")
def health_check():
    """System health and resource monitoring."""
    stats = get_system_metrics()
    return {"status": "healthy", "service": "RootIQ", "system_resources": stats}

@app.get("/topology")
def get_topology():
    """Return microservices dependency graph."""
    if not os.path.exists(DATA_TOPO):
        raise HTTPException(status_code=404, detail="Topology file not found. Run data_generator.py first.")
    with open(DATA_TOPO, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/incidents")
def get_correlated_incidents():
    """Detect anomalies and return correlated incidents."""
    if not os.path.exists(DATA_METRICS):
        raise HTTPException(status_code=404, detail="Telemetry data not found.")

    df = pd.read_csv(DATA_METRICS, parse_dates=["timestamp"])
    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df)

    detector = IsolationForestDetector(contamination=0.15)
    detector.fit(fe_df)
    anom_df = detector.predict(fe_df)

    sdg = ServiceDependencyGraph()
    if os.path.exists(DATA_TOPO):
        sdg.load_from_json(DATA_TOPO)

    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(anom_df, sdg.graph)

    # Return clean serializable response
    clean_incidents = []
    for inc in incidents:
        rec = {k: v for k, v in inc.items() if k != "raw_alert_df"}
        clean_incidents.append(rec)

    return {"count": len(clean_incidents), "incidents": clean_incidents}

@app.get("/root_cause/{incident_id}")
def analyze_root_cause(incident_id: str):
    """Rank probable root causes with 4-dimensional evidence for an incident."""
    if not os.path.exists(DATA_METRICS):
        raise HTTPException(status_code=404, detail="Telemetry data not found.")

    df = pd.read_csv(DATA_METRICS, parse_dates=["timestamp"])
    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df)

    detector = IsolationForestDetector(contamination=0.15)
    detector.fit(fe_df)
    anom_df = detector.predict(fe_df)

    sdg = ServiceDependencyGraph()
    if os.path.exists(DATA_TOPO):
        sdg.load_from_json(DATA_TOPO)
    analyzer = ServiceGraphAnalyzer(sdg)

    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(anom_df, sdg.graph)

    target_inc = next((i for i in incidents if i["incident_id"] == incident_id), None)
    if not target_inc:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")

    normal_df = anom_df[anom_df["is_anomaly"] == 0]
    evidence_engine = EvidenceEngine()
    scorer = RootCauseScorer()
    ranker = RootCauseRanker(scorer)

    evidence = evidence_engine.extract_candidate_evidence(
        target_inc["raw_alert_df"],
        normal_df,
        analyzer
    )
    ranked = ranker.rank_candidates(evidence)

    explainer = IncidentExplainer(use_local_llm=False)
    summary = explainer.explain(target_inc, ranked)

    return {
        "incident_id": incident_id,
        "affected_services": target_inc["affected_services"],
        "ranked_root_causes": ranked,
        "explanation": summary
    }

@app.get("/evaluation")
def get_evaluation_results():
    """Retrieve pre-computed model and system evaluation metrics."""
    res_dir = "outputs/evaluation_results"
    results = {}
    for filename in ["anomaly_evaluation.json", "root_cause_evaluation.json", "incident_correlation_evaluation.json"]:
        p = os.path.join(res_dir, filename)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                results[filename.replace(".json", "")] = json.load(f)
    return results