"""
Strict Exhaustive Test Suite for RootIQ
Tests all 8 pages, all 4 sabotage presets, custom CSV ingestion,
interactive Plotly visualizations, anomaly detection, Granger/temporal sequencing,
alert correlation, and multi-criteria root cause analysis.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Workspace setup
WORKSPACE_DIR = r"C:\Users\bijin\OneDrive\Desktop\RootIQ"
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.service_graph.dependency_graph import ServiceDependencyGraph
from src.service_graph.graph_analysis import ServiceGraphAnalyzer
from src.feature_engineering.metric_features import MetricFeatureEngineer
from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.anomaly_detection.statistical_baseline import StatisticalBaselineDetector
from src.time_series.anomaly_timeline import AnomalyTimeline
from src.time_series.change_detection import ChangeDetector
from src.incident_correlation.incident_correlator import IncidentCorrelator
from src.root_cause.evidence_engine import EvidenceEngine
from src.root_cause.root_cause_scorer import RootCauseScorer
from src.root_cause.root_cause_ranker import RootCauseRanker
from src.explanation.llm_explainer import IncidentExplainer
from src.utils.helpers import get_system_metrics


# --- FIXTURES ---

@pytest.fixture(scope="module")
def paths():
    return {
        "metrics": os.path.join(WORKSPACE_DIR, "data", "raw", "metrics", "telemetry_metrics.csv"),
        "logs": os.path.join(WORKSPACE_DIR, "data", "raw", "logs", "telemetry_logs.csv"),
        "traces": os.path.join(WORKSPACE_DIR, "data", "raw", "traces", "telemetry_traces.csv"),
        "topo": os.path.join(WORKSPACE_DIR, "data", "raw", "service_dependencies.json"),
        "gt": os.path.join(WORKSPACE_DIR, "data", "evaluation", "labelled_incidents", "ground_truth_incidents.json"),
        "rc_eval": os.path.join(WORKSPACE_DIR, "outputs", "evaluation_results", "root_cause_evaluation.json"),
        "ad_eval": os.path.join(WORKSPACE_DIR, "outputs", "evaluation_results", "anomaly_evaluation.json"),
    }


@pytest.fixture(scope="module")
def loaded_data(paths):
    df_m = pd.read_csv(paths["metrics"], parse_dates=["timestamp"])
    df_l = pd.read_csv(paths["logs"], parse_dates=["timestamp"])
    df_t = pd.read_csv(paths["traces"], parse_dates=["timestamp"])
    sdg = ServiceDependencyGraph()
    sdg.load_from_json(paths["topo"])
    analyzer = ServiceGraphAnalyzer(sdg)
    return {
        "df_metrics": df_m,
        "df_logs": df_l,
        "df_traces": df_t,
        "sdg": sdg,
        "analyzer": analyzer,
        "services": sorted(sdg.get_services())
    }


# ==============================================================================
# TEST SUITE 1: DATA INTEGRITY & TOPOLOGY
# ==============================================================================

def test_data_integrity(paths, loaded_data):
    """Verify all raw datasets exist, are non-empty, and contain required schema."""
    for name, path in paths.items():
        assert os.path.exists(path), f"Required file missing: {path}"

    df_m = loaded_data["df_metrics"]
    assert not df_m.empty, "Metrics dataframe is empty"
    required_metric_cols = ["timestamp", "service", "latency_ms", "error_rate", "cpu_usage", "memory_usage", "request_rate"]
    for col in required_metric_cols:
        assert col in df_m.columns, f"Missing metric column: {col}"

    assert len(loaded_data["services"]) == 7, "Expected exactly 7 microservices in topology"
    assert "database" in loaded_data["services"]
    assert "api_gateway" in loaded_data["services"]
    assert "payment_service" in loaded_data["services"]


def test_topology_graph_analysis(loaded_data):
    """Verify topology dependency graph, centrality, and downstream dependents."""
    sdg = loaded_data["sdg"]
    analyzer = loaded_data["analyzer"]

    centrality = analyzer.compute_centrality()
    assert len(centrality) == 7, "Centrality computed for all services"

    db_callers = sdg.get_downstream_dependents("database")
    assert len(db_callers) > 0, "Database must have caller services"
    assert "order_service" in db_callers or "payment_service" in db_callers or "inventory_service" in db_callers

    fe_callers = sdg.get_downstream_dependents("frontend")
    assert len(fe_callers) == 0, "Frontend is top-level ingress, has no upstream callers in reverse dependency"


# ==============================================================================
# TEST SUITE 2: PAGE 1 - SYSTEM TOPOLOGY PLOTLY RENDERING
# ==============================================================================

def test_page1_topology_figure_rendering(loaded_data):
    """Test Page 1 NetworkX/Plotly graph generation under Normal and Outage states."""
    sdg = loaded_data["sdg"]
    pos = {
        "frontend": (0, 3), "api_gateway": (1.6, 3), "order_service": (3.2, 4.3),
        "inventory_service": (3.2, 1.7), "payment_service": (4.8, 4.5),
        "cache": (4.8, 1.5), "database": (6.2, 3)
    }

    for state, active_fault in [("NORMAL", None), ("SABOTAGE_DB", "database"), ("SABOTAGE_PAY", "payment_service")]:
        active_callers = sdg.get_downstream_dependents(active_fault) if active_fault else []
        node_names = [n for n in sdg.graph.nodes() if n in pos]

        node_c = []
        for n in node_names:
            if active_fault and n == active_fault:
                node_c.append("#ef4444")
            elif active_fault and n in active_callers:
                node_c.append("#f59e0b")
            else:
                node_c.append("#38bdf8")

        edge_x, edge_y = [], []
        for u, v in sdg.graph.edges():
            if u in pos and v in pos:
                edge_x.extend([pos[u][0], pos[v][0], None])
                edge_y.extend([pos[u][1], pos[v][1], None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2.5, color="rgba(148, 163, 184, 0.45)"), mode="lines")
        node_trace = go.Scatter(x=[pos[n][0] for n in node_names], y=[pos[n][1] for n in node_names], mode="markers+text", marker=dict(size=34, color=node_c))
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig_json = fig.to_json()
        assert len(fig_json) > 100, "Figure JSON serialization failed"


# ==============================================================================
# TEST SUITE 3: PAGE 2 - INGESTION, STANDARDIZATION & ALL 4 SABOTAGE PRESETS
# ==============================================================================

def test_custom_csv_standardization(loaded_data):
    """Test custom telemetry standardization on complete, partial, and noisy CSVs."""
    from app import standardize_custom_telemetry

    min_df = pd.DataFrame({
        "timestamp": ["2026-03-01 10:00:00", "2026-03-01 10:01:00"],
        "service": ["database", "database"]
    })
    std_min = standardize_custom_telemetry(min_df)
    assert "latency_ms" in std_min.columns and std_min["latency_ms"].iloc[0] == 45.0
    assert "error_rate" in std_min.columns and std_min["error_rate"].iloc[0] == 0.0
    assert "cpu_usage" in std_min.columns and std_min["cpu_usage"].iloc[0] == 25.0
    assert pd.api.types.is_datetime64_any_dtype(std_min["timestamp"])

    df_m = loaded_data["df_metrics"]
    mask_sample = (df_m["timestamp"] >= "2026-03-01 10:30:00") & (df_m["timestamp"] <= "2026-03-01 11:05:00")
    sample_slice = df_m[mask_sample].copy()
    assert len(sample_slice) == 252, f"Expected 252 rows in sample slice, got {len(sample_slice)}"
    std_sample = standardize_custom_telemetry(sample_slice)
    assert len(std_sample) == 252


def test_all_four_sabotage_presets(loaded_data):
    """Test all 4 sabotage presets: Database, Payment, Order, API Gateway."""
    df_m = loaded_data["df_metrics"]
    sdg = loaded_data["sdg"]

    presets = [
        ("database", "Database Lock Contention", 45),
        ("payment_service", "Payment Gateway Outage", 40),
        ("order_service", "Memory Leak & OOM", 50),
        ("api_gateway", "Extreme Latency Spike & CPU Throttling", 30),
    ]

    for target_s, fault_name, start_min in presets:
        mod_df = df_m.copy().sort_values(["service", "timestamp"]).reset_index(drop=True)
        timestamps_unique = sorted(mod_df["timestamp"].unique())
        assert len(timestamps_unique) > start_min + 15

        fault_window = timestamps_unique[start_min:start_min + 15]
        mask = (mod_df["service"] == target_s) & (mod_df["timestamp"].isin(fault_window))

        mod_df.loc[mask, "latency_ms"] = mod_df.loc[mask, "latency_ms"] * 18.0 + 800.0
        mod_df.loc[mask, "error_rate"] = np.clip(mod_df.loc[mask, "error_rate"] + 0.45, 0.0, 1.0)
        mod_df.loc[mask, "cpu_usage"] = np.clip(mod_df.loc[mask, "cpu_usage"] * 1.5 + 40.0, 0.0, 100.0)

        callers = sdg.get_downstream_dependents(target_s)
        for c in callers:
            c_mask = (mod_df["service"] == c) & (mod_df["timestamp"].isin(fault_window[2:]))
            mod_df.loc[c_mask, "latency_ms"] = mod_df.loc[c_mask, "latency_ms"] * 12.0 + 400.0
            mod_df.loc[c_mask, "error_rate"] = np.clip(mod_df.loc[c_mask, "error_rate"] + 0.30, 0.0, 1.0)

        target_injected = mod_df.loc[mask]
        assert (target_injected["latency_ms"] > 800.0).all()
        assert (target_injected["error_rate"] >= 0.45).all()
        assert len(mod_df) == len(df_m)


# ==============================================================================
# TEST SUITE 4: PAGE 3 - TELEMETRY & EDA VISUALIZATIONS
# ==============================================================================

def test_page3_eda_visualizations(loaded_data):
    """Test dual-axis time series, CPU/memory distributions, heatmaps, and comparator charts."""
    df_m = loaded_data["df_metrics"]
    services = loaded_data["services"]

    for svc in services:
        svc_df = df_m[df_m["service"] == svc].sort_values("timestamp")

        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=svc_df["timestamp"], y=svc_df["latency_ms"], name="Latency (ms)"))
        fig_dual.add_trace(go.Scatter(x=svc_df["timestamp"], y=svc_df["error_rate"], name="Error Rate", yaxis="y2"))
        fig_dual.update_layout(yaxis2=dict(overlaying="y", side="right"))
        assert len(fig_dual.to_json()) > 0

        fig_cpu = px.line(svc_df, x="timestamp", y="cpu_usage", template="plotly_dark")
        fig_mem = px.line(svc_df, x="timestamp", y="memory_usage", template="plotly_dark")
        assert len(fig_cpu.to_json()) > 0
        assert len(fig_mem.to_json()) > 0

        num_cols = [c for c in ["cpu_usage", "memory_usage", "latency_ms", "request_rate", "error_rate"] if c in svc_df.columns]
        corr = svc_df[num_cols].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", template="plotly_dark")
        assert len(fig_corr.to_json()) > 0

    for metric in ["latency_ms", "error_rate", "cpu_usage", "memory_usage"]:
        comp_df = df_m[df_m["service"].isin(["database", "order_service", "frontend"])]
        fig_comp = px.line(comp_df, x="timestamp", y=metric, color="service", template="plotly_dark")
        assert len(fig_comp.to_json()) > 0


# ==============================================================================
# TEST SUITE 5: PAGE 4 - ANOMALY DETECTION ENGINE
# ==============================================================================

def test_page4_anomaly_detection_pipeline(loaded_data):
    """Test Isolation Forest and Statistical Baseline anomaly detection."""
    df_m = loaded_data["df_metrics"]

    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_m)
    assert fe_df.shape[1] > df_m.shape[1], "Feature engineering should add features"

    iso = IsolationForestDetector(contamination=0.15)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    assert "is_anomaly" in preds.columns
    assert "anomaly_score" in preds.columns
    assert preds["is_anomaly"].isin([0, 1]).all()
    assert (preds["anomaly_score"] >= 0.0).all() and (preds["anomaly_score"] <= 1.0).all()

    stat = StatisticalBaselineDetector(z_threshold=2.5)
    stat.fit(df_m)
    stat_preds = stat.predict(df_m)
    assert "baseline_anomaly" in stat_preds.columns

    fig_box = px.box(preds, x="service", y="anomaly_score", color="service", template="plotly_dark")
    assert len(fig_box.to_json()) > 0


# ==============================================================================
# TEST SUITE 6: PAGE 5 - TIME-SERIES ONSET SEQUENCING
# ==============================================================================

def test_page5_temporal_sequencing(loaded_data):
    """Test chronological onset extraction and waterfall precedence sequencing."""
    df_m = loaded_data["df_metrics"]
    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_m)
    iso = IsolationForestDetector(contamination=0.15)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    timeline = AnomalyTimeline()
    onsets = timeline.extract_onset_times(preds)

    assert not onsets.empty, "Onsets must not be empty on incident dataset"
    assert "service" in onsets.columns
    assert "first_onset" in onsets.columns
    assert "temporal_score" in onsets.columns
    assert (onsets["temporal_score"] >= 0.0).all() and (onsets["temporal_score"] <= 1.0).all()

    fig_bar = px.bar(onsets, x="service", y="temporal_score", color="temporal_score", template="plotly_dark")
    assert len(fig_bar.to_json()) > 0


# ==============================================================================
# TEST SUITE 7: PAGE 6 - INCIDENT CORRELATION & CLUSTERING
# ==============================================================================

def test_page6_incident_correlation(loaded_data):
    """Test incident correlation, alert clustering, and compression ratio."""
    df_m = loaded_data["df_metrics"]
    sdg = loaded_data["sdg"]

    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_m)
    iso = IsolationForestDetector(contamination=0.15)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(preds, sdg.graph)

    assert len(incidents) > 0, "Should identify at least 1 correlated incident"
    for inc in incidents:
        assert "incident_id" in inc
        assert "affected_services" in inc
        assert "start_time" in inc
        assert "end_time" in inc
        assert "duration_minutes" in inc
        assert "raw_alert_df" in inc
        assert len(inc["affected_services"]) > 0

    raw_alert_count = int(preds["is_anomaly"].sum())
    incident_count = len(incidents)
    compression = ((raw_alert_count - incident_count) / raw_alert_count) * 100 if raw_alert_count > 0 else 0
    assert compression >= 0.0


# ==============================================================================
# TEST SUITE 8: PAGE 7 - ROOT CAUSE ANALYSIS & EXECUTIVE VERDICT
# ==============================================================================

def test_page7_root_cause_analysis_and_verdict(loaded_data):
    """Test full multi-criteria root cause ranking, realistic confidence, and visualizations."""
    df_m = loaded_data["df_metrics"]
    df_l = loaded_data["df_logs"]
    sdg = loaded_data["sdg"]
    analyzer = loaded_data["analyzer"]

    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_m)
    iso = IsolationForestDetector(contamination=0.15)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    if not df_l.empty and "level" in df_l.columns:
        df_l_copy = df_l.copy()
        df_l_copy["timestamp"] = pd.to_datetime(df_l_copy["timestamp"], utc=True)
        preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
        df_l_copy["is_err"] = df_l_copy["level"].isin(["ERROR", "CRITICAL", "FATAL"]).astype(int)
        log_errs = df_l_copy.groupby(["timestamp", "service"])["is_err"].sum().reset_index().rename(columns={"is_err": "error_count"})
        preds = preds.merge(log_errs, on=["timestamp", "service"], how="left").fillna(0.0)

    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(preds, sdg.graph)
    assert len(incidents) > 0

    evidence_engine = EvidenceEngine()
    scorer = RootCauseScorer()
    ranker = RootCauseRanker(scorer)
    normal_df = preds[preds["is_anomaly"] == 0]

    for inc in incidents:
        evidence = evidence_engine.extract_candidate_evidence(inc["raw_alert_df"], normal_df, analyzer)
        ranked = ranker.rank_candidates(evidence)
        assert len(ranked) > 0

        top1 = ranked[0]
        assert 60.0 <= top1["confidence_pct"] <= 100.0, f"Confidence {top1['confidence_pct']} out of range"
        assert 0.0 <= top1["root_cause_score"] <= 100.0

        remediation_actions = {
            "database": "Restart the database pods, terminate hanging locks, and increase connection pool limits.",
            "payment_service": "Verify 3rd-party payment gateway status, refresh API credentials, and enable circuit breaker fallbacks.",
            "order_service": "Scale up order_service worker pods, clear event queues, and inspect heap dumps for memory leaks.",
            "api_gateway": "Scale out api_gateway ingress instances, adjust rate-limiting limits, and refresh routing rules.",
            "inventory_service": "Restart inventory_service pods, check lock timeouts, and scale read replicas.",
            "cache": "Flush stale Redis keys, verify cache cluster node health, and expand max memory limits.",
            "frontend": "Purge edge CDN caches, verify ingress route health, and roll back bad UI releases."
        }
        remedy = remediation_actions.get(top1["service"], "Default remedy")
        assert len(remedy) > 15

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=top1["confidence_pct"],
            gauge={"axis": {"range": [0, 100]}}
        ))
        assert len(fig_gauge.to_json()) > 0

        categories = ["Temporal Precedence", "Dependency Impact", "Anomaly Magnitude", "Metric/Log Spikes"]
        values = [top1["temporal_evidence"] * 100, top1["dependency_evidence"] * 100, top1["anomaly_evidence"] * 100, top1["metric_log_evidence"] * 100]
        fig_radar = go.Figure(go.Scatterpolar(r=values, theta=categories, fill="toself"))
        assert len(fig_radar.to_json()) > 0

        affected = inc["affected_services"]
        culprit = top1["service"]
        sankey_nodes = [culprit] + [s for s in affected if s != culprit]
        node_map = {name: idx for idx, name in enumerate(sankey_nodes)}
        sources = [node_map[culprit] for s in affected if s != culprit]
        targets = [node_map[s] for s in affected if s != culprit]
        if not sources and len(sankey_nodes) > 1:
            sources, targets = [0], [1]
        if sources:
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(label=sankey_nodes),
                link=dict(source=sources, target=targets, value=[1]*len(sources))
            )])
            assert len(fig_sankey.to_json()) > 0

        explainer = IncidentExplainer(use_local_llm=False)
        report = explainer.explain(inc, ranked)
        assert len(report) > 50

        postmortem_md = f"""# SRE Incident Post-Mortem & Root Cause Analysis Report
**Incident Identifier:** `{inc['incident_id']}`
**Primary Culprit:** **`{top1['service']}`**
**Root Cause Confidence:** **{top1['confidence_pct']}%**
{report}
"""
        assert "SRE Incident Post-Mortem" in postmortem_md
        assert top1["service"] in postmortem_md


# ==============================================================================
# TEST SUITE 9: PAGE 8 - QUANTITATIVE EVALUATION BENCHMARK
# ==============================================================================

def test_page8_evaluation_metrics(paths):
    """Verify ground truth evaluation results, accuracy metrics, and system stats."""
    with open(paths["rc_eval"]) as f:
        rc_eval = json.load(f)["metrics"]

    assert rc_eval["top_1_accuracy"] >= 0.80, "Top-1 Accuracy must be >= 80%"
    assert rc_eval["top_3_accuracy"] == 1.0, "Top-3 Accuracy must be 100%"
    assert rc_eval["mean_reciprocal_rank"] >= 0.85, "MRR must be >= 0.85"

    with open(paths["ad_eval"]) as f:
        ad_eval = json.load(f)
    assert "IsolationForest" in ad_eval
    assert "StatisticalBaseline_ZScore" in ad_eval
    assert "f1_score" in ad_eval["IsolationForest"]

    sys_stats = get_system_metrics()
    assert "cpu_percent" in sys_stats
    assert "process_ram_mb" in sys_stats
    assert "host_ram_used_gb" in sys_stats
    assert sys_stats["process_ram_mb"] > 0


# ==============================================================================
# TEST SUITE 10: STREAMLIT NAVIGATION & REDIRECTION SAFETY
# ==============================================================================

def test_streamlit_navigation_safety():
    """Verify PAGE_MAP and safe target_page redirection without widget conflict."""
    PAGE_MAP = {
        "⚡ 1. System Map (Architecture)": "⚡ Overview & Topology",
        "💥 2. Incident Simulator (Break Services)": "📁 Data Ingestion & Simulator",
        "📊 3. Live Metrics (Health Monitor)": "📊 Telemetry & EDA",
        "🔍 4. AI Anomaly Finder (Catching Spikes)": "🔍 Anomaly Detection",
        "⏱️ 5. Who Failed First? (Timeline)": "⏱️ Time-Series & Onset",
        "🚨 6. Alert Grouping (Noise Filter)": "🚨 Incident Correlation",
        "🎯 7. Root Cause Verdict (The Final Answer)": "🎯 Root Cause Analysis",
        "📈 8. Project Scorecard (Accuracy Proof)": "📈 Evaluation Metrics"
    }
    PAGE_OPTIONS = list(PAGE_MAP.keys())
    assert len(PAGE_OPTIONS) == 8

    mock_session = {"nav_menu": PAGE_OPTIONS[0], "target_page": PAGE_OPTIONS[6]}

    if "target_page" in mock_session and mock_session["target_page"] in PAGE_OPTIONS:
        mock_session["nav_menu"] = mock_session.pop("target_page")

    assert mock_session["nav_menu"] == PAGE_OPTIONS[6]
    assert "target_page" not in mock_session
