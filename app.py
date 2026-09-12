"""
RootIQ - Main Streamlit Web Application
Interactive Dashboard for AI-Powered Root Cause Analysis and Incident Intelligence.
Supports Benchmark Data, Custom Telemetry CSV Uploads, and Live Failure Simulation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# Configure page layout
st.set_page_config(
    page_title="RootIQ - AIOps Root Cause Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal RootIQ modules
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

# --- Paths ---
METRICS_PATH = "data/raw/metrics/telemetry_metrics.csv"
LOGS_PATH = "data/raw/logs/telemetry_logs.csv"
TRACES_PATH = "data/raw/traces/telemetry_traces.csv"
TOPO_PATH = "data/raw/service_dependencies.json"
GT_PATH = "data/evaluation/labelled_incidents/ground_truth_incidents.json"

# --- Base Data Loader ---
@st.cache_data
def load_benchmark():
    if not os.path.exists(METRICS_PATH):
        import data_generator
        data_generator.generate_telemetry("data")
    df_m = pd.read_csv(METRICS_PATH, parse_dates=["timestamp"])
    df_l = pd.read_csv(LOGS_PATH, parse_dates=["timestamp"])
    df_t = pd.read_csv(TRACES_PATH, parse_dates=["timestamp"])
    return df_m, df_l, df_t

@st.cache_data
def load_topology():
    sdg = ServiceDependencyGraph()
    if os.path.exists(TOPO_PATH):
        sdg.load_from_json(TOPO_PATH)
    return sdg

sdg = load_topology()
analyzer = ServiceGraphAnalyzer(sdg)
df_metrics_default, df_logs_default, df_traces_default = load_benchmark()

# --- Sidebar Header ---
st.sidebar.image("https://img.icons8.com/fluency/96/server.png", width=70)
st.sidebar.title("RootIQ Core")
st.sidebar.caption("TY B.Sc. Data Science Project")

# --- 1. Telemetry Data Source Selector ---
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Telemetry Source")
data_source = st.sidebar.radio(
    "Choose Data Mode:",
    ["📊 Benchmark Dataset", "📁 Upload Custom CSV", "⚡ Live Failure Simulator"]
)

df_metrics = df_metrics_default.copy()
df_logs = df_logs_default.copy()
df_traces = df_traces_default.copy()
active_source_label = "Benchmark Telemetry (Default Microservices Outages)"

# Render uploader/simulator DIRECTLY under the radio button!
if data_source == "📁 Upload Custom CSV":
    st.sidebar.info("Upload your metrics CSV below:")
    uploaded_file = st.sidebar.file_uploader("Choose CSV", type=["csv"], key="sidebar_csv_uploader")
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            if "timestamp" in custom_df.columns and "service" in custom_df.columns:
                custom_df["timestamp"] = pd.to_datetime(custom_df["timestamp"], utc=True)
                df_metrics = custom_df
                active_source_label = f"Custom CSV: {uploaded_file.name} ({len(df_metrics)} records)"
                st.sidebar.success(f"Loaded {len(df_metrics)} rows!")
            else:
                st.sidebar.error("CSV must contain 'timestamp' and 'service' columns.")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

    sample_csv = df_metrics_default.head(20).to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("⬇️ Download CSV Template", sample_csv, "sample_metrics_template.csv", "text/csv")

elif data_source == "⚡ Live Failure Simulator":
    st.sidebar.info("Select service to crash live:")
    sim_service = st.sidebar.selectbox("Target Microservice:", sorted(sdg.get_services()), index=5, key="sb_sim_svc")
    sim_fault = st.sidebar.selectbox("Outage Scenario:", [
        "Database Lock Contention",
        "Payment Gateway Outage",
        "Memory Leak & OOM",
        "Extreme Latency Spike & CPU Throttling"
    ], key="sb_sim_fault")
    sim_start = st.sidebar.slider("Crash Start (Minute):", 10, 100, 45, key="sb_sim_min")

    if st.sidebar.button("🔥 Inject Failure Now", key="sb_inject_btn"):
        st.session_state["simulated_telemetry"] = True
        st.session_state["sim_service"] = sim_service
        st.session_state["sim_fault"] = sim_fault
        st.session_state["sim_start"] = sim_start

    if st.session_state.get("simulated_telemetry", False):
        target_s = st.session_state.get("sim_service", "database")
        start_min = st.session_state.get("sim_start", 45)
        mod_df = df_metrics_default.copy().sort_values(["service", "timestamp"]).reset_index(drop=True)
        timestamps_unique = sorted(mod_df["timestamp"].unique())
        if len(timestamps_unique) > start_min + 15:
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

            df_metrics = mod_df
            active_source_label = f"Live Outage Injected on '{target_s}' (Min {start_min}-{start_min+15})"
            st.sidebar.success(f"Active Outage on: {target_s}")

# --- 2. Navigation Menu ---
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
menu = st.sidebar.radio(
    "Go to page:",
    [
        "⚡ Overview & Topology",
        "📁 Data Ingestion & Simulator",
        "📊 Telemetry & EDA",
        "🔍 Anomaly Detection",
        "⏱️ Time-Series & Onset",
        "🚨 Incident Correlation",
        "🎯 Root Cause Analysis",
        "📈 Evaluation Metrics"
    ]
)

# --- 3. Engine Parameters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Engine Parameters")
contamination = st.sidebar.slider("Anomaly Contamination", 0.01, 0.30, 0.15, 0.01)
weight_temporal = st.sidebar.slider("Weight: Temporal Onset", 0.1, 0.6, 0.35, 0.05)
weight_dependency = st.sidebar.slider("Weight: Dependency Impact", 0.1, 0.5, 0.25, 0.05)
weight_anomaly = st.sidebar.slider("Weight: Anomaly Score", 0.1, 0.4, 0.20, 0.05)
weight_metric = st.sidebar.slider("Weight: Metric/Log Spikes", 0.1, 0.4, 0.20, 0.05)

# --- Prominent Active Banner on Main Page ---
if data_source == "📁 Upload Custom CSV":
    st.info(f"📁 **Custom Data Mode Active:** `{active_source_label}`. Use the sidebar to upload any metrics CSV.")
elif data_source == "⚡ Live Failure Simulator":
    st.warning(f"⚡ **Live Failure Simulator Mode Active:** `{active_source_label}`. Choose any service in the sidebar to simulate a live failure cascade!")
else:
    st.success(f"📊 **Benchmark Mode Active:** `{active_source_label}`. Running on standard OpenTelemetry microservices benchmark.")

# --- PAGE 1: OVERVIEW & TOPOLOGY ---
if menu == "⚡ Overview & Topology":
    st.title("⚡ RootIQ: AI-Powered Root Cause Analysis")
    st.markdown(f"""
    **RootIQ** transforms noisy multi-source telemetry (metrics, logs, traces) into **ranked, evidence-based root causes**.
    It models microservice dependencies, identifies chronological anomaly onset, and isolates primary failures from cascading downstream symptoms.
    """)

    col1, col2, col3, col4 = st.columns(4)
    sys_stats = get_system_metrics()
    col1.metric("Monitored Services", len(sdg.get_services()))
    col2.metric("Active Telemetry Records", f"{len(df_metrics):,}")
    col3.metric("Host RAM Used", f"{sys_stats['host_ram_used_gb']} GB")
    col4.metric("Engine Health", "Online (CPU-Ready)")

    st.markdown("---")
    st.subheader("🕸️ Microservices Topology & Call Graph")

    pos = {
        "frontend": (0, 3),
        "api_gateway": (1.5, 3),
        "order_service": (3, 4),
        "inventory_service": (3, 2),
        "payment_service": (4.5, 4.5),
        "cache": (4.5, 1.5),
        "database": (5.5, 3)
    }

    edge_x = []
    edge_y = []
    for u, v in sdg.graph.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = [pos[node][0] for node in sdg.graph.nodes() if node in pos]
    node_y = [pos[node][1] for node in sdg.graph.nodes() if node in pos]
    node_names = [node for node in sdg.graph.nodes() if node in pos]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_names,
        textposition="bottom center",
        hoverinfo="text",
        marker=dict(
            size=28,
            color="#2ecc71",
            line=dict(width=2, color="#27ae60")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=20, r=20, t=20),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=400
                    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Service Centrality & Failure Impact Matrix")
    centrality = analyzer.compute_centrality()
    cent_df = pd.DataFrame.from_dict(centrality, orient="index").reset_index().rename(columns={"index": "Service"})
    cent_df["Downstream Dependents"] = cent_df["Service"].apply(lambda s: ", ".join(sdg.get_downstream_dependents(s)) or "None (Edge)")
    st.dataframe(cent_df, use_container_width=True)

# --- PAGE 2: DATA INGESTION & SIMULATOR ---
elif menu == "📁 Data Ingestion & Simulator":
    st.title("📁 Data Ingestion & Live Incident Simulator")
    st.markdown("""
    This module allows you to **bring your own telemetry dataset** or **simulate live system outages** 
    to see RootIQ detect and diagnose root causes in real time!
    """)

    tab1, tab2 = st.tabs(["📁 Upload Custom Telemetry CSV", "⚡ Live Failure Injection Simulator"])

    with tab1:
        st.subheader("Upload Telemetry Data (CSV)")
        st.write("Upload your own metrics CSV to run RootIQ on custom test datasets or production telemetry.")

        up_file = st.file_uploader("Choose a CSV file:", type=["csv"], key="main_tab_uploader")
        if up_file is not None:
            try:
                user_df = pd.read_csv(up_file)
                st.write("### Preview of Uploaded Telemetry:")
                st.dataframe(user_df.head(10), use_container_width=True)

                req_cols = ["timestamp", "service"]
                missing = [c for c in req_cols if c not in user_df.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing mandatory columns: {missing}")
                else:
                    st.success("✅ Valid Telemetry CSV! Select 'Upload Custom CSV' in the sidebar to run analysis across all pages.")
            except Exception as e:
                st.error(f"Error parsing file: {e}")

        st.markdown("---")
        st.subheader("CSV Format Requirements")
        st.markdown("""
        Your CSV file should ideally contain the following columns:
        * `timestamp`: ISO UTC timestamp (e.g. `2026-03-01T10:00:00Z`)
        * `service`: Service identifier (e.g. `database`, `order_service`)
        * `latency_ms`: Response latency in milliseconds (optional)
        * `error_rate`: Failure ratio between 0.0 and 1.0 (optional)
        * `cpu_usage`: CPU utilization percentage 0-100% (optional)
        * `memory_usage`: Memory utilization percentage 0-100% (optional)
        * `request_rate`: Requests per second (optional)
        """)

        sample_csv = df_metrics_default.head(20).to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Sample Telemetry CSV Template", sample_csv, "sample_telemetry.csv", "text/csv")

    with tab2:
        st.subheader("⚡ Live Microservice Failure Simulator")
        st.markdown("""
        Demonstrate RootIQ live during your college viva! Pick any service, inject a failure scenario, 
        and watch how the AI uncovers the true cause behind the cascading errors.
        """)

        c1, c2 = st.columns(2)
        with c1:
            test_svc = st.selectbox("Choose Service to Sabotage:", sorted(sdg.get_services()), index=5, key="main_demo_svc")
            test_type = st.selectbox("Crash Scenario:", [
                "Database Lock Contention",
                "Payment Gateway Outage",
                "Memory Leak & OOM",
                "Extreme Latency Spike & CPU Throttling"
            ], key="main_demo_type")
        with c2:
            test_minute = st.slider("Crash Time Offset (Minute):", 10, 100, 50, key="main_demo_min")
            st.info(f"Target: `{test_svc}` will fail at T+{test_minute}m. Error cascades will automatically propagate to dependent caller services.")

        if st.button("💥 Inject Failure into Pipeline", key="main_inject_btn"):
            st.session_state["simulated_telemetry"] = True
            st.session_state["sim_service"] = test_svc
            st.session_state["sim_fault"] = test_type
            st.session_state["sim_start"] = test_minute
            st.success(f"Failure injected into `{test_svc}`! Navigate to 'Anomaly Detection' or 'Root Cause Analysis' to view the AI diagnosis.")

# --- PAGE 3: TELEMETRY & EDA ---
elif menu == "📊 Telemetry & EDA":
    st.title("📊 Operational Telemetry & Exploratory Analysis")
    st.caption(f"Active Data Mode: {active_source_label}")
    selected_service = st.selectbox("Select Service to Inspect", sorted(df_metrics["service"].unique()))

    svc_df = df_metrics[df_metrics["service"] == selected_service].sort_values("timestamp")

    col1, col2 = st.columns(2)
    with col1:
        fig_lat = px.line(svc_df, x="timestamp", y="latency_ms", title=f"{selected_service} - Latency (ms)", color_discrete_sequence=["#e74c3c"])
        st.plotly_chart(fig_lat, use_container_width=True)
    with col2:
        fig_err = px.line(svc_df, x="timestamp", y="error_rate", title=f"{selected_service} - Error Rate", color_discrete_sequence=["#f39c12"])
        st.plotly_chart(fig_err, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_cpu = px.line(svc_df, x="timestamp", y="cpu_usage", title=f"{selected_service} - CPU Usage (%)", color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig_cpu, use_container_width=True)
    with col4:
        fig_mem = px.line(svc_df, x="timestamp", y="memory_usage", title=f"{selected_service} - Memory Usage (%)", color_discrete_sequence=["#9b59b6"])
        st.plotly_chart(fig_mem, use_container_width=True)

    st.subheader("Telemetry Correlation Matrix")
    num_cols = [c for c in ["cpu_usage", "memory_usage", "latency_ms", "request_rate", "error_rate"] if c in svc_df.columns]
    if len(num_cols) > 1:
        corr = svc_df[num_cols].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", title=f"Metric Correlation: {selected_service}")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- PAGE 4: ANOMALY DETECTION ---
elif menu == "🔍 Anomaly Detection":
    st.title("🔍 Unsupervised Anomaly Detection")
    st.caption(f"Active Data Mode: {active_source_label}")
    st.markdown("""
    RootIQ uses **Isolation Forest** as its primary machine learning model to compute multidimensional anomaly scores.
    Results are benchmarked against a **Rolling 3-Sigma Z-Score** statistical baseline.
    """)

    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_metrics)

    iso = IsolationForestDetector(contamination=contamination)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    stat = StatisticalBaselineDetector(z_threshold=2.5)
    stat.fit(df_metrics)
    stat_preds = stat.predict(df_metrics)

    col1, col2, col3 = st.columns(3)
    col1.metric("Isolation Forest Anomalies", f"{int(preds['is_anomaly'].sum())}")
    col2.metric("Statistical Z-Score Anomalies", f"{int(stat_preds['baseline_anomaly'].sum())}")
    col3.metric("Peak Anomaly Score", f"{preds['anomaly_score'].max():.3f}")

    st.subheader("Anomaly Score Distribution Across Services")
    fig_box = px.box(preds, x="service", y="anomaly_score", color="service", title="Anomaly Score Spread per Service")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Detected Anomalous Telemetry Windows")
    disp_cols = [c for c in ["timestamp", "service", "anomaly_score", "latency_ms", "error_rate", "cpu_usage"] if c in preds.columns]
    st.dataframe(preds[preds["is_anomaly"] == 1][disp_cols].head(25), use_container_width=True)

# --- PAGE 5: TIME-SERIES & ONSET ---
elif menu == "⏱️ Time-Series & Onset":
    st.title("⏱️ Temporal Precedence & Anomaly Onset Analysis")
    st.caption(f"Active Data Mode: {active_source_label}")
    st.markdown("""
    **Temporal Precedence Principle:** The service that exhibits anomalous behavior *first* in a failure propagation window
    has a substantially higher likelihood of being the primary root cause than downstream caller services.
    """)

    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_metrics)
    iso = IsolationForestDetector(contamination=contamination)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    timeline = AnomalyTimeline()
    onsets = timeline.extract_onset_times(preds)

    if onsets.empty:
        st.info("No anomalies detected with current threshold. Adjust contamination slider in the sidebar.")
    else:
        st.subheader("Chronological Anomaly Onset Table")
        st.dataframe(onsets, use_container_width=True)

        st.subheader("Waterfall Timeline: Earliest Onset Lead Times")
        fig_bar = px.bar(
            onsets,
            x="service",
            y="temporal_score",
            color="temporal_score",
            color_continuous_scale="Reds",
            title="Temporal Precedence Score (1.0 = Earliest Failure Originator)",
            text_auto=True
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --- PAGE 6: INCIDENT CORRELATION ---
elif menu == "🚨 Incident Correlation":
    st.title("🚨 Incident Correlation & Alert Clustering")
    st.caption(f"Active Data Mode: {active_source_label}")
    st.markdown("""
    Instead of bombarding on-call engineers with dozens of individual alerts across multiple microservices,
    RootIQ clusters temporally and topologically related alerts into **unified Incident Entities**.
    """)

    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_metrics)
    iso = IsolationForestDetector(contamination=contamination)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(preds, sdg.graph)

    raw_alert_count = int(preds["is_anomaly"].sum())
    incident_count = len(incidents)
    compression = ((raw_alert_count - incident_count) / raw_alert_count) * 100 if raw_alert_count > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Telemetry Alerts", raw_alert_count)
    col2.metric("Correlated Incidents", incident_count)
    col3.metric("Alert Noise Reduction", f"{compression:.1f}%")

    st.subheader("Correlated Incident Registry")
    if not incidents:
        st.info("No correlated incidents detected.")
    else:
        for inc in incidents:
            with st.expander(f"🚨 {inc['incident_id']} (Duration: ~{inc['duration_minutes']} mins | {len(inc['affected_services'])} Affected Services)"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Start Time:** `{inc['start_time']}`")
                c2.write(f"**End Time:** `{inc['end_time']}`")
                c3.write(f"**Peak Anomaly Score:** `{inc['peak_anomaly_score']:.3f}`")
                st.write(f"**Affected Services:** {', '.join([f'`{s}`' for s in inc['affected_services']])}")
                disp_c = [c for c in ["timestamp", "service", "anomaly_score", "latency_ms", "error_rate"] if c in inc["raw_alert_df"].columns]
                st.dataframe(inc["raw_alert_df"][disp_c].head(10), use_container_width=True)

# --- PAGE 7: ROOT CAUSE ANALYSIS ---
elif menu == "🎯 Root Cause Analysis":
    st.title("🎯 Multi-Evidence Root Cause Engine")
    st.caption(f"Active Data Mode: {active_source_label}")
    st.markdown("""
    RootIQ combines **4 dimensions of evidence**:
    1. **Temporal Precedence** (Did this service fail first?)
    2. **Service Dependency Impact** (Does this service sit upstream in the call chain of affected callers?)
    3. **Anomaly Severity** (Magnitude of Isolation Forest score)
    4. **Direct Metric/Log Spikes** (Extreme latency leap, connection errors, HTTP 5xx codes)
    """)

    fe = MetricFeatureEngineer()
    fe_df = fe.transform(df_metrics)
    iso = IsolationForestDetector(contamination=contamination)
    iso.fit(fe_df)
    preds = iso.predict(fe_df)

    if not df_logs.empty and "level" in df_logs.columns:
        df_logs_copy = df_logs.copy()
        df_logs_copy["is_err"] = df_logs_copy["level"].isin(["ERROR", "CRITICAL", "FATAL"]).astype(int)
        log_errs = df_logs_copy.groupby(["timestamp", "service"])["is_err"].sum().reset_index().rename(columns={"is_err": "error_count"})
        preds = preds.merge(log_errs, on=["timestamp", "service"], how="left").fillna(0.0)

    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(preds, sdg.graph)

    if not incidents:
        st.warning("No incidents detected. Adjust contamination threshold in the sidebar.")
    else:
        inc_ids = [inc["incident_id"] for inc in incidents]
        selected_id = st.selectbox("Select Correlated Incident to Diagnose", inc_ids)
        target_inc = next(i for i in incidents if i["incident_id"] == selected_id)

        custom_weights = {
            "temporal": weight_temporal,
            "dependency": weight_dependency,
            "anomaly": weight_anomaly,
            "metric_log": weight_metric
        }

        normal_df = preds[preds["is_anomaly"] == 0]
        evidence_engine = EvidenceEngine()
        scorer = RootCauseScorer(custom_weights)
        ranker = RootCauseRanker(scorer)

        evidence = evidence_engine.extract_candidate_evidence(target_inc["raw_alert_df"], normal_df, analyzer)
        ranked = ranker.rank_candidates(evidence)

        if ranked:
            st.subheader(f"Ranked Probable Causes for `{selected_id}`")
            top1 = ranked[0]
            st.success(f"🏆 **Top-1 Probable Root Cause:** `{top1['service']}` (Confidence: **{top1['confidence_pct']}%** | Composite Score: **{top1['root_cause_score']:.1f}/100**)")

            rank_df = pd.DataFrame(ranked)[["rank", "service", "root_cause_score", "confidence_pct", "temporal_evidence", "dependency_evidence", "anomaly_evidence", "metric_log_evidence"]]
            st.dataframe(rank_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                categories = ["Temporal Precedence", "Dependency Impact", "Anomaly Magnitude", "Metric/Log Spikes"]
                values = [
                    top1["temporal_evidence"] * 100,
                    top1["dependency_evidence"] * 100,
                    top1["anomaly_evidence"] * 100,
                    top1["metric_log_evidence"] * 100
                ]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=top1['service'], line_color="#e74c3c"))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title=f"Evidence Breakdown: {top1['service']}")
                st.plotly_chart(fig_radar, use_container_width=True)

            with col2:
                fig_bar = px.bar(
                    rank_df,
                    x="service",
                    y="root_cause_score",
                    color="confidence_pct",
                    title="Candidate Confidence Distribution (%)",
                    text_auto=True
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            explainer = IncidentExplainer(use_local_llm=False)
            report = explainer.explain(target_inc, ranked)
            st.subheader("💡 Automated Incident Intelligence Diagnosis")
            st.markdown(report)

# --- PAGE 8: EVALUATION METRICS ---
elif menu == "📈 Evaluation Metrics":
    st.title("📈 Model & System Evaluation")
    st.markdown("""
    Formal quantitative metrics evaluating Anomaly Detection accuracy, Root Cause Top-1 / Top-3 accuracy,
    and Mean Reciprocal Rank (MRR) against ground-truth incident benchmarks.
    """)

    res_dir = "outputs/evaluation_results"
    rc_eval_file = os.path.join(res_dir, "root_cause_evaluation.json")
    ad_eval_file = os.path.join(res_dir, "anomaly_evaluation.json")

    if os.path.exists(rc_eval_file):
        with open(rc_eval_file, "r") as f:
            rc_data = json.load(f)["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Root Cause Top-1 Accuracy", f"{rc_data['top_1_accuracy']*100:.1f}%")
        c2.metric("Root Cause Top-3 Accuracy", f"{rc_data['top_3_accuracy']*100:.1f}%")
        c3.metric("Mean Reciprocal Rank (MRR)", f"{rc_data['mean_reciprocal_rank']:.4f}")
    else:
        st.info("Run evaluate_root_cause.py to inspect pre-computed evaluation results.")

    if os.path.exists(ad_eval_file):
        with open(ad_eval_file, "r") as f:
            ad_data = json.load(f)
        st.subheader("Anomaly Detection Benchmark: Isolation Forest vs Z-Score")
        ad_comp_df = pd.DataFrame(ad_data).T[["precision", "recall", "f1_score", "roc_auc", "false_positive_rate"]]
        st.dataframe(ad_comp_df, use_container_width=True)

    st.subheader("System Performance & Laptop Footprint")
    stats = get_system_metrics()
    p1, p2, p3 = st.columns(3)
    p1.metric("CPU Utilization", f"{stats['cpu_percent']}%")
    p2.metric("Process RAM", f"{stats['process_ram_mb']} MB")
    p3.metric("Host RAM Total", f"{stats['host_ram_total_gb']} GB")
