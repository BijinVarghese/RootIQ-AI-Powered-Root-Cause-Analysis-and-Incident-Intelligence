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

# 1. Configure page layout
st.set_page_config(
    page_title="RootIQ - AIOps Root Cause Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Cyber-Ops & Modern Glassmorphism Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 22px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.hero-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 0.96rem;
    font-weight: 400;
}

.badge-group {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.badge-green {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-blue {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.badge-purple {
    background: rgba(192, 132, 252, 0.12);
    color: #c084fc;
    border: 1px solid rgba(192, 132, 252, 0.3);
}

.badge-red {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #34d399;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #34d399;
}

/* Metric KPI Card Styling */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.75) 0%, rgba(31, 41, 55, 0.75) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    border-top: 3px solid #38bdf8;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-top-color: #818cf8;
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.15);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* Button Upgrades */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.45) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

/* Download Button Styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 9px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(16, 185, 129, 0.5) !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0c1017 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.4);
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    color: #94a3b8;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: none;
}

.stTabs [aria-selected="true"] {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #38bdf8 !important;
    border-top: 2px solid #38bdf8 !important;
}

/* Dataframe containers */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

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
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 12px; padding: 10px 0;">
    <span style="font-size: 2.2rem;">⚡</span>
    <div>
        <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">RootIQ</div>
        <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">AIOps Intelligence Core</div>
    </div>
</div>
""", unsafe_allow_html=True)

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
active_source_label = "Benchmark Telemetry (Standard Outages)"

# Render uploader/simulator DIRECTLY under the radio button
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
    st.sidebar.info("Select service to sabotage live:")
    sim_service = st.sidebar.selectbox("Target Microservice:", sorted(sdg.get_services()), index=5, key="sb_sim_svc")
    sim_fault = st.sidebar.selectbox("Outage Scenario:", [
        "Database Lock Contention",
        "Payment Gateway Outage",
        "Memory Leak & OOM",
        "Extreme Latency Spike & CPU Throttling"
    ], key="sb_sim_fault")
    sim_start = st.sidebar.slider("Crash Start (Minute):", 10, 100, 45, key="sb_sim_min")

    if st.sidebar.button("💥 Inject Failure Now", key="sb_inject_btn"):
        st.session_state["simulated_telemetry"] = True
        st.session_state["sim_service"] = sim_service
        st.session_state["sim_fault"] = sim_fault
        st.session_state["sim_start"] = sim_start

# Compute active simulated telemetry if active
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

# --- Top Main Hero Banner ---
is_sim_active = st.session_state.get("simulated_telemetry", False)
sim_target = st.session_state.get("sim_service", "")
status_pill = f'<span class="status-badge badge-red">🚨 Outage on: {sim_target}</span>' if is_sim_active else '<span class="status-badge badge-green"><span class="pulse-dot"></span> All Systems Normal</span>'

st.markdown(f"""
<div class="hero-container">
    <div class="hero-title-row">
        <div>
            <div class="hero-title">⚡ RootIQ AIOps Intelligence</div>
            <div class="hero-subtitle">Autonomous Incident Correlation, Dependency Graph Analytics & Multi-Evidence Root Cause Localization</div>
        </div>
        <div class="badge-group">
            {status_pill}
            <span class="status-badge badge-blue">🌲 Isolation Forest ML</span>
            <span class="status-badge badge-purple">🕸️ 7 Microservices</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Active mode notification badge
if data_source == "📁 Upload Custom CSV":
    st.info(f"📁 **Custom Telemetry Mode Active:** `{active_source_label}`. Running unsupervised inference on user metrics.")
elif is_sim_active:
    st.warning(f"⚡ **Live Outage Simulation Active:** `{active_source_label}`. Failure cascades propagating along caller graph.")
else:
    st.success(f"📊 **Benchmark Mode Active:** `{active_source_label}`. Standard OpenTelemetry baseline active.")

# ==============================================================================
# --- PAGE 1: OVERVIEW & TOPOLOGY ---
# ==============================================================================
if menu == "⚡ Overview & Topology":
    st.subheader("System Topology & Real-Time Operational Footprint")

    col1, col2, col3, col4 = st.columns(4)
    sys_stats = get_system_metrics()
    col1.metric("Monitored Services", len(sdg.get_services()))
    col2.metric("Telemetry Data Points", f"{len(df_metrics):,}")
    col3.metric("Host RAM Footprint", f"{sys_stats['host_ram_used_gb']} GB")
    col4.metric("Engine Health", "Online (CPU-Ready)")

    st.markdown("---")
    
    # Interactive Graph Controls
    ctrl_col1, ctrl_col2 = st.columns([2, 2])
    with ctrl_col1:
        st.subheader("🕸️ Microservices Topology & Threat Map")
    with ctrl_col2:
        highlight_svc = st.selectbox("Highlight Service Call Chain:", ["-- View Full Architecture --"] + sorted(sdg.get_services()))

    pos = {
        "frontend": (0, 3),
        "api_gateway": (1.6, 3),
        "order_service": (3.2, 4.3),
        "inventory_service": (3.2, 1.7),
        "payment_service": (4.8, 4.5),
        "cache": (4.8, 1.5),
        "database": (6.2, 3)
    }

    base_colors = {
        "frontend": "#38bdf8",
        "api_gateway": "#818cf8",
        "order_service": "#c084fc",
        "inventory_service": "#a855f7",
        "payment_service": "#f43f5e",
        "cache": "#fbbf24",
        "database": "#10b981"
    }

    # Determine dynamic node status & colors based on live outage
    node_names = [node for node in sdg.graph.nodes() if node in pos]
    active_fault_svc = st.session_state.get("sim_service", None) if is_sim_active else None
    active_callers = sdg.get_downstream_dependents(active_fault_svc) if active_fault_svc else []

    node_c = []
    node_sizes = []
    node_texts = []
    node_symbols = []

    for n in node_names:
        if active_fault_svc and n == active_fault_svc:
            node_c.append("#ef4444") # Red for Root Cause
            node_sizes.append(48)
            node_texts.append(f"💥 {n}<br><b>[ROOT CAUSE]</b>")
            node_symbols.append("circle")
        elif active_fault_svc and n in active_callers:
            node_c.append("#f59e0b") # Amber for Cascading Degraded
            node_sizes.append(40)
            node_texts.append(f"⚠️ {n}<br>[DEGRADED]")
            node_symbols.append("circle")
        elif highlight_svc != "-- View Full Architecture --" and n == highlight_svc:
            node_c.append("#38bdf8")
            node_sizes.append(44)
            node_texts.append(f"🔍 {n}<br>[SELECTED]")
            node_symbols.append("diamond")
        else:
            node_c.append(base_colors.get(n, "#38bdf8"))
            node_sizes.append(34)
            node_texts.append(f"{n}")
            node_symbols.append("circle")

    # Dynamic edge styling
    edge_x = []
    edge_y = []
    edge_colors = []
    for u, v in sdg.graph.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2.5, color="rgba(148, 163, 184, 0.45)"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = [pos[node][0] for node in node_names]
    node_y = [pos[node][1] for node in node_names]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_texts,
        textposition="top center",
        textfont=dict(color="#f8fafc", size=12, family="Plus Jakarta Sans"),
        hoverinfo="text",
        marker=dict(
            symbol=node_symbols,
            size=node_sizes,
            color=node_c,
            line=dict(width=3, color="rgba(255, 255, 255, 0.6)")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=20, r=20, t=35),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=440
                    ))
    st.plotly_chart(fig, use_container_width=True)

    if is_sim_active:
        st.markdown(f"""
        <div style="display: flex; gap: 15px; padding: 10px 15px; background: rgba(15, 23, 42, 0.6); border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
            <div><span style="color: #ef4444; font-weight: bold;">● Red Node ({active_fault_svc}):</span> Primary Sabotaged Root Cause</div>
            <div><span style="color: #f59e0b; font-weight: bold;">● Amber Nodes ({', '.join(active_callers)}):</span> Cascading Failure Callers</div>
            <div><span style="color: #10b981; font-weight: bold;">● Other Nodes:</span> Unaffected Services</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Architectural Centrality & Downstream Impact Matrix")
    centrality = analyzer.compute_centrality()
    cent_df = pd.DataFrame.from_dict(centrality, orient="index").reset_index().rename(columns={"index": "Service"})
    cent_df["Downstream Dependents"] = cent_df["Service"].apply(lambda s: ", ".join(sdg.get_downstream_dependents(s)) or "None (Leaf)")
    st.dataframe(cent_df, use_container_width=True)

# ==============================================================================
# --- PAGE 2: DATA INGESTION & SIMULATOR ---
# ==============================================================================
elif menu == "📁 Data Ingestion & Simulator":
    st.subheader("Data Ingestion & Live Incident Simulator")
    st.markdown("""
    Test RootIQ with **your own telemetry CSV** or use the **1-Click Failure Presets** 
    to observe automated root cause localization in real time.
    """)

    # Interactive 1-Click Sabotage Presets
    st.markdown("#### ⚡ Quick-Action Live Outage Presets (Click to Trigger):")
    p1, p2, p3 = st.columns(3)
    if p1.button("💥 Crash Database (Pool Exhaustion)", key="quick_db"):
        st.session_state["simulated_telemetry"] = True
        st.session_state["sim_service"] = "database"
        st.session_state["sim_fault"] = "Database Lock Contention"
        st.session_state["sim_start"] = 45
        st.success("Injected Database Outage! Navigate to Anomaly Detection or Root Cause Analysis.")
        st.rerun()

    if p2.button("💳 Sabotage Payment Gateway (502 Outage)", key="quick_pay"):
        st.session_state["simulated_telemetry"] = True
        st.session_state["sim_service"] = "payment_service"
        st.session_state["sim_fault"] = "Payment Gateway Outage"
        st.session_state["sim_start"] = 40
        st.success("Injected Payment Service Outage! Navigate to Anomaly Detection or Root Cause Analysis.")
        st.rerun()

    if p3.button("🧠 Crash Order Service (Memory Leak OOM)", key="quick_order"):
        st.session_state["simulated_telemetry"] = True
        st.session_state["sim_service"] = "order_service"
        st.session_state["sim_fault"] = "Memory Leak & OOM"
        st.session_state["sim_start"] = 50
        st.success("Injected Order Service Outage! Navigate to Anomaly Detection or Root Cause Analysis.")
        st.rerun()

    st.markdown("---")
    tab1, tab2 = st.tabs(["📁 Upload Custom Telemetry CSV", "🎛️ Custom Scenario Builder"])

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
        st.subheader("⚡ Fine-Grained Failure Injection Controls")
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

        if st.button("💥 Inject Custom Failure into Pipeline", key="main_inject_btn"):
            st.session_state["simulated_telemetry"] = True
            st.session_state["sim_service"] = test_svc
            st.session_state["sim_fault"] = test_type
            st.session_state["sim_start"] = test_minute
            st.success(f"Failure injected into `{test_svc}`! Navigate to 'Anomaly Detection' or 'Root Cause Analysis' to view the AI diagnosis.")

# ==============================================================================
# --- PAGE 3: TELEMETRY & EDA ---
# ==============================================================================
elif menu == "📊 Telemetry & EDA":
    st.subheader("Operational Telemetry & Exploratory Analysis")
    st.caption(f"Active Data Mode: {active_source_label}")

    eda_tab1, eda_tab2 = st.tabs(["🔬 Single Service Deep Dive", "📊 Multi-Service Comparative Timeline"])

    with eda_tab1:
        selected_service = st.selectbox("Select Service to Inspect", sorted(df_metrics["service"].unique()))
        svc_df = df_metrics[df_metrics["service"] == selected_service].sort_values("timestamp")

        # Interactive Dual-Axis Chart (Latency + Error Rate)
        st.markdown("#### 📈 Dual-Axis Telemetry Explorer (Latency vs. Error Rate)")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(
            x=svc_df["timestamp"], y=svc_df["latency_ms"],
            name="Latency (ms)", line=dict(color="#38bdf8", width=2.5)
        ))
        fig_dual.add_trace(go.Scatter(
            x=svc_df["timestamp"], y=svc_df["error_rate"],
            name="Error Rate", yaxis="y2", line=dict(color="#f43f5e", width=2, dash="dash")
        ))
        fig_dual.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
            yaxis=dict(title=dict(text="Latency (ms)", font=dict(color="#38bdf8")), tickfont=dict(color="#38bdf8")),
            yaxis2=dict(title=dict(text="Error Rate (0-1)", font=dict(color="#f43f5e")), tickfont=dict(color="#f43f5e"), overlaying="y", side="right"),
            xaxis=dict(rangeslider=dict(visible=True), title="Timeline (Drag Range Slider to Zoom)"),
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_dual, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig_cpu = px.line(svc_df, x="timestamp", y="cpu_usage", title=f"{selected_service} - CPU Usage (%)", color_discrete_sequence=["#a855f7"], template="plotly_dark")
            fig_cpu.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cpu, use_container_width=True)
        with col4:
            fig_mem = px.line(svc_df, x="timestamp", y="memory_usage", title=f"{selected_service} - Memory Usage (%)", color_discrete_sequence=["#fbbf24"], template="plotly_dark")
            fig_mem.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_mem, use_container_width=True)

        st.subheader("Telemetry Correlation Heatmap")
        num_cols = [c for c in ["cpu_usage", "memory_usage", "latency_ms", "request_rate", "error_rate"] if c in svc_df.columns]
        if len(num_cols) > 1:
            corr = svc_df[num_cols].corr().round(2)
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", template="plotly_dark", title=f"Metric Correlation: {selected_service}")
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

    with eda_tab2:
        st.markdown("#### ⚡ Cross-Service Cascade Comparison")
        st.write("Compare multiple microservices simultaneously on the same timeline to observe how latency or errors propagate downstream.")
        
        c_svcs = st.multiselect(
            "Select Services to Compare:",
            sorted(df_metrics["service"].unique()),
            default=["payment_service", "order_service", "frontend"] if "payment_service" in df_metrics["service"].values else sorted(df_metrics["service"].unique())[:3]
        )
        c_metric = st.selectbox("Select Metric:", ["latency_ms", "error_rate", "cpu_usage", "memory_usage"])

        if c_svcs:
            comp_df = df_metrics[df_metrics["service"].isin(c_svcs)].sort_values("timestamp")
            fig_comp = px.line(
                comp_df, x="timestamp", y=c_metric, color="service",
                title=f"Comparative {c_metric.upper()} Across Selected Services",
                template="plotly_dark"
            )
            fig_comp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(rangeslider=dict(visible=True)),
                hovermode="x unified",
                height=450
            )
            st.plotly_chart(fig_comp, use_container_width=True)

# ==============================================================================
# --- PAGE 4: ANOMALY DETECTION ---
# ==============================================================================
elif menu == "🔍 Anomaly Detection":
    st.subheader("Unsupervised Anomaly Detection (Isolation Forest vs Baseline)")
    st.caption(f"Active Data Mode: {active_source_label}")

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

    st.subheader("Anomaly Score Distribution Across Microservices")
    fig_box = px.box(preds, x="service", y="anomaly_score", color="service", template="plotly_dark", title="Anomaly Score Spread per Service")
    fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Detected Anomalous Telemetry Windows")
    disp_cols = [c for c in ["timestamp", "service", "anomaly_score", "latency_ms", "error_rate", "cpu_usage"] if c in preds.columns]
    st.dataframe(preds[preds["is_anomaly"] == 1][disp_cols].head(25), use_container_width=True)

# ==============================================================================
# --- PAGE 5: TIME-SERIES & ONSET ---
# ==============================================================================
elif menu == "⏱️ Time-Series & Onset":
    st.subheader("Temporal Precedence & Anomaly Onset Sequencing")
    st.caption(f"Active Data Mode: {active_source_label}")

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
        st.subheader("Chronological Anomaly Onset Sequence")
        st.dataframe(onsets, use_container_width=True)

        st.subheader("Waterfall Timeline: Earliest Onset Lead Times")
        fig_bar = px.bar(
            onsets,
            x="service",
            y="temporal_score",
            color="temporal_score",
            color_continuous_scale="Reds",
            template="plotly_dark",
            title="Temporal Precedence Score (1.0 = Earliest Failure Originator)",
            text_auto=True
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

# ==============================================================================
# --- PAGE 6: INCIDENT CORRELATION ---
# ==============================================================================
elif menu == "🚨 Incident Correlation":
    st.subheader("Incident Correlation & Alert Clustering")
    st.caption(f"Active Data Mode: {active_source_label}")

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

# ==============================================================================
# --- PAGE 7: ROOT CAUSE ANALYSIS ---
# ==============================================================================
elif menu == "🎯 Root Cause Analysis":
    st.subheader("Multi-Evidence Root Cause Localization Engine")
    st.caption(f"Active Data Mode: {active_source_label}")

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
            top1 = ranked[0]
            affected_str = ", ".join(target_inc["affected_services"])
            onset_str = top1.get("onset_time", target_inc["start_time"])

            # --- EXECUTIVE VERDICT & INCIDENT CONCLUSION BLOCK ---
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 16px; padding: 22px 26px; margin: 10px 0 25px 0; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 16px;">
                    <div>
                        <h3 style="color: #38bdf8; margin: 0; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.3px;">📋 Incident Verdict & Root Cause Conclusion</h3>
                        <div style="color: #94a3b8; font-size: 0.88rem;">Executive summary translating multi-metric anomalies into clear business and engineering insights</div>
                    </div>
                    <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); padding: 5px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
                        ● Status: Triage Complete
                    </span>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px;">
                    
                    <!-- BOX 1: THE PROBLEM -->
                    <div style="background: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; border-radius: 8px; padding: 14px 16px;">
                        <div style="color: #f87171; font-weight: 700; font-size: 0.82rem; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">🚨 1. What Went Wrong (Problem)</div>
                        <div style="color: #f1f5f9; font-size: 0.9rem; line-height: 1.45;">
                            Widespread latency and error spikes impacted <b>{len(target_inc['affected_services'])} microservices</b> ({affected_str}) over a <b>~{target_inc['duration_minutes']}-minute outage window</b>, degrading user requests.
                        </div>
                    </div>

                    <!-- BOX 2: THE AI CONCLUSION -->
                    <div style="background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 14px 16px;">
                        <div style="color: #38bdf8; font-weight: 700; font-size: 0.82rem; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">🎯 2. The AI Verdict (Conclusion)</div>
                        <div style="color: #f1f5f9; font-size: 0.9rem; line-height: 1.45;">
                            RootIQ conclusively isolates <b><span style="color: #38bdf8; font-size: 1.05rem;">{top1['service']}</span></b> as the true root cause with <b>{top1['confidence_pct']}% certainty</b>. Downstream callers were innocent victims of this failure.
                        </div>
                    </div>

                    <!-- BOX 3: THE SMOKING GUN EVIDENCE -->
                    <div style="background: rgba(245, 158, 11, 0.08); border-left: 4px solid #f59e0b; border-radius: 8px; padding: 14px 16px;">
                        <div style="color: #fbbf24; font-weight: 700; font-size: 0.82rem; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">🔍 3. Why It Is The Culprit (Evidence)</div>
                        <div style="color: #f1f5f9; font-size: 0.9rem; line-height: 1.45;">
                            <b>• Failed First:</b> Began degrading at <code>{onset_str}</code>.<br/>
                            <b>• Upstream Caller Impact:</b> Sits upstream of affected caller services.<br/>
                            <b>• Anomaly Severity:</b> Multi-metric spike of <b>{top1['anomaly_evidence']*100:.1f}%</b>.
                        </div>
                    </div>

                    <!-- BOX 4: THE ACTIONABLE REMEDIATION -->
                    <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; border-radius: 8px; padding: 14px 16px;">
                        <div style="color: #34d399; font-weight: 700; font-size: 0.82rem; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">🛠️ 4. Recommended Fix (Remediation)</div>
                        <div style="color: #f1f5f9; font-size: 0.9rem; line-height: 1.45;">
                            • Restart container pods for <code>{top1['service']}</code>.<br/>
                            • Check database connection limits or API rate throttling.<br/>
                            • Verify circuit breaker tripping to stop cascading timeouts.
                        </div>
                    </div>

                </div>
            </div>
            """, unsafe_allow_html=True)

            # High-Impact Hero Card for Top-1 Root Cause
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%); border: 1px solid rgba(239, 68, 68, 0.45); border-radius: 14px; padding: 22px 28px; margin: 15px 0 25px 0; box-shadow: 0 8px 30px rgba(239, 68, 68, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                    <div>
                        <span style="background: #ef4444; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">🏆 Top-1 Primary Culprit</span>
                        <h1 style="color: #ffffff; margin: 10px 0 4px 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">{top1['service']}</h1>
                        <div style="color: #cbd5e1; font-size: 0.95rem;">Identified as the primary root cause behind the cascading microservices outage.</div>
                    </div>
                    <div style="display: flex; gap: 20px; text-align: right;">
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                            <div style="font-size: 1.9rem; font-weight: 800; color: #f87171;">{top1['confidence_pct']}%</div>
                            <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600;">Diagnosis Confidence</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                            <div style="font-size: 1.9rem; font-weight: 800; color: #38bdf8;">{top1['root_cause_score']:.1f}<span style="font-size: 1rem; color: #94a3b8;">/100</span></div>
                            <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600;">Composite Score</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Circular Gauge + 4D Evidence Radar side by side!
            gauge_col, radar_col = st.columns([1.2, 1.8])
            with gauge_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=top1["confidence_pct"],
                    title={"text": "AI Confidence Level", "font": {"size": 16, "color": "#f8fafc"}},
                    number={"suffix": "%", "font": {"size": 38, "color": "#f87171"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                        "bar": {"color": "#ef4444", "thickness": 0.3},
                        "bgcolor": "rgba(30, 41, 59, 0.5)",
                        "borderwidth": 2,
                        "bordercolor": "rgba(255,255,255,0.1)",
                        "steps": [
                            {"range": [0, 50], "color": "rgba(148, 163, 184, 0.2)"},
                            {"range": [50, 75], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [75, 100], "color": "rgba(239, 68, 68, 0.25)"}
                        ],
                        "threshold": {
                            "line": {"color": "#38bdf8", "width": 4},
                            "thickness": 0.8,
                            "value": top1["confidence_pct"]
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f8fafc", "family": "Plus Jakarta Sans"},
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with radar_col:
                categories = ["Temporal Precedence", "Dependency Impact", "Anomaly Magnitude", "Metric/Log Spikes"]
                values = [
                    top1["temporal_evidence"] * 100,
                    top1["dependency_evidence"] * 100,
                    top1["anomaly_evidence"] * 100,
                    top1["metric_log_evidence"] * 100
                ]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=top1['service'],
                    fillcolor="rgba(244, 63, 94, 0.3)",
                    line=dict(color="#f43f5e", width=2.5)
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155", linecolor="#475569"),
                        angularaxis=dict(gridcolor="#334155", linecolor="#475569"),
                        bgcolor="rgba(15, 23, 42, 0.5)"
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
                    showlegend=False,
                    height=300,
                    margin=dict(l=30, r=30, t=30, b=20),
                    title=dict(text=f"4D Evidence Attribution: {top1['service']}", font=dict(size=15, color="#f8fafc"))
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Interactive Failure Cascade Sankey Diagram
            st.subheader("🌊 Cascading Failure Propagation Flow")
            affected = target_inc["affected_services"]
            culprit = top1["service"]
            
            # Construct Sankey Nodes and Links dynamically
            sankey_nodes = [culprit] + [s for s in affected if s != culprit]
            node_map = {name: idx for idx, name in enumerate(sankey_nodes)}
            
            sources = []
            targets = []
            values = []
            link_colors = []

            for s in affected:
                if s != culprit:
                    sources.append(node_map[culprit])
                    targets.append(node_map[s])
                    values.append(1)
                    link_colors.append("rgba(239, 68, 68, 0.4)")

            if not sources and len(sankey_nodes) > 1:
                sources = [0]
                targets = [1]
                values = [1]
                link_colors = ["rgba(239, 68, 68, 0.4)"]

            if sources:
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=[f"{n} (Root Cause)" if n == culprit else f"{n} (Degraded)" for n in sankey_nodes],
                        color=["#ef4444" if n == culprit else "#f59e0b" for n in sankey_nodes]
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors
                    )
                )])
                fig_sankey.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
                    height=260,
                    margin=dict(l=20, r=20, t=30, b=20),
                    title=dict(text=f"Fault Propagation: {culprit} ➔ Downstream Callers", font=dict(size=14))
                )
                st.plotly_chart(fig_sankey, use_container_width=True)

            st.subheader(f"Ranked Probable Causes for `{selected_id}`")
            rank_df = pd.DataFrame(ranked)[["rank", "service", "root_cause_score", "confidence_pct", "temporal_evidence", "dependency_evidence", "anomaly_evidence", "metric_log_evidence"]]
            st.dataframe(rank_df, use_container_width=True)

            explainer = IncidentExplainer(use_local_llm=False)
            report = explainer.explain(target_inc, ranked)
            st.subheader("💡 Automated Incident Intelligence Diagnosis")
            st.markdown(report)

            # Build downloadable SRE Post-Mortem Report
            postmortem_md = f"""# SRE Incident Post-Mortem & Root Cause Analysis Report
**Generated By:** RootIQ Incident Intelligence Engine  
**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Incident Identifier:** `{selected_id}`  
**Impact Window:** `{target_inc['start_time']}` to `{target_inc['end_time']}` (~{target_inc['duration_minutes']} minutes)  
**Peak Anomaly Severity:** `{target_inc['peak_anomaly_score']:.3f}`  
**Affected Microservices ({len(target_inc['affected_services'])}):** {', '.join(target_inc['affected_services'])}

---

## 1. Executive Summary & Root Cause Finding
* **Primary Culprit (Top-1 Root Cause):** **`{top1['service']}`**
* **Root Cause Confidence:** **{top1['confidence_pct']}%**
* **Composite RCA Score:** **{top1['root_cause_score']:.1f} / 100**

### Multi-Dimensional Evidence Attribution:
* **Temporal Precedence Score:** `{top1['temporal_evidence'] * 100:.1f}%` (Service degraded first in failure cascade)
* **Dependency Impact Score:** `{top1['dependency_evidence'] * 100:.1f}%` (Sits upstream of affected caller services)
* **Isolation Forest Anomaly Magnitude:** `{top1['anomaly_evidence'] * 100:.1f}%`
* **Direct Telemetry / Log Spikes:** `{top1['metric_log_evidence'] * 100:.1f}%`

---

## 2. Ranked Root Cause Candidates
| Rank | Service | Root Cause Score | Confidence (%) | Temporal Onset | Dependency Impact | Anomaly Score | Metric Spike |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
            for r in ranked:
                postmortem_md += f"| #{r['rank']} | **{r['service']}** | {r['root_cause_score']:.1f} | {r['confidence_pct']}% | {r['temporal_evidence']*100:.1f}% | {r['dependency_evidence']*100:.1f}% | {r['anomaly_evidence']*100:.1f}% | {r['metric_log_evidence']*100:.1f}% |\n"

            postmortem_md += f"""
---

## 3. Incident Diagnosis & Propagation Narrative
{report}

---
*Report automatically compiled by RootIQ (TY B.Sc. Data Science Final Year Project).*
"""

            st.download_button(
                label="⬇️ Download Incident Post-Mortem Report (Markdown)",
                data=postmortem_md,
                file_name=f"RootIQ_PostMortem_{selected_id}.md",
                mime="text/markdown",
                key="download_postmortem_btn"
            )

# ==============================================================================
# --- PAGE 8: EVALUATION METRICS ---
# ==============================================================================
elif menu == "📈 Evaluation Metrics":
    st.subheader("Quantitative Model & System Evaluation")
    st.markdown("""
    Formal evaluation benchmark measuring Root Cause Top-1 / Top-3 accuracy and 
    Mean Reciprocal Rank (MRR) against ground-truth microservices incidents.
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
