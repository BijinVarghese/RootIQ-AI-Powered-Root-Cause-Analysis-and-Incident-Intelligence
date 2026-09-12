# RootIQ – AI-Powered Root Cause Analysis and Incident Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20FastAPI-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Isolation%20Forest%20%7C%20NetworkX-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bijinvarghese-rootiq-ai-powered-root-cause-analysis--app-dvbsel.streamlit.app/)

> 🚀 **Live Demo Web Application:** [Launch RootIQ on Streamlit Cloud](https://bijinvarghese-rootiq-ai-powered-root-cause-analysis--app-dvbsel.streamlit.app/)


> **TY B.Sc. Data Science Final-Year Project**  
> An analytical decision-support system that processes operational telemetry (logs, metrics, distributed traces), models microservice dependencies, detects anomalous behavior, correlates cascading alerts, and provides evidence-based rankings of probable root causes.

---

## 📌 Executive Summary & Problem Statement

Modern distributed architectures generate overwhelming volumes of telemetry data. When a failure occurs (e.g. database connection pool exhaustion), secondary errors cascade across dependent services (e.g., API Gateway, Order Service, Frontend). Traditional monitoring systems trigger multiple disconnected alerts without clearly identifying the primary component that caused the failure.

**RootIQ** solves this problem by combining:
1. **Unsupervised Machine Learning** (Isolation Forest) for multi-metric anomaly detection.
2. **Time-Series Analysis** to determine earliest anomaly onset and establish temporal precedence.
3. **Service Dependency Graph Analytics** (NetworkX) to track failure propagation along the call graph.
4. **Multi-Evidence Fusion Engine** that computes composite root-cause scores:
   $$\text{Score}(s) = w_{\text{temporal}} S_{\text{temporal}} + w_{\text{dependency}} S_{\text{dependency}} + w_{\text{anomaly}} S_{\text{anomaly}} + w_{\text{metric}} S_{\text{metric}}$$
5. **Human-in-the-Loop Decision Support** with Top-1 & Top-3 candidate ranking, confidence scores, and natural-language incident explanations.

---

## 🏛️ System Architecture

```mermaid
flowchart TB
    subgraph Layer1 ["1. TELEMETRY INGESTION & SIMULATION LAYER"]
        direction TB
        B["📊 Benchmark Telemetry Generator<br/>(7 Microservices • 180 Min Timeline)"]
        U["📁 Custom Telemetry CSV Uploader<br/>(Dynamic External Telemetry Ingestion)"]
        S["⚡ Live Failure Injection Simulator<br/>(Real-Time Interactive Sabotage Engine)"]
        
        M[("📈 Metrics Stream<br/>CPU • RAM • Latency • Errors")]
        L[("📜 Log Stream<br/>INFO • WARN • ERROR • CRITICAL")]
        T[("🔗 Distributed Traces<br/>Call Durations • HTTP Codes")]
        G[("🕸️ NetworkX Service Graph<br/>Caller-Callee Directed Topology")]

        B --> M & L & T
        U --> M
        S --> M
    end

    subgraph Layer2 ["2. PREPROCESSING & FEATURE ENGINEERING LAYER"]
        direction TB
        P["🧹 Time-Window Synchronizer<br/>Timestamp Alignment • Forward/Backward Imputation"]
        FE["🔬 Multi-Dimensional Feature Engineer<br/>• Rolling Z-Scores (3m, 5m, 10m)<br/>• First-Order Rate of Change (Δ)<br/>• Stress Index (CPU × Memory)<br/>• Error Volume Estimation"]
        
        M & L & T --> P
        P --> FE
        G --> FE
    end

    subgraph Layer3 ["3. UNSUPERVISED ML & TEMPORAL SEQUENCING LAYER"]
        direction TB
        IF["🌲 Isolation Forest Model<br/>Multivariate Anomaly Score [0, 1]<br/>(n_estimators=150, contamination=0.15)"]
        SB["📉 Statistical Baseline Detector<br/>Rolling 3-Sigma Z-Score Threshold (Z > 2.5)"]
        TO["⏱️ Temporal Onset Sequencing<br/>Chronological Earliest-Degradation Lead Time<br/>Precedence Score [0, 1]"]
        
        FE --> IF & SB
        IF --> TO
    end

    subgraph Layer4 ["4. INCIDENT CORRELATION & MULTI-EVIDENCE ENGINE LAYER"]
        direction TB
        IC["🚨 Spatio-Temporal Incident Correlator<br/>Sliding Time Window + Graph Reachability<br/><b>97.6% Alert Noise Reduction</b>"]
        
        subgraph Evidence ["🎯 4-Dimensional Evidence Attribution"]
            E1["⏱️ Temporal Precedence (35%)<br/>Earliest Anomaly Originator"]
            E2["🕸️ Dependency Impact (25%)<br/>Upstream Caller Impact Centrality"]
            E3["🌲 Anomaly Magnitude (20%)<br/>Isolation Forest Peak Severity"]
            E4["💥 Metric/Log Spikes (20%)<br/>Direct Error & Latency Jumps"]
        end
        
        RCA["🏆 Probable Cause Scorer & Ranker<br/>Top-1 & Top-3 Candidates with Confidence %"]

        TO --> IC
        G --> IC
        IC --> Evidence
        Evidence --> RCA
    end

    subgraph Layer5 ["5. DECISION SUPPORT & PRESENTATION LAYER"]
        direction TB
        UI["⚡ Interactive Streamlit Dashboard<br/>8 Analytical Pages • Topology • Live Sabotage"]
        API["🚀 FastAPI REST Endpoints<br/>/health • /topology • /incidents • /root_cause"]
        RPT["📄 Automated SRE Post-Mortem<br/>One-Click Markdown Incident Investigation Report"]
        
        RCA --> UI & API & RPT
    end

    %% Styling Classes for Dark and Light Theme Clarity
    classDef l1 fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef l2 fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef l3 fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef l4 fill:#3b0764,stroke:#e879f9,stroke-width:2px,color:#f8fafc;
    classDef l5 fill:#451a03,stroke:#fb923c,stroke-width:2px,color:#f8fafc;

    class B,U,S,M,L,T,G l1;
    class P,FE l2;
    class IF,SB,TO l3;
    class IC,E1,E2,E3,E4,RCA l4;
    class UI,API,RPT l5;
```

### 🧩 Architectural Pipeline Specification

| Architectural Layer | Core Responsibilities | Applied Algorithms & Tech | Primary Output |
| :--- | :--- | :--- | :--- |
| **1. Ingestion & Simulation** | OpenTelemetry metric, log, and trace ingestion; dynamic CSV parsing; real-time failure sabotage | `pandas`, `OpenTelemetry` standard schemas, synthetic fault injector | Synchronized raw telemetry streams & service graph |
| **2. Preprocessing & Features** | Window alignment, imputation, rolling statistics, metric interactions | Rolling Z-Scores (3m, 5m, 10m), rate of change ($\Delta$), Resource Stress Index | 35+ engineered temporal & structural features |
| **3. Detection & Sequencing** | Multivariate anomaly isolation, comparative statistical baselines, chronological onset ordering | `IsolationForest`, `RobustScaler`, 3-$\sigma$ Z-score, Anomaly Timeline Sequencer | Continuous anomaly scores & Temporal Precedence scores |
| **4. Correlation & Root Cause** | Spatio-temporal alert clustering, graph reachability, 4-dimensional weighted evidence fusion | Connected component clustering, `NetworkX` DAG traversal, Multi-Evidence Engine | Unified Incident Entities & Ranked Root Cause Candidates |
| **5. Decision Support & APIs** | Interactive observability dashboard, automated post-mortem exports, production REST integration | `Streamlit`, `Plotly`, `FastAPI`, automated SRE Post-Mortem generator | Web UI, RESTful endpoints, exportable incident reports |

---

## 📂 Project Directory Structure

```text
RootIQ/
├── app.py                              # Main interactive Streamlit dashboard
├── requirements.txt                    # Python dependencies
├── README.md                           # Comprehensive documentation
├── .gitignore                          # Git ignore rules
├── data_generator.py                   # Automated synthetic telemetry generator
│
├── data/
│   ├── raw/
│   │   ├── logs/                       # Raw telemetry logs (CSV)
│   │   ├── metrics/                    # Raw telemetry metrics (CSV)
│   │   └── traces/                     # Raw telemetry traces (CSV)
│   ├── processed/                      # Cleaned & windowed telemetry
│   └── evaluation/
│       └── labelled_incidents/         # Ground-truth benchmark incidents
│
├── src/
│   ├── preprocessing/                  # Log, metric, and trace cleaners
│   ├── eda/                            # Descriptive statistics & correlation
│   ├── feature_engineering/            # Rolling statistics, lags, graph density
│   ├── anomaly_detection/              # Isolation Forest & Z-Score baseline
│   ├── time_series/                    # Anomaly timeline & change point detection
│   ├── service_graph/                  # NetworkX dependency graph & blast radius
│   ├── incident_correlation/           # Temporal-topology alert clustering
│   ├── root_cause/                     # Multi-source evidence engine & ranker
│   ├── explanation/                    # Rule-based & local Qwen2.5 LLM explainer
│   └── utils/                          # Common I/O & system resource helpers
│
├── models/
│   ├── anomaly_detection/              # isolation_forest.pkl
│   ├── root_cause/                     # root_cause_model.pkl
│   └── llm/                            # Setup instructions for local LLMs
│
├── evaluation/
│   ├── anomaly_detection/              # evaluate_anomalies.py
│   ├── root_cause/                     # evaluate_root_cause.py
│   ├── incident_correlation/           # evaluate_incidents.py
│   └── metrics.py                      # Precision, Recall, F1, MRR library
│
├── tests/                              # 12 automated unit tests (pytest)
├── notebooks/                          # 8 educational academic Jupyter notebooks
├── outputs/evaluation_results/         # Pre-computed evaluation JSON benchmarks
└── api/
    └── routes.py                       # FastAPI REST endpoints
```

---

## 🚀 Quick Start & Installation

### 1. Clone or Open the Project
```bash
cd RootIQ
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Telemetry & Train Models
```bash
# Synthesizes microservices telemetry and 2 ground-truth incident scenarios
python data_generator.py

# Run full evaluation and serialize trained model pickles
python evaluation/anomaly_detection/evaluate_anomalies.py
python evaluation/root_cause/evaluate_root_cause.py
python evaluation/incident_correlation/evaluate_incidents.py
```

---

## 🖥️ Running the Application

### Option A: Interactive Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`. Features include:
- **Topology Map**: Interactive Plotly call graph with centrality metrics.
- **Telemetry Explorer**: Real-time metric time-series and correlation heatmaps.
- **Anomaly Detection**: Isolation Forest scores vs Z-Score baseline.
- **Onset Timeline**: Chronological waterfall chart revealing which service failed first.
- **Incident Correlator**: Alert clustering with alert noise compression.
- **Root Cause Engine**: Top-1 and Top-3 ranked root causes with radar charts of multi-source evidence.
- **Evaluation Dashboard**: Precision, Recall, F1, Top-k accuracy, and MRR.

### Option B: FastAPI Backend Endpoints
```bash
python -m uvicorn api.routes:app --reload --port 8000
```
Interactive Swagger docs available at `http://localhost:8000/docs`.
- `GET /health` - System resource status.
- `GET /topology` - Directed service call graph.
- `GET /incidents` - Clustered incident list.
- `GET /root_cause/{incident_id}` - Top-1/Top-3 ranked causes with evidence.
- `GET /evaluation` - Anomaly and root cause benchmark metrics.

### Option C: Run Unit Tests
```bash
python -m pytest tests/ -v
```
Runs 12 automated unit tests across preprocessing, feature engineering, models, time-series, graph analytics, and root-cause ranking.

### Option D: Run Academic Jupyter Notebooks
```bash
jupyter lab notebooks/
```
Walk through notebooks `01_data_exploration.ipynb` through `08_model_evaluation.ipynb`.

---

## 📊 Benchmark Evaluation Results

Evaluated on realistic microservices cascading failure benchmark (`INC-20260301-001` database lock spike & `INC-20260301-002` payment outage):

| Module | Metric | Result | Target |
| :--- | :--- | :--- | :--- |
| **Root Cause Ranking** | **Top-1 Accuracy** | **100.0%** | > 80% |
| **Root Cause Ranking** | **Top-3 Accuracy** | **100.0%** | > 90% |
| **Root Cause Ranking** | **Mean Reciprocal Rank (MRR)** | **1.0000** | > 0.85 |
| **Incident Correlation** | **Alert Noise Compression** | **97.6%** | > 80% |
| **Anomaly Detection** | **Baseline Z-Score ROC-AUC** | **0.9910** | > 0.85 |
| **System Footprint** | **Process RAM Usage** | **~85 MB** | < 1000 MB |

---

## 🛠️ Step-by-Step Guide: How to Push this Project to GitHub

Follow these steps to upload your project to your GitHub account:

### Step 1: Initialize Git in your project folder
Open PowerShell in `C:\Users\bijin\.gemini\antigravity\scratch\RootIQ` and run:
```powershell
git init
```

### Step 2: Configure your Git Identity (if not already done)
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 3: Stage all project files
```powershell
git add .
```
*(The included `.gitignore` will automatically prevent large cache files and virtual environments from being tracked).*

### Step 4: Create your initial commit
```powershell
git commit -m "Initial commit: RootIQ AI-Powered Root Cause Analysis System"
```

### Step 5: Create a new repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. Name the repository **RootIQ** (e.g., `RootIQ`).
3. Set visibility to **Public** (or Private).
4. **Do NOT** check "Add a README file" or "Add .gitignore" (we already have them).
5. Click **Create repository**.

### Step 6: Link and Push to GitHub
Copy the commands shown on GitHub and run them:
```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/RootIQ.git
git push -u origin main
```

Your complete TY B.Sc. Data Science project is now live on GitHub! 🎉

---

## 🎓 Academic Viva & Presentation Highlights
When presenting RootIQ to evaluators:
1. **Explain the Hybrid Architecture**: RootIQ is Data-Science first. It does not rely on a black-box model; it synthesizes 4 explicit analytical dimensions (Anomaly magnitude, Temporal onset order, Metric/log spikes, and Service graph reachability).
2. **Highlight the Distinction Between Cause & Symptom**: Explain how a downstream caller service (e.g. Frontend) has massive error rates, but RootIQ ranks the upstream database as the root cause because the database became anomalous *first* in time and sits upstream in the dependency graph.
3. **Showcase the Decision-Support Role**: The system computes confidence percentages and transparent evidence breakdowns rather than claiming absolute certainty.
4. **Demonstrate Offline / Edge Portability**: The system runs completely locally on modest laptop hardware (~85 MB process RAM) without requiring paid cloud APIs.