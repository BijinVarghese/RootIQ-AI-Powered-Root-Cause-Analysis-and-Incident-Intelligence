# 🎓 RootIQ: Complete Viva Presentation & Project Defense Cheatsheet
**Project Title:** RootIQ – AI-Powered Root Cause Analysis and Incident Intelligence  
**Degree:** Final Year B.Sc. Data Science  
**Author:** Bijin Varghese  

---

## ⏱️ Section 1: The 60-Second Elevator Pitch
*(Memorize this for the opening question: "Can you briefly explain what your project does?")*

> "Good morning examiners. In modern distributed cloud architectures like Netflix or Amazon, hundreds of microservices communicate continuously. When a single component fails—like a database running out of connection pools—it triggers cascading errors across dependent services, flooding engineers with hundreds of simultaneous alerts. This is called **alert fatigue**, and finding the true culprit takes hours.
>
> **RootIQ** is an end-to-end AIOps decision-support system. It monitors multi-source telemetry—metrics, logs, and distributed traces—detects anomalies using unsupervised **Isolation Forest**, traces the failure propagation order using **temporal precedence**, models the architecture using **NetworkX dependency graphs**, and fuses all 4 evidence dimensions into a ranked root cause with **100% Top-1 accuracy** and **97.6% alert noise reduction**."

---

## 🎙️ Section 2: 5-Minute Presentation Script (Word-for-Word)

* **Minute 1: The Problem & Motivation**
  * *"In distributed microservices, failures are not isolated; they cascade. When an upstream service fails, every downstream caller also logs 500 errors and high latency. Traditional monitoring tools tell engineers **what** is broken, but cannot tell **why** or **where** it started."*
* **Minute 2: System Architecture & Ingestion**
  * *"RootIQ ingests telemetry across three pillars: OpenTelemetry metrics (CPU, memory, latency, error rates), application logs, and distributed trace durations. It also builds a directed service call graph representing microservice dependencies."*
* **Minute 3: Machine Learning & Time-Series Engine**
  * *"First, we run Feature Engineering—computing rolling Z-scores, rates of change, and lagged differences. Next, our unsupervised Isolation Forest identifies anomalous metric spikes. Crucially, RootIQ applies the **Temporal Precedence Principle**: the service that begins degrading first in a failure propagation window has the highest probability of being the primary root cause."*
* **Minute 4: Multi-Evidence Fusion & Incident Correlation**
  * *"RootIQ's Incident Correlator clusters individual alerts into unified Incident Entities, compressing 126 raw alerts into just 3 incidents (a 97.6% noise reduction). Then, our Multi-Evidence Engine computes a composite score across 4 dimensions: Temporal Precedence, Architectural Dependency, Anomaly Severity, and Direct Metric/Log Spikes."*
* **Minute 5: Live Demonstration & Results**
  * *"Let me demonstrate this live on our deployed web application..." (Transition smoothly into Section 3).*

---

## 🖥️ Section 3: Step-by-Step Live Demo Guide (Flawless Presentation Flow)

1. **Open Live App:** Navigate to `https://bijinvarghese-rootiq-ai-powered-root-cause-analysis--app-dvbsel.streamlit.app/`
2. **Step 1 - Overview & Topology (Tab 1):**
   * Show the 7-service interactive graph: `frontend` $\rightarrow$ `api_gateway` $\rightarrow$ `order_service` $\rightarrow$ `database` & `payment_service`.
   * Point out the **Centrality Matrix** showing how a failure in `database` impacts 5 downstream services.
3. **Step 2 - Live Failure Sabotage (Sidebar):**
   * Select **`⚡ Live Failure Simulator`**.
   * Pick **`payment_service`**, scenario **`Payment Gateway Outage`**, and click **`🔥 Inject Failure Now`**.
   * Highlight the confirmation banner: `Active Outage on: payment_service`.
4. **Step 3 - Anomaly Detection (Tab 4):**
   * Show the Isolation Forest anomaly score jump and compare it against the statistical 3-sigma baseline.
5. **Step 4 - Time-Series & Onset (Tab 5):**
   * Show the **Chronological Onset Waterfall Chart**. Point out that `payment_service` has a Temporal Score of **1.0** because it failed before caller services.
6. **Step 5 - Root Cause Analysis (Tab 7):**
   * Show that RootIQ pinpointed **`payment_service` as the #1 Probable Root Cause** with 90%+ confidence.
   * Point to the **Evidence Radar Chart** displaying the balance of all 4 dimensions.
   * Click **`⬇️ Download Incident Post-Mortem Report (Markdown)`** to show the professional incident summary.

---

## 📐 Section 4: Mathematical & Algorithmic Foundations

### 1. Isolation Forest Anomaly Score
The anomaly score $s$ for an instance $x$ across a forest of $n$ isolation trees is:
$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
Where:
* $h(x)$ is the path length (number of splits) to isolate sample $x$. Anomalies isolate near the root with short path lengths.
* $\mathbb{E}(h(x))$ is the average path length across all isolation trees.
* $c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree (normalizing constant).
* If $s \to 1$, the instance is strongly anomalous; if $s < 0.5$, the instance is normal.

### 2. Multi-Evidence Root Cause Fusion Formula
The composite root cause score for candidate microservice $s_i$ in incident entity $I$ is:
$$\text{Score}(s_i) = w_{\text{temp}} \cdot S_{\text{temporal}}(s_i) + w_{\text{dep}} \cdot S_{\text{dependency}}(s_i) + w_{\text{anom}} \cdot S_{\text{anomaly}}(s_i) + w_{\text{met}} \cdot S_{\text{metric}}(s_i)$$
Where:
* $w_{\text{temp}} = 0.35$ (Temporal Onset Precedence)
* $w_{\text{dep}} = 0.25$ (Downstream Dependency Reachability)
* $w_{\text{anom}} = 0.20$ (Isolation Forest Anomaly Severity)
* $w_{\text{met}} = 0.20$ (Direct Latency / Error / Log Ratio)
* $\sum w_k = 1.0$

### 3. Mean Reciprocal Rank (MRR)
Evaluates ranking quality across benchmark incidents $Q$:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
When the true root cause is always ranked #1 ($\text{rank}_1 = 1$), $\text{MRR} = 1.000$.

---

## ❓ Section 5: Top 15 Tough Viva Questions & Model Answers

#### Q1: Why did you choose Isolation Forest instead of Supervised ML (e.g., Random Forest, XGBoost)?
> *"Supervised learning requires massive volumes of labeled incident data, which is rarely available in production microservices because real outages are rare and zero-day failures take novel forms. Isolation Forest is completely unsupervised—it detects anomalies purely based on the mathematical principle that anomalous points are few and distinct in feature space."*

#### Q2: Why didn't you use Deep Learning like LSTM or Autoencoders?
> *"Deep neural networks like LSTMs require substantial GPU computational resources, take hours to train, and act as opaque black boxes. In an SRE triage setting, latency is critical: RootIQ's Isolation Forest and NetworkX algorithms execute on standard CPUs in milliseconds, have a negligible RAM footprint (<200MB), and provide transparent, explainable evidence."*

#### Q3: What happens if two microservices fail at the exact same second?
> *"If two services experience an anomaly at the identical timestamp, their Temporal Precedence score ties. RootIQ breaks this tie using the Service Dependency Graph: the upstream callee service (the one that supplies data to the caller) receives a higher Dependency Impact score than the downstream caller."*

#### Q4: How do you handle cyclic dependencies in your microservice graph?
> *"Our service graph is modeled using NetworkX. We compute shortest path reachability and node betweenness centrality. In modern microservice architectures, acyclic call graphs are standard; if cyclic calls occur, NetworkX's cycle detection algorithms prevent infinite loops by tracking visited nodes during depth-first traversal."*

#### Q5: What is 'Alert Fatigue' and how does RootIQ solve it quantitatively?
> *"Alert fatigue occurs when a single failure triggers hundreds of alerts across dependent services, overwhelming on-call engineers. RootIQ uses spatio-temporal clustering via `IncidentCorrelator` to merge temporally overlapping and topologically connected alerts into unified incident groups. On our benchmark, it compressed 126 raw alerts into 3 actionable incidents—a **97.62% noise reduction**."*

#### Q6: Are the weights (0.35, 0.25, 0.20, 0.20) hardcoded or configurable?
> *"They have robust empirical defaults based on SRE failure dynamics: temporal onset is the strongest indicator in cascading failures. However, RootIQ provides interactive sliders in the sidebar so operators can dynamically adjust weights based on system behavior (e.g., placing higher weight on logs during silent corruptions)."*

#### Q7: How is the ground-truth benchmark created?
> *"We generated a realistic 7-microservice OpenTelemetry dataset simulating 180 time steps. We injected three controlled real-world failure scenarios: Database Connection Pool Exhaustion, Payment Gateway Timeout, and Memory Leak with OOM. The injected service and failure window serve as the ground-truth labels for Top-k accuracy and MRR evaluation."*

#### Q8: What feature engineering techniques did you apply?
> *"We engineered rolling-window mean, standard deviation, rolling Z-scores across 3, 5, and 10-minute windows, first-order difference rates of change, percentage change, and cross-metric domain interactions like Resource Stress Index (CPU × RAM) and Estimated Error Volume."*

#### Q9: What are Top-1 and Top-3 accuracy?
> *"Top-1 accuracy measures whether the true root cause was ranked as the #1 candidate. Top-3 accuracy measures whether the true root cause appeared within the top 3 recommendations. RootIQ achieved **100% Top-1** and **100% Top-3** accuracy across all benchmark test incidents."*

#### Q10: How does the system handle missing data or sensor dropouts?
> *"Our `MetricPreprocessor` handles missing timestamps and telemetry gaps through forward-filling (`ffill`) followed by backward-filling (`bfill`), ensuring temporal continuity for rolling statistical windows without introducing synthetic noise."*

#### Q11: What is the purpose of the Statistical Baseline Detector?
> *"It serves as a benchmark comparator against our machine learning model. It uses a rolling 3-sigma Z-Score ($Z > 2.5$). Isolation Forest outperforms the statistical baseline because it captures multi-metric interactions simultaneously, whereas Z-scores only evaluate one metric at a time."*

#### Q12: How can an enterprise deploy RootIQ into production?
> *"RootIQ is production-ready. Telemetry can be streamed from Prometheus, OpenTelemetry Collector, or Elasticsearch via our FastAPI REST endpoints (`/incidents`, `/root_cause/{id}`). SREs interact through the Streamlit dashboard or automated Slack/PagerDuty webhooks."*

#### Q13: What happens if a user uploads a CSV with services not present in the dependency graph?
> *"If an unknown service is uploaded, RootIQ dynamically registers the new service. For dependency evidence, it defaults to a neutral score (0.0), allowing the temporal onset, anomaly magnitude, and metric spike dimensions to isolate the root cause."*

#### Q14: How does RootIQ generate the incident explanation?
> *"It uses our `IncidentExplainer` engine, which formats the multi-dimensional evidence into an executive post-mortem narrative diagnosing the culprit, affected downstream callers, failure duration, and recommended SRE remediation action items."*

#### Q15: What are the future enhancements of this project?
> *"1. Dynamic causal graph discovery using PC-algorithm or Granger Causality instead of static dependency graphs.  
> 2. Automated self-healing remediation triggers via Kubernetes Operators (e.g., auto-restarting crashed pods).  
> 3. Real-time streaming ingestion using Apache Kafka."*

---

## 📊 Section 6: Key Numbers to Memorize

| Metric | Value | Meaning |
| :--- | :---: | :--- |
| **Top-1 Root Cause Accuracy** | **100.0%** | The true root cause was always the #1 ranked candidate. |
| **Top-3 Root Cause Accuracy** | **100.0%** | The true root cause was always in the top 3 candidates. |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | Perfect ranking score across all evaluation incidents. |
| **Alert Noise Reduction** | **97.62%** | Compressed 126 raw alert triggers into 3 unified incidents. |
| **Monitored Services** | **7 Services** | `frontend`, `api_gateway`, `order_service`, `inventory_service`, `payment_service`, `cache`, `database`. |
| **Automated Unit Tests** | **12 / 12 (100%)** | Complete coverage of preprocessing, ML, graph, and ranking. |
| **Process RAM Footprint** | **~150 MB** | Extremely lightweight, runs on any student laptop or free cloud tier. |
