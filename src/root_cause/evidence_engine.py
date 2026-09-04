"""
RootIQ - Evidence Engine
Extracts 4-dimensional quantitative evidence for root-cause candidates:
1. Anomaly Evidence (severity, persistence)
2. Temporal Precedence (earliest onset lead time)
3. Metric & Log Evidence (latency/error/resource spikes)
4. Dependency Graph Evidence (callee/upstream impact on downstream callers)
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional
from ..service_graph.graph_analysis import ServiceGraphAnalyzer

class EvidenceEngine:
    def __init__(self, time_col: str = "timestamp", service_col: str = "service",
                 score_col: str = "anomaly_score", anomaly_col: str = "is_anomaly"):
        self.time_col = time_col
        self.service_col = service_col
        self.score_col = score_col
        self.anomaly_col = anomaly_col

    def extract_candidate_evidence(self,
                                  incident_df: pd.DataFrame,
                                  baseline_df: Optional[pd.DataFrame],
                                  graph_analyzer: Optional[ServiceGraphAnalyzer]) -> Dict[str, Dict[str, Any]]:
        """
        Extract normalized evidence dictionary for all affected services in the incident.
        Returns: { service_name: { 'anomaly': float, 'temporal': float, 'metric_log': float, 'dependency': float, ... } }
        """
        df = incident_df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        affected_services = sorted(list(df[self.service_col].unique()))

        if not affected_services:
            return {}

        evidence_dict = {}

        # 1. Temporal Onset Precedence
        onsets = {}
        for svc in affected_services:
            svc_rows = df[(df[self.service_col] == svc) & (df[self.anomaly_col] == 1)]
            if not svc_rows.empty:
                onsets[svc] = svc_rows[self.time_col].min()
            else:
                onsets[svc] = df[df[self.service_col] == svc][self.time_col].min()

        min_time = min(onsets.values())
        max_time = max(onsets.values())
        time_span = (max_time - min_time).total_seconds()

        # 2. Extract per-candidate evidence
        for svc in affected_services:
            svc_inc = df[df[self.service_col] == svc]
            
            # A. Anomaly Score Evidence
            peak_anom = float(svc_inc[self.score_col].max()) if self.score_col in svc_inc.columns else 0.5
            anom_ratio = float((svc_inc[self.anomaly_col] == 1).mean()) if self.anomaly_col in svc_inc.columns else 0.5
            anomaly_score = (0.6 * peak_anom) + (0.4 * anom_ratio)

            # B. Temporal Evidence: Earlier onset = higher score [0, 1]
            if time_span > 0:
                lead_sec = (onsets[svc] - min_time).total_seconds()
                temporal_score = max(0.1, 1.0 - (lead_sec / time_span))
            else:
                temporal_score = 1.0

            # C. Metric & Log Evidence: Latency, error rate, and CPU spikes
            metric_score = 0.5
            metric_details = []
            
            if "latency_ms" in svc_inc.columns:
                lat_max = svc_inc["latency_ms"].max()
                lat_mean = svc_inc["latency_ms"].mean()
                if baseline_df is not None and "latency_ms" in baseline_df.columns:
                    base_lat = baseline_df[baseline_df[self.service_col] == svc]["latency_ms"].mean()
                    if pd.notna(base_lat) and base_lat > 0:
                        jump = (lat_mean - base_lat) / base_lat
                        if jump > 1.0:
                            metric_score += 0.25
                            metric_details.append(f"Latency spike (+{int(jump*100)}%)")
                elif lat_max > 500:
                    metric_score += 0.2
                    metric_details.append(f"High peak latency ({lat_max:.1f}ms)")

            if "error_rate" in svc_inc.columns:
                err_max = svc_inc["error_rate"].max()
                if err_max > 0.05:
                    metric_score += 0.25
                    metric_details.append(f"Elevated error rate ({err_max*100:.1f}%)")

            if "error_count" in svc_inc.columns:
                err_logs = svc_inc["error_count"].sum()
                if err_logs > 0:
                    metric_score += 0.2
                    metric_details.append(f"Logged {int(err_logs)} critical errors")

            metric_score = min(1.0, metric_score)

            # D. Dependency Evidence
            if graph_analyzer is not None:
                dep_score = graph_analyzer.calculate_dependency_score(svc, affected_services)
                prop_paths = graph_analyzer.find_failure_propagation_path(svc, affected_services)
            else:
                dep_score = 0.5
                prop_paths = []

            evidence_dict[svc] = {
                "service": svc,
                "onset_time": onsets[svc].isoformat(),
                "anomaly_evidence": round(anomaly_score, 3),
                "temporal_evidence": round(temporal_score, 3),
                "metric_log_evidence": round(metric_score, 3),
                "dependency_evidence": round(dep_score, 3),
                "evidence_details": metric_details,
                "propagation_paths": prop_paths
            }

        return evidence_dict