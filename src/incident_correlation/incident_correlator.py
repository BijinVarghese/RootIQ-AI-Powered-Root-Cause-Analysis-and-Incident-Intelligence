"""
RootIQ - Incident Correlator
Groups multi-service telemetry alerts within temporal sliding windows,
deduplicates alerts, incorporates dependency graph topology, and creates incident entities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import networkx as nx

class IncidentCorrelator:
    def __init__(self, time_window_minutes: int = 15, max_gap_minutes: int = 5,
                 time_col: str = "timestamp", service_col: str = "service",
                 anomaly_col: str = "is_anomaly", score_col: str = "anomaly_score"):
        self.time_window_minutes = time_window_minutes
        self.max_gap_minutes = max_gap_minutes
        self.time_col = time_col
        self.service_col = service_col
        self.anomaly_col = anomaly_col
        self.score_col = score_col

    def correlate(self, df: pd.DataFrame, service_graph: Optional[nx.DiGraph] = None) -> List[Dict[str, Any]]:
        """
        Group abnormal observations into unified high-level incident entities.
        """
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        anomalies = df[df[self.anomaly_col] == 1].sort_values(by=self.time_col).reset_index(drop=True)

        if anomalies.empty:
            return []

        incidents = []
        current_alerts = [anomalies.iloc[0]]
        incident_counter = 1

        for i in range(1, len(anomalies)):
            row = anomalies.iloc[i]
            prev_row = current_alerts[-1]
            time_gap = (row[self.time_col] - prev_row[self.time_col]).total_seconds() / 60.0

            # If within gap threshold, cluster into the same incident
            if time_gap <= self.max_gap_minutes:
                current_alerts.append(row)
            else:
                # Close current incident and start next
                incidents.append(self._build_incident_record(current_alerts, incident_counter, service_graph))
                incident_counter += 1
                current_alerts = [row]

        if current_alerts:
            incidents.append(self._build_incident_record(current_alerts, incident_counter, service_graph))

        return incidents

    def _build_incident_record(self, alert_rows: List[pd.Series], seq: int, service_graph: Optional[nx.DiGraph]) -> Dict[str, Any]:
        alert_df = pd.DataFrame(alert_rows)
        start_time = alert_df[self.time_col].min()
        end_time = alert_df[self.time_col].max()
        duration_mins = max(1, int((end_time - start_time).total_seconds() / 60))
        affected_services = sorted(list(alert_df[self.service_col].unique()))
        
        date_str = start_time.strftime("%Y%m%d")
        incident_id = f"INC-{date_str}-{seq:03d}"
        peak_score = float(alert_df[self.score_col].max()) if self.score_col in alert_df.columns else 1.0

        # Graph connectivity check
        connected_cluster = True
        if service_graph and len(affected_services) > 1:
            # Check if any undirected path connects them
            undirected = service_graph.to_undirected()
            sub = undirected.subgraph([s for s in affected_services if undirected.has_node(s)])
            connected_cluster = nx.is_connected(sub) if len(sub.nodes) > 1 else False

        return {
            "incident_id": incident_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_mins,
            "total_alerts": len(alert_df),
            "affected_services": affected_services,
            "affected_count": len(affected_services),
            "peak_anomaly_score": round(peak_score, 4),
            "topology_connected": connected_cluster,
            "raw_alert_df": alert_df
        }