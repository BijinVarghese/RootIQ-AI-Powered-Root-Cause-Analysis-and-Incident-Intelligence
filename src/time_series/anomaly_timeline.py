"""
RootIQ - Anomaly Timeline & Temporal Precedence
Identifies exact anomaly onset timestamps, calculates persistence,
and computes temporal precedence scores (earlier failure = higher root cause probability).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class AnomalyTimeline:
    def __init__(self, time_col: str = "timestamp", service_col: str = "service",
                 anomaly_col: str = "is_anomaly", score_col: str = "anomaly_score"):
        self.time_col = time_col
        self.service_col = service_col
        self.anomaly_col = anomaly_col
        self.score_col = score_col

    def extract_onset_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect the earliest timestamp when each service became anomalous,
        along with peak severity and duration of abnormal state.
        """
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        anom_rows = df[df[self.anomaly_col] == 1]

        if anom_rows.empty:
            return pd.DataFrame(columns=[
                self.service_col, "first_onset", "peak_score", "duration_minutes", "temporal_precedence_rank"
            ])

        records = []
        for service, group in anom_rows.groupby(self.service_col):
            group = group.sort_values(by=self.time_col)
            first_onset = group[self.time_col].min()
            last_onset = group[self.time_col].max()
            duration_mins = max(1, int((last_onset - first_onset).total_seconds() / 60)) + 1
            peak_score = group[self.score_col].max() if self.score_col in group.columns else 1.0

            records.append({
                self.service_col: service,
                "first_onset": first_onset,
                "peak_score": round(float(peak_score), 4),
                "duration_minutes": duration_mins,
                "anomaly_point_count": len(group)
            })

        timeline_df = pd.DataFrame(records).sort_values(by="first_onset").reset_index(drop=True)
        timeline_df["temporal_precedence_rank"] = range(1, len(timeline_df) + 1)
        
        # Calculate normalized temporal score [0, 1] where earliest = 1.0
        earliest = timeline_df["first_onset"].min()
        latest = timeline_df["first_onset"].max()
        time_span = (latest - earliest).total_seconds()
        
        if time_span > 0:
            timeline_df["temporal_score"] = timeline_df["first_onset"].apply(
                lambda t: round(1.0 - ((t - earliest).total_seconds() / time_span), 3)
            )
        else:
            timeline_df["temporal_score"] = 1.0

        return timeline_df

    def build_event_sequence(self, df: pd.DataFrame, metric_trigger_col: str = "anomaly_score") -> List[Dict[str, Any]]:
        """Generate ordered human-readable event chain."""
        onsets = self.extract_onset_times(df)
        events = []
        for _, row in onsets.iterrows():
            events.append({
                "timestamp": str(row["first_onset"]),
                "service": row[self.service_col],
                "event": f"Service '{row[self.service_col]}' exhibited anomalous behavior (peak score: {row['peak_score']})",
                "rank": row["temporal_precedence_rank"],
                "score": row["temporal_score"]
            })
        return events