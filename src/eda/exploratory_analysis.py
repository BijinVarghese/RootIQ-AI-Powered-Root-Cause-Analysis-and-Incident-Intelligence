"""
RootIQ - Exploratory Data Analysis (EDA) Module
Provides statistical distribution profiling, metric correlation analysis,
service breakdown metrics, and normal vs incident comparative analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class ExploratoryAnalysis:
    def __init__(self, time_col: str = "timestamp", service_col: str = "service"):
        self.time_col = time_col
        self.service_col = service_col

    def compute_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate descriptive statistics for all numeric telemetry columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        stats = numeric_df.describe().T
        stats["missing_count"] = numeric_df.isna().sum()
        stats["missing_pct"] = (numeric_df.isna().mean() * 100).round(2)
        stats["skewness"] = numeric_df.skew().round(3)
        return stats.round(3)

    def compute_service_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate operational health metrics grouped by service."""
        if self.service_col not in df.columns:
            raise KeyError(f"Column '{self.service_col}' not found in telemetry.")
            
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != self.service_col]
        agg_map = {col: "mean" for col in numeric_cols}
        
        # Override specific known columns
        for c in numeric_cols:
            if "latency" in c or "cpu" in c or "memory" in c:
                agg_map[c] = "max"
            elif "error" in c:
                agg_map[c] = "sum"
            elif "request" in c:
                agg_map[c] = "sum"

        breakdown = df.groupby(self.service_col).agg(agg_map).reset_index()
        return breakdown.round(3)

    def compute_correlation_matrix(self, df: pd.DataFrame, metric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute Pearson correlation matrix across telemetry metrics."""
        if metric_cols:
            cols = [c for c in metric_cols if c in df.columns]
            subset = df[cols]
        else:
            subset = df.select_dtypes(include=[np.number])
        return subset.corr().round(3)

    def detect_unusual_periods(self, df: pd.DataFrame, metric_col: str, threshold_sigma: float = 3.0) -> pd.DataFrame:
        """Identify timestamp segments where a metric deviates by more than N sigmas."""
        if metric_col not in df.columns:
            raise KeyError(f"Metric '{metric_col}' not found in dataframe.")
            
        mean = df[metric_col].mean()
        std = df[metric_col].std()
        if std == 0 or np.isnan(std):
            return pd.DataFrame()

        upper_bound = mean + (threshold_sigma * std)
        lower_bound = max(0, mean - (threshold_sigma * std))

        anomalies = df[(df[metric_col] > upper_bound) | (df[metric_col] < lower_bound)].copy()
        anomalies["deviation_sigmas"] = ((anomalies[metric_col] - mean) / std).round(2)
        return anomalies[[self.time_col, self.service_col, metric_col, "deviation_sigmas"]] if self.service_col in df.columns else anomalies

    def compare_normal_vs_incident(self, df: pd.DataFrame, incident_start: pd.Timestamp, incident_end: pd.Timestamp) -> pd.DataFrame:
        """Compare metric mean and max between normal baseline and incident period."""
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        incident_start = pd.to_datetime(incident_start, utc=True)
        incident_end = pd.to_datetime(incident_end, utc=True)

        is_incident = (df[self.time_col] >= incident_start) & (df[self.time_col] <= incident_end)
        normal_df = df[~is_incident].select_dtypes(include=[np.number])
        incident_df = df[is_incident].select_dtypes(include=[np.number])

        comparison = []
        for col in normal_df.columns:
            norm_mean = normal_df[col].mean()
            inc_mean = incident_df[col].mean()
            inc_max = incident_df[col].max()
            pct_change = ((inc_mean - norm_mean) / (norm_mean + 1e-6)) * 100
            comparison.append({
                "metric": col,
                "normal_mean": round(norm_mean, 2),
                "incident_mean": round(inc_mean, 2),
                "incident_max": round(inc_max, 2),
                "pct_change": round(pct_change, 2)
            })

        return pd.DataFrame(comparison).sort_values(by="pct_change", ascending=False).reset_index(drop=True)