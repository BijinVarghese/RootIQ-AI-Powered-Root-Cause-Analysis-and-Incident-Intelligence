"""
RootIQ - Metric Feature Engineering
Computes rolling-window statistics, rates of change, error ratios,
and multi-metric interaction features for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

class MetricFeatureEngineer:
    def __init__(self, time_col: str = "timestamp", service_col: str = "service",
                 metric_cols: Optional[List[str]] = None, rolling_windows: Optional[List[int]] = None):
        self.time_col = time_col
        self.service_col = service_col
        self.metric_cols = metric_cols or ["latency_ms", "cpu_usage", "memory_usage", "error_rate", "request_rate"]
        self.rolling_windows = rolling_windows or [3, 5, 10]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling statistical features and ratios per service."""
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        df = df.sort_values(by=[self.service_col, self.time_col])

        present_metrics = [c for c in self.metric_cols if c in df.columns]
        feature_dfs = [df]

        for service, group in df.groupby(self.service_col):
            group_feats = pd.DataFrame(index=group.index)
            
            for col in present_metrics:
                # 1. Rate of change (delta)
                group_feats[f"{col}_diff1"] = group[col].diff().fillna(0.0)
                group_feats[f"{col}_pct_change"] = group[col].pct_change(fill_method=None).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                
                # 2. Rolling statistics
                for w in self.rolling_windows:
                    roll = group[col].rolling(window=w, min_periods=1)
                    r_mean = roll.mean()
                    r_std = roll.std().fillna(1e-6)
                    group_feats[f"{col}_roll_mean_{w}"] = r_mean
                    group_feats[f"{col}_roll_std_{w}"] = r_std
                    group_feats[f"{col}_roll_max_{w}"] = roll.max()
                    # Rolling Z-score
                    group_feats[f"{col}_zscore_{w}"] = ((group[col] - r_mean) / (r_std + 1e-6)).clip(-10, 10)

            # 3. Domain Interactions
            if "cpu_usage" in group.columns and "memory_usage" in group.columns:
                group_feats["resource_stress_index"] = (group["cpu_usage"] * group["memory_usage"]) / 100.0
            if "error_rate" in group.columns and "request_rate" in group.columns:
                group_feats["estimated_error_volume"] = group["error_rate"] * group["request_rate"]

            feature_dfs.append(group_feats)

        # Concatenate generated columns
        engineered = pd.concat([df, pd.concat(feature_dfs[1:], axis=0)], axis=1)
        # Drop duplicate column names if any
        engineered = engineered.loc[:, ~engineered.columns.duplicated()]
        return engineered.fillna(0.0)