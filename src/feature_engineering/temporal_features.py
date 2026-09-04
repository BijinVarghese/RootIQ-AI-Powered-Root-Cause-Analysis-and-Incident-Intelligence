"""
RootIQ - Temporal Feature Engineering
Extracts lag terms, delta windows, and cyclical temporal encodings.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

class TemporalFeatureEngineer:
    def __init__(self, time_col: str = "timestamp", service_col: str = "service",
                 target_metrics: Optional[List[str]] = None, lags: Optional[List[int]] = None):
        self.time_col = time_col
        self.service_col = service_col
        self.target_metrics = target_metrics or ["latency_ms", "error_rate", "cpu_usage"]
        self.lags = lags or [1, 2, 3]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features and cyclical time signals."""
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)
        df = df.sort_values(by=[self.service_col, self.time_col])

        present_metrics = [c for c in self.target_metrics if c in df.columns]
        lag_dfs = []

        for service, group in df.groupby(self.service_col):
            group_lags = pd.DataFrame(index=group.index)
            for m in present_metrics:
                for lag in self.lags:
                    group_lags[f"{m}_lag_{lag}"] = group[m].shift(lag).bfill().fillna(0.0)
            lag_dfs.append(group_lags)

        if lag_dfs:
            lags_combined = pd.concat(lag_dfs, axis=0)
            df = pd.concat([df, lags_combined], axis=1)

        # Cyclical hour and day of week
        hours = df[self.time_col].dt.hour + (df[self.time_col].dt.minute / 60.0)
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        df["day_of_week"] = df[self.time_col].dt.dayofweek

        return df.fillna(0.0)