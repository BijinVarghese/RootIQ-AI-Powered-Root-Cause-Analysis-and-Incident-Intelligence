"""
RootIQ - Metric Preprocessor
Handles metrics normalization (CPU, memory, latency, requests, errors),
missing value imputation, outlier clipping, and continuous time synchronization.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

class MetricPreprocessor:
    DEFAULT_METRICS = ["cpu_usage", "memory_usage", "latency_ms", "request_rate", "error_rate"]

    def __init__(self, time_col: str = "timestamp", service_col: str = "service", metric_cols: Optional[List[str]] = None):
        self.time_col = time_col
        self.service_col = service_col
        self.metric_cols = metric_cols or self.DEFAULT_METRICS

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, sort, interpolate missing values, and clip invalid ranges."""
        df = df.copy()
        
        # Check required
        for col in [self.time_col, self.service_col]:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' missing from metric telemetry.")

        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[self.time_col])
        df[self.service_col] = df[self.service_col].astype(str).str.strip().str.lower()

        # Numeric metric columns
        present_metrics = [c for c in self.metric_cols if c in df.columns]
        if not present_metrics:
            # Fallback to any numeric column except timestamp
            present_metrics = [c for c in df.select_dtypes(include=[np.number]).columns if c != self.time_col]

        for m in present_metrics:
            df[m] = pd.to_numeric(df[m], errors="coerce")
            # Range checks: CPU/Memory/Error rates shouldn't be negative
            if "usage" in m or "rate" in m or "latency" in m:
                df[m] = df[m].clip(lower=0.0)
            if "usage" in m and (df[m].max() <= 1.0):
                # Scale 0-1 to 0-100%
                df[m] = df[m] * 100.0

        # Sort and interpolate missing values per service
        cleaned_groups = []
        for service, group in df.groupby(self.service_col):
            group = group.sort_values(by=self.time_col).drop_duplicates(subset=[self.time_col])
            group[present_metrics] = group[present_metrics].ffill().bfill().fillna(0.0)
            cleaned_groups.append(group)

        if cleaned_groups:
            result = pd.concat(cleaned_groups, ignore_index=True)
            return result.sort_values(by=[self.time_col, self.service_col]).reset_index(drop=True)
        return df

    def resample_sync(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Resample each service to uniform time step with mean/max metrics."""
        df = self.clean(df)
        present_metrics = [c for c in self.metric_cols if c in df.columns]
        
        resampled_list = []
        for service, group in df.groupby(self.service_col):
            group = group.set_index(self.time_col)
            resampled = group[present_metrics].resample(freq).agg(["mean", "max"]).ffill().bfill()
            # Flatten multi-index columns
            resampled.columns = [f"{col}_{stat}" for col, stat in resampled.columns]
            resampled[self.service_col] = service
            resampled_list.append(resampled.reset_index())

        if resampled_list:
            combined = pd.concat(resampled_list, ignore_index=True)
            return combined.sort_values(by=[self.time_col, self.service_col]).reset_index(drop=True)
        return df