"""
RootIQ - Statistical Baseline Anomaly Detector
Rolling Z-score and threshold baseline for performance comparison with ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

class StatisticalBaselineDetector:
    def __init__(self, z_threshold: float = 3.0, service_col: str = "service", time_col: str = "timestamp"):
        self.z_threshold = z_threshold
        self.service_col = service_col
        self.time_col = time_col
        self.baselines: dict = {}

    def fit(self, df: pd.DataFrame, metric_cols: Optional[List[str]] = None) -> "StatisticalBaselineDetector":
        """Compute normal mean and std for each service and metric."""
        if metric_cols is None:
            metric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["is_anomaly", "anomaly_score"]]

        for service, group in df.groupby(self.service_col):
            self.baselines[service] = {}
            for col in metric_cols:
                mean = group[col].mean()
                std = group[col].std()
                self.baselines[service][col] = {
                    "mean": float(mean),
                    "std": float(std) if std > 0 else 1.0
                }
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using z-score threshold across metrics."""
        df = df.copy()
        df["baseline_z_max"] = 0.0
        df["baseline_anomaly"] = 0

        for idx, row in df.iterrows():
            service = row.get(self.service_col)
            svc_baseline = self.baselines.get(service, {})
            max_z = 0.0

            for col, stats in svc_baseline.items():
                if col in row:
                    val = float(row[col])
                    z = abs((val - stats["mean"]) / stats["std"])
                    if z > max_z:
                        max_z = z

            df.at[idx, "baseline_z_max"] = round(max_z, 3)
            df.at[idx, "baseline_anomaly"] = 1 if max_z >= self.z_threshold else 0

        return df