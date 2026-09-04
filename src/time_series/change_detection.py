"""
RootIQ - Change Point & Spike Detection
Implements derivative spike detection and two-sided CUSUM algorithm
for operational telemetry change-point identification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class ChangeDetector:
    def __init__(self, time_col: str = "timestamp"):
        self.time_col = time_col

    def detect_spikes(self, series: pd.Series, threshold_sigmas: float = 3.0) -> pd.Series:
        """
        Detect point spikes where value jumps sharply from preceding rolling baseline.
        Uses lag-1 rolling statistics so the spike itself does not inflate the window.
        """
        roll_mean = series.shift(1).rolling(window=5, min_periods=1).mean().bfill()
        roll_std = series.shift(1).rolling(window=5, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
        z = (series - roll_mean).abs() / roll_std
        return (z >= threshold_sigmas).astype(int)

    def cusum(self, series: pd.Series, drift: float = 0.5, threshold: float = 4.0) -> List[int]:
        """
        Two-sided Cumulative Sum (CUSUM) change-point detector.
        Returns indices where a sustained distribution shift occurred.
        """
        values = series.fillna(0.0).values
        if len(values) < 2:
            return []

        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val < 1e-6:
            return []

        standardized = (values - mean_val) / std_val
        pos_sum = 0.0
        neg_sum = 0.0
        change_indices = []

        for i, val in enumerate(standardized):
            pos_sum = max(0.0, pos_sum + val - drift)
            neg_sum = min(0.0, neg_sum + val + drift)

            if pos_sum > threshold or neg_sum < -threshold:
                change_indices.append(i)
                pos_sum = 0.0
                neg_sum = 0.0

        return change_indices