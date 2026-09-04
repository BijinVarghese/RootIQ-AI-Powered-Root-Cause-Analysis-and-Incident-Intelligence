"""
RootIQ - Log Preprocessor
Handles timestamp parsing, log severity normalization, missing value imputation,
and time-window aggregation of error/warning spikes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class LogPreprocessor:
    SEVERITY_MAP = {
        "DEBUG": 0,
        "INFO": 0,
        "NOTICE": 0,
        "WARN": 1,
        "WARNING": 1,
        "ERROR": 2,
        "ERR": 2,
        "CRITICAL": 3,
        "FATAL": 3,
        "SEVERE": 3
    }

    def __init__(self, time_col: str = "timestamp", service_col: str = "service", level_col: str = "level", message_col: str = "message"):
        self.time_col = time_col
        self.service_col = service_col
        self.level_col = level_col
        self.message_col = message_col

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw logs: drop duplicates, parse dates, normalize log level."""
        df = df.copy()
        
        # Check required columns
        for col in [self.time_col, self.service_col]:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' missing from log telemetry.")
                
        if self.level_col not in df.columns:
            df[self.level_col] = "INFO"
        if self.message_col not in df.columns:
            df[self.message_col] = ""

        # Normalize timestamp
        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[self.time_col])

        # Normalize strings
        df[self.service_col] = df[self.service_col].astype(str).str.strip().str.lower()
        df[self.level_col] = df[self.level_col].astype(str).str.strip().str.upper()
        df[self.message_col] = df[self.message_col].fillna("").astype(str)

        # Drop duplicates
        df = df.drop_duplicates(subset=[self.time_col, self.service_col, self.level_col, self.message_col])
        
        # Add severity score
        df["severity_score"] = df[self.level_col].map(lambda x: self.SEVERITY_MAP.get(x, 0))
        df["is_error"] = (df["severity_score"] >= 2).astype(int)
        df["is_warning"] = (df["severity_score"] == 1).astype(int)

        return df.sort_values(by=self.time_col).reset_index(drop=True)

    def aggregate_to_window(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Aggregate log counts and error rates into fixed time windows per service."""
        cleaned = self.clean(df)
        cleaned["window"] = cleaned[self.time_col].dt.floor(freq)
        
        agg = cleaned.groupby(["window", self.service_col]).agg(
            total_logs=("severity_score", "count"),
            error_count=("is_error", "sum"),
            warning_count=("is_warning", "sum"),
            max_severity=("severity_score", "max"),
            mean_severity=("severity_score", "mean")
        ).reset_index()

        agg["error_ratio"] = agg["error_count"] / np.maximum(agg["total_logs"], 1)
        return agg.rename(columns={"window": self.time_col})