"""
RootIQ - Trace Preprocessor
Aggregates distributed request traces into latency percentiles (P50, P95, P99),
error counts, and service dependency call links.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class TracePreprocessor:
    def __init__(self, time_col: str = "timestamp", service_col: str = "service",
                 duration_col: str = "duration_ms", status_col: str = "status_code",
                 trace_id_col: str = "trace_id", parent_service_col: Optional[str] = "parent_service"):
        self.time_col = time_col
        self.service_col = service_col
        self.duration_col = duration_col
        self.status_col = status_col
        self.trace_id_col = trace_id_col
        self.parent_service_col = parent_service_col

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean raw trace spans."""
        df = df.copy()
        for col in [self.time_col, self.service_col, self.duration_col]:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' missing from trace telemetry.")

        df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[self.time_col])
        df[self.service_col] = df[self.service_col].astype(str).str.strip().str.lower()
        df[self.duration_col] = pd.to_numeric(df[self.duration_col], errors="coerce").clip(lower=0.0).fillna(0.0)

        if self.status_col in df.columns:
            df[self.status_col] = pd.to_numeric(df[self.status_col], errors="coerce").fillna(200).astype(int)
            df["is_trace_error"] = (df[self.status_col] >= 400).astype(int)
        else:
            df["is_trace_error"] = 0

        return df.sort_values(by=self.time_col).reset_index(drop=True)

    def aggregate_to_window(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Aggregate trace spans into windowed latency distributions."""
        cleaned = self.clean(df)
        cleaned["window"] = cleaned[self.time_col].dt.floor(freq)

        grouped = cleaned.groupby(["window", self.service_col])
        
        agg = grouped.agg(
            span_count=(self.duration_col, "count"),
            trace_error_count=("is_trace_error", "sum"),
            trace_latency_p50=(self.duration_col, lambda x: np.percentile(x, 50) if len(x) > 0 else 0),
            trace_latency_p95=(self.duration_col, lambda x: np.percentile(x, 95) if len(x) > 0 else 0),
            trace_latency_max=(self.duration_col, "max"),
        ).reset_index()

        agg["trace_error_rate"] = agg["trace_error_count"] / np.maximum(agg["span_count"], 1)
        return agg.rename(columns={"window": self.time_col})

    def extract_dependency_edges(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Extract caller -> callee service dependency pairs from traces."""
        cleaned = self.clean(df)
        if self.parent_service_col and self.parent_service_col in cleaned.columns:
            pairs = cleaned[[self.parent_service_col, self.service_col]].dropna()
            pairs = pairs[pairs[self.parent_service_col] != ""]
            unique_pairs = pairs.drop_duplicates().values.tolist()
            return [(str(p[0]).strip().lower(), str(p[1]).strip().lower()) for p in unique_pairs if p[0] != p[1]]
        return []