"""
RootIQ Utility Helpers
Provides common I/O, timestamp normalization, telemetry window synchronization,
and system resource tracking functions.
"""

import os
import json
import psutil
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union

def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)
    return path

def load_json(filepath: str) -> Dict[str, Any]:
    """Safely load a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, filepath: str, indent: int = 4) -> None:
    """Save data to JSON with directory creation."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)

def load_csv(filepath: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """Safely load a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    return pd.read_csv(filepath, parse_dates=parse_dates)

def save_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """Save DataFrame to CSV with directory creation."""
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=index)

def standardize_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Standardize timestamp column to UTC datetime and sort chronologically."""
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' missing from dataframe.")
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    return df

def aggregate_time_window(df: pd.DataFrame, time_col: str = "timestamp", freq: str = "1min", group_col: str = "service") -> pd.DataFrame:
    """Bucket telemetry into uniform time windows per service."""
    df = standardize_timestamp(df, time_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df["window_time"] = df[time_col].dt.floor(freq)
    agg_dict = {col: ["mean", "max", "min", "std"] for col in numeric_cols}
    if "request_count" in df.columns:
        agg_dict["request_count"] = ["sum"]
    if "error_count" in df.columns:
        agg_dict["error_count"] = ["sum"]
    grouped = df.groupby(["window_time", group_col]).agg(agg_dict)
    grouped.columns = ["_".join(filter(None, col)).strip() for col in grouped.columns.values]
    return grouped.reset_index().rename(columns={"window_time": "timestamp"})

def get_system_metrics() -> Dict[str, Union[float, int]]:
    """Return current process & host memory/CPU usage for evaluation logging."""
    process = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "host_ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "host_ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "process_ram_mb": round(process.memory_info().rss / (1024 ** 2), 2),
    }

def format_evidence_summary(candidate: str, score: float, evidence_dict: Dict[str, Any]) -> str:
    """Format an interpretable text summary of candidate root-cause evidence."""
    lines = [
        f"Candidate: {candidate} (Root Cause Score: {score:.1f}/100)",
        f"  - Anomaly Evidence: {evidence_dict.get('anomaly_evidence', 'N/A')}",
        f"  - Temporal Precedence: {evidence_dict.get('temporal_evidence', 'N/A')}",
        f"  - Metric/Log Spikes: {evidence_dict.get('metric_log_evidence', 'N/A')}",
        f"  - Dependency Impact: {evidence_dict.get('dependency_evidence', 'N/A')}"
    ]
    return "\n".join(lines)