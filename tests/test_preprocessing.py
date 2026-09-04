"""Unit tests for Telemetry Preprocessing."""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.log_preprocessor import LogPreprocessor
from src.preprocessing.metric_preprocessor import MetricPreprocessor
from src.preprocessing.trace_preprocessor import TracePreprocessor

def test_log_preprocessor_clean():
    data = {
        "timestamp": ["2026-03-01T10:00:00Z", "2026-03-01T10:00:00Z", "invalid_date"],
        "service": ["ORDER_SERVICE", "order_service", "database"],
        "level": ["ERROR", "error", "INFO"],
        "message": ["timeout", "timeout", "ok"]
    }
    df = pd.DataFrame(data)
    lp = LogPreprocessor()
    cleaned = lp.clean(df)
    assert len(cleaned) == 1  # drops invalid date and duplicate
    assert cleaned.iloc[0]["service"] == "order_service"
    assert cleaned.iloc[0]["is_error"] == 1

def test_metric_preprocessor_clean():
    data = {
        "timestamp": ["2026-03-01T10:00:00Z", "2026-03-01T10:01:00Z"],
        "service": ["database", "database"],
        "cpu_usage": [120.0, -10.0],
        "latency_ms": [-5.0, 200.0]
    }
    df = pd.DataFrame(data)
    mp = MetricPreprocessor()
    cleaned = mp.clean(df)
    assert cleaned["latency_ms"].min() >= 0.0

def test_trace_preprocessor_aggregation():
    data = {
        "timestamp": ["2026-03-01T10:00:10Z", "2026-03-01T10:00:40Z"],
        "service": ["frontend", "frontend"],
        "duration_ms": [50.0, 150.0],
        "status_code": [200, 500]
    }
    df = pd.DataFrame(data)
    tp = TracePreprocessor()
    windowed = tp.aggregate_to_window(df, freq="1min")
    assert len(windowed) == 1
    assert windowed.iloc[0]["span_count"] == 2
    assert windowed.iloc[0]["trace_error_rate"] == 0.5