"""Unit tests for Feature Engineering."""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.metric_features import MetricFeatureEngineer
from src.feature_engineering.temporal_features import TemporalFeatureEngineer

def test_metric_feature_engineering():
    dates = pd.date_range("2026-03-01 10:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "service": ["database"] * 5,
        "latency_ms": [50, 60, 70, 80, 500],
        "cpu_usage": [20, 22, 25, 30, 90]
    })
    fe = MetricFeatureEngineer(rolling_windows=[3])
    res = fe.transform(df)
    assert "latency_ms_roll_mean_3" in res.columns
    assert "latency_ms_diff1" in res.columns
    assert "resource_stress_index" in res.columns or res["latency_ms_diff1"].iloc[-1] == 420

def test_temporal_features():
    dates = pd.date_range("2026-03-01 10:00", periods=4, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "service": ["auth"] * 4,
        "latency_ms": [10, 20, 30, 40]
    })
    tfe = TemporalFeatureEngineer(lags=[1])
    res = tfe.transform(df)
    assert "latency_ms_lag_1" in res.columns
    assert "hour_sin" in res.columns