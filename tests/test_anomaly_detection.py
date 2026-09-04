"""Unit tests for Anomaly Detection."""
import pytest
import pandas as pd
import numpy as np
from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.anomaly_detection.statistical_baseline import StatisticalBaselineDetector

def test_isolation_forest_detector():
    np.random.seed(42)
    normal = np.random.normal(50, 5, (100, 2))
    outliers = np.array([[500, 500], [600, 600]])
    data = np.vstack([normal, outliers])
    df = pd.DataFrame(data, columns=["metric1", "metric2"])

    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(df)
    preds = detector.predict(df)

    assert "is_anomaly" in preds.columns
    assert "anomaly_score" in preds.columns
    # Outliers should be detected as anomalies
    assert preds.iloc[-1]["is_anomaly"] == 1
    assert preds.iloc[-1]["anomaly_score"] > preds.iloc[0]["anomaly_score"]

def test_statistical_baseline():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-03-01", periods=10, freq="1min"),
        "service": ["db"] * 10,
        "val": [10, 10, 10, 10, 10, 10, 10, 10, 10, 100]
    })
    base = StatisticalBaselineDetector(z_threshold=2.0)
    base.fit(df, ["val"])
    preds = base.predict(df)
    assert preds.iloc[-1]["baseline_anomaly"] == 1