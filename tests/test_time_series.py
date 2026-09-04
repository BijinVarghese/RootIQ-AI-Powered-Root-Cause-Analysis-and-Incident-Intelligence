"""Unit tests for Time Series and Onset Detection."""
import pytest
import pandas as pd
from src.time_series.anomaly_timeline import AnomalyTimeline
from src.time_series.change_detection import ChangeDetector

def test_anomaly_timeline_precedence():
    dates = pd.date_range("2026-03-01 10:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": [dates[0], dates[1], dates[2], dates[3]],
        "service": ["db", "db", "api", "api"],
        "is_anomaly": [1, 1, 1, 1],
        "anomaly_score": [0.9, 0.95, 0.8, 0.85]
    })
    timeline = AnomalyTimeline()
    onsets = timeline.extract_onset_times(df)
    assert len(onsets) == 2
    # db started at dates[0], api at dates[2] -> db has rank 1
    assert onsets.iloc[0]["service"] == "db"
    assert onsets.iloc[0]["temporal_precedence_rank"] == 1
    assert onsets.iloc[0]["temporal_score"] >= onsets.iloc[1]["temporal_score"]

def test_change_detector_spikes():
    s = pd.Series([10, 10, 11, 9, 10, 100, 10, 10])
    cd = ChangeDetector()
    spikes = cd.detect_spikes(s, threshold_sigmas=3.0)
    assert spikes.iloc[5] == 1