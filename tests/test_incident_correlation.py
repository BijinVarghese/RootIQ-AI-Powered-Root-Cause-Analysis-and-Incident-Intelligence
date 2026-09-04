"""Unit tests for Incident Correlation."""
import pytest
import pandas as pd
from src.incident_correlation.incident_correlator import IncidentCorrelator

def test_incident_clustering():
    times = [
        "2026-03-01 10:00:00Z",
        "2026-03-01 10:02:00Z",
        "2026-03-01 10:03:00Z",
        "2026-03-01 10:40:00Z", # Gap > 5 min -> separate incident
        "2026-03-01 10:42:00Z"
    ]
    df = pd.DataFrame({
        "timestamp": times,
        "service": ["db", "api", "frontend", "payment", "order"],
        "is_anomaly": [1, 1, 1, 1, 1],
        "anomaly_score": [0.8, 0.7, 0.6, 0.9, 0.7]
    })
    correlator = IncidentCorrelator(max_gap_minutes=5)
    incidents = correlator.correlate(df)
    assert len(incidents) == 2
    assert "db" in incidents[0]["affected_services"]
    assert "payment" in incidents[1]["affected_services"]