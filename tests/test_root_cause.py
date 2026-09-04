"""Unit tests for Root Cause Ranking."""
import pytest
import pandas as pd
from src.root_cause.evidence_engine import EvidenceEngine
from src.root_cause.root_cause_scorer import RootCauseScorer
from src.root_cause.root_cause_ranker import RootCauseRanker

def test_root_cause_ranking():
    evidence = {
        "database": {
            "service": "database",
            "onset_time": "2026-03-01T10:00:00Z",
            "anomaly_evidence": 0.95,
            "temporal_evidence": 1.0,   # Failed first!
            "metric_log_evidence": 0.90,
            "dependency_evidence": 0.85,
            "evidence_details": ["Connection pool exhausted"]
        },
        "frontend": {
            "service": "frontend",
            "onset_time": "2026-03-01T10:03:00Z",
            "anomaly_evidence": 0.70,
            "temporal_evidence": 0.2,   # Failed later as downstream effect
            "metric_log_evidence": 0.60,
            "dependency_evidence": 0.20,
            "evidence_details": ["HTTP 504 timeout"]
        }
    }

    scorer = RootCauseScorer()
    ranker = RootCauseRanker(scorer)
    ranked = ranker.rank_candidates(evidence)

    assert len(ranked) == 2
    assert ranked[0]["service"] == "database"
    assert ranked[0]["root_cause_score"] > ranked[1]["root_cause_score"]
    assert ranked[0]["confidence_pct"] > ranked[1]["confidence_pct"]

    eval_res = ranker.evaluate_ranking(ranked, true_root_cause="database")
    assert eval_res["top_1"] == 1
    assert eval_res["top_3"] == 1
    assert eval_res["reciprocal_rank"] == 1.0