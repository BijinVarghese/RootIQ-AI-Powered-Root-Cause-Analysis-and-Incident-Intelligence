"""
RootIQ - Root Cause Scorer
Combines multi-dimensional evidence into composite root-cause scores.
"""

from typing import Dict, Any, Optional

class RootCauseScorer:
    DEFAULT_WEIGHTS = {
        "temporal": 0.35,     # Earlier onset is the strongest indicator of primary cause
        "dependency": 0.25,   # Upstream position explaining downstream callers
        "anomaly": 0.20,      # Severity of anomalous observation
        "metric_log": 0.20    # Direct error rates and latency jumps
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        total_w = sum(self.weights.values())
        # Normalize weights to sum to 1.0
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def score_candidate(self, evidence: Dict[str, Any]) -> float:
        """Compute composite score on a 0-100 scale."""
        score = (
            self.weights["temporal"] * evidence.get("temporal_evidence", 0.0) +
            self.weights["dependency"] * evidence.get("dependency_evidence", 0.0) +
            self.weights["anomaly"] * evidence.get("anomaly_evidence", 0.0) +
            self.weights["metric_log"] * evidence.get("metric_log_evidence", 0.0)
        )
        return round(score * 100.0, 2)