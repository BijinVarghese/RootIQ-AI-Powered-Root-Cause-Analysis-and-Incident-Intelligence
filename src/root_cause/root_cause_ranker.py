"""
RootIQ - Root Cause Ranker
Ranks candidates into Top-1 and Top-3 probable causes with confidence distribution
and reciprocal rank computation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .evidence_engine import EvidenceEngine
from .root_cause_scorer import RootCauseScorer

class RootCauseRanker:
    def __init__(self, scorer: Optional[RootCauseScorer] = None):
        self.scorer = scorer or RootCauseScorer()

    def rank_candidates(self, evidence_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank all candidate services based on composite score."""
        if not evidence_dict:
            return []

        ranked_list = []
        for svc, ev in evidence_dict.items():
            composite_score = self.scorer.score_candidate(ev)
            ranked_list.append({
                "service": svc,
                "root_cause_score": composite_score,
                "onset_time": ev["onset_time"],
                "anomaly_evidence": ev["anomaly_evidence"],
                "temporal_evidence": ev["temporal_evidence"],
                "metric_log_evidence": ev["metric_log_evidence"],
                "dependency_evidence": ev["dependency_evidence"],
                "evidence_details": ev["evidence_details"],
                "propagation_paths": ev.get("propagation_paths", [])
            })

        # Sort descending by score
        ranked_list = sorted(ranked_list, key=lambda x: x["root_cause_score"], reverse=True)

        # Softmax-style confidence percentage
        scores = np.array([item["root_cause_score"] for item in ranked_list])
        exp_scores = np.exp((scores - np.max(scores)) / 15.0) # temperature scaling
        confidences = (exp_scores / np.sum(exp_scores)) * 100.0

        for i, item in enumerate(ranked_list):
            item["rank"] = i + 1
            item["confidence_pct"] = round(float(confidences[i]), 1)

        return ranked_list

    def evaluate_ranking(self, ranked_list: List[Dict[str, Any]], true_root_cause: str) -> Dict[str, Any]:
        """Compute Top-1, Top-3, and Reciprocal Rank against ground truth."""
        true_rc = true_root_cause.strip().lower()
        ranks = [item["service"].strip().lower() for item in ranked_list]

        if true_rc in ranks:
            rank_idx = ranks.index(true_rc) + 1
            reciprocal_rank = 1.0 / rank_idx
            is_top_1 = int(rank_idx == 1)
            is_top_3 = int(rank_idx <= 3)
        else:
            rank_idx = -1
            reciprocal_rank = 0.0
            is_top_1 = 0
            is_top_3 = 0

        return {
            "true_root_cause": true_root_cause,
            "predicted_root_cause": ranked_list[0]["service"] if ranked_list else "None",
            "true_rank": rank_idx,
            "top_1": is_top_1,
            "top_3": is_top_3,
            "reciprocal_rank": round(reciprocal_rank, 4)
        }