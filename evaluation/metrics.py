"""
RootIQ - Evaluation Metrics Library
Computes formal evaluation metrics for AIOps modules:
1. Anomaly Detection: Precision, Recall, F1, FPR, ROC-AUC
2. Root Cause Analysis: Top-1 Accuracy, Top-3 Accuracy, MRR
3. Incident Correlation: Grouping Accuracy, Alert Compression Ratio
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import List, Dict, Any, Union, Optional

def calculate_anomaly_metrics(y_true: List[int], y_pred: List[int], y_scores: Optional[List[float]] = None) -> Dict[str, float]:
    """Calculate standard binary classification metrics for anomaly detection."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    roc_auc = 0.5
    if y_scores is not None and len(np.unique(y_true_arr)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_scores))
        except Exception:
            roc_auc = 0.5

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "roc_auc": round(roc_auc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

def calculate_root_cause_metrics(evaluation_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes Top-1 Accuracy, Top-3 Accuracy, and Mean Reciprocal Rank (MRR).
    Each record must contain: {'top_1': 1/0, 'top_3': 1/0, 'reciprocal_rank': float}
    """
    if not evaluation_records:
        return {"top_1_accuracy": 0.0, "top_3_accuracy": 0.0, "mean_reciprocal_rank": 0.0, "total_incidents": 0}

    n = len(evaluation_records)
    top_1_acc = sum(r.get("top_1", 0) for r in evaluation_records) / n
    top_3_acc = sum(r.get("top_3", 0) for r in evaluation_records) / n
    mrr = sum(r.get("reciprocal_rank", 0.0) for r in evaluation_records) / n

    return {
        "total_incidents": n,
        "top_1_accuracy": round(float(top_1_acc), 4),
        "top_3_accuracy": round(float(top_3_acc), 4),
        "mean_reciprocal_rank": round(float(mrr), 4)
    }

def calculate_correlation_metrics(total_alerts: int, grouped_incidents: int) -> Dict[str, float]:
    """Calculate alert deduplication and compression ratio."""
    compression = ((total_alerts - grouped_incidents) / total_alerts) * 100 if total_alerts > 0 else 0.0
    return {
        "total_raw_alerts": total_alerts,
        "correlated_incidents": grouped_incidents,
        "alert_compression_pct": round(compression, 2)
    }