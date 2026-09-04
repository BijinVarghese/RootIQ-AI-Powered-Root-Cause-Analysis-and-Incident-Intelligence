"""
RootIQ - Isolation Forest Anomaly Detector
Unsupervised machine-learning model for multi-metric telemetry anomaly detection.
Generates continuous anomaly scores [0, 1] and binary anomaly flags.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from typing import List, Optional, Tuple, Dict, Any

class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 150, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "IsolationForestDetector":
        """Fit scaler and Isolation Forest on training data."""
        if feature_cols is None:
            self.feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.feature_names = [c for c in feature_cols if c in df.columns]

        if not self.feature_names:
            raise ValueError("No numeric features provided to fit Isolation Forest.")

        X = df[self.feature_names].fillna(0.0).values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute anomaly flags and normalized anomaly score in range [0, 1].
        1.0 indicates maximum anomaly severity, 0.0 indicates completely normal.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        df = df.copy()
        X = df[self.feature_names].fillna(0.0).values
        X_scaled = self.scaler.transform(X)

        # Raw scores: negative indicates anomaly, higher is more normal
        raw_scores = self.model.decision_function(X_scaled)
        preds = self.model.predict(X_scaled) # -1 is anomaly, 1 is normal

        # Normalize score into [0, 1] where 1.0 is highest anomaly severity
        # decision_function typically ranges from [-0.5, 0.5]
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            norm_anomaly_score = 1.0 - ((raw_scores - min_s) / (max_s - min_s))
        else:
            norm_anomaly_score = np.zeros_like(raw_scores)

        df["raw_decision_score"] = raw_scores.round(4)
        df["anomaly_score"] = norm_anomaly_score.round(4)
        df["is_anomaly"] = (preds == -1).astype(int)

        return df

    def save_model(self, filepath: str) -> None:
        """Serialize trained model and scaler."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "is_fitted": self.is_fitted
        }, filepath)

    def load_model(self, filepath: str) -> "IsolationForestDetector":
        """Load serialized model and scaler."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.contamination = data.get("contamination", 0.05)
        self.is_fitted = data["is_fitted"]
        return self