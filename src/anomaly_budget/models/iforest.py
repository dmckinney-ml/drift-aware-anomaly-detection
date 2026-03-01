# src/anomaly_budget/models/iforest.py
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_iforest(X_train: np.ndarray, cfg: Dict[str, Any]) -> IsolationForest:
    """
    Fit IsolationForest on training features.
    """
    model = IsolationForest(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_samples=cfg.get("max_samples", "auto"),
        max_features=float(cfg.get("max_features", 1.0)),
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=int(cfg.get("n_jobs", -1)),
        contamination="auto",  # we control alert rate via thresholding
    )
    model.fit(X_train)
    return model


def score_iforest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Return anomaly scores where HIGHER => more anomalous.
    sklearn's score_samples: higher => more normal, so we negate it.
    """
    return -model.score_samples(X)
