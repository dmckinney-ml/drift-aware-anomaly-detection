# src/anomaly_budget/thresholds.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdResult:
    method: str
    threshold: float
    alert_rate: float
    alerts: int
    n: int


def apply_threshold(
    scored_df: pd.DataFrame,
    *,
    score_col: str = "score",
    threshold: float,
    out_pred_col: str = "pred",
) -> pd.DataFrame:
    out = scored_df.copy()
    s = out[score_col].to_numpy(dtype=float)
    pred = np.zeros(len(out), dtype=int)
    finite = np.isfinite(s)
    pred[finite] = (s[finite] > float(threshold)).astype(int)
    out[out_pred_col] = pred
    return out


def threshold_from_percentile(
    train_scores: np.ndarray,
    percentile: float,
) -> float:
    s = np.asarray(train_scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        raise ValueError("train_scores is empty after removing NaN/inf")
    return float(np.percentile(s, percentile))


def threshold_for_alert_rate(
    train_scores: np.ndarray,
    alert_rate: float,
) -> float:
    """
    Pick threshold so that approx alert_rate fraction of train points alert.
    threshold = percentile((1 - alert_rate) * 100)
    """
    if not (0.0 < alert_rate < 1.0):
        raise ValueError("alert_rate must be in (0, 1)")
    percentile = (1.0 - alert_rate) * 100.0
    return threshold_from_percentile(train_scores, percentile)


def evaluate_alert_rate(
    scores: np.ndarray,
    threshold: float,
) -> ThresholdResult:
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    s = s[finite]
    pred = (s > float(threshold)).astype(int)
    alerts = int(pred.sum())
    n = int(len(pred))
    rate = float(alerts / n) if n else 0.0
    return ThresholdResult(method="fixed", threshold=float(threshold), alert_rate=rate, alerts=alerts, n=n)