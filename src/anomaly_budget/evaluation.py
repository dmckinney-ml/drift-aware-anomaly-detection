# src/anomaly_budget/evaluation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


@dataclass(frozen=True)
class PointMetrics:
    precision: float
    recall: float
    f1: float
    ap: float
    n_true: int
    n_pred: int
    n: int


def compute_point_metrics(
    df: pd.DataFrame,
    *,
    label_col: str = "is_anomaly",
    pred_col: str = "pred",
    score_col: Optional[str] = "score",
) -> PointMetrics:
    y_true = df[label_col].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy()
    n_true = int(y_true.sum())
    n_pred = int(y_pred.sum())
    n = int(len(df))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    ap = float("nan")
    if score_col and score_col in df.columns and n_true > 0:
        scores = df[score_col].to_numpy(dtype=float)
        # average_precision_score requires finite scores
        ap = float(average_precision_score(y_true, scores))

    return PointMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        ap=ap,
        n_true=n_true,
        n_pred=n_pred,
        n=n,
    )


@dataclass(frozen=True)
class EventWindowMetrics:
    event_recall: float
    event_precision: float
    matched_events: int
    total_events: int
    matched_predictions: int
    total_predictions: int


def _merge_timestamps_to_windows(
    ts: pd.Series,
    *,
    merge_gap: pd.Timedelta,
    min_width: pd.Timedelta,
) -> pd.DataFrame:
    """
    Convert a sorted timestamp Series into merged windows.
    A new window starts when (current_ts - prev_ts) > merge_gap.
    Returns DataFrame with columns: start, end
    """
    if ts.empty:
        return pd.DataFrame(columns=["start", "end"])

    # Ensure sorted + unique-ish
    ts = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
    if ts.empty:
        return pd.DataFrame(columns=["start", "end"])

    starts = [ts.iloc[0]]
    ends = [ts.iloc[0]]

    for t in ts.iloc[1:]:
        if (t - ends[-1]) <= merge_gap:
            ends[-1] = t
        else:
            starts.append(t)
            ends.append(t)

    out = pd.DataFrame({"start": starts, "end": ends})

    # Enforce minimum width by expanding end if needed
    width = out["end"] - out["start"]
    too_small = width < min_width
    if too_small.any():
        out.loc[too_small, "end"] = out.loc[too_small, "start"] + min_width

    return out


def compute_event_window_metrics(
    scored_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    series_key: str,
    ts_col: str = "timestamp",
    pred_col: str = "pred",
    match_mode: str = "exact",        # exact | suffix
    window_tolerance: str = "0min",   # expands labeled windows by +/- tolerance
    pred_merge_gap: str = "0min",     # merge predicted points into windows if <= gap
    pred_min_width: str = "0min",     # minimum duration of a predicted window
) -> EventWindowMetrics:
    """
    Event-window evaluation (alerting-style).

    Labeled windows:
      windows_df columns: series_key, start, end

    Predicted windows:
      - take all predicted points (pred_col==1)
      - merge points into windows using pred_merge_gap
      - optionally enforce minimum predicted window width (pred_min_width)

    Metrics:
      - event_recall: fraction of labeled windows hit by any predicted window
      - event_precision: fraction of predicted windows that overlap any labeled window
    """
    if match_mode not in {"exact", "suffix"}:
        raise ValueError("match_mode must be 'exact' or 'suffix'")

    if windows_df.empty:
        total_pred_pts = int(scored_df[pred_col].sum())
        return EventWindowMetrics(0.0, 0.0, 0, 0, 0, total_pred_pts)

    # Select series windows
    if match_mode == "exact":
        w = windows_df[windows_df["series_key"] == series_key].copy()
    else:
        w = windows_df[windows_df["series_key"].astype(str).str.endswith(series_key)].copy()

    total_events = int(len(w))
    if total_events == 0:
        total_pred_pts = int(scored_df[pred_col].sum())
        return EventWindowMetrics(0.0, 0.0, 0, 0, 0, total_pred_pts)

    # Normalize + apply tolerance to labeled windows
    tol = pd.Timedelta(window_tolerance)
    w["start"] = pd.to_datetime(w["start"], errors="coerce") - tol
    w["end"] = pd.to_datetime(w["end"], errors="coerce") + tol
    w = w.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)

    total_events = int(len(w))
    if total_events == 0:
        total_pred_pts = int(scored_df[pred_col].sum())
        return EventWindowMetrics(0.0, 0.0, 0, 0, 0, total_pred_pts)

    # Predicted points -> prediction windows
    pred_ts = pd.to_datetime(
        scored_df.loc[scored_df[pred_col] == 1, ts_col],
        errors="coerce",
    ).dropna().sort_values()

    total_pred_points = int(len(pred_ts))

    merge_gap = pd.Timedelta(pred_merge_gap)
    min_width = pd.Timedelta(pred_min_width)

    pred_windows = _merge_timestamps_to_windows(
        pred_ts,
        merge_gap=merge_gap,
        min_width=min_width,
    )

    total_predictions = int(len(pred_windows))
    if total_predictions == 0:
        return EventWindowMetrics(0.0, 0.0, 0, total_events, 0, 0)

    # Helpers: overlap check
    def _overlaps(a_start, a_end, b_start, b_end) -> bool:
        return (a_start <= b_end) and (b_start <= a_end)

    # Event recall: how many labeled windows are hit by ANY pred window
    matched_events = 0
    for _, ev in w.iterrows():
        hit = False
        for _, pw in pred_windows.iterrows():
            if _overlaps(ev["start"], ev["end"], pw["start"], pw["end"]):
                hit = True
                break
        if hit:
            matched_events += 1

    # Prediction precision-like: how many pred windows overlap ANY event window
    matched_predictions = 0
    for _, pw in pred_windows.iterrows():
        hit = False
        for _, ev in w.iterrows():
            if _overlaps(ev["start"], ev["end"], pw["start"], pw["end"]):
                hit = True
                break
        if hit:
            matched_predictions += 1

    event_recall = float(matched_events / total_events) if total_events else 0.0
    event_precision = float(matched_predictions / total_predictions) if total_predictions else 0.0

    return EventWindowMetrics(
        event_recall=event_recall,
        event_precision=event_precision,
        matched_events=matched_events,
        total_events=total_events,
        matched_predictions=matched_predictions,
        total_predictions=total_predictions,
    )