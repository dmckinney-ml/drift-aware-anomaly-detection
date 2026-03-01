# Baseline MAD anomaly detection model
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class RollingMADConfig:
    value_col: str = "value"
    window: int = 24
    min_periods: Optional[int] = None
    eps: float = 1e-9
    # 1.4826 scales MAD to be comparable to std under Normal assumption
    mad_scale: float = 1.4826


def rolling_mad_score(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    window: int = 24,
    min_periods: Optional[int] = None,
    eps: float = 1e-9,
    mad_scale: float = 1.4826,
) -> pd.Series:
    """
    Rolling robust z-score based on Median Absolute Deviation (MAD).

    score(t) = |x_t - median_window(t)| / (mad_scale * MAD_window(t) + eps)

    Returns a float Series aligned with df.index.
    Higher score => more anomalous.
    """
    if value_col not in df.columns:
        raise ValueError(f"value_col='{value_col}' not found in df")

    if window <= 1:
        raise ValueError("window must be > 1")

    if min_periods is None:
        min_periods = max(10, window // 3)

    x = pd.to_numeric(df[value_col], errors="coerce")
    roll = x.rolling(window=window, min_periods=min_periods)

    med = roll.median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window=window, min_periods=min_periods).median()

    score = abs_dev / (mad_scale * mad + eps)
    return score.astype(float)


def score_frame_mad(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    window: int = 24,
    min_periods: Optional[int] = None,
    eps: float = 1e-9,
    mad_scale: float = 1.4826,
    out_score_col: str = "score",
) -> pd.DataFrame:
    """
    Convenience: returns a copy of df with a score column.
    """
    out = df.copy()
    out[out_score_col] = rolling_mad_score(
        out,
        value_col=value_col,
        window=window,
        min_periods=min_periods,
        eps=eps,
        mad_scale=mad_scale,
    )
    return out