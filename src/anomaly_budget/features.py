from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd


class FeatureError(ValueError):
    """Raised when feature generation fails due to config or data issues."""


def build_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    value_col: str,
    kind: str = "raw",
    period: int = 48,
    roll_window: int = 24,
    long_window: int = 168,
    include_raw: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create features and return (feature_df, feature_cols).
    Does not drop NaNs—callers should dropna per split to avoid leakage.
    """
    if timestamp_col not in df.columns:
        raise FeatureError(f"Missing timestamp_col '{timestamp_col}'")
    if value_col not in df.columns:
        raise FeatureError(f"Missing value_col '{value_col}'")

    out = df.copy()

    # Ensure sorted by time
    out = out.sort_values(timestamp_col).reset_index(drop=True)

    if kind == "raw":
        feature_cols = [value_col]
        return out, feature_cols

    if kind == "seasonal_diff":
        if period <= 0:
            raise FeatureError("period must be positive for seasonal_diff")
        out["seasonal_diff"] = out[value_col] - out[value_col].shift(period)
        feature_cols = ["seasonal_diff"]
        return out, feature_cols

    if kind == "rolling_context":
        if roll_window <= 1:
            raise FeatureError("roll_window must be > 1")
        if long_window <= 1:
            raise FeatureError("long_window must be > 1")

        # base columns
        feature_cols: List[str] = []

        if include_raw:
            feature_cols.append(value_col)

        # deltas
        out["delta1"] = out[value_col].diff(1)
        out["deltaW"] = out[value_col].diff(roll_window)  # "window" delta
        feature_cols += ["delta1", "deltaW"]

        # rolling stats (short)
        out["rolling_mean"] = out[value_col].rolling(roll_window, min_periods=max(2, roll_window // 2)).mean()
        out["rolling_std"]  = out[value_col].rolling(roll_window, min_periods=max(2, roll_window // 2)).std(ddof=0)
        feature_cols += ["rolling_mean", "rolling_std"]

        # long-horizon deviation
        out["rolling_mean_long"] = out[value_col].rolling(long_window, min_periods=max(2, long_window // 2)).mean()
        out["long_dev"] = out[value_col] - out["rolling_mean_long"]
        feature_cols += ["rolling_mean_long", "long_dev"]

        # guard: infinite -> nan
        for c in feature_cols:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)

        return out, feature_cols

    raise FeatureError(f"Unknown kind='{kind}'. Expected raw|seasonal_diff|rolling_context.")


def build_features_from_config(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience wrapper around build_features using validated config.
    """
    ds = cfg["dataset"]
    feat = cfg["features"]

    kind = feat["kind"]
    kwargs = dict(
        timestamp_col=ds["timestamp_col"],
        value_col=ds["value_col"],
        kind=kind,
    )

    if kind == "seasonal_diff":
        kwargs["period"] = int(feat["period"])

    if kind == "rolling_context":
        kwargs["roll_window"] = int(feat["roll_window"])
        kwargs["long_window"] = int(feat["long_window"])
        kwargs["include_raw"] = bool(feat["include_raw"])

    return build_features(df, **kwargs)