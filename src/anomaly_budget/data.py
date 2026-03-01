# src/anomaly_budget/data.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ast
import json

import pandas as pd


class DataError(ValueError):
    """Raised when dataset cannot be loaded or parsed correctly."""


# ---------------------------------------------------------------------
# Core dataset loading
# ---------------------------------------------------------------------
def load_timeseries_csv(
    path: str | Path,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a CSV time series file with at least timestamp + value.

    Enforces:
      - timestamp parsed to datetime
      - sorted by timestamp
      - value numeric
      - label coerced to 0/1 int if provided
    """
    p = Path(path)
    if not p.exists():
        raise DataError(f"CSV not found: {p}")

    df = pd.read_csv(p)

    for col in [timestamp_col, value_col]:
        if col not in df.columns:
            raise DataError(f"Missing required column '{col}' in {p}. Found: {list(df.columns)}")

    # parse timestamps
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    if df[timestamp_col].isna().any():
        bad = df[df[timestamp_col].isna()].head(5)
        raise DataError(
            f"Found unparsable timestamps in '{timestamp_col}' for {p}. Example rows:\n{bad}"
        )

    # numeric values
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if df[value_col].isna().any():
        bad = df[df[value_col].isna()].head(5)
        raise DataError(
            f"Found non-numeric values in '{value_col}' for {p}. Example rows:\n{bad}"
        )

    # optional labels (if present in file)
    if label_col:
        if label_col not in df.columns:
            raise DataError(f"label_col='{label_col}' not found in {p}. Found: {list(df.columns)}")

        lab = df[label_col]
        if lab.dtype == bool:
            df[label_col] = lab.astype(int)
        else:
            df[label_col] = pd.to_numeric(lab, errors="coerce").fillna(0).astype(int)
        df[label_col] = (df[label_col] > 0).astype(int)

    # sort
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split preserving order."""
    if not (0.0 < train_frac < 1.0):
        raise DataError("train_frac must be between 0 and 1.")
    split_idx = int(len(df) * train_frac)
    if split_idx <= 0 or split_idx >= len(df):
        raise DataError("train_frac produces empty train or test split.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ---------------------------------------------------------------------
# NAB combined_labels.* loading (JSON points -> windows)
# ---------------------------------------------------------------------
def load_nab_combined_points(labels_path: str | Path) -> pd.DataFrame:
    """
    Load NAB combined labels (JSON preferred; CSV fallback) and normalize to POINT timestamps:
      series_key, timestamp

    Supports NAB JSON dict:
      { "realKnownCause/nyc_taxi.csv": ["2014-...", ...], ... }

    Also supports JSON list-of-records variants and legacy CSV variants.
    """
    p = Path(labels_path)
    if not p.exists():
        raise DataError(f"Labels file not found: {p}")

    suffix = p.suffix.lower()
    rows: list[dict[str, Any]] = []

    def _add_points(series_key: str, items: Any) -> None:
        if items is None:
            return

        if isinstance(items, (list, tuple)):
            for it in items:
                # tolerate ["start","end"] style entries; keep the start as the point label
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    ts = pd.to_datetime(it[0], errors="coerce")
                else:
                    ts = pd.to_datetime(it, errors="coerce")
                if pd.isna(ts):
                    continue
                rows.append({"series_key": str(series_key), "timestamp": ts})
        else:
            ts = pd.to_datetime(items, errors="coerce")
            if pd.isna(ts):
                return
            rows.append({"series_key": str(series_key), "timestamp": ts})

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            for series_key, items in obj.items():
                _add_points(series_key, items)

        elif isinstance(obj, list):
            possible_series_cols = ["filename", "file", "series", "path"]
            possible_window_cols = ["anomaly_window", "anomaly_windows", "windows", "anomalies"]

            for rec in obj:
                if not isinstance(rec, dict):
                    continue
                series_col = next((c for c in possible_series_cols if c in rec), None)
                window_col = next((c for c in possible_window_cols if c in rec), None)
                if not series_col or not window_col:
                    continue
                _add_points(str(rec[series_col]), rec[window_col])
        else:
            raise DataError(f"Unsupported JSON structure in {p}: {type(obj)}")

    else:
        # CSV fallback
        lab = pd.read_csv(p)

        possible_series_cols = ["filename", "file", "series", "path"]
        series_col = next((c for c in possible_series_cols if c in lab.columns), None)
        if series_col is None:
            raise DataError(
                f"Could not find series identifier column in {p}. "
                f"Tried: {possible_series_cols}. Found: {list(lab.columns)}"
            )

        possible_window_cols = ["anomaly_window", "anomaly_windows", "windows", "anomalies"]
        window_col = next((c for c in possible_window_cols if c in lab.columns), None)
        if window_col is None:
            raise DataError(
                f"Could not find anomaly window column in {p}. "
                f"Tried: {possible_window_cols}. Found: {list(lab.columns)}"
            )

        for _, r in lab.iterrows():
            series_key = str(r[series_col])
            cell = r[window_col]
            try:
                parsed = (
                    json.loads(cell)
                    if isinstance(cell, str) and cell.strip().startswith("[")
                    else ast.literal_eval(cell)
                )
            except Exception as e:
                raise DataError(f"Failed parsing windows for {series_key}: {e}")

            _add_points(series_key, parsed)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["series_key", "timestamp"])
    return out.sort_values(["series_key", "timestamp"]).reset_index(drop=True)


def points_to_windows(points_df: pd.DataFrame, window_minutes: int = 60) -> pd.DataFrame:
    """
    Convert point labels into [start,end] windows of fixed duration.
    Output columns: series_key, start, end
    """
    if points_df.empty:
        return pd.DataFrame(columns=["series_key", "start", "end"])
    if window_minutes <= 0:
        raise DataError("window_minutes must be positive")

    delta = pd.Timedelta(minutes=int(window_minutes))
    out = points_df.copy()
    out = out.rename(columns={"timestamp": "start"})
    out["end"] = out["start"] + delta
    return out[["series_key", "start", "end"]].sort_values(["series_key", "start"]).reset_index(drop=True)


def load_nab_combined_labels(labels_path: str | Path, window_minutes: int = 60) -> pd.DataFrame:
    """
    Compatibility wrapper for the rest of your pipeline:
    load points -> convert to windows
    """
    pts = load_nab_combined_points(labels_path)
    return points_to_windows(pts, window_minutes=window_minutes)


def add_point_labels_from_windows(
    df: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    series_key: str,
    ts_col: str = "timestamp",
    out_label_col: str = "is_anomaly",
    match_mode: str = "exact",  # exact | suffix
) -> pd.DataFrame:
    out = df.copy()
    out[out_label_col] = 0

    if windows_df.empty:
        return out

    if match_mode not in {"exact", "suffix"}:
        raise DataError("match_mode must be 'exact' or 'suffix'")

    if match_mode == "exact":
        w = windows_df[windows_df["series_key"] == series_key]
    else:
        w = windows_df[windows_df["series_key"].astype(str).str.endswith(series_key)]

    if w.empty:
        return out

    ts = out[ts_col]
    mask = pd.Series(False, index=out.index)
    for _, row in w.iterrows():
        mask |= (ts >= row["start"]) & (ts <= row["end"])

    out.loc[mask, out_label_col] = 1
    return out


# ---------------------------------------------------------------------
# Config-driven labeling
# ---------------------------------------------------------------------
def apply_labels_from_config(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply labels if cfg includes a supported labels block. Otherwise ensure is_anomaly exists.

    Supports:
      labels:
        type: nab_combined
        path: NAB/labels/combined_labels.json
        series_key: realKnownCause/nyc_taxi.csv
        match_mode: suffix|exact
        window_minutes: 60
    """
    ds = cfg["dataset"]
    labels_cfg = cfg.get("labels")

    # If dataset already contains labels (label_col), normalize to is_anomaly for consistency
    label_col = ds.get("label_col")
    if label_col and label_col in df.columns:
        out = df.copy()
        out["is_anomaly"] = out[label_col].astype(int)
        return out

    if labels_cfg and labels_cfg.get("type") == "nab_combined":
        window_minutes = int(labels_cfg.get("window_minutes", 60))
        windows_df = load_nab_combined_labels(labels_cfg["path"], window_minutes=window_minutes)

        out = add_point_labels_from_windows(
            df,
            windows_df,
            series_key=str(labels_cfg["series_key"]),
            ts_col=str(ds["timestamp_col"]),
            out_label_col="is_anomaly",
            match_mode=str(labels_cfg.get("match_mode", "exact")),
        )
        return out

    # Default: consistent schema
    out = df.copy()
    if "is_anomaly" not in out.columns:
        out["is_anomaly"] = 0
    return out