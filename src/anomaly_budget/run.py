# src/anomaly_budget/run.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from anomaly_budget.config import load_and_validate_config
from anomaly_budget.data import (
    load_timeseries_csv,
    apply_labels_from_config,
    time_split,
    load_nab_combined_labels,
)
from anomaly_budget.features import build_features_from_config
from anomaly_budget.models.baseline_mad import score_frame_mad
from anomaly_budget.models.iforest import fit_iforest, score_iforest
from anomaly_budget.thresholds import (
    threshold_for_alert_rate,
    threshold_from_percentile,
    apply_threshold,
    evaluate_alert_rate,
)
from anomaly_budget.evaluation import compute_point_metrics, compute_event_window_metrics
from anomaly_budget.artifacts import (
    save_config,
    save_metrics,
    save_scored_frame,
    ensure_output_dir,
)


def run_from_config_path(config_path: str | Path) -> int:
    cfg = load_and_validate_config(config_path)
    return run_experiment(cfg)


def _dropna_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=feature_cols)


def _resolve_threshold(cfg: Dict[str, Any], train_scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (threshold, threshold_meta).
    Supports:
      - alert_rate: choose threshold from percentile (1-alert_rate)
      - fixed: use cfg threshold.threshold
      - percentile: use cfg threshold.percentile
    """
    thr_cfg = cfg.get("threshold", {}) or {}
    method = str(thr_cfg.get("method", "alert_rate"))

    if method == "fixed":
        if "threshold" not in thr_cfg:
            raise ValueError("threshold.threshold is required when threshold.method='fixed'")
        thr = float(thr_cfg["threshold"])
        meta = {"method": "fixed", "threshold": thr}
        return thr, meta

    if method == "percentile":
        if "percentile" not in thr_cfg:
            raise ValueError("threshold.percentile is required when threshold.method='percentile'")
        pct = float(thr_cfg["percentile"])
        thr = threshold_from_percentile(train_scores, pct)
        meta = {"method": "percentile", "percentile": pct, "threshold": thr}
        return thr, meta

    if method == "alert_rate":
        alert_rate = float(thr_cfg.get("alert_rate", 0.05))
        thr = threshold_for_alert_rate(train_scores, alert_rate=alert_rate)
        meta = {"method": "alert_rate", "alert_rate": alert_rate, "threshold": thr}
        return thr, meta

    raise ValueError(f"Unknown threshold.method='{method}'. Expected fixed|percentile|alert_rate")


def run_experiment(cfg: Dict[str, Any]) -> int:
    ds = cfg["dataset"]

    # ---- load + labels ----
    df = load_timeseries_csv(
        ds["path"],
        timestamp_col=ds["timestamp_col"],
        value_col=ds["value_col"],
        label_col=ds.get("label_col"),
    )
    df = apply_labels_from_config(df, cfg)

    train_df, test_df = time_split(df, cfg["split"]["train_frac"])

    model_type = str(cfg.get("model", {}).get("type", "baseline_mad"))

    # ---- score train/test ----
    if model_type == "baseline_mad":
        # Build features for baseline too (so seasonal_diff can exist)
        train_feat, feature_cols = build_features_from_config(train_df, cfg)
        test_feat, _ = build_features_from_config(test_df, cfg)

        # baseline_mad expects a single value-like column to score
        mad_cfg = cfg.get("baseline_mad", {})
        mad_window = int(mad_cfg.get("window", 24))

        # If config explicitly sets value_col, use it; otherwise use the first feature col
        mad_value_col = str(mad_cfg.get("value_col") or feature_cols[0])

        if mad_value_col not in train_feat.columns:
            raise ValueError(
                f"baseline_mad.value_col='{mad_value_col}' not found. "
                f"Available columns include: {list(train_feat.columns)[:20]} ..."
            )

        scored_train = score_frame_mad(
            train_feat,
            value_col=mad_value_col,
            window=mad_window,
            out_score_col="score",
        )
        scored_test = score_frame_mad(
            test_feat,
            value_col=mad_value_col,
            window=mad_window,
            out_score_col="score",
        )

        model_meta = {"type": "baseline_mad", "window": mad_window, "value_col": mad_value_col}

    elif model_type == "iforest":
        train_feat, feature_cols = build_features_from_config(train_df, cfg)
        test_feat, _ = build_features_from_config(test_df, cfg)

        train_feat = _dropna_features(train_feat, feature_cols)
        test_feat = _dropna_features(test_feat, feature_cols)

        X_train = train_feat[feature_cols].to_numpy()
        X_test = test_feat[feature_cols].to_numpy()

        if_cfg = cfg.get("iforest", {}) or {}
        model = fit_iforest(X_train, if_cfg)

        train_scores = score_iforest(model, X_train)
        test_scores = score_iforest(model, X_test)

        scored_train = train_feat.copy()
        scored_train["score"] = train_scores

        scored_test = test_feat.copy()
        scored_test["score"] = test_scores

        model_meta = {"type": "iforest", "feature_cols": feature_cols, "params": dict(if_cfg)}

    else:
        raise ValueError(f"Unknown model.type='{model_type}'. Expected baseline_mad|iforest.")

    # ---- thresholding ----
    train_scores_np = scored_train["score"].dropna().to_numpy(dtype=float)
    thr, thr_meta = _resolve_threshold(cfg, train_scores_np)

    scored_test = apply_threshold(scored_test, threshold=thr, out_pred_col="pred")

    # ---- evaluation ----
    # Avoid AP crash: drop rows with NaN score before metrics
    eval_df = scored_test.dropna(subset=["score"]).copy()

    point = compute_point_metrics(
        eval_df,
        label_col="is_anomaly",
        pred_col="pred",
        score_col="score",
    )

    # optional: report actual alert rate on test
    test_alert = evaluate_alert_rate(eval_df["score"].to_numpy(dtype=float), thr)

    event = None
    labels_cfg = cfg.get("labels")
    if labels_cfg and labels_cfg.get("type") == "nab_combined":
        window_minutes = int(labels_cfg.get("window_minutes", 60))
        windows_df = load_nab_combined_labels(labels_cfg["path"], window_minutes=window_minutes)

        eval_cfg = cfg.get("evaluation", {}) or {}
        event = compute_event_window_metrics(
            eval_df,
            windows_df,
            series_key=labels_cfg["series_key"],
            ts_col=ds["timestamp_col"],
            pred_col="pred",
            match_mode=str(labels_cfg.get("match_mode", "exact")),
            window_tolerance=str(eval_cfg.get("window_tolerance", "0min")),
            pred_merge_gap=str(eval_cfg.get("pred_merge_gap", "60min")),
            pred_min_width=str(eval_cfg.get("pred_min_width", "0min")),
        )

    # ---- artifacts ----
    out_dir = cfg.get("output", {}).get("dir", "outputs/latest")
    out_dir = ensure_output_dir(out_dir)

    save_config(out_dir, cfg)
    save_scored_frame(out_dir, scored_test, filename="scored_test.csv")  # keep full frame (incl NaNs)

    metrics: Dict[str, Any] = {
        "point": point,
        "threshold": thr_meta,
        "test_alert_rate_actual": {
            "threshold": float(test_alert.threshold),
            "alert_rate": float(test_alert.alert_rate),
            "alerts": int(test_alert.alerts),
            "n": int(test_alert.n),
        },
        "model": model_meta,
    }
    if event is not None:
        metrics["event"] = event
        metrics["labels"] = {
            "type": "nab_combined",
            "series_key": labels_cfg["series_key"],
            "window_minutes": window_minutes,
            "match_mode": labels_cfg.get("match_mode", "exact"),
        }

    save_metrics(out_dir, metrics)

    print(f"[ok] output_dir={out_dir}")
    print(f"[metrics] point={point}")
    print(f"[metrics] threshold={thr_meta}")
    print(f"[metrics] test_alert_rate={test_alert}")
    if event is not None:
        print(f"[metrics] event={event}")

    return 0