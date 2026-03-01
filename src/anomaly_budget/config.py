from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    """Raised when configuration is invalid or missing required keys."""


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key '{ctx}.{key}'")
    return d[key]


def _as_path(p: Any, base_dir: Path) -> Path:
    if p is None:
        raise ConfigError("Path is None")
    path = Path(str(p)).expanduser()
    return path if path.is_absolute() else (base_dir / path)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML config from disk. Returns raw dict (not yet validated).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a YAML mapping (top-level dict).")

    cfg["_meta"] = {
        "config_path": str(config_path.resolve()),
        "config_dir": str(config_path.parent.resolve()),
    }
    return cfg


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize config. Returns a resolved dict with defaults applied.
    Path policy:
      - dataset/labels paths resolve relative to config_dir (usually ./configs)
      - output.dir resolves relative to project root (config_dir/..)
    """
    meta = cfg.get("_meta") or {}
    config_dir = Path(meta.get("config_dir") or ".").resolve()
    project_root = config_dir.parent

    dataset = _require(cfg, "dataset", "root")
    if not isinstance(dataset, dict):
        raise ConfigError("root.dataset must be a mapping")

    split = cfg.get("split", {}) or {}
    if not isinstance(split, dict):
        raise ConfigError("root.split must be a mapping if provided")

    features = cfg.get("features", {}) or {}
    if not isinstance(features, dict):
        raise ConfigError("root.features must be a mapping if provided")

    labels = cfg.get("labels", None)
    if labels is not None and not isinstance(labels, dict):
        raise ConfigError("root.labels must be a mapping if provided")

    threshold = cfg.get("threshold", None)
    if threshold is not None and not isinstance(threshold, dict):
        raise ConfigError("root.threshold must be a mapping if provided")

    output = cfg.get("output", None)
    if output is not None and not isinstance(output, dict):
        raise ConfigError("root.output must be a mapping if provided")

    baseline_mad = cfg.get("baseline_mad", None)
    if baseline_mad is not None and not isinstance(baseline_mad, dict):
        raise ConfigError("root.baseline_mad must be a mapping if provided")

    model = cfg.get("model", None)
    if model is not None and not isinstance(model, dict):
        raise ConfigError("root.model must be a mapping if provided")

    iforest = cfg.get("iforest", None)
    if iforest is not None and not isinstance(iforest, dict):
        raise ConfigError("root.iforest must be a mapping if provided")

    # --------------------
    # dataset
    # --------------------
    ds_name = str(_require(dataset, "name", "dataset"))
    ds_path_raw = _require(dataset, "path", "dataset")
    ds_path = _as_path(ds_path_raw, config_dir).resolve()

    timestamp_col = str(dataset.get("timestamp_col", "timestamp"))
    value_col = str(dataset.get("value_col", "value"))
    label_col = dataset.get("label_col", None)
    label_col = str(label_col) if label_col not in (None, "", "null") else None

    # --------------------
    # split
    # --------------------
    train_frac = float(split.get("train_frac", 0.7))
    if not (0.5 <= train_frac < 1.0):
        raise ConfigError("split.train_frac must be in [0.5, 1.0).")

    # --------------------
    # features
    # --------------------
    kind = str(features.get("kind", "raw"))  # raw | seasonal_diff | rolling_context
    if kind not in {"raw", "seasonal_diff", "rolling_context"}:
        raise ConfigError("features.kind must be one of: raw, seasonal_diff, rolling_context")

    resolved: Dict[str, Any] = {
        "dataset": {
            "name": ds_name,
            "path": str(ds_path),
            "timestamp_col": timestamp_col,
            "value_col": value_col,
            "label_col": label_col,
        },
        "split": {"train_frac": train_frac},
        "features": {"kind": kind},
        "_meta": meta,
    }

    # feature options
    if kind == "seasonal_diff":
        period = int(features.get("period", 48))
        if period <= 0:
            raise ConfigError("features.period must be a positive integer.")
        resolved["features"]["period"] = period

    if kind == "rolling_context":
        roll_window = int(features.get("roll_window", features.get("window", 24)))
        if roll_window <= 1:
            raise ConfigError("features.roll_window must be > 1.")
        resolved["features"]["roll_window"] = roll_window

        long_window = int(features.get("long_window", 168))
        if long_window <= 1:
            raise ConfigError("features.long_window must be > 1.")
        resolved["features"]["long_window"] = long_window

        include_raw = bool(features.get("include_raw", True))
        resolved["features"]["include_raw"] = include_raw

    # --------------------
    # labels (optional)
    # --------------------
    if labels:
        labels_type = str(_require(labels, "type", "labels"))
        if labels_type not in {"nab_combined"}:
            raise ConfigError("labels.type must be 'nab_combined' (for now)")

        labels_path_raw = _require(labels, "path", "labels")
        labels_path = _as_path(labels_path_raw, config_dir).resolve()

        series_key = str(_require(labels, "series_key", "labels"))

        match_mode = str(labels.get("match_mode", "exact"))
        if match_mode not in {"exact", "suffix"}:
            raise ConfigError("labels.match_mode must be 'exact' or 'suffix'")

        window_minutes = int(labels.get("window_minutes", 60))
        if window_minutes <= 0:
            raise ConfigError("labels.window_minutes must be > 0")

        resolved["labels"] = {
            "type": labels_type,
            "path": str(labels_path),
            "series_key": series_key,
            "match_mode": match_mode,
            "window_minutes": window_minutes,
        }

    # --------------------
    # threshold (optional)
    # --------------------
    if threshold:
        method = str(threshold.get("method", "alert_rate"))

        if method not in {"alert_rate", "fixed", "percentile"}:
            raise ConfigError("threshold.method must be one of: alert_rate, fixed, percentile")

        out_thr: Dict[str, Any] = {"method": method}

        if method == "alert_rate":
            alert_rate = float(threshold.get("alert_rate", 0.05))
            if not (0.0 < alert_rate < 1.0):
                raise ConfigError("threshold.alert_rate must be between 0 and 1")
            out_thr["alert_rate"] = alert_rate

        elif method == "fixed":
            if "threshold" not in threshold:
                raise ConfigError("threshold.threshold is required when method='fixed'")
            thr = float(threshold["threshold"])
            out_thr["threshold"] = thr

        elif method == "percentile":
            if "percentile" not in threshold:
                raise ConfigError("threshold.percentile is required when method='percentile'")
            pct = float(threshold["percentile"])
            if not (0.0 < pct < 100.0):
                raise ConfigError("threshold.percentile must be in (0, 100)")
            out_thr["percentile"] = pct

        resolved["threshold"] = out_thr

    # --------------------
    # output (optional)
    # --------------------
    if output:
        out_dir_raw = output.get("dir", "outputs/latest")
        out_dir_path = Path(str(out_dir_raw)).expanduser()

        # IMPORTANT: output resolves relative to PROJECT ROOT (configs/..)
        out_dir = out_dir_path if out_dir_path.is_absolute() else (project_root / out_dir_path)
        out_dir = out_dir.resolve()

        out_cfg = dict(output)
        out_cfg["dir"] = str(out_dir)
        out_cfg.setdefault("save_scored_frame", True)
        out_cfg.setdefault("save_metrics", True)
        resolved["output"] = out_cfg

    # --------------------
    # model / baseline_mad / iforest (pass-through + light validation)
    # --------------------
    if model:
        resolved["model"] = dict(model)

    if baseline_mad:
        bm = dict(baseline_mad)
        if "window" in bm:
            w = int(bm["window"])
            if w <= 1:
                raise ConfigError("baseline_mad.window must be > 1")
            bm["window"] = w
        resolved["baseline_mad"] = bm

    if iforest:
        resolved["iforest"] = dict(iforest)

    return resolved


def load_and_validate_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Convenience function: load YAML and validate/normalize.
    """
    raw = load_yaml_config(config_path)
    return validate_config(raw)