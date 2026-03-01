# Artifact management for experiment outputs
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # fallback
    return str(obj)


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_config(out_dir: str | Path, cfg: Dict[str, Any], filename: str = "config.resolved.json") -> Path:
    out = ensure_output_dir(out_dir)
    p = out / filename
    p.write_text(json.dumps(_jsonable(cfg), indent=2, sort_keys=True), encoding="utf-8")
    return p


def save_metrics(out_dir: str | Path, metrics: Dict[str, Any], filename: str = "metrics.json") -> Path:
    out = ensure_output_dir(out_dir)
    p = out / filename
    p.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True), encoding="utf-8")
    return p


def save_scored_frame(
    out_dir: str | Path,
    df: pd.DataFrame,
    filename: str = "scored.csv",
) -> Path:
    out = ensure_output_dir(out_dir)
    p = out / filename
    df.to_csv(p, index=False)
    return p