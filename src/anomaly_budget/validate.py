from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# ----------------------------
# Friendly, user-facing errors
# ----------------------------
class UserInputError(ValueError):
    """Raise this for errors that should be shown directly to HF Space users."""


@dataclass(frozen=True)
class InputContract:
    timestamp_col: str = "timestamp"
    value_col: str = "value"
    label_col: Optional[str] = "is_anomaly"  # optional in public mode

    # Limits
    max_rows: int = 200_000
    max_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Missing value policy
    max_missing_value_frac: float = 0.01  # 1%
    interpolate_missing: bool = True

    # Cadence checks (warnings, not hard-fail unless extreme)
    irregular_cadence_warn_cv: float = 0.50  # coefficient of variation
    irregular_cadence_fail_cv: float = 2.00  # extreme irregularity


@dataclass(frozen=True)
class InputWarnings:
    warnings: List[str]

    def to_text(self) -> str:
        return "\n".join(f"- {w}" for w in self.warnings)


def _coerce_label(series: pd.Series) -> pd.Series:
    # Accept bool/int/float/str; coerce to 0/1
    if series.dtype == bool:
        out = series.astype(int)
    else:
        out = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return (out > 0).astype(int)


def validate_and_prepare_timeseries(
    df: pd.DataFrame,
    *,
    contract: InputContract = InputContract(),
    file_bytes: Optional[int] = None,
) -> Tuple[pd.DataFrame, InputWarnings]:
    """
    Validate + normalize a user-provided time series dataframe.

    Returns:
      (clean_df, warnings)
    clean_df is sorted by timestamp and contains at least:
      - timestamp_col (datetime, consistent tz-awareness)
      - value_col (float)
      - label_col if present (0/1 int)
    """

    # ---- size checks ----
    if file_bytes is not None and file_bytes > contract.max_bytes:
        raise UserInputError(
            f"File too large ({file_bytes/1024/1024:.1f} MB). "
            f"Max allowed is {contract.max_bytes/1024/1024:.0f} MB."
        )

    if len(df) > contract.max_rows:
        raise UserInputError(
            f"Too many rows ({len(df):,}). Max allowed is {contract.max_rows:,} rows."
        )

    ts_col = contract.timestamp_col
    val_col = contract.value_col
    lab_col = contract.label_col

    # ---- column checks ----
    missing = [c for c in [ts_col, val_col] if c not in df.columns]
    if missing:
        raise UserInputError(
            f"Missing required column(s): {missing}. "
            f"Your CSV must contain '{ts_col}' and '{val_col}'."
        )

    out = df.copy()

    # ---- timestamps ----
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")

    if out[ts_col].isna().any():
        bad = out.loc[out[ts_col].isna(), [ts_col]].head(5)
        raise UserInputError(
            f"Some timestamps could not be parsed. "
            f"Please use ISO-8601 if possible (e.g., 2015-01-01T01:00:00Z).\n"
            f"Example bad rows:\n{bad.to_string(index=False)}"
        )

    # Reject mixed tz-aware and tz-naive (pandas can produce object dtype in weird mixes)
    # We enforce: either all tz-aware or all tz-naive
    # If tz-aware, pandas stores dtype datetime64[ns, tz]
    if pd.api.types.is_object_dtype(out[ts_col].dtype):
        # attempt to detect mixed tz by inspecting a small sample
        sample = out[ts_col].iloc[:50]
        tzinfo = [getattr(getattr(x, "tzinfo", None), "utcoffset", None) for x in sample]
        if any(tzinfo) and not all(tzinfo):
            raise UserInputError(
                "Timestamps appear to mix timezone-aware and timezone-naive values. "
                "Please make them consistent (all with timezone, or all without)."
            )

    # ---- values ----
    out[val_col] = pd.to_numeric(out[val_col], errors="coerce")

    if out[val_col].isna().any():
        frac = float(out[val_col].isna().mean())
        if frac > contract.max_missing_value_frac:
            bad = out.loc[out[val_col].isna(), [ts_col, val_col]].head(5)
            raise UserInputError(
                f"'{val_col}' has too many missing/non-numeric values "
                f"({frac*100:.2f}% > {contract.max_missing_value_frac*100:.2f}%).\n"
                f"Example bad rows:\n{bad.to_string(index=False)}"
            )

    # ---- optional labels ----
    if lab_col and lab_col in out.columns:
        out[lab_col] = _coerce_label(out[lab_col])

    # ---- sort and de-dup timestamps ----
    out = out.sort_values(ts_col).reset_index(drop=True)

    if out[ts_col].duplicated().any():
        # Keep last by default (common in telemetry)
        out = out.drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)

    # ---- handle missing values ----
    warnings: List[str] = []
    if out[val_col].isna().any():
        if contract.interpolate_missing:
            # time-based interpolation if possible
            out = out.set_index(ts_col)
            out[val_col] = out[val_col].interpolate(method="time", limit_direction="both")
            out = out.reset_index()
            if out[val_col].isna().any():
                # fallback
                out[val_col] = out[val_col].interpolate(limit_direction="both")
            warnings.append("Filled a small number of missing values via interpolation.")
        else:
            out = out.dropna(subset=[val_col]).reset_index(drop=True)
            warnings.append("Dropped rows with missing values.")

    # ---- cadence warnings ----
    # Irregular cadence isn't fatal for all methods, but we should warn.
    deltas = out[ts_col].diff().dropna()
    if len(deltas) >= 10:
        secs = deltas.dt.total_seconds().to_numpy()
        med = float(np.median(secs))
        if med > 0:
            cv = float(np.std(secs) / (np.mean(secs) + 1e-12))
            if cv > contract.irregular_cadence_fail_cv:
                raise UserInputError(
                    "Sampling cadence is extremely irregular (time gaps vary widely). "
                    "Please resample to a regular interval before uploading."
                )
            if cv > contract.irregular_cadence_warn_cv:
                warnings.append(
                    "Sampling cadence looks irregular. Rolling-window features may be less reliable."
                )

    return out, InputWarnings(warnings=warnings)