#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build next-generation event-level SCADA features for Farm C.

This script supplements the existing four-detector outputs. It does NOT replace
or overwrite the validated four-detector engine. Instead, it adds three kinds
of lower-level evidence that were missing from the first event-level model:

1. Dual baseline evidence
   - fixed pre-event baseline (default: 48 hours)
   - mode-aware rolling dynamic baseline (default: previous 24 hours)

2. Operating-mode stratification
   - generating
   - low_power
   - stopped
   - unknown

3. True post-event recovery/right-censoring evidence
   - reads each event's row_scores.csv
   - checks raw detector flags after metadata event_end
   - distinguishes observed recovery, continuing abnormality and missing data

The output all_events_event_level_features_v3.csv can be used directly by
train_event_level_ml_vote_v3.py.

Recommended project layout
--------------------------
wind_farm_fault_detection/
├─ data/raw/Wind Farm C/
│  ├─ event_info.csv
│  └─ datasets/<event_id>.csv
├─ outputs/farmC_four_detector_event_level/farm_C/
│  ├─ all_events_event_level_features.csv
│  └─ event_<id>/row_scores.csv
└─ scripts/
   └─ build_event_level_features_v3.py

Example
-------
python .\\wind_farm_fault_detection\\scripts\\build_event_level_features_v3.py `
  --farm C `
  --metadata ".\\wind_farm_fault_detection\\data\\raw\\Wind Farm C\\event_info.csv" `
  --event-dir ".\\wind_farm_fault_detection\\data\\raw\\Wind Farm C\\datasets" `
  --detector-output-dir ".\\wind_farm_fault_detection\\outputs\\farmC_four_detector_event_level\\farm_C" `
  --output-dir ".\\wind_farm_fault_detection\\outputs\\farmC_event_features_v3" `
  --fixed-reference-hours 48 `
  --dynamic-window-hours 24 `
  --post-event-hours 12 `
  --measurement-mode avg_only `
  --power-signals "power_2,power_5,power_6,power_17"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


EPSILON = 1e-9
TRUE_TEXT = {"true", "1", "yes", "y", "t"}
TIMESTAMP_CANDIDATES = [
    "time_stamp", "timestamp", "datetime", "date_time", "time", "date"
]
ROW_ID_CANDIDATES = ["id", "row_id", "scada_id", "record_id"]
STAT_SUFFIX_PATTERN = re.compile(r"_(avg|max|min|std)$", re.IGNORECASE)
NON_MEASUREMENT_COLUMNS = {
    "time_stamp", "timestamp", "datetime", "date_time", "time", "date",
    "asset_id", "id", "row_id", "scada_id", "record_id", "train_test",
    "status_type_id", "event_id", "event_label",
}


@dataclass(frozen=True)
class EventMetadata:
    farm_id: str
    event_id: str
    event_label: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    event_description: str


# =============================================================================
# 1. GENERAL UTILITIES
# =============================================================================

def normalise_event_id(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna(False).astype(str).str.strip().str.lower().isin(TRUE_TEXT)
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, ensure_ascii=False, indent=2)


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep=";", low_memory=False)
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";", engine="python", low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lower_map = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def infer_sampling_minutes(timestamps: pd.Series) -> float:
    diffs = (
        pd.to_datetime(timestamps, errors="coerce")
        .sort_values()
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .dropna()
    )
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        return 10.0
    return float(diffs.median())


def robust_location_scale(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    median = frame.median(axis=0, skipna=True)
    mad = (frame - median).abs().median(axis=0, skipna=True) * 1.4826
    q25 = frame.quantile(0.25)
    q75 = frame.quantile(0.75)
    iqr = (q75 - q25) / 1.349
    std = frame.std(axis=0, skipna=True)

    scale = mad.copy()
    bad = (~np.isfinite(scale)) | (scale <= EPSILON)
    scale.loc[bad] = iqr.loc[bad]
    bad = (~np.isfinite(scale)) | (scale <= EPSILON)
    scale.loc[bad] = std.loc[bad]
    bad = (~np.isfinite(scale)) | (scale <= EPSILON)
    scale.loc[bad] = np.nan
    return median, scale


def longest_true_run_hours(flag: pd.Series, sampling_minutes: float) -> float:
    values = flag.fillna(False).astype(bool).to_numpy()
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest * sampling_minutes / 60.0)


def summarise_fraction_series(
    series: pd.Series,
    prefix: str,
    row_threshold: float,
    sampling_minutes: float,
) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_row_coverage": 0.0,
            f"{prefix}_longest_hours": 0.0,
        }
    full = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p95": float(values.quantile(0.95)),
        f"{prefix}_row_coverage": float((values >= row_threshold).mean()),
        f"{prefix}_longest_hours": longest_true_run_hours(
            full >= row_threshold, sampling_minutes
        ),
    }


# =============================================================================
# 2. INPUT LOADING
# =============================================================================

def load_metadata(path: Path, farm_id: str) -> list[EventMetadata]:
    df = read_semicolon_csv(path)
    required = [
        "event_id", "event_label", "event_start", "event_end", "event_description"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["event_start"] = pd.to_datetime(df["event_start"], errors="coerce")
    df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")
    df = df.dropna(subset=["event_start", "event_end"])
    df = df.loc[df["event_end"] > df["event_start"]].copy()

    events: list[EventMetadata] = []
    for _, row in df.iterrows():
        events.append(
            EventMetadata(
                farm_id=str(farm_id),
                event_id=normalise_event_id(row["event_id"]),
                event_label=str(row.get("event_label", "")),
                event_start=pd.Timestamp(row["event_start"]),
                event_end=pd.Timestamp(row["event_end"]),
                event_description=str(row.get("event_description", "")),
            )
        )
    return events


def resolve_event_file(event_dir: Path, event_id: str) -> Path:
    candidates = [
        event_dir / f"{event_id}.csv",
        event_dir / f"event_{event_id}.csv",
        event_dir / f"Event_{event_id}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(event_dir.glob(f"*{event_id}*.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple candidate files for event {event_id}: {matches}"
        )
    raise FileNotFoundError(f"No event CSV found for event {event_id} in {event_dir}")


def resolve_base_features_file(
    detector_output_dir: Path,
    explicit: Optional[Path],
) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Base feature table not found: {explicit}")
        return explicit

    exact = detector_output_dir / "all_events_event_level_features.csv"
    if exact.exists():
        return exact

    matches = sorted(detector_output_dir.glob("all_events_event_level_features*.csv"))
    if not matches:
        raise FileNotFoundError(
            "Could not find all_events_event_level_features*.csv under "
            f"{detector_output_dir}"
        )
    return matches[-1]


def select_measurement_columns(
    df: pd.DataFrame,
    measurement_mode: str,
    excluded_base_signals: set[str],
    minimum_valid_fraction: float,
) -> list[str]:
    selected: list[str] = []
    for column in df.columns:
        lower = column.lower()
        if lower in NON_MEASUREMENT_COLUMNS:
            continue
        match = STAT_SUFFIX_PATTERN.search(column)
        if match is None:
            continue
        statistic = match.group(1).lower()
        if measurement_mode == "avg_only" and statistic != "avg":
            continue
        base_signal = STAT_SUFFIX_PATTERN.sub("", column)
        if base_signal in excluded_base_signals:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if float(numeric.notna().mean()) < minimum_valid_fraction:
            continue
        selected.append(column)
    return selected


# =============================================================================
# 3. OPERATING-MODE CLASSIFICATION
# =============================================================================

def parse_power_signals(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def identify_power_average_columns(
    df: pd.DataFrame,
    power_signals: list[str],
) -> list[str]:
    columns: list[str] = []
    for base in power_signals:
        candidate = f"{base}_avg"
        if candidate in df.columns:
            columns.append(candidate)
    return columns


def classify_operating_mode(
    df: pd.DataFrame,
    power_columns: list[str],
    reference_mask: pd.Series,
    stopped_ratio: float,
    generating_ratio: float,
) -> tuple[pd.Series, pd.Series, float]:
    if not power_columns:
        mode = pd.Series("unknown", index=df.index, dtype=object)
        return mode, pd.Series(np.nan, index=df.index), np.nan

    power = df[power_columns].apply(pd.to_numeric, errors="coerce")
    consensus = power.median(axis=1, skipna=True)
    reference = consensus.loc[reference_mask].dropna()
    if reference.empty:
        active_scale = float(consensus.dropna().quantile(0.90))
    else:
        active_scale = float(reference.quantile(0.90))

    if not np.isfinite(active_scale) or abs(active_scale) <= EPSILON:
        mode = pd.Series("unknown", index=df.index, dtype=object)
        return mode, pd.Series(np.nan, index=df.index), active_scale

    normalised = consensus / max(abs(active_scale), EPSILON)
    mode = pd.Series("unknown", index=df.index, dtype=object)
    valid = consensus.notna()
    mode.loc[valid & (normalised <= stopped_ratio)] = "stopped"
    mode.loc[
        valid & (normalised > stopped_ratio) & (normalised < generating_ratio)
    ] = "low_power"
    mode.loc[valid & (normalised >= generating_ratio)] = "generating"
    return mode, normalised, active_scale


# =============================================================================
# 4. DUAL-BASELINE ROW FEATURES
# =============================================================================

def calculate_dual_baseline_row_features(
    df: pd.DataFrame,
    timestamp_col: str,
    measurement_columns: list[str],
    reference_mask: pd.Series,
    operating_mode: pd.Series,
    sampling_minutes: float,
    dynamic_window_hours: float,
    minimum_dynamic_points: int,
    minimum_mode_reference_points: int,
    z_threshold: float,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not measurement_columns:
        raise ValueError("No measurement columns selected for dual-baseline features.")

    numeric = df[measurement_columns].apply(pd.to_numeric, errors="coerce")
    n_rows = len(df)
    n_signals = len(measurement_columns)
    dynamic_window_points = max(
        minimum_dynamic_points,
        int(round(dynamic_window_hours * 60.0 / sampling_minutes)),
    )

    global_reference = numeric.loc[reference_mask]
    global_median, global_scale = robust_location_scale(global_reference)

    modes = ["generating", "low_power", "stopped"]
    mode_medians: dict[str, pd.Series] = {}
    mode_scales: dict[str, pd.Series] = {}
    mode_reference_counts: dict[str, int] = {}
    mode_reference_available: dict[str, bool] = {}

    for mode in modes:
        mask = reference_mask & (operating_mode == mode)
        count = int(mask.sum())
        mode_reference_counts[mode] = count
        available = count >= minimum_mode_reference_points
        mode_reference_available[mode] = available
        if available:
            centre, scale = robust_location_scale(numeric.loc[mask])
            mode_medians[mode] = centre
            mode_scales[mode] = scale
        else:
            mode_medians[mode] = global_median
            mode_scales[mode] = global_scale

    fixed_abnormal_count = np.zeros(n_rows, dtype=np.int32)
    dynamic_abnormal_count = np.zeros(n_rows, dtype=np.int32)
    dual_abnormal_count = np.zeros(n_rows, dtype=np.int32)
    fixed_only_count = np.zeros(n_rows, dtype=np.int32)
    dynamic_only_count = np.zeros(n_rows, dtype=np.int32)
    dynamic_valid_count = np.zeros(n_rows, dtype=np.int32)
    fixed_valid_count = np.zeros(n_rows, dtype=np.int32)

    for batch_start in range(0, n_signals, batch_size):
        batch_columns = measurement_columns[batch_start:batch_start + batch_size]
        batch = numeric[batch_columns]

        fixed_z = pd.DataFrame(np.nan, index=df.index, columns=batch_columns)
        for mode in modes:
            rows = operating_mode == mode
            if not bool(rows.any()):
                continue
            centre = mode_medians[mode].reindex(batch_columns)
            scale = mode_scales[mode].reindex(batch_columns)
            fixed_z.loc[rows, batch_columns] = (
                batch.loc[rows, batch_columns] - centre
            ).abs().div(scale, axis=1)

        unknown_rows = ~operating_mode.isin(modes)
        if bool(unknown_rows.any()):
            fixed_z.loc[unknown_rows, batch_columns] = (
                batch.loc[unknown_rows, batch_columns]
                - global_median.reindex(batch_columns)
            ).abs().div(global_scale.reindex(batch_columns), axis=1)

        dynamic_z = pd.DataFrame(np.nan, index=df.index, columns=batch_columns)
        for mode in modes:
            rows = operating_mode == mode
            if not bool(rows.any()):
                continue
            masked = batch.where(rows)
            rolling = masked.rolling(
                window=dynamic_window_points,
                min_periods=minimum_dynamic_points,
            )
            centre = rolling.median().shift(1)
            q25 = rolling.quantile(0.25).shift(1)
            q75 = rolling.quantile(0.75).shift(1)
            scale = (q75 - q25) / 1.349
            std = rolling.std().shift(1)
            bad = (~np.isfinite(scale)) | (scale <= EPSILON)
            scale = scale.mask(bad, std)
            bad = (~np.isfinite(scale)) | (scale <= EPSILON)
            scale = scale.mask(bad)
            dynamic_z.loc[rows, batch_columns] = (
                batch.loc[rows, batch_columns] - centre.loc[rows, batch_columns]
            ).abs().div(scale.loc[rows, batch_columns])

        fixed_valid = fixed_z.notna().to_numpy(dtype=bool)
        dynamic_valid = dynamic_z.notna().to_numpy(dtype=bool)
        fixed_abnormal = (fixed_z >= z_threshold).fillna(False).to_numpy(dtype=bool)
        dynamic_abnormal = (
            (dynamic_z >= z_threshold).fillna(False).to_numpy(dtype=bool)
        )
        comparable = fixed_valid & dynamic_valid

        fixed_valid_count += fixed_valid.sum(axis=1).astype(np.int32)
        dynamic_valid_count += dynamic_valid.sum(axis=1).astype(np.int32)
        fixed_abnormal_count += fixed_abnormal.sum(axis=1).astype(np.int32)
        dynamic_abnormal_count += dynamic_abnormal.sum(axis=1).astype(np.int32)
        dual_abnormal_count += (
            fixed_abnormal & dynamic_abnormal & comparable
        ).sum(axis=1).astype(np.int32)
        fixed_only_count += (
            fixed_abnormal & ~dynamic_abnormal & comparable
        ).sum(axis=1).astype(np.int32)
        dynamic_only_count += (
            ~fixed_abnormal & dynamic_abnormal & comparable
        ).sum(axis=1).astype(np.int32)

    fixed_denominator = np.maximum(fixed_valid_count, 1)
    dynamic_denominator = np.maximum(dynamic_valid_count, 1)

    row_features = pd.DataFrame(
        {
            "time_stamp": pd.to_datetime(df[timestamp_col], errors="coerce"),
            "operating_mode": operating_mode.astype(str),
            "fixed_valid_signal_count": fixed_valid_count,
            "dynamic_valid_signal_count": dynamic_valid_count,
            "fixed_abnormal_signal_count": fixed_abnormal_count,
            "dynamic_abnormal_signal_count": dynamic_abnormal_count,
            "dual_abnormal_signal_count": dual_abnormal_count,
            "fixed_only_abnormal_signal_count": fixed_only_count,
            "dynamic_only_abnormal_signal_count": dynamic_only_count,
            "fixed_abnormal_fraction": fixed_abnormal_count / fixed_denominator,
            "dynamic_abnormal_fraction": dynamic_abnormal_count / dynamic_denominator,
            "dual_abnormal_fraction": dual_abnormal_count / dynamic_denominator,
            "fixed_only_abnormal_fraction": fixed_only_count / dynamic_denominator,
            "dynamic_only_abnormal_fraction": dynamic_only_count / dynamic_denominator,
            "baseline_disagreement_fraction": (
                fixed_only_count + dynamic_only_count
            ) / dynamic_denominator,
            "dynamic_reference_signal_fraction": dynamic_valid_count / max(n_signals, 1),
        },
        index=df.index,
    )

    row_features.loc[
        row_features["dynamic_valid_signal_count"] <= 0,
        [
            "dynamic_abnormal_fraction",
            "dual_abnormal_fraction",
            "fixed_only_abnormal_fraction",
            "dynamic_only_abnormal_fraction",
            "baseline_disagreement_fraction",
        ],
    ] = np.nan

    diagnostics = {
        "measurement_signal_count": n_signals,
        "dynamic_window_points": dynamic_window_points,
        "mode_reference_counts": mode_reference_counts,
        "mode_reference_available": mode_reference_available,
        "global_reference_rows": int(reference_mask.sum()),
    }
    return row_features, diagnostics


# =============================================================================
# 5. TRUE POST-EVENT RIGHT-CENSORING
# =============================================================================

def first_false_run_start(
    flag: pd.Series,
    minimum_false_points: int,
) -> Optional[int]:
    values = flag.fillna(False).astype(bool).to_numpy()
    run_start: Optional[int] = None
    run_length = 0
    for position, value in enumerate(values):
        if not value:
            if run_start is None:
                run_start = position
            run_length += 1
            if run_length >= minimum_false_points:
                return run_start
        else:
            run_start = None
            run_length = 0
    return None


def detector_post_event_features(
    rows: pd.DataFrame,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    raw_flag_column: str,
    sampling_minutes: float,
    tail_points: int,
    post_check_points: int,
    recovery_false_points: int,
    active_fraction_threshold: float,
) -> dict[str, Any]:
    prefix = raw_flag_column.replace("_raw_flag", "")
    output = {
        f"{prefix}_tail_raw_active_fraction": 0.0,
        f"{prefix}_post_end_raw_active_fraction": 0.0,
        f"{prefix}_active_after_event": False,
        f"{prefix}_recovery_observed_after_event": False,
        f"{prefix}_recovery_delay_hours": np.nan,
        f"{prefix}_right_censored": False,
        f"{prefix}_post_event_incomplete": False,
    }

    if raw_flag_column not in rows.columns or rows.empty:
        output[f"{prefix}_post_event_incomplete"] = True
        return output

    event_rows = rows.loc[
        (rows["time_stamp"] >= event_start)
        & (rows["time_stamp"] <= event_end)
    ]
    post_rows = rows.loc[rows["time_stamp"] > event_end].copy()

    tail = to_bool_series(event_rows[raw_flag_column].tail(tail_points))
    tail_fraction = float(tail.mean()) if len(tail) else 0.0
    output[f"{prefix}_tail_raw_active_fraction"] = tail_fraction
    tail_active = tail_fraction >= active_fraction_threshold

    if post_rows.empty:
        output[f"{prefix}_post_event_incomplete"] = bool(tail_active)
        return output

    post_flags = to_bool_series(post_rows[raw_flag_column])
    first_flags = post_flags.head(post_check_points)
    post_fraction = float(first_flags.mean()) if len(first_flags) else 0.0
    output[f"{prefix}_post_end_raw_active_fraction"] = post_fraction
    output[f"{prefix}_active_after_event"] = bool(
        post_fraction >= active_fraction_threshold
    )

    if not tail_active:
        return output

    recovery_position = first_false_run_start(
        post_flags,
        minimum_false_points=recovery_false_points,
    )
    if recovery_position is not None:
        recovery_time = pd.Timestamp(post_rows.iloc[recovery_position]["time_stamp"])
        output[f"{prefix}_recovery_observed_after_event"] = True
        output[f"{prefix}_recovery_delay_hours"] = float(
            (recovery_time - event_end).total_seconds() / 3600.0
        )
    else:
        output[f"{prefix}_right_censored"] = bool(
            output[f"{prefix}_active_after_event"]
        )
        output[f"{prefix}_post_event_incomplete"] = bool(
            len(post_rows) < post_check_points
        )

    return output


def calculate_post_event_censoring(
    row_scores_path: Path,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    post_event_hours: float,
    tail_points: int,
    post_check_points: int,
    recovery_false_points: int,
    active_fraction_threshold: float,
) -> dict[str, Any]:
    base = {
        "row_scores_available": False,
        "post_event_data_available": False,
        "post_event_points_available": 0,
        "post_event_hours_available": 0.0,
        "right_censored_confirmed": False,
        "post_event_data_incomplete": False,
    }
    if not row_scores_path.exists():
        base["post_event_data_incomplete"] = True
        return base

    rows = pd.read_csv(row_scores_path, low_memory=False)
    if "time_stamp" not in rows.columns:
        base["post_event_data_incomplete"] = True
        return base

    rows["time_stamp"] = pd.to_datetime(rows["time_stamp"], errors="coerce")
    rows = rows.dropna(subset=["time_stamp"]).sort_values("time_stamp")
    rows = rows.loc[
        rows["time_stamp"]
        <= event_end + pd.Timedelta(hours=post_event_hours)
    ].copy()
    if rows.empty:
        base["post_event_data_incomplete"] = True
        return base

    sampling_minutes = infer_sampling_minutes(rows["time_stamp"])
    post_rows = rows.loc[rows["time_stamp"] > event_end]
    base["row_scores_available"] = True
    base["post_event_data_available"] = not post_rows.empty
    base["post_event_points_available"] = int(len(post_rows))
    if not post_rows.empty:
        base["post_event_hours_available"] = float(
            (
                post_rows["time_stamp"].max()
                - event_end
            ).total_seconds()
            / 3600.0
        )

    localized = detector_post_event_features(
        rows=rows,
        event_start=event_start,
        event_end=event_end,
        raw_flag_column="localized_persistent_subsystem_state_raw_flag",
        sampling_minutes=sampling_minutes,
        tail_points=tail_points,
        post_check_points=post_check_points,
        recovery_false_points=recovery_false_points,
        active_fraction_threshold=active_fraction_threshold,
    )
    persistent = detector_post_event_features(
        rows=rows,
        event_start=event_start,
        event_end=event_end,
        raw_flag_column="persistent_system_state_raw_flag",
        sampling_minutes=sampling_minutes,
        tail_points=tail_points,
        post_check_points=post_check_points,
        recovery_false_points=recovery_false_points,
        active_fraction_threshold=active_fraction_threshold,
    )
    base.update(localized)
    base.update(persistent)

    base["right_censored_confirmed"] = bool(
        localized.get(
            "localized_persistent_subsystem_state_right_censored", False
        )
        or persistent.get("persistent_system_state_right_censored", False)
    )
    base["post_event_data_incomplete"] = bool(
        not base["post_event_data_available"]
        or localized.get(
            "localized_persistent_subsystem_state_post_event_incomplete", False
        )
        or persistent.get("persistent_system_state_post_event_incomplete", False)
    )
    return base


# =============================================================================
# 6. EVENT-LEVEL AGGREGATION
# =============================================================================

def aggregate_event_features(
    metadata: EventMetadata,
    row_features: pd.DataFrame,
    normalised_power: pd.Series,
    diagnostics: dict[str, Any],
    sampling_minutes: float,
    row_abnormal_fraction_threshold: float,
    censoring: dict[str, Any],
) -> dict[str, Any]:
    event_mask = (
        (row_features["time_stamp"] >= metadata.event_start)
        & (row_features["time_stamp"] <= metadata.event_end)
    )
    event_rows = row_features.loc[event_mask].copy()
    if event_rows.empty:
        raise ValueError(
            f"No dual-baseline rows inside event interval {metadata.event_id}."
        )

    event_power = pd.to_numeric(normalised_power.loc[event_rows.index], errors="coerce")
    event_duration_hours = max(
        (metadata.event_end - metadata.event_start).total_seconds() / 3600.0,
        sampling_minutes / 60.0,
    )
    event_duration_days = event_duration_hours / 24.0

    output: dict[str, Any] = {
        "farm_id": metadata.farm_id,
        "event_id": metadata.event_id,
        "metadata_label_v3": metadata.event_label,
        "metadata_start_v3": metadata.event_start,
        "metadata_end_v3": metadata.event_end,
        "dual_baseline_measurement_signal_count": diagnostics[
            "measurement_signal_count"
        ],
        "dual_baseline_dynamic_window_points": diagnostics[
            "dynamic_window_points"
        ],
        "dual_baseline_reference_rows": diagnostics["global_reference_rows"],
        "mode_generating_reference_rows": diagnostics[
            "mode_reference_counts"
        ]["generating"],
        "mode_low_power_reference_rows": diagnostics[
            "mode_reference_counts"
        ]["low_power"],
        "mode_stopped_reference_rows": diagnostics[
            "mode_reference_counts"
        ]["stopped"],
        "mode_generating_reference_available": diagnostics[
            "mode_reference_available"
        ]["generating"],
        "mode_low_power_reference_available": diagnostics[
            "mode_reference_available"
        ]["low_power"],
        "mode_stopped_reference_available": diagnostics[
            "mode_reference_available"
        ]["stopped"],
        "dynamic_reference_available_fraction": float(
            event_rows["dynamic_reference_signal_fraction"].fillna(0.0).mean()
        ),
        "normalised_power_mean": float(event_power.mean())
        if event_power.notna().any()
        else np.nan,
        "normalised_power_std": float(event_power.std())
        if event_power.notna().sum() > 1
        else 0.0,
    }

    for mode in ["generating", "low_power", "stopped", "unknown"]:
        output[f"operating_{mode}_fraction"] = float(
            (event_rows["operating_mode"] == mode).mean()
        )

    mode_values = event_rows["operating_mode"].astype(str)
    transition_count = int((mode_values != mode_values.shift(1)).sum() - 1)
    transition_count = max(0, transition_count)
    output["operating_mode_transition_count"] = transition_count
    output["operating_mode_transition_rate_per_day"] = (
        transition_count / event_duration_days if event_duration_days > 0 else 0.0
    )

    fraction_columns = {
        "fixed_abnormal_fraction": "fixed_abnormal_fraction",
        "dynamic_abnormal_fraction": "dynamic_abnormal_fraction",
        "dual_abnormal_fraction": "dual_abnormal_fraction",
        "fixed_only_abnormal_fraction": "fixed_only_abnormal_fraction",
        "dynamic_only_abnormal_fraction": "dynamic_only_abnormal_fraction",
        "baseline_disagreement_fraction": "baseline_disagreement_fraction",
    }
    for column, prefix in fraction_columns.items():
        output.update(
            summarise_fraction_series(
                event_rows[column],
                prefix=prefix,
                row_threshold=row_abnormal_fraction_threshold,
                sampling_minutes=sampling_minutes,
            )
        )

    for mode in ["generating", "low_power", "stopped"]:
        mode_rows = event_rows.loc[event_rows["operating_mode"] == mode]
        output[f"{mode}_row_count"] = int(len(mode_rows))
        for column in [
            "fixed_abnormal_fraction",
            "dynamic_abnormal_fraction",
            "dual_abnormal_fraction",
            "fixed_only_abnormal_fraction",
            "dynamic_only_abnormal_fraction",
        ]:
            values = pd.to_numeric(mode_rows.get(column), errors="coerce").dropna()
            output[f"{mode}_{column}_mean"] = (
                float(values.mean()) if not values.empty else 0.0
            )
            output[f"{mode}_{column}_max"] = (
                float(values.max()) if not values.empty else 0.0
            )
            output[f"{mode}_{column}_row_coverage"] = (
                float((values >= row_abnormal_fraction_threshold).mean())
                if not values.empty
                else 0.0
            )

    output.update(censoring)
    return output


def analyse_one_event(
    metadata: EventMetadata,
    event_file: Path,
    detector_output_dir: Path,
    event_output_dir: Path,
    fixed_reference_hours: float,
    dynamic_window_hours: float,
    post_event_hours: float,
    measurement_mode: str,
    power_signals: list[str],
    stopped_ratio: float,
    generating_ratio: float,
    minimum_reference_points: int,
    minimum_mode_reference_points: int,
    minimum_dynamic_points: int,
    minimum_valid_fraction: float,
    z_threshold: float,
    row_abnormal_fraction_threshold: float,
    batch_size: int,
    tail_points: int,
    post_check_points: int,
    recovery_false_points: int,
    active_fraction_threshold: float,
    save_row_features: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print(f"[INFO] Event {metadata.event_id}: {event_file}")
    raw = read_semicolon_csv(event_file)
    timestamp_col = find_column(raw.columns, TIMESTAMP_CANDIDATES)
    if timestamp_col is None:
        raise ValueError(f"No timestamp column found in {event_file}")
    raw[timestamp_col] = pd.to_datetime(raw[timestamp_col], errors="coerce")
    raw = raw.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    raw = raw.drop_duplicates(subset=[timestamp_col]).reset_index(drop=True)

    context_start = metadata.event_start - pd.Timedelta(hours=fixed_reference_hours)
    context_end = metadata.event_end + pd.Timedelta(hours=post_event_hours)
    df = raw.loc[
        (raw[timestamp_col] >= context_start)
        & (raw[timestamp_col] <= context_end)
    ].copy()
    if df.empty:
        raise ValueError("No rows remain inside requested analysis context.")
    df = df.reset_index(drop=True)

    sampling_minutes = infer_sampling_minutes(df[timestamp_col])
    reference_mask = (
        (df[timestamp_col] >= context_start)
        & (df[timestamp_col] < metadata.event_start)
    )
    if int(reference_mask.sum()) < minimum_reference_points:
        fallback = df[timestamp_col] < metadata.event_start
        if int(fallback.sum()) < minimum_reference_points:
            raise ValueError(
                f"Only {int(fallback.sum())} pre-event rows; "
                f"minimum={minimum_reference_points}."
            )
        warnings.warn(
            f"Event {metadata.event_id}: using all available pre-event rows "
            "because the preferred fixed-reference window is incomplete.",
            RuntimeWarning,
        )
        reference_mask = fallback

    power_columns = identify_power_average_columns(df, power_signals)
    operating_mode, normalised_power, power_scale = classify_operating_mode(
        df=df,
        power_columns=power_columns,
        reference_mask=reference_mask,
        stopped_ratio=stopped_ratio,
        generating_ratio=generating_ratio,
    )

    measurements = select_measurement_columns(
        df=df,
        measurement_mode=measurement_mode,
        excluded_base_signals=set(power_signals),
        minimum_valid_fraction=minimum_valid_fraction,
    )
    if not measurements:
        raise ValueError("No usable measurement columns were selected.")

    row_features, diagnostics = calculate_dual_baseline_row_features(
        df=df,
        timestamp_col=timestamp_col,
        measurement_columns=measurements,
        reference_mask=reference_mask,
        operating_mode=operating_mode,
        sampling_minutes=sampling_minutes,
        dynamic_window_hours=dynamic_window_hours,
        minimum_dynamic_points=minimum_dynamic_points,
        minimum_mode_reference_points=minimum_mode_reference_points,
        z_threshold=z_threshold,
        batch_size=batch_size,
    )
    row_features["normalised_consensus_power"] = normalised_power

    row_scores_path = (
        detector_output_dir / f"event_{metadata.event_id}" / "row_scores.csv"
    )
    censoring = calculate_post_event_censoring(
        row_scores_path=row_scores_path,
        event_start=metadata.event_start,
        event_end=metadata.event_end,
        post_event_hours=post_event_hours,
        tail_points=tail_points,
        post_check_points=post_check_points,
        recovery_false_points=recovery_false_points,
        active_fraction_threshold=active_fraction_threshold,
    )

    features = aggregate_event_features(
        metadata=metadata,
        row_features=row_features,
        normalised_power=normalised_power,
        diagnostics=diagnostics,
        sampling_minutes=sampling_minutes,
        row_abnormal_fraction_threshold=row_abnormal_fraction_threshold,
        censoring=censoring,
    )

    event_output_dir.mkdir(parents=True, exist_ok=True)
    if save_row_features:
        row_features.to_csv(
            event_output_dir / "dual_baseline_mode_row_features.csv",
            index=False,
        )
    write_json(
        event_output_dir / "dual_baseline_mode_configuration.json",
        {
            "farm_id": metadata.farm_id,
            "event_id": metadata.event_id,
            "event_file": event_file,
            "sampling_minutes": sampling_minutes,
            "fixed_reference_hours": fixed_reference_hours,
            "dynamic_window_hours": dynamic_window_hours,
            "post_event_hours": post_event_hours,
            "measurement_mode": measurement_mode,
            "measurement_columns": len(measurements),
            "power_columns": power_columns,
            "power_reference_scale": power_scale,
            "stopped_ratio": stopped_ratio,
            "generating_ratio": generating_ratio,
            "z_threshold": z_threshold,
            "row_abnormal_fraction_threshold": row_abnormal_fraction_threshold,
            "diagnostics": diagnostics,
            "censoring": censoring,
        },
    )

    summary = {
        "farm_id": metadata.farm_id,
        "event_id": metadata.event_id,
        "event_label": metadata.event_label,
        "event_file": str(event_file),
        "row_scores_path": str(row_scores_path),
        "sampling_minutes": sampling_minutes,
        "measurement_columns": len(measurements),
        "power_columns": ";".join(power_columns),
        "reference_rows": int(reference_mask.sum()),
        "event_rows": int(
            (
                (row_features["time_stamp"] >= metadata.event_start)
                & (row_features["time_stamp"] <= metadata.event_end)
            ).sum()
        ),
        "status": "success",
        "error": "",
    }
    return features, summary


# =============================================================================
# 7. COMMAND LINE AND MAIN
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed+dynamic baseline, operating-mode and post-event "
            "right-censoring features for event-level SCADA classification."
        )
    )
    parser.add_argument("--farm", default="C")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--event-dir", required=True, type=Path)
    parser.add_argument("--detector-output-dir", required=True, type=Path)
    parser.add_argument("--base-features-file", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--event-id", default="all")
    parser.add_argument("--fixed-reference-hours", type=float, default=48.0)
    parser.add_argument("--dynamic-window-hours", type=float, default=24.0)
    parser.add_argument("--post-event-hours", type=float, default=12.0)
    parser.add_argument(
        "--measurement-mode",
        choices=["avg_only", "all"],
        default="avg_only",
    )
    parser.add_argument(
        "--power-signals",
        default="power_2,power_5,power_6,power_17",
    )
    parser.add_argument("--stopped-power-ratio", type=float, default=0.05)
    parser.add_argument("--generating-power-ratio", type=float, default=0.30)
    parser.add_argument("--minimum-reference-points", type=int, default=72)
    parser.add_argument("--minimum-mode-reference-points", type=int, default=12)
    parser.add_argument("--minimum-dynamic-points", type=int, default=12)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.50)
    parser.add_argument("--dual-baseline-z-threshold", type=float, default=8.0)
    parser.add_argument(
        "--row-abnormal-fraction-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument("--column-batch-size", type=int, default=32)
    parser.add_argument("--censor-tail-points", type=int, default=6)
    parser.add_argument("--post-check-points", type=int, default=6)
    parser.add_argument("--recovery-false-points", type=int, default=3)
    parser.add_argument(
        "--post-active-fraction-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--save-row-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.fixed_reference_hours <= 0 or args.dynamic_window_hours <= 0:
        parser.error("Reference and dynamic window hours must be > 0.")
    if args.post_event_hours < 0:
        parser.error("--post-event-hours must be >= 0.")
    if not 0 <= args.stopped_power_ratio < args.generating_power_ratio:
        parser.error(
            "Power ratios must satisfy 0 <= stopped < generating."
        )
    for name in [
        "minimum_reference_points",
        "minimum_mode_reference_points",
        "minimum_dynamic_points",
        "column_batch_size",
        "censor_tail_points",
        "post_check_points",
        "recovery_false_points",
    ]:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0.")
    for name in [
        "minimum_valid_fraction",
        "row_abnormal_fraction_threshold",
        "post_active_fraction_threshold",
    ]:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    power_signals = parse_power_signals(args.power_signals)
    events = load_metadata(args.metadata, args.farm)
    if str(args.event_id).lower() != "all":
        target = normalise_event_id(args.event_id)
        events = [event for event in events if event.event_id == target]
    if not events:
        print("[ERROR] No metadata rows matched the requested event selection.")
        return 1

    base_features_path = resolve_base_features_file(
        args.detector_output_dir,
        args.base_features_file,
    )
    print(f"[INPUT] Base features: {base_features_path}")

    feature_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for event in events:
        try:
            event_file = resolve_event_file(args.event_dir, event.event_id)
            event_output_dir = args.output_dir / f"event_{event.event_id}"
            features, summary = analyse_one_event(
                metadata=event,
                event_file=event_file,
                detector_output_dir=args.detector_output_dir,
                event_output_dir=event_output_dir,
                fixed_reference_hours=args.fixed_reference_hours,
                dynamic_window_hours=args.dynamic_window_hours,
                post_event_hours=args.post_event_hours,
                measurement_mode=args.measurement_mode,
                power_signals=power_signals,
                stopped_ratio=args.stopped_power_ratio,
                generating_ratio=args.generating_power_ratio,
                minimum_reference_points=args.minimum_reference_points,
                minimum_mode_reference_points=args.minimum_mode_reference_points,
                minimum_dynamic_points=args.minimum_dynamic_points,
                minimum_valid_fraction=args.minimum_valid_fraction,
                z_threshold=args.dual_baseline_z_threshold,
                row_abnormal_fraction_threshold=(
                    args.row_abnormal_fraction_threshold
                ),
                batch_size=args.column_batch_size,
                tail_points=args.censor_tail_points,
                post_check_points=args.post_check_points,
                recovery_false_points=args.recovery_false_points,
                active_fraction_threshold=(
                    args.post_active_fraction_threshold
                ),
                save_row_features=args.save_row_features,
            )
            feature_rows.append(features)
            run_rows.append(summary)
        except Exception as exc:
            print(
                f"[ERROR] Event {event.event_id}: {exc}",
                file=sys.stderr,
            )
            run_rows.append(
                {
                    "farm_id": event.farm_id,
                    "event_id": event.event_id,
                    "event_label": event.event_label,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    new_features = pd.DataFrame(feature_rows)
    run_summary = pd.DataFrame(run_rows)
    new_features.to_csv(
        args.output_dir / "all_events_dual_baseline_mode_features.csv",
        index=False,
    )
    run_summary.to_csv(
        args.output_dir / "feature_build_run_summary.csv",
        index=False,
    )

    base = pd.read_csv(base_features_path, low_memory=False)
    if "event_id" not in base.columns:
        raise ValueError("Base feature table does not contain event_id.")
    base["event_id"] = base["event_id"].map(normalise_event_id)
    if not new_features.empty:
        new_features["event_id"] = new_features["event_id"].map(
            normalise_event_id
        )
        duplicate_columns = [
            column
            for column in new_features.columns
            if column in base.columns and column != "event_id"
        ]
        if duplicate_columns:
            new_features = new_features.drop(columns=duplicate_columns)
        merged = base.merge(new_features, on="event_id", how="left")
    else:
        merged = base.copy()

    merged.to_csv(
        args.output_dir / "all_events_event_level_features_v3.csv",
        index=False,
    )

    manifest = {
        "metadata": args.metadata,
        "event_dir": args.event_dir,
        "detector_output_dir": args.detector_output_dir,
        "base_features_file": base_features_path,
        "output_dir": args.output_dir,
        "selected_event_count": len(events),
        "successful_event_count": int((run_summary["status"] == "success").sum()),
        "failed_event_count": int((run_summary["status"] == "failed").sum()),
        "fixed_reference_hours": args.fixed_reference_hours,
        "dynamic_window_hours": args.dynamic_window_hours,
        "post_event_hours": args.post_event_hours,
        "measurement_mode": args.measurement_mode,
        "power_signals": power_signals,
        "stopped_power_ratio": args.stopped_power_ratio,
        "generating_power_ratio": args.generating_power_ratio,
        "dual_baseline_z_threshold": args.dual_baseline_z_threshold,
        "row_abnormal_fraction_threshold": (
            args.row_abnormal_fraction_threshold
        ),
    }
    write_json(args.output_dir / "feature_build_manifest.json", manifest)

    print(
        f"[DONE] Success={manifest['successful_event_count']}; "
        f"failed={manifest['failed_event_count']}"
    )
    print(
        "[OUTPUT] "
        f"{args.output_dir / 'all_events_event_level_features_v3.csv'}"
    )
    return 0 if manifest["failed_event_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
