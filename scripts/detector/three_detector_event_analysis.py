#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-detector SCADA anomaly analysis for wind-turbine events.

Final tuned adaptive-threshold revision:
- short and persistent thresholds are calibrated from each event's pre-event data;
- short-standstill reference quantile increased from 0.985 to 0.995;
- persistent-system fraction floor increased from 0.15 to 0.45;
- short standstill requires stronger 75% power-signal consensus;
- intermittent clusters are built from repeated confirmed short-stop rows;
- adjacent short-stop flags are reduced to one strongest 10-minute candidate;
- --analysis-end can override one event's metadata end time without editing CSV.

Detectors
---------
1. short_standstill
   - Designed for a single 10-minute row or a very short interruption.
   - MUST have multisensor evidence AND power evidence.
   - Power evidence means:
       a) clear power dip, and
       b) next-row recovery OR strong within-window min/range/std evidence.

2. intermittent_cluster
   - Designed for repeated short stops, converter-torque oscillations,
     intermittent control changes, and short abnormal clusters.

3. persistent_system_state
   - Designed for long-lasting standstill or persistent system-level changes.

The same detector parameters are used for every event. Event-specific adaptation
comes from the pre-event reference period, which is used to calculate robust
median/MAD baselines for each signal.

Expected metadata columns
-------------------------
event_id;event_label;event_start;event_start_id;event_end;event_end_id;event_description

Expected event CSV
------------------
One file per event, normally named <event_id>.csv, containing:
time_stamp, id, and measurement columns such as:
sensor_0_avg, sensor_0_max, sensor_0_min, sensor_0_std,
power_2_avg, power_2_max, power_2_min, power_2_std, ...

Examples
--------
Single event:
python three_detector_event_analysis.py \
    --farm C \
    --metadata data/farm_C/event_metadata.csv \
    --event-dir data/farm_C/events \
    --event-id 35 \
    --output-dir outputs/three_detector

All events in one farm:
python three_detector_event_analysis.py \
    --farm C \
    --metadata data/farm_C/event_metadata.csv \
    --event-dir data/farm_C/events \
    --event-id all \
    --output-dir outputs/three_detector

Manual power-signal override:
python three_detector_event_analysis.py \
    --farm C \
    --metadata data/farm_C/event_metadata.csv \
    --event-dir data/farm_C/events \
    --event-id 35 \
    --power-signals power_2,power_5,power_6,power_17 \
    --output-dir outputs/three_detector

Override the analysis end time for one event:
python three_detector_event_analysis.py \
    --farm C \
    --metadata data/farm_C/event_metadata.csv \
    --event-dir data/farm_C/events \
    --event-id 18 \
    --analysis-end "2025-09-18 23:50:00" \
    --power-signals power_2,power_5,power_6,power_17 \
    --output-dir outputs/three_detector_event18
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

MEASUREMENT_MODE = "all"  # "all" or "avg_only"

REFERENCE_HOURS = 48
POST_EVENT_HOURS = 12
MIN_REFERENCE_POINTS = 72
MIN_REFERENCE_VALID_FRACTION = 0.50

LOCAL_POWER_REFERENCE_POINTS = 6
TOP_TRANSITION_CANDIDATES = 30
TOP_CONTRIBUTORS_PER_CANDIDATE = 20
PLOT_TOP_CONTRIBUTOR_SIGNALS = 20
TOP_CONTRIBUTOR_BAR_COUNT = 20
PLOT_OUTPUTS = True

ROBUST_Z_CLIP = 1_000_000.0
EPSILON = 1e-9

# The same detector logic is applied to every event. The abnormal-fraction
# thresholds for short and persistent detectors are calculated automatically
# from each event's pre-event reference period.
DETECTOR_CONFIGS: dict[str, dict[str, Any]] = {
    "short_standstill": {
        "z_threshold": 8.0,
        "fraction_reference_quantile": 0.995,
        "fraction_floor": 0.05,
        "rolling_points": 1,
        "min_segment_points": 1,
        "max_segment_points": 1,
        "max_gap_points": 0,
        "require_power_evidence": True,
    },
    "intermittent_cluster": {
        # This detector is based on the density of already-confirmed
        # short-standstill rows, rather than a second loose absolute threshold.
        "source": "confirmed_short_standstill_density",
        "rolling_points": 12,          # approximately two hours at 10-min sampling
        "minimum_short_flags": 2,      # at least two confirmed short-stop rows
        "min_segment_points": 6,       # at least about one hour of abnormal activity
        "max_gap_points": 3,
        "require_power_evidence": False,
    },
    "persistent_system_state": {
        "z_threshold": 12.0,
        "fraction_reference_quantile": 0.995,
        "fraction_floor": 0.45,
        "rolling_points": 6,
        "min_segment_points": 12,
        "max_gap_points": 2,
        "require_power_evidence": False,
    },
}

POWER_EVIDENCE_CONFIG: dict[str, Any] = {
    "local_reference_points": LOCAL_POWER_REFERENCE_POINTS,
    "minimum_power_drop_ratio": 0.40,
    "minimum_recovery_ratio": 0.70,
    "maximum_min_power_ratio": 0.20,
    "power_dip_z_threshold": 6.0,
    "recovery_z_threshold": 5.0,
    "variability_z_threshold": 5.0,

    # Stronger agreement than the previous 50% / two-signal rule.
    # With four Farm C power signals, at least three must agree.
    "power_signal_consensus_fraction": 0.75,
    "minimum_power_signal_count": 3,

    "active_power_reference_quantile": 0.30,
}

# Optional farm-specific overrides.
# Edit these only when engineering knowledge confirms the correct active-power
# base signals. Auto-detection is used when a farm is not listed.
POWER_SIGNAL_OVERRIDES: dict[str, list[str]] = {
    "C": ["power_2", "power_5", "power_6", "power_17"],
}

TIMESTAMP_CANDIDATES = [
    "time_stamp", "timestamp", "datetime", "date_time", "time", "date"
]
ROW_ID_CANDIDATES = ["id", "row_id", "scada_id", "record_id"]

NON_MEASUREMENT_COLUMNS = {
    "time_stamp", "timestamp", "datetime", "date_time", "time", "date",
    "asset_id", "id", "row_id", "scada_id", "record_id",
    "train_test", "status_type_id", "event_id", "event_label",
}

STAT_SUFFIX_PATTERN = re.compile(r"_(avg|max|min|std)$", re.IGNORECASE)


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

@dataclass
class EventMetadata:
    farm_id: str
    event_id: str
    event_label: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    event_start_id: Optional[int]
    event_end_id: Optional[int]
    event_description: str


@dataclass
class Segment:
    farm_id: str
    event_id: str
    detector: str
    event_description: str
    segment_id: str
    segment_start: pd.Timestamp
    segment_end: pd.Timestamp
    start_id: Optional[int]
    end_id: Optional[int]
    n_points: int
    duration_minutes: float
    strongest_candidate_time: pd.Timestamp
    strongest_candidate_id: Optional[int]
    detector_score_max: float
    abnormal_fraction_max: float
    power_evidence_available: bool
    power_evidence_confirmed: bool
    power_signal_count: int
    power_dip_signal_count_max: int
    power_recovery_signal_count_max: int
    power_variability_signal_count_max: int
    power_dip_score_max: float
    power_recovery_score_max: float
    power_variability_score_max: float
    confidence: str
    parent_cluster_id: Optional[str] = None


# =============================================================================
# 3. BASIC UTILITIES
# =============================================================================

def safe_int(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalise_event_id(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lower_map = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=";", low_memory=False)

    # Fallback for malformed reads where the complete header became one column.
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";", engine="python", low_memory=False)

    df.columns = [str(col).strip() for col in df.columns]
    return df


def infer_sampling_minutes(timestamps: pd.Series) -> float:
    diffs = timestamps.sort_values().diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        raise ValueError("Could not infer sampling interval from timestamps.")
    return float(diffs.median())


def robust_location_scale(reference: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Robust centre and scale per column.

    Priority:
      1. MAD × 1.4826
      2. IQR / 1.349
      3. standard deviation
    """
    median = reference.median(axis=0, skipna=True)
    absolute_deviation = (reference - median).abs()
    mad_scale = absolute_deviation.median(axis=0, skipna=True) * 1.4826

    q25 = reference.quantile(0.25)
    q75 = reference.quantile(0.75)
    iqr_scale = (q75 - q25) / 1.349

    std_scale = reference.std(axis=0, skipna=True)

    scale = mad_scale.copy()
    bad = (~np.isfinite(scale)) | (scale <= EPSILON)
    scale.loc[bad] = iqr_scale.loc[bad]

    bad = (~np.isfinite(scale)) | (scale <= EPSILON)
    scale.loc[bad] = std_scale.loc[bad]

    bad = (~np.isfinite(scale)) | (scale <= EPSILON)
    scale.loc[bad] = np.nan

    return median, scale


def robust_scale_1d(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan

    median = float(values.median())
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if np.isfinite(mad) and mad > EPSILON:
        return mad

    q25, q75 = values.quantile([0.25, 0.75])
    iqr = float((q75 - q25) / 1.349)
    if np.isfinite(iqr) and iqr > EPSILON:
        return iqr

    std = float(values.std())
    if np.isfinite(std) and std > EPSILON:
        return std

    return np.nan


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).fillna(0.0)


# =============================================================================
# 4. METADATA AND EVENT FILE LOADING
# =============================================================================

def load_metadata(metadata_path: Path, farm_id: str) -> list[EventMetadata]:
    df = read_semicolon_csv(metadata_path)

    required = [
        "event_id", "event_label", "event_start", "event_end", "event_description"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["event_start"] = pd.to_datetime(df["event_start"], errors="coerce")
    df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")

    if df["event_start"].isna().any() or df["event_end"].isna().any():
        raise ValueError("Some metadata event_start/event_end values could not be parsed.")

    events: list[EventMetadata] = []
    for _, row in df.iterrows():
        events.append(
            EventMetadata(
                farm_id=str(farm_id),
                event_id=normalise_event_id(row["event_id"]),
                event_label=str(row.get("event_label", "")),
                event_start=pd.Timestamp(row["event_start"]),
                event_end=pd.Timestamp(row["event_end"]),
                event_start_id=safe_int(row.get("event_start_id")),
                event_end_id=safe_int(row.get("event_end_id")),
                event_description=str(row.get("event_description", "")),
            )
        )
    return events


def resolve_event_file(
    event_dir: Path,
    event_id: str,
    explicit_event_file: Optional[Path] = None,
) -> Path:
    if explicit_event_file is not None:
        if not explicit_event_file.exists():
            raise FileNotFoundError(f"Explicit event file not found: {explicit_event_file}")
        return explicit_event_file

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
            f"Multiple possible files found for event {event_id}: {matches}. "
            "Use --event-file for a single-event run."
        )

    raise FileNotFoundError(
        f"No event CSV found for event {event_id} in {event_dir}."
    )


def prepare_event_dataframe(
    event_file: Path,
    metadata: EventMetadata,
) -> tuple[pd.DataFrame, str, Optional[str], float]:
    df = read_semicolon_csv(event_file)

    timestamp_col = find_column(df.columns, TIMESTAMP_CANDIDATES)
    if timestamp_col is None:
        raise ValueError(
            f"No timestamp column found. Tried: {TIMESTAMP_CANDIDATES}"
        )

    row_id_col = find_column(df.columns, ROW_ID_CANDIDATES)

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).copy()
    df = df.sort_values(timestamp_col).drop_duplicates(timestamp_col).reset_index(drop=True)

    context_start = metadata.event_start - pd.Timedelta(hours=REFERENCE_HOURS)
    context_end = metadata.event_end + pd.Timedelta(hours=POST_EVENT_HOURS)

    context_df = df.loc[
        (df[timestamp_col] >= context_start)
        & (df[timestamp_col] <= context_end)
    ].copy()

    # If the file does not include the requested full context, keep all available
    # rows up to the post-event end. This still allows a larger pre-event fallback.
    if context_df.empty:
        context_df = df.loc[df[timestamp_col] <= context_end].copy()

    if context_df.empty:
        raise ValueError("No usable rows remain after time filtering.")

    sampling_minutes = infer_sampling_minutes(context_df[timestamp_col])
    return context_df.reset_index(drop=True), timestamp_col, row_id_col, sampling_minutes


# =============================================================================
# 5. MEASUREMENT AND BASE-SIGNAL PREPARATION
# =============================================================================

def select_measurement_columns(df: pd.DataFrame) -> list[str]:
    selected: list[str] = []

    for col in df.columns:
        lower = col.lower()
        if lower in NON_MEASUREMENT_COLUMNS:
            continue

        suffix_match = STAT_SUFFIX_PATTERN.search(col)
        if suffix_match is None:
            continue

        stat = suffix_match.group(1).lower()
        if MEASUREMENT_MODE == "avg_only" and stat != "avg":
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        valid_fraction = float(numeric.notna().mean())
        if valid_fraction < MIN_REFERENCE_VALID_FRACTION:
            continue

        selected.append(col)

    if not selected:
        raise ValueError(
            "No measurement columns were selected. Check column names and "
            "MEASUREMENT_MODE."
        )

    return selected


def get_base_signal_name(column: str) -> str:
    return STAT_SUFFIX_PATTERN.sub("", column)


def group_measurements_by_base_signal(
    measurement_columns: list[str],
) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}

    for column in measurement_columns:
        match = STAT_SUFFIX_PATTERN.search(column)
        if match is None:
            continue

        statistic = match.group(1).lower()
        base_signal = get_base_signal_name(column)
        groups.setdefault(base_signal, {})
        groups[base_signal][statistic] = column

    return groups


def select_reference_mask(
    df: pd.DataFrame,
    timestamp_col: str,
    metadata: EventMetadata,
) -> pd.Series:
    preferred_start = metadata.event_start - pd.Timedelta(hours=REFERENCE_HOURS)

    preferred = (
        (df[timestamp_col] >= preferred_start)
        & (df[timestamp_col] < metadata.event_start)
    )

    if int(preferred.sum()) >= MIN_REFERENCE_POINTS:
        return preferred

    # Fallback: use every available row before event_start.
    fallback = df[timestamp_col] < metadata.event_start
    if int(fallback.sum()) >= MIN_REFERENCE_POINTS:
        warnings.warn(
            f"Event {metadata.event_id}: fewer than {MIN_REFERENCE_POINTS} rows "
            f"in the preferred {REFERENCE_HOURS}-hour reference period. "
            "Using all available pre-event rows.",
            RuntimeWarning,
        )
        return fallback

    raise ValueError(
        f"Event {metadata.event_id}: only {int(fallback.sum())} pre-event rows "
        f"are available; at least {MIN_REFERENCE_POINTS} are required. "
        "The event is skipped rather than using contaminated event data."
    )


def calculate_measurement_and_base_z(
    df: pd.DataFrame,
    measurement_columns: list[str],
    reference_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    numeric = df[measurement_columns].apply(pd.to_numeric, errors="coerce")
    reference = numeric.loc[reference_mask]

    valid_fraction = reference.notna().mean()
    valid_columns = valid_fraction[
        valid_fraction >= MIN_REFERENCE_VALID_FRACTION
    ].index.tolist()

    reference = reference[valid_columns]
    numeric = numeric[valid_columns]

    centre, scale = robust_location_scale(reference)
    usable_columns = scale.dropna().index.tolist()

    if not usable_columns:
        raise ValueError("No measurement columns have a usable reference scale.")

    z_df = (numeric[usable_columns] - centre[usable_columns]).abs()
    z_df = z_df.div(scale[usable_columns], axis=1)
    z_df = z_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z_df = z_df.clip(upper=ROBUST_Z_CLIP)

    groups = group_measurements_by_base_signal(usable_columns)

    # Build all base-signal columns first, then create the DataFrame once.
    # This avoids repeatedly inserting hundreds of columns into a DataFrame,
    # which causes pandas PerformanceWarning: DataFrame is highly fragmented.
    base_z_columns: dict[str, pd.Series] = {}
    clean_groups: dict[str, dict[str, str]] = {}

    for base_signal, stat_map in groups.items():
        cols = [
            col
            for col in stat_map.values()
            if col in z_df.columns
        ]

        if not cols:
            continue

        # One base signal may have avg/max/min/std columns.
        # Its row-level anomaly score is the maximum robust z-score
        # across the available statistics.
        base_z_columns[base_signal] = z_df[cols].max(axis=1)

        clean_groups[base_signal] = {
            stat: col
            for stat, col in stat_map.items()
            if col in z_df.columns
        }

    if not base_z_columns:
        raise ValueError(
            "No base signals could be created from measurement columns."
        )

    # Create the whole DataFrame in one operation. This is the key fix for
    # the fragmentation warning.
    base_z = pd.DataFrame(
        base_z_columns,
        index=df.index,
    ).copy()

    return z_df, base_z, clean_groups


# =============================================================================
# 6. POWER SIGNAL IDENTIFICATION AND POWER EVIDENCE
# =============================================================================

def identify_power_signal_groups(
    signal_groups: dict[str, dict[str, str]],
    farm_id: str,
    manual_power_signals: Optional[list[str]] = None,
) -> dict[str, dict[str, str]]:
    requested: Optional[list[str]] = None

    if manual_power_signals:
        requested = manual_power_signals
    elif farm_id in POWER_SIGNAL_OVERRIDES:
        requested = POWER_SIGNAL_OVERRIDES[farm_id]

    if requested:
        selected = {
            base: signal_groups[base]
            for base in requested
            if base in signal_groups and "avg" in signal_groups[base]
        }
        if selected:
            return selected
        warnings.warn(
            f"No requested power signals were found for farm {farm_id}. "
            "Falling back to automatic identification.",
            RuntimeWarning,
        )

    include_patterns = [
        r"(^|_)power($|_)",
        r"active_power",
        r"generator_power",
        r"electrical_power",
        r"(^|_)pwr($|_)",
    ]
    exclude_patterns = [
        r"reactive",
        r"power_factor",
        r"setpoint",
        r"limit",
        r"rated",
    ]

    selected: dict[str, dict[str, str]] = {}
    for base_signal, stat_map in signal_groups.items():
        name = base_signal.lower()

        included = any(re.search(pattern, name) for pattern in include_patterns)
        excluded = any(re.search(pattern, name) for pattern in exclude_patterns)

        if included and not excluded and "avg" in stat_map:
            selected[base_signal] = stat_map

    return selected


def calculate_power_evidence(
    df: pd.DataFrame,
    power_groups: dict[str, dict[str, str]],
    reference_mask: pd.Series,
) -> pd.DataFrame:
    cfg = POWER_EVIDENCE_CONFIG
    evidence = pd.DataFrame(index=df.index)

    if not power_groups:
        evidence["power_evidence_available"] = False
        evidence["power_signal_count"] = 0
        evidence["required_power_signal_count"] = 0
        evidence["power_dip_signal_count"] = 0
        evidence["power_recovery_signal_count"] = 0
        evidence["power_variability_signal_count"] = 0
        evidence["power_dip_consensus"] = False
        evidence["power_recovery_consensus"] = False
        evidence["power_within_window_consensus"] = False
        evidence["power_evidence_confirmed"] = False
        evidence["power_dip_score_max"] = 0.0
        evidence["power_recovery_score_max"] = 0.0
        evidence["power_variability_score_max"] = 0.0
        return evidence

    dip_flags: list[pd.Series] = []
    recovery_flags: list[pd.Series] = []
    variability_flags: list[pd.Series] = []

    dip_scores: list[pd.Series] = []
    recovery_scores: list[pd.Series] = []
    variability_scores: list[pd.Series] = []

    for base_signal, columns in power_groups.items():
        avg_col = columns.get("avg")
        if avg_col is None:
            continue

        avg = pd.to_numeric(df[avg_col], errors="coerce")
        reference_avg = avg.loc[reference_mask].dropna()
        if len(reference_avg) < MIN_REFERENCE_POINTS // 2:
            continue

        previous_normal = (
            avg.rolling(
                window=int(cfg["local_reference_points"]),
                min_periods=max(2, int(cfg["local_reference_points"]) // 2),
            )
            .median()
            .shift(1)
        )
        next_avg = avg.shift(-1)

        denominator = previous_normal.abs().clip(lower=EPSILON)
        drop_ratio = (previous_normal - avg) / denominator
        recovery_ratio = next_avg / denominator

        change_scale = robust_scale_1d(reference_avg.diff())
        if not np.isfinite(change_scale) or change_scale <= EPSILON:
            continue

        dip_z = (previous_normal - avg) / change_scale
        recovery_z = (next_avg - avg) / change_scale

        q_active = float(
            reference_avg.quantile(float(cfg["active_power_reference_quantile"]))
        )
        q90 = float(reference_avg.quantile(0.90))

        # For nonnegative power signals, this avoids classifying a near-zero
        # preceding state as active generation.
        active_threshold = max(q_active, 0.10 * q90)
        was_generating = previous_normal >= active_threshold

        dip_flag = was_generating & (
            (drop_ratio >= float(cfg["minimum_power_drop_ratio"]))
            | (dip_z >= float(cfg["power_dip_z_threshold"]))
        )

        recovery_flag = dip_flag & (
            (recovery_ratio >= float(cfg["minimum_recovery_ratio"]))
            | (recovery_z >= float(cfg["recovery_z_threshold"]))
        )

        variability_flag = pd.Series(False, index=df.index)
        variability_z = pd.Series(0.0, index=df.index, dtype=float)

        min_col = columns.get("min")
        max_col = columns.get("max")
        std_col = columns.get("std")

        if min_col is not None and max_col is not None:
            minimum = pd.to_numeric(df[min_col], errors="coerce")
            maximum = pd.to_numeric(df[max_col], errors="coerce")
            power_range = maximum - minimum

            reference_range = power_range.loc[reference_mask].dropna()
            range_scale = robust_scale_1d(reference_range)

            if np.isfinite(range_scale) and range_scale > EPSILON:
                range_median = float(reference_range.median())
                range_z = (power_range - range_median) / range_scale
                min_ratio = minimum / denominator

                range_flag = (
                    was_generating
                    & (min_ratio <= float(cfg["maximum_min_power_ratio"]))
                    & (range_z >= float(cfg["variability_z_threshold"]))
                )

                variability_flag = variability_flag | range_flag
                variability_z = pd.concat(
                    [variability_z, range_z.fillna(0.0)], axis=1
                ).max(axis=1)

        if std_col is not None:
            std = pd.to_numeric(df[std_col], errors="coerce")
            reference_std = std.loc[reference_mask].dropna()
            std_scale = robust_scale_1d(reference_std)

            if np.isfinite(std_scale) and std_scale > EPSILON:
                std_median = float(reference_std.median())
                std_z = (std - std_median) / std_scale

                std_flag = dip_flag & (
                    std_z >= float(cfg["variability_z_threshold"])
                )
                variability_flag = variability_flag | std_flag
                variability_z = pd.concat(
                    [variability_z, std_z.fillna(0.0)], axis=1
                ).max(axis=1)

        dip_flags.append(dip_flag.rename(base_signal))
        recovery_flags.append(recovery_flag.rename(base_signal))
        variability_flags.append(variability_flag.rename(base_signal))

        dip_scores.append(dip_z.fillna(0.0).rename(base_signal))
        recovery_scores.append(recovery_z.fillna(0.0).rename(base_signal))
        variability_scores.append(variability_z.fillna(0.0).rename(base_signal))

    if not dip_flags:
        return calculate_power_evidence(df, {}, reference_mask)

    dip_flag_df = pd.concat(dip_flags, axis=1)
    recovery_flag_df = pd.concat(recovery_flags, axis=1)
    variability_flag_df = pd.concat(variability_flags, axis=1)

    n_power_signals = dip_flag_df.shape[1]
    required_count = min(
        n_power_signals,
        max(
            1 if n_power_signals == 1 else int(cfg["minimum_power_signal_count"]),
            math.ceil(
                n_power_signals
                * float(cfg["power_signal_consensus_fraction"])
            ),
        ),
    )

    dip_count = dip_flag_df.sum(axis=1).astype(int)
    recovery_count = recovery_flag_df.sum(axis=1).astype(int)
    variability_count = variability_flag_df.sum(axis=1).astype(int)

    dip_consensus = dip_count >= required_count
    recovery_consensus = recovery_count >= required_count
    variability_consensus = variability_count >= required_count

    evidence["power_evidence_available"] = True
    evidence["power_signal_count"] = n_power_signals
    evidence["required_power_signal_count"] = required_count

    evidence["power_dip_signal_count"] = dip_count
    evidence["power_recovery_signal_count"] = recovery_count
    evidence["power_variability_signal_count"] = variability_count

    evidence["power_dip_consensus"] = dip_consensus
    evidence["power_recovery_consensus"] = recovery_consensus
    evidence["power_within_window_consensus"] = variability_consensus

    evidence["power_evidence_confirmed"] = (
        dip_consensus & (recovery_consensus | variability_consensus)
    )

    evidence["power_dip_score_max"] = (
        pd.concat(dip_scores, axis=1).max(axis=1).clip(lower=0.0)
    )
    evidence["power_recovery_score_max"] = (
        pd.concat(recovery_scores, axis=1).max(axis=1).clip(lower=0.0)
    )
    evidence["power_variability_score_max"] = (
        pd.concat(variability_scores, axis=1).max(axis=1).clip(lower=0.0)
    )

    return evidence


# =============================================================================
# 7. THREE DETECTORS
# =============================================================================

def calculate_top_k_mean(base_z: pd.DataFrame, k: int = 20) -> pd.Series:
    values = base_z.to_numpy(dtype=float)
    if values.shape[1] == 0:
        return pd.Series(0.0, index=base_z.index)

    k = min(k, values.shape[1])
    partitioned = np.partition(values, values.shape[1] - k, axis=1)[:, -k:]
    return pd.Series(np.nanmean(partitioned, axis=1), index=base_z.index)


def calculate_adaptive_fraction_threshold(
    fraction: pd.Series,
    reference_mask: pd.Series,
    quantile: float,
    floor: float,
    rolling_points: int = 1,
) -> tuple[float, pd.Series]:
    """
    Calculate an event-specific threshold from the pre-event reference period.

    The detector parameters are unchanged across events, but the final
    abnormal-fraction threshold adapts to each event's own reference data.
    """
    reference_fraction = fraction.loc[reference_mask].dropna()

    if reference_fraction.empty:
        raise ValueError(
            "Cannot calculate adaptive threshold because the reference "
            "abnormal-fraction series is empty."
        )

    if rolling_points > 1:
        reference_for_threshold = reference_fraction.rolling(
            window=rolling_points,
            min_periods=1,
            center=True,
        ).mean()
    else:
        reference_for_threshold = reference_fraction

    threshold = max(
        float(floor),
        float(reference_for_threshold.quantile(float(quantile))),
    )

    return threshold, reference_for_threshold


def run_detectors(
    df: pd.DataFrame,
    base_z: pd.DataFrame,
    power_evidence: pd.DataFrame,
    reference_mask: pd.Series,
    event_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run the three detectors and return row-level scores plus calibration data.

    1. short_standstill
       Adaptive multisensor threshold + mandatory power evidence.

    2. intermittent_cluster
       Rolling density of confirmed short-standstill rows.

    3. persistent_system_state
       Adaptive persistent multisensor threshold.
    """
    scores = pd.DataFrame(index=df.index)
    calibration: dict[str, Any] = {}

    scores["top20_base_local_change_mean"] = calculate_top_k_mean(base_z, k=20)

    # -----------------------------------------------------------------
    # Detector 1: confirmed short standstill
    # -----------------------------------------------------------------
    short_cfg = DETECTOR_CONFIGS["short_standstill"]
    short_abnormal = base_z >= float(short_cfg["z_threshold"])
    short_fraction = short_abnormal.mean(axis=1)

    short_threshold, short_reference_distribution = (
        calculate_adaptive_fraction_threshold(
            fraction=short_fraction,
            reference_mask=reference_mask,
            quantile=float(short_cfg["fraction_reference_quantile"]),
            floor=float(short_cfg["fraction_floor"]),
            rolling_points=int(short_cfg["rolling_points"]),
        )
    )

    short_smoothed = short_fraction.copy()
    short_raw_flag = short_smoothed >= short_threshold

    short_final_flag = (
        short_raw_flag
        & power_evidence["power_evidence_confirmed"].fillna(False)
        & event_mask
    )

    scores["short_standstill_abnormal_fraction"] = short_fraction
    scores["short_standstill_smoothed_fraction"] = short_smoothed
    scores["short_standstill_threshold"] = short_threshold
    scores["short_standstill_raw_flag"] = short_raw_flag
    scores["short_standstill_flag"] = short_final_flag

    calibration["short_standstill"] = {
        "z_threshold": float(short_cfg["z_threshold"]),
        "adaptive_fraction_threshold": short_threshold,
        "reference_quantile": float(short_cfg["fraction_reference_quantile"]),
        "fraction_floor": float(short_cfg["fraction_floor"]),
        "reference_raw_trigger_rate": float(
            short_raw_flag.loc[reference_mask].mean()
        ),
        "reference_power_confirmed_rate": float(
            power_evidence.loc[
                reference_mask, "power_evidence_confirmed"
            ].mean()
        ),
        "reference_combined_trigger_rate": float(
            (
                short_raw_flag
                & power_evidence["power_evidence_confirmed"].fillna(False)
            ).loc[reference_mask].mean()
        ),
        "reference_fraction_median": float(
            short_reference_distribution.median()
        ),
        "reference_fraction_p95": float(
            short_reference_distribution.quantile(0.95)
        ),
        "reference_fraction_p985": float(
            short_reference_distribution.quantile(0.985)
        ),
    }

    # -----------------------------------------------------------------
    # Detector 2: intermittent cluster
    # -----------------------------------------------------------------
    intermittent_cfg = DETECTOR_CONFIGS["intermittent_cluster"]
    intermittent_window = int(intermittent_cfg["rolling_points"])
    minimum_short_flags = int(intermittent_cfg["minimum_short_flags"])

    confirmed_short_numeric = short_final_flag.astype(int)

    intermittent_count = confirmed_short_numeric.rolling(
        window=intermittent_window,
        min_periods=1,
        center=True,
    ).sum()

    intermittent_density = confirmed_short_numeric.rolling(
        window=intermittent_window,
        min_periods=1,
        center=True,
    ).mean()

    intermittent_raw_flag = intermittent_count >= minimum_short_flags
    intermittent_final_flag = intermittent_raw_flag & event_mask

    scores["intermittent_cluster_abnormal_fraction"] = (
        confirmed_short_numeric.astype(float)
    )
    scores["intermittent_cluster_smoothed_fraction"] = intermittent_density
    scores["intermittent_cluster_short_flag_count"] = intermittent_count
    scores["intermittent_cluster_threshold"] = (
        minimum_short_flags / intermittent_window
    )
    scores["intermittent_cluster_minimum_short_flags"] = minimum_short_flags
    scores["intermittent_cluster_raw_flag"] = intermittent_raw_flag
    scores["intermittent_cluster_flag"] = intermittent_final_flag

    calibration["intermittent_cluster"] = {
        "source": intermittent_cfg["source"],
        "rolling_points": intermittent_window,
        "minimum_short_flags": minimum_short_flags,
        "density_threshold": minimum_short_flags / intermittent_window,
        "event_confirmed_short_rows": int(short_final_flag.loc[event_mask].sum()),
        "event_cluster_flag_rate": float(
            intermittent_final_flag.loc[event_mask].mean()
        ),
    }

    # -----------------------------------------------------------------
    # Detector 3: persistent system state
    # -----------------------------------------------------------------
    persistent_cfg = DETECTOR_CONFIGS["persistent_system_state"]
    persistent_abnormal = base_z >= float(persistent_cfg["z_threshold"])
    persistent_fraction = persistent_abnormal.mean(axis=1)

    persistent_rolling_points = int(persistent_cfg["rolling_points"])
    persistent_smoothed = persistent_fraction.rolling(
        window=persistent_rolling_points,
        min_periods=1,
        center=True,
    ).mean()

    persistent_threshold, persistent_reference_distribution = (
        calculate_adaptive_fraction_threshold(
            fraction=persistent_fraction,
            reference_mask=reference_mask,
            quantile=float(
                persistent_cfg["fraction_reference_quantile"]
            ),
            floor=float(persistent_cfg["fraction_floor"]),
            rolling_points=persistent_rolling_points,
        )
    )

    persistent_raw_flag = persistent_smoothed >= persistent_threshold
    persistent_final_flag = persistent_raw_flag & event_mask

    scores["persistent_system_state_abnormal_fraction"] = persistent_fraction
    scores["persistent_system_state_smoothed_fraction"] = persistent_smoothed
    scores["persistent_system_state_threshold"] = persistent_threshold
    scores["persistent_system_state_raw_flag"] = persistent_raw_flag
    scores["persistent_system_state_flag"] = persistent_final_flag

    calibration["persistent_system_state"] = {
        "z_threshold": float(persistent_cfg["z_threshold"]),
        "adaptive_fraction_threshold": persistent_threshold,
        "reference_quantile": float(
            persistent_cfg["fraction_reference_quantile"]
        ),
        "fraction_floor": float(persistent_cfg["fraction_floor"]),
        "rolling_points": persistent_rolling_points,
        "reference_trigger_rate": float(
            (
                persistent_reference_distribution
                >= persistent_threshold
            ).mean()
        ),
        "reference_fraction_median": float(
            persistent_reference_distribution.median()
        ),
        "reference_fraction_p95": float(
            persistent_reference_distribution.quantile(0.95)
        ),
        "reference_fraction_p995": float(
            persistent_reference_distribution.quantile(0.995)
        ),
    }

    # Composite candidate score is a relative ranking tool, not a probability.
    components = [
        percentile_rank(scores["top20_base_local_change_mean"]),
        percentile_rank(scores["short_standstill_smoothed_fraction"]),
        percentile_rank(scores["intermittent_cluster_smoothed_fraction"]),
        percentile_rank(scores["persistent_system_state_smoothed_fraction"]),
        percentile_rank(power_evidence["power_dip_score_max"]),
        percentile_rank(power_evidence["power_recovery_score_max"]),
        percentile_rank(power_evidence["power_variability_score_max"]),
    ]
    scores["exploratory_composite_rank"] = (
        pd.concat(components, axis=1).mean(axis=1)
    )
    scores.loc[~event_mask, "exploratory_composite_rank"] = np.nan

    return scores, calibration


# =============================================================================
# 8. SEGMENT EXTRACTION
# =============================================================================

def bridge_short_false_gaps(flag: pd.Series, max_gap_points: int) -> pd.Series:
    values = flag.fillna(False).astype(bool).to_numpy().copy()
    if max_gap_points <= 0 or len(values) == 0:
        return pd.Series(values, index=flag.index)

    i = 0
    while i < len(values):
        if values[i]:
            i += 1
            continue

        start = i
        while i < len(values) and not values[i]:
            i += 1
        end = i - 1
        length = end - start + 1

        bounded_left = start > 0 and values[start - 1]
        bounded_right = i < len(values) and values[i]

        if bounded_left and bounded_right and length <= max_gap_points:
            values[start:i] = True

    return pd.Series(values, index=flag.index)


def contiguous_true_ranges(flag: pd.Series) -> list[tuple[int, int]]:
    values = flag.fillna(False).astype(bool).to_numpy()
    ranges: list[tuple[int, int]] = []

    start: Optional[int] = None
    for position, value in enumerate(values):
        if value and start is None:
            start = position
        elif not value and start is not None:
            ranges.append((start, position - 1))
            start = None

    if start is not None:
        ranges.append((start, len(values) - 1))

    return ranges


def confidence_for_segment(
    detector: str,
    segment_rows: pd.DataFrame,
) -> str:
    if detector == "short_standstill":
        all_three = bool(
            segment_rows["power_dip_consensus"].any()
            and segment_rows["power_recovery_consensus"].any()
            and segment_rows["power_within_window_consensus"].any()
        )
        return "Very High" if all_three else "High"

    if detector == "intermittent_cluster":
        maximum_count = float(
            segment_rows["intermittent_cluster_short_flag_count"].max()
        )
        minimum_count = float(
            segment_rows["intermittent_cluster_minimum_short_flags"].iloc[0]
        )

        if maximum_count >= 2.0 * minimum_count:
            return "High"
        if maximum_count >= 1.5 * minimum_count:
            return "Medium-High"
        return "Medium"

    threshold_column = f"{detector}_threshold"
    threshold = float(segment_rows[threshold_column].iloc[0])
    max_fraction = float(
        segment_rows[f"{detector}_smoothed_fraction"].max()
    )

    if max_fraction >= 1.5 * threshold:
        return "High"
    if max_fraction >= 1.2 * threshold:
        return "Medium-High"
    return "Medium"


def extract_segments_for_detector(
    df: pd.DataFrame,
    timestamp_col: str,
    row_id_col: Optional[str],
    metadata: EventMetadata,
    sampling_minutes: float,
    detector: str,
    row_output: pd.DataFrame,
) -> list[Segment]:
    cfg = DETECTOR_CONFIGS[detector]

    bridged = bridge_short_false_gaps(
        row_output[f"{detector}_flag"],
        max_gap_points=int(cfg["max_gap_points"]),
    )

    ranges = contiguous_true_ranges(bridged)
    ranges = [
        (start, end)
        for start, end in ranges
        if (end - start + 1) >= int(cfg["min_segment_points"])
    ]

    # A short standstill should be reported as one 10-minute candidate row.
    # If several adjacent rows satisfy the short detector, retain only the
    # strongest row in that contiguous run. The broader activity is handled
    # separately by intermittent_cluster.
    if detector == "short_standstill":
        single_row_ranges: list[tuple[int, int]] = []

        for start, end in ranges:
            candidate_positions = row_output.index[start:end + 1]
            strongest_index = row_output.loc[
                candidate_positions, "detector_score"
            ].idxmax()

            strongest_position = int(
                row_output.index.get_loc(strongest_index)
            )
            single_row_ranges.append(
                (strongest_position, strongest_position)
            )

        ranges = single_row_ranges

    segments: list[Segment] = []

    for number, (start_pos, end_pos) in enumerate(ranges, start=1):
        idx = row_output.index[start_pos:end_pos + 1]
        segment_rows = row_output.loc[idx]

        strongest_idx = segment_rows["detector_score"].idxmax()
        start_time = pd.Timestamp(df.loc[idx[0], timestamp_col])
        last_row_time = pd.Timestamp(df.loc[idx[-1], timestamp_col])
        segment_end = last_row_time + pd.Timedelta(minutes=sampling_minutes)

        start_id = safe_int(df.loc[idx[0], row_id_col]) if row_id_col else None
        end_id = safe_int(df.loc[idx[-1], row_id_col]) if row_id_col else None
        strongest_id = (
            safe_int(df.loc[strongest_idx, row_id_col]) if row_id_col else None
        )

        segment_id = (
            f"{metadata.farm_id}_event{metadata.event_id}_"
            f"{detector}_{number:02d}"
        )

        segments.append(
            Segment(
                farm_id=metadata.farm_id,
                event_id=metadata.event_id,
                detector=detector,
                event_description=metadata.event_description,
                segment_id=segment_id,
                segment_start=start_time,
                segment_end=segment_end,
                start_id=start_id,
                end_id=end_id,
                n_points=len(idx),
                duration_minutes=len(idx) * sampling_minutes,
                strongest_candidate_time=pd.Timestamp(
                    df.loc[strongest_idx, timestamp_col]
                ),
                strongest_candidate_id=strongest_id,
                detector_score_max=float(segment_rows["detector_score"].max()),
                abnormal_fraction_max=float(
                    segment_rows[f"{detector}_smoothed_fraction"].max()
                ),
                power_evidence_available=bool(
                    segment_rows["power_evidence_available"].any()
                ),
                power_evidence_confirmed=bool(
                    segment_rows["power_evidence_confirmed"].any()
                ),
                power_signal_count=int(
                    segment_rows["power_signal_count"].max()
                ),
                power_dip_signal_count_max=int(
                    segment_rows["power_dip_signal_count"].max()
                ),
                power_recovery_signal_count_max=int(
                    segment_rows["power_recovery_signal_count"].max()
                ),
                power_variability_signal_count_max=int(
                    segment_rows["power_variability_signal_count"].max()
                ),
                power_dip_score_max=float(
                    segment_rows["power_dip_score_max"].max()
                ),
                power_recovery_score_max=float(
                    segment_rows["power_recovery_score_max"].max()
                ),
                power_variability_score_max=float(
                    segment_rows["power_variability_score_max"].max()
                ),
                confidence=confidence_for_segment(detector, segment_rows),
            )
        )

    return segments


def attach_parent_cluster_ids(segments: list[Segment]) -> list[Segment]:
    intermittent = [
        segment for segment in segments
        if segment.detector == "intermittent_cluster"
    ]
    persistent = [
        segment for segment in segments
        if segment.detector == "persistent_system_state"
    ]

    for segment in segments:
        if segment.detector != "short_standstill":
            continue

        parent = next(
            (
                cluster for cluster in intermittent
                if cluster.segment_start <= segment.segment_start
                and cluster.segment_end >= segment.segment_end
            ),
            None,
        )

        if parent is None:
            parent = next(
                (
                    cluster for cluster in persistent
                    if cluster.segment_start <= segment.segment_start
                    and cluster.segment_end >= segment.segment_end
                ),
                None,
            )

        segment.parent_cluster_id = parent.segment_id if parent else None

    return segments


# =============================================================================
# 9. CANDIDATES AND CONTRIBUTORS
# =============================================================================

def create_row_output(
    df: pd.DataFrame,
    timestamp_col: str,
    row_id_col: Optional[str],
    detector_scores: pd.DataFrame,
    power_evidence: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)
    output["time_stamp"] = df[timestamp_col]
    output["row_id"] = (
        df[row_id_col].map(safe_int) if row_id_col else pd.Series(index=df.index)
    )

    output = pd.concat(
        [output, detector_scores, power_evidence],
        axis=1,
    )

    # Main score for choosing the strongest row inside a detected segment.
    output["detector_score"] = (
        percentile_rank(output["top20_base_local_change_mean"])
        + percentile_rank(output["exploratory_composite_rank"].fillna(0.0))
    ) / 2.0

    return output


def select_top_candidates(
    row_output: pd.DataFrame,
    event_mask: pd.Series,
) -> pd.DataFrame:
    candidates = row_output.loc[event_mask].copy()
    candidates = candidates.sort_values(
        [
            "exploratory_composite_rank",
            "power_evidence_confirmed",
            "top20_base_local_change_mean",
        ],
        ascending=[False, False, False],
    ).head(TOP_TRANSITION_CANDIDATES)

    candidates.insert(0, "candidate_rank", range(1, len(candidates) + 1))
    return candidates.reset_index(names="source_index")


def create_candidate_contributors(
    top_candidates: pd.DataFrame,
    z_df: pd.DataFrame,
    base_z: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, candidate in top_candidates.iterrows():
        source_index = int(candidate["source_index"])

        measurement_scores = z_df.loc[source_index].sort_values(ascending=False)
        base_scores = base_z.loc[source_index].sort_values(ascending=False)

        for contributor_rank, (measurement, score) in enumerate(
            measurement_scores.head(TOP_CONTRIBUTORS_PER_CANDIDATE).items(),
            start=1,
        ):
            rows.append(
                {
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "candidate_time": candidate["time_stamp"],
                    "row_id": candidate["row_id"],
                    "contributor_level": "measurement",
                    "contributor_rank": contributor_rank,
                    "contributor": measurement,
                    "base_signal": get_base_signal_name(measurement),
                    "robust_z": float(score),
                }
            )

        for contributor_rank, (base_signal, score) in enumerate(
            base_scores.head(TOP_CONTRIBUTORS_PER_CANDIDATE).items(),
            start=1,
        ):
            rows.append(
                {
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "candidate_time": candidate["time_stamp"],
                    "row_id": candidate["row_id"],
                    "contributor_level": "base_signal",
                    "contributor_rank": contributor_rank,
                    "contributor": base_signal,
                    "base_signal": base_signal,
                    "robust_z": float(score),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# 10. PLOTS
# =============================================================================

def plot_detector_timelines(
    row_output: pd.DataFrame,
    metadata: EventMetadata,
    output_dir: Path,
) -> None:
    for detector_name in DETECTOR_CONFIGS:
        fig, ax = plt.subplots(figsize=(16, 5))

        if detector_name == "intermittent_cluster":
            y_column = "intermittent_cluster_short_flag_count"
            threshold = float(
                row_output[
                    "intermittent_cluster_minimum_short_flags"
                ].iloc[0]
            )
            y_label = (
                "Confirmed short-standstill rows in rolling window"
            )
            line_label = (
                "rolling count of confirmed short-standstill rows"
            )
        else:
            y_column = f"{detector_name}_smoothed_fraction"
            threshold = float(
                row_output[f"{detector_name}_threshold"].iloc[0]
            )
            y_label = "Base-signal abnormal fraction"
            line_label = f"{detector_name} detector score"

        ax.plot(
            row_output["time_stamp"],
            row_output[y_column],
            label=line_label,
        )
        ax.axhline(
            threshold,
            linestyle="--",
            label=f"adaptive threshold = {threshold:.4g}",
        )
        ax.axvline(
            metadata.event_start,
            linestyle="--",
            label="metadata start",
        )
        ax.axvline(
            metadata.event_end,
            linestyle="--",
            label="metadata end",
        )
        ax.set_title(
            f"Farm {metadata.farm_id} Event "
            f"{metadata.event_id}: {detector_name}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel(y_label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{detector_name}_timeline.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_power_evidence(
    row_output: pd.DataFrame,
    metadata: EventMetadata,
    output_dir: Path,
) -> None:
    for column, title in [
        ("power_dip_score_max", "Maximum power-dip evidence"),
        ("power_recovery_score_max", "Maximum next-row recovery evidence"),
        ("power_variability_score_max", "Maximum within-window power variability"),
    ]:
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(row_output["time_stamp"], row_output[column], label=column)
        ax.axvline(metadata.event_start, linestyle="--", label="metadata start")
        ax.axvline(metadata.event_end, linestyle="--", label="metadata end")

        confirmed = row_output["power_evidence_confirmed"].fillna(False)
        ax.scatter(
            row_output.loc[confirmed, "time_stamp"],
            row_output.loc[confirmed, column],
            s=25,
            label="confirmed power evidence",
        )

        ax.set_title(
            f"Farm {metadata.farm_id} Event {metadata.event_id}: {title}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Robust evidence score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{column}_timeline.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_top_base_contributors(
    base_z: pd.DataFrame,
    event_mask: pd.Series,
    metadata: EventMetadata,
    output_dir: Path,
) -> None:
    maxima = base_z.loc[event_mask].max(axis=0).sort_values(ascending=False)
    top = maxima.head(TOP_CONTRIBUTOR_BAR_COUNT).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top.index, top.values)
    ax.set_title(
        f"Farm {metadata.farm_id} Event {metadata.event_id}: "
        "Top base signals by maximum robust local change"
    )
    ax.set_xlabel("Maximum robust z-score during metadata interval")
    ax.set_ylabel("Base signal")
    fig.tight_layout()
    fig.savefig(
        output_dir / "top_base_signal_contributors.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_candidate_signal_contexts(
    df: pd.DataFrame,
    timestamp_col: str,
    top_candidates: pd.DataFrame,
    z_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    candidate_plot_dir = output_dir / "candidate_signal_contexts"
    candidate_plot_dir.mkdir(parents=True, exist_ok=True)

    for _, candidate in top_candidates.iterrows():
        source_index = int(candidate["source_index"])
        candidate_time = pd.Timestamp(candidate["time_stamp"])

        top_measurements = (
            z_df.loc[source_index]
            .sort_values(ascending=False)
            .head(PLOT_TOP_CONTRIBUTOR_SIGNALS)
            .index
            .tolist()
        )

        context_mask = (
            (df[timestamp_col] >= candidate_time - pd.Timedelta(hours=2))
            & (df[timestamp_col] <= candidate_time + pd.Timedelta(hours=2))
        )
        context = df.loc[context_mask]

        if context.empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 7))
        for measurement in top_measurements:
            values = pd.to_numeric(context[measurement], errors="coerce")
            reference = values.dropna()
            if reference.empty:
                continue

            # Standardise only for visual comparison of different units.
            scale = robust_scale_1d(reference)
            if not np.isfinite(scale) or scale <= EPSILON:
                continue
            standardised = (values - reference.median()) / scale
            ax.plot(context[timestamp_col], standardised, alpha=0.65)

        ax.axvline(candidate_time, linestyle="--", label="candidate time")
        ax.set_title(
            f"Candidate {int(candidate['candidate_rank'])}: "
            f"{candidate_time:%Y-%m-%d %H:%M}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Locally standardised signal value")
        ax.legend()
        fig.tight_layout()

        filename = (
            f"candidate_{int(candidate['candidate_rank']):02d}_"
            f"{candidate_time:%Y%m%d_%H%M}.png"
        )
        fig.savefig(candidate_plot_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 11. EVENT ANALYSIS
# =============================================================================

def analyse_event(
    metadata: EventMetadata,
    event_file: Path,
    output_root: Path,
    manual_power_signals: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(
        f"[INFO] Analysing farm {metadata.farm_id}, "
        f"event {metadata.event_id}: {event_file}"
    )

    event_output_dir = (
        output_root
        / f"farm_{metadata.farm_id}"
        / f"event_{metadata.event_id}"
    )
    event_output_dir.mkdir(parents=True, exist_ok=True)

    df, timestamp_col, row_id_col, sampling_minutes = prepare_event_dataframe(
        event_file, metadata
    )

    reference_mask = select_reference_mask(df, timestamp_col, metadata)
    event_mask = (
        (df[timestamp_col] >= metadata.event_start)
        & (df[timestamp_col] <= metadata.event_end)
    )

    if int(event_mask.sum()) == 0:
        raise ValueError(
            f"No rows fall inside metadata interval for event {metadata.event_id}."
        )

    measurement_columns = select_measurement_columns(df)

    z_df, base_z, base_groups = calculate_measurement_and_base_z(
        df,
        measurement_columns,
        reference_mask,
    )

    power_groups = identify_power_signal_groups(
        base_groups,
        farm_id=metadata.farm_id,
        manual_power_signals=manual_power_signals,
    )

    power_evidence = calculate_power_evidence(
        df,
        power_groups,
        reference_mask,
    )

    detector_scores, detector_calibration = run_detectors(
        df,
        base_z,
        power_evidence,
        reference_mask,
        event_mask,
    )

    row_output = create_row_output(
        df,
        timestamp_col,
        row_id_col,
        detector_scores,
        power_evidence,
    )

    segments: list[Segment] = []
    for detector_name in DETECTOR_CONFIGS:
        segments.extend(
            extract_segments_for_detector(
                df=df,
                timestamp_col=timestamp_col,
                row_id_col=row_id_col,
                metadata=metadata,
                sampling_minutes=sampling_minutes,
                detector=detector_name,
                row_output=row_output,
            )
        )

    segments = attach_parent_cluster_ids(segments)
    segment_df = pd.DataFrame([asdict(segment) for segment in segments])

    top_candidates = select_top_candidates(row_output, event_mask)
    contributors = create_candidate_contributors(
        top_candidates,
        z_df,
        base_z,
    )

    row_output.to_csv(
        event_output_dir / "row_scores.csv",
        index=False,
    )
    segment_df.to_csv(
        event_output_dir / "detected_segments.csv",
        index=False,
    )
    top_candidates.to_csv(
        event_output_dir / "top_transition_candidates.csv",
        index=False,
    )
    contributors.to_csv(
        event_output_dir / "candidate_contributors.csv",
        index=False,
    )

    configuration = {
        "farm_id": metadata.farm_id,
        "event_id": metadata.event_id,
        "event_file": str(event_file),
        "analysis_start_used": str(metadata.event_start),
        "analysis_end_used": str(metadata.event_end),
        "sampling_minutes": sampling_minutes,
        "reference_rows": int(reference_mask.sum()),
        "event_rows": int(event_mask.sum()),
        "measurement_columns": len(z_df.columns),
        "base_signals": len(base_z.columns),
        "power_base_signals": list(power_groups.keys()),
        "measurement_mode": MEASUREMENT_MODE,
        "detector_configs": DETECTOR_CONFIGS,
        "power_evidence_config": POWER_EVIDENCE_CONFIG,
        "detector_calibration": detector_calibration,
    }

    with open(
        event_output_dir / "analysis_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, ensure_ascii=False, indent=2, default=str)

    if PLOT_OUTPUTS:
        plot_detector_timelines(row_output, metadata, event_output_dir)
        plot_power_evidence(row_output, metadata, event_output_dir)
        plot_top_base_contributors(
            base_z, event_mask, metadata, event_output_dir
        )
        plot_candidate_signal_contexts(
            df,
            timestamp_col,
            top_candidates,
            z_df,
            event_output_dir,
        )

    event_summary = {
        "farm_id": metadata.farm_id,
        "event_id": metadata.event_id,
        "event_label": metadata.event_label,
        "metadata_start": metadata.event_start,
        "metadata_end": metadata.event_end,
        "event_description": metadata.event_description,
        "sampling_minutes": sampling_minutes,
        "reference_rows": int(reference_mask.sum()),
        "event_rows": int(event_mask.sum()),
        "measurement_columns": len(z_df.columns),
        "base_signals": len(base_z.columns),
        "power_signal_count": len(power_groups),
        "power_signals": ",".join(power_groups.keys()),
        "short_standstill_segments": int(
            (segment_df.get("detector") == "short_standstill").sum()
        ) if not segment_df.empty else 0,
        "intermittent_cluster_segments": int(
            (segment_df.get("detector") == "intermittent_cluster").sum()
        ) if not segment_df.empty else 0,
        "persistent_system_state_segments": int(
            (segment_df.get("detector") == "persistent_system_state").sum()
        ) if not segment_df.empty else 0,
        "status": "success",
        "error": "",
    }

    return segment_df, event_summary


# =============================================================================
# 12. BATCH RUNNER
# =============================================================================

def parse_power_signals(text: Optional[str]) -> Optional[list[str]]:
    if text is None or not text.strip():
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_analysis_end(
    text: Optional[str],
    event_start: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """
    Parse and validate an optional command-line analysis-end override.

    The metadata CSV is never edited. A copied EventMetadata object is used
    only for the current run.
    """
    if text is None or not str(text).strip():
        return None

    parsed = pd.to_datetime(str(text).strip(), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(
            f"Could not parse --analysis-end value: {text!r}. "
            'Use a format such as "2025-09-18 23:50:00".'
        )

    parsed = pd.Timestamp(parsed)

    if parsed <= event_start:
        raise ValueError(
            f"--analysis-end ({parsed}) must be later than "
            f"event_start ({event_start})."
        )

    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run three SCADA anomaly detectors for one or all events."
    )
    parser.add_argument("--farm", required=True, help="Farm ID, e.g. A, B or C.")
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Semicolon-separated event metadata CSV.",
    )
    parser.add_argument(
        "--event-dir",
        required=True,
        type=Path,
        help="Directory containing one CSV per event.",
    )
    parser.add_argument(
        "--event-id",
        default="all",
        help='Specific event ID, e.g. "35", or "all".',
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        default=None,
        help="Explicit event CSV path. Only valid for a single event.",
    )
    parser.add_argument(
        "--analysis-end",
        default=None,
        help=(
            "Optional end-time override for a single event, for example "
            '"2025-09-18 23:50:00". The metadata file is not modified. '
            "This option cannot be used with --event-id all."
        ),
    )
    parser.add_argument(
        "--label-filter",
        default=None,
        help='Optional metadata label filter, e.g. "anomaly".',
    )
    parser.add_argument(
        "--power-signals",
        default=None,
        help=(
            "Optional comma-separated base signal names, e.g. "
            "power_2,power_5,power_6,power_17."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/three_detector"),
        help="Root output directory.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.event_file is not None and str(args.event_id).lower() == "all":
        parser.error("--event-file can only be used with a specific --event-id.")

    if args.analysis_end is not None and str(args.event_id).lower() == "all":
        parser.error(
            "--analysis-end can only be used with a specific --event-id."
        )

    manual_power_signals = parse_power_signals(args.power_signals)

    events = load_metadata(args.metadata, args.farm)

    if args.label_filter:
        events = [
            event for event in events
            if event.event_label.lower() == args.label_filter.lower()
        ]

    if str(args.event_id).lower() != "all":
        target_id = normalise_event_id(args.event_id)
        events = [event for event in events if event.event_id == target_id]

    if not events:
        print("[ERROR] No metadata rows matched the requested selection.")
        return 1

    # Apply an optional analysis-end override only to the selected event.
    # The metadata CSV remains unchanged.
    original_metadata_end_by_event: dict[str, pd.Timestamp] = {}

    if args.analysis_end is not None:
        if len(events) != 1:
            parser.error(
                "--analysis-end requires exactly one selected event."
            )

        selected_event = events[0]
        override_end = parse_analysis_end(
            args.analysis_end,
            selected_event.event_start,
        )

        original_metadata_end_by_event[selected_event.event_id] = (
            selected_event.event_end
        )

        events = [
            replace(
                selected_event,
                event_end=override_end,
                event_end_id=None,
            )
        ]

        print(
            "[INFO] Analysis-end override applied for "
            f"event {selected_event.event_id}: "
            f"{selected_event.event_end} -> {override_end}"
        )

    all_segments: list[pd.DataFrame] = []
    event_summaries: list[dict[str, Any]] = []

    for event in events:
        try:
            event_file = resolve_event_file(
                args.event_dir,
                event.event_id,
                explicit_event_file=args.event_file,
            )

            segment_df, event_summary = analyse_event(
                metadata=event,
                event_file=event_file,
                output_root=args.output_dir,
                manual_power_signals=manual_power_signals,
            )

            if not segment_df.empty:
                all_segments.append(segment_df)

            original_end = original_metadata_end_by_event.get(
                event.event_id
            )
            event_summary["original_metadata_end"] = (
                original_end if original_end is not None else event.event_end
            )
            event_summary["analysis_end_used"] = event.event_end
            event_summary["analysis_end_overridden"] = (
                original_end is not None
            )

            event_summaries.append(event_summary)

        except Exception as exc:
            print(
                f"[ERROR] Farm {event.farm_id}, event {event.event_id}: {exc}",
                file=sys.stderr,
            )
            event_summaries.append(
                {
                    "farm_id": event.farm_id,
                    "event_id": event.event_id,
                    "event_label": event.event_label,
                    "metadata_start": event.event_start,
                    "metadata_end": event.event_end,
                    "event_description": event.event_description,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    batch_dir = args.output_dir / f"farm_{args.farm}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(event_summaries)
    summary_df.to_csv(batch_dir / "all_events_run_summary.csv", index=False)

    if all_segments:
        all_segments_df = pd.concat(all_segments, ignore_index=True)
    else:
        all_segments_df = pd.DataFrame()

    all_segments_df.to_csv(
        batch_dir / "all_events_detected_segments.csv",
        index=False,
    )

    success_count = int((summary_df["status"] == "success").sum())
    failed_count = int((summary_df["status"] == "failed").sum())

    print(
        f"[DONE] Success: {success_count}; failed: {failed_count}. "
        f"Outputs: {batch_dir}"
    )
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())