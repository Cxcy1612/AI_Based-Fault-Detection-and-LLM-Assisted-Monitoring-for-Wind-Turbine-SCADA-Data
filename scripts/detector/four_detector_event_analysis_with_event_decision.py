#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated four-detector SCADA analysis with an interpretable
event-level decision layer.

This script uses four_detector_event_analysis_power_confirmed.py as the
validated detector engine, then adds:

1. Independent short-episode construction
2. Episode-based intermittent-cluster reconstruction
3. Persistent and localized duration aggregation
4. Confirmed anomaly coverage
5. Right-censoring and incomplete-recovery checks
6. Layered event-level decisions
7. Batch event-level evaluation

Required neighbouring file
---------------------------
four_detector_event_analysis_power_confirmed.py

Main outputs for each event
---------------------------
row_scores.csv
detected_segments_raw.csv
detected_segments.csv
short_episodes.csv
derived_short_clusters.csv
event_level_features.csv
event_level_decision.json
event_level_explanation.md

Batch outputs
-------------
all_events_detected_segments.csv
all_events_detected_segments_raw.csv
all_events_run_summary.csv
all_events_event_level_features.csv
all_events_event_level_decisions.csv
event_level_confusion_matrix.csv
event_level_evaluation.json

Example
-------
python four_detector_event_analysis_with_event_decision.py ^
  --farm C ^
  --metadata "..\\data\\raw\\Wind Farm C\\event_info.csv" ^
  --event-dir "..\\data\\raw\\Wind Farm C\\datasets" ^
  --event-id 35 ^
  --power-signals "power_2,power_5,power_6,power_17" ^
  --localized-recovery-window-points 3 ^
  --localized-recovery-ratio 0.60 ^
  --localized-drop-ratio 0.40 ^
  --localized-power-consensus-fraction 0.75 ^
  --localized-min-power-signals 3 ^
  --output-dir "..\\outputs\\four_detector_event35_with_decision"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import four_detector_event_analysis_power_confirmed as detector_engine
except ImportError as exc:
    raise ImportError(
        "Could not import four_detector_event_analysis_power_confirmed.py. "
        "Place this script in the same directory as the original four-detector "
        "script."
    ) from exc


# =============================================================================
# 1. EVENT-LEVEL CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class EventDecisionConfig:
    """
    Thresholds used by the event-level decision layer.

    These values are initial engineering thresholds developed from the inspected
    Farm C examples. They should later be calibrated on a training set and
    checked on a validation set.
    """

    # Merge confirmed short rows/segments into one independent short episode.
    short_merge_gap_minutes: float = 20.0

    # Density windows.
    short_window_2h_hours: float = 2.0
    short_window_24h_hours: float = 24.0

    # Episode-based clusters.
    weak_cluster_min_episodes: int = 2
    strong_cluster_min_episodes: int = 3
    cluster_window_hours: float = 2.0

    # Persistent-system anomaly path.
    persistent_max_hours_threshold: float = 2.0
    persistent_total_hours_threshold: float = 4.0

    # Power-confirmed localized anomaly path.
    localized_max_hours_threshold: float = 1.0

    # Repeated-short anomaly path.
    anomaly_max_short_2h: int = 4
    anomaly_max_short_24h: int = 6
    anomaly_strong_cluster_count: int = 2
    anomaly_min_short_episode_count: int = 8

    # Multi-detector anomaly path.
    anomaly_coverage_fraction: float = 0.03
    normal_transient_coverage_fraction: float = 0.005

    # Review region.
    review_max_short_2h_min: int = 2
    review_max_short_2h_max: int = 3
    review_max_short_24h_min: int = 3
    review_max_short_24h_max: int = 5
    review_rejected_localized_hours: float = 2.0

    # Tail inspection for possible right censoring.
    censor_tail_points: int = 3
    censor_tail_required_fraction: float = 0.67

    # A review result is conservatively mapped to normal in the binary output.
    review_binary_label: str = "normal"


# =============================================================================
# 2. GENERAL UTILITIES
# =============================================================================

def normalise_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed CSV boolean values into a clean boolean Series."""
    if series.empty:
        return pd.Series(False, index=series.index, dtype=bool)

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    true_values = {"true", "1", "yes", "y", "t"}
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(true_values)
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(result):
        return default
    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def json_ready(value: Any) -> Any:
    """Convert pandas and NumPy values into JSON-compatible objects."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]

    if pd.isna(value):
        return None

    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_ready(payload),
            file,
            ensure_ascii=False,
            indent=2,
        )


def prepare_segment_table(segment_df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the Raw segment table."""
    if segment_df is None or segment_df.empty:
        return pd.DataFrame(
            columns=[
                "farm_id",
                "event_id",
                "detector",
                "segment_id",
                "segment_start",
                "segment_end",
                "duration_minutes",
                "include_in_final",
                "candidate_status",
                "rejection_reason",
            ]
        )

    result = segment_df.copy()

    for column in ["segment_start", "segment_end"]:
        if column not in result.columns:
            result[column] = pd.NaT
        result[column] = pd.to_datetime(result[column], errors="coerce")

    if "duration_minutes" not in result.columns:
        result["duration_minutes"] = (
            result["segment_end"] - result["segment_start"]
        ).dt.total_seconds() / 60.0
    else:
        result["duration_minutes"] = pd.to_numeric(
            result["duration_minutes"],
            errors="coerce",
        )

    if "include_in_final" not in result.columns:
        result["include_in_final"] = True
    else:
        result["include_in_final"] = normalise_bool_series(
            result["include_in_final"]
        )

    result = result.dropna(subset=["segment_start", "segment_end"])
    result = result.loc[result["segment_end"] > result["segment_start"]]
    result = result.sort_values(["segment_start", "segment_end"]).reset_index(
        drop=True
    )

    return result


def clip_interval(
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    clipped_start = max(pd.Timestamp(start), pd.Timestamp(event_start))
    clipped_end = min(pd.Timestamp(end), pd.Timestamp(event_end))

    if clipped_end <= clipped_start:
        return None

    return clipped_start, clipped_end


def merge_intervals(
    intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
    gap_minutes: float = 0.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge overlapping intervals and intervals separated by a small gap."""
    clean_intervals = sorted(
        [
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in intervals
            if pd.notna(start)
            and pd.notna(end)
            and pd.Timestamp(end) > pd.Timestamp(start)
        ],
        key=lambda item: (item[0], item[1]),
    )

    if not clean_intervals:
        return []

    maximum_gap = pd.Timedelta(minutes=float(gap_minutes))

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start, current_end = clean_intervals[0]

    for start, end in clean_intervals[1:]:
        if start <= current_end + maximum_gap:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def intervals_total_minutes(
    intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    return float(
        sum(
            (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60.0
            for start, end in intervals
        )
    )


def maximum_episode_count_in_window(
    timestamps: pd.Series,
    window_hours: float,
) -> int:
    """Maximum number of episode starts in any fixed-duration time window."""
    times = (
        pd.to_datetime(timestamps, errors="coerce")
        .dropna()
        .sort_values()
        .tolist()
    )

    if not times:
        return 0

    window = pd.Timedelta(hours=float(window_hours))
    left = 0
    maximum_count = 0

    for right, current_time in enumerate(times):
        while left <= right and current_time - times[left] > window:
            left += 1

        maximum_count = max(maximum_count, right - left + 1)

    return maximum_count


# =============================================================================
# 3. SHORT EPISODES
# =============================================================================

def confirmed_detector_segments(
    raw_segments: pd.DataFrame,
    detector: str,
) -> pd.DataFrame:
    if raw_segments.empty:
        return raw_segments.copy()

    selected = raw_segments.loc[
        raw_segments["detector"].astype(str) == detector
    ].copy()

    if selected.empty:
        return selected

    # Localized candidates must pass the explicit power-boundary confirmation.
    if detector == "localized_persistent_subsystem_state":
        selected = selected.loc[
            normalise_bool_series(selected["include_in_final"])
        ].copy()

    return selected.sort_values("segment_start").reset_index(drop=True)


def power_consensus_requirement(row: pd.Series) -> int:
    signal_count = safe_int(row.get("power_signal_count"), default=0)

    if signal_count <= 0:
        return 0

    return max(1, int(math.ceil(0.75 * signal_count)))


def short_has_direct_recovery(row: pd.Series) -> bool:
    required = power_consensus_requirement(row)

    if required <= 0:
        return False

    recovery_count = safe_int(
        row.get("power_recovery_signal_count_max"),
        default=0,
    )
    return recovery_count >= required


def short_has_variability_support(row: pd.Series) -> bool:
    required = power_consensus_requirement(row)

    if required <= 0:
        return False

    variability_count = safe_int(
        row.get("power_variability_signal_count_max"),
        default=0,
    )
    return variability_count >= required


def build_short_episodes(
    raw_segments: pd.DataFrame,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    config: EventDecisionConfig,
) -> pd.DataFrame:
    """
    Merge adjacent confirmed short segments into independent short episodes.

    Example:
        19:40-19:50
        19:50-20:00
    becomes one 19:40-20:00 episode.
    """
    short_segments = confirmed_detector_segments(
        raw_segments,
        "short_standstill",
    )

    output_columns = [
        "short_episode_id",
        "episode_start",
        "episode_end",
        "duration_minutes",
        "source_short_segment_count",
        "source_short_segment_ids",
        "direct_recovery_segment_count",
        "variability_supported_segment_count",
        "episode_evidence_type",
        "maximum_detector_score",
        "maximum_abnormal_fraction",
    ]

    if short_segments.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict[str, Any]] = []

    for _, row in short_segments.iterrows():
        clipped = clip_interval(
            row["segment_start"],
            row["segment_end"],
            event_start,
            event_end,
        )
        if clipped is None:
            continue

        start, end = clipped

        records.append(
            {
                "start": start,
                "end": end,
                "segment_id": str(row.get("segment_id", "")),
                "direct_recovery": short_has_direct_recovery(row),
                "variability_support": short_has_variability_support(row),
                "detector_score": safe_float(
                    row.get("detector_score_max"),
                    default=0.0,
                ),
                "abnormal_fraction": safe_float(
                    row.get("abnormal_fraction_max"),
                    default=0.0,
                ),
            }
        )

    if not records:
        return pd.DataFrame(columns=output_columns)

    records.sort(key=lambda item: (item["start"], item["end"]))
    gap = pd.Timedelta(minutes=config.short_merge_gap_minutes)

    merged_records: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    for record in records:
        if current is None:
            current = {
                "start": record["start"],
                "end": record["end"],
                "segment_ids": [record["segment_id"]],
                "direct_recovery_count": int(record["direct_recovery"]),
                "variability_support_count": int(
                    record["variability_support"]
                ),
                "detector_scores": [record["detector_score"]],
                "abnormal_fractions": [record["abnormal_fraction"]],
            }
            continue

        if record["start"] <= current["end"] + gap:
            current["end"] = max(current["end"], record["end"])
            current["segment_ids"].append(record["segment_id"])
            current["direct_recovery_count"] += int(
                record["direct_recovery"]
            )
            current["variability_support_count"] += int(
                record["variability_support"]
            )
            current["detector_scores"].append(record["detector_score"])
            current["abnormal_fractions"].append(
                record["abnormal_fraction"]
            )
        else:
            merged_records.append(current)
            current = {
                "start": record["start"],
                "end": record["end"],
                "segment_ids": [record["segment_id"]],
                "direct_recovery_count": int(record["direct_recovery"]),
                "variability_support_count": int(
                    record["variability_support"]
                ),
                "detector_scores": [record["detector_score"]],
                "abnormal_fractions": [record["abnormal_fraction"]],
            }

    if current is not None:
        merged_records.append(current)

    rows: list[dict[str, Any]] = []

    for number, record in enumerate(merged_records, start=1):
        if record["direct_recovery_count"] > 0:
            evidence_type = "direct_power_recovery"
        elif record["variability_support_count"] > 0:
            evidence_type = "variability_supported"
        else:
            evidence_type = "power_confirmed_other"

        rows.append(
            {
                "short_episode_id": f"short_episode_{number:03d}",
                "episode_start": record["start"],
                "episode_end": record["end"],
                "duration_minutes": (
                    record["end"] - record["start"]
                ).total_seconds() / 60.0,
                "source_short_segment_count": len(record["segment_ids"]),
                "source_short_segment_ids": ";".join(
                    item for item in record["segment_ids"] if item
                ),
                "direct_recovery_segment_count": int(
                    record["direct_recovery_count"]
                ),
                "variability_supported_segment_count": int(
                    record["variability_support_count"]
                ),
                "episode_evidence_type": evidence_type,
                "maximum_detector_score": max(
                    record["detector_scores"],
                    default=0.0,
                ),
                "maximum_abnormal_fraction": max(
                    record["abnormal_fractions"],
                    default=0.0,
                ),
            }
        )

    return pd.DataFrame(rows, columns=output_columns)


# =============================================================================
# 4. EPISODE-BASED CLUSTERS
# =============================================================================

def derive_episode_clusters(
    short_episodes: pd.DataFrame,
    window_hours: float,
    minimum_episodes: int,
    cluster_type: str,
) -> pd.DataFrame:
    """
    Construct non-duplicated clusters from independent short episodes.

    Qualifying sliding windows are first identified, then overlapping qualifying
    windows are merged into one cluster. This prevents one dense fault period
    from being counted repeatedly.
    """
    columns = [
        "cluster_id",
        "cluster_type",
        "cluster_start",
        "cluster_end",
        "cluster_span_minutes",
        "episode_count",
        "episode_ids",
    ]

    if short_episodes.empty:
        return pd.DataFrame(columns=columns)

    episodes = short_episodes.copy()
    episodes["episode_start"] = pd.to_datetime(
        episodes["episode_start"],
        errors="coerce",
    )
    episodes["episode_end"] = pd.to_datetime(
        episodes["episode_end"],
        errors="coerce",
    )
    episodes = episodes.dropna(
        subset=["episode_start", "episode_end"]
    ).sort_values("episode_start").reset_index(drop=True)

    if len(episodes) < minimum_episodes:
        return pd.DataFrame(columns=columns)

    window = pd.Timedelta(hours=float(window_hours))
    qualifying: list[dict[str, Any]] = []

    for left in range(len(episodes)):
        right = left

        while (
            right + 1 < len(episodes)
            and episodes.loc[right + 1, "episode_start"]
            - episodes.loc[left, "episode_start"]
            <= window
        ):
            right += 1

        count = right - left + 1

        if count >= minimum_episodes:
            subset = episodes.iloc[left:right + 1]
            qualifying.append(
                {
                    "start": subset["episode_start"].min(),
                    "end": subset["episode_end"].max(),
                    "episode_ids": set(
                        subset["short_episode_id"].astype(str)
                    ),
                }
            )

    if not qualifying:
        return pd.DataFrame(columns=columns)

    qualifying.sort(key=lambda item: (item["start"], item["end"]))

    merged: list[dict[str, Any]] = []
    current = qualifying[0]

    for candidate in qualifying[1:]:
        if candidate["start"] <= current["end"]:
            current["end"] = max(current["end"], candidate["end"])
            current["episode_ids"].update(candidate["episode_ids"])
        else:
            merged.append(current)
            current = candidate

    merged.append(current)

    rows: list[dict[str, Any]] = []

    for number, cluster in enumerate(merged, start=1):
        episode_ids = sorted(cluster["episode_ids"])

        rows.append(
            {
                "cluster_id": f"{cluster_type}_cluster_{number:03d}",
                "cluster_type": cluster_type,
                "cluster_start": cluster["start"],
                "cluster_end": cluster["end"],
                "cluster_span_minutes": (
                    cluster["end"] - cluster["start"]
                ).total_seconds() / 60.0,
                "episode_count": len(episode_ids),
                "episode_ids": ";".join(episode_ids),
            }
        )

    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# 5. PERSISTENT AND LOCALIZED STATE STATISTICS
# =============================================================================

def detector_interval_statistics(
    segment_df: pd.DataFrame,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> dict[str, Any]:
    if segment_df.empty:
        return {
            "segment_count": 0,
            "merged_interval_count": 0,
            "max_hours": 0.0,
            "total_hours": 0.0,
            "merged_intervals": [],
        }

    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for _, row in segment_df.iterrows():
        clipped = clip_interval(
            row["segment_start"],
            row["segment_end"],
            event_start,
            event_end,
        )
        if clipped is not None:
            intervals.append(clipped)

    merged = merge_intervals(intervals)

    durations_hours = [
        (end - start).total_seconds() / 3600.0
        for start, end in merged
    ]

    return {
        "segment_count": int(len(segment_df)),
        "merged_interval_count": int(len(merged)),
        "max_hours": float(max(durations_hours, default=0.0)),
        "total_hours": float(sum(durations_hours)),
        "merged_intervals": merged,
    }


def rejected_localized_statistics(
    raw_segments: pd.DataFrame,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> dict[str, Any]:
    if raw_segments.empty:
        return {
            "candidate_count": 0,
            "max_hours": 0.0,
            "total_hours": 0.0,
            "ends_near_event_boundary": False,
        }

    localized = raw_segments.loc[
        raw_segments["detector"].astype(str)
        == "localized_persistent_subsystem_state"
    ].copy()

    if localized.empty:
        return {
            "candidate_count": 0,
            "max_hours": 0.0,
            "total_hours": 0.0,
            "ends_near_event_boundary": False,
        }

    rejected = localized.loc[
        ~normalise_bool_series(localized["include_in_final"])
    ].copy()

    if rejected.empty:
        return {
            "candidate_count": 0,
            "max_hours": 0.0,
            "total_hours": 0.0,
            "ends_near_event_boundary": False,
        }

    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for _, row in rejected.iterrows():
        clipped = clip_interval(
            row["segment_start"],
            row["segment_end"],
            event_start,
            event_end,
        )
        if clipped is not None:
            intervals.append(clipped)

    merged = merge_intervals(intervals)
    duration_hours = [
        (end - start).total_seconds() / 3600.0
        for start, end in merged
    ]

    boundary_tolerance = pd.Timedelta(minutes=20)

    ends_near_boundary = any(
        pd.Timestamp(end) >= pd.Timestamp(event_end) - boundary_tolerance
        for _, end in merged
    )

    return {
        "candidate_count": int(len(rejected)),
        "max_hours": float(max(duration_hours, default=0.0)),
        "total_hours": float(sum(duration_hours)),
        "ends_near_event_boundary": bool(ends_near_boundary),
    }


# =============================================================================
# 6. RIGHT-CENSORING AND DATA COMPLETENESS
# =============================================================================

def tail_flag_is_active(
    event_rows: pd.DataFrame,
    flag_column: str,
    config: EventDecisionConfig,
) -> bool:
    if flag_column not in event_rows.columns or event_rows.empty:
        return False

    tail_points = min(config.censor_tail_points, len(event_rows))
    if tail_points <= 0:
        return False

    values = normalise_bool_series(
        event_rows[flag_column].tail(tail_points)
    )

    required = max(
        1,
        int(
            math.ceil(
                tail_points * config.censor_tail_required_fraction
            )
        ),
    )

    return int(values.sum()) >= required


def calculate_censoring_features(
    row_scores: pd.DataFrame,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    sampling_minutes: float,
    rejected_localized: dict[str, Any],
    config: EventDecisionConfig,
) -> dict[str, Any]:
    if row_scores.empty or "time_stamp" not in row_scores.columns:
        return {
            "analysis_data_start": pd.NaT,
            "analysis_data_end": pd.NaT,
            "persistent_active_at_event_end": False,
            "localized_active_at_event_end": False,
            "raw_localized_reaches_event_end": bool(
                rejected_localized["ends_near_event_boundary"]
            ),
            "right_censored": bool(
                rejected_localized["ends_near_event_boundary"]
            ),
        }

    rows = row_scores.copy()
    rows["time_stamp"] = pd.to_datetime(
        rows["time_stamp"],
        errors="coerce",
    )
    rows = rows.dropna(subset=["time_stamp"]).sort_values("time_stamp")

    event_rows = rows.loc[
        (rows["time_stamp"] >= event_start)
        & (rows["time_stamp"] <= event_end)
    ].copy()

    persistent_tail = tail_flag_is_active(
        event_rows,
        "persistent_system_state_flag",
        config,
    )
    localized_tail = tail_flag_is_active(
        event_rows,
        "localized_persistent_subsystem_state_flag",
        config,
    )

    raw_localized_reaches_end = bool(
        rejected_localized["ends_near_event_boundary"]
    )

    right_censored = bool(
        persistent_tail
        or localized_tail
        or raw_localized_reaches_end
    )

    return {
        "analysis_data_start": (
            rows["time_stamp"].min() if not rows.empty else pd.NaT
        ),
        "analysis_data_end": (
            rows["time_stamp"].max() if not rows.empty else pd.NaT
        ),
        "persistent_active_at_event_end": persistent_tail,
        "localized_active_at_event_end": localized_tail,
        "raw_localized_reaches_event_end": raw_localized_reaches_end,
        "right_censored": right_censored,
        "sampling_minutes_used_for_censoring": sampling_minutes,
    }


# =============================================================================
# 7. EVENT-LEVEL FEATURE CONSTRUCTION
# =============================================================================

def build_event_level_features(
    metadata: Any,
    raw_segment_df: pd.DataFrame,
    row_scores: pd.DataFrame,
    sampling_minutes: float,
    config: EventDecisionConfig,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    event_start = pd.Timestamp(metadata.event_start)
    event_end = pd.Timestamp(metadata.event_end)

    event_duration_minutes = max(
        (event_end - event_start).total_seconds() / 60.0,
        sampling_minutes,
    )
    event_duration_hours = event_duration_minutes / 60.0
    event_duration_days = event_duration_hours / 24.0

    raw_segments = prepare_segment_table(raw_segment_df)

    short_episodes = build_short_episodes(
        raw_segments=raw_segments,
        event_start=event_start,
        event_end=event_end,
        config=config,
    )

    weak_clusters = derive_episode_clusters(
        short_episodes=short_episodes,
        window_hours=config.cluster_window_hours,
        minimum_episodes=config.weak_cluster_min_episodes,
        cluster_type="weak",
    )

    strong_clusters = derive_episode_clusters(
        short_episodes=short_episodes,
        window_hours=config.cluster_window_hours,
        minimum_episodes=config.strong_cluster_min_episodes,
        cluster_type="strong",
    )

    derived_clusters = pd.concat(
        [weak_clusters, strong_clusters],
        ignore_index=True,
    )

    persistent_segments = confirmed_detector_segments(
        raw_segments,
        "persistent_system_state",
    )
    localized_segments = confirmed_detector_segments(
        raw_segments,
        "localized_persistent_subsystem_state",
    )

    persistent_stats = detector_interval_statistics(
        persistent_segments,
        event_start,
        event_end,
    )
    localized_stats = detector_interval_statistics(
        localized_segments,
        event_start,
        event_end,
    )

    rejected_localized = rejected_localized_statistics(
        raw_segments,
        event_start,
        event_end,
    )

    max_short_2h = maximum_episode_count_in_window(
        short_episodes.get(
            "episode_start",
            pd.Series(dtype="datetime64[ns]"),
        ),
        config.short_window_2h_hours,
    )
    max_short_24h = maximum_episode_count_in_window(
        short_episodes.get(
            "episode_start",
            pd.Series(dtype="datetime64[ns]"),
        ),
        config.short_window_24h_hours,
    )

    # Coverage uses only actual confirmed abnormal intervals:
    # short episodes + persistent states + confirmed localized states.
    # Derived cluster windows are not added because a cluster window does not
    # mean that the turbine was continuously abnormal for the whole window.
    coverage_intervals: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ] = []

    if not short_episodes.empty:
        coverage_intervals.extend(
            list(
                zip(
                    short_episodes["episode_start"],
                    short_episodes["episode_end"],
                )
            )
        )

    coverage_intervals.extend(persistent_stats["merged_intervals"])
    coverage_intervals.extend(localized_stats["merged_intervals"])

    coverage_intervals = merge_intervals(coverage_intervals)
    coverage_minutes = intervals_total_minutes(coverage_intervals)
    coverage_fraction = min(
        1.0,
        coverage_minutes / event_duration_minutes,
    )

    short_count = int(len(short_episodes))
    direct_recovery_count = (
        int(
            (
                short_episodes["direct_recovery_segment_count"] > 0
            ).sum()
        )
        if not short_episodes.empty
        else 0
    )
    variability_only_count = (
        int(
            (
                (
                    short_episodes["direct_recovery_segment_count"] == 0
                )
                & (
                    short_episodes[
                        "variability_supported_segment_count"
                    ] > 0
                )
            ).sum()
        )
        if not short_episodes.empty
        else 0
    )

    detector_families: list[str] = []

    if short_count > 0:
        detector_families.append("short_standstill")

    if len(strong_clusters) > 0:
        detector_families.append("intermittent_cluster")

    if persistent_stats["segment_count"] > 0:
        detector_families.append("persistent_system_state")

    if localized_stats["segment_count"] > 0:
        detector_families.append(
            "localized_persistent_subsystem_state"
        )

    censoring = calculate_censoring_features(
        row_scores=row_scores,
        event_start=event_start,
        event_end=event_end,
        sampling_minutes=sampling_minutes,
        rejected_localized=rejected_localized,
        config=config,
    )

    features = {
        "farm_id": metadata.farm_id,
        "event_id": metadata.event_id,
        "metadata_label": metadata.event_label,
        "metadata_start": event_start,
        "metadata_end": event_end,
        "event_description": metadata.event_description,

        "event_duration_minutes": event_duration_minutes,
        "event_duration_hours": event_duration_hours,
        "event_duration_days": event_duration_days,
        "sampling_minutes": sampling_minutes,

        "short_episode_count": short_count,
        "short_rate_per_day": (
            short_count / event_duration_days
            if event_duration_days > 0
            else 0.0
        ),
        "direct_recovery_short_episode_count": (
            direct_recovery_count
        ),
        "variability_only_short_episode_count": (
            variability_only_count
        ),
        "max_short_count_2h": max_short_2h,
        "max_short_count_24h": max_short_24h,

        "weak_cluster_count": int(len(weak_clusters)),
        "strong_cluster_count": int(len(strong_clusters)),
        "maximum_episodes_in_strong_cluster": (
            int(strong_clusters["episode_count"].max())
            if not strong_clusters.empty
            else 0
        ),

        "persistent_segment_count": int(
            persistent_stats["segment_count"]
        ),
        "persistent_merged_interval_count": int(
            persistent_stats["merged_interval_count"]
        ),
        "max_persistent_hours": float(
            persistent_stats["max_hours"]
        ),
        "total_persistent_hours": float(
            persistent_stats["total_hours"]
        ),

        "confirmed_localized_segment_count": int(
            localized_stats["segment_count"]
        ),
        "confirmed_localized_merged_interval_count": int(
            localized_stats["merged_interval_count"]
        ),
        "max_confirmed_localized_hours": float(
            localized_stats["max_hours"]
        ),
        "total_confirmed_localized_hours": float(
            localized_stats["total_hours"]
        ),

        "rejected_localized_candidate_count": int(
            rejected_localized["candidate_count"]
        ),
        "max_rejected_localized_hours": float(
            rejected_localized["max_hours"]
        ),
        "total_rejected_localized_hours": float(
            rejected_localized["total_hours"]
        ),

        "confirmed_anomaly_coverage_minutes": coverage_minutes,
        "confirmed_anomaly_coverage_fraction": coverage_fraction,

        "detector_family_count": len(detector_families),
        "detector_families": ";".join(detector_families),

        **censoring,
    }

    return features, short_episodes, derived_clusters


# =============================================================================
# 8. LAYERED EVENT-LEVEL DECISION
# =============================================================================

def decide_event_level(
    features: dict[str, Any],
    config: EventDecisionConfig,
) -> dict[str, Any]:
    short_count = safe_int(features["short_episode_count"])
    max_short_2h = safe_int(features["max_short_count_2h"])
    max_short_24h = safe_int(features["max_short_count_24h"])
    strong_cluster_count = safe_int(
        features["strong_cluster_count"]
    )

    max_persistent = safe_float(
        features["max_persistent_hours"]
    )
    total_persistent = safe_float(
        features["total_persistent_hours"]
    )

    max_localized = safe_float(
        features["max_confirmed_localized_hours"]
    )
    total_localized = safe_float(
        features["total_confirmed_localized_hours"]
    )

    coverage = safe_float(
        features["confirmed_anomaly_coverage_fraction"]
    )
    detector_family_count = safe_int(
        features["detector_family_count"]
    )

    rejected_localized_max = safe_float(
        features["max_rejected_localized_hours"]
    )
    right_censored = bool(features["right_censored"])

    strong_paths: list[str] = []
    reasons: list[str] = []

    # -------------------------------------------------------------------------
    # Path A: persistent system-wide anomaly
    # -------------------------------------------------------------------------
    persistent_path = bool(
        max_persistent >= config.persistent_max_hours_threshold
        or total_persistent >= config.persistent_total_hours_threshold
    )

    if persistent_path:
        strong_paths.append("persistent_system_state")
        reasons.append(
            "Persistent-system evidence exceeded the event-level threshold: "
            f"longest={max_persistent:.2f} h, "
            f"total={total_persistent:.2f} h."
        )

    # -------------------------------------------------------------------------
    # Path B: power-confirmed localized persistent anomaly
    # -------------------------------------------------------------------------
    localized_path = bool(
        max_localized >= config.localized_max_hours_threshold
    )

    if localized_path:
        strong_paths.append("localized_persistent_state")
        reasons.append(
            "A power-confirmed localized state exceeded the duration "
            f"threshold: longest={max_localized:.2f} h, "
            f"total={total_localized:.2f} h."
        )

    # -------------------------------------------------------------------------
    # Path C: repeated short-standstill anomaly
    # -------------------------------------------------------------------------
    repeated_short_density_path = bool(
        max_short_2h >= config.anomaly_max_short_2h
        or max_short_24h >= config.anomaly_max_short_24h
    )

    intermittent_cluster_path = bool(
        strong_cluster_count
        >= config.anomaly_strong_cluster_count
        and short_count
        >= config.anomaly_min_short_episode_count
    )

    repeated_short_path = bool(
        repeated_short_density_path or intermittent_cluster_path
    )

    if repeated_short_path:
        strong_paths.append("repeated_short_standstill")
        reasons.append(
            "Repeated short-stop evidence exceeded the event-level threshold: "
            f"short episodes={short_count}, "
            f"max in 2 h={max_short_2h}, "
            f"max in 24 h={max_short_24h}, "
            f"strong clusters={strong_cluster_count}."
        )

    # -------------------------------------------------------------------------
    # Path D: multi-detector coverage support
    # -------------------------------------------------------------------------
    state_detector_present = bool(
        features["persistent_segment_count"] > 0
        or features["confirmed_localized_segment_count"] > 0
    )

    multi_detector_path = bool(
        coverage >= config.anomaly_coverage_fraction
        and detector_family_count >= 2
        and state_detector_present
    )

    if multi_detector_path:
        reasons.append(
            "Confirmed abnormal intervals covered "
            f"{coverage:.2%} of the event and were supported by "
            f"{detector_family_count} detector families."
        )

    # -------------------------------------------------------------------------
    # Strong anomaly result
    # -------------------------------------------------------------------------
    if strong_paths or multi_detector_path:
        independent_paths = list(dict.fromkeys(strong_paths))

        if len(independent_paths) >= 2 or multi_detector_path:
            detailed_status = "anomaly_multi_detector"
        elif independent_paths[0] == "persistent_system_state":
            detailed_status = "anomaly_persistent_system_state"
        elif independent_paths[0] == "localized_persistent_state":
            detailed_status = "anomaly_localized_persistent_state"
        else:
            detailed_status = "anomaly_repeated_short_standstill"

        if right_censored:
            reasons.append(
                "The abnormal state also reached the event or data boundary; "
                "the exact recovery time may be right-censored."
            )

        return {
            "binary_decision": "anomaly",
            "detailed_status": detailed_status,
            "decision_level": 3,
            "automatic_anomaly": True,
            "manual_review": right_censored,
            "strong_evidence_paths": ";".join(independent_paths),
            "decision_reason": " ".join(reasons),
        }

    # -------------------------------------------------------------------------
    # Review layer
    # -------------------------------------------------------------------------
    medium_short_density = bool(
        config.review_max_short_2h_min
        <= max_short_2h
        <= config.review_max_short_2h_max
        or config.review_max_short_24h_min
        <= max_short_24h
        <= config.review_max_short_24h_max
    )

    one_strong_cluster = strong_cluster_count == 1

    medium_coverage = bool(
        config.normal_transient_coverage_fraction
        <= coverage
        < config.anomaly_coverage_fraction
    )

    long_rejected_localized = bool(
        rejected_localized_max
        >= config.review_rejected_localized_hours
    )

    review_required = bool(
        medium_short_density
        or one_strong_cluster
        or medium_coverage
        or long_rejected_localized
        or right_censored
    )

    if review_required:
        review_reasons: list[str] = []

        if medium_short_density:
            review_reasons.append(
                "Short episodes showed moderate local density "
                f"(2 h maximum={max_short_2h}, "
                f"24 h maximum={max_short_24h})."
            )

        if one_strong_cluster:
            review_reasons.append(
                "One strong episode-based short-stop cluster was detected."
            )

        if medium_coverage:
            review_reasons.append(
                "Confirmed anomaly coverage was in the review region: "
                f"{coverage:.2%}."
            )

        if long_rejected_localized:
            review_reasons.append(
                "A long sensor-state localized candidate was detected but "
                "did not pass complete power-boundary confirmation: "
                f"longest={rejected_localized_max:.2f} h."
            )

        if right_censored:
            review_reasons.append(
                "An abnormal state reached the event or data boundary, so "
                "recovery could not be fully confirmed."
            )

        return {
            "binary_decision": config.review_binary_label,
            "detailed_status": "review_required",
            "decision_level": 2,
            "automatic_anomaly": False,
            "manual_review": True,
            "strong_evidence_paths": "",
            "decision_reason": " ".join(review_reasons),
        }

    # -------------------------------------------------------------------------
    # Normal with isolated operational transients
    # -------------------------------------------------------------------------
    if short_count > 0:
        return {
            "binary_decision": "normal",
            "detailed_status": "normal_with_transients",
            "decision_level": 1,
            "automatic_anomaly": False,
            "manual_review": False,
            "strong_evidence_paths": "",
            "decision_reason": (
                f"{short_count} isolated short episode(s) were detected, "
                "but no high-density, persistent, power-confirmed localized "
                "or multi-detector event-level threshold was reached."
            ),
        }

    # -------------------------------------------------------------------------
    # No confirmed event-level evidence
    # -------------------------------------------------------------------------
    return {
        "binary_decision": "normal",
        "detailed_status": "normal",
        "decision_level": 0,
        "automatic_anomaly": False,
        "manual_review": False,
        "strong_evidence_paths": "",
        "decision_reason": (
            "No confirmed event-level anomaly evidence was detected."
        ),
    }


# =============================================================================
# 9. HUMAN-READABLE EXPLANATION
# =============================================================================

def build_event_explanation(
    features: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    metadata_label = str(features["metadata_label"]).lower()
    predicted_label = str(decision["binary_decision"]).lower()

    label_match = (
        metadata_label == predicted_label
        if metadata_label in {"normal", "anomaly"}
        else None
    )

    lines = [
        f"# Event-level decision: Farm {features['farm_id']} "
        f"Event {features['event_id']}",
        "",
        "## Final decision",
        "",
        f"- Metadata label: **{features['metadata_label']}**",
        f"- Predicted binary label: **{decision['binary_decision']}**",
        f"- Detailed status: **{decision['detailed_status']}**",
        f"- Manual review required: **{decision['manual_review']}**",
        f"- Metadata agreement: **{label_match}**",
        "",
        "## Decision reason",
        "",
        decision["decision_reason"],
        "",
        "## Event-level features",
        "",
        f"- Event duration: {features['event_duration_hours']:.2f} hours",
        f"- Independent short episodes: "
        f"{features['short_episode_count']}",
        f"- Short rate: {features['short_rate_per_day']:.3f} per day",
        f"- Maximum short episodes in 2 hours: "
        f"{features['max_short_count_2h']}",
        f"- Maximum short episodes in 24 hours: "
        f"{features['max_short_count_24h']}",
        f"- Weak clusters: {features['weak_cluster_count']}",
        f"- Strong clusters: {features['strong_cluster_count']}",
        f"- Longest persistent interval: "
        f"{features['max_persistent_hours']:.2f} hours",
        f"- Total persistent duration: "
        f"{features['total_persistent_hours']:.2f} hours",
        f"- Longest confirmed localized interval: "
        f"{features['max_confirmed_localized_hours']:.2f} hours",
        f"- Total confirmed localized duration: "
        f"{features['total_confirmed_localized_hours']:.2f} hours",
        f"- Longest rejected localized candidate: "
        f"{features['max_rejected_localized_hours']:.2f} hours",
        f"- Confirmed anomaly coverage: "
        f"{features['confirmed_anomaly_coverage_fraction']:.2%}",
        f"- Detector families: "
        f"{features['detector_families'] or 'none'}",
        f"- Right-censored: {features['right_censored']}",
        "",
        "## Interpretation",
        "",
    ]

    status = decision["detailed_status"]

    if status == "normal":
        lines.append(
            "No confirmed temporal pattern was strong enough to support an "
            "event-level anomaly."
        )

    elif status == "normal_with_transients":
        lines.append(
            "The event contains isolated power-confirmed transient disturbances, "
            "but their frequency, density, duration and coverage remain consistent "
            "with a normal event containing occasional operational transients."
        )

    elif status == "review_required":
        lines.append(
            "The event contains borderline or incomplete evidence. It is not "
            "automatically classified as anomalous, but the detected pattern "
            "should be inspected before operational use."
        )

    elif status == "anomaly_repeated_short_standstill":
        lines.append(
            "The event is dominated by repeated short interruptions occurring "
            "at a frequency or local density that is substantially higher than "
            "the isolated transient pattern observed in normal events."
        )

    elif status == "anomaly_persistent_system_state":
        lines.append(
            "The event contains a broad, long-lasting SCADA state change. The "
            "persistent detector provides sufficient independent evidence for "
            "an event-level anomaly."
        )

    elif status == "anomaly_localized_persistent_state":
        lines.append(
            "The event contains a stable localized SCADA state change with "
            "confirmed power-dip and power-recovery boundary evidence."
        )

    elif status == "anomaly_multi_detector":
        lines.append(
            "Multiple independent detector paths support the event-level anomaly. "
            "The conclusion is not based on a single short transient."
        )

    return "\n".join(lines) + "\n"


# =============================================================================
# 10. EVENT OUTPUT
# =============================================================================

def save_event_level_outputs(
    event_output_dir: Path,
    features: dict[str, Any],
    decision: dict[str, Any],
    short_episodes: pd.DataFrame,
    derived_clusters: pd.DataFrame,
    config: EventDecisionConfig,
) -> None:
    event_output_dir.mkdir(parents=True, exist_ok=True)

    short_episodes.to_csv(
        event_output_dir / "short_episodes.csv",
        index=False,
    )
    derived_clusters.to_csv(
        event_output_dir / "derived_short_clusters.csv",
        index=False,
    )

    feature_row = {
        **features,
        **decision,
    }
    pd.DataFrame([feature_row]).to_csv(
        event_output_dir / "event_level_features.csv",
        index=False,
    )

    write_json(
        event_output_dir / "event_level_decision.json",
        {
            "features": features,
            "decision": decision,
            "event_decision_config": asdict(config),
        },
    )

    explanation = build_event_explanation(features, decision)
    with (
        event_output_dir / "event_level_explanation.md"
    ).open("w", encoding="utf-8") as file:
        file.write(explanation)


# =============================================================================
# 11. BATCH EVALUATION
# =============================================================================

def normalise_ground_truth_label(value: Any) -> Optional[str]:
    text = str(value).strip().lower()

    if text == "anomaly":
        return "anomaly"

    if text == "normal":
        return "normal"

    return None


def evaluate_event_level_predictions(
    decision_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if decision_df.empty:
        confusion = pd.DataFrame(
            [[0, 0], [0, 0]],
            index=["actual_normal", "actual_anomaly"],
            columns=["predicted_normal", "predicted_anomaly"],
        )
        return confusion, {
            "evaluated_events": 0,
            "accuracy": None,
            "precision_anomaly": None,
            "recall_anomaly": None,
            "f1_anomaly": None,
            "specificity_normal": None,
        }

    evaluation = decision_df.copy()
    evaluation["actual"] = evaluation["metadata_label"].map(
        normalise_ground_truth_label
    )
    evaluation["predicted"] = (
        evaluation["binary_decision"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    evaluation = evaluation.loc[
        evaluation["actual"].isin(["normal", "anomaly"])
        & evaluation["predicted"].isin(["normal", "anomaly"])
    ].copy()

    if evaluation.empty:
        return evaluate_event_level_predictions(pd.DataFrame())

    tp = int(
        (
            (evaluation["actual"] == "anomaly")
            & (evaluation["predicted"] == "anomaly")
        ).sum()
    )
    tn = int(
        (
            (evaluation["actual"] == "normal")
            & (evaluation["predicted"] == "normal")
        ).sum()
    )
    fp = int(
        (
            (evaluation["actual"] == "normal")
            & (evaluation["predicted"] == "anomaly")
        ).sum()
    )
    fn = int(
        (
            (evaluation["actual"] == "anomaly")
            & (evaluation["predicted"] == "normal")
        ).sum()
    )

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    if np.isfinite(precision) and np.isfinite(recall):
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    else:
        f1 = np.nan

    confusion = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp],
        ],
        index=["actual_normal", "actual_anomaly"],
        columns=["predicted_normal", "predicted_anomaly"],
    )

    metrics = {
        "evaluated_events": total,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision_anomaly": precision,
        "recall_anomaly": recall,
        "f1_anomaly": f1,
        "specificity_normal": specificity,
        "normal_false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else np.nan
        ),
    }

    return confusion, metrics


# =============================================================================
# 12. COMMAND-LINE CONFIGURATION
# =============================================================================

def add_event_level_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        "event-level decision options"
    )

    group.add_argument(
        "--event-short-merge-gap-minutes",
        type=float,
        default=20.0,
        help=(
            "Maximum normal gap used to merge nearby confirmed short "
            "segments into one independent episode. Default: 20 minutes."
        ),
    )
    group.add_argument(
        "--event-persistent-max-hours",
        type=float,
        default=2.0,
        help=(
            "Longest persistent interval required for automatic anomaly. "
            "Default: 2 hours."
        ),
    )
    group.add_argument(
        "--event-persistent-total-hours",
        type=float,
        default=4.0,
        help=(
            "Total persistent duration required for automatic anomaly. "
            "Default: 4 hours."
        ),
    )
    group.add_argument(
        "--event-localized-max-hours",
        type=float,
        default=1.0,
        help=(
            "Longest power-confirmed localized interval required for "
            "automatic anomaly. Default: 1 hour."
        ),
    )
    group.add_argument(
        "--event-max-short-2h",
        type=int,
        default=4,
        help=(
            "Independent short episodes within two hours required for "
            "automatic anomaly. Default: 4."
        ),
    )
    group.add_argument(
        "--event-max-short-24h",
        type=int,
        default=6,
        help=(
            "Independent short episodes within 24 hours required for "
            "automatic anomaly. Default: 6."
        ),
    )
    group.add_argument(
        "--event-strong-clusters",
        type=int,
        default=2,
        help=(
            "Strong episode-based clusters required for the cluster anomaly "
            "path. Default: 2."
        ),
    )
    group.add_argument(
        "--event-min-shorts-for-cluster-path",
        type=int,
        default=8,
        help=(
            "Minimum total short episodes required together with multiple "
            "strong clusters. Default: 8."
        ),
    )
    group.add_argument(
        "--event-anomaly-coverage",
        type=float,
        default=0.03,
        help=(
            "Confirmed event coverage required for the multi-detector path. "
            "Use a fraction, e.g. 0.03 means 3 percent."
        ),
    )
    group.add_argument(
        "--event-normal-transient-coverage",
        type=float,
        default=0.005,
        help=(
            "Upper isolated-transient coverage boundary. "
            "Default: 0.005, or 0.5 percent."
        ),
    )
    group.add_argument(
        "--event-review-binary-label",
        choices=["normal", "anomaly"],
        default="normal",
        help=(
            "Binary label assigned to review_required events. "
            "Default: normal."
        ),
    )

    return parser


def build_event_decision_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> EventDecisionConfig:
    positive_values = {
        "event_short_merge_gap_minutes":
            args.event_short_merge_gap_minutes,
        "event_persistent_max_hours":
            args.event_persistent_max_hours,
        "event_persistent_total_hours":
            args.event_persistent_total_hours,
        "event_localized_max_hours":
            args.event_localized_max_hours,
    }

    for name, value in positive_values.items():
        if float(value) < 0:
            parser.error(
                f"--{name.replace('_', '-')} must be >= 0."
            )

    positive_integer_values = {
        "event_max_short_2h": args.event_max_short_2h,
        "event_max_short_24h": args.event_max_short_24h,
        "event_strong_clusters": args.event_strong_clusters,
        "event_min_shorts_for_cluster_path":
            args.event_min_shorts_for_cluster_path,
    }

    for name, value in positive_integer_values.items():
        if int(value) <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be > 0."
            )

    for name, value in {
        "event_anomaly_coverage": args.event_anomaly_coverage,
        "event_normal_transient_coverage":
            args.event_normal_transient_coverage,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            parser.error(
                f"--{name.replace('_', '-')} must be in [0, 1]."
            )

    if (
        args.event_normal_transient_coverage
        >= args.event_anomaly_coverage
    ):
        parser.error(
            "--event-normal-transient-coverage must be smaller than "
            "--event-anomaly-coverage."
        )

    return EventDecisionConfig(
        short_merge_gap_minutes=float(
            args.event_short_merge_gap_minutes
        ),
        persistent_max_hours_threshold=float(
            args.event_persistent_max_hours
        ),
        persistent_total_hours_threshold=float(
            args.event_persistent_total_hours
        ),
        localized_max_hours_threshold=float(
            args.event_localized_max_hours
        ),
        anomaly_max_short_2h=int(args.event_max_short_2h),
        anomaly_max_short_24h=int(args.event_max_short_24h),
        anomaly_strong_cluster_count=int(
            args.event_strong_clusters
        ),
        anomaly_min_short_episode_count=int(
            args.event_min_shorts_for_cluster_path
        ),
        anomaly_coverage_fraction=float(
            args.event_anomaly_coverage
        ),
        normal_transient_coverage_fraction=float(
            args.event_normal_transient_coverage
        ),
        review_binary_label=str(
            args.event_review_binary_label
        ),
    )


# =============================================================================
# 13. MAIN INTEGRATED RUNNER
# =============================================================================

def main() -> int:
    parser = detector_engine.build_argument_parser()
    parser.description = (
        "Run four SCADA detectors and an interpretable event-level "
        "normal/anomaly decision layer."
    )
    parser = add_event_level_arguments(parser)
    args = parser.parse_args()

    if (
        args.event_file is not None
        and str(args.event_id).lower() == "all"
    ):
        parser.error(
            "--event-file can only be used with a specific --event-id."
        )

    if (
        args.analysis_end is not None
        and str(args.event_id).lower() == "all"
    ):
        parser.error(
            "--analysis-end can only be used with a specific --event-id."
        )

    detector_engine.apply_localized_runtime_overrides(
        args,
        parser,
    )

    event_config = build_event_decision_config(
        args,
        parser,
    )

    manual_power_signals = detector_engine.parse_power_signals(
        args.power_signals
    )

    events = detector_engine.load_metadata(
        args.metadata,
        args.farm,
    )

    if args.label_filter:
        events = [
            event
            for event in events
            if event.event_label.lower()
            == args.label_filter.lower()
        ]

    if str(args.event_id).lower() != "all":
        target_id = detector_engine.normalise_event_id(
            args.event_id
        )
        events = [
            event
            for event in events
            if event.event_id == target_id
        ]

    if not events:
        print(
            "[ERROR] No metadata rows matched the requested selection.",
            file=sys.stderr,
        )
        return 1

    original_metadata_end_by_event: dict[
        str,
        pd.Timestamp,
    ] = {}

    if args.analysis_end is not None:
        if len(events) != 1:
            parser.error(
                "--analysis-end requires exactly one selected event."
            )

        selected_event = events[0]
        override_end = detector_engine.parse_analysis_end(
            args.analysis_end,
            selected_event.event_start,
        )

        original_metadata_end_by_event[
            selected_event.event_id
        ] = selected_event.event_end

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

    all_display_segments: list[pd.DataFrame] = []
    all_raw_segments: list[pd.DataFrame] = []
    event_summaries: list[dict[str, Any]] = []
    event_feature_rows: list[dict[str, Any]] = []
    event_decision_rows: list[dict[str, Any]] = []

    for event in events:
        try:
            event_file = detector_engine.resolve_event_file(
                args.event_dir,
                event.event_id,
                explicit_event_file=args.event_file,
            )

            (
                display_segment_df,
                raw_segment_df,
                event_summary,
            ) = detector_engine.analyse_event(
                metadata=event,
                event_file=event_file,
                output_root=args.output_dir,
                manual_power_signals=manual_power_signals,
            )

            event_output_dir = (
                args.output_dir
                / f"farm_{event.farm_id}"
                / f"event_{event.event_id}"
            )

            row_score_path = (
                event_output_dir / "row_scores.csv"
            )

            if not row_score_path.exists():
                raise FileNotFoundError(
                    f"Expected row-score output was not found: "
                    f"{row_score_path}"
                )

            row_scores = pd.read_csv(
                row_score_path,
                low_memory=False,
            )

            sampling_minutes = safe_float(
                event_summary.get("sampling_minutes"),
                default=10.0,
            )

            (
                features,
                short_episodes,
                derived_clusters,
            ) = build_event_level_features(
                metadata=event,
                raw_segment_df=raw_segment_df,
                row_scores=row_scores,
                sampling_minutes=sampling_minutes,
                config=event_config,
            )

            decision = decide_event_level(
                features,
                event_config,
            )

            save_event_level_outputs(
                event_output_dir=event_output_dir,
                features=features,
                decision=decision,
                short_episodes=short_episodes,
                derived_clusters=derived_clusters,
                config=event_config,
            )

            original_end = original_metadata_end_by_event.get(
                event.event_id
            )

            event_summary["original_metadata_end"] = (
                original_end
                if original_end is not None
                else event.event_end
            )
            event_summary["analysis_end_used"] = event.event_end
            event_summary["analysis_end_overridden"] = (
                original_end is not None
            )

            event_summary.update(
                {
                    "event_level_binary_decision":
                        decision["binary_decision"],
                    "event_level_detailed_status":
                        decision["detailed_status"],
                    "event_level_manual_review":
                        decision["manual_review"],
                    "event_level_decision_reason":
                        decision["decision_reason"],
                    "short_episode_count":
                        features["short_episode_count"],
                    "max_short_count_2h":
                        features["max_short_count_2h"],
                    "max_short_count_24h":
                        features["max_short_count_24h"],
                    "max_persistent_hours":
                        features["max_persistent_hours"],
                    "max_confirmed_localized_hours":
                        features[
                            "max_confirmed_localized_hours"
                        ],
                    "confirmed_anomaly_coverage_fraction":
                        features[
                            "confirmed_anomaly_coverage_fraction"
                        ],
                    "right_censored":
                        features["right_censored"],
                }
            )

            event_summaries.append(event_summary)

            feature_row = {
                **features,
                **decision,
            }
            event_feature_rows.append(feature_row)

            event_decision_rows.append(
                {
                    "farm_id": event.farm_id,
                    "event_id": event.event_id,
                    "metadata_label": event.event_label,
                    "binary_decision":
                        decision["binary_decision"],
                    "detailed_status":
                        decision["detailed_status"],
                    "manual_review":
                        decision["manual_review"],
                    "strong_evidence_paths":
                        decision["strong_evidence_paths"],
                    "decision_reason":
                        decision["decision_reason"],
                    "metadata_match": (
                        normalise_ground_truth_label(
                            event.event_label
                        )
                        == decision["binary_decision"]
                        if normalise_ground_truth_label(
                            event.event_label
                        ) is not None
                        else np.nan
                    ),
                }
            )

            if not display_segment_df.empty:
                all_display_segments.append(
                    display_segment_df
                )

            if not raw_segment_df.empty:
                all_raw_segments.append(raw_segment_df)

            print(
                "[DECISION] "
                f"Farm {event.farm_id}, Event {event.event_id}: "
                f"{decision['binary_decision']} / "
                f"{decision['detailed_status']}"
            )
            print(
                f"[REASON] {decision['decision_reason']}"
            )

        except Exception as exc:
            print(
                f"[ERROR] Farm {event.farm_id}, "
                f"event {event.event_id}: {exc}",
                file=sys.stderr,
            )

            event_summaries.append(
                {
                    "farm_id": event.farm_id,
                    "event_id": event.event_id,
                    "event_label": event.event_label,
                    "metadata_start": event.event_start,
                    "metadata_end": event.event_end,
                    "event_description":
                        event.event_description,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    batch_dir = args.output_dir / f"farm_{args.farm}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(event_summaries)
    feature_df = pd.DataFrame(event_feature_rows)
    decision_df = pd.DataFrame(event_decision_rows)

    if all_display_segments:
        all_display_segments_df = pd.concat(
            all_display_segments,
            ignore_index=True,
        )
    else:
        all_display_segments_df = pd.DataFrame()

    if all_raw_segments:
        all_raw_segments_df = pd.concat(
            all_raw_segments,
            ignore_index=True,
        )
    else:
        all_raw_segments_df = pd.DataFrame()

    summary_df.to_csv(
        batch_dir / "all_events_run_summary.csv",
        index=False,
    )
    feature_df.to_csv(
        batch_dir / "all_events_event_level_features.csv",
        index=False,
    )
    decision_df.to_csv(
        batch_dir / "all_events_event_level_decisions.csv",
        index=False,
    )
    all_display_segments_df.to_csv(
        batch_dir / "all_events_detected_segments.csv",
        index=False,
    )
    all_raw_segments_df.to_csv(
        batch_dir / "all_events_detected_segments_raw.csv",
        index=False,
    )

    confusion_matrix, evaluation_metrics = (
        evaluate_event_level_predictions(decision_df)
    )

    confusion_matrix.to_csv(
        batch_dir / "event_level_confusion_matrix.csv"
    )
    write_json(
        batch_dir / "event_level_evaluation.json",
        evaluation_metrics,
    )
    write_json(
        batch_dir / "event_level_decision_configuration.json",
        asdict(event_config),
    )

    success_count = int(
        (summary_df.get("status", pd.Series(dtype=str)) == "success").sum()
    )
    failed_count = int(
        (summary_df.get("status", pd.Series(dtype=str)) == "failed").sum()
    )

    anomaly_count = int(
        (
            decision_df.get(
                "binary_decision",
                pd.Series(dtype=str),
            ) == "anomaly"
        ).sum()
    )
    normal_count = int(
        (
            decision_df.get(
                "binary_decision",
                pd.Series(dtype=str),
            ) == "normal"
        ).sum()
    )
    review_count = int(
        normalise_bool_series(
            decision_df.get(
                "manual_review",
                pd.Series(dtype=bool),
            )
        ).sum()
    )

    print(
        f"[DONE] Success: {success_count}; "
        f"failed: {failed_count}; "
        f"predicted anomaly: {anomaly_count}; "
        f"predicted normal: {normal_count}; "
        f"manual review: {review_count}."
    )
    print(f"[OUTPUT] {batch_dir}")

    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())