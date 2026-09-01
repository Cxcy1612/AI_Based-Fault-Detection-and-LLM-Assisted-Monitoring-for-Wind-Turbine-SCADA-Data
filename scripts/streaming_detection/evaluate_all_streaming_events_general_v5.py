#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrate and evaluate streaming replay results for all labelled events.

This script reads:
1. Farm metadata (for example, Wind Farm C/event_info.csv)
2. The output directory produced by
   streaming_detector_replay_v4_episode_trigger_support.py

Expected streaming output structure
-----------------------------------
<stream-output-dir>/
├─ <event_id>/
│  ├─ stream_row_scores.csv
│  ├─ stream_detected_episodes.csv
│  └─ stream_replay_summary.json
├─ all_stream_detected_episodes.csv
├─ stream_replay_manifest.json
└─ stream_replay_failures.csv

The script generates:
---------------------
event_level_streaming_evaluation.csv
episode_level_streaming_evaluation.csv
overall_streaming_metrics.json
overall_streaming_metrics.csv
missed_anomaly_events.csv
false_positive_normal_events.csv
high_alert_burden_events.csv
detector_family_summary.csv
streaming_evaluation_report.md
figure_1_event_level_confusion_matrix.png
figure_2_detection_delay_distribution.png
figure_3_active_fraction_by_label.png
figure_4_episode_count_by_event.png

Important interpretation
------------------------
- Event-level detection is based on whether at least one ACTIVE alert overlaps the
  metadata interval.
- For anomaly events, an overlapping alert is a true detection.
- For normal events, an overlapping alert is an event-level false positive.
- Alerts outside the metadata interval are reported separately as background
  alert burden and are not used directly in the event-level confusion matrix.
- Negative detection delay means the first overlapping alert became active
  before metadata_start and remained active into the labelled event interval.

Example
-------
python .\\wind_farm_fault_detection\\scripts\\evaluate_all_streaming_events.py `
  --metadata ".\\wind_farm_fault_detection\\data\\raw\\Wind Farm C\\event_info.csv" `
  --stream-output-dir ".\\wind_farm_fault_detection\\outputs\\farmC_streaming_replay_v4_all" `
  --output-dir ".\\wind_farm_fault_detection\\outputs\\farmC_streaming_replay_v4_evaluation" `
  --pre-event-hours 168
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

TRUE_TEXT = {"true", "1", "yes", "y", "t"}
FALSE_TEXT = {"false", "0", "no", "n", "f"}
TARGET_MAP = {"normal": 0, "anomaly": 1}
LABEL_MAP = {0: "normal", 1: "anomaly"}


# =============================================================================
# 1. GENERAL UTILITIES
# =============================================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read semicolon- or comma-separated CSV files."""
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=";", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", low_memory=False)
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";", engine="python", low_memory=False)

    df.columns = [str(column).strip() for column in df.columns]
    return df


def find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def normalise_event_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            lambda value: True
            if value in TRUE_TEXT
            else False
            if value in FALSE_TEXT
            else False
        )
        .astype(bool)
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)


def weighted_quantile(values: pd.Series, quantile: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(quantile))


# =============================================================================
# 2. LOAD METADATA AND STREAMING OUTPUTS
# =============================================================================

def load_metadata(path: Path, farm_id: Optional[str]) -> pd.DataFrame:
    df = read_csv_auto(path)

    id_col = find_column(df.columns, ["event_id", "id"])
    label_col = find_column(
        df.columns,
        ["event_label", "metadata_label", "label"],
    )
    start_col = find_column(
        df.columns,
        ["event_start", "metadata_start", "start"],
    )
    end_col = find_column(
        df.columns,
        ["event_end", "metadata_end", "end"],
    )
    description_col = find_column(
        df.columns,
        ["event_description", "description"],
    )

    if not all([id_col, label_col, start_col, end_col]):
        raise ValueError(
            "Metadata must contain event id, label, event start and event end."
        )

    output = pd.DataFrame(
        {
            "event_id": df[id_col].map(normalise_event_id),
            "metadata_label": (
                df[label_col].astype(str).str.strip().str.lower()
            ),
            "metadata_start": pd.to_datetime(
                df[start_col], errors="coerce"
            ),
            "metadata_end": pd.to_datetime(
                df[end_col], errors="coerce"
            ),
            "event_description": (
                df[description_col].fillna("").astype(str)
                if description_col
                else ""
            ),
        }
    )

    if farm_id:
        output["farm_id"] = str(farm_id)
    elif "farm_id" in df.columns:
        output["farm_id"] = df["farm_id"].astype(str)
    else:
        output["farm_id"] = ""

    output = output.loc[
        output["metadata_label"].isin(TARGET_MAP)
    ].copy()
    output = output.dropna(
        subset=["metadata_start", "metadata_end"]
    )
    output = output.loc[
        output["metadata_end"] >= output["metadata_start"]
    ].copy()
    output = output.drop_duplicates(
        subset=["event_id"], keep="last"
    )
    output["event_duration_hours"] = (
        output["metadata_end"] - output["metadata_start"]
    ).dt.total_seconds() / 3600.0
    return output.sort_values("metadata_start").reset_index(drop=True)


def read_manifest(stream_output_dir: Path) -> dict[str, Any]:
    path = stream_output_dir / "stream_replay_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_combined_episodes(stream_output_dir: Path) -> pd.DataFrame:
    combined_path = stream_output_dir / "all_stream_detected_episodes.csv"

    if combined_path.exists():
        episodes = pd.read_csv(combined_path, low_memory=False)
    else:
        frames: list[pd.DataFrame] = []
        for path in sorted(
            stream_output_dir.glob("*/stream_detected_episodes.csv")
        ):
            try:
                frame = pd.read_csv(path, low_memory=False)
            except pd.errors.EmptyDataError:
                continue
            if not frame.empty:
                frames.append(frame)
        episodes = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )

    if episodes.empty:
        return episodes

    for column in [
        "start_time",
        "end_time",
        "first_active_time",
        "last_evidence_time",
    ]:
        if column in episodes.columns:
            episodes[column] = pd.to_datetime(
                episodes[column], errors="coerce"
            )

    if "source_id" in episodes.columns:
        episodes["source_id"] = episodes["source_id"].map(
            normalise_event_id
        )

    for column in ["right_censored", "metadata_overlap"]:
        if column in episodes.columns:
            episodes[column] = to_bool_series(episodes[column])

    return episodes


def load_row_scores(
    stream_output_dir: Path,
    event_id: str,
) -> pd.DataFrame:
    path = stream_output_dir / event_id / "stream_row_scores.csv"
    if not path.exists():
        return pd.DataFrame()

    try:
        rows = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if rows.empty:
        return rows

    if "timestamp" not in rows.columns:
        raise ValueError(f"{path} does not contain timestamp.")

    rows["timestamp"] = pd.to_datetime(
        rows["timestamp"], errors="coerce"
    )
    rows = rows.dropna(subset=["timestamp"]).sort_values("timestamp")

    bool_columns = [
        column
        for column in [
            "warmup_complete",
            "baseline_updated",
            "review_flag",
            "active_alert_flag",
            "confirmed_evidence_flag",
            "short_episode_start_flag",
            "short_cluster_qualified_flag",
            "strong_short_trigger_flag",
            "short_cluster_rearm_flag",
            "short_cluster_support_flag",
            "short_confirmed_flag",
            "intermittent_cluster_flag",
            "persistent_system_state_flag",
            "localized_candidate_flag",
            "localized_confirmed_flag",
            "status_candidate_flag",
            "status_confirmed_flag",
            "semantic_candidate_flag",
            "semantic_confirmed_flag",
            "semantic_onset_flag",
            "semantic_chronic_flag",
            "semantic_new_burst_flag",
            "semantic_fusion_candidate_flag",
            "semantic_fusion_trigger_flag",
            "semantic_fusion_confirmed_flag",
            "semantic_fusion_rearm_flag",
            "targeted_change_candidate_flag",
            "targeted_change_trigger_flag",
            "communication_control_candidate_flag",
            "communication_control_confirmed_flag",
            "status_red_confirmed_flag",
            "semantic_multi_group_confirmed_flag",
            "right_censored",
        ]
        if column in rows.columns
    ]
    for column in bool_columns:
        rows[column] = to_bool_series(rows[column])

    return rows


# =============================================================================
# 3. EVENT-LEVEL EVALUATION
# =============================================================================

def interval_overlap_hours(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> float:
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    return max(0.0, (end - start).total_seconds() / 3600.0)


def union_interval_hours(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    clean = sorted(
        [
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in intervals
            if pd.notna(start)
            and pd.notna(end)
            and pd.Timestamp(end) >= pd.Timestamp(start)
        ],
        key=lambda item: (item[0], item[1]),
    )
    if not clean:
        return 0.0

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start, current_end = clean[0]

    for start, end in clean[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return float(
        sum(
            (end - start).total_seconds() / 3600.0
            for start, end in merged
        )
    )


def detector_family_set(text: Any) -> set[str]:
    if pd.isna(text):
        return set()
    tokens = re.split(r"[;,|]+", str(text))
    return {token.strip() for token in tokens if token.strip()}


def summarise_event(
    metadata_row: pd.Series,
    rows: pd.DataFrame,
    episodes: pd.DataFrame,
    pre_event_hours: float,
) -> dict[str, Any]:
    event_id = normalise_event_id(metadata_row["event_id"])
    label = str(metadata_row["metadata_label"]).lower()
    metadata_start = pd.Timestamp(metadata_row["metadata_start"])
    metadata_end = pd.Timestamp(metadata_row["metadata_end"])
    event_hours = max(
        0.0,
        (metadata_end - metadata_start).total_seconds() / 3600.0,
    )

    event_episodes = episodes.copy()
    if not event_episodes.empty and "source_id" in event_episodes.columns:
        event_episodes = event_episodes.loc[
            event_episodes["source_id"] == event_id
        ].copy()

    overlapping_episodes = pd.DataFrame()
    if not event_episodes.empty:
        overlap_mask = (
            (event_episodes["end_time"] >= metadata_start)
            & (event_episodes["start_time"] <= metadata_end)
        )
        overlapping_episodes = event_episodes.loc[overlap_mask].copy()
        if not overlapping_episodes.empty:
            overlapping_episodes["computed_overlap_hours"] = (
                overlapping_episodes.apply(
                    lambda item: interval_overlap_hours(
                        pd.Timestamp(item["start_time"]),
                        pd.Timestamp(item["end_time"]),
                        metadata_start,
                        metadata_end,
                    ),
                    axis=1,
                )
            )

    event_detected = not overlapping_episodes.empty
    predicted_binary = int(event_detected)
    actual_binary = TARGET_MAP[label]

    first_alert_time = pd.NaT
    detection_delay_hours = float("nan")
    if event_detected:
        active_times = pd.to_datetime(
            overlapping_episodes.get(
                "first_active_time",
                overlapping_episodes["start_time"],
            ),
            errors="coerce",
        ).dropna()
        if not active_times.empty:
            first_alert_time = active_times.min()
            detection_delay_hours = float(
                (first_alert_time - metadata_start).total_seconds()
                / 3600.0
            )

    overlapping_intervals: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ] = []
    if not overlapping_episodes.empty:
        for _, episode in overlapping_episodes.iterrows():
            overlapping_intervals.append(
                (
                    max(pd.Timestamp(episode["start_time"]), metadata_start),
                    min(pd.Timestamp(episode["end_time"]), metadata_end),
                )
            )

    episode_overlap_union_hours = union_interval_hours(
        overlapping_intervals
    )
    episode_overlap_fraction = min(
        1.0,
        safe_divide(episode_overlap_union_hours, event_hours)
        if event_hours > 0
        else 0.0,
    )

    # Three fixed event-level evaluation tiers.
    any_overlap_detected = bool(event_detected)
    meaningful_detected = bool(
        any_overlap_detected
        and (
            episode_overlap_union_hours >= 1.0
            or episode_overlap_fraction >= 0.05
        )
    )
    operational_detected = bool(
        meaningful_detected
        and np.isfinite(detection_delay_hours)
        and -24.0 <= detection_delay_hours <= 24.0
    )

    row_count_total = 0
    valid_row_count_total = 0
    active_rows_total = 0
    review_rows_total = 0
    active_fraction_total = float("nan")
    review_fraction_total = float("nan")
    active_rows_inside = 0
    row_count_inside = 0
    active_fraction_inside = float("nan")
    review_rows_inside = 0
    confirmed_rows_inside = 0
    baseline_frozen_rows_total = 0
    pre_event_row_count = 0
    pre_event_active_rows = 0
    pre_event_active_fraction = float("nan")
    pre_event_first_active_time = pd.NaT
    pre_event_active_hours = 0.0

    row_metrics: dict[str, int] = {
        "short_episode_starts_inside": 0,
        "short_cluster_qualified_rows_inside": 0,
        "strong_short_trigger_rows_inside": 0,
        "short_cluster_rearm_rows_inside": 0,
        "short_cluster_support_rows_inside": 0,
        "persistent_confirmed_rows_inside": 0,
        "localized_candidate_rows_inside": 0,
        "localized_confirmed_rows_inside": 0,
        "status_candidate_rows_inside": 0,
        "status_confirmed_rows_inside": 0,
        "semantic_candidate_rows_inside": 0,
        "semantic_confirmed_rows_inside": 0,
        "semantic_onset_rows_inside": 0,
        "semantic_chronic_rows_inside": 0,
        "semantic_fusion_confirmed_rows_inside": 0,
        "targeted_change_trigger_rows_inside": 0,
        "communication_control_confirmed_rows_inside": 0,
        "status_red_confirmed_rows_inside": 0,
    }

    if not rows.empty:
        row_count_total = int(len(rows))
        if "warmup_complete" in rows.columns:
            valid_rows = rows.loc[rows["warmup_complete"]].copy()
        else:
            valid_rows = rows.copy()

        valid_row_count_total = int(len(valid_rows))

        active_series = (
            valid_rows["active_alert_flag"]
            if "active_alert_flag" in valid_rows.columns
            else pd.Series(False, index=valid_rows.index)
        )
        review_series = (
            valid_rows["review_flag"]
            if "review_flag" in valid_rows.columns
            else pd.Series(False, index=valid_rows.index)
        )

        active_rows_total = int(active_series.sum())
        review_rows_total = int(review_series.sum())
        active_fraction_total = safe_divide(
            active_rows_total, valid_row_count_total
        )
        review_fraction_total = safe_divide(
            review_rows_total, valid_row_count_total
        )

        if "baseline_freeze_flag" in valid_rows.columns:
            baseline_frozen_rows_total = int(
                to_bool_series(
                    valid_rows["baseline_freeze_flag"]
                ).sum()
            )

        inside = valid_rows.loc[
            (valid_rows["timestamp"] >= metadata_start)
            & (valid_rows["timestamp"] <= metadata_end)
        ].copy()
        row_count_inside = int(len(inside))

        if not inside.empty:
            active_rows_inside = int(
                inside.get(
                    "active_alert_flag",
                    pd.Series(False, index=inside.index),
                ).sum()
            )
            review_rows_inside = int(
                inside.get(
                    "review_flag",
                    pd.Series(False, index=inside.index),
                ).sum()
            )
            confirmed_rows_inside = int(
                inside.get(
                    "confirmed_evidence_flag",
                    pd.Series(False, index=inside.index),
                ).sum()
            )
            active_fraction_inside = safe_divide(
                active_rows_inside, row_count_inside
            )

            flag_mapping = {
                "short_episode_starts_inside":
                    "short_episode_start_flag",
                "short_cluster_qualified_rows_inside":
                    "short_cluster_qualified_flag",
                "strong_short_trigger_rows_inside":
                    "strong_short_trigger_flag",
                "short_cluster_rearm_rows_inside":
                    "short_cluster_rearm_flag",
                "short_cluster_support_rows_inside":
                    "short_cluster_support_flag",
                "persistent_confirmed_rows_inside":
                    "persistent_system_state_flag",
                "localized_candidate_rows_inside":
                    "localized_candidate_flag",
                "localized_confirmed_rows_inside":
                    "localized_confirmed_flag",
                "status_candidate_rows_inside":
                    "status_candidate_flag",
                "status_confirmed_rows_inside":
                    "status_confirmed_flag",
                "semantic_candidate_rows_inside":
                    "semantic_candidate_flag",
                "semantic_confirmed_rows_inside":
                    "semantic_confirmed_flag",
                "semantic_onset_rows_inside":
                    "semantic_onset_flag",
                "semantic_chronic_rows_inside":
                    "semantic_chronic_flag",
                "semantic_fusion_confirmed_rows_inside":
                    "semantic_fusion_trigger_flag",
                "targeted_change_trigger_rows_inside":
                    "targeted_change_trigger_flag",
                "communication_control_confirmed_rows_inside":
                    "communication_control_confirmed_flag",
                "status_red_confirmed_rows_inside":
                    "status_red_confirmed_flag",
            }
            for output_name, input_name in flag_mapping.items():
                if input_name in inside.columns:
                    row_metrics[output_name] = int(
                        inside[input_name].sum()
                    )

        pre_start = metadata_start - pd.Timedelta(
            hours=pre_event_hours
        )
        pre_rows = valid_rows.loc[
            (valid_rows["timestamp"] >= pre_start)
            & (valid_rows["timestamp"] < metadata_start)
        ].copy()
        pre_event_row_count = int(len(pre_rows))
        if not pre_rows.empty:
            pre_active = pre_rows.loc[
                pre_rows.get(
                    "active_alert_flag",
                    pd.Series(False, index=pre_rows.index),
                )
            ]
            pre_event_active_rows = int(len(pre_active))
            pre_event_active_fraction = safe_divide(
                pre_event_active_rows, pre_event_row_count
            )
            if not pre_active.empty:
                pre_event_first_active_time = pd.Timestamp(
                    pre_active["timestamp"].min()
                )

            sampling_minutes = (
                float(
                    pre_rows["timestamp"]
                    .sort_values()
                    .diff()
                    .dt.total_seconds()
                    .div(60.0)
                    .dropna()
                    .median()
                )
                if len(pre_rows) > 1
                else 0.0
            )
            if np.isfinite(sampling_minutes):
                pre_event_active_hours = (
                    pre_event_active_rows * sampling_minutes / 60.0
                )

    detector_counter: Counter[str] = Counter()
    if not overlapping_episodes.empty:
        for value in overlapping_episodes.get(
            "detector_families",
            pd.Series("", index=overlapping_episodes.index),
        ):
            detector_counter.update(detector_family_set(value))

    dominant_detector_family = ""
    if detector_counter:
        dominant_detector_family = detector_counter.most_common(1)[0][0]

    event_episode_count = int(len(event_episodes))
    overlap_episode_count = int(len(overlapping_episodes))
    background_episode_count = max(
        0, event_episode_count - overlap_episode_count
    )

    total_episode_hours = 0.0
    max_episode_hours = 0.0
    right_censored_episode_count = 0
    if not event_episodes.empty:
        duration_values = pd.to_numeric(
            event_episodes.get(
                "duration_hours",
                pd.Series(0.0, index=event_episodes.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        total_episode_hours = float(duration_values.sum())
        max_episode_hours = float(duration_values.max())
        if "right_censored" in event_episodes.columns:
            right_censored_episode_count = int(
                event_episodes["right_censored"].sum()
            )

    status = (
        "true_positive"
        if actual_binary == 1 and predicted_binary == 1
        else "false_negative"
        if actual_binary == 1 and predicted_binary == 0
        else "false_positive"
        if actual_binary == 0 and predicted_binary == 1
        else "true_negative"
    )

    result: dict[str, Any] = {
        "farm_id": metadata_row.get("farm_id", ""),
        "event_id": event_id,
        "metadata_label": label,
        "actual_binary": actual_binary,
        "predicted_binary": predicted_binary,
        "predicted_label": LABEL_MAP[predicted_binary],
        "event_level_status": status,
        "event_detected": event_detected,
        "any_overlap_detected": any_overlap_detected,
        "meaningful_detected": meaningful_detected,
        "operational_detected": operational_detected,
        "metadata_start": metadata_start,
        "metadata_end": metadata_end,
        "event_duration_hours": event_hours,
        "event_description": metadata_row.get(
            "event_description", ""
        ),
        "source_result_available": not rows.empty,
        "event_episode_count": event_episode_count,
        "overlap_episode_count": overlap_episode_count,
        "background_episode_count": background_episode_count,
        "first_overlapping_alert_time": first_alert_time,
        "detection_delay_hours": detection_delay_hours,
        "early_alert": bool(
            np.isfinite(detection_delay_hours)
            and detection_delay_hours < 0
        ),
        "episode_overlap_union_hours": episode_overlap_union_hours,
        "episode_overlap_fraction": episode_overlap_fraction,
        "total_episode_hours_full_file": total_episode_hours,
        "max_episode_hours_full_file": max_episode_hours,
        "right_censored_episode_count": right_censored_episode_count,
        "row_count_total": row_count_total,
        "valid_row_count_total": valid_row_count_total,
        "active_rows_total": active_rows_total,
        "active_fraction_total": active_fraction_total,
        "review_rows_total": review_rows_total,
        "review_fraction_total": review_fraction_total,
        "baseline_frozen_rows_total": baseline_frozen_rows_total,
        "row_count_inside_metadata": row_count_inside,
        "active_rows_inside_metadata": active_rows_inside,
        "active_fraction_inside_metadata": active_fraction_inside,
        "review_rows_inside_metadata": review_rows_inside,
        "confirmed_rows_inside_metadata": confirmed_rows_inside,
        "pre_event_window_hours": pre_event_hours,
        "pre_event_row_count": pre_event_row_count,
        "pre_event_active_rows": pre_event_active_rows,
        "pre_event_active_fraction": pre_event_active_fraction,
        "pre_event_active_hours": pre_event_active_hours,
        "pre_event_first_active_time": pre_event_first_active_time,
        "dominant_detector_family": dominant_detector_family,
        "overlap_detector_families": ";".join(
            sorted(detector_counter)
        ),
        **row_metrics,
    }

    return result


def build_event_level_table(
    metadata: pd.DataFrame,
    stream_output_dir: Path,
    episodes: pd.DataFrame,
    pre_event_hours: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, metadata_row in metadata.iterrows():
        event_id = normalise_event_id(metadata_row["event_id"])
        print(f"[evaluation] Event {event_id}")
        row_scores = load_row_scores(stream_output_dir, event_id)
        rows.append(
            summarise_event(
                metadata_row=metadata_row,
                rows=row_scores,
                episodes=episodes,
                pre_event_hours=pre_event_hours,
            )
        )

    return pd.DataFrame(rows).sort_values(
        ["metadata_start", "event_id"]
    ).reset_index(drop=True)


# =============================================================================
# 4. EPISODE-LEVEL ANNOTATION
# =============================================================================

def annotate_episode_table(
    episodes: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()

    metadata_lookup = metadata.set_index("event_id")
    output = episodes.copy()

    output["source_id"] = output["source_id"].map(normalise_event_id)
    output["source_metadata_label"] = ""
    output["source_metadata_start"] = pd.NaT
    output["source_metadata_end"] = pd.NaT
    output["computed_overlap_hours"] = 0.0
    output["overlaps_source_metadata"] = False
    output["episode_relation"] = "background"

    for index, row in output.iterrows():
        source_id = normalise_event_id(row["source_id"])
        if source_id not in metadata_lookup.index:
            continue

        item = metadata_lookup.loc[source_id]
        metadata_start = pd.Timestamp(item["metadata_start"])
        metadata_end = pd.Timestamp(item["metadata_end"])

        start = pd.Timestamp(row["start_time"])
        end = pd.Timestamp(row["end_time"])

        overlap_hours = interval_overlap_hours(
            start, end, metadata_start, metadata_end
        )

        output.at[index, "source_metadata_label"] = item[
            "metadata_label"
        ]
        output.at[index, "source_metadata_start"] = metadata_start
        output.at[index, "source_metadata_end"] = metadata_end
        output.at[index, "computed_overlap_hours"] = overlap_hours
        output.at[index, "overlaps_source_metadata"] = (
            overlap_hours > 0
        )

        if overlap_hours > 0:
            output.at[index, "episode_relation"] = (
                "overlap_anomaly"
                if item["metadata_label"] == "anomaly"
                else "overlap_normal"
            )
        elif end < metadata_start:
            output.at[index, "episode_relation"] = "before_metadata"
        elif start > metadata_end:
            output.at[index, "episode_relation"] = "after_metadata"

    return output


# =============================================================================
# 5. OVERALL METRICS
# =============================================================================

def tier_confusion_metrics(
    event_table: pd.DataFrame,
    prediction_column: str,
    prefix: str,
) -> dict[str, Any]:
    actual = event_table["actual_binary"].astype(int)
    predicted = event_table[prediction_column].astype(bool).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        f"{prefix}_true_negative": int(tn),
        f"{prefix}_false_positive": int(fp),
        f"{prefix}_false_negative": int(fn),
        f"{prefix}_true_positive": int(tp),
        f"{prefix}_accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        f"{prefix}_recall": safe_divide(tp, tp + fn),
        f"{prefix}_precision": safe_divide(tp, tp + fp),
        f"{prefix}_specificity": safe_divide(tn, tn + fp),
        f"{prefix}_false_positive_rate": safe_divide(fp, tn + fp),
        f"{prefix}_balanced_accuracy": float(
            np.nanmean(
                [
                    safe_divide(tp, tp + fn),
                    safe_divide(tn, tn + fp),
                ]
            )
        ),
    }


def calculate_overall_metrics(
    event_table: pd.DataFrame,
    episode_table: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    actual = event_table["actual_binary"].astype(int)
    predicted = event_table["predicted_binary"].astype(int)

    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    anomaly_rows = event_table.loc[
        event_table["metadata_label"] == "anomaly"
    ].copy()
    normal_rows = event_table.loc[
        event_table["metadata_label"] == "normal"
    ].copy()
    detected_anomalies = anomaly_rows.loc[
        anomaly_rows["event_detected"]
    ].copy()

    delay_values = pd.to_numeric(
        detected_anomalies["detection_delay_hours"],
        errors="coerce",
    ).dropna()

    positive_delays = delay_values.loc[delay_values >= 0]
    negative_delays = delay_values.loc[delay_values < 0]

    total_valid_rows = float(
        pd.to_numeric(
            event_table["valid_row_count_total"],
            errors="coerce",
        ).fillna(0).sum()
    )
    total_active_rows = float(
        pd.to_numeric(
            event_table["active_rows_total"],
            errors="coerce",
        ).fillna(0).sum()
    )

    background_episode_count = 0
    background_episode_hours = 0.0
    if not episode_table.empty:
        background = episode_table.loc[
            ~episode_table["overlaps_source_metadata"]
        ].copy()
        background_episode_count = int(len(background))
        background_episode_hours = float(
            pd.to_numeric(
                background.get(
                    "duration_hours",
                    pd.Series(0.0, index=background.index),
                ),
                errors="coerce",
            ).fillna(0.0).sum()
        )

    tier_metrics: dict[str, Any] = {}
    tier_metrics.update(
        tier_confusion_metrics(
            event_table, "any_overlap_detected", "any_overlap"
        )
    )
    tier_metrics.update(
        tier_confusion_metrics(
            event_table, "meaningful_detected", "meaningful"
        )
    )
    tier_metrics.update(
        tier_confusion_metrics(
            event_table, "operational_detected", "operational"
        )
    )

    metrics: dict[str, Any] = {
        "event_count": int(len(event_table)),
        "anomaly_event_count": int(len(anomaly_rows)),
        "normal_event_count": int(len(normal_rows)),
        "true_positive": int(tp),
        "false_negative": int(fn),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "event_level_accuracy": safe_divide(
            tp + tn, tp + tn + fp + fn
        ),
        "event_level_balanced_accuracy": float(
            np.nanmean(
                [
                    safe_divide(tp, tp + fn),
                    safe_divide(tn, tn + fp),
                ]
            )
        ),
        "anomaly_event_recall": safe_divide(tp, tp + fn),
        "anomaly_event_precision": safe_divide(tp, tp + fp),
        "normal_event_specificity": safe_divide(tn, tn + fp),
        "normal_event_false_positive_rate": safe_divide(
            fp, tn + fp
        ),
        "detected_anomaly_event_count": int(tp),
        "missed_anomaly_event_count": int(fn),
        "false_positive_normal_event_count": int(fp),
        "events_with_available_row_results": int(
            event_table["source_result_available"].sum()
        ),
        "events_missing_row_results": int(
            (~event_table["source_result_available"]).sum()
        ),
        "total_episode_count": int(
            len(episode_table)
            if not episode_table.empty
            else event_table["event_episode_count"].sum()
        ),
        "background_episode_count": background_episode_count,
        "background_episode_hours": background_episode_hours,
        "mean_episodes_per_event": float(
            event_table["event_episode_count"].mean()
        ),
        "median_episodes_per_event": float(
            event_table["event_episode_count"].median()
        ),
        "mean_active_fraction_full_file": float(
            pd.to_numeric(
                event_table["active_fraction_total"],
                errors="coerce",
            ).mean()
        ),
        "median_active_fraction_full_file": float(
            pd.to_numeric(
                event_table["active_fraction_total"],
                errors="coerce",
            ).median()
        ),
        "global_active_row_fraction": safe_divide(
            total_active_rows, total_valid_rows
        ),
        "mean_active_fraction_inside_anomaly_events": float(
            pd.to_numeric(
                anomaly_rows["active_fraction_inside_metadata"],
                errors="coerce",
            ).mean()
        ),
        "mean_active_fraction_inside_normal_events": float(
            pd.to_numeric(
                normal_rows["active_fraction_inside_metadata"],
                errors="coerce",
            ).mean()
        ),
        "median_detection_delay_hours_all_detected_anomalies":
            float(delay_values.median())
            if not delay_values.empty
            else float("nan"),
        "mean_detection_delay_hours_all_detected_anomalies":
            float(delay_values.mean())
            if not delay_values.empty
            else float("nan"),
        "median_positive_detection_delay_hours":
            float(positive_delays.median())
            if not positive_delays.empty
            else float("nan"),
        "mean_positive_detection_delay_hours":
            float(positive_delays.mean())
            if not positive_delays.empty
            else float("nan"),
        "early_alert_anomaly_count": int(len(negative_delays)),
        "early_alert_anomaly_fraction": safe_divide(
            len(negative_delays), len(anomaly_rows)
        ),
        "median_early_alert_lead_hours":
            float((-negative_delays).median())
            if not negative_delays.empty
            else float("nan"),
        "manifest_processed_sources": manifest.get(
            "processed_sources"
        ),
        "manifest_failed_sources": manifest.get("failed_sources"),
        "manifest_total_detected_episodes": manifest.get(
            "total_detected_episodes"
        ),
        **tier_metrics,
    }

    return metrics


def metrics_to_table(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in metrics.items()
        ]
    )


# =============================================================================
# 6. DETECTOR-FAMILY ANALYSIS
# =============================================================================

def build_detector_family_summary(
    episode_table: pd.DataFrame,
) -> pd.DataFrame:
    if episode_table.empty or "detector_families" not in episode_table.columns:
        return pd.DataFrame(
            columns=[
                "detector_family",
                "episode_count",
                "overlap_anomaly_episode_count",
                "overlap_normal_episode_count",
                "background_episode_count",
                "total_duration_hours",
            ]
        )

    rows: list[dict[str, Any]] = []
    for _, episode in episode_table.iterrows():
        families = detector_family_set(
            episode.get("detector_families", "")
        )
        if not families:
            families = {"unclassified"}

        for family in families:
            rows.append(
                {
                    "detector_family": family,
                    "episode_relation": episode.get(
                        "episode_relation", "background"
                    ),
                    "duration_hours": pd.to_numeric(
                        episode.get("duration_hours", 0.0),
                        errors="coerce",
                    ),
                }
            )

    expanded = pd.DataFrame(rows)
    output_rows: list[dict[str, Any]] = []

    for family, group in expanded.groupby("detector_family"):
        relation_counts = group["episode_relation"].value_counts()
        output_rows.append(
            {
                "detector_family": family,
                "episode_count": int(len(group)),
                "overlap_anomaly_episode_count": int(
                    relation_counts.get("overlap_anomaly", 0)
                ),
                "overlap_normal_episode_count": int(
                    relation_counts.get("overlap_normal", 0)
                ),
                "background_episode_count": int(
                    relation_counts.get("background", 0)
                    + relation_counts.get("before_metadata", 0)
                    + relation_counts.get("after_metadata", 0)
                ),
                "total_duration_hours": float(
                    pd.to_numeric(
                        group["duration_hours"], errors="coerce"
                    ).fillna(0.0).sum()
                ),
            }
        )

    return pd.DataFrame(output_rows).sort_values(
        ["episode_count", "detector_family"],
        ascending=[False, True],
    ).reset_index(drop=True)


# =============================================================================
# 7. FIGURES
# =============================================================================

def save_confusion_matrix_figure(
    event_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    actual = event_table["actual_binary"].astype(int)
    tiers = [
        ("any_overlap_detected", "Any-overlap", "figure_1a_confusion_matrix_any_overlap.png"),
        ("meaningful_detected", "Meaningful", "figure_1b_confusion_matrix_meaningful.png"),
        ("operational_detected", "Operational", "figure_1c_confusion_matrix_operational.png"),
    ]

    for column, title, filename in tiers:
        predicted = event_table[column].astype(bool).astype(int)
        matrix = confusion_matrix(actual, predicted, labels=[0, 1])

        figure, axis = plt.subplots(figsize=(5.5, 5))
        image = axis.imshow(matrix)
        axis.set_xticks([0, 1], labels=["Normal", "Anomaly"])
        axis.set_yticks([0, 1], labels=["Normal", "Anomaly"])
        axis.set_xlabel("Predicted event label")
        axis.set_ylabel("Metadata event label")
        axis.set_title(f"{title} event-level confusion matrix")

        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                )

        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def save_delay_figure(
    event_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    anomaly = event_table.loc[
        (event_table["metadata_label"] == "anomaly")
        & event_table["event_detected"]
    ].copy()

    delays = pd.to_numeric(
        anomaly["detection_delay_hours"], errors="coerce"
    ).dropna()

    figure, axis = plt.subplots(figsize=(8, 5))
    if delays.empty:
        axis.text(
            0.5,
            0.5,
            "No detected anomaly events with valid delay",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        bins = min(20, max(5, int(math.sqrt(len(delays))) + 1))
        axis.hist(delays, bins=bins)
        axis.axvline(0.0, linestyle="--")
    axis.set_xlabel(
        "Detection delay (hours; negative means alert before metadata start)"
    )
    axis.set_ylabel("Detected anomaly events")
    axis.set_title("Detection-delay distribution")
    figure.tight_layout()
    figure.savefig(
        output_dir / "figure_2_detection_delay_distribution.png",
        dpi=180,
    )
    plt.close(figure)


def save_active_fraction_figure(
    event_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    normal = pd.to_numeric(
        event_table.loc[
            event_table["metadata_label"] == "normal",
            "active_fraction_inside_metadata",
        ],
        errors="coerce",
    ).dropna()

    anomaly = pd.to_numeric(
        event_table.loc[
            event_table["metadata_label"] == "anomaly",
            "active_fraction_inside_metadata",
        ],
        errors="coerce",
    ).dropna()

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(
        [normal.to_numpy(), anomaly.to_numpy()],
        labels=["Normal", "Anomaly"],
        showfliers=True,
    )
    axis.set_ylabel("Active-alert fraction inside metadata interval")
    axis.set_title("Alert coverage by metadata label")
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(
        output_dir / "figure_3_active_fraction_by_label.png",
        dpi=180,
    )
    plt.close(figure)


def save_episode_count_figure(
    event_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    ordered = event_table.sort_values(
        ["event_episode_count", "event_id"],
        ascending=[False, True],
    ).copy()

    figure, axis = plt.subplots(
        figsize=(max(10, len(ordered) * 0.22), 5)
    )
    axis.bar(
        ordered["event_id"].astype(str),
        ordered["event_episode_count"],
    )
    axis.set_xlabel("Event ID")
    axis.set_ylabel("Episodes detected in source file")
    axis.set_title("Streaming Episode count by Event source")
    axis.tick_params(axis="x", rotation=90)
    figure.tight_layout()
    figure.savefig(
        output_dir / "figure_4_episode_count_by_event.png",
        dpi=180,
    )
    plt.close(figure)


# =============================================================================
# 8. REPORT
# =============================================================================

def format_metric(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"
    return str(value)


def build_markdown_report(
    event_table: pd.DataFrame,
    metrics: dict[str, Any],
    missed: pd.DataFrame,
    false_positive: pd.DataFrame,
    high_burden: pd.DataFrame,
    detector_summary: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Streaming SCADA evaluation report")
    lines.append("")
    lines.append("## Overall Event-level performance")
    lines.append("")
    lines.append(
        f"- Events evaluated: **{metrics['event_count']}** "
        f"({metrics['anomaly_event_count']} anomaly, "
        f"{metrics['normal_event_count']} normal)"
    )
    lines.append(
        f"- Any-overlap recall: "
        f"**{format_metric(metrics['any_overlap_recall'])}**; "
        f"normal false-positive rate: "
        f"**{format_metric(metrics['any_overlap_false_positive_rate'])}**"
    )
    lines.append(
        f"- Meaningful recall: "
        f"**{format_metric(metrics['meaningful_recall'])}**; "
        f"normal false-positive rate: "
        f"**{format_metric(metrics['meaningful_false_positive_rate'])}**"
    )
    lines.append(
        f"- Operational recall: "
        f"**{format_metric(metrics['operational_recall'])}**; "
        f"normal false-positive rate: "
        f"**{format_metric(metrics['operational_false_positive_rate'])}**"
    )
    lines.append(
        f"- Normal Event false-positive rate: "
        f"**{format_metric(metrics['normal_event_false_positive_rate'])}**"
    )
    lines.append(
        f"- Balanced accuracy: "
        f"**{format_metric(metrics['event_level_balanced_accuracy'])}**"
    )
    lines.append(
        f"- Confusion matrix: TN={metrics['true_negative']}, "
        f"FP={metrics['false_positive']}, "
        f"FN={metrics['false_negative']}, "
        f"TP={metrics['true_positive']}"
    )
    lines.append(
        f"- Global active-row fraction: "
        f"**{format_metric(metrics['global_active_row_fraction'])}**"
    )
    lines.append(
        f"- Median detection delay across detected anomaly Events: "
        f"**{format_metric(metrics['median_detection_delay_hours_all_detected_anomalies'])} hours**"
    )
    lines.append(
        f"- Early-alert anomaly Events: "
        f"**{metrics['early_alert_anomaly_count']}**"
    )
    lines.append("")

    lines.append("## Missed anomaly Events")
    lines.append("")
    if missed.empty:
        lines.append("No anomaly Event was missed.")
    else:
        display_columns = [
            "event_id",
            "metadata_start",
            "metadata_end",
            "event_description",
            "active_fraction_total",
            "event_episode_count",
        ]
        lines.append(
            missed[display_columns]
            .to_markdown(index=False)
        )
    lines.append("")

    lines.append("## False-positive normal Events")
    lines.append("")
    if false_positive.empty:
        lines.append("No normal Event produced an overlapping active alert.")
    else:
        display_columns = [
            "event_id",
            "metadata_start",
            "metadata_end",
            "active_fraction_inside_metadata",
            "overlap_episode_count",
            "dominant_detector_family",
            "event_description",
        ]
        lines.append(
            false_positive[display_columns]
            .to_markdown(index=False)
        )
    lines.append("")

    lines.append("## Highest alert-burden Events")
    lines.append("")
    if high_burden.empty:
        lines.append("No high-burden Events were identified.")
    else:
        display_columns = [
            "event_id",
            "metadata_label",
            "active_fraction_total",
            "active_fraction_inside_metadata",
            "event_episode_count",
            "background_episode_count",
            "dominant_detector_family",
        ]
        lines.append(
            high_burden[display_columns]
            .head(15)
            .to_markdown(index=False)
        )
    lines.append("")

    lines.append("## Detector-family Episode summary")
    lines.append("")
    if detector_summary.empty:
        lines.append("No detector-family Episode data were available.")
    else:
        lines.append(
            detector_summary.head(20).to_markdown(index=False)
        )
    lines.append("")

    lines.append("## Recommended manual review order")
    lines.append("")
    lines.append(
        "1. Missed anomaly Events."
    )
    lines.append(
        "2. False-positive normal Events with the highest active fraction."
    )
    lines.append(
        "3. Anomaly Events with very negative delay, because the alert may "
        "represent background burden rather than a precise early warning."
    )
    lines.append(
        "4. Anomaly Events with long positive detection delay or low alert coverage."
    )
    lines.append(
        "5. A small set of representative successful Short, Persistent, "
        "Localized and multi-detector cases."
    )
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# 9. COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Farm event metadata CSV.",
    )
    parser.add_argument(
        "--stream-output-dir",
        type=Path,
        required=True,
        help="Output directory generated by the streaming replay script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for integrated evaluation outputs.",
    )
    parser.add_argument(
        "--farm",
        default="",
        help="Optional farm identifier written to the output table.",
    )
    parser.add_argument(
        "--pre-event-hours",
        type=float,
        default=168.0,
        help="Causal pre-event window used to quantify background alerts.",
    )
    parser.add_argument(
        "--high-active-fraction",
        type=float,
        default=0.25,
        help=(
            "Full-file active fraction used to flag a high alert-burden Event."
        ),
    )
    parser.add_argument(
        "--high-episode-count",
        type=int,
        default=20,
        help=(
            "Episode count used to flag a high alert-burden Event."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(args.metadata)
    if not args.stream_output_dir.exists():
        raise FileNotFoundError(args.stream_output_dir)
    if args.pre_event_hours < 0:
        raise ValueError("--pre-event-hours must be non-negative.")
    if not 0 <= args.high_active_fraction <= 1:
        raise ValueError(
            "--high-active-fraction must be between 0 and 1."
        )
    if args.high_episode_count < 0:
        raise ValueError("--high-episode-count must be non-negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(
        path=args.metadata,
        farm_id=args.farm or None,
    )
    episodes = load_combined_episodes(args.stream_output_dir)
    manifest = read_manifest(args.stream_output_dir)

    event_table = build_event_level_table(
        metadata=metadata,
        stream_output_dir=args.stream_output_dir,
        episodes=episodes,
        pre_event_hours=args.pre_event_hours,
    )

    episode_table = annotate_episode_table(
        episodes=episodes,
        metadata=metadata,
    )

    metrics = calculate_overall_metrics(
        event_table=event_table,
        episode_table=episode_table,
        manifest=manifest,
    )

    detector_summary = build_detector_family_summary(
        episode_table
    )

    missed_anomalies = event_table.loc[
        (event_table["metadata_label"] == "anomaly")
        & (~event_table["event_detected"])
    ].copy()

    false_positive_normals = event_table.loc[
        (event_table["metadata_label"] == "normal")
        & event_table["event_detected"]
    ].copy().sort_values(
        [
            "active_fraction_inside_metadata",
            "overlap_episode_count",
        ],
        ascending=[False, False],
    )

    high_burden = event_table.loc[
        (
            pd.to_numeric(
                event_table["active_fraction_total"],
                errors="coerce",
            ).fillna(0.0)
            >= args.high_active_fraction
        )
        | (
            pd.to_numeric(
                event_table["event_episode_count"],
                errors="coerce",
            ).fillna(0)
            >= args.high_episode_count
        )
    ].copy().sort_values(
        [
            "active_fraction_total",
            "event_episode_count",
        ],
        ascending=[False, False],
    )

    # Extra ranked diagnostic tables.
    anomaly_delay_ranking = event_table.loc[
        (event_table["metadata_label"] == "anomaly")
        & event_table["event_detected"]
    ].copy().sort_values(
        "detection_delay_hours",
        ascending=True,
    )

    low_coverage_anomalies = event_table.loc[
        event_table["metadata_label"] == "anomaly"
    ].copy().sort_values(
        [
            "active_fraction_inside_metadata",
            "detection_delay_hours",
        ],
        ascending=[True, False],
    )

    # Save tables.
    event_table.to_csv(
        args.output_dir / "event_level_streaming_evaluation.csv",
        index=False,
    )
    episode_table.to_csv(
        args.output_dir / "episode_level_streaming_evaluation.csv",
        index=False,
    )
    metrics_to_table(metrics).to_csv(
        args.output_dir / "overall_streaming_metrics.csv",
        index=False,
    )
    write_json(
        args.output_dir / "overall_streaming_metrics.json",
        metrics,
    )
    missed_anomalies.to_csv(
        args.output_dir / "missed_anomaly_events.csv",
        index=False,
    )
    false_positive_normals.to_csv(
        args.output_dir / "false_positive_normal_events.csv",
        index=False,
    )
    high_burden.to_csv(
        args.output_dir / "high_alert_burden_events.csv",
        index=False,
    )
    detector_summary.to_csv(
        args.output_dir / "detector_family_summary.csv",
        index=False,
    )
    anomaly_delay_ranking.to_csv(
        args.output_dir / "anomaly_detection_delay_ranking.csv",
        index=False,
    )
    low_coverage_anomalies.to_csv(
        args.output_dir / "low_coverage_anomaly_events.csv",
        index=False,
    )
    event_table.loc[
        (event_table["metadata_label"] == "anomaly")
        & (~event_table["meaningful_detected"])
    ].to_csv(
        args.output_dir / "missed_meaningful_anomaly_events.csv",
        index=False,
    )
    event_table.loc[
        (event_table["metadata_label"] == "anomaly")
        & (~event_table["operational_detected"])
    ].to_csv(
        args.output_dir / "missed_operational_anomaly_events.csv",
        index=False,
    )
    event_table.loc[
        (event_table["metadata_label"] == "normal")
        & event_table["meaningful_detected"]
    ].to_csv(
        args.output_dir / "meaningful_false_positive_normal_events.csv",
        index=False,
    )
    event_table.loc[
        (event_table["metadata_label"] == "normal")
        & event_table["operational_detected"]
    ].to_csv(
        args.output_dir / "operational_false_positive_normal_events.csv",
        index=False,
    )

    # Save figures.
    save_confusion_matrix_figure(event_table, args.output_dir)
    save_delay_figure(event_table, args.output_dir)
    save_active_fraction_figure(event_table, args.output_dir)
    save_episode_count_figure(event_table, args.output_dir)

    report = build_markdown_report(
        event_table=event_table,
        metrics=metrics,
        missed=missed_anomalies,
        false_positive=false_positive_normals,
        high_burden=high_burden,
        detector_summary=detector_summary,
    )
    with (
        args.output_dir / "streaming_evaluation_report.md"
    ).open("w", encoding="utf-8") as handle:
        handle.write(report)

    print("")
    print("Streaming evaluation completed.")
    print(
        f"Events: {metrics['event_count']} | "
        f"TP={metrics['true_positive']} "
        f"FN={metrics['false_negative']} "
        f"FP={metrics['false_positive']} "
        f"TN={metrics['true_negative']}"
    )
    print(
        "Any-overlap recall / FPR: "
        f"{format_metric(metrics['any_overlap_recall'])} / "
        f"{format_metric(metrics['any_overlap_false_positive_rate'])}"
    )
    print(
        "Meaningful recall / FPR: "
        f"{format_metric(metrics['meaningful_recall'])} / "
        f"{format_metric(metrics['meaningful_false_positive_rate'])}"
    )
    print(
        "Operational recall / FPR: "
        f"{format_metric(metrics['operational_recall'])} / "
        f"{format_metric(metrics['operational_false_positive_rate'])}"
    )
    print(
        "Normal Event false-positive rate: "
        f"{format_metric(metrics['normal_event_false_positive_rate'])}"
    )
    print(
        "Global active-row fraction: "
        f"{format_metric(metrics['global_active_row_fraction'])}"
    )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
