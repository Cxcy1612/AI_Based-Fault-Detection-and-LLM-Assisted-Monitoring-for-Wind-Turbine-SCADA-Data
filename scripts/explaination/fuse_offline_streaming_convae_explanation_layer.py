#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuse three complementary model-only evidence sources for logged analysis windows:

1. Offline event-level detection (V2)
2. Streaming time-localised detection (V5)
3. Convolutional Autoencoder early-warning evidence before the logged window

Part A uses deterministic rules to combine the three evidence sources into one
operator-facing status: anomaly, normal, or review_required.  The LLM is NOT
allowed to decide or override that status; it only explains the supplied evidence.

Part B remains streaming-only and unchanged in purpose: additional V5 review
candidates outside the logged Event window and its temporal buffer.

Ground-truth/maintenance semantics such as metadata labels, Event descriptions,
recorded diagnoses, failure types, and folder-name label suffixes are never
included in LLM payloads.

This script does NOT call an LLM. It prepares auditable structured payloads.

Recommended inputs
------------------
1. V2:
   ml_event_predictions.csv
2. V2 optional:
   ml_event_explanations.csv
3. V5:
   event_level_streaming_evaluation.csv
4. V5:
   all_stream_detected_episodes.csv

Example
-------
python fuse_v2_v5_explanation_layer.py ^
  --v2-predictions "outputs/event_level_ml/ml_event_predictions.csv" ^
  --v2-explanations "outputs/event_level_ml/ml_event_explanations.csv" ^
  --v5-event-evaluation "outputs/stream_eval/event_level_streaming_evaluation.csv" ^
  --v5-episodes "outputs/stream_replay/all_stream_detected_episodes.csv" ^
  --output-dir "outputs/v2_v5_explanation_fusion" ^
  --event-buffer-hours 24 ^
  --offlog-merge-gap-hours 6 ^
  --max-offlog-candidates 5 ^
  --minimum-offlog-priority 0.70
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


TRUE_TEXT = {"true", "1", "yes", "y", "t"}
FALSE_TEXT = {"false", "0", "no", "n", "f"}

# These fields may exist in source files but are intentionally excluded from
# all LLM payloads produced by this script.
PROHIBITED_LLM_FIELDS = {
    "metadata_label",
    "actual_label",
    "actual_binary",
    "event_label",
    "event_description",
    "description",
    "fault_description",
    "maintenance_action",
    "recorded_diagnosis",
    "failure_type",
    "component_name",
    "prediction_correct",
    "event_level_status",
}

DETECTOR_LABELS = {
    "fault_like_short_standstill":
        "a fault-like short shutdown followed by recovery",
    "intermittent_cluster":
        "repeated fault-like short shutdowns within a limited period",
    "persistent_system_state":
        "a sustained broad deviation across multiple signals",
    "localized_persistent_subsystem_state":
        "a sustained deviation affecting a smaller stable signal group",
    "communication_data_quality":
        "a communication or data-quality abnormality",
    "slow_trend":
        "a gradual multi-signal trend deviation",
    "status_code":
        "an unusual or rapidly changing status-code pattern",
    "communication_control_fault":
        "combined communication and control-state abnormality",
    "semantic_subsystem_fusion":
        "a subsystem-specific deviation supported by independent evidence",
    "targeted_subsystem_change":
        "a new high-severity change in a monitored subsystem",
    "unclassified_confirmed_episode":
        "confirmed alert evidence without a classified detector family",
}


@dataclass(frozen=True)
class FusionConfig:
    event_buffer_hours: float = 24.0
    operational_window_hours: float = 24.0
    meaningful_minimum_hours: float = 1.0
    meaningful_minimum_fraction: float = 0.05
    fragmented_minimum_episode_count: int = 3

    # Convolutional Autoencoder early-warning evidence for Part A.
    early_warning_lookback_days: float = 30.0
    early_warning_top_signals: int = 5

    offlog_merge_gap_hours: float = 6.0
    offlog_minimum_duration_hours: float = 1.0 / 3.0  # 20 minutes
    offlog_minimum_alert_score: float = 0.90
    offlog_minimum_priority: float = 0.70
    max_offlog_candidates: int = 5
    offlog_minimum_separation_hours: float = 24.0
    targeted_only_penalty: float = 0.65


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    # Most generated output files are comma-separated, but support semicolon
    # inputs to avoid fragile manual conversion.
    first = pd.read_csv(path, low_memory=False)
    if first.shape[1] > 1:
        first.columns = [str(c).strip() for c in first.columns]
        return first

    second = pd.read_csv(path, sep=";", low_memory=False)
    second.columns = [str(c).strip() for c in second.columns]
    return second


def normalise_event_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    return False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(json_ready(payload), ensure_ascii=False) + "\n"
            )


def split_detector_families(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    parts = re.split(r"[;,|]+", str(value))
    return {part.strip() for part in parts if part.strip()}


def detector_text(families: Iterable[str]) -> list[str]:
    return [
        DETECTOR_LABELS.get(name, name.replace("_", " "))
        for name in sorted(set(families))
    ]


def interval_overlap_hours(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> float:
    start = max(pd.Timestamp(start_a), pd.Timestamp(start_b))
    end = min(pd.Timestamp(end_a), pd.Timestamp(end_b))
    return max(0.0, (end - start).total_seconds() / 3600.0)


def union_interval_hours(
    intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    clean = sorted(
        [
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in intervals
            if pd.notna(start)
            and pd.notna(end)
            and pd.Timestamp(end) >= pd.Timestamp(start)
        ],
        key=lambda pair: (pair[0], pair[1]),
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
        sum((end - start).total_seconds() / 3600.0
            for start, end in merged)
    )


def nearest_interval_distance_hours(
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    if not intervals:
        return float("inf")

    distances: list[float] = []
    for other_start, other_end in intervals:
        if end >= other_start and start <= other_end:
            return 0.0
        if end < other_start:
            distances.append(
                (other_start - end).total_seconds() / 3600.0
            )
        else:
            distances.append(
                (start - other_end).total_seconds() / 3600.0
            )
    return float(min(distances))


def sanitise_for_llm(payload: Any) -> Any:
    """Recursively remove prohibited log/ground-truth fields."""
    if isinstance(payload, dict):
        return {
            key: sanitise_for_llm(value)
            for key, value in payload.items()
            if key not in PROHIBITED_LLM_FIELDS
        }
    if isinstance(payload, list):
        return [sanitise_for_llm(value) for value in payload]
    return payload


# =============================================================================
# LOAD AND STANDARDISE V2
# =============================================================================

def load_v2_predictions(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    required = {"event_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"V2 predictions missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["event_id"] = df["event_id"].map(normalise_event_id)

    if "metadata_start" in df.columns:
        df["metadata_start"] = pd.to_datetime(
            df["metadata_start"], errors="coerce"
        )

    # Canonical V2 names. Missing optional columns remain blank/NaN.
    aliases = {
        "v2_model": ["model"],
        "v2_predicted_label": ["predicted_label"],
        "v2_final_status": ["final_status"],
        "v2_vote_fraction": ["anomaly_vote_fraction"],
        "v2_mean_probability": ["mean_anomaly_probability"],
        "v2_prediction_stability": ["prediction_stability"],
        "v2_manual_review": ["manual_review"],
        "v2_cv_prediction_count": ["cv_prediction_count"],
        "v2_vote_margin_from_half": ["vote_margin_from_half"],
    }
    output = pd.DataFrame({"event_id": df["event_id"]})
    for target, candidates in aliases.items():
        source = next((c for c in candidates if c in df.columns), None)
        output[target] = df[source] if source else np.nan

    if "metadata_start" in df.columns:
        output["v2_metadata_start"] = df["metadata_start"]

    for column in [
        "v2_vote_fraction",
        "v2_mean_probability",
        "v2_cv_prediction_count",
        "v2_vote_margin_from_half",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output["v2_manual_review"] = output["v2_manual_review"].map(to_bool)
    return output.drop_duplicates("event_id", keep="last")


def load_v2_explanations(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["event_id"])
    df = read_csv_auto(path)
    if "event_id" not in df.columns:
        raise ValueError("V2 explanation file must contain event_id.")

    df = df.copy()
    df["event_id"] = df["event_id"].map(normalise_event_id)

    keep = ["event_id"]
    rename: dict[str, str] = {}
    for source, target in [
        ("main_supporting_features", "v2_supporting_features"),
        ("main_opposing_features", "v2_opposing_features"),
        ("explanation_method", "v2_explanation_method"),
    ]:
        if source in df.columns:
            keep.append(source)
            rename[source] = target

    return (
        df[keep]
        .rename(columns=rename)
        .drop_duplicates("event_id", keep="last")
    )


# =============================================================================
# LOAD AND STANDARDISE V5
# =============================================================================

def load_v5_event_evaluation(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    required = {"event_id", "metadata_start", "metadata_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"V5 event evaluation missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["event_id"] = df["event_id"].map(normalise_event_id)
    df["metadata_start"] = pd.to_datetime(
        df["metadata_start"], errors="coerce"
    )
    df["metadata_end"] = pd.to_datetime(
        df["metadata_end"], errors="coerce"
    )
    df = df.dropna(subset=["metadata_start", "metadata_end"])

    # Keep only model and time-window information. Ground-truth fields are
    # deliberately omitted from the returned table.
    columns = [
        "farm_id",
        "event_id",
        "metadata_start",
        "metadata_end",
        "any_overlap_detected",
        "meaningful_detected",
        "operational_detected",
        "event_episode_count",
        "overlap_episode_count",
        "background_episode_count",
        "first_overlapping_alert_time",
        "detection_delay_hours",
        "episode_overlap_union_hours",
        "episode_overlap_fraction",
        "dominant_detector_family",
        "overlap_detector_families",
        "active_fraction_inside_metadata",
        "review_fraction_total",
        "confirmed_rows_inside_metadata",
        "short_episode_starts_inside",
        "strong_short_trigger_rows_inside",
        "persistent_confirmed_rows_inside",
        "localized_candidate_rows_inside",
        "localized_confirmed_rows_inside",
        "semantic_fusion_confirmed_rows_inside",
        "targeted_change_trigger_rows_inside",
        "communication_control_confirmed_rows_inside",
    ]
    output = df[[c for c in columns if c in df.columns]].copy()

    for column in [
        "any_overlap_detected",
        "meaningful_detected",
        "operational_detected",
    ]:
        if column in output.columns:
            output[column] = output[column].map(to_bool)
        else:
            output[column] = False

    if "first_overlapping_alert_time" in output.columns:
        output["first_overlapping_alert_time"] = pd.to_datetime(
            output["first_overlapping_alert_time"], errors="coerce"
        )

    return output.drop_duplicates("event_id", keep="last")


def load_v5_episodes(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    if df.empty:
        return df

    required = {"start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"V5 episode table missing columns: {sorted(missing)}"
        )

    df = df.copy()
    for column in [
        "start_time",
        "end_time",
        "first_active_time",
        "last_evidence_time",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if "first_active_time" not in df.columns:
        df["first_active_time"] = df["start_time"]

    df = df.dropna(subset=["start_time", "end_time"])
    df = df.loc[df["end_time"] >= df["start_time"]].copy()

    if "source_id" in df.columns:
        df["source_id"] = df["source_id"].map(normalise_event_id)
    else:
        df["source_id"] = ""

    if "asset_id" not in df.columns:
        df["asset_id"] = ""
    if "farm_id" not in df.columns:
        df["farm_id"] = ""
    if "source_file" not in df.columns:
        df["source_file"] = ""
    if "episode_id" not in df.columns:
        df["episode_id"] = [
            f"episode_{index:06d}" for index in range(len(df))
        ]

    numeric_columns = [
        "duration_hours",
        "max_alert_score",
        "max_abnormal_fraction_z8",
        "max_abnormal_fraction_z12",
        "short_confirmed_count",
        "short_episode_count",
        "fault_like_short_episode_count",
        "intermittent_flag_points",
        "persistent_flag_points",
        "localized_candidate_points",
        "localized_confirmed_points",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    if "duration_hours" not in df.columns or (
        df["duration_hours"] <= 0
    ).all():
        df["duration_hours"] = (
            df["end_time"] - df["start_time"]
        ).dt.total_seconds() / 3600.0

    if "right_censored" not in df.columns:
        df["right_censored"] = False
    df["right_censored"] = df["right_censored"].map(to_bool)

    if "detector_families" not in df.columns:
        df["detector_families"] = ""

    return df.sort_values(
        ["farm_id", "asset_id", "source_id", "start_time", "end_time"]
    ).reset_index(drop=True)


# =============================================================================
# LOGGED-WINDOW V5 SUMMARY
# =============================================================================

def episode_relation(
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    buffer_hours: float,
) -> str:
    pre_start = window_start - pd.Timedelta(hours=buffer_hours)
    post_end = window_end + pd.Timedelta(hours=buffer_hours)

    if episode_end >= window_start and episode_start <= window_end:
        return "overlap"
    if episode_end < window_start and episode_end >= pre_start:
        return "pre_window"
    if episode_start > window_end and episode_start <= post_end:
        return "post_window"
    return "unrelated"


def select_primary_episode(
    related: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    cfg: FusionConfig,
) -> Optional[pd.Series]:
    if related.empty:
        return None

    candidates = related.copy()
    candidates["overlap_hours"] = candidates.apply(
        lambda row: interval_overlap_hours(
            row["start_time"],
            row["end_time"],
            window_start,
            window_end,
        ),
        axis=1,
    )
    window_hours = max(
        (window_end - window_start).total_seconds() / 3600.0,
        1e-9,
    )
    candidates["overlap_fraction"] = (
        candidates["overlap_hours"] / window_hours
    ).clip(0, 1)

    active_time = pd.to_datetime(
        candidates["first_active_time"], errors="coerce"
    ).fillna(candidates["start_time"])
    candidates["delay_hours"] = (
        active_time - window_start
    ).dt.total_seconds() / 3600.0

    candidates["meaningful"] = (
        (candidates["overlap_hours"] >= cfg.meaningful_minimum_hours)
        | (
            candidates["overlap_fraction"]
            >= cfg.meaningful_minimum_fraction
        )
    )
    candidates["operational"] = (
        candidates["meaningful"]
        & candidates["delay_hours"].between(
            -cfg.operational_window_hours,
            cfg.operational_window_hours,
            inclusive="both",
        )
    )

    relation_rank = {
        "overlap": 3,
        "pre_window": 2,
        "post_window": 1,
        "unrelated": 0,
    }
    candidates["relation_rank"] = candidates["relation"].map(
        relation_rank
    ).fillna(0)

    candidates = candidates.sort_values(
        [
            "operational",
            "meaningful",
            "relation_rank",
            "max_alert_score",
            "overlap_hours",
            "duration_hours",
        ],
        ascending=[False, False, False, False, False, False],
    )
    return candidates.iloc[0]


def build_v5_window_summary(
    event_row: pd.Series,
    episodes: pd.DataFrame,
    cfg: FusionConfig,
) -> dict[str, Any]:
    event_id = normalise_event_id(event_row["event_id"])
    start = pd.Timestamp(event_row["metadata_start"])
    end = pd.Timestamp(event_row["metadata_end"])
    farm_id = str(event_row.get("farm_id", ""))

    candidates = episodes.copy()
    if farm_id and "farm_id" in candidates.columns:
        same_farm = candidates["farm_id"].astype(str) == farm_id
        if same_farm.any():
            candidates = candidates.loc[same_farm].copy()

    # When replay outputs are organised per source Event, match the episode
    # table to the same source Event.  If the table contains non-empty source IDs
    # but none match this Event, do NOT fall back to a different Event merely
    # because its timestamps are nearby; different Event files may represent
    # different turbines.
    if "source_id" in candidates.columns and event_id:
        source_text = candidates["source_id"].map(normalise_event_id)
        same_source = source_text == event_id
        has_source_ids = (source_text != "").any()
        if same_source.any():
            candidates = candidates.loc[same_source].copy()
        elif has_source_ids:
            candidates = candidates.iloc[0:0].copy()

    candidates["relation"] = candidates.apply(
        lambda row: episode_relation(
            pd.Timestamp(row["start_time"]),
            pd.Timestamp(row["end_time"]),
            start,
            end,
            cfg.event_buffer_hours,
        ),
        axis=1,
    )
    related = candidates.loc[
        candidates["relation"] != "unrelated"
    ].copy()
    overlap = related.loc[related["relation"] == "overlap"].copy()
    pre = related.loc[related["relation"] == "pre_window"].copy()
    post = related.loc[related["relation"] == "post_window"].copy()

    overlap_intervals = [
        (
            max(pd.Timestamp(row["start_time"]), start),
            min(pd.Timestamp(row["end_time"]), end),
        )
        for _, row in overlap.iterrows()
    ]
    overlap_union_hours = union_interval_hours(overlap_intervals)
    event_hours = max((end - start).total_seconds() / 3600.0, 1e-9)
    overlap_fraction = min(1.0, overlap_union_hours / event_hours)

    primary = select_primary_episode(related, start, end, cfg)

    first_active = pd.NaT
    delay_hours = float("nan")
    if not overlap.empty:
        active_times = pd.to_datetime(
            overlap["first_active_time"], errors="coerce"
        ).dropna()
        if not active_times.empty:
            first_active = active_times.min()
            delay_hours = (
                first_active - start
            ).total_seconds() / 3600.0

    any_overlap = not overlap.empty
    meaningful = bool(
        any_overlap
        and (
            overlap_union_hours >= cfg.meaningful_minimum_hours
            or overlap_fraction >= cfg.meaningful_minimum_fraction
        )
    )
    operational = bool(
        meaningful
        and np.isfinite(delay_hours)
        and -cfg.operational_window_hours
        <= delay_hours
        <= cfg.operational_window_hours
    )

    family_counter: Counter[str] = Counter()
    for value in related["detector_families"]:
        family_counter.update(split_detector_families(value))

    all_families = sorted(family_counter)
    primary_families = (
        sorted(split_detector_families(primary["detector_families"]))
        if primary is not None
        else []
    )

    max_score = (
        safe_float(related["max_alert_score"].max())
        if not related.empty
        else 0.0
    )

    if operational and len(overlap) >= cfg.fragmented_minimum_episode_count:
        streaming_pattern = "operational_fragmented"
    elif operational:
        streaming_pattern = "operational_concentrated"
    elif meaningful:
        streaming_pattern = "meaningful_late_or_untimely"
    elif any_overlap:
        streaming_pattern = "weak_overlap_only"
    elif not pre.empty:
        streaming_pattern = "pre_window_alert_only"
    elif not post.empty:
        streaming_pattern = "post_window_alert_only"
    else:
        streaming_pattern = "no_qualifying_episode"

    reason: list[str] = []
    if streaming_pattern == "operational_fragmented":
        reason.append(
            "confirmed streaming evidence occurred near the start of the "
            "analysis window but was split across several episodes"
        )
    elif streaming_pattern == "operational_concentrated":
        reason.append(
            "a confirmed streaming episode occurred within the operational "
            "timing window"
        )
    elif streaming_pattern == "meaningful_late_or_untimely":
        reason.append(
            "meaningful streaming overlap was present, but the first alert "
            "did not satisfy the operational timing criterion"
        )
    elif streaming_pattern == "weak_overlap_only":
        reason.append(
            "an alert overlapped the analysis window, but the overlap was "
            "too short or too small to satisfy the meaningful criterion"
        )
    elif streaming_pattern == "pre_window_alert_only":
        reason.append(
            "the nearest confirmed streaming evidence occurred before the "
            "analysis window and did not remain active into it"
        )
    elif streaming_pattern == "post_window_alert_only":
        reason.append(
            "the nearest confirmed streaming evidence occurred after the "
            "analysis window"
        )
    else:
        reason.append(
            "no confirmed V5 episode was found inside the analysis window "
            "or its configured temporal buffer"
        )

    if primary_families:
        reason.append(
            "the primary episode was supported by: "
            + "; ".join(detector_text(primary_families))
        )
    if len(overlap) >= cfg.fragmented_minimum_episode_count:
        reason.append(
            f"{len(overlap)} overlapping episodes indicate fragmented "
            "rather than continuous evidence"
        )

    return {
        "event_id": event_id,
        "v5_streaming_pattern": streaming_pattern,
        "v5_any_overlap_detected": any_overlap,
        "v5_meaningful_detected": meaningful,
        "v5_operational_detected": operational,
        "v5_first_active_time": first_active,
        "v5_detection_delay_hours": delay_hours,
        "v5_related_episode_count": int(len(related)),
        "v5_overlap_episode_count": int(len(overlap)),
        "v5_pre_window_episode_count": int(len(pre)),
        "v5_post_window_episode_count": int(len(post)),
        "v5_overlap_union_hours": overlap_union_hours,
        "v5_overlap_fraction": overlap_fraction,
        "v5_max_alert_score": max_score,
        "v5_primary_episode_id":
            str(primary["episode_id"]) if primary is not None else "",
        "v5_primary_episode_start":
            primary["start_time"] if primary is not None else pd.NaT,
        "v5_primary_episode_end":
            primary["end_time"] if primary is not None else pd.NaT,
        "v5_primary_detector_families":
            ";".join(primary_families),
        "v5_all_related_detector_families":
            ";".join(all_families),
        "v5_reason": " | ".join(reason),
    }


# =============================================================================
# DETERMINISTIC V2/V5 RELATIONSHIP
# =============================================================================

def classify_evidence_relationship(row: pd.Series) -> tuple[str, str]:
    v2_status = str(row.get("v2_final_status", "")).strip().lower()
    v2_label = str(row.get("v2_predicted_label", "")).strip().lower()
    v5_pattern = str(row.get("v5_streaming_pattern", "")).strip().lower()
    operational = to_bool(row.get("v5_operational_detected", False))
    meaningful = to_bool(row.get("v5_meaningful_detected", False))
    any_overlap = to_bool(row.get("v5_any_overlap_detected", False))

    v2_anomaly = v2_status == "anomaly" or (
        not v2_status and v2_label == "anomaly"
    )
    v2_normal = v2_status == "normal" or (
        not v2_status and v2_label == "normal"
    )
    v2_review = v2_status == "review_required"

    if v2_anomaly and operational:
        return (
            "dual_support",
            "V2 supports an Event-wide abnormal pattern and V5 provides "
            "timely confirmed streaming evidence.",
        )
    if v2_anomaly and meaningful:
        return (
            "partial_dual_support",
            "Both systems provide abnormal evidence, but the V5 evidence "
            "is late or does not satisfy the operational timing criterion.",
        )
    if v2_anomaly and not any_overlap:
        return (
            "offline_only_support",
            "V2 supports an aggregate abnormal pattern, while V5 does not "
            "locate a qualifying confirmed episode near the window.",
        )
    if v2_review and operational:
        return (
            "streaming_supports_offline_review",
            "V2 remains uncertain, while V5 provides timely confirmed "
            "streaming evidence.",
        )
    if v2_review and (meaningful or any_overlap):
        return (
            "jointly_inconclusive_with_temporal_support",
            "V2 is uncertain and V5 provides limited or untimely temporal "
            "support.",
        )
    if v2_review:
        return (
            "jointly_inconclusive",
            "V2 is uncertain and V5 does not provide strong confirmed "
            "temporal support.",
        )
    if v2_normal and operational:
        return (
            "streaming_only_support",
            "V5 identifies a timely localised alert, while V2 classifies "
            "the complete analysis window as normal.",
        )
    if v2_normal and (meaningful or any_overlap):
        return (
            "weak_streaming_only_support",
            "V2 classifies the complete window as normal, while V5 provides "
            "limited or untimely localised evidence.",
        )
    if v2_normal and not any_overlap:
        return (
            "consistent_low_model_evidence",
            "Neither system provides strong abnormal evidence for this "
            "analysis window.",
        )
    return (
        "insufficient_model_output",
        "One or both model outputs are unavailable or incomplete.",
    )


def build_v2_reason(row: pd.Series) -> str:
    parts: list[str] = []

    status = str(row.get("v2_final_status", "")).strip()
    label = str(row.get("v2_predicted_label", "")).strip()
    vote = row.get("v2_vote_fraction", np.nan)
    stability = str(row.get("v2_prediction_stability", "")).strip()

    if status:
        parts.append(f"V2 final status was {status}")
    elif label:
        parts.append(f"V2 binary prediction was {label}")

    if pd.notna(vote):
        parts.append(
            f"{safe_float(vote):.2f} of repeated outer-fold predictions "
            "voted for anomaly"
        )
    if stability:
        parts.append(f"prediction stability was {stability}")

    support = str(row.get("v2_supporting_features", "")).strip()
    oppose = str(row.get("v2_opposing_features", "")).strip()
    if support and support.lower() not in {"nan", "none"}:
        parts.append("reported supporting evidence: " + support)
    if oppose and oppose.lower() not in {"nan", "none"}:
        parts.append("reported opposing evidence: " + oppose)

    return " | ".join(parts)



# =============================================================================
# CONVOLUTIONAL AUTOENCODER EARLY-WARNING EVIDENCE
# =============================================================================

def first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    """Return the first candidate column that exists, case-insensitively."""
    if df is None or df.empty and not len(df.columns):
        return None
    lookup = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        found = lookup.get(str(candidate).strip().lower())
        if found is not None:
            return found
    return None


def load_convae_event_index(root: Path) -> pd.DataFrame:
    """
    Load only a safe Event index from all_turbines_event_results.csv.

    The source table may contain ground-truth evaluation columns.  Those columns
    are intentionally NOT propagated.  The table is used only to confirm that a
    ConvAE result exists for an Event and, when available, to retain safe model
    metadata such as asset_id.
    """
    path = root / "all_turbines_event_results.csv"
    df = read_csv_auto(path)

    event_column = first_existing_column(
        df,
        ["event_id", "event", "source_id"],
    )
    if event_column is None:
        raise ValueError(
            "ConvAE all_turbines_event_results.csv must contain an Event ID column."
        )

    output = pd.DataFrame(
        {
            "event_id": df[event_column].map(normalise_event_id),
            "cae_event_result_available": True,
        }
    )

    asset_column = first_existing_column(
        df,
        ["asset_id", "turbine_id", "asset", "turbine"],
    )
    if asset_column is not None:
        output["cae_asset_id"] = df[asset_column].astype(str)

    # Safe audit-only model metadata.  These fields are not required by the LLM.
    for target, aliases in {
        "cae_validation_far": [
            "validation_confirmed_window_far",
            "validation_far",
            "val_far",
        ],
        "cae_global_threshold": [
            "global_threshold",
            "selected_threshold",
            "threshold",
        ],
        "cae_selected_feature_count": [
            "selected_feature_count",
            "n_selected_features",
            "feature_count",
        ],
    }.items():
        source = first_existing_column(df, aliases)
        if source is not None:
            output[target] = pd.to_numeric(df[source], errors="coerce")

    return (
        output.loc[output["event_id"] != ""]
        .drop_duplicates("event_id", keep="last")
        .reset_index(drop=True)
    )


def load_feature_description_map(root: Path) -> dict[str, str]:
    """Load raw-feature -> human-readable description mapping."""
    path = root / "parsed_feature_description.csv"
    if not path.exists():
        return {}

    df = read_csv_auto(path)
    if df.empty:
        return {}

    feature_column = first_existing_column(
        df,
        [
            "feature",
            "feature_name",
            "sensor",
            "sensor_name",
            "variable",
            "column",
            "raw_feature",
            "scada_feature",
            "name",
        ],
    )
    description_column = first_existing_column(
        df,
        [
            "description",
            "feature_description",
            "sensor_description",
            "parsed_description",
            "meaning",
            "human_readable_description",
            "label",
        ],
    )

    if feature_column is None or description_column is None:
        return {}

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        feature = str(row.get(feature_column, "")).strip()
        description = str(row.get(description_column, "")).strip()
        if (
            feature
            and description
            and feature.lower() not in {"nan", "none"}
            and description.lower() not in {"nan", "none"}
        ):
            mapping[feature] = description
    return mapping


def find_convae_event_dir(root: Path, event_id: str) -> Optional[Path]:
    """
    Find event_<id>_* without exposing the folder name to downstream payloads.

    Folder names contain _normal/_anomaly suffixes in the current experiment
    output.  Those suffixes are ground-truth semantics and are used only as an
    internal filesystem locator, never as model evidence.
    """
    event_id = normalise_event_id(event_id)
    if not event_id:
        return None

    pattern = re.compile(rf"^event_{re.escape(event_id)}(?:_|$)", re.IGNORECASE)
    matches = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and pattern.match(path.name)
        ],
        key=lambda path: path.name.lower(),
    )
    return matches[0] if matches else None


def load_event_description_fallback(event_dir: Path) -> dict[str, str]:
    """
    Use feature_ranking_with_descriptions.csv only as a description fallback.

    Its ranking is NOT treated as causal or Event-specific anomaly attribution.
    """
    path = event_dir / "feature_ranking_with_descriptions.csv"
    if not path.exists():
        return {}

    df = read_csv_auto(path)
    if df.empty:
        return {}

    feature_column = first_existing_column(
        df,
        [
            "feature",
            "feature_name",
            "sensor",
            "sensor_name",
            "variable",
            "column",
            "name",
        ],
    )
    description_column = first_existing_column(
        df,
        [
            "description",
            "feature_description",
            "sensor_description",
            "parsed_description",
            "meaning",
            "human_readable_description",
        ],
    )
    if feature_column is None or description_column is None:
        return {}

    output: dict[str, str] = {}
    for _, row in df.iterrows():
        feature = str(row.get(feature_column, "")).strip()
        description = str(row.get(description_column, "")).strip()
        if (
            feature
            and description
            and feature.lower() not in {"nan", "none"}
            and description.lower() not in {"nan", "none"}
        ):
            output[feature] = description
    return output


def normalise_episode_table(path: Path) -> pd.DataFrame:
    """
    Load a ConvAE warning_episodes.csv with flexible time-column aliases.

    warning_episodes.csv is already the retained/confirmed episode table from the
    ConvAE experiment, so no ground-truth label is needed to decide whether an
    episode is model-generated warning evidence.
    """
    if not path.exists():
        return pd.DataFrame(columns=["start_time", "end_time"])

    df = read_csv_auto(path)
    if df.empty:
        return pd.DataFrame(columns=["start_time", "end_time"])

    start_column = first_existing_column(
        df,
        [
            "start_time",
            "episode_start",
            "warning_start",
            "episode_start_time",
            "start",
            "first_active_time",
        ],
    )
    end_column = first_existing_column(
        df,
        [
            "end_time",
            "episode_end",
            "warning_end",
            "episode_end_time",
            "end",
            "last_evidence_time",
        ],
    )

    if start_column is None:
        raise ValueError(
            f"{path.name} does not contain a recognised warning start-time column."
        )

    output = pd.DataFrame()
    output["start_time"] = pd.to_datetime(df[start_column], errors="coerce")

    if end_column is not None:
        output["end_time"] = pd.to_datetime(df[end_column], errors="coerce")
    else:
        output["end_time"] = output["start_time"]

    duration_column = first_existing_column(
        df,
        ["duration_hours", "episode_duration_hours", "duration"],
    )
    if duration_column is not None:
        output["reported_duration_hours"] = pd.to_numeric(
            df[duration_column], errors="coerce"
        )

    score_column = first_existing_column(
        df,
        [
            "max_score",
            "maximum_score",
            "max_alert_score",
            "max_reconstruction_score",
            "max_smoothed_score",
            "peak_score",
        ],
    )
    if score_column is not None:
        output["episode_max_score"] = pd.to_numeric(
            df[score_column], errors="coerce"
        )

    output = output.dropna(subset=["start_time"]).copy()
    output["end_time"] = output["end_time"].fillna(output["start_time"])
    output.loc[
        output["end_time"] < output["start_time"], "end_time"
    ] = output.loc[
        output["end_time"] < output["start_time"], "start_time"
    ]

    return output.sort_values(["start_time", "end_time"]).reset_index(drop=True)


def interval_overlaps_any(
    timestamp: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    return any(start <= timestamp <= end for start, end in intervals)


def describe_top_reconstruction_signals(
    event_dir: Path,
    pre_warning_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    description_map: dict[str, str],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Summarise the strongest reconstruction-error signals during pre-window
    warning episodes.

    Only human-readable descriptions are returned.  Raw sensor names are never
    included in the LLM payload.  These signals are supporting reconstruction
    evidence, not causal fault attribution.
    """
    if not pre_warning_intervals or top_k <= 0:
        return []

    path = event_dir / "sensor_reconstruction_errors.csv"
    if not path.exists():
        return []

    df = read_csv_auto(path)
    if df.empty:
        return []

    time_column = first_existing_column(
        df,
        [
            "window_end",
            "time_stamp",
            "timestamp",
            "time",
            "end_time",
            "window_time",
        ],
    )
    if time_column is None:
        # Without time alignment, using the whole error table could mix pre-window
        # and in-window/post-window evidence, so skip signal attribution.
        return []

    times = pd.to_datetime(df[time_column], errors="coerce")
    mask = times.map(
        lambda value: (
            False
            if pd.isna(value)
            else interval_overlaps_any(pd.Timestamp(value), pre_warning_intervals)
        )
    )
    subset = df.loc[mask].copy()
    if subset.empty:
        return []

    excluded = {
        time_column,
        "event_id",
        "asset_id",
        "label",
        "event_label",
        "metadata_label",
    }
    numeric_candidates: list[str] = []
    for column in subset.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(subset[column], errors="coerce")
        if converted.notna().any():
            subset[column] = converted
            numeric_candidates.append(column)

    if not numeric_candidates:
        return []

    ranking = []
    fallback = load_event_description_fallback(event_dir)

    for feature in numeric_candidates:
        series = pd.to_numeric(subset[feature], errors="coerce").dropna()
        if series.empty:
            continue

        description = (
            description_map.get(feature)
            or fallback.get(feature)
            or ""
        )
        description = str(description).strip()
        if not description or description.lower() in {"nan", "none"}:
            # Do not expose raw feature names merely because a description is absent.
            continue

        ranking.append(
            {
                "signal_description": description,
                "mean_reconstruction_error": float(series.mean()),
                "maximum_reconstruction_error": float(series.max()),
                "supporting_window_count": int(series.notna().sum()),
            }
        )

    ranking.sort(
        key=lambda item: (
            item["mean_reconstruction_error"],
            item["maximum_reconstruction_error"],
        ),
        reverse=True,
    )

    # Remove repeated descriptions while preserving score order.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranking:
        key = item["signal_description"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= top_k:
            break

    return unique


def build_convae_window_summary(
    event_row: pd.Series,
    convae_root: Path,
    description_map: dict[str, str],
    cfg: FusionConfig,
) -> dict[str, Any]:
    """
    Recompute model-only pre-window early-warning evidence from warning episodes.

    This deliberately uses only the Event ID and analysis-window start time.
    It does not use the Event's normal/anomaly label.  Therefore the same rule is
    applied to both logged anomaly and logged normal windows.
    """
    event_id = normalise_event_id(event_row["event_id"])
    event_start = pd.Timestamp(event_row["metadata_start"])
    lookback_start = event_start - pd.Timedelta(
        days=float(cfg.early_warning_lookback_days)
    )

    event_dir = find_convae_event_dir(convae_root, event_id)
    if event_dir is None:
        return {
            "event_id": event_id,
            "cae_evidence_available": False,
            "cae_pre_window_warning_detected": False,
            "cae_warning_within_30d": False,
            "cae_warning_within_14d": False,
            "cae_warning_within_7d": False,
            "cae_earliest_warning_time": pd.NaT,
            "cae_latest_warning_time": pd.NaT,
            "cae_earliest_lead_time_days": np.nan,
            "cae_closest_lead_time_days": np.nan,
            "cae_pre_window_episode_count": 0,
            "cae_total_pre_window_warning_hours": 0.0,
            "cae_main_signal_evidence": "[]",
            "cae_reason": (
                "no matching Convolutional Autoencoder Event output folder was found"
            ),
        }

    warning_path = event_dir / "warning_episodes.csv"
    if not warning_path.exists():
        return {
            "event_id": event_id,
            "cae_evidence_available": False,
            "cae_pre_window_warning_detected": False,
            "cae_warning_within_30d": False,
            "cae_warning_within_14d": False,
            "cae_warning_within_7d": False,
            "cae_earliest_warning_time": pd.NaT,
            "cae_latest_warning_time": pd.NaT,
            "cae_earliest_lead_time_days": np.nan,
            "cae_closest_lead_time_days": np.nan,
            "cae_pre_window_episode_count": 0,
            "cae_total_pre_window_warning_hours": 0.0,
            "cae_main_signal_evidence": "[]",
            "cae_reason": (
                "the matching Convolutional Autoencoder Event output exists, "
                "but warning_episodes.csv is unavailable"
            ),
        }

    episodes = normalise_episode_table(warning_path)

    # Keep any confirmed warning episode that overlaps the configured pre-window
    # lookback interval.  Clip it at Event start so no in-window information is
    # accidentally counted as early-warning evidence.
    selected_rows: list[dict[str, Any]] = []
    for _, row in episodes.iterrows():
        start = pd.Timestamp(row["start_time"])
        end = pd.Timestamp(row["end_time"])

        if end < lookback_start or start >= event_start:
            continue

        clipped_start = max(start, lookback_start)
        clipped_end = min(end, event_start)
        if clipped_end < clipped_start:
            continue

        selected_rows.append(
            {
                "start_time": clipped_start,
                "end_time": clipped_end,
                "duration_hours": max(
                    0.0,
                    (clipped_end - clipped_start).total_seconds() / 3600.0,
                ),
            }
        )

    selected = pd.DataFrame(selected_rows)

    detected = not selected.empty
    if detected:
        intervals = [
            (pd.Timestamp(row["start_time"]), pd.Timestamp(row["end_time"]))
            for _, row in selected.iterrows()
        ]
        earliest = min(start for start, _ in intervals)
        latest = max(end for _, end in intervals)
        total_hours = union_interval_hours(intervals)

        earliest_lead_days = (
            event_start - earliest
        ).total_seconds() / 86400.0
        closest_lead_days = max(
            0.0,
            min(
                (event_start - end).total_seconds() / 86400.0
                for _, end in intervals
            ),
        )

        def within(days: float) -> bool:
            horizon_start = event_start - pd.Timedelta(days=days)
            return any(
                end >= horizon_start and start < event_start
                for start, end in intervals
            )

        warning_30 = within(min(30.0, cfg.early_warning_lookback_days))
        warning_14 = within(14.0)
        warning_7 = within(7.0)

        signal_evidence = describe_top_reconstruction_signals(
            event_dir=event_dir,
            pre_warning_intervals=intervals,
            description_map=description_map,
            top_k=cfg.early_warning_top_signals,
        )

        reason_parts = [
            f"{len(intervals)} confirmed early-warning episode(s) occurred "
            f"within the {cfg.early_warning_lookback_days:g}-day pre-window period",
            f"the earliest evidence began about {earliest_lead_days:.2f} days "
            "before the analysis window",
            f"the combined pre-window warning duration was {total_hours:.2f} hours",
        ]
        if signal_evidence:
            reason_parts.append(
                "the strongest reconstruction-error support was concentrated in "
                + "; ".join(
                    item["signal_description"] for item in signal_evidence
                )
            )

        return {
            "event_id": event_id,
            "cae_evidence_available": True,
            "cae_pre_window_warning_detected": True,
            "cae_warning_within_30d": bool(warning_30),
            "cae_warning_within_14d": bool(warning_14),
            "cae_warning_within_7d": bool(warning_7),
            "cae_earliest_warning_time": earliest,
            "cae_latest_warning_time": latest,
            "cae_earliest_lead_time_days": float(earliest_lead_days),
            "cae_closest_lead_time_days": float(closest_lead_days),
            "cae_pre_window_episode_count": int(len(intervals)),
            "cae_total_pre_window_warning_hours": float(total_hours),
            "cae_main_signal_evidence": json.dumps(
                json_ready(signal_evidence), ensure_ascii=False
            ),
            "cae_reason": " | ".join(reason_parts),
        }

    return {
        "event_id": event_id,
        "cae_evidence_available": True,
        "cae_pre_window_warning_detected": False,
        "cae_warning_within_30d": False,
        "cae_warning_within_14d": False,
        "cae_warning_within_7d": False,
        "cae_earliest_warning_time": pd.NaT,
        "cae_latest_warning_time": pd.NaT,
        "cae_earliest_lead_time_days": np.nan,
        "cae_closest_lead_time_days": np.nan,
        "cae_pre_window_episode_count": 0,
        "cae_total_pre_window_warning_hours": 0.0,
        "cae_main_signal_evidence": "[]",
        "cae_reason": (
            f"no confirmed Convolutional Autoencoder warning episode occurred "
            f"within the {cfg.early_warning_lookback_days:g}-day period before "
            "the analysis window"
        ),
    }


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return []
    return result if isinstance(result, list) else []


def classify_three_source_fusion(
    row: pd.Series,
) -> tuple[str, str, str]:
    """
    Deterministically combine offline, streaming, and early-warning evidence.

    Important asymmetry:
    - Streaming/early-warning alerts are positive abnormal evidence.
    - Absence of a streaming or early-warning alert is NOT proof of normality.
    - An integrated 'normal' status therefore requires an offline-normal result
      plus available low evidence from both temporal detectors.
    """
    v2_status = str(row.get("v2_final_status", "")).strip().lower()
    v2_label = str(row.get("v2_predicted_label", "")).strip().lower()

    offline_anomaly = v2_status == "anomaly" or (
        not v2_status and v2_label == "anomaly"
    )
    offline_normal = v2_status == "normal" or (
        not v2_status and v2_label == "normal"
    )
    offline_review = v2_status == "review_required"

    streaming_operational = to_bool(
        row.get("v5_operational_detected", False)
    )
    streaming_meaningful = to_bool(
        row.get("v5_meaningful_detected", False)
    )
    streaming_overlap = to_bool(
        row.get("v5_any_overlap_detected", False)
    )
    streaming_pattern = str(
        row.get("v5_streaming_pattern", "")
    ).strip().lower()
    streaming_weak_temporal = streaming_pattern in {
        "pre_window_alert_only",
        "post_window_alert_only",
        "weak_overlap_only",
        "meaningful_late_or_untimely",
    }
    streaming_any_abnormal = bool(
        streaming_operational
        or streaming_meaningful
        or streaming_overlap
        or streaming_weak_temporal
    )

    early_available = to_bool(
        row.get("cae_evidence_available", False)
    )
    early_warning = to_bool(
        row.get("cae_pre_window_warning_detected", False)
    )

    # Strongest integrated abnormal status: offline Event-level abnormality plus
    # independent temporal support from streaming and/or early warning.
    if offline_anomaly and (
        streaming_operational
        or streaming_meaningful
        or early_warning
    ):
        supporters = []
        if streaming_operational:
            supporters.append("timely streaming evidence")
        elif streaming_meaningful:
            supporters.append("meaningful streaming evidence")
        if early_warning:
            supporters.append("pre-window early-warning evidence")

        return (
            "anomaly",
            "multi_source_abnormal_support",
            "Offline detection identifies an Event-wide abnormal pattern and "
            + " and ".join(supporters)
            + " provides independent temporal support. The integrated status is "
              "therefore anomaly, while the evidence remains a model-based "
              "monitoring assessment rather than a confirmed physical diagnosis.",
        )

    # Offline anomaly without meaningful/timely independent support is retained as
    # a disagreement that deserves review rather than being silently downgraded.
    if offline_anomaly:
        return (
            "review_required",
            "offline_abnormal_without_independent_confirmation",
            "Offline detection identifies an Event-wide abnormal pattern, but "
            "streaming detection does not provide meaningful/timely confirmation "
            "and the early-warning detector does not provide pre-window support "
            "or is unavailable. The disagreement should be reviewed manually.",
        )

    # If offline is normal but either temporal detector reports abnormal evidence,
    # do not use a 2-of-3 vote.  The sources answer different temporal questions,
    # so the conflict is explicitly escalated to review.
    if offline_normal and (streaming_any_abnormal or early_warning):
        evidence = []
        if streaming_any_abnormal:
            evidence.append("streaming detection")
        if early_warning:
            evidence.append("early-warning detection")
        return (
            "review_required",
            "temporal_evidence_conflicts_with_offline_normal",
            "Offline detection classifies the complete analysis window as normal, "
            "while "
            + " and ".join(evidence)
            + " provides abnormal temporal evidence. Because the approaches assess "
              "different temporal aspects, the conflict is escalated to review "
              "rather than resolved by majority voting.",
        )

    # Normal is deliberately conservative: the Event-wide detector must be normal,
    # both temporal evidence sources must be available, and neither may report
    # abnormal evidence.
    if (
        offline_normal
        and not streaming_any_abnormal
        and early_available
        and not early_warning
    ):
        return (
            "normal",
            "consistent_low_model_evidence",
            "Offline detection classifies the complete analysis window as normal, "
            "streaming detection finds no qualifying abnormal episode, and the "
            "early-warning detector finds no confirmed warning in the configured "
            "pre-window period. This is a low-model-evidence normal assessment, "
            "not proof that the turbine is physically fault-free.",
        )

    if offline_review:
        if streaming_any_abnormal or early_warning:
            return (
                "review_required",
                "offline_uncertain_with_temporal_support",
                "Offline detection remains uncertain while one or both temporal "
                "detectors provide abnormal evidence. The integrated status remains "
                "review_required so that the additional temporal evidence can guide "
                "manual inspection.",
            )
        return (
            "review_required",
            "offline_uncertain_without_strong_temporal_support",
            "Offline detection remains uncertain and the temporal detectors do not "
            "provide strong abnormal support. The integrated result remains "
            "review_required rather than being treated as normal.",
        )

    # Missing ConvAE evidence blocks a confident integrated normal assessment.
    if offline_normal and not early_available:
        return (
            "review_required",
            "early_warning_evidence_unavailable",
            "Offline detection is normal and streaming evidence is low, but the "
            "early-warning evidence is unavailable. The three-source fusion is "
            "therefore incomplete and requires review.",
        )

    return (
        "review_required",
        "insufficient_three_source_output",
        "One or more required model outputs are unavailable or incomplete, so the "
        "three-source fusion cannot make a complete low-evidence or abnormal "
        "assessment.",
    )



# =============================================================================
# OFF-LOG CANDIDATE DISCOVERY
# =============================================================================

def build_expanded_logged_intervals(
    events: pd.DataFrame,
    buffer_hours: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals = sorted(
        [
            (
                pd.Timestamp(row["metadata_start"])
                - pd.Timedelta(hours=buffer_hours),
                pd.Timestamp(row["metadata_end"])
                + pd.Timedelta(hours=buffer_hours),
            )
            for _, row in events.iterrows()
        ],
        key=lambda pair: (pair[0], pair[1]),
    )
    if not intervals:
        return []

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def is_outside_intervals(
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    return all(end < other_start or start > other_end
               for other_start, other_end in intervals)


def find_source_event_context(
    source_event_id: str,
    candidate_start: pd.Timestamp,
    candidate_end: pd.Timestamp,
    events: pd.DataFrame,
    buffer_hours: float,
) -> dict[str, Any]:
    """Compare a candidate only with its own source Event window.

    Different source Event files may represent different turbines, so the
    candidate must never be matched to the globally nearest Event by time.
    """
    source_event_id = normalise_event_id(source_event_id)
    matched = events.loc[events["event_id"].map(normalise_event_id) == source_event_id]

    if not source_event_id or matched.empty:
        return {
            "source_logged_event_id": source_event_id,
            "source_logged_event_start": pd.NaT,
            "source_logged_event_end": pd.NaT,
            "distance_to_source_event_window_hours": float("nan"),
            "position_relative_to_source_event": "source_event_window_unavailable",
            "inside_source_event_window": False,
            "inside_source_event_buffer": False,
        }

    event = matched.iloc[-1]
    event_start = pd.Timestamp(event["metadata_start"])
    event_end = pd.Timestamp(event["metadata_end"])
    expanded_start = event_start - pd.Timedelta(hours=buffer_hours)
    expanded_end = event_end + pd.Timedelta(hours=buffer_hours)

    inside_window = bool(candidate_end >= event_start and candidate_start <= event_end)
    inside_buffer = bool(candidate_end >= expanded_start and candidate_start <= expanded_end)

    if candidate_end < event_start:
        distance = (event_start - candidate_end).total_seconds() / 3600.0
        position = "before_source_event"
    elif candidate_start > event_end:
        distance = (candidate_start - event_end).total_seconds() / 3600.0
        position = "after_source_event"
    else:
        distance = 0.0
        position = "overlapping_source_event"

    return {
        "source_logged_event_id": source_event_id,
        "source_logged_event_start": event_start,
        "source_logged_event_end": event_end,
        "distance_to_source_event_window_hours": float(distance),
        "position_relative_to_source_event": position,
        "inside_source_event_window": inside_window,
        "inside_source_event_buffer": inside_buffer,
    }


def episode_is_offlog_for_own_source(
    row: pd.Series,
    events: pd.DataFrame,
    buffer_hours: float,
) -> bool:
    """Return True only when an episode is outside its own Event buffer.

    Episodes without a valid source_id-to-Event match are not promoted as
    off-log candidates because their turbine/Event context is ambiguous.
    """
    source_id = normalise_event_id(row.get("source_id", ""))
    context = find_source_event_context(
        source_id,
        pd.Timestamp(row["start_time"]),
        pd.Timestamp(row["end_time"]),
        events,
        buffer_hours,
    )
    if context["position_relative_to_source_event"] == "source_event_window_unavailable":
        return False
    return not context["inside_source_event_buffer"]


def can_merge_offlog(
    current: dict[str, Any],
    row: pd.Series,
    cfg: FusionConfig,
) -> bool:
    same_farm = str(current["farm_id"]) == str(row.get("farm_id", ""))
    same_asset = str(current["asset_id"]) == str(row.get("asset_id", ""))
    row_source_id = normalise_event_id(row.get("source_id", ""))
    same_source = row_source_id in set(current.get("source_ids", set()))
    gap_hours = (
        pd.Timestamp(row["start_time"])
        - pd.Timestamp(current["end_time"])
    ).total_seconds() / 3600.0

    current_families = set(current["families"])
    row_families = split_detector_families(row["detector_families"])
    family_compatible = bool(current_families & row_families)

    # Empty asset identifiers are common in some exports. Time and shared
    # detector family are then still required.
    return bool(
        same_farm
        and same_asset
        and same_source
        and 0 <= gap_hours <= cfg.offlog_merge_gap_hours
        and family_compatible
    )


def merge_offlog_episodes(
    episodes: pd.DataFrame,
    cfg: FusionConfig,
) -> list[dict[str, Any]]:
    if episodes.empty:
        return []

    ordered = episodes.sort_values(
        ["farm_id", "asset_id", "start_time", "end_time"]
    )
    merged: list[dict[str, Any]] = []

    for _, row in ordered.iterrows():
        families = split_detector_families(row["detector_families"])
        source_id = normalise_event_id(row.get("source_id", ""))
        source_file = str(row.get("source_file", "")).strip()
        item = {
            "farm_id": str(row.get("farm_id", "")),
            "asset_id": str(row.get("asset_id", "")),
            "source_ids": {source_id} if source_id else set(),
            "source_files": {source_file} if source_file else set(),
            "start_time": pd.Timestamp(row["start_time"]),
            "end_time": pd.Timestamp(row["end_time"]),
            "first_active_time": pd.Timestamp(row["first_active_time"]),
            "episode_ids": [str(row.get("episode_id", ""))],
            "families": set(families),
            "component_episode_count": 1,
            "maximum_alert_score": safe_float(
                row.get("max_alert_score", 0.0)
            ),
            "maximum_abnormal_fraction_z8": safe_float(
                row.get("max_abnormal_fraction_z8", 0.0)
            ),
            "maximum_abnormal_fraction_z12": safe_float(
                row.get("max_abnormal_fraction_z12", 0.0)
            ),
            "short_episode_count": safe_float(
                row.get("short_episode_count", 0.0)
            ),
            "fault_like_short_episode_count": safe_float(
                row.get("fault_like_short_episode_count", 0.0)
            ),
            "intermittent_flag_points": safe_float(
                row.get("intermittent_flag_points", 0.0)
            ),
            "persistent_flag_points": safe_float(
                row.get("persistent_flag_points", 0.0)
            ),
            "localized_candidate_points": safe_float(
                row.get("localized_candidate_points", 0.0)
            ),
            "localized_confirmed_points": safe_float(
                row.get("localized_confirmed_points", 0.0)
            ),
            "right_censored": to_bool(row.get("right_censored", False)),
        }

        if merged and can_merge_offlog(merged[-1], row, cfg):
            current = merged[-1]
            current["end_time"] = max(
                pd.Timestamp(current["end_time"]),
                pd.Timestamp(row["end_time"]),
            )
            current["first_active_time"] = min(
                pd.Timestamp(current["first_active_time"]),
                pd.Timestamp(row["first_active_time"]),
            )
            current["episode_ids"].append(str(row.get("episode_id", "")))
            current["families"].update(families)
            current["source_ids"].update(item["source_ids"])
            current["source_files"].update(item["source_files"])
            current["component_episode_count"] += 1
            current["maximum_alert_score"] = max(
                current["maximum_alert_score"],
                item["maximum_alert_score"],
            )
            for name in [
                "maximum_abnormal_fraction_z8",
                "maximum_abnormal_fraction_z12",
            ]:
                current[name] = max(current[name], item[name])
            for name in [
                "short_episode_count",
                "fault_like_short_episode_count",
                "intermittent_flag_points",
                "persistent_flag_points",
                "localized_candidate_points",
                "localized_confirmed_points",
            ]:
                current[name] += item[name]
            current["right_censored"] = (
                current["right_censored"] or item["right_censored"]
            )
        else:
            merged.append(item)

    for item in merged:
        item["duration_hours"] = max(
            0.0,
            (
                pd.Timestamp(item["end_time"])
                - pd.Timestamp(item["start_time"])
            ).total_seconds() / 3600.0,
        )
    return merged


def calculate_offlog_priority(
    item: dict[str, Any],
    logged_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    cfg: FusionConfig,
) -> tuple[float, dict[str, float]]:
    intensity = np.clip(
        safe_float(item["maximum_alert_score"]), 0.0, 1.0
    )
    duration = np.clip(
        math.log1p(max(0.0, safe_float(item["duration_hours"])))
        / math.log1p(24.0),
        0.0,
        1.0,
    )

    families = set(item["families"])
    diversity = min(1.0, len(families) / 3.0)

    repeated_raw = (
        safe_float(item["component_episode_count"])
        + 0.5 * safe_float(item["fault_like_short_episode_count"])
        + 0.02 * safe_float(item["persistent_flag_points"])
        + 0.03 * safe_float(item["localized_confirmed_points"])
    )
    repeated_support = min(1.0, math.log1p(repeated_raw) / math.log1p(10.0))

    distance_hours = nearest_interval_distance_hours(
        pd.Timestamp(item["start_time"]),
        pd.Timestamp(item["end_time"]),
        logged_intervals,
    )
    independence = (
        1.0
        if not np.isfinite(distance_hours)
        else min(1.0, max(0.0, distance_hours) / 168.0)
    )

    score = (
        0.35 * intensity
        + 0.20 * duration
        + 0.20 * diversity
        + 0.20 * repeated_support
        + 0.05 * independence
    )

    targeted_only = families == {"targeted_subsystem_change"}
    if targeted_only:
        score *= cfg.targeted_only_penalty

    # Right-censored intervals are not deleted, but receive a small caution
    # penalty because their true end is unknown.
    if item["right_censored"]:
        score *= 0.90

    components = {
        "intensity": float(intensity),
        "duration": float(duration),
        "detector_diversity": float(diversity),
        "repeated_or_persistent_support": float(repeated_support),
        "temporal_independence": float(independence),
        "targeted_only_penalty_applied": float(targeted_only),
        "right_censored_penalty_applied": float(item["right_censored"]),
        "distance_to_nearest_logged_window_hours":
            float(distance_hours) if np.isfinite(distance_hours) else np.nan,
    }
    return float(np.clip(score, 0.0, 1.0)), components


def select_nonredundant_offlog_candidates(
    candidates: pd.DataFrame,
    cfg: FusionConfig,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    selected_rows: list[pd.Series] = []
    for _, row in candidates.sort_values(
        "review_priority_score", ascending=False
    ).iterrows():
        if len(selected_rows) >= cfg.max_offlog_candidates:
            break

        redundant = False
        for existing in selected_rows:
            same_asset = str(row["asset_id"]) == str(existing["asset_id"])
            separation = min(
                abs(
                    (
                        pd.Timestamp(row["start_time"])
                        - pd.Timestamp(existing["start_time"])
                    ).total_seconds()
                    / 3600.0
                ),
                abs(
                    (
                        pd.Timestamp(row["end_time"])
                        - pd.Timestamp(existing["end_time"])
                    ).total_seconds()
                    / 3600.0
                ),
            )
            if same_asset and separation < cfg.offlog_minimum_separation_hours:
                redundant = True
                break

        if not redundant:
            selected_rows.append(row)

    if not selected_rows:
        return candidates.iloc[0:0].copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_offlog_candidates(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    cfg: FusionConfig,
) -> pd.DataFrame:
    # Each episode is compared only with the logged window belonging to its
    # own source Event. This prevents cross-turbine temporal matching.
    outside_mask = episodes.apply(
        lambda row: episode_is_offlog_for_own_source(
            row, events, cfg.event_buffer_hours
        ),
        axis=1,
    )
    outside = episodes.loc[outside_mask].copy()

    merged = merge_offlog_episodes(outside, cfg)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(merged, start=1):
        if item["duration_hours"] < cfg.offlog_minimum_duration_hours:
            continue
        if item["maximum_alert_score"] < cfg.offlog_minimum_alert_score:
            continue

        source_event_ids = sorted(item["source_ids"])
        source_event_id = source_event_ids[0] if len(source_event_ids) == 1 else ""
        source_context = find_source_event_context(
            source_event_id,
            pd.Timestamp(item["start_time"]),
            pd.Timestamp(item["end_time"]),
            events,
            cfg.event_buffer_hours,
        )
        source_expanded_intervals = []
        if pd.notna(source_context["source_logged_event_start"]):
            source_expanded_intervals = [(
                pd.Timestamp(source_context["source_logged_event_start"])
                - pd.Timedelta(hours=cfg.event_buffer_hours),
                pd.Timestamp(source_context["source_logged_event_end"])
                + pd.Timedelta(hours=cfg.event_buffer_hours),
            )]

        priority, components = calculate_offlog_priority(
            item, source_expanded_intervals, cfg
        )
        if priority < cfg.offlog_minimum_priority:
            continue

        families = sorted(item["families"])
        source_files = sorted(item["source_files"])

        evidence: list[str] = [
            f"maximum alert score reached {item['maximum_alert_score']:.2f}",
            f"the merged interval lasted {item['duration_hours']:.2f} hours",
            f"{item['component_episode_count']} V5 episode(s) were merged",
        ]
        if families:
            evidence.append(
                "detector evidence included: "
                + "; ".join(detector_text(families))
            )
        if item["localized_confirmed_points"] > 0:
            evidence.append(
                "localized confirmed evidence was present"
            )
        if item["fault_like_short_episode_count"] > 0:
            evidence.append(
                "one or more independent fault-like short shutdowns were present"
            )
        if item["persistent_flag_points"] > 0:
            evidence.append(
                "persistent system-state support was present"
            )

        rows.append(
            {
                "candidate_id": f"offlog_candidate_{index:04d}",
                "farm_id": item["farm_id"],
                "asset_id": item["asset_id"],
                "source_event_ids": ";".join(source_event_ids),
                "source_files": ";".join(source_files),
                "source_logged_event_id":
                    source_context["source_logged_event_id"],
                "source_logged_event_start":
                    source_context["source_logged_event_start"],
                "source_logged_event_end":
                    source_context["source_logged_event_end"],
                "distance_to_source_event_window_hours":
                    source_context["distance_to_source_event_window_hours"],
                "position_relative_to_source_event":
                    source_context["position_relative_to_source_event"],
                "inside_source_event_window":
                    source_context["inside_source_event_window"],
                "inside_source_event_buffer":
                    source_context["inside_source_event_buffer"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "first_active_time": item["first_active_time"],
                "duration_hours": item["duration_hours"],
                "component_episode_count":
                    item["component_episode_count"],
                "episode_ids": ";".join(item["episode_ids"]),
                "maximum_alert_score": item["maximum_alert_score"],
                "detector_family_count": len(families),
                "detector_families": ";".join(families),
                "right_censored": item["right_censored"],
                "review_priority_score": priority,
                "priority_components": json.dumps(
                    json_ready(components), ensure_ascii=False
                ),
                "evidence_summary": " | ".join(evidence),
                "caution":
                    "This interval is a model-generated review candidate, "
                    "not a confirmed fault.",
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    return select_nonredundant_offlog_candidates(candidates, cfg)


# =============================================================================
# PAYLOADS AND REPORTS
# =============================================================================

def build_logged_payload(row: pd.Series) -> dict[str, Any]:
    signal_evidence = parse_json_list(
        row.get("cae_main_signal_evidence", "[]")
    )

    payload = {
        "task": "logged_analysis_window_model_only_explanation",
        "analysis_window": {
            "event_id": normalise_event_id(row["event_id"]),
            "farm_id": str(row.get("farm_id", "")),
            "start": row["metadata_start"],
            "end": row["metadata_end"],
        },
        "offline_v2": {
            "predicted_label": row.get("v2_predicted_label"),
            "final_status": row.get("v2_final_status"),
            "anomaly_vote_fraction": row.get("v2_vote_fraction"),
            "mean_anomaly_probability": row.get("v2_mean_probability"),
            "prediction_stability":
                row.get("v2_prediction_stability"),
            "cv_prediction_count":
                row.get("v2_cv_prediction_count"),
            "reason": row.get("v2_reason"),
            "explanation_method":
                row.get("v2_explanation_method"),
        },
        "streaming_v5": {
            "streaming_pattern":
                row.get("v5_streaming_pattern"),
            "any_overlap_detected":
                row.get("v5_any_overlap_detected"),
            "meaningful_detected":
                row.get("v5_meaningful_detected"),
            "operational_detected":
                row.get("v5_operational_detected"),
            "first_active_time":
                row.get("v5_first_active_time"),
            "detection_delay_hours":
                row.get("v5_detection_delay_hours"),
            "related_episode_count":
                row.get("v5_related_episode_count"),
            "overlap_episode_count":
                row.get("v5_overlap_episode_count"),
            "pre_window_episode_count":
                row.get("v5_pre_window_episode_count"),
            "post_window_episode_count":
                row.get("v5_post_window_episode_count"),
            "overlap_union_hours":
                row.get("v5_overlap_union_hours"),
            "overlap_fraction":
                row.get("v5_overlap_fraction"),
            "maximum_alert_score":
                row.get("v5_max_alert_score"),
            "primary_episode_id":
                row.get("v5_primary_episode_id"),
            "primary_episode_start":
                row.get("v5_primary_episode_start"),
            "primary_episode_end":
                row.get("v5_primary_episode_end"),
            "primary_detector_families":
                split_detector_families(
                    row.get("v5_primary_detector_families", "")
                ),
            "reason": row.get("v5_reason"),
        },
        "early_warning_cae": {
            "evidence_available":
                row.get("cae_evidence_available"),
            "pre_window_warning_detected":
                row.get("cae_pre_window_warning_detected"),
            "warning_within_30_days":
                row.get("cae_warning_within_30d"),
            "warning_within_14_days":
                row.get("cae_warning_within_14d"),
            "warning_within_7_days":
                row.get("cae_warning_within_7d"),
            "earliest_warning_time":
                row.get("cae_earliest_warning_time"),
            "latest_warning_time":
                row.get("cae_latest_warning_time"),
            "earliest_lead_time_days":
                row.get("cae_earliest_lead_time_days"),
            "closest_lead_time_days":
                row.get("cae_closest_lead_time_days"),
            "pre_window_episode_count":
                row.get("cae_pre_window_episode_count"),
            "total_pre_window_warning_hours":
                row.get("cae_total_pre_window_warning_hours"),
            "main_signal_evidence": signal_evidence,
            "signal_evidence_note":
                (
                    "These are signals with comparatively strong reconstruction "
                    "errors during confirmed pre-window warning episodes. They are "
                    "supporting model evidence, not identified root causes."
                ),
            "reason": row.get("cae_reason"),
        },
        "deterministic_synthesis": {
            "final_status":
                row.get("integrated_final_status"),
            "three_source_relationship":
                row.get("three_source_relationship"),
            "relationship_explanation":
                row.get("three_source_explanation"),
            "offline_streaming_relationship":
                row.get("v2_v5_evidence_relationship"),
            "offline_streaming_explanation":
                row.get("v2_v5_relationship_explanation"),
        },
        "instructions": {
            "do_not_use_ground_truth": True,
            "do_not_invent_fault_type": True,
            "do_not_override_model_outputs": True,
            "do_not_override_final_status": True,
            "do_not_treat_no_alert_as_proof_of_normality": True,
            "do_not_treat_signal_evidence_as_root_cause": True,
            "required_sections": [
                "Final assessment",
                "Offline detection",
                "Streaming detection",
                "Early-warning detection",
                "Overall interpretation",
                "Recommended action",
            ],
        },
    }
    return sanitise_for_llm(payload)

def build_offlog_payload(row: pd.Series) -> dict[str, Any]:
    payload = {
        "task": "additional_unlogged_v5_review_candidate",
        "candidate": {
            "candidate_id": row["candidate_id"],
            "farm_id": row.get("farm_id", ""),
            "asset_id": row.get("asset_id", ""),
            "source_event_ids": [
                value
                for value in str(row.get("source_event_ids", "")).split(";")
                if value
            ],
            "source_files": [
                value
                for value in str(row.get("source_files", "")).split(";")
                if value
            ],
            "source_logged_event_id":
                row.get("source_logged_event_id", ""),
            "source_logged_event_start":
                row.get("source_logged_event_start"),
            "source_logged_event_end":
                row.get("source_logged_event_end"),
            "distance_to_source_event_window_hours":
                row.get("distance_to_source_event_window_hours"),
            "position_relative_to_source_event":
                row.get("position_relative_to_source_event"),
            "inside_source_event_window":
                row.get("inside_source_event_window", False),
            "inside_source_event_buffer":
                row.get("inside_source_event_buffer", False),
            "start": row["start_time"],
            "end": row["end_time"],
            "first_active_time": row["first_active_time"],
            "duration_hours": row["duration_hours"],
            "component_episode_count":
                row["component_episode_count"],
            "maximum_alert_score": row["maximum_alert_score"],
            "review_priority_score":
                row["review_priority_score"],
            "detector_families":
                split_detector_families(
                    row.get("detector_families", "")
                ),
            "evidence_summary": row["evidence_summary"],
            "right_censored": row["right_censored"],
        },
        "instructions": {
            "describe_as_review_candidate_not_confirmed_fault": True,
            "recommend_manual_review": True,
            "do_not_invent_component_failure": True,
            "mention_no_maintenance_validation_was_used": True,
            "compare_candidate_only_with_its_own_source_event_window": True,
            "do_not_compare_with_events_from_other_turbines": True,
        },
    }
    return sanitise_for_llm(payload)


def write_markdown_report(
    path: Path,
    logged: pd.DataFrame,
    offlog: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Offline–Streaming–Early-Warning Explanation-Layer Fusion",
        "",
        "Part A combines three model-only evidence sources: offline Event-level "
        "assessment, streaming time-localised detection, and Convolutional "
        "Autoencoder pre-window early-warning evidence. "
        "Ground-truth labels, Event descriptions and recorded diagnoses are not "
        "used in the LLM payloads.",
        "",
        "## Part A — Logged analysis windows",
        "",
    ]

    for _, row in logged.iterrows():
        lines.extend(
            [
                f"### Event {row['event_id']}",
                "",
                f"- Window: {row['metadata_start']} to "
                f"{row['metadata_end']}",
                f"- Integrated final status: "
                f"{row.get('integrated_final_status', '')}",
                f"- Three-source relationship: "
                f"{row.get('three_source_relationship', '')}",
                f"- Offline status: {row.get('v2_final_status', '')}",
                f"- Offline anomaly vote fraction: "
                f"{row.get('v2_vote_fraction', np.nan)}",
                f"- Streaming pattern: "
                f"{row.get('v5_streaming_pattern', '')}",
                f"- Streaming first alert delay (hours): "
                f"{row.get('v5_detection_delay_hours', np.nan)}",
                f"- Early-warning evidence available: "
                f"{row.get('cae_evidence_available', False)}",
                f"- Pre-window early warning detected: "
                f"{row.get('cae_pre_window_warning_detected', False)}",
                f"- Warning within 30/14/7 days: "
                f"{row.get('cae_warning_within_30d', False)} / "
                f"{row.get('cae_warning_within_14d', False)} / "
                f"{row.get('cae_warning_within_7d', False)}",
                f"- Earliest pre-window lead time (days): "
                f"{row.get('cae_earliest_lead_time_days', np.nan)}",
                f"- Pre-window warning episode count: "
                f"{row.get('cae_pre_window_episode_count', 0)}",
                "",
                f"**Offline reason:** {row.get('v2_reason', '')}",
                "",
                f"**Streaming reason:** {row.get('v5_reason', '')}",
                "",
                f"**Early-warning reason:** {row.get('cae_reason', '')}",
                "",
                f"**Deterministic three-source synthesis:** "
                f"{row.get('three_source_explanation', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Part B — Additional streaming-only review candidates",
            "",
        ]
    )

    if offlog.empty:
        lines.append(
            "No off-log candidate satisfied the configured review threshold."
        )
    else:
        for _, row in offlog.iterrows():
            lines.extend(
                [
                    f"### {row['candidate_id']}",
                    "",
                    f"- Source Event ID(s): "
                    f"{row.get('source_event_ids', '') or 'Unavailable'}",
                    f"- Source SCADA file(s): "
                    f"{row.get('source_files', '') or 'Unavailable'}",
                    f"- Asset ID: {row.get('asset_id', '')}",
                    f"- Source logged Event: "
                    f"{row.get('source_logged_event_id', '') or 'Unavailable'}",
                    f"- Position relative to source Event: "
                    f"{row.get('position_relative_to_source_event', '')}",
                    f"- Distance to source Event window (hours): "
                    f"{row.get('distance_to_source_event_window_hours', np.nan)}",
                    f"- Inside source Event window: "
                    f"{row.get('inside_source_event_window', False)}",
                    f"- Inside source Event ± buffer: "
                    f"{row.get('inside_source_event_buffer', False)}",
                    f"- Interval: {row['start_time']} to {row['end_time']}",
                    f"- Duration: {row['duration_hours']:.2f} hours",
                    f"- Review-priority score: "
                    f"{row['review_priority_score']:.3f}",
                    f"- Maximum alert score: "
                    f"{row['maximum_alert_score']:.3f}",
                    f"- Detector families: "
                    f"{row['detector_families']}",
                    "",
                    f"**Evidence:** {row['evidence_summary']}",
                    "",
                    f"**Caution:** {row['caution']}",
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")

# =============================================================================
# MAIN
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare three-source Part-A explanation payloads "
            "(offline + streaming + ConvAE early warning) while keeping Part B "
            "as streaming-only off-log review candidates."
        )
    )
    parser.add_argument(
        "--v2-predictions",
        type=Path,
        required=True,
        help="Offline ml_event_predictions.csv",
    )
    parser.add_argument(
        "--v2-explanations",
        type=Path,
        default=None,
        help="Optional offline ml_event_explanations.csv",
    )
    parser.add_argument(
        "--v5-event-evaluation",
        type=Path,
        required=True,
        help="Streaming event_level_streaming_evaluation.csv",
    )
    parser.add_argument(
        "--v5-episodes",
        type=Path,
        required=True,
        help="Streaming all_stream_detected_episodes.csv",
    )
    parser.add_argument(
        "--convae-root",
        type=Path,
        required=True,
        help=(
            "Root directory of all_turbines_autoencoders_improved. It must "
            "contain all_turbines_event_results.csv and the per-Event folders "
            "event_<id>_*; parsed_feature_description.csv is recommended."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--event-buffer-hours",
        type=float,
        default=24.0,
        help=(
            "Temporal buffer before/after every logged analysis window. "
            "Also used to exclude nearby episodes from Part-B off-log candidates."
        ),
    )
    parser.add_argument(
        "--operational-window-hours",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--early-warning-lookback-days",
        type=float,
        default=30.0,
        help=(
            "Pre-window period in which ConvAE warning_episodes.csv is searched "
            "for model-only early-warning evidence."
        ),
    )
    parser.add_argument(
        "--early-warning-top-signals",
        type=int,
        default=5,
        help=(
            "Maximum number of human-readable reconstruction-error signal "
            "descriptions included in each Part-A payload."
        ),
    )
    parser.add_argument(
        "--offlog-merge-gap-hours",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--offlog-minimum-duration-hours",
        type=float,
        default=1.0 / 3.0,
    )
    parser.add_argument(
        "--offlog-minimum-alert-score",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--minimum-offlog-priority",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--max-offlog-candidates",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--offlog-minimum-separation-hours",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--targeted-only-penalty",
        type=float,
        default=0.65,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.early_warning_lookback_days <= 0:
        raise ValueError("--early-warning-lookback-days must be > 0.")
    if args.early_warning_top_signals < 0:
        raise ValueError("--early-warning-top-signals must be >= 0.")

    cfg = FusionConfig(
        event_buffer_hours=args.event_buffer_hours,
        operational_window_hours=args.operational_window_hours,
        early_warning_lookback_days=args.early_warning_lookback_days,
        early_warning_top_signals=args.early_warning_top_signals,
        offlog_merge_gap_hours=args.offlog_merge_gap_hours,
        offlog_minimum_duration_hours=
            args.offlog_minimum_duration_hours,
        offlog_minimum_alert_score=
            args.offlog_minimum_alert_score,
        offlog_minimum_priority=args.minimum_offlog_priority,
        max_offlog_candidates=args.max_offlog_candidates,
        offlog_minimum_separation_hours=
            args.offlog_minimum_separation_hours,
        targeted_only_penalty=args.targeted_only_penalty,
    )

    v2 = load_v2_predictions(args.v2_predictions)
    v2_explanations = load_v2_explanations(args.v2_explanations)
    v5_events = load_v5_event_evaluation(args.v5_event_evaluation)
    v5_episodes = load_v5_episodes(args.v5_episodes)

    if not args.convae_root.exists():
        raise FileNotFoundError(args.convae_root)
    convae_index = load_convae_event_index(args.convae_root)
    description_map = load_feature_description_map(args.convae_root)

    # V5 event evaluation supplies the logged analysis-window boundaries.
    event_master_columns = [
        column
        for column in [
            "farm_id",
            "event_id",
            "metadata_start",
            "metadata_end",
        ]
        if column in v5_events.columns
    ]
    event_master = v5_events[event_master_columns].copy()

    integrated = event_master.merge(v2, on="event_id", how="left")
    if not v2_explanations.empty:
        integrated = integrated.merge(
            v2_explanations, on="event_id", how="left"
        )

    # Streaming evidence for the logged window.
    v5_summaries = pd.DataFrame(
        [
            build_v5_window_summary(row, v5_episodes, cfg)
            for _, row in event_master.iterrows()
        ]
    )
    integrated = integrated.merge(
        v5_summaries, on="event_id", how="left"
    )

    # ConvAE early-warning evidence is recomputed from warning_episodes.csv using
    # only Event ID + logged-window start time.  No normal/anomaly folder suffix
    # or source evaluation label is propagated.
    cae_summaries = pd.DataFrame(
        [
            build_convae_window_summary(
                event_row=row,
                convae_root=args.convae_root,
                description_map=description_map,
                cfg=cfg,
            )
            for _, row in event_master.iterrows()
        ]
    )
    integrated = integrated.merge(
        cae_summaries, on="event_id", how="left"
    )
    integrated = integrated.merge(
        convae_index, on="event_id", how="left"
    )
    integrated["cae_event_result_available"] = (
        integrated["cae_event_result_available"]
        .fillna(False)
        .map(to_bool)
    )

    integrated["v2_reason"] = integrated.apply(
        build_v2_reason, axis=1
    )

    # Preserve the original offline/streaming relationship as an auditable
    # intermediate result.
    v2_v5_relationships = integrated.apply(
        classify_evidence_relationship, axis=1
    )
    integrated["v2_v5_evidence_relationship"] = [
        item[0] for item in v2_v5_relationships
    ]
    integrated["v2_v5_relationship_explanation"] = [
        item[1] for item in v2_v5_relationships
    ]

    # New final deterministic three-source fusion for Part A.
    three_source = integrated.apply(
        classify_three_source_fusion, axis=1
    )
    integrated["integrated_final_status"] = [
        item[0] for item in three_source
    ]
    integrated["three_source_relationship"] = [
        item[1] for item in three_source
    ]
    integrated["three_source_explanation"] = [
        item[2] for item in three_source
    ]

    # Remove any accidental ground-truth/log columns before saving the
    # explanation-layer table.
    integrated = integrated[
        [
            column
            for column in integrated.columns
            if column not in PROHIBITED_LLM_FIELDS
        ]
    ].sort_values(["metadata_start", "event_id"])

    # Part B remains exactly streaming-only in purpose.
    offlog = build_offlog_candidates(
        event_master, v5_episodes, cfg
    )

    logged_payloads = [
        build_logged_payload(row)
        for _, row in integrated.iterrows()
    ]
    offlog_payloads = [
        build_offlog_payload(row)
        for _, row in offlog.iterrows()
    ]

    integrated.to_csv(
        args.output_dir / "integrated_logged_event_evidence.csv",
        index=False,
    )
    cae_summaries.to_csv(
        args.output_dir / "early_warning_event_evidence.csv",
        index=False,
    )
    offlog.to_csv(
        args.output_dir / "additional_v5_review_candidates.csv",
        index=False,
    )
    write_jsonl(
        args.output_dir / "logged_event_llm_payloads.jsonl",
        logged_payloads,
    )
    write_jsonl(
        args.output_dir / "offlog_candidate_llm_payloads.jsonl",
        offlog_payloads,
    )
    write_markdown_report(
        args.output_dir / "deterministic_fusion_report.md",
        integrated,
        offlog,
    )

    status_counts = (
        integrated["integrated_final_status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    manifest = {
        "inputs": {
            "v2_predictions": args.v2_predictions,
            "v2_explanations": args.v2_explanations,
            "v5_event_evaluation": args.v5_event_evaluation,
            "v5_episodes": args.v5_episodes,
            "convae_root": args.convae_root,
            "convae_event_results":
                args.convae_root / "all_turbines_event_results.csv",
            "convae_feature_descriptions":
                args.convae_root / "parsed_feature_description.csv",
        },
        "configuration": cfg.__dict__,
        "logged_window_count": int(len(integrated)),
        "integrated_status_counts": status_counts,
        "early_warning_evidence_available_count": int(
            integrated["cae_evidence_available"].fillna(False).map(to_bool).sum()
        ),
        "offlog_candidate_count": int(len(offlog)),
        "part_b_streaming_only": True,
        "ground_truth_or_log_semantics_excluded": sorted(
            PROHIBITED_LLM_FIELDS
        ),
        "notes": [
            "ConvAE folder suffixes such as _normal/_anomaly are used only to "
            "locate files and are never copied to the LLM payload.",
            "Pre-window early-warning evidence is recomputed from "
            "warning_episodes.csv relative to metadata_start, so the same model-only "
            "rule is applied to every logged Event.",
            "Raw sensor names are not sent to the LLM. Human-readable descriptions "
            "are used when available.",
            "The LLM must not override integrated_final_status.",
        ],
        "outputs": [
            "integrated_logged_event_evidence.csv",
            "early_warning_event_evidence.csv",
            "additional_v5_review_candidates.csv",
            "logged_event_llm_payloads.jsonl",
            "offlog_candidate_llm_payloads.jsonl",
            "deterministic_fusion_report.md",
        ],
    }
    write_json(args.output_dir / "fusion_manifest.json", manifest)

    print("[DONE] Three-source Part-A explanation fusion completed.")
    print(f"[LOGGED WINDOWS] {len(integrated)}")
    print(f"[STATUS COUNTS] {status_counts}")
    print(
        "[EARLY-WARNING EVIDENCE AVAILABLE] "
        f"{manifest['early_warning_evidence_available_count']}"
    )
    print(f"[OFF-LOG STREAMING-ONLY CANDIDATES] {len(offlog)}")
    print(f"[OUTPUT] {args.output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
