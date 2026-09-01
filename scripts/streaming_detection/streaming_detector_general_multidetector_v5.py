#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General multi-detector V5 causal replay with latched semantic fusion and targeted subsystem onset detectors.

This script is the streaming/online counterpart of
``four_detector_event_analysis_power_confirmed.py``.  It deliberately does not
use metadata event_start/event_end to gate detection.  Metadata is optional and
is used only after detection to annotate rows for historical evaluation.

Design principles
-----------------
1. Strictly causal: a score at time t uses only data available at or before t.
2. Ten-minute replay: files are processed chronologically as if rows arrived one
   at a time.
3. Baseline freeze: the adaptive baseline updates only in a stable NORMAL state;
   it is frozen during candidate/active/recovery states to reduce fault
   contamination.
4. Automatic episode construction: a state machine creates alert episodes
   without knowing metadata boundaries.
5. Two-level evidence: review-only signals cannot promote themselves to red.
   Red alerts require an explicitly confirmed detector family.
6. Edge-triggered episodes: repeated-short clusters fire once per burst and
   re-arm only after a causal quiet period.
7. Targeted subsystem onset: oil, hydraulic, yaw, electrical, cooling and
   communication-control changes use strict onset gates and per-family latches.

Main outputs
------------
<output-dir>/<source_id>/stream_row_scores.csv
<output-dir>/<source_id>/stream_detected_episodes.csv
<output-dir>/<source_id>/stream_replay_summary.json
<output-dir>/all_stream_detected_episodes.csv
<output-dir>/stream_replay_manifest.json

Example
-------
python streaming_detector_replay.py ^
  --farm C ^
  --metadata "data/raw/Wind Farm C/event_info.csv" ^
  --event-dir "data/raw/Wind Farm C/datasets" ^
  --event-id all ^
  --output-dir "outputs/farmC_streaming_replay" ^
  --measurement-mode avg_only ^
  --power-signals "power_2,power_5,power_6,power_17"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

EPSILON = 1e-9
TRUE_TEXT = {"true", "1", "yes", "y", "t"}
STAT_SUFFIX_PATTERN = re.compile(r"_(avg|max|min|std)$", re.IGNORECASE)
TIMESTAMP_CANDIDATES = [
    "time_stamp", "timestamp", "time", "datetime", "date_time", "date",
]
ROW_ID_CANDIDATES = ["id", "row_id", "record_id", "index"]
NON_MEASUREMENT_COLUMNS = {
    "time_stamp", "timestamp", "time", "datetime", "date_time", "date",
    "asset_id", "id", "row_id", "record_id", "index", "train_test",
    "status_type_id", "event_id", "event_label", "metadata_label",
}


@dataclass(frozen=True)
class ReplayConfig:
    sampling_minutes: float
    warmup_hours: float = 48.0
    baseline_effective_hours: float = 48.0
    minimum_warmup_valid_fraction: float = 0.50
    baseline_clip_z: float = 6.0
    baseline_freeze_abnormal_fraction: float = 0.05

    short_z_threshold: float = 8.0
    short_fraction_floor: float = 0.05
    short_local_power_points: int = 6
    short_minimum_power_drop_ratio: float = 0.40
    short_minimum_recovery_ratio: float = 0.70
    power_dip_z_threshold: float = 6.0
    power_recovery_z_threshold: float = 5.0
    power_consensus_fraction: float = 0.75
    minimum_power_signal_count: int = 3

    # V4 rolling short-standstill aggregation inspired by the V2 event-level model.
    # A single confirmed short stop is review evidence only. Repeated stops within
    # causal 2 h / 24 h windows form weak or strong clusters.
    intermittent_window_points: int = 12
    intermittent_minimum_short_flags: int = 2
    short_6h_window_hours: float = 6.0
    short_24h_window_hours: float = 24.0
    strong_short_count_2h: int = 3
    strong_short_count_6h: int = 2
    # Reduced from 8 to 6 to recover sensitivity; chronic cycling is handled
    # separately by the causal seven-day background-burden suppressor.
    strong_short_count_24h: int = 6
    # A repeated-short pattern can trigger a red alert only when the new short
    # also has stronger process evidence. This reduces normal operational cycling.
    strong_short_minimum_abnormal_fraction: float = 0.10
    short_support_minutes: float = 60.0
    # V6 fault-like short gate: a short stop must follow a stable generating
    # period and have a sufficiently strong power drop/recovery pattern before
    # it can contribute to an intermittent fault cluster.
    short_pre_generating_points: int = 6
    short_pre_generating_fraction: float = 0.80
    short_minimum_drop_score: float = 1.00

    # V6.1: do not confirm a fault-like short from a single recovery sample.
    # A pending dip must recover and then remain in a stable generating state for
    # several causal samples. Closely spaced confirmations are de-duplicated.
    short_stable_recovery_points: int = 3
    short_stable_recovery_fraction: float = 0.80
    short_pending_timeout_minutes: float = 180.0
    short_independent_min_gap_minutes: float = 60.0

    # A fault-like short must start after genuinely stable generation and contain
    # a bounded low-power interval. With ten-minute data, 1-6 points means
    # approximately 10-120 minutes.
    short_stable_generation_ratio: float = 0.30
    short_minimum_dip_points: int = 1
    short_maximum_dip_points: int = 12

    # Causal chronic-cycling suppression. A repeated-short trigger is downgraded
    # to review when the previous seven days already contain both high alert
    # burden and many independent short episodes. New localized confirmation can
    # override the suppression.
    background_window_hours: float = 168.0
    background_minimum_history_hours: float = 48.0
    background_active_fraction_threshold: float = 0.20
    background_short_count_threshold: int = 20
    background_suppression_enabled: bool = True

    # In chronic-cycling conditions, do not fully suppress the detector.
    # Instead, require a denser new burst before promotion to red alert.
    background_strong_short_count_2h: int = 4
    background_strong_short_count_6h: int = 3
    background_strong_short_count_24h: int = 8

    # Multi-review promotion created many background episodes in V6. It is
    # disabled by default and can be enabled explicitly for controlled testing.
    # V3 permanently disables generic multi-review red promotion. The V2 results
    # showed that this path generated many background episodes and normal overlaps.
    # The fields are retained only for output/config compatibility.
    enable_multi_review_promotion: bool = False
    multi_review_confirmation_points: int = 6

    # Repeated-short clusters use edge triggering. Once a burst has fired, another
    # alert cannot fire until the recent short burden has remained quiet.
    short_cluster_quiet_hours: float = 6.0
    short_cluster_rearm_max_count_6h: int = 1

    # Operating-state gate based on robust warm-up power quantiles.
    operating_stopped_ratio: float = 0.03
    operating_transition_ratio: float = 0.15
    operating_generating_ratio: float = 0.15
    operating_state_confirmation_points: int = 3

    # Communication / data-quality detector.
    communication_missing_candidate_fraction: float = 0.10
    communication_missing_confirmed_fraction: float = 0.20
    communication_frozen_candidate_fraction: float = 0.70
    communication_frozen_confirmed_fraction: float = 0.85
    communication_confirmation_points: int = 3
    communication_timestamp_gap_multiplier: float = 1.5
    communication_confirmed_gap_minutes: float = 30.0
    # Generic missing/frozen/timestamp-gap evidence is a data-quality warning by
    # default, not a confirmed turbine-fault detector. It can be promoted only in
    # a controlled ablation experiment.
    communication_red_alert_enabled: bool = False

    # Generic slow-trend detector using causal standardized-value windows.
    slow_trend_window_hours: float = 6.0
    slow_trend_shift_hours: float = 24.0
    slow_trend_minimum_slope_z_per_hour: float = 0.25
    slow_trend_minimum_shift_z: float = 2.5
    slow_trend_minimum_signals: int = 2
    slow_trend_confirmation_points: int = 6
    # Generic trend scanning across anonymous SCADA columns is review-only by
    # default. Enable red-alert promotion only after a semantic sensor whitelist
    # and load-corrected residual model have been validated.
    slow_trend_red_alert_enabled: bool = False

    # Status-code detector. The detector learns status frequencies causally from
    # warm-up/history and does not require a hard-coded status dictionary.
    status_history_hours: float = 720.0
    status_minimum_history_hours: float = 48.0
    status_rare_frequency_threshold: float = 0.0005
    status_novel_confirmation_points: int = 2
    status_rare_confirmation_points: int = 3
    status_transition_window_hours: float = 1.0
    status_transition_count_threshold: int = 4
    status_confirmed_support_hours: float = 6.0

    # Semantic subsystem detector driven by feature_description.csv. It scans
    # physically related groups rather than all anonymous sensors together.
    semantic_candidate_z_threshold: float = 6.0
    semantic_confirmed_z_threshold: float = 8.0
    semantic_minimum_abnormal_signals: int = 2
    semantic_maximum_required_signals: int = 8
    semantic_required_fraction: float = 0.20
    semantic_confirmation_points: int = 3
    semantic_multi_group_confirmation_points: int = 3

    # A semantic subsystem deviation is review-only by default. It can become
    # red-alert evidence only when it is a new causal burst, not a chronic
    # background deviation, and an independent detector corroborates it.
    semantic_burden_window_hours: float = 24.0
    semantic_burden_minimum_history_hours: float = 6.0
    semantic_chronic_fraction_threshold: float = 0.20
    semantic_onset_quiet_fraction_threshold: float = 0.10
    semantic_fusion_confirmation_points: int = 2
    semantic_confirmed_support_hours: float = 1.0
    semantic_fusion_quiet_hours: float = 12.0
    semantic_fusion_maximum_group_count: int = 2

    # V5 targeted causal onset detectors. These use the semantic mapping but are
    # stricter than generic semantic review: high severity, low prior burden,
    # bounded group count, confirmation, latch and quiet-period re-arm.
    targeted_change_z_threshold: float = 10.0
    targeted_change_confirmation_points: int = 2
    targeted_change_quiet_hours: float = 12.0
    targeted_change_support_hours: float = 1.0
    targeted_change_maximum_group_count: int = 2

    # Status and generic persistent deviations are review/fusion evidence only.
    status_independent_red_enabled: bool = False
    communication_control_confirmation_points: int = 2

    persistent_z_threshold: float = 12.0
    persistent_fraction_threshold: float = 0.45
    persistent_smoothing_points: int = 6
    persistent_minimum_points: int = 12

    localized_z_threshold: float = 8.0
    localized_minimum_abnormal_signals: int = 3
    localized_coverage_window_points: int = 6
    localized_minimum_signal_coverage: float = 0.60
    localized_minimum_stable_signals: int = 3
    localized_overlap_window_points: int = 3
    localized_minimum_overlap: float = 0.40
    localized_strength_window_points: int = 3
    localized_strength_threshold: float = 8.0
    localized_minimum_points: int = 6
    # Localized evidence must be confirmed by a power dip near the START of the
    # localized raw segment. Confirmation is latched once per segment and expires
    # after a bounded support period, preventing a days-long candidate from being
    # re-confirmed by unrelated later power changes.
    localized_confirmation_window_minutes: float = 120.0
    localized_confirmed_support_hours: float = 6.0

    candidate_confirmation_points: int = 3
    recovery_confirmation_points: int = 6
    # Keep an episode logically open after recovery so related evidence that
    # returns shortly afterwards is merged into the same episode. During this
    # cooldown the red active_alert flag is off.
    episode_merge_gap_hours: float = 6.0
    immediate_alert_score: float = 0.90
    review_score: float = 0.55

    def warmup_points(self) -> int:
        return max(12, int(round(self.warmup_hours * 60.0 / self.sampling_minutes)))

    def baseline_alpha(self) -> float:
        n = max(2.0, self.baseline_effective_hours * 60.0 / self.sampling_minutes)
        return 2.0 / (n + 1.0)

    def short_6h_points(self) -> int:
        return max(1, int(round(self.short_6h_window_hours * 60.0 / self.sampling_minutes)))

    def short_24h_points(self) -> int:
        return max(1, int(round(self.short_24h_window_hours * 60.0 / self.sampling_minutes)))

    def background_window_points(self) -> int:
        return max(
            1,
            int(round(self.background_window_hours * 60.0 / self.sampling_minutes)),
        )

    def background_minimum_history_points(self) -> int:
        return max(
            1,
            int(
                round(
                    self.background_minimum_history_hours
                    * 60.0
                    / self.sampling_minutes
                )
            ),
        )

    def status_history_points(self) -> int:
        return max(
            1,
            int(round(self.status_history_hours * 60.0 / self.sampling_minutes)),
        )

    def status_minimum_history_points(self) -> int:
        return max(
            1,
            int(
                round(
                    self.status_minimum_history_hours
                    * 60.0
                    / self.sampling_minutes
                )
            ),
        )

    def status_transition_window_points(self) -> int:
        return max(
            2,
            int(
                round(
                    self.status_transition_window_hours
                    * 60.0
                    / self.sampling_minutes
                )
            ),
        )

    def status_confirmed_support_points(self) -> int:
        return max(
            1,
            int(
                round(
                    self.status_confirmed_support_hours
                    * 60.0
                    / self.sampling_minutes
                )
            ),
        )

    def semantic_burden_window_points(self) -> int:
        return max(
            1,
            int(
                round(
                    self.semantic_burden_window_hours
                    * 60.0
                    / self.sampling_minutes
                )
            ),
        )

    def semantic_burden_minimum_history_points(self) -> int:
        return max(
            1,
            int(
                round(
                    self.semantic_burden_minimum_history_hours
                    * 60.0
                    / self.sampling_minutes
                )
            ),
        )

    def semantic_confirmed_support_points(self) -> int:
        return max(
            1,
            int(round(self.semantic_confirmed_support_hours * 60.0 / self.sampling_minutes)),
        )

    def semantic_fusion_quiet_points(self) -> int:
        return max(1, int(round(self.semantic_fusion_quiet_hours * 60.0 / self.sampling_minutes)))

    def targeted_change_quiet_points(self) -> int:
        return max(1, int(round(self.targeted_change_quiet_hours * 60.0 / self.sampling_minutes)))

    def targeted_change_support_points(self) -> int:
        return max(1, int(round(self.targeted_change_support_hours * 60.0 / self.sampling_minutes)))

    def slow_trend_window_points(self) -> int:
        return max(4, int(round(self.slow_trend_window_hours * 60.0 / self.sampling_minutes)))

    def slow_trend_shift_points(self) -> int:
        return max(
            self.slow_trend_window_points() + 1,
            int(round(self.slow_trend_shift_hours * 60.0 / self.sampling_minutes)),
        )

    def short_cluster_quiet_points(self) -> int:
        return max(
            1,
            int(
                round(
                    self.short_cluster_quiet_hours
                    * 60.0
                    / self.sampling_minutes
                )
            ),
        )

    def short_support_points(self) -> int:
        return max(1, int(round(self.short_support_minutes / self.sampling_minutes)))

    def short_pending_timeout_points(self) -> int:
        return max(
            1,
            int(round(self.short_pending_timeout_minutes / self.sampling_minutes)),
        )

    def short_independent_min_gap_points(self) -> int:
        return max(
            1,
            int(round(self.short_independent_min_gap_minutes / self.sampling_minutes)),
        )

    def localized_confirmation_window_points(self) -> int:
        return max(
            1,
            int(round(self.localized_confirmation_window_minutes / self.sampling_minutes)),
        )

    def localized_confirmed_support_points(self) -> int:
        return max(
            1,
            int(round(self.localized_confirmed_support_hours * 60.0 / self.sampling_minutes)),
        )

    def episode_merge_gap_points(self) -> int:
        return max(1, int(round(self.episode_merge_gap_hours * 60.0 / self.sampling_minutes)))


@dataclass
class Episode:
    farm_id: str
    source_id: str
    source_file: str
    asset_id: str
    episode_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    first_active_time: pd.Timestamp
    last_evidence_time: pd.Timestamp
    duration_hours: float
    right_censored: bool
    max_alert_score: float
    max_abnormal_fraction_z8: float
    max_abnormal_fraction_z12: float
    short_confirmed_count: int
    short_episode_count: int
    fault_like_short_episode_count: int
    intermittent_flag_points: int
    persistent_flag_points: int
    localized_candidate_points: int
    localized_confirmed_points: int
    detector_families: str
    metadata_overlap: bool
    metadata_event_id: str
    metadata_label: str
    metadata_overlap_hours: float


class OnlineEWMABaseline:
    """Per-measurement causal EWMA location/scale with clipped updates."""

    def __init__(
        self,
        initial_rows: np.ndarray,
        alpha: float,
        minimum_valid_fraction: float,
        clip_z: float,
    ) -> None:
        if initial_rows.ndim != 2 or initial_rows.shape[0] < 2:
            raise ValueError("At least two warm-up rows are required.")

        self.alpha = float(alpha)
        self.clip_z = float(clip_z)
        valid_fraction = np.mean(np.isfinite(initial_rows), axis=0)
        self.usable = valid_fraction >= float(minimum_valid_fraction)

        centre = np.nanmedian(initial_rows, axis=0)
        mad = np.nanmedian(np.abs(initial_rows - centre), axis=0)
        scale = 1.4826 * mad

        q75 = np.nanpercentile(initial_rows, 75, axis=0)
        q25 = np.nanpercentile(initial_rows, 25, axis=0)
        iqr_scale = (q75 - q25) / 1.349
        scale = np.where(np.isfinite(scale) & (scale > EPSILON), scale, iqr_scale)

        std = np.nanstd(initial_rows, axis=0, ddof=1)
        scale = np.where(np.isfinite(scale) & (scale > EPSILON), scale, std)
        scale = np.where(np.isfinite(scale) & (scale > EPSILON), scale, np.nan)

        self.usable &= np.isfinite(centre) & np.isfinite(scale)
        self.mean = np.where(self.usable, centre, 0.0).astype(float)
        self.var = np.where(self.usable, np.square(scale), 1.0).astype(float)

    @property
    def scale(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, EPSILON))

    def score(self, row: np.ndarray) -> np.ndarray:
        z = np.zeros_like(row, dtype=float)
        valid = self.usable & np.isfinite(row)
        z[valid] = np.abs(row[valid] - self.mean[valid]) / self.scale[valid]
        return np.clip(z, 0.0, 1_000_000.0)

    def update(self, row: np.ndarray) -> None:
        valid = self.usable & np.isfinite(row)
        if not np.any(valid):
            return
        scale = self.scale
        residual = row - self.mean
        clipped = np.clip(residual, -self.clip_z * scale, self.clip_z * scale)
        new_mean = self.mean + self.alpha * clipped
        centred = row - new_mean
        new_var = (1.0 - self.alpha) * self.var + self.alpha * np.square(centred)
        self.mean[valid] = new_mean[valid]
        self.var[valid] = np.maximum(new_var[valid], EPSILON)


def normalise_event_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=";", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", low_memory=False)
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";", engine="python", low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def infer_sampling_minutes(timestamps: pd.Series) -> float:
    diffs = timestamps.sort_values().diff().dt.total_seconds().div(60.0)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        raise ValueError("Could not infer the sampling interval.")
    return float(diffs.median())


def select_measurement_columns(
    df: pd.DataFrame,
    mode: str,
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
        if mode == "avg_only" and match.group(1).lower() != "avg":
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if float(numeric.notna().mean()) >= minimum_valid_fraction:
            selected.append(column)
    if not selected:
        raise ValueError("No measurement columns were selected.")
    return selected


def base_signal_name(column: str) -> str:
    return STAT_SUFFIX_PATTERN.sub("", column)


def build_base_groups(columns: list[str]) -> tuple[list[str], list[np.ndarray]]:
    mapping: dict[str, list[int]] = {}
    for index, column in enumerate(columns):
        mapping.setdefault(base_signal_name(column), []).append(index)
    names = list(mapping)
    indices = [np.asarray(mapping[name], dtype=int) for name in names]
    return names, indices


def identify_power_bases(
    base_names: list[str],
    requested: Optional[list[str]],
) -> list[str]:
    available = set(base_names)
    if requested:
        selected = [name for name in requested if name in available]
        if selected:
            return selected
        warnings.warn("Requested power signals were not found; using automatic matching.")

    include = [r"(^|_)power($|_)", r"active_power", r"generator_power", r"electrical_power", r"(^|_)pwr($|_)"]
    exclude = [r"reactive", r"factor", r"setpoint", r"limit", r"rated"]
    result: list[str] = []
    for name in base_names:
        lower = name.lower()
        if any(re.search(pattern, lower) for pattern in include) and not any(
            re.search(pattern, lower) for pattern in exclude
        ):
            result.append(name)
    return result


def read_feature_descriptions(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Load semicolon/comma separated sensor descriptions."""
    if path is None:
        return {}
    frame = read_csv_auto(path)
    sensor_col = find_column(frame.columns, ["sensor_name", "feature", "signal"])
    description_col = find_column(
        frame.columns, ["description", "sensor_description", "meaning"]
    )
    if sensor_col is None or description_col is None:
        raise ValueError(
            "Feature description file must contain sensor_name and description."
        )
    angle_col = find_column(frame.columns, ["is_angle"])
    counter_col = find_column(frame.columns, ["is_counter"])
    result: dict[str, dict[str, Any]] = {}
    for _, item in frame.iterrows():
        name = str(item[sensor_col]).strip()
        if not name:
            continue
        result[name] = {
            "description": str(item[description_col]).strip(),
            "is_angle": (
                str(item[angle_col]).strip().lower() in TRUE_TEXT
                if angle_col else False
            ),
            "is_counter": (
                str(item[counter_col]).strip().lower() in TRUE_TEXT
                if counter_col else False
            ),
        }
    return result


SEMANTIC_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "dc_link_voltage": (
        "dc link", "direct current link", "dc-link",
    ),
    "grid_voltage": (
        "grid voltage", "line voltage", "mains voltage",
        "phase voltage", "transformer voltage",
    ),
    "generator_current": (
        "generator current", "stator current", "phase current",
    ),
    "converter_electrical": (
        "converter", "inverter", "rectifier",
    ),
    "auxiliary_supply": (
        "supply voltage", "battery voltage", "control voltage",
        "cabinet voltage", "auxiliary voltage",
    ),
    "pitch_axis": (
        "pitch", "blade", "axis 1", "axis 2", "axis 3", "pitching",
    ),
    "yaw": ("yaw",),
    "gearbox_oil": (
        "gear oil", "gearbox oil", "oil pump", "oil pressure",
        "oil level", "lubric",
    ),
    "gearbox_thermal": (
        "gearbox temperature", "gear bearing temperature",
        "gearbox bearing temperature",
    ),
    "generator_thermal": (
        "generator temperature", "generator bearing temperature",
        "stator temperature", "rotor temperature",
    ),
    "converter_thermal": (
        "converter temperature", "inverter temperature",
        "power module temperature",
    ),
    "cabinet_thermal": (
        "cabinet temperature", "nacelle temperature",
        "hub temperature", "ambient cabinet",
    ),
    "cooling_system": (
        "cooling", "coolant", "cooler", "heat exchanger",
        "water pump", "cooling fan",
    ),
    "hydraulic_brake": (
        "hydraulic", "brake", "accumulator pressure",
    ),
    "vibration_mechanical": (
        "vibration", "acceleration", "shaft", "coupling",
        "main bearing",
    ),
}



def build_semantic_groups(
    base_names: list[str],
    feature_descriptions: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    groups: dict[str, list[int]] = {
        name: [] for name in SEMANTIC_GROUP_PATTERNS
    }
    descriptions: dict[str, str] = {}
    for index, base_name in enumerate(base_names):
        item = feature_descriptions.get(base_name, {})
        description = str(item.get("description", "")).strip()
        descriptions[base_name] = description
        if not description or item.get("is_angle") or item.get("is_counter"):
            continue
        lower = description.lower()
        for group_name, patterns in SEMANTIC_GROUP_PATTERNS.items():
            if any(pattern in lower for pattern in patterns):
                groups[group_name].append(index)
                break
    return (
        {
            name: np.asarray(indices, dtype=int)
            for name, indices in groups.items()
            if indices
        },
        descriptions,
    )


def status_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def required_consensus(n_signals: int, fraction: float, minimum: int) -> int:
    if n_signals <= 0:
        return 0
    return min(n_signals, max(1 if n_signals == 1 else minimum, math.ceil(n_signals * fraction)))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def top_k_mean(values: np.ndarray, k: int) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    k = min(k, finite.size)
    return float(np.partition(finite, finite.size - k)[-k:].mean())


def causal_slope(values: np.ndarray, sampling_hours: float) -> float:
    """Least-squares slope using only the supplied causal history."""
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if int(finite.sum()) < 4:
        return 0.0
    y = array[finite]
    x = np.arange(array.size, dtype=float)[finite] * sampling_hours
    x = x - float(x.mean())
    denominator = float(np.square(x).sum())
    if denominator <= EPSILON:
        return 0.0
    return float(np.sum(x * (y - float(y.mean()))) / denominator)


def consecutive_run_length(values: pd.Series) -> pd.Series:
    array = values.fillna(False).astype(bool).to_numpy()
    result = np.zeros(len(array), dtype=int)
    run = 0
    for i, value in enumerate(array):
        run = run + 1 if value else 0
        result[i] = run
    return pd.Series(result, index=values.index)


def load_metadata(path: Optional[Path], farm_id: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = read_csv_auto(path)
    id_col = find_column(df.columns, ["event_id", "id"])
    label_col = find_column(df.columns, ["event_label", "metadata_label", "label"])
    start_col = find_column(df.columns, ["event_start", "metadata_start", "start"])
    end_col = find_column(df.columns, ["event_end", "metadata_end", "end"])
    description_col = find_column(df.columns, ["event_description", "description"])
    if not all([id_col, label_col, start_col, end_col]):
        raise ValueError("Metadata must contain event id, label, start and end columns.")
    out = pd.DataFrame(
        {
            "event_id": df[id_col].map(normalise_event_id),
            "metadata_label": df[label_col].astype(str).str.strip().str.lower(),
            "metadata_start": pd.to_datetime(df[start_col], errors="coerce"),
            "metadata_end": pd.to_datetime(df[end_col], errors="coerce"),
            "event_description": df[description_col].fillna("").astype(str) if description_col else "",
            "farm_id": farm_id,
        }
    )
    return out.dropna(subset=["metadata_start", "metadata_end"]).reset_index(drop=True)


def metadata_for_source(metadata: pd.DataFrame, source_id: str) -> pd.DataFrame:
    if metadata.empty:
        return metadata.copy()
    exact = metadata.loc[metadata["event_id"] == normalise_event_id(source_id)].copy()
    return exact if not exact.empty else pd.DataFrame(columns=metadata.columns)


def annotate_metadata_rows(
    output: pd.DataFrame,
    source_metadata: pd.DataFrame,
) -> pd.DataFrame:
    result = output.copy()
    result["metadata_event_id"] = ""
    result["metadata_label"] = ""
    result["metadata_start"] = pd.NaT
    result["metadata_end"] = pd.NaT
    result["inside_metadata_interval"] = False
    result["target_anomaly"] = 0
    if source_metadata.empty:
        return result

    for _, item in source_metadata.iterrows():
        mask = (
            (result["timestamp"] >= item["metadata_start"])
            & (result["timestamp"] <= item["metadata_end"])
        )
        result.loc[mask, "metadata_event_id"] = str(item["event_id"])
        result.loc[mask, "metadata_label"] = str(item["metadata_label"])
        result.loc[mask, "metadata_start"] = item["metadata_start"]
        result.loc[mask, "metadata_end"] = item["metadata_end"]
        result.loc[mask, "inside_metadata_interval"] = True
        if str(item["metadata_label"]).lower() == "anomaly":
            result.loc[mask, "target_anomaly"] = 1
    return result


def overlap_episode_with_metadata(
    start: pd.Timestamp,
    end: pd.Timestamp,
    source_metadata: pd.DataFrame,
) -> tuple[bool, str, str, float]:
    if source_metadata.empty:
        return False, "", "", 0.0
    best_hours = 0.0
    best_id = ""
    best_label = ""
    for _, item in source_metadata.iterrows():
        overlap_start = max(start, item["metadata_start"])
        overlap_end = min(end, item["metadata_end"])
        hours = max(0.0, (overlap_end - overlap_start).total_seconds() / 3600.0)
        if hours > best_hours:
            best_hours = hours
            best_id = str(item["event_id"])
            best_label = str(item["metadata_label"])
    return best_hours > 0.0, best_id, best_label, best_hours


def replay_one_file(
    event_file: Path,
    source_id: str,
    farm_id: str,
    source_metadata: pd.DataFrame,
    output_dir: Path,
    measurement_mode: str,
    manual_power_signals: Optional[list[str]],
    feature_descriptions: dict[str, dict[str, Any]],
    config_overrides: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = read_csv_auto(event_file)
    timestamp_col = find_column(df.columns, TIMESTAMP_CANDIDATES)
    if timestamp_col is None:
        raise ValueError(f"No timestamp column found in {event_file.name}.")
    row_id_col = find_column(df.columns, ROW_ID_CANDIDATES)
    asset_col = find_column(df.columns, ["asset_id", "turbine_id", "asset"])
    status_col = find_column(
        df.columns,
        ["status_type_id", "status_id", "status_code", "operating_status"],
    )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df = df.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("The source file contains too few usable rows.")

    sampling_minutes = infer_sampling_minutes(df[timestamp_col])
    cfg = ReplayConfig(sampling_minutes=sampling_minutes, **config_overrides)
    warmup_points = cfg.warmup_points()
    if len(df) <= warmup_points:
        raise ValueError(
            f"{event_file.name} has {len(df)} rows but requires more than "
            f"{warmup_points} warm-up rows."
        )

    measurement_columns = select_measurement_columns(
        df, measurement_mode, cfg.minimum_warmup_valid_fraction
    )
    numeric_df = df[measurement_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=float)
    base_names, base_groups = build_base_groups(measurement_columns)
    base_index = {name: index for index, name in enumerate(base_names)}
    semantic_groups, semantic_descriptions = build_semantic_groups(
        base_names, feature_descriptions
    )
    power_bases = identify_power_bases(base_names, manual_power_signals)
    power_base_indices = [base_index[name] for name in power_bases]

    # Robust power reference used only for operating-state classification.
    warmup_power_matrix = (
        values[:warmup_points, :]
        if power_base_indices
        else np.empty((warmup_points, 0), dtype=float)
    )

    # Prefer the avg column for each requested power base.
    power_measurement_indices: list[int] = []
    for power_name in power_bases:
        candidates = [
            index for index, column in enumerate(measurement_columns)
            if base_signal_name(column) == power_name and column.lower().endswith("_avg")
        ]
        if candidates:
            power_measurement_indices.append(candidates[0])

    power_high_reference = float("nan")
    if power_measurement_indices:
        warmup_power_values = values[:warmup_points, power_measurement_indices]
        row_power_median = np.nanmedian(warmup_power_values, axis=1)
        finite_power = row_power_median[np.isfinite(row_power_median)]
        if finite_power.size:
            power_high_reference = float(np.nanpercentile(finite_power, 90))

    baseline = OnlineEWMABaseline(
        initial_rows=values[:warmup_points],
        alpha=cfg.baseline_alpha(),
        minimum_valid_fraction=cfg.minimum_warmup_valid_fraction,
        clip_z=cfg.baseline_clip_z,
    )

    short_history_2h: deque[int] = deque(maxlen=cfg.intermittent_window_points)
    short_history_6h: deque[int] = deque(maxlen=cfg.short_6h_points())
    short_history_24h: deque[int] = deque(maxlen=cfg.short_24h_points())
    recent_short_start_history: deque[int] = deque(maxlen=cfg.short_support_points())
    stable_generation_history: deque[int] = deque(
        maxlen=cfg.short_pre_generating_points
    )
    background_short_history: deque[int] = deque(
        maxlen=cfg.background_window_points()
    )
    background_active_history: deque[int] = deque(
        maxlen=cfg.background_window_points()
    )
    localized_power_dip_history: deque[tuple[pd.Timestamp, bool]] = deque(
        maxlen=cfg.localized_confirmation_window_points()
    )
    persistent_fraction_history: deque[float] = deque(maxlen=cfg.persistent_smoothing_points)
    localized_active_history: deque[np.ndarray] = deque(maxlen=cfg.localized_coverage_window_points)
    localized_overlap_history: deque[float] = deque(maxlen=cfg.localized_overlap_window_points)
    localized_strength_history: deque[float] = deque(maxlen=cfg.localized_strength_window_points)
    power_histories: dict[int, deque[float]] = {
        index: deque(maxlen=cfg.short_local_power_points) for index in power_measurement_indices
    }
    power_generating_histories: dict[int, deque[int]] = {
        index: deque(maxlen=cfg.short_pre_generating_points)
        for index in power_measurement_indices
    }
    standardized_history: deque[np.ndarray] = deque(
        maxlen=cfg.slow_trend_shift_points()
    )

    status_history: deque[str] = deque(maxlen=cfg.status_history_points())
    status_transition_history: deque[int] = deque(
        maxlen=cfg.status_transition_window_points()
    )
    status_confirmed_support_history: deque[int] = deque(
        maxlen=cfg.status_confirmed_support_points()
    )
    semantic_confirmed_support_history: deque[int] = deque(
        maxlen=cfg.semantic_confirmed_support_points()
    )
    semantic_group_runs: dict[str, int] = {
        name: 0 for name in semantic_groups
    }
    semantic_group_burden_history: dict[str, deque[int]] = {
        name: deque(maxlen=cfg.semantic_burden_window_points())
        for name in semantic_groups
    }
    semantic_multi_group_run = 0
    semantic_fusion_run = 0
    semantic_fusion_latched = False
    semantic_fusion_quiet_run = 0
    semantic_fusion_support_history: deque[int] = deque(
        maxlen=cfg.semantic_confirmed_support_points()
    )
    targeted_group_runs: dict[str, int] = {name: 0 for name in semantic_groups}
    targeted_group_latched: dict[str, bool] = {name: False for name in semantic_groups}
    targeted_group_quiet_runs: dict[str, int] = {name: 0 for name in semantic_groups}
    targeted_change_support_history: deque[int] = deque(
        maxlen=cfg.targeted_change_support_points()
    )
    communication_control_run = 0
    previous_status = ""

    previous_active_set = np.zeros(len(base_names), dtype=bool)
    previous_power_dip_flags = np.zeros(len(power_measurement_indices), dtype=bool)
    previous_power_reference = np.full(len(power_measurement_indices), np.nan)
    previous_power_value = np.full(len(power_measurement_indices), np.nan)
    previous_short_confirmed_flag = False
    previous_fault_like_short_confirmed_flag = False
    short_cluster_latched = False
    short_cluster_quiet_run = 0

    # V6.1 pending-short state. A dip candidate is not confirmed until recovery
    # has occurred and stable generation has persisted for several samples.
    pending_fault_like_short = False
    pending_fault_like_start_time: Optional[pd.Timestamp] = None
    pending_fault_like_age_points = 0
    pending_fault_like_recovery_seen = False
    pending_fault_like_recovery_run = 0
    pending_fault_like_dip_points = 0
    pending_fault_like_duration_valid = False
    last_independent_fault_like_short_time: Optional[pd.Timestamp] = None

    previous_localized_raw_flag = False
    localized_raw_start_time: Optional[pd.Timestamp] = None
    localized_confirmation_latched = False
    localized_confirmation_time: Optional[pd.Timestamp] = None

    persistent_run = 0
    localized_run = 0
    communication_missing_run = 0
    communication_frozen_run = 0
    slow_trend_run = 0
    current_status_run = 0
    previous_timestamp: Optional[pd.Timestamp] = None
    previous_measurement_row: Optional[np.ndarray] = None
    operating_state = "unknown"
    pending_operating_state = "unknown"
    pending_operating_state_run = 0
    candidate_run = 0
    recovery_run = 0
    inactive_pending_run = 0
    stream_state = "normal"
    candidate_start_time: Optional[pd.Timestamp] = None
    active_start_time: Optional[pd.Timestamp] = None
    last_evidence_time: Optional[pd.Timestamp] = None
    episode_counter = 0
    current_episode_rows: list[dict[str, Any]] = []
    episodes: list[Episode] = []
    rows: list[dict[str, Any]] = []

    asset_id = ""
    if asset_col is not None and df[asset_col].notna().any():
        asset_id = str(df.loc[df[asset_col].first_valid_index(), asset_col])

    required_power = required_consensus(
        len(power_measurement_indices),
        cfg.power_consensus_fraction,
        cfg.minimum_power_signal_count,
    )

    for position in range(len(df)):
        timestamp = pd.Timestamp(df.at[position, timestamp_col])
        row_id = df.at[position, row_id_col] if row_id_col is not None else position
        row = values[position]

        # Data-quality measurements are causal and independent of fault metadata.
        missing_fraction = float(np.mean(~np.isfinite(row)))
        frozen_fraction = 0.0
        if previous_measurement_row is not None:
            comparable = np.isfinite(row) & np.isfinite(previous_measurement_row)
            if np.any(comparable):
                frozen_fraction = float(
                    np.mean(
                        np.isclose(
                            row[comparable],
                            previous_measurement_row[comparable],
                            rtol=1e-9,
                            atol=1e-12,
                        )
                    )
                )

        timestamp_gap_minutes = 0.0
        if previous_timestamp is not None:
            timestamp_gap_minutes = max(
                0.0,
                (timestamp - previous_timestamp).total_seconds() / 60.0,
            )

        if position < warmup_points:
            rows.append(
                {
                    "farm_id": farm_id,
                    "source_id": source_id,
                    "source_file": str(event_file),
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "row_id": row_id,
                    "warmup_complete": False,
                    "stream_state": "warmup",
                    "baseline_updated": False,
                    "alert_score": 0.0,
                    "review_flag": False,
                    "active_alert_flag": False,
                }
            )
            if status_col is not None:
                warmup_status = status_token(df.at[position, status_col])
                if warmup_status:
                    status_history.append(warmup_status)
                    if warmup_status == previous_status:
                        current_status_run += 1
                    else:
                        current_status_run = 1
                    previous_status = warmup_status
            previous_timestamp = timestamp
            previous_measurement_row = row.copy()
            continue

        # Operating-state classification from robust warm-up power reference.
        current_power_median = float("nan")
        power_ratio = float("nan")
        if power_measurement_indices:
            current_power_values = row[power_measurement_indices]
            if np.isfinite(current_power_values).any():
                current_power_median = float(np.nanmedian(current_power_values))
        if (
            np.isfinite(current_power_median)
            and np.isfinite(power_high_reference)
            and abs(power_high_reference) > EPSILON
        ):
            power_ratio = current_power_median / abs(power_high_reference)

        if not np.isfinite(power_ratio):
            proposed_operating_state = "unknown"
        elif power_ratio < cfg.operating_stopped_ratio:
            proposed_operating_state = "stopped"
        elif power_ratio < cfg.operating_transition_ratio:
            proposed_operating_state = "transition"
        else:
            proposed_operating_state = "generating"

        if proposed_operating_state == pending_operating_state:
            pending_operating_state_run += 1
        else:
            pending_operating_state = proposed_operating_state
            pending_operating_state_run = 1

        if (
            pending_operating_state_run
            >= cfg.operating_state_confirmation_points
        ):
            operating_state = pending_operating_state

        generating_state_flag = operating_state == "generating"
        transition_state_flag = operating_state == "transition"
        stopped_state_flag = operating_state == "stopped"

        stable_generation_now = bool(
            np.isfinite(power_ratio)
            and power_ratio >= cfg.short_stable_generation_ratio
        )
        pre_stable_generation_fraction = (
            float(np.mean(np.asarray(stable_generation_history, dtype=float)))
            if stable_generation_history
            else 0.0
        )
        pre_stable_generation_consensus = bool(
            len(stable_generation_history)
            >= cfg.short_pre_generating_points
            and pre_stable_generation_fraction
            >= cfg.short_pre_generating_fraction
        )

        measurement_z = baseline.score(row)
        base_z = np.zeros(len(base_names), dtype=float)
        for base_position, indices in enumerate(base_groups):
            base_z[base_position] = float(np.max(measurement_z[indices]))

        abnormal_z8 = base_z >= cfg.short_z_threshold
        abnormal_z12 = base_z >= cfg.persistent_z_threshold
        abnormal_fraction_z8 = float(abnormal_z8.mean())
        abnormal_fraction_z12 = float(abnormal_z12.mean())

        # --------------------------------------------------------------
        # Causal status-code detector
        # --------------------------------------------------------------
        current_status = (
            status_token(df.at[position, status_col])
            if status_col is not None
            else ""
        )
        if current_status:
            if current_status == previous_status:
                current_status_run += 1
            else:
                current_status_run = 1

        status_counts = Counter(status_history)
        status_history_count = len(status_history)
        current_status_count = status_counts.get(current_status, 0)
        status_frequency = (
            current_status_count / status_history_count
            if current_status and status_history_count
            else 0.0
        )
        status_history_ready = bool(
            status_history_count >= cfg.status_minimum_history_points()
        )
        status_novel_flag = bool(
            current_status
            and status_history_ready
            and current_status_count == 0
        )
        status_rare_flag = bool(
            current_status
            and status_history_ready
            and current_status_count > 0
            and status_frequency <= cfg.status_rare_frequency_threshold
        )
        status_transition_flag = bool(
            current_status
            and previous_status
            and current_status != previous_status
        )
        status_transition_count = int(sum(status_transition_history)) + int(
            status_transition_flag
        )
        status_transition_burst_flag = bool(
            status_transition_count
            >= cfg.status_transition_count_threshold
        )
        status_confirmed_flag = bool(
            (
                status_novel_flag
                and current_status_run
                >= cfg.status_novel_confirmation_points
            )
            or (
                status_rare_flag
                and current_status_run
                >= cfg.status_rare_confirmation_points
            )
            or (
                status_transition_burst_flag
                and (status_novel_flag or status_rare_flag)
            )
        )
        status_support_flag = bool(
            status_confirmed_flag
            or any(status_confirmed_support_history)
        )
        status_candidate_flag = bool(
            status_novel_flag
            or status_rare_flag
            or status_transition_burst_flag
        )
        status_score = float(
            1.0
            if status_confirmed_flag
            else 0.75
            if status_novel_flag or status_rare_flag
            else 0.60
            if status_transition_burst_flag
            else 0.0
        )

        # --------------------------------------------------------------
        # Semantic subsystem detector
        # --------------------------------------------------------------
        # Raw semantic deviations are diagnostic/review evidence. A semantic
        # detector can promote to red only through semantic_fusion_confirmed_flag
        # below, which requires a new onset and independent corroboration.
        semantic_candidate_groups: list[str] = []
        semantic_confirmed_groups: list[str] = []
        semantic_onset_groups: list[str] = []
        semantic_chronic_groups: list[str] = []
        semantic_group_scores: dict[str, float] = {}
        semantic_group_abnormal_counts: dict[str, int] = {}
        semantic_group_max_z: dict[str, float] = {}
        semantic_group_burden_fractions: dict[str, float] = {}

        stable_generation_required_groups = {
            "dc_link_voltage",
            "grid_voltage",
            "generator_current",
            "converter_electrical",
            "auxiliary_supply",
            "gearbox_thermal",
            "generator_thermal",
            "converter_thermal",
            "cabinet_thermal",
            "cooling_system",
            "vibration_mechanical",
        }

        for group_name, group_indices in semantic_groups.items():
            group_z = base_z[group_indices]
            candidate_count = int(
                np.sum(group_z >= cfg.semantic_candidate_z_threshold)
            )
            confirmed_count = int(
                np.sum(group_z >= cfg.semantic_confirmed_z_threshold)
            )
            required_count = min(
                len(group_indices),
                cfg.semantic_maximum_required_signals,
                max(
                    cfg.semantic_minimum_abnormal_signals,
                    int(
                        math.ceil(
                            len(group_indices)
                            * cfg.semantic_required_fraction
                        )
                    ),
                ),
            )
            group_max_z = (
                float(np.nanmax(group_z)) if group_z.size else 0.0
            )

            burden_history = semantic_group_burden_history[group_name]
            burden_history_ready = bool(
                len(burden_history)
                >= cfg.semantic_burden_minimum_history_points()
            )
            burden_fraction = (
                float(np.mean(np.asarray(burden_history, dtype=float)))
                if burden_history
                else 0.0
            )
            semantic_group_burden_fractions[group_name] = burden_fraction

            operating_gate = bool(
                stable_generation_now
                if group_name in stable_generation_required_groups
                else not stopped_state_flag
            )
            group_candidate = bool(
                candidate_count >= required_count
                and operating_gate
                and not transition_state_flag
            )
            group_onset = bool(
                group_candidate
                and burden_history_ready
                and burden_fraction
                <= cfg.semantic_onset_quiet_fraction_threshold
            )
            group_chronic = bool(
                burden_history_ready
                and burden_fraction
                >= cfg.semantic_chronic_fraction_threshold
            )

            semantic_group_runs[group_name] = (
                semantic_group_runs[group_name] + 1
                if group_candidate
                else 0
            )
            group_confirmed = bool(
                semantic_group_runs[group_name]
                >= cfg.semantic_confirmation_points
                and confirmed_count >= required_count
            )
            group_score = min(
                1.0,
                max(
                    candidate_count / max(required_count, 1),
                    group_max_z
                    / max(cfg.semantic_confirmed_z_threshold, EPSILON),
                ),
            )
            semantic_group_scores[group_name] = float(group_score)
            semantic_group_abnormal_counts[group_name] = candidate_count
            semantic_group_max_z[group_name] = group_max_z

            if group_candidate:
                semantic_candidate_groups.append(group_name)
            if group_confirmed:
                semantic_confirmed_groups.append(group_name)
            if group_onset:
                semantic_onset_groups.append(group_name)
            if group_chronic:
                semantic_chronic_groups.append(group_name)

        semantic_candidate_group_count = len(semantic_candidate_groups)
        semantic_confirmed_group_count = len(semantic_confirmed_groups)
        semantic_onset_group_count = len(semantic_onset_groups)
        semantic_chronic_group_count = len(semantic_chronic_groups)

        semantic_multi_group_candidate_flag = bool(
            semantic_candidate_group_count >= 2
        )
        semantic_multi_group_run = (
            semantic_multi_group_run + 1
            if semantic_multi_group_candidate_flag
            else 0
        )
        semantic_multi_group_confirmed_flag = bool(
            semantic_multi_group_run
            >= cfg.semantic_multi_group_confirmation_points
        )

        # This remains a review/diagnostic flag and cannot independently open red.
        semantic_confirmed_flag = bool(
            semantic_confirmed_group_count >= 1
            or semantic_multi_group_confirmed_flag
        )
        semantic_candidate_flag = bool(
            semantic_candidate_group_count >= 1
        )
        semantic_onset_flag = bool(
            semantic_onset_group_count >= 1
        )
        semantic_chronic_flag = bool(
            semantic_chronic_group_count >= 1
            and not semantic_onset_flag
        )
        semantic_new_burst_flag = bool(
            semantic_confirmed_flag
            and semantic_onset_flag
            and not semantic_chronic_flag
        )
        semantic_support_flag = bool(
            semantic_confirmed_flag
            or any(semantic_confirmed_support_history)
        )
        semantic_score = float(
            max(semantic_group_scores.values(), default=0.0)
        )

        communication_missing_candidate_flag = bool(
            missing_fraction >= cfg.communication_missing_candidate_fraction
        )
        communication_missing_run = (
            communication_missing_run + 1
            if communication_missing_candidate_flag
            else 0
        )
        communication_frozen_candidate_flag = bool(
            frozen_fraction >= cfg.communication_frozen_candidate_fraction
            and not stopped_state_flag
        )
        communication_frozen_run = (
            communication_frozen_run + 1
            if communication_frozen_candidate_flag
            else 0
        )
        timestamp_gap_candidate_flag = bool(
            timestamp_gap_minutes
            > cfg.communication_timestamp_gap_multiplier
            * cfg.sampling_minutes
        )
        communication_candidate_flag = bool(
            communication_missing_candidate_flag
            or communication_frozen_candidate_flag
            or timestamp_gap_candidate_flag
        )
        communication_data_quality_confirmed_flag = bool(
            (
                missing_fraction
                >= cfg.communication_missing_confirmed_fraction
                and communication_missing_run
                >= cfg.communication_confirmation_points
            )
            or (
                frozen_fraction
                >= cfg.communication_frozen_confirmed_fraction
                and communication_frozen_run
                >= cfg.communication_confirmation_points
                and not stopped_state_flag
            )
            or timestamp_gap_minutes
            >= cfg.communication_confirmed_gap_minutes
        )
        communication_alert_confirmed_flag = bool(
            cfg.communication_red_alert_enabled
            and communication_data_quality_confirmed_flag
        )

        power_dip_flags = np.zeros(len(power_measurement_indices), dtype=bool)
        power_recovery_flags = np.zeros(len(power_measurement_indices), dtype=bool)
        power_drop_scores = np.zeros(len(power_measurement_indices), dtype=float)
        power_recovery_scores = np.zeros(len(power_measurement_indices), dtype=float)
        pre_generating_flags = np.zeros(len(power_measurement_indices), dtype=bool)
        current_generating_flags = np.zeros(
            len(power_measurement_indices), dtype=bool
        )

        # Evaluate the state immediately BEFORE the current row. This causal gate
        # prevents normal low-power/stopped operation from being interpreted as a
        # fault-like short standstill.
        for j, measurement_index in enumerate(power_measurement_indices):
            generating_history = power_generating_histories[measurement_index]
            if len(generating_history) >= cfg.short_pre_generating_points:
                pre_generating_flags[j] = bool(
                    np.mean(np.asarray(generating_history, dtype=float))
                    >= cfg.short_pre_generating_fraction
                )

        for j, measurement_index in enumerate(power_measurement_indices):
            current = row[measurement_index]
            history = power_histories[measurement_index]
            reference = float(np.nanmedian(np.asarray(history, dtype=float))) if history else np.nan
            scale = float(baseline.scale[measurement_index])

            if np.isfinite(current) and np.isfinite(reference) and abs(reference) > EPSILON:
                drop_ratio = (reference - current) / max(abs(reference), EPSILON)
                dip_z = (reference - current) / max(scale, EPSILON)
                active_threshold = max(0.10 * abs(reference), EPSILON)
                generating = reference >= active_threshold
                power_dip_flags[j] = bool(
                    generating
                    and (
                        drop_ratio >= cfg.short_minimum_power_drop_ratio
                        or dip_z >= cfg.power_dip_z_threshold
                    )
                )
                power_drop_scores[j] = max(drop_ratio, dip_z / max(cfg.power_dip_z_threshold, EPSILON), 0.0)

            # Recovery at t confirms a dip observed at t-1.  This is causal and
            # introduces one sampling-period confirmation delay.
            if (
                np.isfinite(current)
                and previous_power_dip_flags[j]
                and np.isfinite(previous_power_reference[j])
            ):
                denominator = max(abs(previous_power_reference[j]), EPSILON)
                recovery_ratio = current / denominator
                recovery_z = (current - previous_power_value[j]) / max(scale, EPSILON)
                power_recovery_flags[j] = bool(
                    recovery_ratio >= cfg.short_minimum_recovery_ratio
                    or recovery_z >= cfg.power_recovery_z_threshold
                )
                power_recovery_scores[j] = max(
                    recovery_ratio / max(cfg.short_minimum_recovery_ratio, EPSILON),
                    recovery_z / max(cfg.power_recovery_z_threshold, EPSILON),
                    0.0,
                )

            if np.isfinite(current):
                history.append(float(current))
                generating_reference = (
                    float(np.nanmedian(np.asarray(history, dtype=float)))
                    if history
                    else np.nan
                )
                generating_now = bool(
                    np.isfinite(generating_reference)
                    and generating_reference > EPSILON
                    and current >= 0.10 * abs(generating_reference)
                )
                current_generating_flags[j] = generating_now
                power_generating_histories[measurement_index].append(
                    int(generating_now)
                )
            previous_power_reference[j] = reference
            previous_power_value[j] = current

        dip_count = int(power_dip_flags.sum())
        recovery_count = int(power_recovery_flags.sum())
        dip_consensus = required_power > 0 and dip_count >= required_power
        recovery_consensus = required_power > 0 and recovery_count >= required_power
        pre_generating_count = int(pre_generating_flags.sum())
        pre_generating_consensus = bool(
            required_power > 0 and pre_generating_count >= required_power
        )
        current_generating_count = int(current_generating_flags.sum())
        current_generating_consensus = bool(
            required_power > 0
            and current_generating_count >= required_power
        )
        confirmed_drop_scores = power_drop_scores[power_dip_flags]
        power_drop_score_mean_confirmed = float(
            np.mean(confirmed_drop_scores)
        ) if confirmed_drop_scores.size else 0.0

        short_candidate_flag = bool(
            pre_stable_generation_consensus
            and abnormal_fraction_z8 >= cfg.short_fraction_floor
            and dip_consensus
        )
        short_confirmed_flag = bool(
            recovery_consensus
            and int(previous_power_dip_flags.sum()) >= required_power
        )

        # V6: distinguish ordinary operational short stops from fault-like short
        # stops. Only the latter may enter the intermittent-cluster histories.
        fault_like_short_candidate_flag = bool(
            short_candidate_flag
            and pre_generating_consensus
            and pre_stable_generation_consensus
            and abnormal_fraction_z8
            >= cfg.strong_short_minimum_abnormal_fraction
            and power_drop_score_mean_confirmed
            >= cfg.short_minimum_drop_score
        )
        # Balanced short confirmation:
        # 1. A qualified dip after stable generation opens a pending candidate.
        # 2. The low-power interval must last between configured minimum and
        #    maximum numbers of samples.
        # 3. Recovery must then remain stable for several causal samples.
        if fault_like_short_candidate_flag and not pending_fault_like_short:
            pending_fault_like_short = True
            pending_fault_like_start_time = timestamp
            pending_fault_like_age_points = 0
            pending_fault_like_recovery_seen = False
            pending_fault_like_recovery_run = 0
            pending_fault_like_dip_points = 1
            pending_fault_like_duration_valid = False

        fault_like_short_recovery_observed_flag = False
        stable_recovery_generating_flag = False
        fault_like_short_confirmed_flag = False
        short_duration_rejected_flag = False

        if pending_fault_like_short:
            pending_fault_like_age_points += 1

            if not pending_fault_like_recovery_seen:
                # Count causal samples from the initial dip until recovery.
                if pending_fault_like_age_points > 1:
                    pending_fault_like_dip_points += 1

                if short_confirmed_flag:
                    pending_fault_like_duration_valid = bool(
                        cfg.short_minimum_dip_points
                        <= pending_fault_like_dip_points
                        <= cfg.short_maximum_dip_points
                    )
                    if pending_fault_like_duration_valid:
                        pending_fault_like_recovery_seen = True
                        fault_like_short_recovery_observed_flag = True
                    else:
                        short_duration_rejected_flag = True

            if pending_fault_like_recovery_seen:
                stable_recovery_generating_flag = bool(
                    stable_generation_now and current_generating_consensus
                )
                if stable_recovery_generating_flag:
                    pending_fault_like_recovery_run += 1
                else:
                    pending_fault_like_recovery_run = 0

                required_stable_points = max(
                    1, cfg.short_stable_recovery_points
                )
                required_stable_count = max(
                    1,
                    int(
                        math.ceil(
                            required_stable_points
                            * cfg.short_stable_recovery_fraction
                        )
                    ),
                )
                fault_like_short_confirmed_flag = bool(
                    pending_fault_like_duration_valid
                    and pending_fault_like_recovery_run
                    >= required_stable_count
                )

            dip_duration_expired = bool(
                not pending_fault_like_recovery_seen
                and pending_fault_like_dip_points
                > cfg.short_maximum_dip_points
            )
            pending_timeout = bool(
                pending_fault_like_age_points
                >= cfg.short_pending_timeout_points()
            )

            if (
                fault_like_short_confirmed_flag
                or short_duration_rejected_flag
                or dip_duration_expired
                or pending_timeout
            ):
                pending_fault_like_short = False
                pending_fault_like_start_time = None
                pending_fault_like_age_points = 0
                pending_fault_like_recovery_seen = False
                pending_fault_like_recovery_run = 0
                pending_fault_like_dip_points = 0
                pending_fault_like_duration_valid = False

        previous_power_dip_flags = power_dip_flags.copy()

        # Count episode starts rather than True samples. The cluster histories now
        # contain only fault-like short episodes; ordinary short stops remain review
        # evidence and cannot create an intermittent red alert by themselves.
        short_episode_start_flag = bool(
            short_confirmed_flag and not previous_short_confirmed_flag
        )
        raw_fault_like_short_episode_start_flag = bool(
            fault_like_short_confirmed_flag
            and not previous_fault_like_short_confirmed_flag
        )

        independent_gap_ok = bool(
            last_independent_fault_like_short_time is None
            or (
                timestamp - last_independent_fault_like_short_time
            ).total_seconds()
            >= cfg.short_independent_min_gap_minutes * 60.0
        )
        fault_like_short_episode_start_flag = bool(
            raw_fault_like_short_episode_start_flag
            and independent_gap_ok
        )
        if fault_like_short_episode_start_flag:
            last_independent_fault_like_short_time = timestamp

        previous_short_confirmed_flag = short_confirmed_flag
        previous_fault_like_short_confirmed_flag = (
            fault_like_short_confirmed_flag
        )

        short_history_2h.append(int(fault_like_short_episode_start_flag))
        short_history_6h.append(int(fault_like_short_episode_start_flag))
        short_history_24h.append(int(fault_like_short_episode_start_flag))
        recent_short_start_history.append(
            int(fault_like_short_episode_start_flag)
        )
        short_episode_count_2h = int(sum(short_history_2h))
        short_episode_count_6h = int(sum(short_history_6h))
        short_episode_count_24h = int(sum(short_history_24h))
        recent_short_episode_count = int(sum(recent_short_start_history))

        weak_short_cluster_flag = bool(
            short_episode_count_2h >= cfg.intermittent_minimum_short_flags
        )

        # V5 separates repeated occurrence from fault quality. A frequent short
        # pattern becomes a red trigger only when the newest short is accompanied
        # by a sufficiently broad SCADA disturbance or another detector family.
        background_history_ready = bool(
            len(background_active_history)
            >= cfg.background_minimum_history_points()
        )
        background_short_count = int(sum(background_short_history))
        background_active_fraction = (
            float(np.mean(np.asarray(background_active_history, dtype=float)))
            if background_active_history
            else 0.0
        )
        background_cycling_flag = bool(
            cfg.background_suppression_enabled
            and background_history_ready
            and (
                background_active_fraction
                >= cfg.background_active_fraction_threshold
                or background_short_count
                >= cfg.background_short_count_threshold
            )
        )

        effective_strong_short_count_2h = (
            cfg.background_strong_short_count_2h
            if background_cycling_flag
            else cfg.strong_short_count_2h
        )
        effective_strong_short_count_6h = (
            cfg.background_strong_short_count_6h
            if background_cycling_flag
            else cfg.strong_short_count_6h
        )
        effective_strong_short_count_24h = (
            cfg.background_strong_short_count_24h
            if background_cycling_flag
            else cfg.strong_short_count_24h
        )

        short_cluster_count_condition = bool(
            short_episode_count_2h
            >= effective_strong_short_count_2h
            or (
                short_episode_count_6h
                >= effective_strong_short_count_6h
                and short_episode_count_24h
                >= effective_strong_short_count_24h
            )
        )
        # Persistent/localized quality evidence is added after those detectors
        # are computed for the current row.
        strong_short_quality_flag = False
        strong_short_trigger_flag = False

        # Short evidence supports an open alert only for a short recent interval,
        # rather than for the whole trailing two-hour cluster window.
        short_recent_support_flag = bool(
            fault_like_short_confirmed_flag
            or recent_short_episode_count > 0
        )
        short_cluster_support_flag = short_recent_support_flag

        persistent_fraction_history.append(abnormal_fraction_z12)
        persistent_smoothed_fraction = float(np.mean(persistent_fraction_history))
        persistent_raw_flag = bool(
            len(persistent_fraction_history) >= cfg.persistent_smoothing_points
            and persistent_smoothed_fraction >= cfg.persistent_fraction_threshold
        )
        persistent_run = persistent_run + 1 if persistent_raw_flag else 0
        persistent_flag = persistent_run >= cfg.persistent_minimum_points

        localized_active = base_z >= cfg.localized_z_threshold
        localized_active_history.append(localized_active.copy())
        active_stack = np.stack(localized_active_history, axis=0)
        localized_coverage = active_stack.mean(axis=0)
        stable_mask = localized_coverage >= cfg.localized_minimum_signal_coverage
        stable_count = int(stable_mask.sum())

        current_overlap = jaccard(localized_active, previous_active_set)
        previous_active_set = localized_active.copy()
        localized_overlap_history.append(current_overlap)
        overlap_smoothed = float(np.mean(localized_overlap_history))

        localized_strength = top_k_mean(
            base_z, cfg.localized_minimum_abnormal_signals
        )
        localized_strength_history.append(localized_strength)
        localized_strength_smoothed = float(np.mean(localized_strength_history))

        localized_raw_flag = bool(
            len(localized_active_history) >= cfg.localized_coverage_window_points
            and stable_count >= cfg.localized_minimum_stable_signals
            and overlap_smoothed >= cfg.localized_minimum_overlap
            and localized_strength_smoothed >= cfg.localized_strength_threshold
        )
        localized_run = localized_run + 1 if localized_raw_flag else 0
        localized_candidate_flag = localized_run >= cfg.localized_minimum_points

        # V5 boundary confirmation. Start timing from the raw localized segment,
        # because the candidate itself is delayed by localized_minimum_points.
        localized_raw_start_flag = bool(
            localized_raw_flag and not previous_localized_raw_flag
        )
        localized_raw_end_flag = bool(
            previous_localized_raw_flag and not localized_raw_flag
        )
        if localized_raw_start_flag:
            localized_raw_start_time = timestamp
            localized_confirmation_latched = False
            localized_confirmation_time = None
            localized_power_dip_history.clear()

        localized_power_dip_history.append((timestamp, bool(dip_consensus)))

        localized_age_minutes = float("nan")
        if localized_raw_start_time is not None:
            localized_age_minutes = (
                timestamp - localized_raw_start_time
            ).total_seconds() / 60.0

        localized_boundary_dip_flag = bool(
            localized_raw_start_time is not None
            and any(
                dip
                and time >= localized_raw_start_time
                for time, dip in localized_power_dip_history
            )
        )
        localized_confirmation_trigger_flag = bool(
            generating_state_flag
            and localized_candidate_flag
            and not localized_confirmation_latched
            and np.isfinite(localized_age_minutes)
            and localized_age_minutes <= cfg.localized_confirmation_window_minutes
            and localized_boundary_dip_flag
        )
        if localized_confirmation_trigger_flag:
            localized_confirmation_latched = True
            localized_confirmation_time = timestamp

        localized_confirmation_age_points = 0
        if localized_confirmation_time is not None:
            localized_confirmation_age_points = max(
                0,
                int(
                    round(
                        (timestamp - localized_confirmation_time).total_seconds()
                        / 60.0
                        / cfg.sampling_minutes
                    )
                ),
            )

        localized_confirmed_flag = bool(
            localized_candidate_flag
            and localized_confirmation_latched
            and localized_confirmation_age_points
            <= cfg.localized_confirmed_support_points()
        )

        if localized_raw_end_flag:
            localized_raw_start_time = None
            localized_confirmation_latched = False
            localized_confirmation_time = None
            localized_power_dip_history.clear()

        previous_localized_raw_flag = localized_raw_flag

        # Generic causal slow-trend detector. It uses standardized residual history
        # and is enabled only during stable generation to reduce load-state effects.
        signed_standardized = np.zeros_like(row, dtype=float)
        valid_standardized = baseline.usable & np.isfinite(row)
        signed_standardized[valid_standardized] = (
            row[valid_standardized] - baseline.mean[valid_standardized]
        ) / baseline.scale[valid_standardized]
        standardized_history.append(signed_standardized.copy())

        slow_trend_max_abs_slope = 0.0
        slow_trend_max_abs_shift = 0.0
        slow_trend_signal_count = 0
        if (
            generating_state_flag
            and len(standardized_history)
            >= cfg.slow_trend_window_points()
        ):
            history_array = np.stack(standardized_history, axis=0)
            recent = history_array[-cfg.slow_trend_window_points():]
            sampling_hours = cfg.sampling_minutes / 60.0
            slopes = np.asarray(
                [
                    causal_slope(recent[:, index], sampling_hours)
                    for index in range(recent.shape[1])
                ],
                dtype=float,
            )
            slow_trend_max_abs_slope = float(
                np.nanmax(np.abs(slopes))
            ) if slopes.size else 0.0

            if len(standardized_history) >= cfg.slow_trend_shift_points():
                older = history_array[
                    : max(
                        1,
                        history_array.shape[0]
                        - cfg.slow_trend_window_points(),
                    )
                ]
                recent_median = np.nanmedian(recent, axis=0)
                older_median = np.nanmedian(older, axis=0)
                shifts = recent_median - older_median
            else:
                shifts = np.nanmedian(recent, axis=0)

            slow_trend_max_abs_shift = float(
                np.nanmax(np.abs(shifts))
            ) if shifts.size else 0.0
            slow_mask = (
                np.abs(slopes)
                >= cfg.slow_trend_minimum_slope_z_per_hour
            ) & (
                np.abs(shifts)
                >= cfg.slow_trend_minimum_shift_z
            )
            slow_trend_signal_count = int(np.nansum(slow_mask))

        slow_trend_candidate_flag = bool(
            generating_state_flag
            and slow_trend_signal_count
            >= cfg.slow_trend_minimum_signals
        )
        slow_trend_run = (
            slow_trend_run + 1
            if slow_trend_candidate_flag
            else 0
        )
        slow_trend_pattern_confirmed_flag = bool(
            slow_trend_run >= cfg.slow_trend_confirmation_points
        )
        slow_trend_alert_confirmed_flag = bool(
            cfg.slow_trend_red_alert_enabled
            and slow_trend_pattern_confirmed_flag
        )

        # Finalise short trigger after other detector evidence is available.
        # are available for the current row.
        strong_short_quality_flag = bool(
            fault_like_short_episode_start_flag
            and (
                persistent_raw_flag
                or localized_candidate_flag
                or abnormal_fraction_z8
                >= cfg.strong_short_minimum_abnormal_fraction
            )
        )
        background_override_flag = bool(localized_confirmed_flag)

        short_cluster_qualified_flag = bool(
            fault_like_short_episode_start_flag
            and short_cluster_count_condition
            and strong_short_quality_flag
        )

        # Edge-trigger a repeated-short burst once. The latch is re-armed only
        # after the causal six-hour short burden has remained low long enough.
        if (
            short_episode_count_6h
            <= cfg.short_cluster_rearm_max_count_6h
            and not fault_like_short_episode_start_flag
        ):
            short_cluster_quiet_run += 1
        else:
            short_cluster_quiet_run = 0

        short_cluster_rearm_flag = bool(
            short_cluster_latched
            and short_cluster_quiet_run >= cfg.short_cluster_quiet_points()
        )
        if short_cluster_rearm_flag:
            short_cluster_latched = False
            short_cluster_quiet_run = 0

        strong_short_trigger_flag = bool(
            short_cluster_qualified_flag
            and not short_cluster_latched
        )
        if strong_short_trigger_flag:
            short_cluster_latched = True

        intermittent_raw_flag = bool(
            short_cluster_count_condition
            and short_recent_support_flag
        )

        short_score = min(1.0, abnormal_fraction_z8 / max(cfg.short_fraction_floor, EPSILON)) if short_candidate_flag or short_confirmed_flag else 0.0
        cluster_score = max(
            min(
                1.0,
                short_episode_count_2h
                / max(effective_strong_short_count_2h, 1),
            ),
            min(
                1.0,
                short_episode_count_24h
                / max(effective_strong_short_count_24h, 1),
            ),
        )
        persistent_score = min(1.0, persistent_smoothed_fraction / max(cfg.persistent_fraction_threshold, EPSILON))
        localized_score = min(
            1.0,
            0.5 * stable_count / max(cfg.localized_minimum_stable_signals, 1)
            + 0.5 * localized_strength_smoothed / max(cfg.localized_strength_threshold, EPSILON),
        )

        # ------------------------------------------------------------------
        # Two evidence levels
        # ------------------------------------------------------------------
        # A localized sensor-only candidate is intentionally REVIEW evidence
        # only.  In the previous version it was counted as a confirmed detector
        # family and could promote itself to a red alert after three points.
        # That produced long false-alert episodes and froze the baseline.
        short_review_flag = bool(short_candidate_flag or short_confirmed_flag)
        review_family_count = (
            int(short_review_flag)
            + int(persistent_raw_flag)
            + int(localized_candidate_flag)
            + int(communication_candidate_flag)
            + int(slow_trend_candidate_flag)
            + int(status_candidate_flag)
            + int(semantic_candidate_flag)
        )
        # Generic semantic fusion is now strict and edge-triggered. Review-only
        # localized/persistent/short-support evidence cannot corroborate it.
        semantic_independent_corroboration_flag = bool(
            localized_confirmed_flag
            or strong_short_trigger_flag
            or (
                status_confirmed_flag
                and (status_novel_flag or status_rare_flag)
            )
        )
        semantic_group_count_gate = bool(
            1 <= semantic_confirmed_group_count
            <= cfg.semantic_fusion_maximum_group_count
        )
        semantic_fusion_candidate_flag = bool(
            semantic_new_burst_flag
            and semantic_group_count_gate
            and semantic_independent_corroboration_flag
        )
        semantic_fusion_run = semantic_fusion_run + 1 if semantic_fusion_candidate_flag else 0
        if semantic_fusion_candidate_flag:
            semantic_fusion_quiet_run = 0
        else:
            semantic_fusion_quiet_run += 1
        semantic_fusion_rearm_flag = bool(
            semantic_fusion_latched
            and semantic_fusion_quiet_run >= cfg.semantic_fusion_quiet_points()
        )
        if semantic_fusion_rearm_flag:
            semantic_fusion_latched = False
            semantic_fusion_quiet_run = 0
        semantic_fusion_trigger_flag = bool(
            semantic_fusion_run >= cfg.semantic_fusion_confirmation_points
            and not semantic_fusion_latched
        )
        if semantic_fusion_trigger_flag:
            semantic_fusion_latched = True
        semantic_fusion_confirmed_flag = bool(semantic_fusion_trigger_flag)
        semantic_fusion_support_flag = bool(
            semantic_fusion_trigger_flag or any(semantic_fusion_support_history)
        )

        # Targeted subsystem onset detectors for fault families repeatedly missed
        # by V3. They remain causal and require a new high-severity, low-burden
        # onset in no more than two physical groups.
        targeted_group_whitelist = {
            "yaw", "gearbox_oil", "hydraulic_brake", "dc_link_voltage",
            "auxiliary_supply", "cooling_system", "converter_electrical",
            "grid_voltage", "pitch_axis",
        }
        targeted_change_candidate_groups: list[str] = []
        targeted_change_trigger_groups: list[str] = []
        for group_name in semantic_groups:
            group_candidate = bool(
                group_name in targeted_group_whitelist
                and group_name in semantic_onset_groups
                and semantic_group_max_z.get(group_name, 0.0)
                    >= cfg.targeted_change_z_threshold
                and group_name not in semantic_chronic_groups
            )
            targeted_group_runs[group_name] = (
                targeted_group_runs[group_name] + 1 if group_candidate else 0
            )
            if group_candidate:
                targeted_change_candidate_groups.append(group_name)
                targeted_group_quiet_runs[group_name] = 0
            else:
                targeted_group_quiet_runs[group_name] += 1
            if (
                targeted_group_latched[group_name]
                and targeted_group_quiet_runs[group_name]
                    >= cfg.targeted_change_quiet_points()
            ):
                targeted_group_latched[group_name] = False
                targeted_group_quiet_runs[group_name] = 0
            if (
                targeted_group_runs[group_name]
                    >= cfg.targeted_change_confirmation_points
                and not targeted_group_latched[group_name]
            ):
                targeted_change_trigger_groups.append(group_name)
                targeted_group_latched[group_name] = True

        targeted_change_candidate_flag = bool(
            1 <= len(targeted_change_candidate_groups)
            <= cfg.targeted_change_maximum_group_count
        )
        targeted_change_trigger_flag = bool(
            1 <= len(targeted_change_trigger_groups)
            <= cfg.targeted_change_maximum_group_count
        )
        targeted_change_support_flag = bool(
            targeted_change_trigger_flag or any(targeted_change_support_history)
        )
        targeted_change_score = float(
            max(
                [semantic_group_scores.get(name, 0.0) for name in targeted_change_trigger_groups],
                default=0.0,
            )
        )

        # Communication/control fault path: generic data-quality confirmation is
        # not sufficient alone. It must coincide with a rare/novel status pattern.
        communication_control_candidate_flag = bool(
            communication_data_quality_confirmed_flag
            and status_candidate_flag
            and (status_novel_flag or status_rare_flag or status_transition_burst_flag)
        )
        communication_control_run = (
            communication_control_run + 1 if communication_control_candidate_flag else 0
        )
        communication_control_confirmed_flag = bool(
            communication_control_run >= cfg.communication_control_confirmation_points
        )

        # Status is fusion-only unless explicitly enabled for an ablation.
        status_red_confirmed_flag = bool(
            cfg.status_independent_red_enabled and status_confirmed_flag
        )

        confirmed_family_count = (
            int(strong_short_trigger_flag)
            + int(localized_confirmed_flag)
            + int(communication_alert_confirmed_flag)
            + int(communication_control_confirmed_flag)
            + int(slow_trend_alert_confirmed_flag)
            + int(status_red_confirmed_flag)
            + int(semantic_fusion_trigger_flag)
            + int(targeted_change_trigger_flag)
        )

        review_evidence_flag = bool(
            short_review_flag
            or weak_short_cluster_flag
            or persistent_raw_flag
            or localized_candidate_flag
            or communication_candidate_flag
            or slow_trend_candidate_flag
            or status_candidate_flag
            or semantic_candidate_flag
        )
        short_cross_detector_flag = bool(
            fault_like_short_episode_start_flag
            and (
                localized_confirmed_flag
                or status_red_confirmed_flag
                or targeted_change_trigger_flag
                or communication_control_confirmed_flag
                or semantic_fusion_trigger_flag
                or communication_alert_confirmed_flag
                or slow_trend_alert_confirmed_flag
            )
        )
        confirmed_evidence_flag = bool(
            strong_short_trigger_flag
            or localized_confirmed_flag
            or communication_alert_confirmed_flag
            or slow_trend_alert_confirmed_flag
            or status_red_confirmed_flag
            or communication_control_confirmed_flag
            or semantic_fusion_trigger_flag
            or targeted_change_trigger_flag
            or short_cross_detector_flag
        )

        # Localized candidate evidence may create a yellow review state, but its
        # score is capped below the immediate red-alert threshold unless power
        # evidence confirms it.  The multi-detector bonus uses confirmed
        # families only.
        alert_score = float(
            max(
                0.60 * short_score if short_candidate_flag else 0.0,
                0.70 * short_score if short_confirmed_flag else 0.0,
                0.65 * cluster_score if weak_short_cluster_flag else 0.0,
                0.90 * cluster_score if strong_short_trigger_flag else 0.0,
                0.95 * persistent_score if persistent_flag else 0.55 * persistent_score if persistent_raw_flag else 0.0,
                0.95 * localized_score if localized_confirmed_flag else 0.55 * localized_score if localized_candidate_flag else 0.0,
                0.98 if communication_alert_confirmed_flag else 0.60 if communication_candidate_flag else 0.0,
                0.92 if slow_trend_alert_confirmed_flag else 0.58 if slow_trend_candidate_flag else 0.0,
                status_score if status_red_confirmed_flag else 0.65 * status_score,
                0.98 if communication_control_confirmed_flag else 0.0,
                0.97 * targeted_change_score if targeted_change_trigger_flag else 0.0,
                0.96 * semantic_score
                if semantic_fusion_trigger_flag
                else 0.68 * semantic_score
                if semantic_confirmed_flag
                else 0.55 * semantic_score
                if semantic_candidate_flag
                else 0.0,
                min(1.0, 0.55 + 0.20 * confirmed_family_count)
                if confirmed_family_count >= 2
                else 0.0,
            )
        )

        # Two simultaneous review families may be promoted after the configured
        # persistence period.  A localized-only candidate can never promote
        # itself to a red alert.
        candidate_persistent_flag = bool(
            candidate_run >= cfg.candidate_confirmation_points
        )
        multi_review_support_flag = bool(review_family_count >= 2)
        # Diagnostic only in V3. Generic combinations of review evidence can no
        # longer open a red episode, even if an old command line contains the
        # legacy enable flag.
        multi_review_trigger_flag = False
        active_trigger_flag = bool(confirmed_evidence_flag)
        active_support_flag = bool(
            short_recent_support_flag
            or localized_confirmed_flag
            or communication_alert_confirmed_flag
            or slow_trend_alert_confirmed_flag
            or (status_support_flag and status_red_confirmed_flag)
            or communication_control_confirmed_flag
            or semantic_fusion_support_flag
            or targeted_change_support_flag
        )
        strong_evidence_flag = bool(
            confirmed_family_count >= 2
            or alert_score >= cfg.immediate_alert_score
        )

        # Keep legacy output name for downstream compatibility.  It now means
        # evidence capable of supporting a red alert, not every raw candidate.
        evidence_flag = active_support_flag
        detector_family_count = confirmed_family_count

        if stream_state == "normal":
            if active_trigger_flag:
                # Confirmed evidence can open an alert immediately; no future
                # data and no extra one-row delay are required.
                stream_state = "active_alert"
                candidate_run = 0
                candidate_start_time = timestamp
                active_start_time = timestamp
                last_evidence_time = timestamp
                recovery_run = 0
                episode_counter += 1
                current_episode_rows = []
            elif review_evidence_flag:
                stream_state = "candidate"
                candidate_run = 1
                candidate_start_time = timestamp
                last_evidence_time = timestamp
                current_episode_rows = []
            else:
                candidate_run = 0

        elif stream_state == "candidate":
            if active_trigger_flag:
                stream_state = "active_alert"
                active_start_time = timestamp
                last_evidence_time = timestamp
                recovery_run = 0
                episode_counter += 1
                # Do not clear current_episode_rows: retain the yellow lead-in.
            elif review_evidence_flag:
                candidate_run += 1
                last_evidence_time = timestamp
                if multi_review_trigger_flag:
                    stream_state = "active_alert"
                    active_start_time = timestamp
                    recovery_run = 0
                    episode_counter += 1
            else:
                stream_state = "normal"
                candidate_run = 0
                candidate_start_time = None
                last_evidence_time = None
                current_episode_rows = []

        elif stream_state == "active_alert":
            if active_support_flag:
                last_evidence_time = timestamp
                recovery_run = 0
            else:
                stream_state = "recovery_pending"
                recovery_run = 1

        elif stream_state == "recovery_pending":
            if active_support_flag:
                stream_state = "active_alert"
                recovery_run = 0
                last_evidence_time = timestamp
            else:
                recovery_run += 1
                if recovery_run >= cfg.recovery_confirmation_points:
                    # The red alert is cleared, but the logical episode remains
                    # open for a causal merge cooldown. Related evidence returning
                    # within this gap reopens the same episode instead of creating
                    # another fragment.
                    stream_state = "inactive_pending"
                    inactive_pending_run = 0
                    recovery_run = 0

        elif stream_state == "inactive_pending":
            if active_trigger_flag:
                stream_state = "active_alert"
                inactive_pending_run = 0
                recovery_run = 0
                last_evidence_time = timestamp
            else:
                inactive_pending_run += 1
                if inactive_pending_run >= cfg.episode_merge_gap_points():
                    episode_start = candidate_start_time or active_start_time or timestamp
                    episode_end = last_evidence_time or timestamp
                    overlap, meta_id, meta_label, overlap_hours = overlap_episode_with_metadata(
                        episode_start, episode_end, source_metadata
                    )
                    frame = pd.DataFrame(current_episode_rows)
                    detectors: list[str] = []
                    for detector_name, column in [
                        ("fault_like_short_standstill", "strong_short_trigger_flag"),
                        ("intermittent_cluster", "strong_short_trigger_flag"),
                        ("persistent_system_state", "persistent_system_state_flag"),
                        ("localized_persistent_subsystem_state", "localized_confirmed_flag"),
                        ("communication_data_quality", "communication_alert_confirmed_flag"),
                        ("slow_trend", "slow_trend_alert_confirmed_flag"),
                        ("status_code", "status_red_confirmed_flag"),
                        ("communication_control_fault", "communication_control_confirmed_flag"),
                        ("semantic_subsystem_fusion", "semantic_fusion_trigger_flag"),
                        ("targeted_subsystem_change", "targeted_change_trigger_flag"),
                    ]:
                        if not frame.empty and bool(
                            frame.get(column, pd.Series(dtype=bool)).fillna(False).any()
                        ):
                            detectors.append(detector_name)
                    if not detectors:
                        detectors.append("unclassified_confirmed_episode")

                    episodes.append(
                        Episode(
                            farm_id=farm_id,
                            source_id=source_id,
                            source_file=str(event_file),
                            asset_id=asset_id,
                            episode_id=f"{farm_id}_{source_id}_stream_{episode_counter:03d}",
                            start_time=episode_start,
                            end_time=episode_end,
                            first_active_time=active_start_time or episode_start,
                            last_evidence_time=episode_end,
                            duration_hours=max(0.0, (episode_end - episode_start).total_seconds() / 3600.0),
                            right_censored=False,
                            max_alert_score=float(frame["alert_score"].max()) if not frame.empty else alert_score,
                            max_abnormal_fraction_z8=float(frame["abnormal_fraction_z8"].max()) if not frame.empty else abnormal_fraction_z8,
                            max_abnormal_fraction_z12=float(frame["abnormal_fraction_z12"].max()) if not frame.empty else abnormal_fraction_z12,
                            short_confirmed_count=int(frame["short_confirmed_flag"].sum()) if not frame.empty else 0,
                            short_episode_count=int(frame.get("short_episode_start_flag", pd.Series(dtype=bool)).fillna(False).sum()) if not frame.empty else 0,
                            fault_like_short_episode_count=int(frame.get("fault_like_short_episode_start_flag", pd.Series(dtype=bool)).fillna(False).sum()) if not frame.empty else 0,
                            intermittent_flag_points=int(frame["intermittent_cluster_flag"].sum()) if not frame.empty else 0,
                            persistent_flag_points=int(frame["persistent_system_state_flag"].sum()) if not frame.empty else 0,
                            localized_candidate_points=int(frame["localized_candidate_flag"].sum()) if not frame.empty else 0,
                            localized_confirmed_points=int(frame["localized_confirmed_flag"].sum()) if not frame.empty else 0,
                            detector_families=";".join(detectors),
                            metadata_overlap=overlap,
                            metadata_event_id=meta_id,
                            metadata_label=meta_label,
                            metadata_overlap_hours=overlap_hours,
                        )
                    )
                    stream_state = "normal"
                    candidate_run = 0
                    recovery_run = 0
                    inactive_pending_run = 0
                    candidate_start_time = None
                    active_start_time = None
                    last_evidence_time = None
                    current_episode_rows = []

        active_alert_flag = stream_state in {"active_alert", "recovery_pending"}
        review_flag = bool(
            stream_state == "candidate"
            or (
                not active_alert_flag
                and review_evidence_flag
                and alert_score >= cfg.review_score
            )
        )

        # Do not freeze the adaptive baseline for a localized-only sensor
        # candidate.  Freeze it for power/short evidence, persistent evidence,
        # confirmed localized evidence, or simultaneous review families.
        baseline_freeze_flag = bool(
            short_candidate_flag
            or short_confirmed_flag
            or short_recent_support_flag
            or persistent_raw_flag
            or localized_confirmation_trigger_flag
            or localized_confirmed_flag
            or communication_alert_confirmed_flag
            or slow_trend_alert_confirmed_flag
            or review_family_count >= 2
        )
        baseline_update_allowed = bool(
            stream_state in {"normal", "candidate"}
            and operating_state in {"generating", "stopped"}
            and not baseline_freeze_flag
            and abnormal_fraction_z8 < cfg.baseline_freeze_abnormal_fraction
        )
        if baseline_update_allowed:
            baseline.update(row)

        output_row = {
            "farm_id": farm_id,
            "source_id": source_id,
            "source_file": str(event_file),
            "asset_id": asset_id,
            "timestamp": timestamp,
            "row_id": row_id,
            "warmup_complete": True,
            "stream_state": stream_state,
            "baseline_updated": baseline_update_allowed,
            "operating_state": operating_state,
            "power_high_reference": power_high_reference,
            "current_power_median": current_power_median,
            "power_ratio": power_ratio,
            "generating_state_flag": generating_state_flag,
            "stable_generation_now": stable_generation_now,
            "pre_stable_generation_fraction": pre_stable_generation_fraction,
            "pre_stable_generation_consensus": pre_stable_generation_consensus,
            "transition_state_flag": transition_state_flag,
            "stopped_state_flag": stopped_state_flag,
            "missing_fraction": missing_fraction,
            "frozen_fraction": frozen_fraction,
            "timestamp_gap_minutes": timestamp_gap_minutes,
            "communication_candidate_flag": communication_candidate_flag,
            "communication_data_quality_confirmed_flag": communication_data_quality_confirmed_flag,
            "communication_alert_confirmed_flag": communication_alert_confirmed_flag,
            # Legacy alias now means alert-eligible confirmation.
            "communication_confirmed_flag": communication_alert_confirmed_flag,
            "communication_missing_run": communication_missing_run,
            "communication_frozen_run": communication_frozen_run,
            "slow_trend_max_abs_slope": slow_trend_max_abs_slope,
            "slow_trend_max_abs_shift": slow_trend_max_abs_shift,
            "slow_trend_signal_count": slow_trend_signal_count,
            "slow_trend_candidate_flag": slow_trend_candidate_flag,
            "slow_trend_pattern_confirmed_flag": slow_trend_pattern_confirmed_flag,
            "slow_trend_alert_confirmed_flag": slow_trend_alert_confirmed_flag,
            # Legacy alias now means alert-eligible confirmation.
            "slow_trend_confirmed_flag": slow_trend_alert_confirmed_flag,
            "slow_trend_run_points": slow_trend_run,
            "abnormal_fraction_z8": abnormal_fraction_z8,
            "abnormal_fraction_z12": abnormal_fraction_z12,
            "top3_base_z_mean": top_k_mean(base_z, 3),
            "top10_base_z_mean": top_k_mean(base_z, 10),
            "power_signal_count": len(power_measurement_indices),
            "required_power_signal_count": required_power,
            "power_dip_signal_count": dip_count,
            "power_recovery_signal_count": recovery_count,
            "power_dip_consensus": dip_consensus,
            "power_recovery_consensus": recovery_consensus,
            "power_dip_score_max": float(np.max(power_drop_scores)) if power_drop_scores.size else 0.0,
            "power_recovery_score_max": float(np.max(power_recovery_scores)) if power_recovery_scores.size else 0.0,
            "pre_generating_signal_count": pre_generating_count,
            "pre_generating_consensus": pre_generating_consensus,
            "power_drop_score_mean_confirmed": power_drop_score_mean_confirmed,
            "short_candidate_flag": short_candidate_flag,
            "fault_like_short_candidate_flag": fault_like_short_candidate_flag,
            "fault_like_short_recovery_observed_flag": fault_like_short_recovery_observed_flag,
            "stable_recovery_generating_flag": stable_recovery_generating_flag,
            "pending_fault_like_short": pending_fault_like_short,
            "pending_fault_like_age_points": pending_fault_like_age_points,
            "pending_fault_like_dip_points": pending_fault_like_dip_points,
            "pending_fault_like_duration_valid": pending_fault_like_duration_valid,
            "short_duration_rejected_flag": short_duration_rejected_flag,
            "pending_fault_like_recovery_run": pending_fault_like_recovery_run,
            "fault_like_short_confirmed_flag": fault_like_short_confirmed_flag,
            "current_generating_signal_count": current_generating_count,
            "current_generating_consensus": current_generating_consensus,
            "short_confirmed_flag": short_confirmed_flag,
            "short_episode_start_flag": short_episode_start_flag,
            "raw_fault_like_short_episode_start_flag": raw_fault_like_short_episode_start_flag,
            "independent_fault_like_short_gap_ok": independent_gap_ok,
            "fault_like_short_episode_start_flag": fault_like_short_episode_start_flag,
            "short_episode_count_trailing_2h": short_episode_count_2h,
            "short_episode_count_trailing_6h": short_episode_count_6h,
            "short_episode_count_trailing_24h": short_episode_count_24h,
            # Legacy aliases retained for downstream scripts.
            "short_count_trailing_2h": short_episode_count_2h,
            "short_count_trailing_24h": short_episode_count_24h,
            "weak_short_cluster_flag": weak_short_cluster_flag,
            "short_cluster_qualified_flag": short_cluster_qualified_flag,
            "strong_short_trigger_flag": strong_short_trigger_flag,
            "strong_short_cluster_flag": strong_short_trigger_flag,
            "short_cluster_latched": short_cluster_latched,
            "short_cluster_quiet_run": short_cluster_quiet_run,
            "short_cluster_rearm_flag": short_cluster_rearm_flag,
            "short_cluster_support_flag": short_cluster_support_flag,
            "short_recent_support_flag": short_recent_support_flag,
            "recent_short_episode_count": recent_short_episode_count,
            "strong_short_quality_flag": strong_short_quality_flag,
            "background_history_ready": background_history_ready,
            "background_short_count_7d": background_short_count,
            "background_active_fraction_7d": background_active_fraction,
            "background_cycling_flag": background_cycling_flag,
            "effective_strong_short_count_2h": effective_strong_short_count_2h,
            "effective_strong_short_count_6h": effective_strong_short_count_6h,
            "effective_strong_short_count_24h": effective_strong_short_count_24h,
            "background_override_flag": background_override_flag,
            "short_cross_detector_flag": short_cross_detector_flag,
            "intermittent_cluster_flag": intermittent_raw_flag,
            "persistent_smoothed_fraction": persistent_smoothed_fraction,
            "persistent_raw_flag": persistent_raw_flag,
            "persistent_run_points": persistent_run,
            "persistent_system_state_flag": persistent_flag,
            "localized_active_signal_count": int(localized_active.sum()),
            "localized_stable_signal_count": stable_count,
            "localized_sensor_overlap": current_overlap,
            "localized_sensor_overlap_smoothed": overlap_smoothed,
            "localized_strength": localized_strength,
            "localized_strength_smoothed": localized_strength_smoothed,
            "localized_raw_flag": localized_raw_flag,
            "localized_run_points": localized_run,
            "localized_candidate_flag": localized_candidate_flag,
            "localized_raw_start_flag": localized_raw_start_flag,
            "localized_raw_end_flag": localized_raw_end_flag,
            "localized_age_minutes": localized_age_minutes,
            "localized_boundary_dip_flag": localized_boundary_dip_flag,
            "localized_confirmation_trigger_flag": localized_confirmation_trigger_flag,
            "localized_confirmation_latched": localized_confirmation_latched,
            "localized_confirmation_age_points": localized_confirmation_age_points,
            "localized_confirmed_flag": localized_confirmed_flag,
            "status_type_id": current_status,
            "status_history_ready": status_history_ready,
            "status_frequency": status_frequency,
            "status_run_points": current_status_run,
            "status_novel_flag": status_novel_flag,
            "status_rare_flag": status_rare_flag,
            "status_transition_flag": status_transition_flag,
            "status_transition_count_1h": status_transition_count,
            "status_transition_burst_flag": status_transition_burst_flag,
            "status_candidate_flag": status_candidate_flag,
            "status_confirmed_flag": status_confirmed_flag,
            "status_support_flag": status_support_flag,
            "status_score": status_score,
            "semantic_candidate_group_count": semantic_candidate_group_count,
            "semantic_confirmed_group_count": semantic_confirmed_group_count,
            "semantic_onset_group_count": semantic_onset_group_count,
            "semantic_chronic_group_count": semantic_chronic_group_count,
            "semantic_candidate_groups": ";".join(semantic_candidate_groups),
            "semantic_confirmed_groups": ";".join(semantic_confirmed_groups),
            "semantic_onset_groups": ";".join(semantic_onset_groups),
            "semantic_chronic_groups": ";".join(semantic_chronic_groups),
            "semantic_group_abnormal_counts": json.dumps(
                semantic_group_abnormal_counts, sort_keys=True
            ),
            "semantic_group_max_z": json.dumps(
                semantic_group_max_z, sort_keys=True
            ),
            "semantic_group_burden_fractions": json.dumps(
                semantic_group_burden_fractions, sort_keys=True
            ),
            "semantic_multi_group_candidate_flag": semantic_multi_group_candidate_flag,
            "semantic_multi_group_run": semantic_multi_group_run,
            "semantic_multi_group_confirmed_flag": semantic_multi_group_confirmed_flag,
            "semantic_candidate_flag": semantic_candidate_flag,
            "semantic_confirmed_flag": semantic_confirmed_flag,
            "semantic_onset_flag": semantic_onset_flag,
            "semantic_chronic_flag": semantic_chronic_flag,
            "semantic_new_burst_flag": semantic_new_burst_flag,
            "semantic_independent_corroboration_flag": semantic_independent_corroboration_flag,
            "semantic_fusion_candidate_flag": semantic_fusion_candidate_flag,
            "semantic_fusion_run": semantic_fusion_run,
            "semantic_fusion_latched": semantic_fusion_latched,
            "semantic_fusion_quiet_run": semantic_fusion_quiet_run,
            "semantic_fusion_rearm_flag": semantic_fusion_rearm_flag,
            "semantic_fusion_trigger_flag": semantic_fusion_trigger_flag,
            "semantic_fusion_confirmed_flag": semantic_fusion_confirmed_flag,
            "semantic_fusion_support_flag": semantic_fusion_support_flag,
            "targeted_change_candidate_groups": ";".join(targeted_change_candidate_groups),
            "targeted_change_trigger_groups": ";".join(targeted_change_trigger_groups),
            "targeted_change_candidate_flag": targeted_change_candidate_flag,
            "targeted_change_trigger_flag": targeted_change_trigger_flag,
            "targeted_change_support_flag": targeted_change_support_flag,
            "targeted_change_score": targeted_change_score,
            "communication_control_candidate_flag": communication_control_candidate_flag,
            "communication_control_confirmed_flag": communication_control_confirmed_flag,
            "status_red_confirmed_flag": status_red_confirmed_flag,
            "semantic_support_flag": semantic_support_flag,
            "semantic_score": semantic_score,
            "review_family_count": review_family_count,
            "confirmed_family_count": confirmed_family_count,
            "detector_family_count": detector_family_count,
            "review_evidence_flag": review_evidence_flag,
            "confirmed_evidence_flag": confirmed_evidence_flag,
            "multi_review_support_flag": multi_review_support_flag,
            "multi_review_trigger_flag": False,
            "multi_review_promotion_hard_disabled": True,
            "candidate_persistent_flag": candidate_persistent_flag,
            "active_trigger_flag": active_trigger_flag,
            "active_support_flag": active_support_flag,
            "baseline_freeze_flag": baseline_freeze_flag,
            "evidence_flag": evidence_flag,
            "strong_evidence_flag": strong_evidence_flag,
            "alert_score": alert_score,
            "review_flag": review_flag,
            "active_alert_flag": active_alert_flag,
        }
        rows.append(output_row)

        # Update long-run histories after scoring and state transition so current
        # observations cannot influence their own trigger decision.
        stable_generation_history.append(int(stable_generation_now))
        background_short_history.append(
            int(fault_like_short_episode_start_flag)
        )
        background_active_history.append(int(active_alert_flag))
        if current_status:
            status_history.append(current_status)
        status_transition_history.append(int(status_transition_flag))
        status_confirmed_support_history.append(int(status_confirmed_flag))
        semantic_confirmed_support_history.append(int(semantic_confirmed_flag))
        semantic_fusion_support_history.append(int(semantic_fusion_trigger_flag))
        targeted_change_support_history.append(int(targeted_change_trigger_flag))
        for group_name in semantic_groups:
            semantic_group_burden_history[group_name].append(
                int(group_name in semantic_candidate_groups)
            )
        if current_status:
            previous_status = current_status

        previous_timestamp = timestamp
        previous_measurement_row = row.copy()
        if active_alert_flag or stream_state == "candidate":
            current_episode_rows.append(output_row.copy())

    # Close an episode that remains open at end-of-file as right-censored.
    if stream_state in {"active_alert", "recovery_pending", "inactive_pending"} and candidate_start_time is not None:
        episode_start = candidate_start_time
        episode_end = last_evidence_time or pd.Timestamp(df.iloc[-1][timestamp_col])
        overlap, meta_id, meta_label, overlap_hours = overlap_episode_with_metadata(
            episode_start, episode_end, source_metadata
        )
        frame = pd.DataFrame(current_episode_rows)
        detectors = []
        for detector_name, column in [
            ("fault_like_short_standstill", "strong_short_trigger_flag"),
            ("intermittent_cluster", "strong_short_trigger_flag"),
            ("persistent_system_state", "persistent_system_state_flag"),
            ("localized_persistent_subsystem_state", "localized_confirmed_flag"),
            ("communication_data_quality", "communication_alert_confirmed_flag"),
            ("slow_trend", "slow_trend_alert_confirmed_flag"),
            ("status_code", "status_red_confirmed_flag"),
            ("communication_control_fault", "communication_control_confirmed_flag"),
            ("semantic_subsystem_fusion", "semantic_fusion_trigger_flag"),
            ("targeted_subsystem_change", "targeted_change_trigger_flag"),
        ]:
            if not frame.empty and bool(frame.get(column, pd.Series(dtype=bool)).fillna(False).any()):
                detectors.append(detector_name)
        episodes.append(
            Episode(
                farm_id=farm_id,
                source_id=source_id,
                source_file=str(event_file),
                asset_id=asset_id,
                episode_id=f"{farm_id}_{source_id}_stream_{episode_counter:03d}",
                start_time=episode_start,
                end_time=episode_end,
                first_active_time=active_start_time or episode_start,
                last_evidence_time=episode_end,
                duration_hours=max(0.0, (episode_end - episode_start).total_seconds() / 3600.0),
                right_censored=True,
                max_alert_score=float(frame["alert_score"].max()) if not frame.empty else 0.0,
                max_abnormal_fraction_z8=float(frame["abnormal_fraction_z8"].max()) if not frame.empty else 0.0,
                max_abnormal_fraction_z12=float(frame["abnormal_fraction_z12"].max()) if not frame.empty else 0.0,
                short_confirmed_count=int(frame["short_confirmed_flag"].sum()) if not frame.empty else 0,
                short_episode_count=int(frame.get("short_episode_start_flag", pd.Series(dtype=bool)).fillna(False).sum()) if not frame.empty else 0,
                fault_like_short_episode_count=int(frame.get("fault_like_short_episode_start_flag", pd.Series(dtype=bool)).fillna(False).sum()) if not frame.empty else 0,
                intermittent_flag_points=int(frame["intermittent_cluster_flag"].sum()) if not frame.empty else 0,
                persistent_flag_points=int(frame["persistent_system_state_flag"].sum()) if not frame.empty else 0,
                localized_candidate_points=int(frame["localized_candidate_flag"].sum()) if not frame.empty else 0,
                localized_confirmed_points=int(frame["localized_confirmed_flag"].sum()) if not frame.empty else 0,
                detector_families=";".join(detectors),
                metadata_overlap=overlap,
                metadata_event_id=meta_id,
                metadata_label=meta_label,
                metadata_overlap_hours=overlap_hours,
            )
        )

    row_output = pd.DataFrame(rows)
    row_output = annotate_metadata_rows(row_output, source_metadata)
    episode_output = pd.DataFrame([asdict(episode) for episode in episodes])

    source_output_dir = output_dir / source_id
    source_output_dir.mkdir(parents=True, exist_ok=True)
    row_output.to_csv(source_output_dir / "stream_row_scores.csv", index=False)
    episode_output.to_csv(source_output_dir / "stream_detected_episodes.csv", index=False)

    summary = {
        "farm_id": farm_id,
        "source_id": source_id,
        "source_file": str(event_file),
        "rows": int(len(df)),
        "sampling_minutes": sampling_minutes,
        "warmup_points": warmup_points,
        "measurement_mode": measurement_mode,
        "measurement_columns": len(measurement_columns),
        "base_signals": len(base_names),
        "power_bases": power_bases,
        "status_column": status_col or "",
        "semantic_group_signal_counts": {
            name: int(len(indices))
            for name, indices in semantic_groups.items()
        },
        "detected_episodes": int(len(episode_output)),
        "active_alert_rows": int(row_output.get("active_alert_flag", pd.Series(dtype=bool)).fillna(False).sum()),
        "review_rows": int(row_output.get("review_flag", pd.Series(dtype=bool)).fillna(False).sum()),
        "confirmed_evidence_rows": int(
            row_output.get(
                "confirmed_evidence_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "status_candidate_rows": int(
            row_output.get(
                "status_candidate_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "status_confirmed_rows": int(
            row_output.get(
                "status_confirmed_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "semantic_candidate_rows": int(
            row_output.get(
                "semantic_candidate_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "semantic_confirmed_rows": int(
            row_output.get(
                "semantic_confirmed_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "semantic_onset_rows": int(
            row_output.get(
                "semantic_onset_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "semantic_chronic_rows": int(
            row_output.get(
                "semantic_chronic_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "semantic_fusion_confirmed_rows": int(
            row_output.get("semantic_fusion_trigger_flag", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "targeted_change_trigger_rows": int(
            row_output.get("targeted_change_trigger_flag", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "communication_control_confirmed_rows": int(
            row_output.get("communication_control_confirmed_flag", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "short_episode_starts": int(
            row_output.get("short_episode_start_flag", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "raw_fault_like_short_episode_starts": int(
            row_output.get(
                "raw_fault_like_short_episode_start_flag",
                pd.Series(dtype=bool),
            ).fillna(False).sum()
        ),
        "fault_like_short_episode_starts": int(
            row_output.get(
                "fault_like_short_episode_start_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "fault_like_short_confirmed_rows": int(
            row_output.get(
                "fault_like_short_confirmed_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "strong_short_trigger_rows": int(
            row_output.get("strong_short_trigger_flag", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "short_cluster_qualified_rows": int(
            row_output.get(
                "short_cluster_qualified_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "short_cluster_rearm_rows": int(
            row_output.get(
                "short_cluster_rearm_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "multi_review_trigger_rows": 0,
        "background_cycling_rows": int(
            row_output.get(
                "background_cycling_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "dynamic_background_threshold_rows": int(
            row_output.get(
                "background_cycling_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "short_duration_rejected_rows": int(
            row_output.get(
                "short_duration_rejected_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "short_cluster_support_rows": int(
            row_output.get("short_cluster_support_flag", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "short_recent_support_rows": int(
            row_output.get("short_recent_support_flag", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "localized_confirmation_trigger_rows": int(
            row_output.get(
                "localized_confirmation_trigger_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "communication_data_quality_confirmed_rows": int(
            row_output.get(
                "communication_data_quality_confirmed_flag",
                pd.Series(dtype=bool),
            ).fillna(False).sum()
        ),
        "communication_alert_confirmed_rows": int(
            row_output.get(
                "communication_alert_confirmed_flag",
                pd.Series(dtype=bool),
            ).fillna(False).sum()
        ),
        "slow_trend_pattern_confirmed_rows": int(
            row_output.get(
                "slow_trend_pattern_confirmed_flag",
                pd.Series(dtype=bool),
            ).fillna(False).sum()
        ),
        "slow_trend_alert_confirmed_rows": int(
            row_output.get(
                "slow_trend_alert_confirmed_flag",
                pd.Series(dtype=bool),
            ).fillna(False).sum()
        ),
        "localized_candidate_rows": int(
            row_output.get(
                "localized_candidate_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "localized_confirmed_rows": int(
            row_output.get(
                "localized_confirmed_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "baseline_frozen_rows": int(
            row_output.get(
                "baseline_freeze_flag", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "metadata_rows": int(len(source_metadata)),
        "config": asdict(cfg),
    }
    with (source_output_dir / "stream_replay_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return row_output, episode_output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm", required=True, help="Farm identifier, e.g. C")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--event-id", default="all", help="all or one comma-separated list")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measurement-mode", choices=["avg_only", "all"], default="avg_only")
    parser.add_argument("--power-signals", default="")
    parser.add_argument(
        "--feature-description",
        type=Path,
        default=None,
        help="feature_description.csv used to build semantic subsystem groups.",
    )
    parser.add_argument("--warmup-hours", type=float, default=48.0)
    parser.add_argument("--baseline-effective-hours", type=float, default=48.0)
    parser.add_argument("--candidate-confirmation-points", type=int, default=3)
    parser.add_argument("--recovery-confirmation-points", type=int, default=6)
    parser.add_argument("--episode-merge-gap-hours", type=float, default=6.0)
    parser.add_argument("--strong-short-count-2h", type=int, default=3)
    parser.add_argument("--strong-short-count-6h", type=int, default=2)
    parser.add_argument("--strong-short-count-24h", type=int, default=6)
    parser.add_argument("--strong-short-minimum-abnormal-fraction", type=float, default=0.10)
    parser.add_argument("--short-support-minutes", type=float, default=60.0)
    parser.add_argument("--short-pre-generating-points", type=int, default=6)
    parser.add_argument("--short-pre-generating-fraction", type=float, default=0.80)
    parser.add_argument("--short-minimum-drop-score", type=float, default=1.00)
    parser.add_argument("--short-stable-recovery-points", type=int, default=3)
    parser.add_argument("--short-stable-recovery-fraction", type=float, default=0.80)
    parser.add_argument("--short-pending-timeout-minutes", type=float, default=180.0)
    parser.add_argument("--short-independent-min-gap-minutes", type=float, default=60.0)
    parser.add_argument("--short-stable-generation-ratio", type=float, default=0.30)
    parser.add_argument("--short-minimum-dip-points", type=int, default=1)
    parser.add_argument("--short-maximum-dip-points", type=int, default=12)
    parser.add_argument("--background-window-hours", type=float, default=168.0)
    parser.add_argument("--background-minimum-history-hours", type=float, default=48.0)
    parser.add_argument("--background-active-fraction-threshold", type=float, default=0.20)
    parser.add_argument("--background-short-count-threshold", type=int, default=20)
    parser.add_argument("--background-strong-short-count-2h", type=int, default=4)
    parser.add_argument("--background-strong-short-count-6h", type=int, default=3)
    parser.add_argument("--background-strong-short-count-24h", type=int, default=8)
    parser.add_argument(
        "--disable-background-suppression",
        action="store_true",
        help="Disable causal chronic-cycling suppression for ablation testing.",
    )
    parser.add_argument(
        "--enable-multi-review-promotion",
        action="store_true",
        help="Legacy compatibility flag; ignored because V3 hard-disables this path.",
    )
    parser.add_argument("--multi-review-confirmation-points", type=int, default=6)
    parser.add_argument("--short-cluster-quiet-hours", type=float, default=6.0)
    parser.add_argument("--short-cluster-rearm-max-count-6h", type=int, default=1)
    parser.add_argument("--operating-stopped-ratio", type=float, default=0.03)
    parser.add_argument("--operating-transition-ratio", type=float, default=0.15)
    parser.add_argument("--operating-state-confirmation-points", type=int, default=3)
    parser.add_argument("--communication-missing-candidate-fraction", type=float, default=0.10)
    parser.add_argument("--communication-missing-confirmed-fraction", type=float, default=0.20)
    parser.add_argument("--communication-frozen-candidate-fraction", type=float, default=0.70)
    parser.add_argument("--communication-frozen-confirmed-fraction", type=float, default=0.85)
    parser.add_argument("--communication-confirmation-points", type=int, default=3)
    parser.add_argument("--communication-confirmed-gap-minutes", type=float, default=30.0)
    parser.add_argument(
        "--enable-communication-red-alert",
        action="store_true",
        help=(
            "Allow generic missing/frozen/timestamp-gap evidence to open a red "
            "turbine-fault alert. Disabled by default."
        ),
    )
    parser.add_argument("--slow-trend-window-hours", type=float, default=6.0)
    parser.add_argument("--slow-trend-shift-hours", type=float, default=24.0)
    parser.add_argument("--slow-trend-minimum-slope-z-per-hour", type=float, default=0.25)
    parser.add_argument("--slow-trend-minimum-shift-z", type=float, default=2.5)
    parser.add_argument("--slow-trend-minimum-signals", type=int, default=2)
    parser.add_argument("--slow-trend-confirmation-points", type=int, default=6)
    parser.add_argument(
        "--enable-slow-trend-red-alert",
        action="store_true",
        help=(
            "Allow the generic anonymous-sensor slow-trend detector to open a "
            "red alert. Disabled by default; use only for ablation experiments."
        ),
    )
    parser.add_argument("--status-history-hours", type=float, default=720.0)
    parser.add_argument("--status-minimum-history-hours", type=float, default=48.0)
    parser.add_argument("--status-rare-frequency-threshold", type=float, default=0.0005)
    parser.add_argument("--status-novel-confirmation-points", type=int, default=2)
    parser.add_argument("--status-rare-confirmation-points", type=int, default=3)
    parser.add_argument("--status-transition-window-hours", type=float, default=1.0)
    parser.add_argument("--status-transition-count-threshold", type=int, default=4)
    parser.add_argument("--status-confirmed-support-hours", type=float, default=6.0)
    parser.add_argument("--semantic-candidate-z-threshold", type=float, default=6.0)
    parser.add_argument("--semantic-confirmed-z-threshold", type=float, default=8.0)
    parser.add_argument("--semantic-minimum-abnormal-signals", type=int, default=2)
    parser.add_argument("--semantic-maximum-required-signals", type=int, default=8)
    parser.add_argument("--semantic-required-fraction", type=float, default=0.20)
    parser.add_argument("--semantic-confirmation-points", type=int, default=3)
    parser.add_argument("--semantic-multi-group-confirmation-points", type=int, default=3)
    parser.add_argument("--semantic-burden-window-hours", type=float, default=24.0)
    parser.add_argument("--semantic-burden-minimum-history-hours", type=float, default=6.0)
    parser.add_argument("--semantic-chronic-fraction-threshold", type=float, default=0.20)
    parser.add_argument("--semantic-onset-quiet-fraction-threshold", type=float, default=0.10)
    parser.add_argument("--semantic-fusion-confirmation-points", type=int, default=2)
    parser.add_argument("--semantic-confirmed-support-hours", type=float, default=1.0)
    parser.add_argument("--semantic-fusion-quiet-hours", type=float, default=12.0)
    parser.add_argument("--semantic-fusion-maximum-group-count", type=int, default=2)
    parser.add_argument("--targeted-change-z-threshold", type=float, default=10.0)
    parser.add_argument("--targeted-change-confirmation-points", type=int, default=2)
    parser.add_argument("--targeted-change-quiet-hours", type=float, default=12.0)
    parser.add_argument("--targeted-change-support-hours", type=float, default=1.0)
    parser.add_argument("--targeted-change-maximum-group-count", type=int, default=2)
    parser.add_argument("--communication-control-confirmation-points", type=int, default=2)
    parser.add_argument("--localized-confirmation-window-minutes", type=float, default=120.0)
    parser.add_argument("--localized-confirmed-support-hours", type=float, default=6.0)
    parser.add_argument("--review-score", type=float, default=0.55)
    parser.add_argument("--immediate-alert-score", type=float, default=0.90)
    return parser.parse_args()


def discover_files(event_dir: Path, event_id_argument: str) -> list[tuple[str, Path]]:
    if not event_dir.exists():
        raise FileNotFoundError(event_dir)
    requested = None
    if event_id_argument.strip().lower() != "all":
        requested = {normalise_event_id(item) for item in event_id_argument.split(",") if item.strip()}
    files: list[tuple[str, Path]] = []
    for path in sorted(event_dir.glob("*.csv")):
        source_id = normalise_event_id(path.stem)
        if requested is None or source_id in requested:
            files.append((source_id, path))
    if not files:
        raise FileNotFoundError("No matching CSV files were found.")
    return files


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata, args.farm)
    feature_descriptions = read_feature_descriptions(args.feature_description)
    power_signals = [item.strip() for item in args.power_signals.split(",") if item.strip()] or None
    overrides = {
        "warmup_hours": args.warmup_hours,
        "baseline_effective_hours": args.baseline_effective_hours,
        "candidate_confirmation_points": args.candidate_confirmation_points,
        "recovery_confirmation_points": args.recovery_confirmation_points,
        "episode_merge_gap_hours": args.episode_merge_gap_hours,
        "strong_short_count_2h": args.strong_short_count_2h,
        "strong_short_count_6h": args.strong_short_count_6h,
        "strong_short_count_24h": args.strong_short_count_24h,
        "strong_short_minimum_abnormal_fraction": args.strong_short_minimum_abnormal_fraction,
        "short_support_minutes": args.short_support_minutes,
        "short_pre_generating_points": args.short_pre_generating_points,
        "short_pre_generating_fraction": args.short_pre_generating_fraction,
        "short_minimum_drop_score": args.short_minimum_drop_score,
        "short_stable_recovery_points": args.short_stable_recovery_points,
        "short_stable_recovery_fraction": args.short_stable_recovery_fraction,
        "short_pending_timeout_minutes": args.short_pending_timeout_minutes,
        "short_independent_min_gap_minutes": args.short_independent_min_gap_minutes,
        "short_stable_generation_ratio": args.short_stable_generation_ratio,
        "short_minimum_dip_points": args.short_minimum_dip_points,
        "short_maximum_dip_points": args.short_maximum_dip_points,
        "background_window_hours": args.background_window_hours,
        "background_minimum_history_hours": args.background_minimum_history_hours,
        "background_active_fraction_threshold": args.background_active_fraction_threshold,
        "background_short_count_threshold": args.background_short_count_threshold,
        "background_strong_short_count_2h": args.background_strong_short_count_2h,
        "background_strong_short_count_6h": args.background_strong_short_count_6h,
        "background_strong_short_count_24h": args.background_strong_short_count_24h,
        "background_suppression_enabled": not args.disable_background_suppression,
        "enable_multi_review_promotion": False,
        "multi_review_confirmation_points": args.multi_review_confirmation_points,
        "short_cluster_quiet_hours": args.short_cluster_quiet_hours,
        "short_cluster_rearm_max_count_6h": args.short_cluster_rearm_max_count_6h,
        "operating_stopped_ratio": args.operating_stopped_ratio,
        "operating_transition_ratio": args.operating_transition_ratio,
        "operating_generating_ratio": args.operating_transition_ratio,
        "operating_state_confirmation_points": args.operating_state_confirmation_points,
        "communication_missing_candidate_fraction": args.communication_missing_candidate_fraction,
        "communication_missing_confirmed_fraction": args.communication_missing_confirmed_fraction,
        "communication_frozen_candidate_fraction": args.communication_frozen_candidate_fraction,
        "communication_frozen_confirmed_fraction": args.communication_frozen_confirmed_fraction,
        "communication_confirmation_points": args.communication_confirmation_points,
        "communication_confirmed_gap_minutes": args.communication_confirmed_gap_minutes,
        "communication_red_alert_enabled": args.enable_communication_red_alert,
        "slow_trend_window_hours": args.slow_trend_window_hours,
        "slow_trend_shift_hours": args.slow_trend_shift_hours,
        "slow_trend_minimum_slope_z_per_hour": args.slow_trend_minimum_slope_z_per_hour,
        "slow_trend_minimum_shift_z": args.slow_trend_minimum_shift_z,
        "slow_trend_minimum_signals": args.slow_trend_minimum_signals,
        "slow_trend_confirmation_points": args.slow_trend_confirmation_points,
        "slow_trend_red_alert_enabled": args.enable_slow_trend_red_alert,
        "status_history_hours": args.status_history_hours,
        "status_minimum_history_hours": args.status_minimum_history_hours,
        "status_rare_frequency_threshold": args.status_rare_frequency_threshold,
        "status_novel_confirmation_points": args.status_novel_confirmation_points,
        "status_rare_confirmation_points": args.status_rare_confirmation_points,
        "status_transition_window_hours": args.status_transition_window_hours,
        "status_transition_count_threshold": args.status_transition_count_threshold,
        "status_confirmed_support_hours": args.status_confirmed_support_hours,
        "semantic_candidate_z_threshold": args.semantic_candidate_z_threshold,
        "semantic_confirmed_z_threshold": args.semantic_confirmed_z_threshold,
        "semantic_minimum_abnormal_signals": args.semantic_minimum_abnormal_signals,
        "semantic_maximum_required_signals": args.semantic_maximum_required_signals,
        "semantic_required_fraction": args.semantic_required_fraction,
        "semantic_confirmation_points": args.semantic_confirmation_points,
        "semantic_multi_group_confirmation_points": args.semantic_multi_group_confirmation_points,
        "semantic_burden_window_hours": args.semantic_burden_window_hours,
        "semantic_burden_minimum_history_hours": args.semantic_burden_minimum_history_hours,
        "semantic_chronic_fraction_threshold": args.semantic_chronic_fraction_threshold,
        "semantic_onset_quiet_fraction_threshold": args.semantic_onset_quiet_fraction_threshold,
        "semantic_fusion_confirmation_points": args.semantic_fusion_confirmation_points,
        "semantic_confirmed_support_hours": args.semantic_confirmed_support_hours,
        "semantic_fusion_quiet_hours": args.semantic_fusion_quiet_hours,
        "semantic_fusion_maximum_group_count": args.semantic_fusion_maximum_group_count,
        "targeted_change_z_threshold": args.targeted_change_z_threshold,
        "targeted_change_confirmation_points": args.targeted_change_confirmation_points,
        "targeted_change_quiet_hours": args.targeted_change_quiet_hours,
        "targeted_change_support_hours": args.targeted_change_support_hours,
        "targeted_change_maximum_group_count": args.targeted_change_maximum_group_count,
        "status_independent_red_enabled": False,
        "communication_control_confirmation_points": args.communication_control_confirmation_points,
        "localized_confirmation_window_minutes": args.localized_confirmation_window_minutes,
        "localized_confirmed_support_hours": args.localized_confirmed_support_hours,
        "review_score": args.review_score,
        "immediate_alert_score": args.immediate_alert_score,
    }

    summaries: list[dict[str, Any]] = []
    all_episodes: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for source_id, event_file in discover_files(args.event_dir, args.event_id):
        print(f"[stream replay] {event_file.name}")
        try:
            _, episodes, summary = replay_one_file(
                event_file=event_file,
                source_id=source_id,
                farm_id=args.farm,
                source_metadata=metadata_for_source(metadata, source_id),
                output_dir=args.output_dir,
                measurement_mode=args.measurement_mode,
                manual_power_signals=power_signals,
                feature_descriptions=feature_descriptions,
                config_overrides=overrides,
            )
            summaries.append(summary)
            if not episodes.empty:
                all_episodes.append(episodes)
        except Exception as exc:  # continue other files, record the failure
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append({"source_id": source_id, "source_file": str(event_file), "error": str(exc)})

    combined = pd.concat(all_episodes, ignore_index=True) if all_episodes else pd.DataFrame()
    combined.to_csv(args.output_dir / "all_stream_detected_episodes.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "stream_replay_failures.csv", index=False)

    manifest = {
        "farm_id": args.farm,
        "feature_description_file": (
            str(args.feature_description)
            if args.feature_description is not None
            else ""
        ),
        "feature_description_sensor_count": len(feature_descriptions),
        "multi_review_promotion_hard_disabled": True,
        "status_independent_red_disabled": True,
        "persistent_review_only": True,
        "version": "general_multidetector_v5",
        "processed_sources": len(summaries),
        "failed_sources": len(failures),
        "total_detected_episodes": int(len(combined)),
        "summaries": summaries,
        "failures": failures,
    }
    with (args.output_dir / "stream_replay_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"Completed: {len(summaries)} source(s), {len(failures)} failure(s).")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())