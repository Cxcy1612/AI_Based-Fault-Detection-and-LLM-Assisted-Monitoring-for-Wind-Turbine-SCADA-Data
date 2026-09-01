#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a causal streaming-window feature table from streaming replay outputs.

Input is produced by ``streaming_detector_replay.py``.  Every output sample is a
snapshot at time t and all rolling features use only rows at or before t.
Metadata labels are used only as targets after the streaming detector has run.

Default sampling keeps every anomaly snapshot and one normal snapshot per hour
for ten-minute SCADA data.  This reduces the extreme correlation and class
imbalance caused by saving every normal ten-minute row.

Example
-------
python build_streaming_window_dataset.py ^
  --input-dir "outputs/farmC_streaming_replay" ^
  --output-file "outputs/farmC_streaming_dataset/streaming_window_features.csv" ^
  --positive-every-points 1 ^
  --negative-every-points 6 ^
  --prediction-horizon-hours 0 ^
  --normal-exclusion-buffer-hours 6
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

TRUE_TEXT = {"true", "1", "yes", "y", "t"}

CURRENT_NUMERIC_COLUMNS = [
    "abnormal_fraction_z8",
    "abnormal_fraction_z12",
    "top3_base_z_mean",
    "top10_base_z_mean",
    "power_dip_signal_count",
    "power_recovery_signal_count",
    "power_dip_score_max",
    "power_recovery_score_max",
    "short_count_trailing_2h",
    "persistent_smoothed_fraction",
    "persistent_run_points",
    "localized_active_signal_count",
    "localized_stable_signal_count",
    "localized_sensor_overlap",
    "localized_sensor_overlap_smoothed",
    "localized_strength",
    "localized_strength_smoothed",
    "localized_run_points",
    "detector_family_count",
    "alert_score",
]

BINARY_COLUMNS = [
    "power_dip_consensus",
    "power_recovery_consensus",
    "short_candidate_flag",
    "short_confirmed_flag",
    "intermittent_cluster_flag",
    "persistent_raw_flag",
    "persistent_system_state_flag",
    "localized_raw_flag",
    "localized_candidate_flag",
    "localized_confirmed_flag",
    "evidence_flag",
    "strong_evidence_flag",
]

ROLLING_NUMERIC_COLUMNS = [
    "abnormal_fraction_z8",
    "abnormal_fraction_z12",
    "top3_base_z_mean",
    "top10_base_z_mean",
    "power_dip_signal_count",
    "power_recovery_signal_count",
    "power_dip_score_max",
    "power_recovery_score_max",
    "persistent_smoothed_fraction",
    "localized_stable_signal_count",
    "localized_sensor_overlap_smoothed",
    "localized_strength_smoothed",
    "detector_family_count",
    "alert_score",
]

ROLLING_BINARY_COLUMNS = [
    "short_candidate_flag",
    "short_confirmed_flag",
    "intermittent_cluster_flag",
    "persistent_raw_flag",
    "persistent_system_state_flag",
    "localized_raw_flag",
    "localized_candidate_flag",
    "localized_confirmed_flag",
    "evidence_flag",
    "strong_evidence_flag",
]

METADATA_COLUMNS = [
    "farm_id",
    "source_id",
    "source_file",
    "asset_id",
    "timestamp",
    "row_id",
    "metadata_event_id",
    "metadata_label",
    "metadata_start",
    "metadata_end",
    "inside_metadata_interval",
    "target_anomaly",
]


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype("string").fillna("").str.strip().str.lower()
    return text.isin(TRUE_TEXT)


def infer_sampling_minutes(timestamps: pd.Series) -> float:
    diffs = timestamps.sort_values().diff().dt.total_seconds().div(60.0)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        raise ValueError("Could not infer sampling interval.")
    return float(diffs.median())


def causal_future_max(values: pd.Series, points: int) -> pd.Series:
    """Maximum from current row through the next ``points`` rows."""
    if points <= 0:
        return values.astype(int)
    reversed_values = values.iloc[::-1]
    result = reversed_values.rolling(points + 1, min_periods=1).max().iloc[::-1]
    return result.astype(int)


def consecutive_run_length(values: pd.Series) -> pd.Series:
    array = to_bool(values).to_numpy()
    result = np.zeros(len(array), dtype=int)
    run = 0
    for index, value in enumerate(array):
        run = run + 1 if value else 0
        result[index] = run
    return pd.Series(result, index=values.index)


def discover_row_score_files(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    files = sorted(input_dir.rglob("stream_row_scores.csv"))
    if not files:
        raise FileNotFoundError(
            f"No stream_row_scores.csv files were found below {input_dir}."
        )
    return files


def safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def normal_exclusion_mask(
    frame: pd.DataFrame,
    buffer_hours: float,
) -> pd.Series:
    if buffer_hours <= 0:
        return pd.Series(False, index=frame.index)
    positive_times = frame.loc[frame["target_anomaly_current"] == 1, "timestamp"]
    if positive_times.empty:
        return pd.Series(False, index=frame.index)

    start = positive_times.min() - pd.Timedelta(hours=buffer_hours)
    end = positive_times.max() + pd.Timedelta(hours=buffer_hours)
    return (
        (frame["target_anomaly_current"] == 0)
        & (frame["timestamp"] >= start)
        & (frame["timestamp"] <= end)
    )


def build_features_for_source(
    path: Path,
    windows_hours: list[float],
    prediction_horizon_hours: float,
    positive_every_points: int,
    negative_every_points: int,
    normal_exclusion_buffer_hours: float,
    minimum_history_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    if "timestamp" not in frame:
        raise ValueError(f"{path} does not contain a timestamp column.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path} contains no usable rows.")

    sampling_minutes = infer_sampling_minutes(frame["timestamp"])
    frame["warmup_complete"] = to_bool(
        frame.get("warmup_complete", pd.Series(False, index=frame.index))
    )
    frame["target_anomaly_current"] = pd.to_numeric(
        frame.get("target_anomaly", 0), errors="coerce"
    ).fillna(0).astype(int)

    for column in BINARY_COLUMNS:
        frame[column] = to_bool(frame.get(column, pd.Series(False, index=frame.index))).astype(int)
    for column in CURRENT_NUMERIC_COLUMNS:
        frame[column] = safe_numeric(frame, column)

    features = pd.DataFrame(index=frame.index)
    for column in CURRENT_NUMERIC_COLUMNS + BINARY_COLUMNS:
        features[f"current__{column}"] = frame[column]

    # Causal run lengths and online durations.
    for column in [
        "short_candidate_flag",
        "short_confirmed_flag",
        "intermittent_cluster_flag",
        "persistent_raw_flag",
        "persistent_system_state_flag",
        "localized_raw_flag",
        "localized_candidate_flag",
        "localized_confirmed_flag",
        "evidence_flag",
        "strong_evidence_flag",
    ]:
        run = consecutive_run_length(frame[column])
        features[f"run_points__{column}"] = run
        features[f"run_hours__{column}"] = run * sampling_minutes / 60.0

    window_points: dict[str, int] = {}
    complete_history = pd.Series(True, index=frame.index)
    for hours in windows_hours:
        points = max(1, int(round(hours * 60.0 / sampling_minutes)))
        label = str(hours).replace(".", "p") + "h"
        window_points[label] = points
        min_periods = max(1, int(math.ceil(points * minimum_history_fraction)))
        complete_history &= frame.index.to_series() >= (min_periods - 1)

        for column in ROLLING_NUMERIC_COLUMNS:
            series = frame[column]
            rolling = series.rolling(points, min_periods=min_periods)
            features[f"mean_{label}__{column}"] = rolling.mean()
            features[f"max_{label}__{column}"] = rolling.max()
            features[f"std_{label}__{column}"] = rolling.std(ddof=0).fillna(0.0)
            if column in {"abnormal_fraction_z8", "abnormal_fraction_z12", "alert_score"}:
                features[f"p95_{label}__{column}"] = rolling.quantile(0.95)

        for column in ROLLING_BINARY_COLUMNS:
            series = frame[column]
            rolling = series.rolling(points, min_periods=min_periods)
            features[f"rate_{label}__{column}"] = rolling.mean()
            features[f"count_{label}__{column}"] = rolling.sum()

    # Recent changes: current compared with one and six hours earlier.
    for lag_hours in [1.0, 6.0, 24.0]:
        points = max(1, int(round(lag_hours * 60.0 / sampling_minutes)))
        label = str(lag_hours).replace(".", "p") + "h"
        for column in [
            "abnormal_fraction_z8",
            "abnormal_fraction_z12",
            "top3_base_z_mean",
            "persistent_smoothed_fraction",
            "localized_strength_smoothed",
            "alert_score",
        ]:
            features[f"delta_{label}__{column}"] = frame[column] - frame[column].shift(points)

    horizon_points = max(0, int(round(prediction_horizon_hours * 60.0 / sampling_minutes)))
    target = causal_future_max(frame["target_anomaly_current"], horizon_points)

    metadata = pd.DataFrame(index=frame.index)
    for column in METADATA_COLUMNS:
        if column in frame:
            metadata[column] = frame[column]
        else:
            metadata[column] = "" if column not in {"inside_metadata_interval", "target_anomaly"} else 0

    metadata["snapshot_id"] = (
        metadata.get("source_id", "").astype(str)
        + "__"
        + frame["timestamp"].dt.strftime("%Y%m%dT%H%M%S")
    )
    asset = metadata.get("asset_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    source = metadata.get("source_id", pd.Series(path.parent.name, index=frame.index)).fillna(path.parent.name).astype(str)
    metadata["source_group_id"] = "source__" + source
    metadata["asset_group_id"] = np.where(
        asset.str.len() > 0, "asset__" + asset, metadata["source_group_id"]
    )
    # Default to source-level grouping so the current Farm C dataset retains
    # enough independent groups for validation.  When several source files are
    # overlapping copies from the same turbine/year, train with
    # --group-column asset_group_id instead.
    metadata["group_id"] = metadata["source_group_id"]
    metadata["sampling_minutes"] = sampling_minutes
    metadata["target_anomaly_current"] = frame["target_anomaly_current"]
    metadata["target_anomaly"] = target
    metadata["prediction_horizon_hours"] = prediction_horizon_hours
    metadata["source_row_score_file"] = str(path)

    combined = pd.concat([metadata.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    combined = combined.loc[frame["warmup_complete"].to_numpy()].copy()
    combined = combined.loc[complete_history.loc[combined.index].fillna(False).to_numpy()].copy()

    # Remove normal snapshots close to a labelled anomaly.  They are often
    # transition/recovery points and can create ambiguous supervision.
    aligned_frame = frame.loc[combined.index].copy()
    aligned_frame["target_anomaly_current"] = combined["target_anomaly_current"].to_numpy()
    exclusion = normal_exclusion_mask(aligned_frame, normal_exclusion_buffer_hours)
    combined = combined.loc[~exclusion.to_numpy()].copy()

    # Deterministic temporal down-sampling within each class.
    positive_counter = combined.groupby("target_anomaly").cumcount()
    keep_positive = (combined["target_anomaly"] == 1) & (
        positive_counter.mod(max(1, positive_every_points)) == 0
    )
    keep_negative = (combined["target_anomaly"] == 0) & (
        positive_counter.mod(max(1, negative_every_points)) == 0
    )
    combined = combined.loc[keep_positive | keep_negative].copy()

    numeric_feature_columns = [
        column for column in combined.columns
        if column not in metadata.columns
    ]
    combined[numeric_feature_columns] = combined[numeric_feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    summary = {
        "source": str(path),
        "sampling_minutes": sampling_minutes,
        "window_points": window_points,
        "horizon_points": horizon_points,
        "rows_input": int(len(frame)),
        "snapshots_output": int(len(combined)),
        "positive_snapshots": int((combined["target_anomaly"] == 1).sum()),
        "negative_snapshots": int((combined["target_anomaly"] == 0).sum()),
        "feature_count": len(numeric_feature_columns),
    }
    return combined.reset_index(drop=True), summary


def parse_hours(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Window hours must be positive comma-separated values.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--windows-hours", type=parse_hours, default=parse_hours("1,2,6,24"))
    parser.add_argument("--prediction-horizon-hours", type=float, default=0.0)
    parser.add_argument("--positive-every-points", type=int, default=1)
    parser.add_argument("--negative-every-points", type=int, default=6)
    parser.add_argument("--normal-exclusion-buffer-hours", type=float, default=6.0)
    parser.add_argument("--minimum-history-fraction", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for path in discover_row_score_files(args.input_dir):
        print(f"[build snapshots] {path}")
        try:
            table, summary = build_features_for_source(
                path=path,
                windows_hours=args.windows_hours,
                prediction_horizon_hours=args.prediction_horizon_hours,
                positive_every_points=args.positive_every_points,
                negative_every_points=args.negative_every_points,
                normal_exclusion_buffer_hours=args.normal_exclusion_buffer_hours,
                minimum_history_fraction=args.minimum_history_fraction,
            )
            if not table.empty:
                tables.append(table)
            summaries.append(summary)
        except Exception as exc:
            failures.append({"source": str(path), "error": str(exc)})
            print(f"  FAILED: {exc}")

    if not tables:
        raise RuntimeError("No streaming snapshot table was created.")

    dataset = pd.concat(tables, ignore_index=True, sort=False)
    dataset = dataset.sort_values(["timestamp", "group_id", "snapshot_id"]).reset_index(drop=True)
    dataset.to_csv(args.output_file, index=False)

    manifest = {
        "output_file": str(args.output_file),
        "rows": int(len(dataset)),
        "groups": int(dataset["group_id"].nunique()),
        "positive_rows": int((dataset["target_anomaly"] == 1).sum()),
        "negative_rows": int((dataset["target_anomaly"] == 0).sum()),
        "prediction_horizon_hours": args.prediction_horizon_hours,
        "windows_hours": args.windows_hours,
        "positive_every_points": args.positive_every_points,
        "negative_every_points": args.negative_every_points,
        "normal_exclusion_buffer_hours": args.normal_exclusion_buffer_hours,
        "sources": summaries,
        "failures": failures,
    }
    manifest_path = args.output_file.with_name(args.output_file.stem + "_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    pd.DataFrame(summaries).to_csv(
        args.output_file.with_name(args.output_file.stem + "_source_summary.csv"),
        index=False,
    )
    pd.DataFrame(failures).to_csv(
        args.output_file.with_name(args.output_file.stem + "_failures.csv"),
        index=False,
    )
    print(f"Saved {len(dataset)} snapshots to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
