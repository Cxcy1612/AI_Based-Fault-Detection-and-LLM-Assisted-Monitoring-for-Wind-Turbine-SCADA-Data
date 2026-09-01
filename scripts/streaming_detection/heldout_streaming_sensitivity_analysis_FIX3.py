#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strong held-out sensitivity analysis for the causal Streaming V5 detector.

Purpose
-------
This script addresses the risk that global Streaming V5 thresholds were refined
while inspecting the same labelled Events later used for retrospective evaluation.
It performs a post-hoc robustness / sensitivity analysis in which:

1. A pre-specified bank of plausible Streaming V5 configurations is evaluated.
   Detector replay itself is label-blind: Event labels and metadata boundaries are
   NOT passed into ``replay_one_file``.
2. Threshold/configuration selection is performed using development Events only.
3. The selected configuration is frozen before held-out evaluation.
4. Three complementary checks are produced:
   a) one canonical stratified 70/30 held-out split;
   b) one chronological latest-Events held-out split;
   c) repeated nested stratified Event-level CV (default 5 folds x 10 repeats).
5. Event-level bootstrap intervals are computed from repeated out-of-sample
   detection rates.

Important limitation
--------------------
This cannot make historically inspected Events genuinely untouched again.  The
original Streaming V5 design was developed using the available Event collection.
Therefore this analysis should be described as a *post-hoc held-out sensitivity
analysis* or *robustness check*, not as external validation.  Its value is that,
within each new split, held-out labels are not used to select the configuration.

The script dynamically imports the user's existing:
- streaming_detector_general_multidetector_v5.py
- evaluate_all_streaming_events_general_v5.py

No modification of those source files is required.

Example (Windows PowerShell)
----------------------------
python "C:\\Users\\Lenovo\\Documents\\u5712870\\wind_farm_fault_detection\\scripts\\streaming_detection\\heldout_streaming_sensitivity_analysis.py" `
  --stream-script "C:\\Users\\Lenovo\\Documents\\u5712870\\wind_farm_fault_detection\\scripts\\streaming_detection\\streaming_detector_general_multidetector_v5.py" `
  --evaluator-script "C:\\Users\\Lenovo\\Documents\\u5712870\\wind_farm_fault_detection\\scripts\\streaming_detection\\evaluate_all_streaming_events_general_v5.py" `
  --metadata "C:\\Users\\Lenovo\\Documents\\u5712870\\wind_farm_fault_detection\\data\\raw\\Wind Farm C\\event_info.csv" `
  --event-dir "C:\\Users\\Lenovo\\Documents\\u5712870\\wind_farm_fault_detection\\data\\raw\\Wind Farm C\\datasets" `
  --feature-description "C:\\Users\\Lenovo\\Documents\\u5712870\\wind_farm_fault_detection\\data\\raw\\Wind Farm C\\feature_description.csv" `
  --output-dir "C:\\Users\\Lenovo\\Documents\\u5712870\\wind_farm_fault_detection\\outputs\\streaming_heldout_sensitivity_v5" `
  --farm C `
  --measurement-mode avg_only `
  --power-signals "power_2,power_5,power_6,power_17" `
  --candidate-set full `
  --selection-target operational `
  --max-normal-fpr 0.35 `
  --outer-splits 5 `
  --outer-repeats 10 `
  --inner-splits 4 `
  --canonical-holdout-fraction 0.30 `
  --bootstrap-repeats 10000 `
  --random-state 42 `
  --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)


SCRIPT_BUILD = "2026-08-25-fix3-empty-metadata-dataframe"

# =============================================================================
# 0. CONSTANTS
# =============================================================================

LABEL_TO_INT = {"normal": 0, "anomaly": 1}
DETECTION_COLUMNS = {
    "any_overlap": "any_overlap_detected",
    "meaningful": "meaningful_detected",
    "operational": "operational_detected",
    "in_interval_active": "in_interval_active_detected",
}


# =============================================================================
# 1. GENERAL UTILITIES
# =============================================================================

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
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)


def normalise_event_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def safe_divide(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "p2_5": np.nan,
            "p97_5": np.nan,
            "std": np.nan,
            "n": 0,
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "p2_5": float(np.quantile(arr, 0.025)),
        "p97_5": float(np.quantile(arr, 0.975)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def import_module_from_path(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses and some dynamic-module behaviours.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def append_row_csv(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
    )


# =============================================================================
# 2. CANDIDATE CONFIGURATION BANK
# =============================================================================

def load_base_config(v5_module, base_config_json: Optional[Path]) -> dict[str, Any]:
    """Return ReplayConfig defaults, optionally overlaid by a JSON file."""
    default = asdict(v5_module.ReplayConfig(sampling_minutes=10.0))
    default.pop("sampling_minutes", None)

    if base_config_json is not None:
        if not base_config_json.exists():
            raise FileNotFoundError(base_config_json)
        with base_config_json.open("r", encoding="utf-8") as handle:
            overlay = json.load(handle)
        if not isinstance(overlay, dict):
            raise ValueError("--base-config-json must contain a JSON object.")
        valid_fields = {f.name for f in fields(v5_module.ReplayConfig)} - {"sampling_minutes"}
        unknown = sorted(set(overlay) - valid_fields)
        if unknown:
            raise ValueError(
                "Unknown ReplayConfig fields in base config JSON: " + ", ".join(unknown)
            )
        default.update(overlay)

    # Hard safety: preserve the intended final architecture.
    default["enable_multi_review_promotion"] = False
    default["status_independent_red_enabled"] = False
    default["communication_red_alert_enabled"] = False
    default["slow_trend_red_alert_enabled"] = False
    return default


def make_candidate_bank(
    v5_module,
    base_config: dict[str, Any],
    candidate_set: str,
) -> list[dict[str, Any]]:
    """
    Build a deliberately small, interpretable, pre-specified bank.

    This is intentionally NOT a huge Cartesian grid.  With only 58 Events, a huge
    grid would overfit the development subsets.  Profiles perturb coherent groups
    of engineering thresholds around the current V5 configuration.
    """

    profiles: list[tuple[str, dict[str, Any]]] = [
        ("default", {}),

        # Repeated-short path.
        (
            "short_sensitive",
            {
                "short_z_threshold": 7.0,
                "strong_short_count_24h": 5,
                "strong_short_minimum_abnormal_fraction": 0.08,
                "short_minimum_drop_score": 0.90,
            },
        ),
        (
            "short_conservative",
            {
                "short_z_threshold": 9.0,
                "strong_short_count_24h": 7,
                "strong_short_minimum_abnormal_fraction": 0.12,
                "short_minimum_drop_score": 1.10,
            },
        ),

        # Persistent path (review/fusion evidence, but can also affect state history).
        (
            "persistent_sensitive",
            {
                "persistent_z_threshold": 10.0,
                "persistent_fraction_threshold": 0.40,
                "persistent_minimum_points": 9,
            },
        ),
        (
            "persistent_conservative",
            {
                "persistent_z_threshold": 14.0,
                "persistent_fraction_threshold": 0.50,
                "persistent_minimum_points": 15,
            },
        ),

        # Localized subsystem path.
        (
            "localized_sensitive",
            {
                "localized_z_threshold": 7.0,
                "localized_strength_threshold": 7.0,
                "localized_minimum_overlap": 0.35,
                "localized_confirmation_window_minutes": 180.0,
                "localized_confirmed_support_hours": 8.0,
            },
        ),
        (
            "localized_conservative",
            {
                "localized_z_threshold": 9.0,
                "localized_strength_threshold": 9.0,
                "localized_minimum_overlap": 0.50,
                "localized_confirmation_window_minutes": 90.0,
                "localized_confirmed_support_hours": 4.0,
            },
        ),

        # Semantic fusion path.
        (
            "semantic_sensitive",
            {
                "semantic_candidate_z_threshold": 5.5,
                "semantic_confirmed_z_threshold": 7.0,
                "semantic_confirmation_points": 2,
                "semantic_fusion_confirmation_points": 2,
            },
        ),
        (
            "semantic_conservative",
            {
                "semantic_candidate_z_threshold": 6.5,
                "semantic_confirmed_z_threshold": 9.0,
                "semantic_confirmation_points": 4,
                "semantic_fusion_confirmation_points": 3,
            },
        ),

        # Targeted semantic-onset path.
        (
            "targeted_sensitive",
            {
                "targeted_change_z_threshold": 9.0,
                "targeted_change_confirmation_points": 2,
                "targeted_change_support_hours": 1.5,
            },
        ),
        (
            "targeted_conservative",
            {
                "targeted_change_z_threshold": 11.0,
                "targeted_change_confirmation_points": 3,
                "targeted_change_support_hours": 0.75,
            },
        ),

        # Chronic-background suppression.
        (
            "background_less_suppression",
            {
                "background_active_fraction_threshold": 0.25,
                "background_short_count_threshold": 25,
                "background_strong_short_count_24h": 7,
            },
        ),
        (
            "background_more_suppression",
            {
                "background_active_fraction_threshold": 0.15,
                "background_short_count_threshold": 15,
                "background_strong_short_count_24h": 9,
            },
        ),

        # Edge-trigger re-arm behaviour.
        (
            "rearm_faster",
            {
                "short_cluster_quiet_hours": 4.0,
                "short_cluster_rearm_max_count_6h": 2,
                "semantic_fusion_quiet_hours": 8.0,
                "targeted_change_quiet_hours": 8.0,
            },
        ),
        (
            "rearm_slower",
            {
                "short_cluster_quiet_hours": 8.0,
                "short_cluster_rearm_max_count_6h": 0,
                "semantic_fusion_quiet_hours": 18.0,
                "targeted_change_quiet_hours": 18.0,
            },
        ),

        # State-machine episode persistence.
        (
            "episode_shorter",
            {
                "recovery_confirmation_points": 4,
                "episode_merge_gap_hours": 3.0,
                "localized_confirmed_support_hours": 4.0,
            },
        ),
        (
            "episode_longer",
            {
                "recovery_confirmation_points": 8,
                "episode_merge_gap_hours": 9.0,
                "localized_confirmed_support_hours": 8.0,
            },
        ),

        # Status-assisted corroboration.
        (
            "status_fusion_sensitive",
            {
                "status_rare_frequency_threshold": 0.001,
                "status_novel_confirmation_points": 1,
                "status_rare_confirmation_points": 2,
                "status_confirmed_support_hours": 8.0,
            },
        ),
        (
            "status_fusion_conservative",
            {
                "status_rare_frequency_threshold": 0.0002,
                "status_novel_confirmation_points": 3,
                "status_rare_confirmation_points": 4,
                "status_confirmed_support_hours": 4.0,
            },
        ),

        # Coherent joint profiles.
        (
            "joint_sensitive",
            {
                "short_z_threshold": 7.0,
                "strong_short_count_24h": 5,
                "localized_z_threshold": 7.0,
                "localized_strength_threshold": 7.0,
                "semantic_confirmed_z_threshold": 7.0,
                "targeted_change_z_threshold": 9.0,
                "background_active_fraction_threshold": 0.25,
            },
        ),
        (
            "joint_conservative",
            {
                "short_z_threshold": 9.0,
                "strong_short_count_24h": 7,
                "localized_z_threshold": 9.0,
                "localized_strength_threshold": 9.0,
                "semantic_confirmed_z_threshold": 9.0,
                "targeted_change_z_threshold": 11.0,
                "background_active_fraction_threshold": 0.15,
            },
        ),
    ]

    if candidate_set == "compact":
        keep = {
            "default",
            "short_sensitive",
            "short_conservative",
            "localized_sensitive",
            "localized_conservative",
            "semantic_sensitive",
            "semantic_conservative",
            "joint_sensitive",
            "joint_conservative",
        }
        profiles = [item for item in profiles if item[0] in keep]

    valid_fields = {f.name for f in fields(v5_module.ReplayConfig)} - {"sampling_minutes"}
    candidates: list[dict[str, Any]] = []
    for name, changes in profiles:
        unknown = sorted(set(changes) - valid_fields)
        if unknown:
            raise ValueError(f"Candidate {name} contains invalid ReplayConfig fields: {unknown}")
        config = dict(base_config)
        config.update(changes)
        # Architecture safety remains fixed across every candidate.
        config["enable_multi_review_promotion"] = False
        config["status_independent_red_enabled"] = False
        config["communication_red_alert_enabled"] = False
        config["slow_trend_red_alert_enabled"] = False
        candidates.append(
            {
                "candidate_name": name,
                "changes": changes,
                "config": config,
                "complexity_distance": int(len(changes)),
            }
        )
    return candidates


def candidate_bank_table(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in candidates:
        row = {
            "candidate_name": item["candidate_name"],
            "complexity_distance": item["complexity_distance"],
            "changed_parameter_count": len(item["changes"]),
            "changes_json": json.dumps(item["changes"], sort_keys=True),
        }
        row.update({f"change__{k}": v for k, v in item["changes"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 3. PRECOMPUTE LABEL-BLIND REPLAY RESULTS
# =============================================================================

def build_event_file_map(v5_module, event_dir: Path) -> dict[str, Path]:
    pairs = v5_module.discover_files(event_dir, "all")
    return {normalise_event_id(source_id): path for source_id, path in pairs}


def active_inside_metrics(evaluator_module, rows: pd.DataFrame, metadata_row: pd.Series) -> dict[str, Any]:
    if rows.empty or "timestamp" not in rows.columns:
        return {
            "in_interval_active_detected": False,
            "first_active_inside_metadata": pd.NaT,
            "in_interval_detection_delay_hours": np.nan,
        }
    frame = rows.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if "active_alert_flag" in frame.columns:
        active = evaluator_module.to_bool_series(frame["active_alert_flag"])
    else:
        active = pd.Series(False, index=frame.index)
    start = pd.Timestamp(metadata_row["metadata_start"])
    end = pd.Timestamp(metadata_row["metadata_end"])
    inside = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] <= end) & active]
    if inside.empty:
        return {
            "in_interval_active_detected": False,
            "first_active_inside_metadata": pd.NaT,
            "in_interval_detection_delay_hours": np.nan,
        }
    first = pd.Timestamp(inside["timestamp"].min())
    return {
        "in_interval_active_detected": True,
        "first_active_inside_metadata": first,
        "in_interval_detection_delay_hours": float((first - start).total_seconds() / 3600.0),
    }


def precompute_candidate_event_metrics(
    *,
    args: argparse.Namespace,
    v5_module,
    evaluator_module,
    metadata: pd.DataFrame,
    event_file_map: dict[str, Path],
    feature_descriptions: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> pd.DataFrame:
    cache_dir = args.output_dir / "cache_candidate_event_metrics"
    temp_root = args.output_dir / "_temporary_replay_outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    power_signals = [item.strip() for item in args.power_signals.split(",") if item.strip()] or None
    all_tables: list[pd.DataFrame] = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        name = candidate["candidate_name"]
        cache_file = cache_dir / f"{name}.csv"
        print(f"\n[candidate {candidate_index}/{len(candidates)}] {name}")

        if cache_file.exists() and args.resume:
            cached = pd.read_csv(cache_file, low_memory=False)
            if "event_id" in cached.columns:
                cached["event_id"] = cached["event_id"].map(normalise_event_id)

            # Resume only genuinely successful Event replays.  Failed rows are
            # removed from the cache so that a corrected script/configuration
            # automatically retries them instead of treating them as complete.
            if "replay_status" in cached.columns:
                success_mask = cached["replay_status"].fillna("").astype(str).eq("success")
                cached = cached.loc[success_mask].copy()
                cached.to_csv(cache_file, index=False)

            completed = set(cached.get("event_id", pd.Series(dtype=str)).astype(str))
        else:
            if cache_file.exists():
                cache_file.unlink()
            cached = pd.DataFrame()
            completed = set()

        expected_ids = set(metadata["event_id"].map(normalise_event_id).astype(str))
        if completed >= expected_ids:
            print(f"  cache complete: {cache_file}")
            cached["candidate_name"] = name
            all_tables.append(cached)
            continue

        temp_candidate_dir = temp_root / name
        temp_candidate_dir.mkdir(parents=True, exist_ok=True)

        for row_number, (_, metadata_row) in enumerate(metadata.iterrows(), start=1):
            event_id = normalise_event_id(metadata_row["event_id"])
            if event_id in completed:
                continue
            if event_id not in event_file_map:
                raise FileNotFoundError(f"No raw Event CSV found for Event {event_id}")

            event_file = event_file_map[event_id]
            print(f"  [{row_number:02d}/{len(metadata)}] Event {event_id}")
            start_time = time.time()
            try:
                # IMPORTANT: source_metadata is deliberately empty.  The detector
                # therefore does not receive label, Event start, or Event end.
                empty_source_metadata = pd.DataFrame(
                    columns=["event_id", "metadata_label", "metadata_start", "metadata_end"]
                )
                if not isinstance(empty_source_metadata, pd.DataFrame):
                    raise TypeError(
                        f"Internal error: empty_source_metadata is {type(empty_source_metadata)!r}, "
                        "expected pandas.DataFrame"
                    )
                rows, episodes, replay_summary = v5_module.replay_one_file(
                    event_file=event_file,
                    source_id=event_id,
                    farm_id=args.farm,
                    # Label-blind replay: provide an EMPTY DataFrame, never a list.
                    # The V5 API calls `source_metadata.empty`.
                    source_metadata=empty_source_metadata.copy(),
                    output_dir=temp_candidate_dir,
                    measurement_mode=args.measurement_mode,
                    manual_power_signals=power_signals,
                    feature_descriptions=feature_descriptions,
                    config_overrides=candidate["config"],
                )

                event_summary = evaluator_module.summarise_event(
                    metadata_row=metadata_row,
                    rows=rows,
                    episodes=episodes,
                    pre_event_hours=args.pre_event_hours,
                )
                event_summary.update(active_inside_metrics(evaluator_module, rows, metadata_row))
                event_summary.update(
                    {
                        "candidate_name": name,
                        "candidate_complexity_distance": candidate["complexity_distance"],
                        "replay_status": "success",
                        "replay_error": "",
                        "replay_seconds": float(time.time() - start_time),
                    }
                )
            except Exception as exc:
                event_summary = {
                    "candidate_name": name,
                    "candidate_complexity_distance": candidate["complexity_distance"],
                    "event_id": event_id,
                    "metadata_label": str(metadata_row["metadata_label"]),
                    "metadata_start": metadata_row["metadata_start"],
                    "metadata_end": metadata_row["metadata_end"],
                    "replay_status": "failed",
                    "replay_error": str(exc),
                    "replay_seconds": float(time.time() - start_time),
                }
                print(f"    FAILED: {exc}", file=sys.stderr)
                traceback.print_exc()

            append_row_csv(cache_file, event_summary)
            completed.add(event_id)

            if not args.keep_temporary_replay_outputs:
                event_output_dir = temp_candidate_dir / event_id
                if event_output_dir.exists():
                    shutil.rmtree(event_output_dir, ignore_errors=True)

        table = pd.read_csv(cache_file, low_memory=False)
        table["event_id"] = table["event_id"].map(normalise_event_id)
        table["candidate_name"] = name
        all_tables.append(table)

        failures = table.loc[table.get("replay_status", "success") != "success"]
        if not failures.empty:
            raise RuntimeError(
                f"Candidate {name} had {len(failures)} replay failures. "
                f"See {cache_file} before continuing."
            )

    if not args.keep_temporary_replay_outputs:
        shutil.rmtree(temp_root, ignore_errors=True)

    combined = pd.concat(all_tables, ignore_index=True, sort=False)
    combined["event_id"] = combined["event_id"].map(normalise_event_id)
    combined.to_csv(args.output_dir / "candidate_event_metrics.csv", index=False)
    return combined


# =============================================================================
# 4. EVENT-LEVEL METRICS AND CONFIG SELECTION
# =============================================================================

def normalise_detection_columns(frame: pd.DataFrame, evaluator_module) -> pd.DataFrame:
    output = frame.copy()
    for column in DETECTION_COLUMNS.values():
        if column in output.columns:
            output[column] = evaluator_module.to_bool_series(output[column])
    return output


def calculate_subset_metrics(
    frame: pd.DataFrame,
    detection_column: str,
    delay_column: str = "detection_delay_hours",
) -> dict[str, Any]:
    if frame.empty:
        return {
            "n_events": 0,
            "n_anomaly": 0,
            "n_normal": 0,
            "tp": 0,
            "fn": 0,
            "tn": 0,
            "fp": 0,
            "recall": np.nan,
            "specificity": np.nan,
            "normal_fpr": np.nan,
            "balanced_accuracy": np.nan,
            "median_delay_hours": np.nan,
        }

    label = frame["metadata_label"].astype(str).str.lower()
    detected = frame[detection_column].astype(bool)
    anomaly = label.eq("anomaly")
    normal = label.eq("normal")

    tp = int((anomaly & detected).sum())
    fn = int((anomaly & ~detected).sum())
    fp = int((normal & detected).sum())
    tn = int((normal & ~detected).sum())
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, tn + fp)
    balanced = (
        float((recall + specificity) / 2.0)
        if np.isfinite(recall) and np.isfinite(specificity)
        else np.nan
    )

    delay_values = pd.to_numeric(
        frame.loc[anomaly & detected, delay_column]
        if delay_column in frame.columns
        else pd.Series(dtype=float),
        errors="coerce",
    ).dropna()

    return {
        "n_events": int(len(frame)),
        "n_anomaly": int(anomaly.sum()),
        "n_normal": int(normal.sum()),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "recall": recall,
        "specificity": specificity,
        "normal_fpr": fpr,
        "balanced_accuracy": balanced,
        "median_delay_hours": float(delay_values.median()) if not delay_values.empty else np.nan,
    }


def metrics_for_all_tiers(frame: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for tier, column in DETECTION_COLUMNS.items():
        delay_column = (
            "in_interval_detection_delay_hours"
            if tier == "in_interval_active"
            else "detection_delay_hours"
        )
        metrics = calculate_subset_metrics(frame, column, delay_column=delay_column)
        for key, value in metrics.items():
            payload[f"{tier}__{key}"] = value
    return payload


def inner_stability_metrics(
    candidate_frame: pd.DataFrame,
    development_event_ids: list[str],
    development_labels: np.ndarray,
    detection_column: str,
    inner_splits: int,
    random_state: int,
    max_normal_fpr: float,
) -> dict[str, Any]:
    ids = np.asarray(development_event_ids, dtype=object)
    labels = np.asarray(development_labels, dtype=int)
    min_class = int(pd.Series(labels).value_counts().min())
    n_splits = max(2, min(inner_splits, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    recalls: list[float] = []
    fprs: list[float] = []
    balanced: list[float] = []
    specificities: list[float] = []

    by_id = candidate_frame.set_index("event_id", drop=False)
    for _, val_idx in cv.split(ids, labels):
        fold_ids = ids[val_idx].tolist()
        fold = by_id.loc[fold_ids].reset_index(drop=True)
        metrics = calculate_subset_metrics(fold, detection_column)
        recalls.append(metrics["recall"])
        fprs.append(metrics["normal_fpr"])
        balanced.append(metrics["balanced_accuracy"])
        specificities.append(metrics["specificity"])

    def safe_nanmedian(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        return float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan

    def safe_nanstd(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0 if finite.size == 1 else np.nan

    return {
        "inner_fold_count": n_splits,
        "inner_median_recall": safe_nanmedian(recalls),
        "inner_median_normal_fpr": safe_nanmedian(fprs),
        "inner_median_balanced_accuracy": safe_nanmedian(balanced),
        "inner_median_specificity": safe_nanmedian(specificities),
        "inner_recall_std": safe_nanstd(recalls),
        "inner_fpr_std": safe_nanstd(fprs),
        "inner_fpr_constraint_violation_fraction": float(
            np.mean([float(v > max_normal_fpr) for v in fprs if np.isfinite(v)])
        ) if any(np.isfinite(v) for v in fprs) else np.nan,
    }


def choose_candidate(
    *,
    candidate_metrics: pd.DataFrame,
    candidates: list[dict[str, Any]],
    development_event_ids: list[str],
    metadata_lookup: pd.DataFrame,
    selection_target: str,
    max_normal_fpr: float,
    inner_splits: int,
    random_state: int,
) -> tuple[str, pd.DataFrame]:
    detection_column = DETECTION_COLUMNS[selection_target]
    secondary_target = (
        "in_interval_active" if selection_target != "in_interval_active" else "operational"
    )
    secondary_column = DETECTION_COLUMNS[secondary_target]

    dev_meta = metadata_lookup.set_index("event_id").loc[development_event_ids]
    dev_labels = dev_meta["metadata_label"].map(LABEL_TO_INT).to_numpy(dtype=int)

    rows: list[dict[str, Any]] = []
    complexity = {item["candidate_name"]: item["complexity_distance"] for item in candidates}

    for candidate_name, group in candidate_metrics.groupby("candidate_name", sort=False):
        group = group.loc[group["event_id"].isin(development_event_ids)].copy()
        if len(group) != len(development_event_ids):
            missing = sorted(set(development_event_ids) - set(group["event_id"]))
            raise RuntimeError(f"Candidate {candidate_name} missing development Events: {missing}")

        primary = calculate_subset_metrics(group, detection_column)
        secondary = calculate_subset_metrics(group, secondary_column)
        stability = inner_stability_metrics(
            candidate_frame=group,
            development_event_ids=development_event_ids,
            development_labels=dev_labels,
            detection_column=detection_column,
            inner_splits=inner_splits,
            random_state=random_state,
            max_normal_fpr=max_normal_fpr,
        )
        feasible = bool(
            np.isfinite(primary["normal_fpr"])
            and primary["normal_fpr"] <= max_normal_fpr
        )
        rows.append(
            {
                "candidate_name": candidate_name,
                "complexity_distance": int(complexity[candidate_name]),
                "feasible": feasible,
                "pooled_recall": primary["recall"],
                "pooled_normal_fpr": primary["normal_fpr"],
                "pooled_specificity": primary["specificity"],
                "pooled_balanced_accuracy": primary["balanced_accuracy"],
                "secondary_recall": secondary["recall"],
                **stability,
            }
        )

    table = pd.DataFrame(rows)
    feasible_table = table.loc[table["feasible"]].copy()

    if not feasible_table.empty:
        ranked = feasible_table.sort_values(
            by=[
                "inner_median_recall",
                "pooled_recall",
                "inner_median_balanced_accuracy",
                "secondary_recall",
                "pooled_specificity",
                "inner_recall_std",
                "pooled_normal_fpr",
                "complexity_distance",
                "candidate_name",
            ],
            ascending=[False, False, False, False, False, True, True, True, True],
            kind="mergesort",
        )
    else:
        # If no profile satisfies the FPR cap, prefer the lowest FPR first.
        ranked = table.sort_values(
            by=[
                "pooled_normal_fpr",
                "inner_median_normal_fpr",
                "inner_median_recall",
                "pooled_recall",
                "complexity_distance",
                "candidate_name",
            ],
            ascending=[True, True, False, False, True, True],
            kind="mergesort",
        )

    selected = str(ranked.iloc[0]["candidate_name"])
    table["selected"] = table["candidate_name"].eq(selected)
    table["selection_target"] = selection_target
    table["max_normal_fpr"] = max_normal_fpr
    return selected, table


# =============================================================================
# 5. CANONICAL / CHRONOLOGICAL HELD-OUT CHECKS
# =============================================================================

def run_single_holdout(
    *,
    name: str,
    development_ids: list[str],
    holdout_ids: list[str],
    candidate_metrics: pd.DataFrame,
    candidates: list[dict[str, Any]],
    metadata: pd.DataFrame,
    args: argparse.Namespace,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected, selection_table = choose_candidate(
        candidate_metrics=candidate_metrics,
        candidates=candidates,
        development_event_ids=development_ids,
        metadata_lookup=metadata,
        selection_target=args.selection_target,
        max_normal_fpr=args.max_normal_fpr,
        inner_splits=args.inner_splits,
        random_state=random_state,
    )

    held = candidate_metrics.loc[
        candidate_metrics["candidate_name"].eq(selected)
        & candidate_metrics["event_id"].isin(holdout_ids)
    ].copy()
    held = held.sort_values("metadata_start").reset_index(drop=True)
    metrics = {
        "analysis_name": name,
        "selected_candidate": selected,
        "development_events": len(development_ids),
        "holdout_events": len(holdout_ids),
        "development_event_ids": ";".join(development_ids),
        "holdout_event_ids": ";".join(holdout_ids),
        **metrics_for_all_tiers(held),
    }
    return metrics, selection_table, held


# =============================================================================
# 6. REPEATED NESTED STRATIFIED EVENT-LEVEL VALIDATION
# =============================================================================

def run_repeated_nested_cv(
    *,
    candidate_metrics: pd.DataFrame,
    candidates: list[dict[str, Any]],
    metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = metadata.copy().reset_index(drop=True)
    y = meta["metadata_label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    event_ids = meta["event_id"].astype(str).to_numpy(dtype=object)

    rkf = RepeatedStratifiedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    assignment_rows: list[dict[str, Any]] = []

    folds_per_repeat = args.outer_splits
    for outer_index, (dev_idx, hold_idx) in enumerate(rkf.split(event_ids, y), start=1):
        repeat_index = (outer_index - 1) // folds_per_repeat + 1
        fold_index = (outer_index - 1) % folds_per_repeat + 1
        dev_ids = event_ids[dev_idx].tolist()
        hold_ids = event_ids[hold_idx].tolist()

        print(
            f"[outer] repeat {repeat_index}/{args.outer_repeats}, "
            f"fold {fold_index}/{args.outer_splits}: "
            f"dev={len(dev_ids)}, holdout={len(hold_ids)}"
        )

        selected, selection_table = choose_candidate(
            candidate_metrics=candidate_metrics,
            candidates=candidates,
            development_event_ids=dev_ids,
            metadata_lookup=meta,
            selection_target=args.selection_target,
            max_normal_fpr=args.max_normal_fpr,
            inner_splits=args.inner_splits,
            random_state=args.random_state + outer_index * 101,
        )
        selection_table.insert(0, "outer_index", outer_index)
        selection_table.insert(1, "repeat_index", repeat_index)
        selection_table.insert(2, "fold_index", fold_index)
        selection_rows.append(selection_table)

        held = candidate_metrics.loc[
            candidate_metrics["candidate_name"].eq(selected)
            & candidate_metrics["event_id"].isin(hold_ids)
        ].copy()
        held["outer_index"] = outer_index
        held["repeat_index"] = repeat_index
        held["fold_index"] = fold_index
        held["selected_candidate"] = selected
        prediction_rows.append(held)

        fold_metric = {
            "outer_index": outer_index,
            "repeat_index": repeat_index,
            "fold_index": fold_index,
            "selected_candidate": selected,
            "development_events": len(dev_ids),
            "holdout_events": len(hold_ids),
            **metrics_for_all_tiers(held),
        }
        fold_rows.append(fold_metric)

        dev_set = set(dev_ids)
        hold_set = set(hold_ids)
        for event_id in event_ids:
            assignment_rows.append(
                {
                    "outer_index": outer_index,
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    "event_id": event_id,
                    "subset": "development" if event_id in dev_set else "holdout" if event_id in hold_set else "",
                    "selected_candidate": selected,
                }
            )

    fold_table = pd.DataFrame(fold_rows)
    prediction_table = pd.concat(prediction_rows, ignore_index=True, sort=False)
    selection_table = pd.concat(selection_rows, ignore_index=True, sort=False)
    assignment_table = pd.DataFrame(assignment_rows)
    return fold_table, prediction_table, selection_table, assignment_table


# =============================================================================
# 7. OUT-OF-SAMPLE EVENT AGGREGATION + BOOTSTRAP
# =============================================================================

def build_per_event_holdout_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_id, group in predictions.groupby("event_id", sort=False):
        label = str(group["metadata_label"].iloc[0])
        row: dict[str, Any] = {
            "event_id": event_id,
            "metadata_label": label,
            "holdout_appearances": int(len(group)),
            "metadata_start": pd.to_datetime(group["metadata_start"], errors="coerce").min(),
            "metadata_end": pd.to_datetime(group["metadata_end"], errors="coerce").max(),
        }
        for tier, column in DETECTION_COLUMNS.items():
            values = group[column].astype(bool).astype(float)
            row[f"{tier}_detection_rate"] = float(values.mean())
            row[f"{tier}_detected_count"] = int(values.sum())
        operational_delay = pd.to_numeric(
            group.loc[group[DETECTION_COLUMNS["operational"]].astype(bool), "detection_delay_hours"],
            errors="coerce",
        ).dropna()
        adjusted_delay = pd.to_numeric(
            group.loc[group[DETECTION_COLUMNS["in_interval_active"]].astype(bool), "in_interval_detection_delay_hours"],
            errors="coerce",
        ).dropna()
        row["median_operational_delay_hours_when_detected"] = (
            float(operational_delay.median()) if not operational_delay.empty else np.nan
        )
        row["median_in_interval_delay_hours_when_detected"] = (
            float(adjusted_delay.median()) if not adjusted_delay.empty else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["metadata_label", "event_id"]).reset_index(drop=True)


def bootstrap_mean(
    values: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    estimate = float(values.mean())
    if values.size == 1 or repeats <= 0:
        return estimate, estimate, estimate
    samples = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    return estimate, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def bootstrap_event_level_rates(
    per_event: pd.DataFrame,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    anomaly = per_event.loc[per_event["metadata_label"].eq("anomaly")]
    normal = per_event.loc[per_event["metadata_label"].eq("normal")]
    rows: list[dict[str, Any]] = []

    for tier in DETECTION_COLUMNS:
        col = f"{tier}_detection_rate"
        recall, recall_low, recall_high = bootstrap_mean(
            anomaly[col].to_numpy(float), repeats, rng
        )
        fpr, fpr_low, fpr_high = bootstrap_mean(
            normal[col].to_numpy(float), repeats, rng
        )
        rows.append(
            {
                "tier": tier,
                "out_of_sample_mean_recall": recall,
                "bootstrap_95_low_recall": recall_low,
                "bootstrap_95_high_recall": recall_high,
                "out_of_sample_mean_normal_fpr": fpr,
                "bootstrap_95_low_normal_fpr": fpr_low,
                "bootstrap_95_high_normal_fpr": fpr_high,
                "out_of_sample_mean_specificity": 1.0 - fpr if np.isfinite(fpr) else np.nan,
                "bootstrap_95_low_specificity": 1.0 - fpr_high if np.isfinite(fpr_high) else np.nan,
                "bootstrap_95_high_specificity": 1.0 - fpr_low if np.isfinite(fpr_low) else np.nan,
                "anomaly_events": int(len(anomaly)),
                "normal_events": int(len(normal)),
                "bootstrap_repeats": repeats,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 8. SUMMARIES + FIGURES
# =============================================================================

def summarise_outer_folds(fold_table: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in fold_table.columns
        if column.endswith("__recall")
        or column.endswith("__specificity")
        or column.endswith("__normal_fpr")
        or column.endswith("__balanced_accuracy")
        or column.endswith("__median_delay_hours")
    ]
    rows = []
    for column in metric_columns:
        stats = percentile_summary(pd.to_numeric(fold_table[column], errors="coerce"))
        rows.append({"metric": column, **stats})
    return pd.DataFrame(rows)


def plot_outer_metric_distributions(fold_table: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        "operational__recall",
        "in_interval_active__recall",
        "operational__specificity",
        "operational__normal_fpr",
    ]
    labels = [
        "Strict operational recall",
        "In-interval active recall",
        "Strict operational specificity",
        "Strict operational normal FPR",
    ]
    values = [pd.to_numeric(fold_table[m], errors="coerce").dropna().to_numpy() for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_title("Repeated held-out metric distributions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_repeated_holdout_metric_distributions.png", dpi=180)
    plt.close(fig)


def plot_selected_candidate_frequency(fold_table: pd.DataFrame, output_dir: Path) -> None:
    counts = fold_table["selected_candidate"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(counts))))
    ax.barh(counts.index, counts.values)
    ax.set_xlabel("Number of outer held-out folds selected")
    ax.set_ylabel("Candidate configuration")
    ax.set_title("Development-only configuration selection frequency")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_selected_candidate_frequency.png", dpi=180)
    plt.close(fig)


def plot_per_event_detection_rates(per_event: pd.DataFrame, output_dir: Path) -> None:
    table = per_event.sort_values(
        ["metadata_label", "operational_detection_rate", "event_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, table["operational_detection_rate"].to_numpy(float))
    ax.set_xticks(x)
    ax.set_xticklabels(table["event_id"].astype(str), rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Out-of-sample operational detection rate")
    ax.set_xlabel("Event ID")
    ax.set_title("Per-Event held-out detection stability across repeats")
    boundary = int((table["metadata_label"] == "anomaly").sum())
    # If lexical label ordering places anomaly first, mark the class boundary.
    if 0 < boundary < len(table):
        ax.axvline(boundary - 0.5, linestyle="--", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_per_event_holdout_detection_rate.png", dpi=180)
    plt.close(fig)


def plot_default_vs_holdout(
    default_metrics: dict[str, Any],
    bootstrap_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    operational = bootstrap_table.loc[bootstrap_table["tier"].eq("operational")].iloc[0]
    adjusted = bootstrap_table.loc[bootstrap_table["tier"].eq("in_interval_active")].iloc[0]
    labels = [
        "Full retrospective\nstrict recall",
        "Repeated held-out\nstrict recall",
        "Full retrospective\nin-interval recall",
        "Repeated held-out\nin-interval recall",
    ]
    values = [
        default_metrics["operational__recall"],
        operational["out_of_sample_mean_recall"],
        default_metrics["in_interval_active__recall"],
        adjusted["out_of_sample_mean_recall"],
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Retrospective default vs repeated held-out sensitivity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_4_retrospective_vs_heldout_recall.png", dpi=180)
    plt.close(fig)


def build_markdown_report(
    *,
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    candidates: list[dict[str, Any]],
    default_metrics: dict[str, Any],
    canonical_metrics: dict[str, Any],
    chronological_metrics: Optional[dict[str, Any]],
    fold_summary: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    frequency_table: pd.DataFrame,
) -> str:
    def pct(value: Any) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NA"
        return f"{100.0 * value:.1f}%" if np.isfinite(value) else "NA"

    lines: list[str] = []
    lines.append("# Streaming V5 Held-out Sensitivity Analysis")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This is a post-hoc robustness check for global threshold tuning. "
        "Detector replay is label-blind, configuration selection uses development "
        "Events only, and the selected configuration is frozen before held-out evaluation."
    )
    lines.append("")
    lines.append(
        "Because the 58 Events had already been inspected during the historical development "
        "of Streaming V5, this analysis must not be described as a completely untouched "
        "external test."
    )
    lines.append("")
    lines.append("## Dataset and design")
    lines.append("")
    lines.append(f"- Events: **{len(metadata)}**")
    lines.append(f"- Anomaly Events: **{int((metadata['metadata_label'] == 'anomaly').sum())}**")
    lines.append(f"- Normal Events: **{int((metadata['metadata_label'] == 'normal').sum())}**")
    lines.append(f"- Candidate configurations: **{len(candidates)}**")
    lines.append(
        f"- Repeated nested Event-level validation: **{args.outer_splits} folds x "
        f"{args.outer_repeats} repeats = {args.outer_splits * args.outer_repeats} outer hold-outs**"
    )
    lines.append(f"- Primary selection target: **{args.selection_target}**")
    lines.append(f"- Development normal-FPR constraint: **<= {args.max_normal_fpr:.2f}**")
    lines.append("")
    lines.append("## Full retrospective default configuration")
    lines.append("")
    lines.append(f"- Strict operational recall: **{pct(default_metrics['operational__recall'])}**")
    lines.append(f"- Strict operational specificity: **{pct(default_metrics['operational__specificity'])}**")
    lines.append(f"- Strict operational normal FPR: **{pct(default_metrics['operational__normal_fpr'])}**")
    lines.append(f"- In-interval Active recall: **{pct(default_metrics['in_interval_active__recall'])}**")
    lines.append("")
    lines.append("## Canonical stratified held-out split")
    lines.append("")
    lines.append(f"- Selected candidate: **{canonical_metrics['selected_candidate']}**")
    lines.append(f"- Strict operational recall: **{pct(canonical_metrics['operational__recall'])}**")
    lines.append(f"- Strict operational normal FPR: **{pct(canonical_metrics['operational__normal_fpr'])}**")
    lines.append(f"- In-interval Active recall: **{pct(canonical_metrics['in_interval_active__recall'])}**")
    lines.append("")

    if chronological_metrics is not None:
        lines.append("## Chronological latest-Events held-out split")
        lines.append("")
        lines.append(f"- Selected candidate: **{chronological_metrics['selected_candidate']}**")
        lines.append(f"- Strict operational recall: **{pct(chronological_metrics['operational__recall'])}**")
        lines.append(f"- Strict operational normal FPR: **{pct(chronological_metrics['operational__normal_fpr'])}**")
        lines.append(f"- In-interval Active recall: **{pct(chronological_metrics['in_interval_active__recall'])}**")
        lines.append("")

    lines.append("## Repeated held-out out-of-sample estimates")
    lines.append("")
    display = bootstrap_table.copy()
    for col in [
        "out_of_sample_mean_recall",
        "bootstrap_95_low_recall",
        "bootstrap_95_high_recall",
        "out_of_sample_mean_normal_fpr",
        "bootstrap_95_low_normal_fpr",
        "bootstrap_95_high_normal_fpr",
    ]:
        display[col] = display[col].map(pct)
    lines.append(display.to_markdown(index=False))
    lines.append("")
    lines.append("## Configuration stability")
    lines.append("")
    lines.append(frequency_table.head(15).to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append(
        "If the held-out distributions remain broadly comparable with the full retrospective "
        "results, the original conclusions are less sensitive to one particular threshold "
        "configuration. If performance collapses or selected configurations vary strongly "
        "across splits, the retrospective results should be interpreted as threshold-sensitive."
    )
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# 9. MAIN ANALYSIS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-script", type=Path, required=True)
    parser.add_argument("--evaluator-script", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--feature-description", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--farm", default="C")
    parser.add_argument("--measurement-mode", choices=["avg_only", "all"], default="avg_only")
    parser.add_argument("--power-signals", default="power_2,power_5,power_6,power_17")
    parser.add_argument("--base-config-json", type=Path, default=None)
    parser.add_argument("--candidate-set", choices=["compact", "full"], default="full")
    parser.add_argument(
        "--selection-target",
        choices=list(DETECTION_COLUMNS),
        default="operational",
        help="Primary development-only target used to select the candidate configuration.",
    )
    parser.add_argument("--max-normal-fpr", type=float, default=0.35)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--canonical-holdout-fraction", type=float, default=0.30)
    parser.add_argument("--pre-event-hours", type=float, default=168.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--stage",
        choices=["all", "precompute", "analyse"],
        default="all",
        help="Use precompute for the expensive detector pass, analyse to reuse cached metrics.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-temporary-replay-outputs", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in [args.stream_script, args.evaluator_script, args.metadata, args.event_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.feature_description is not None and not args.feature_description.exists():
        raise FileNotFoundError(args.feature_description)
    if not (0.0 < args.max_normal_fpr < 1.0):
        raise ValueError("--max-normal-fpr must be in (0, 1).")
    if args.outer_splits < 2 or args.outer_repeats < 1 or args.inner_splits < 2:
        raise ValueError("CV split/repeat counts are invalid.")
    if not (0.1 <= args.canonical_holdout_fraction <= 0.5):
        raise ValueError("--canonical-holdout-fraction should be between 0.10 and 0.50.")


def main() -> int:
    args = parse_args()
    print(f"Sensitivity script build: {SCRIPT_BUILD}")
    print(f"Sensitivity script path: {Path(__file__).resolve()}")
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v5 = import_module_from_path(args.stream_script, "streaming_v5_for_sensitivity")
    evaluator = import_module_from_path(args.evaluator_script, "streaming_eval_for_sensitivity")

    metadata = evaluator.load_metadata(args.metadata, args.farm)
    metadata["event_id"] = metadata["event_id"].map(normalise_event_id)
    metadata["metadata_label"] = metadata["metadata_label"].astype(str).str.lower()
    metadata = metadata.loc[metadata["metadata_label"].isin(LABEL_TO_INT)].copy()
    metadata = metadata.sort_values(["metadata_start", "event_id"]).reset_index(drop=True)

    counts = metadata["metadata_label"].value_counts().to_dict()
    if counts.get("anomaly", 0) < args.outer_splits or counts.get("normal", 0) < args.outer_splits:
        raise ValueError(
            "Each class must contain at least --outer-splits Events for repeated stratified CV."
        )

    event_file_map = build_event_file_map(v5, args.event_dir)
    missing_files = sorted(set(metadata["event_id"]) - set(event_file_map))
    if missing_files:
        raise FileNotFoundError(f"Missing raw Event CSVs for Events: {missing_files}")

    feature_descriptions = v5.read_feature_descriptions(args.feature_description)
    base_config = load_base_config(v5, args.base_config_json)
    candidates = make_candidate_bank(v5, base_config, args.candidate_set)
    candidate_bank_table(candidates).to_csv(args.output_dir / "candidate_bank.csv", index=False)
    write_json(args.output_dir / "base_replay_config.json", base_config)
    write_json(
        args.output_dir / "candidate_configs.json",
        {item["candidate_name"]: item["config"] for item in candidates},
    )

    if args.stage in {"all", "precompute"}:
        candidate_metrics = precompute_candidate_event_metrics(
            args=args,
            v5_module=v5,
            evaluator_module=evaluator,
            metadata=metadata,
            event_file_map=event_file_map,
            feature_descriptions=feature_descriptions,
            candidates=candidates,
        )
        if args.stage == "precompute":
            print("Precomputation complete. Re-run with --stage analyse to perform held-out analysis.")
            return 0
    else:
        combined_path = args.output_dir / "candidate_event_metrics.csv"
        if not combined_path.exists():
            # Rebuild compact combined table from per-candidate caches if possible.
            cache_dir = args.output_dir / "cache_candidate_event_metrics"
            tables = []
            for candidate in candidates:
                path = cache_dir / f"{candidate['candidate_name']}.csv"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing {combined_path} and candidate cache {path}. Run --stage precompute first."
                    )
                table = pd.read_csv(path, low_memory=False)
                table["candidate_name"] = candidate["candidate_name"]
                tables.append(table)
            candidate_metrics = pd.concat(tables, ignore_index=True, sort=False)
            candidate_metrics.to_csv(combined_path, index=False)
        else:
            candidate_metrics = pd.read_csv(combined_path, low_memory=False)

    candidate_metrics["event_id"] = candidate_metrics["event_id"].map(normalise_event_id)
    candidate_metrics = normalise_detection_columns(candidate_metrics, evaluator)

    expected_rows = len(metadata) * len(candidates)
    if len(candidate_metrics) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} candidate-event rows but found {len(candidate_metrics)}. "
            "Check replay failures or stale caches."
        )

    # -------------------------------------------------------------------------
    # Full retrospective baseline: current/default candidate on all 58 Events.
    # -------------------------------------------------------------------------
    default_all = candidate_metrics.loc[candidate_metrics["candidate_name"].eq("default")].copy()
    default_metrics = metrics_for_all_tiers(default_all)
    pd.DataFrame([default_metrics]).to_csv(
        args.output_dir / "full_retrospective_default_metrics.csv", index=False
    )

    # -------------------------------------------------------------------------
    # Canonical 70/30 stratified held-out subset.
    # -------------------------------------------------------------------------
    all_ids = metadata["event_id"].astype(str).to_numpy()
    all_y = metadata["metadata_label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    dev_ids, hold_ids = train_test_split(
        all_ids,
        test_size=args.canonical_holdout_fraction,
        random_state=args.random_state,
        stratify=all_y,
    )
    canonical_metrics, canonical_selection, canonical_predictions = run_single_holdout(
        name="canonical_stratified_holdout",
        development_ids=sorted(dev_ids.tolist()),
        holdout_ids=sorted(hold_ids.tolist()),
        candidate_metrics=candidate_metrics,
        candidates=candidates,
        metadata=metadata,
        args=args,
        random_state=args.random_state + 5000,
    )
    pd.DataFrame([canonical_metrics]).to_csv(
        args.output_dir / "canonical_holdout_metrics.csv", index=False
    )
    write_json(args.output_dir / "canonical_holdout_metrics.json", canonical_metrics)
    canonical_selection.to_csv(args.output_dir / "canonical_development_candidate_selection.csv", index=False)
    canonical_predictions.to_csv(args.output_dir / "canonical_holdout_event_predictions.csv", index=False)

    # -------------------------------------------------------------------------
    # Chronological latest-Events held-out subset.
    # -------------------------------------------------------------------------
    chronological_metrics: Optional[dict[str, Any]] = None
    chronological_selection = pd.DataFrame()
    chronological_predictions = pd.DataFrame()
    chrono_count = max(1, int(math.ceil(len(metadata) * args.canonical_holdout_fraction)))
    chrono_count = min(chrono_count, len(metadata) - 2)
    chrono_hold = metadata.tail(chrono_count).copy()
    chrono_dev = metadata.iloc[:-chrono_count].copy()
    if chrono_hold["metadata_label"].nunique() == 2 and chrono_dev["metadata_label"].nunique() == 2:
        chronological_metrics, chronological_selection, chronological_predictions = run_single_holdout(
            name="chronological_latest_events_holdout",
            development_ids=chrono_dev["event_id"].astype(str).tolist(),
            holdout_ids=chrono_hold["event_id"].astype(str).tolist(),
            candidate_metrics=candidate_metrics,
            candidates=candidates,
            metadata=metadata,
            args=args,
            random_state=args.random_state + 6000,
        )
        pd.DataFrame([chronological_metrics]).to_csv(
            args.output_dir / "chronological_holdout_metrics.csv", index=False
        )
        write_json(args.output_dir / "chronological_holdout_metrics.json", chronological_metrics)
        chronological_selection.to_csv(
            args.output_dir / "chronological_development_candidate_selection.csv", index=False
        )
        chronological_predictions.to_csv(
            args.output_dir / "chronological_holdout_event_predictions.csv", index=False
        )
    else:
        print(
            "[warning] Chronological holdout did not contain both classes in both subsets; "
            "chronological sensitivity check skipped."
        )

    # -------------------------------------------------------------------------
    # Strongest robustness component: repeated nested stratified outer hold-outs.
    # -------------------------------------------------------------------------
    fold_table, outer_predictions, outer_selection, split_assignments = run_repeated_nested_cv(
        candidate_metrics=candidate_metrics,
        candidates=candidates,
        metadata=metadata,
        args=args,
    )
    fold_table.to_csv(args.output_dir / "outer_fold_metrics.csv", index=False)
    outer_predictions.to_csv(args.output_dir / "outer_holdout_event_predictions.csv", index=False)
    outer_selection.to_csv(args.output_dir / "inner_candidate_selection_by_outer_fold.csv", index=False)
    split_assignments.to_csv(args.output_dir / "outer_split_assignments.csv", index=False)

    fold_summary = summarise_outer_folds(fold_table)
    fold_summary.to_csv(args.output_dir / "outer_fold_metric_summary.csv", index=False)

    per_event = build_per_event_holdout_summary(outer_predictions)
    per_event.to_csv(args.output_dir / "per_event_holdout_summary.csv", index=False)

    bootstrap_table = bootstrap_event_level_rates(
        per_event,
        repeats=args.bootstrap_repeats,
        random_state=args.random_state + 9000,
    )
    bootstrap_table.to_csv(args.output_dir / "event_level_bootstrap_summary.csv", index=False)

    frequency = (
        fold_table["selected_candidate"]
        .value_counts()
        .rename_axis("candidate_name")
        .reset_index(name="selected_outer_folds")
    )
    frequency["selection_fraction"] = frequency["selected_outer_folds"] / len(fold_table)
    frequency.to_csv(args.output_dir / "selected_candidate_frequency.csv", index=False)

    # Figures.
    plot_outer_metric_distributions(fold_table, args.output_dir)
    plot_selected_candidate_frequency(fold_table, args.output_dir)
    plot_per_event_detection_rates(per_event, args.output_dir)
    plot_default_vs_holdout(default_metrics, bootstrap_table, args.output_dir)

    report = build_markdown_report(
        args=args,
        metadata=metadata,
        candidates=candidates,
        default_metrics=default_metrics,
        canonical_metrics=canonical_metrics,
        chronological_metrics=chronological_metrics,
        fold_summary=fold_summary,
        bootstrap_table=bootstrap_table,
        frequency_table=frequency,
    )
    (args.output_dir / "heldout_sensitivity_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "analysis": "Streaming V5 held-out sensitivity / robustness analysis",
        "important_interpretation": (
            "Post-hoc robustness check only; historically inspected Events are not a truly untouched external test."
        ),
        "stream_script": str(args.stream_script),
        "evaluator_script": str(args.evaluator_script),
        "metadata": str(args.metadata),
        "event_dir": str(args.event_dir),
        "feature_description": str(args.feature_description) if args.feature_description else "",
        "event_count": int(len(metadata)),
        "anomaly_events": int((metadata["metadata_label"] == "anomaly").sum()),
        "normal_events": int((metadata["metadata_label"] == "normal").sum()),
        "candidate_count": len(candidates),
        "candidate_set": args.candidate_set,
        "selection_target": args.selection_target,
        "max_normal_fpr": args.max_normal_fpr,
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "outer_holdouts": int(args.outer_splits * args.outer_repeats),
        "inner_splits": args.inner_splits,
        "canonical_holdout_fraction": args.canonical_holdout_fraction,
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_state": args.random_state,
        "detector_received_metadata_during_precompute": False,
        "architecture_constraints": {
            "multi_review_promotion": False,
            "status_independent_red": False,
            "communication_red_alert": False,
            "slow_trend_red_alert": False,
        },
        "outputs": [
            "candidate_bank.csv",
            "candidate_event_metrics.csv",
            "full_retrospective_default_metrics.csv",
            "canonical_holdout_metrics.csv",
            "chronological_holdout_metrics.csv (if both classes available)",
            "outer_fold_metrics.csv",
            "outer_holdout_event_predictions.csv",
            "per_event_holdout_summary.csv",
            "event_level_bootstrap_summary.csv",
            "selected_candidate_frequency.csv",
            "heldout_sensitivity_report.md",
        ],
    }
    write_json(args.output_dir / "analysis_manifest.json", manifest)

    print("\n[DONE]")
    print(f"Results: {args.output_dir}")
    print(
        "Primary repeated held-out summary: "
        f"{args.output_dir / 'event_level_bootstrap_summary.csv'}"
    )
    print(
        "Human-readable report: "
        f"{args.output_dir / 'heldout_sensitivity_report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
