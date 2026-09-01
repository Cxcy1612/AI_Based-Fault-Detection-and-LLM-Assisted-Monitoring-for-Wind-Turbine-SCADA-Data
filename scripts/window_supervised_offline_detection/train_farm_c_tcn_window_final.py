#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a Farm C supervised TCN window classifier.

This version is a fair architecture comparison with the best 1D-CNN setup. It
keeps the same Event-grouped folds, training-only feature selection and scaling,
window construction, class sampling, validation-only FAR-constrained threshold
selection, and window-level evaluation. Only the neural architecture is changed
from a compact 1D-CNN to a compact causal dilated Temporal Convolutional Network
(TCN).

The TCN uses residual causal Conv1D blocks with dilation rates 1, 2, 4 and 8,
followed by combined global-average and global-max pooling. The maximum-pooling
branch is included to preserve short local anomaly activations that may be
diluted by average pooling. No Event-level aggregation is applied.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass
class Config:
    metadata: str
    event_csv_dir: str
    output_dir: str
    event_id_column: str = "event_id"
    event_label_column: str = "event_label"
    event_start_column: str = "event_start"
    event_end_column: str = "event_end"
    timestamp_column: str = "time_stamp"
    n_splits: int = 5
    seed: int = 42
    window_size: int = 144
    evaluation_step_size: int = 72
    train_anomaly_step_size: int = 48
    train_normal_step_size: int = 72
    anomaly_overlap_threshold: float = 0.10
    normal_gap_hours: float = 24.0
    feature_mode: str = "avg_std"
    top_k_features: int = 20
    missing_threshold: float = 0.30
    event_coverage_threshold: float = 0.80
    near_constant_threshold: float = 0.99
    correlation_threshold: float = 0.95
    feature_sampling_rows_per_event: int = 4000
    train_normal_to_anomaly_ratio: float = 3.0
    batch_size: int = 64
    epochs: int = 80
    early_stopping_patience: int = 10
    learning_rate: float = 1e-3
    tcn_filters: int = 32
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.30
    tcn_l2: float = 5e-4
    window_threshold: float = 0.50
    threshold_target_far: float = 0.05
    threshold_grid_size: int = 501


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install it with:\n"
            "python -m pip install tensorflow"
        ) from exc
    return tf


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    last_error: Exception | None = None
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(columns: Iterable[str], requested: str, alternatives: Iterable[str]) -> str:
    mapping = {norm_name(col): col for col in columns}
    for candidate in [requested, *alternatives]:
        if norm_name(candidate) in mapping:
            return mapping[norm_name(candidate)]
    raise KeyError(f"Cannot find {requested!r}. Available: {list(columns)[:30]}")


def normalize_event_id(value: Any) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"0", "normal", "healthy", "false", "no", "negative"}:
        return 0
    if text in {"1", "anomaly", "abnormal", "fault", "faulty", "true", "yes", "positive"}:
        return 1
    try:
        return int(float(value) > 0)
    except Exception as exc:
        raise ValueError(f"Unsupported label: {value!r}") from exc


def find_event_csv(root: Path, event_id: str) -> Path:
    for candidate in (
        root / f"{event_id}.csv",
        root / f"event_{event_id}.csv",
        root / f"Event_{event_id}.csv",
    ):
        if candidate.exists():
            return candidate
    matches = list(root.rglob(f"{event_id}.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No CSV found for Event {event_id} under {root}")
    raise RuntimeError(f"Multiple CSVs found for Event {event_id}: {matches}")


def load_metadata(cfg: Config) -> pd.DataFrame:
    raw = read_csv_auto(Path(cfg.metadata))
    id_col = find_column(raw.columns, cfg.event_id_column, ["event", "eventid", "id"])
    label_col = find_column(raw.columns, cfg.event_label_column, ["label", "class"])
    start_col = find_column(raw.columns, cfg.event_start_column, ["start", "start_time"])
    end_col = find_column(raw.columns, cfg.event_end_column, ["end", "end_time"])
    meta = pd.DataFrame({
        "event_id": raw[id_col].map(normalize_event_id),
        "event_label": raw[label_col].map(normalize_label),
        "event_start": pd.to_datetime(raw[start_col], errors="coerce"),
        "event_end": pd.to_datetime(raw[end_col], errors="coerce"),
    })
    if meta["event_id"].duplicated().any():
        raise ValueError("Duplicate Event IDs found in metadata.")
    if meta[["event_start", "event_end"]].isna().any().any():
        raise ValueError("Invalid event_start/event_end values found.")
    root = Path(cfg.event_csv_dir)
    meta["event_csv"] = meta["event_id"].map(lambda x: str(find_event_csv(root, x)))
    return meta.sort_values("event_id").reset_index(drop=True)


def load_event(event_row: pd.Series, cfg: Config, selected: list[str] | None = None) -> pd.DataFrame:
    df = read_csv_auto(Path(event_row["event_csv"]))
    time_col = find_column(df.columns, cfg.timestamp_column, ["timestamp", "time", "datetime", "date_time"])
    if selected is not None:
        available = [col for col in selected if col in df.columns]
        df = df[[time_col, *available]].copy()
    else:
        df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="last")
    return df.rename(columns={time_col: "__timestamp__"}).reset_index(drop=True)


LEAKAGE_TOKENS = {
    "label", "target", "fault", "failure", "anomaly", "maintenance",
    "description", "diagnosis", "prediction", "event_start", "event_end",
    "train_test",
}
NON_SENSOR_NAMES = {
    "time_stamp", "timestamp", "time", "datetime", "date_time", "asset_id",
    "event_id", "id", "train_test", "status_type_id",
}


def feature_allowed(name: str, mode: str) -> bool:
    norm = norm_name(name)
    if norm in NON_SENSOR_NAMES or any(token in norm for token in LEAKAGE_TOKENS):
        return False
    if mode == "avg":
        return norm.endswith("_avg")
    if mode == "avg_std":
        return norm.endswith("_avg") or norm.endswith("_std")
    if mode == "all_numeric":
        return True
    raise ValueError("feature_mode must be avg, avg_std or all_numeric")


def discover_candidates(train_events: pd.DataFrame, cfg: Config) -> list[str]:
    common: set[str] | None = None
    for _, row in train_events.iterrows():
        df = load_event(row, cfg)
        numeric = set()
        for col in df.columns:
            if col == "__timestamp__" or not feature_allowed(col, cfg.feature_mode):
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                numeric.add(col)
        common = numeric if common is None else common & numeric
    if not common:
        raise RuntimeError("No common numeric candidate features found.")
    return sorted(common)


def sample_training_rows(
    train_events: pd.DataFrame,
    features: list[str],
    cfg: Config,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    samples, coverage_rows = [], []
    for _, row in train_events.iterrows():
        df = load_event(row, cfg, features)
        numeric = df.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        coverage = {"event_id": row["event_id"]}
        coverage.update({col: float(numeric[col].notna().mean()) for col in features})
        coverage_rows.append(coverage)
        if len(numeric) > cfg.feature_sampling_rows_per_event:
            idx = rng.choice(len(numeric), cfg.feature_sampling_rows_per_event, replace=False)
            numeric = numeric.iloc[np.sort(idx)]
        samples.append(numeric)
    return pd.concat(samples, ignore_index=True), pd.DataFrame(coverage_rows)


def quality_filter(sampled: pd.DataFrame, coverage: pd.DataFrame, features: list[str], cfg: Config) -> list[str]:
    kept = []
    for col in features:
        series = pd.to_numeric(sampled[col], errors="coerce")
        non_missing = series.dropna()
        if non_missing.empty:
            continue
        missing_rate = float(series.isna().mean())
        event_coverage = float((coverage[col] > 0).mean())
        dominant = float(non_missing.value_counts(normalize=True).iloc[0])
        std = float(non_missing.std(ddof=0))
        if missing_rate > cfg.missing_threshold:
            continue
        if event_coverage < cfg.event_coverage_threshold:
            continue
        if dominant >= cfg.near_constant_threshold:
            continue
        if not np.isfinite(std) or std <= 1e-12:
            continue
        kept.append(col)
    if not kept:
        raise RuntimeError("All features removed by quality filters.")
    return kept


def correlation_prune(sampled: pd.DataFrame, features: list[str], threshold: float) -> list[str]:
    data = sampled[features].apply(pd.to_numeric, errors="coerce")
    data = data.fillna(data.median(numeric_only=True))
    corr = data.corr(method="spearman").abs()
    kept: list[str] = []
    for col in features:
        if not any(pd.notna(corr.loc[col, prev]) and corr.loc[col, prev] > threshold for prev in kept):
            kept.append(col)
    return kept


def slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)[mask]
    y = values[mask]
    x = x - x.mean()
    denom = float(np.sum(x * x))
    return 0.0 if denom <= 0 else float(np.sum(x * (y - y.mean())) / denom)


def build_event_summary(train_events: pd.DataFrame, features: list[str], cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    rows, labels = [], []
    for _, event_row in train_events.iterrows():
        df = load_event(event_row, cfg, features)
        if int(event_row["event_label"]) == 1:
            mask = (df["__timestamp__"] >= event_row["event_start"]) & (df["__timestamp__"] <= event_row["event_end"])
            if mask.sum() >= 3:
                df = df.loc[mask]
        record = {"event_id": event_row["event_id"]}
        for col in features:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                stats = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            else:
                stats = (
                    float(np.mean(finite)), float(np.std(finite)), float(np.min(finite)),
                    float(np.max(finite)), slope(values),
                    float(1.0 - finite.size / max(len(values), 1)),
                )
            for suffix, value in zip(("mean", "std", "min", "max", "slope", "missing"), stats):
                record[f"{col}__{suffix}"] = value
        rows.append(record)
        labels.append(int(event_row["event_label"]))
    matrix = pd.DataFrame(rows).set_index("event_id")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    return matrix, np.asarray(labels, dtype=int)


def rank_features(train_events: pd.DataFrame, features: list[str], cfg: Config, seed: int) -> pd.DataFrame:
    X, y = build_event_summary(train_events, features, cfg)
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    scores = {feature: 0.0 for feature in features}
    for name, importance in zip(X.columns, model.feature_importances_):
        channel = name.rsplit("__", 1)[0]
        if channel in scores:
            scores[channel] += float(importance)
    return pd.DataFrame({"feature": list(scores), "importance": list(scores.values())}).sort_values(
        ["importance", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def select_features(train_events: pd.DataFrame, cfg: Config, seed: int, fold_dir: Path) -> list[str]:
    candidates = discover_candidates(train_events, cfg)
    sampled, coverage = sample_training_rows(train_events, candidates, cfg, seed)
    quality = quality_filter(sampled, coverage, candidates, cfg)
    pruned = correlation_prune(sampled, quality, cfg.correlation_threshold)
    ranking = rank_features(train_events, pruned, cfg, seed)
    selected = ranking.head(cfg.top_k_features)["feature"].tolist()
    pd.DataFrame({"candidate_feature": candidates}).to_csv(fold_dir / "candidate_features.csv", index=False)
    pd.DataFrame({"quality_feature": quality}).to_csv(fold_dir / "quality_filtered_features.csv", index=False)
    pd.DataFrame({"pruned_feature": pruned}).to_csv(fold_dir / "correlation_pruned_features.csv", index=False)
    ranking.to_csv(fold_dir / "feature_ranking.csv", index=False)
    pd.DataFrame({"selected_feature": selected}).to_csv(fold_dir / "selected_features.csv", index=False)
    return selected


@dataclass
class ChannelScaler:
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]

    def transform(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        output = np.empty((len(frame), len(features)), dtype=np.float32)
        for index, col in enumerate(features):
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, self.medians[col])
            values = np.clip((values - self.means[col]) / self.scales[col], -10.0, 10.0)
            output[:, index] = values.astype(np.float32)
        return output


def fit_scaler(train_events: pd.DataFrame, features: list[str], cfg: Config, seed: int) -> ChannelScaler:
    rng = np.random.default_rng(seed)
    collected = {feature: [] for feature in features}
    for _, row in train_events.iterrows():
        df = load_event(row, cfg, features)
        if len(df) > cfg.feature_sampling_rows_per_event:
            idx = rng.choice(len(df), cfg.feature_sampling_rows_per_event, replace=False)
            df = df.iloc[np.sort(idx)]
        for col in features:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                collected[col].append(values)
    medians, means, scales = {}, {}, {}
    for col in features:
        values = np.concatenate(collected[col]) if collected[col] else np.asarray([0.0])
        medians[col] = float(np.median(values))
        means[col] = float(np.mean(values))
        scale_value = float(np.std(values))
        scales[col] = scale_value if np.isfinite(scale_value) and scale_value > 1e-8 else 1.0
    return ChannelScaler(medians, means, scales)


def assign_window_label(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    event_label: int,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    cfg: Config,
) -> tuple[int | None, float]:
    if event_label == 0:
        return 0, 0.0
    overlap_start = max(window_start, event_start)
    overlap_end = min(window_end, event_end)
    overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
    window_seconds = max(1.0, (window_end - window_start).total_seconds())
    ratio = overlap_seconds / window_seconds
    if ratio >= cfg.anomaly_overlap_threshold:
        return 1, float(ratio)
    if overlap_seconds == 0:
        if window_end < event_start:
            distance = (event_start - window_end).total_seconds() / 3600.0
        elif window_start > event_end:
            distance = (window_start - event_end).total_seconds() / 3600.0
        else:
            distance = 0.0
        if distance >= cfg.normal_gap_hours:
            return 0, 0.0
    return None, float(ratio)


def generate_event_windows(
    event_row: pd.Series,
    features: list[str],
    scaler: ChannelScaler,
    cfg: Config,
    training: bool,
):
    """Generate windows inside one Event only.

    Event splitting is completed before this function is called. During training,
    anomalous windows are sampled more densely than normal windows. Validation
    and test Events always use a single label-independent evaluation stride.
    """
    df = load_event(event_row, cfg, features)
    if len(df) < cfg.window_size:
        return [], [], []

    values = scaler.transform(df, features)
    times = df["__timestamp__"].reset_index(drop=True)
    last_start = len(df) - cfg.window_size

    if training:
        if cfg.train_anomaly_step_size <= 0 or cfg.train_normal_step_size <= 0:
            raise ValueError("Training step sizes must be positive.")
        anomaly_starts = set(range(0, last_start + 1, cfg.train_anomaly_step_size))
        normal_starts = set(range(0, last_start + 1, cfg.train_normal_step_size))
        candidate_starts = sorted(anomaly_starts | normal_starts)
    else:
        if cfg.evaluation_step_size <= 0:
            raise ValueError("evaluation_step_size must be positive.")
        anomaly_starts = normal_starts = set()
        candidate_starts = range(0, last_start + 1, cfg.evaluation_step_size)

    windows, labels, records = [], [], []
    for start in candidate_starts:
        end = start + cfg.window_size
        label, overlap = assign_window_label(
            times.iloc[start],
            times.iloc[end - 1],
            int(event_row["event_label"]),
            event_row["event_start"],
            event_row["event_end"],
            cfg,
        )
        if label is None:
            continue

        # Training-only supervised oversampling:
        # anomaly windows follow the denser anomaly stride, while normal windows
        # follow the standard normal stride. Evaluation never uses labels to
        # choose the sampling stride.
        if training:
            if label == 1 and start not in anomaly_starts:
                continue
            if label == 0 and start not in normal_starts:
                continue

        windows.append(values[start:end].astype(np.float32))
        labels.append(label)
        records.append({
            "event_id": str(event_row["event_id"]),
            "event_label": int(event_row["event_label"]),
            "window_start": str(times.iloc[start]),
            "window_end": str(times.iloc[end - 1]),
            "overlap_ratio": float(overlap),
            "window_label": int(label),
        })
    return windows, labels, records


def build_dataset(
    events: pd.DataFrame,
    features: list[str],
    scaler: ChannelScaler,
    cfg: Config,
    training: bool,
):
    windows, labels, records = [], [], []
    for _, row in events.iterrows():
        event_windows, event_labels, event_records = generate_event_windows(
            row, features, scaler, cfg, training=training
        )
        windows.extend(event_windows)
        labels.extend(event_labels)
        records.extend(event_records)

    if not windows:
        raise RuntimeError("No eligible windows generated.")
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        pd.DataFrame(records),
    )


def balance_training_windows(
    X: np.ndarray,
    y: np.ndarray,
    records: pd.DataFrame,
    normal_to_anomaly_ratio: float,
    seed: int,
):
    """Keep all anomaly windows and undersample normal windows by Event.

    Normal-window selection is distributed across source Events as evenly as
    possible so that one long Event does not dominate the balanced training set.
    """
    if normal_to_anomaly_ratio <= 0:
        raise ValueError("train_normal_to_anomaly_ratio must be greater than zero.")

    y_int = y.astype(int)
    anomaly_idx = np.where(y_int == 1)[0]
    normal_idx = np.where(y_int == 0)[0]
    if len(anomaly_idx) == 0 or len(normal_idx) == 0:
        raise RuntimeError(
            f"Training windows require both classes; found normal={len(normal_idx)}, "
            f"anomaly={len(anomaly_idx)}."
        )

    target_normal = min(
        len(normal_idx),
        max(1, int(round(len(anomaly_idx) * normal_to_anomaly_ratio))),
    )
    rng = np.random.default_rng(seed)

    normal_frame = records.iloc[normal_idx].copy()
    normal_frame["__original_index__"] = normal_idx
    event_pools: dict[str, list[int]] = {}
    for event_id, group in normal_frame.groupby("event_id", sort=True):
        pool = group["__original_index__"].astype(int).tolist()
        rng.shuffle(pool)
        event_pools[str(event_id)] = pool

    selected_normal: list[int] = []
    event_ids = list(event_pools)
    rng.shuffle(event_ids)
    while len(selected_normal) < target_normal:
        added = False
        for event_id in event_ids:
            pool = event_pools[event_id]
            if pool and len(selected_normal) < target_normal:
                selected_normal.append(pool.pop())
                added = True
        if not added:
            break

    selected = np.concatenate([
        anomaly_idx,
        np.asarray(selected_normal, dtype=int),
    ])
    rng.shuffle(selected)

    balanced_records = records.iloc[selected].reset_index(drop=True)
    return X[selected], y[selected], balanced_records


def event_balanced_weights(y: np.ndarray, records: pd.DataFrame) -> np.ndarray:
    """Give each source Event approximately equal total training weight.

    Class imbalance is already controlled through explicit window sampling, so
    no additional class-weight multiplier is applied here.
    """
    event_counts = records["event_id"].value_counts().to_dict()
    weights = records["event_id"].map(
        lambda event_id: 1.0 / event_counts[event_id]
    ).to_numpy(dtype=float)
    return (weights / max(weights.mean(), 1e-12)).astype(np.float32)


def _tcn_residual_block(
    x,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    regularizer,
    block_name: str,
):
    """Build one causal dilated residual TCN block."""
    tf = require_tensorflow()
    residual = x

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizer,
        name=f"{block_name}_conv1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{block_name}_relu1")(x)
    x = tf.keras.layers.SpatialDropout1D(
        dropout_rate,
        name=f"{block_name}_drop1",
    )(x)

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizer,
        name=f"{block_name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = tf.keras.layers.Activation("relu", name=f"{block_name}_relu2")(x)
    x = tf.keras.layers.SpatialDropout1D(
        dropout_rate,
        name=f"{block_name}_drop2",
    )(x)

    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=regularizer,
            name=f"{block_name}_residual_projection",
        )(residual)

    x = tf.keras.layers.Add(name=f"{block_name}_add")([x, residual])
    return tf.keras.layers.Activation(
        "relu",
        name=f"{block_name}_out",
    )(x)


def build_model(
    window_size: int,
    n_features: int,
    learning_rate: float,
    tcn_filters: int = 32,
    tcn_kernel_size: int = 3,
    tcn_dropout: float = 0.30,
    tcn_l2: float = 5e-4,
):
    """Build a compact causal dilated TCN for cross-Event generalisation."""
    tf = require_tensorflow()
    regularizer = tf.keras.regularizers.l2(tcn_l2)

    inputs = tf.keras.Input(
        shape=(window_size, n_features),
        name="scada_window",
    )

    x = inputs
    for dilation in (1, 2, 4, 8):
        x = _tcn_residual_block(
            x=x,
            filters=tcn_filters,
            kernel_size=tcn_kernel_size,
            dilation_rate=dilation,
            dropout_rate=tcn_dropout,
            regularizer=regularizer,
            block_name=f"tcn_d{dilation}",
        )

    average_pool = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pool"
    )(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(
        name="global_max_pool"
    )(x)
    x = tf.keras.layers.Concatenate(name="combined_temporal_pool")([
        average_pool,
        max_pool,
    ])

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense_32",
    )(x)
    x = tf.keras.layers.Dropout(0.40, name="dense_dropout")(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="anomaly_probability",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="farm_c_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model

def safe_roc_auc(y_true, y_prob) -> float:
    return float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true, y_prob) -> float:
    return float("nan") if len(np.unique(y_true)) < 2 else float(average_precision_score(y_true, y_prob))


def metrics(y_true, y_prob, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold_by_far(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_far: float,
    grid_size: int,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """Select a threshold on validation data with FAR constrained to target.

    Among thresholds satisfying FAR <= target_far, choose the one with highest
    recall. Ties are resolved by higher precision, then higher F1, then the
    lower threshold. If no threshold satisfies the FAR constraint, choose the
    threshold with the lowest FAR and then highest recall.
    """
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("threshold_target_far must be between 0 and 1.")
    if grid_size < 2:
        raise ValueError("threshold_grid_size must be at least 2.")

    thresholds = np.linspace(0.0, 1.0, grid_size)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        row = metrics(y_true, y_prob, float(threshold))
        row["threshold"] = float(threshold)
        rows.append(row)

    table = pd.DataFrame(rows)
    eligible = table[table["false_alarm_rate"] <= target_far + 1e-12].copy()

    if not eligible.empty:
        chosen = eligible.sort_values(
            ["recall", "precision", "f1", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        selection_rule = "max_recall_subject_to_far"
    else:
        chosen = table.sort_values(
            ["false_alarm_rate", "recall", "precision", "f1", "threshold"],
            ascending=[True, False, False, False, True],
        ).iloc[0]
        selection_rule = "minimum_far_fallback"

    selected_threshold = float(chosen["threshold"])
    selected_metrics = {
        key: (
            int(chosen[key])
            if key in {"count", "tn", "fp", "fn", "tp"}
            else float(chosen[key])
        )
        for key in [
            "count", "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "false_alarm_rate",
            "tn", "fp", "fn", "tp",
        ]
    }
    selected_metrics["threshold"] = selected_threshold
    selected_metrics["target_false_alarm_rate"] = float(target_far)
    selected_metrics["selection_rule"] = selection_rule
    return selected_threshold, selected_metrics, table


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    tf = require_tensorflow()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(cfg)
    meta.to_csv(out / "resolved_metadata.csv", index=False)
    save_json(out / "config.json", asdict(cfg))
    print(meta["event_label"].value_counts().sort_index())

    splitter = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    X_placeholder = np.zeros((len(meta), 1))
    y_event = meta["event_label"].to_numpy(dtype=int)
    groups = meta["event_id"].to_numpy()
    all_windows, all_metrics = [], []

    for fold, (outer_train_idx, test_idx) in enumerate(splitter.split(X_placeholder, y_event, groups), start=1):
        fold_seed = cfg.seed + fold * 1000
        set_seed(fold_seed)
        fold_dir = out / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        outer_train = meta.iloc[outer_train_idx].reset_index(drop=True)
        test_events = meta.iloc[test_idx].reset_index(drop=True)
        train_events, val_events = train_test_split(
            outer_train,
            test_size=0.20,
            random_state=fold_seed,
            stratify=outer_train["event_label"],
        )
        train_events = train_events.reset_index(drop=True)
        val_events = val_events.reset_index(drop=True)
        split_table = pd.concat([
            train_events.assign(split="train"),
            val_events.assign(split="validation"),
            test_events.assign(split="test"),
        ], ignore_index=True)
        split_table.to_csv(fold_dir / "event_split.csv", index=False)
        train_ids, val_ids, test_ids = set(train_events.event_id), set(val_events.event_id), set(test_events.event_id)
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise AssertionError("Event leakage detected.")

        print(f"\n[FOLD {fold}] train={len(train_events)} val={len(val_events)} test={len(test_events)}")
        features = select_features(train_events, cfg, fold_seed, fold_dir)
        scaler = fit_scaler(train_events, features, cfg, fold_seed)
        save_json(fold_dir / "scaler.json", asdict(scaler))
        X_train_raw, y_train_raw, rec_train_raw = build_dataset(
            train_events, features, scaler, cfg, training=True
        )
        X_val, y_val, rec_val = build_dataset(
            val_events, features, scaler, cfg, training=False
        )
        X_test, y_test, rec_test = build_dataset(
            test_events, features, scaler, cfg, training=False
        )

        X_train, y_train, rec_train = balance_training_windows(
            X_train_raw,
            y_train_raw,
            rec_train_raw,
            cfg.train_normal_to_anomaly_ratio,
            fold_seed,
        )

        raw_counts = pd.Series(y_train_raw.astype(int)).value_counts().sort_index().to_dict()
        balanced_counts = pd.Series(y_train.astype(int)).value_counts().sort_index().to_dict()
        val_counts = pd.Series(y_val.astype(int)).value_counts().sort_index().to_dict()
        test_counts = pd.Series(y_test.astype(int)).value_counts().sort_index().to_dict()
        window_counts = {
            "train_before_balance": {str(k): int(v) for k, v in raw_counts.items()},
            "train_after_balance": {str(k): int(v) for k, v in balanced_counts.items()},
            "validation": {str(k): int(v) for k, v in val_counts.items()},
            "test": {str(k): int(v) for k, v in test_counts.items()},
        }
        save_json(fold_dir / "window_class_counts.json", window_counts)
        rec_train_raw.to_csv(fold_dir / "training_windows_before_balance.csv", index=False)
        rec_train.to_csv(fold_dir / "training_windows_after_balance.csv", index=False)

        print(
            f"[WINDOWS] train_raw={len(y_train_raw)} "
            f"train_balanced={len(y_train)} val={len(y_val)} "
            f"test={len(y_test)} features={len(features)}"
        )
        print(f"[CLASS COUNTS] {window_counts}")
        if len(np.unique(y_train.astype(int))) < 2:
            raise RuntimeError(f"Fold {fold} training windows have only one class.")

        sample_weight = event_balanced_weights(y_train, rec_train)
        model = build_model(
            cfg.window_size,
            len(features),
            cfg.learning_rate,
            tcn_filters=cfg.tcn_filters,
            tcn_kernel_size=cfg.tcn_kernel_size,
            tcn_dropout=cfg.tcn_dropout,
            tcn_l2=cfg.tcn_l2,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=cfg.early_stopping_patience,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc", mode="max", factor=0.5,
                patience=max(2, cfg.early_stopping_patience // 3), min_lr=1e-6, verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(fold_dir / "best_model.weights.h5"),
                monitor="val_pr_auc",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
        ]
        history = model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            validation_data=(X_val, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        pd.DataFrame(history.history).to_csv(
            fold_dir / "training_history.csv",
            index=False,
        )

        val_probabilities = model.predict(
            X_val,
            batch_size=cfg.batch_size,
            verbose=0,
        ).reshape(-1)
        selected_threshold, validation_threshold_metrics, threshold_table = (
            select_threshold_by_far(
                y_val,
                val_probabilities,
                cfg.threshold_target_far,
                cfg.threshold_grid_size,
            )
        )

        rec_val = rec_val.copy()
        rec_val["probability"] = val_probabilities
        rec_val["prediction"] = (
            val_probabilities >= selected_threshold
        ).astype(int)
        rec_val["fold"] = fold
        rec_val.to_csv(
            fold_dir / "validation_window_predictions.csv",
            index=False,
        )
        threshold_table.to_csv(
            fold_dir / "validation_threshold_search.csv",
            index=False,
        )
        save_json(
            fold_dir / "selected_threshold.json",
            validation_threshold_metrics,
        )

        probabilities = model.predict(
            X_test,
            batch_size=cfg.batch_size,
            verbose=0,
        ).reshape(-1)

        rec_test = rec_test.copy()
        rec_test["probability"] = probabilities
        rec_test["prediction"] = (
            probabilities >= selected_threshold
        ).astype(int)
        rec_test["selected_threshold"] = selected_threshold
        rec_test["fold"] = fold
        rec_test.to_csv(
            fold_dir / "window_predictions.csv",
            index=False,
        )
        all_windows.append(rec_test)

        window_metrics = metrics(
            y_test,
            probabilities,
            selected_threshold,
        )
        window_metrics.update({
            "fold": fold,
            "level": "window",
            "selected_threshold": selected_threshold,
            "validation_target_far": cfg.threshold_target_far,
        })
        all_metrics.append(window_metrics)

        save_json(
            fold_dir / "metrics.json",
            {
                "validation_threshold_selection": validation_threshold_metrics,
                "window": window_metrics,
            },
        )
        print("[SELECTED THRESHOLD]", selected_threshold)
        print("[WINDOW METRICS]", window_metrics)

        del X_train_raw, y_train_raw, X_train, y_train, X_val, y_val, X_test, y_test, model
        tf.keras.backend.clear_session()
        gc.collect()

    all_window_df = pd.concat(all_windows, ignore_index=True)
    fold_metrics_df = pd.DataFrame(all_metrics)
    all_window_df.to_csv(out / "all_window_predictions.csv", index=False)
    fold_metrics_df.to_csv(out / "fold_metrics.csv", index=False)

    summary_rows = []
    for level, group in fold_metrics_df.groupby("level"):
        for metric_name in [
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "false_alarm_rate",
        ]:
            values = pd.to_numeric(group[metric_name], errors="coerce")
            summary_rows.append({
                "level": level,
                "metric": metric_name,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    pd.DataFrame(summary_rows).to_csv(out / "metrics_summary.csv", index=False)

    overall_true = all_window_df["window_label"].to_numpy(dtype=int)
    overall_pred = all_window_df["prediction"].to_numpy(dtype=int)
    overall_prob = all_window_df["probability"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(
        overall_true,
        overall_pred,
        labels=[0, 1],
    ).ravel()

    final = {
        "overall_window_metrics": {
            "count": int(len(overall_true)),
            "accuracy": float(accuracy_score(overall_true, overall_pred)),
            "precision": float(
                precision_score(overall_true, overall_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(overall_true, overall_pred, zero_division=0)
            ),
            "f1": float(
                f1_score(overall_true, overall_pred, zero_division=0)
            ),
            "roc_auc": safe_roc_auc(overall_true, overall_prob),
            "pr_auc": safe_pr_auc(overall_true, overall_prob),
            "false_alarm_rate": (
                float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "event_count": int(len(meta)),
        "out_of_fold_window_predictions": int(len(all_window_df)),
        "evaluation_unit": "24-hour window",
        "event_aggregation_applied": False,
        "threshold_selection": (
            "validation-based per fold; maximise recall subject to FAR constraint"
        ),
        "threshold_target_far": float(cfg.threshold_target_far),
    }
    save_json(out / "final_window_metrics.json", final)
    print("\n[DONE]")
    print(json.dumps(final, indent=2))
    print(f"[OUTPUT] {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Event-grouped Farm C TCN.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--event-csv-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--event-id-column", default="event_id")
    parser.add_argument("--event-label-column", default="event_label")
    parser.add_argument("--event-start-column", default="event_start")
    parser.add_argument("--event-end-column", default="event_end")
    parser.add_argument("--timestamp-column", default="time_stamp")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=144)
    parser.add_argument("--evaluation-step-size", type=int, default=72)
    parser.add_argument("--train-anomaly-step-size", type=int, default=48)
    parser.add_argument("--train-normal-step-size", type=int, default=72)
    parser.add_argument("--anomaly-overlap-threshold", type=float, default=0.10)
    parser.add_argument("--normal-gap-hours", type=float, default=24.0)
    parser.add_argument("--feature-mode", choices=["avg", "avg_std", "all_numeric"], default="avg_std")
    parser.add_argument("--top-k-features", type=int, default=20)
    parser.add_argument("--missing-threshold", type=float, default=0.30)
    parser.add_argument("--event-coverage-threshold", type=float, default=0.80)
    parser.add_argument("--near-constant-threshold", type=float, default=0.99)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--feature-sampling-rows-per-event", type=int, default=4000)
    parser.add_argument("--train-normal-to-anomaly-ratio", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tcn-filters", type=int, default=32)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--tcn-dropout", type=float, default=0.30)
    parser.add_argument("--tcn-l2", type=float, default=5e-4)
    parser.add_argument("--window-threshold", type=float, default=0.50)
    parser.add_argument("--threshold-target-far", type=float, default=0.05)
    parser.add_argument("--threshold-grid-size", type=int, default=501)
    return parser


def main() -> int:
    cfg = Config(**vars(build_parser().parse_args()))
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
