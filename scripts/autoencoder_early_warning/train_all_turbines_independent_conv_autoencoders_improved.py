#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Improved fleet-wide turbine-specific convolutional Autoencoder framework.

Each event CSV is treated as one turbine/asset time series. The framework is
shared across all turbines, but the following are fitted independently for each
CSV/turbine using only that turbine's earlier normal history:

- feature selection
- missing-value medians
- RobustScaler
- convolutional Autoencoder weights
- reconstruction-error threshold
- sensor-level reconstruction-error baseline

The metadata/manual file may contain both anomaly and normal intervals.

For anomaly rows, the script evaluates:
1. whether a persistent warning occurs before the recorded interval;
2. lead time to the recorded event start;
3. whether the recorded interval itself contains warnings.

For normal rows, the script evaluates:
1. whether the recorded normal interval remains warning-free;
2. warning-window fraction and warning-episode count inside the interval.

The feature-description CSV is used to translate anonymous names such as
sensor_43_std into physical descriptions such as a stator-cooler flow meter.

Important limitation
--------------------
A reconstruction-error warning indicates deviation from learned normal SCADA
behaviour. It does not by itself diagnose the exact component failure described
in the maintenance manual.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass
class Config:
    datasets_dir: str
    metadata_file: str
    feature_description_file: str
    output_dir: str

    timestamp_column: str = "time_stamp"
    asset_id_column: str = "asset_id"
    event_id_column: str = "event_id"
    label_column: str = "event_label"
    start_column: str = "event_start"
    end_column: str = "event_end"
    description_column: str = "event_description"

    sampling_minutes: int = 10
    feature_mode: str = "avg_std"  # avg | avg_std | all_numeric
    top_k_features: int = 30
    max_missing_fraction: float = 0.20
    max_constant_fraction: float = 0.995
    max_abs_correlation: float = 0.98

    window_size: int = 144
    train_step: int = 72
    validation_step: int = 72
    test_step: int = 6
    min_window_valid_fraction: float = 0.98

    anomaly_pre_exclusion_days: float = 14.0
    normal_pre_exclusion_days: float = 1.0
    train_fraction_before_validation: float = 0.80
    split_gap_hours: float = 24.0
    test_lookback_days: float = 30.0
    post_interval_test_days: float = 2.0
    warning_match_days: float = 30.0

    seed: int = 42
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    latent_channels: int = 16
    dropout: float = 0.20
    l2: float = 5e-4
    patience: int = 12

    # Improved calibration: dense normal-validation scoring, stricter global
    # threshold, local-sensor confirmation, and k-of-n temporal persistence.
    threshold_quantile: float = 0.99
    target_validation_far: float = 0.02
    threshold_quantile_grid: str = "0.95,0.975,0.99,0.995,0.9975"
    score_smoothing_windows: int = 3
    persistence_lookback_windows: int = 12
    persistence_min_positive: int = 8
    local_threshold_quantile: float = 0.99
    minimum_local_sensor_count: int = 1
    minimum_episode_hours: float = 6.0
    max_validation_episodes_per_30d: float = 1.0
    merge_gap_hours: float = 2.0

    include_labels: str = "all"  # all | anomaly | normal
    event_ids: str = ""          # comma-separated filter
    overwrite: bool = False


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description=(
            "Train one independent Conv Autoencoder per turbine/event CSV and "
            "evaluate anomaly and normal manual intervals."
        )
    )
    p.add_argument("--datasets-dir", required=True)
    p.add_argument("--metadata-file", required=True)
    p.add_argument("--feature-description-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--timestamp-column", default="time_stamp")
    p.add_argument("--asset-id-column", default="asset_id")
    p.add_argument("--feature-mode", choices=["avg", "avg_std", "all_numeric"], default="avg_std")
    p.add_argument("--top-k-features", type=int, default=30)
    p.add_argument("--anomaly-pre-exclusion-days", type=float, default=14.0)
    p.add_argument("--normal-pre-exclusion-days", type=float, default=1.0)
    p.add_argument("--test-lookback-days", type=float, default=30.0)
    p.add_argument("--post-interval-test-days", type=float, default=2.0)
    p.add_argument("--warning-match-days", type=float, default=30.0)
    p.add_argument("--threshold-quantile", type=float, default=0.99)
    p.add_argument("--target-validation-far", type=float, default=0.02)
    p.add_argument("--threshold-quantile-grid", default="0.95,0.975,0.99,0.995,0.9975")
    p.add_argument("--score-smoothing-windows", type=int, default=3)
    p.add_argument("--persistence-lookback-windows", type=int, default=12)
    p.add_argument("--persistence-min-positive", type=int, default=8)
    p.add_argument("--local-threshold-quantile", type=float, default=0.99)
    p.add_argument("--minimum-local-sensor-count", type=int, default=1)
    p.add_argument("--minimum-episode-hours", type=float, default=6.0)
    p.add_argument("--max-validation-episodes-per-30d", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-labels", choices=["all", "anomaly", "normal"], default="all")
    p.add_argument("--event-ids", default="", help="Optional comma-separated event IDs")
    p.add_argument("--overwrite", action="store_true")
    ns = p.parse_args()
    cfg = Config(
        datasets_dir=ns.datasets_dir,
        metadata_file=ns.metadata_file,
        feature_description_file=ns.feature_description_file,
        output_dir=ns.output_dir,
    )
    for key, value in vars(ns).items():
        if hasattr(cfg, key) and key not in {
            "datasets_dir", "metadata_file", "feature_description_file", "output_dir"
        }:
            setattr(cfg, key, value)
    return cfg


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
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
            "TensorFlow is required. Install with: python -m pip install tensorflow"
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


def find_column(columns: Iterable[str], requested: str, alternatives: Sequence[str]) -> str:
    mapping = {norm_name(c): c for c in columns}
    for candidate in [requested, *alternatives]:
        key = norm_name(candidate)
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Cannot find {requested!r}. Available columns: {list(columns)[:30]}")


NON_SENSOR = {
    "time_stamp", "timestamp", "time", "datetime", "date_time", "asset_id",
    "event_id", "id", "train_test", "status_type_id", "label", "target",
    "event_label", "event_start", "event_end", "description",
}
LEAKAGE_TOKENS = {"fault", "failure", "anomaly", "label", "target", "prediction", "event_"}


def feature_allowed(name: str, mode: str) -> bool:
    n = norm_name(name)
    if n in NON_SENSOR or any(tok in n for tok in LEAKAGE_TOKENS):
        return False
    if mode == "avg":
        return n.endswith("_avg")
    if mode == "avg_std":
        return n.endswith("_avg") or n.endswith("_std")
    return True


def canonical_label(value: object) -> str:
    s = norm_name(str(value))
    if s in {"anomaly", "abnormal", "fault", "failure", "1", "true"}:
        return "anomaly"
    if s in {"normal", "healthy", "0", "false"}:
        return "normal"
    return s


def load_metadata(cfg: Config) -> pd.DataFrame:
    meta = read_csv_auto(Path(cfg.metadata_file))
    id_col = find_column(meta.columns, cfg.event_id_column, ["id", "source_id"])
    label_col = find_column(meta.columns, cfg.label_column, ["label", "event_type"])
    start_col = find_column(meta.columns, cfg.start_column, ["start", "start_time"])
    end_col = find_column(meta.columns, cfg.end_column, ["end", "end_time"])
    try:
        desc_col = find_column(meta.columns, cfg.description_column, ["description", "event_desc"])
    except KeyError:
        desc_col = None

    out = pd.DataFrame({
        "event_id": meta[id_col].astype(str).str.strip(),
        "label": meta[label_col].map(canonical_label),
        "interval_start": pd.to_datetime(meta[start_col], errors="coerce"),
        "interval_end": pd.to_datetime(meta[end_col], errors="coerce"),
        "event_description": "" if desc_col is None else meta[desc_col].fillna("").astype(str),
    })
    out = out.dropna(subset=["interval_start", "interval_end"])
    out = out[out["interval_end"] > out["interval_start"]].copy()
    out = out.drop_duplicates(subset=["event_id"], keep="last").reset_index(drop=True)

    if cfg.include_labels != "all":
        out = out[out["label"] == cfg.include_labels]
    if cfg.event_ids.strip():
        wanted = {x.strip() for x in cfg.event_ids.split(",") if x.strip()}
        out = out[out["event_id"].isin(wanted)]
    return out.reset_index(drop=True)


def load_feature_description(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    cols = {norm_name(c): c for c in df.columns}
    sensor_col = cols.get("sensor_name") or cols.get("feature") or cols.get("name")
    desc_col = cols.get("description") or cols.get("feature_description")
    unit_col = cols.get("unit")
    stats_col = cols.get("statistics_type") or cols.get("statistics")
    if sensor_col is None or desc_col is None:
        raise ValueError(
            "Feature-description file must contain sensor_name and description columns."
        )
    out = pd.DataFrame({
        "base_signal": df[sensor_col].astype(str).str.strip(),
        "description": df[desc_col].fillna("").astype(str).str.strip(),
        "unit": "" if unit_col is None else df[unit_col].fillna("").astype(str).str.strip(),
        "statistics_type": "" if stats_col is None else df[stats_col].fillna("").astype(str).str.strip(),
    })
    out["base_signal_norm"] = out["base_signal"].map(norm_name)
    return out.drop_duplicates("base_signal_norm", keep="first")


def parse_measurement_name(feature: str) -> tuple[str, str]:
    n = norm_name(feature)
    suffix_map = {
        "_avg": "average",
        "_std": "std_dev",
        "_min": "minimum",
        "_max": "maximum",
    }
    for suffix, stat in suffix_map.items():
        if n.endswith(suffix):
            return n[: -len(suffix)], stat
    return n, "raw"


def enrich_features(features: Sequence[str], feature_desc: pd.DataFrame) -> pd.DataFrame:
    lookup = feature_desc.set_index("base_signal_norm").to_dict("index")
    rows = []
    for feature in features:
        base, statistic = parse_measurement_name(feature)
        info = lookup.get(base, {})
        rows.append({
            "feature": feature,
            "base_signal": base,
            "statistic": statistic,
            "physical_description": info.get("description", "UNKNOWN"),
            "unit": info.get("unit", ""),
            "available_statistics": info.get("statistics_type", ""),
        })
    return pd.DataFrame(rows)


def load_and_prepare(csv_path: Path, cfg: Config) -> tuple[pd.DataFrame, str | None]:
    raw = read_csv_auto(csv_path)
    time_col = find_column(raw.columns, cfg.timestamp_column, ["timestamp", "datetime", "date_time", "time"])
    try:
        asset_col = find_column(raw.columns, cfg.asset_id_column, ["asset", "turbine_id", "wt_id"])
    except KeyError:
        asset_col = None

    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).sort_values(time_col)
    raw = raw.drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)

    asset_id = None
    if asset_col is not None:
        assets = raw[asset_col].dropna().astype(str).unique().tolist()
        if len(assets) > 1:
            raise ValueError(f"Expected one asset in {csv_path.name}, found {assets[:10]}")
        asset_id = assets[0] if assets else None

    raw = raw.rename(columns={time_col: "__timestamp__"})
    return raw, asset_id


def split_contiguous_segments(df: pd.DataFrame, sampling_minutes: int) -> list[pd.DataFrame]:
    expected = pd.Timedelta(minutes=sampling_minutes)
    breaks = df["__timestamp__"].diff().fillna(expected) > expected * 1.5
    segment_id = breaks.cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(segment_id) if len(g)]


def robust_variability(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 10:
        return 0.0
    q25, q75 = np.nanpercentile(x, [25, 75])
    iqr = float(q75 - q25)
    mad = float(np.nanmedian(np.abs(x - np.nanmedian(x))))
    return max(iqr, 1.4826 * mad)


def select_features(train_rows: pd.DataFrame, cfg: Config) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    candidates: list[str] = []
    for col in train_rows.columns:
        if col == "__timestamp__" or not feature_allowed(col, cfg.feature_mode):
            continue
        numeric = pd.to_numeric(train_rows[col], errors="coerce")
        if numeric.notna().sum() < max(100, cfg.window_size):
            continue
        missing = float(numeric.isna().mean())
        if missing > cfg.max_missing_fraction:
            continue
        vc = numeric.dropna().value_counts(normalize=True)
        constant_fraction = float(vc.iloc[0]) if len(vc) else 1.0
        if constant_fraction >= cfg.max_constant_fraction:
            continue
        variability = robust_variability(numeric)
        if not np.isfinite(variability) or variability <= 0:
            continue
        candidates.append(col)
        rows.append({
            "feature": col,
            "missing_fraction": missing,
            "largest_value_fraction": constant_fraction,
            "robust_variability": variability,
        })
    if not candidates:
        raise ValueError("No usable numeric sensor features found")

    ranking = pd.DataFrame(rows).sort_values(
        ["robust_variability", "missing_fraction"], ascending=[False, True]
    ).reset_index(drop=True)

    sample = train_rows[candidates].apply(pd.to_numeric, errors="coerce")
    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=cfg.seed)
    sample = sample.interpolate(limit_direction="both").fillna(sample.median())
    corr = sample.corr(method="spearman").abs()

    selected: list[str] = []
    for col in ranking["feature"]:
        if all(float(corr.loc[col, kept]) < cfg.max_abs_correlation for kept in selected):
            selected.append(col)
        if len(selected) >= cfg.top_k_features:
            break
    if len(selected) < 3:
        raise ValueError(f"Only {len(selected)} features survived selection")
    ranking["selected"] = ranking["feature"].isin(selected)
    return selected, ranking


def fit_scaler(train_rows: pd.DataFrame, features: list[str]) -> tuple[RobustScaler, pd.Series]:
    x = train_rows[features].apply(pd.to_numeric, errors="coerce")
    medians = x.median()
    x = x.interpolate(limit_direction="both").fillna(medians)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(x.to_numpy(dtype=np.float32))
    return scaler, medians


def transform_segment(seg, features, scaler, medians):
    x = seg[features].apply(pd.to_numeric, errors="coerce")
    valid_fraction = x.notna().mean(axis=1).to_numpy(dtype=float)
    x = x.interpolate(limit_direction="both").fillna(medians)
    arr = scaler.transform(x.to_numpy(dtype=np.float32)).astype(np.float32)
    return arr, valid_fraction


def make_windows(df, features, scaler, medians, cfg, start, end, step):
    subset = df[(df["__timestamp__"] >= start) & (df["__timestamp__"] <= end)].copy()
    windows, meta = [], []
    for seg in split_contiguous_segments(subset, cfg.sampling_minutes):
        if len(seg) < cfg.window_size:
            continue
        arr, row_valid = transform_segment(seg, features, scaler, medians)
        for i in range(0, len(seg) - cfg.window_size + 1, step):
            j = i + cfg.window_size
            if float(np.mean(row_valid[i:j])) < cfg.min_window_valid_fraction:
                continue
            windows.append(arr[i:j])
            meta.append({
                "window_start": seg["__timestamp__"].iloc[i],
                "window_end": seg["__timestamp__"].iloc[j - 1],
            })
    if not windows:
        return np.empty((0, cfg.window_size, len(features)), dtype=np.float32), pd.DataFrame(meta)
    return np.stack(windows).astype(np.float32), pd.DataFrame(meta)


def build_model(cfg: Config, n_features: int):
    tf = require_tensorflow()
    reg = tf.keras.regularizers.l2(cfg.l2)
    inp = tf.keras.Input(shape=(cfg.window_size, n_features), name="scada_window")
    x = tf.keras.layers.Conv1D(32, 5, padding="same", kernel_regularizer=reg)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2, padding="same")(x)
    x = tf.keras.layers.SpatialDropout1D(cfg.dropout)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2, padding="same")(x)
    x = tf.keras.layers.SpatialDropout1D(cfg.dropout)(x)
    z = tf.keras.layers.Conv1D(cfg.latent_channels, 3, padding="same", activation="relu", name="latent")(x)
    x = tf.keras.layers.UpSampling1D(2)(z)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu", kernel_regularizer=reg)(x)
    out = tf.keras.layers.Conv1D(n_features, 1, padding="same", activation="linear", name="reconstruction")(x)
    model = tf.keras.Model(inp, out, name="improved_turbine_specific_conv_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss="mae")
    return model


def reconstruction_scores(model, x: np.ndarray, batch_size: int):
    if len(x) == 0:
        return np.empty(0), np.empty((0, x.shape[-1])), np.empty((0, x.shape[1]))
    pred = model.predict(x, batch_size=batch_size, verbose=0)
    abs_err = np.abs(x - pred)
    return abs_err.mean(axis=(1, 2)), abs_err.mean(axis=1), abs_err.mean(axis=2)


def rolling_median(values: np.ndarray, windows: int) -> np.ndarray:
    """Causal rolling median used to reduce isolated reconstruction spikes."""
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rolling(max(1, int(windows)), min_periods=1).median().to_numpy(dtype=float)


def apply_k_of_n(flags: np.ndarray, lookback: int, minimum_positive: int) -> np.ndarray:
    """Confirm a warning when at least k of the latest n dense scores are positive.

    Unlike three consecutive highly overlapping windows, k-of-n requires broader
    temporal support and is less sensitive to one short unusual operating episode.
    The implementation is causal: only current and previous scores are used.
    """
    flags = np.asarray(flags, dtype=bool)
    n = max(1, int(lookback))
    k = min(max(1, int(minimum_positive)), n)
    counts = pd.Series(flags.astype(int)).rolling(n, min_periods=n).sum().to_numpy()
    return np.nan_to_num(counts, nan=0.0) >= k


def count_boolean_episodes(times: pd.Series, flags: np.ndarray, cfg: Config) -> int:
    frame = pd.DataFrame({"window_end": pd.to_datetime(times), "flag": np.asarray(flags, bool)})
    active = frame[frame["flag"]].sort_values("window_end")
    if active.empty:
        return 0
    step_hours = cfg.test_step * cfg.sampling_minutes / 60.0
    max_gap = pd.Timedelta(hours=cfg.merge_gap_hours + step_hours)
    starts = 1
    prev = active["window_end"].iloc[0]
    for t in active["window_end"].iloc[1:]:
        if t - prev > max_gap:
            starts += 1
        prev = t
    return starts


def calibrate_global_threshold(
    validation_scores: np.ndarray,
    validation_sensor_scores: np.ndarray,
    validation_times: pd.Series,
    sensor_thresholds: np.ndarray,
    cfg: Config,
) -> dict[str, float]:
    """Choose the least strict candidate satisfying validation false-alert targets.

    Calibration uses dense validation windows with the same step as testing. A
    candidate warning requires a smoothed global exceedance, local-sensor
    confirmation, and k-of-n persistence. This aligns threshold calibration with
    the final online decision rule instead of calibrating only single windows.
    """
    candidates = []
    for token in str(cfg.threshold_quantile_grid).split(","):
        try:
            candidates.append(float(token.strip()))
        except ValueError:
            pass
    candidates.extend([cfg.threshold_quantile, 1.0 - cfg.target_validation_far])
    candidates = sorted({min(max(q, 0.50), 0.9999) for q in candidates})
    smoothed = rolling_median(validation_scores, cfg.score_smoothing_windows)
    local_count = (validation_sensor_scores > sensor_thresholds[None, :]).sum(axis=1)
    duration_days = max(
        1e-9,
        (pd.to_datetime(validation_times).max() - pd.to_datetime(validation_times).min()).total_seconds() / 86400.0,
    )
    rows = []
    chosen = None
    for q in candidates:
        threshold = float(np.quantile(validation_scores, q))
        candidate = (smoothed > threshold) & (local_count >= cfg.minimum_local_sensor_count)
        confirmed = apply_k_of_n(
            candidate, cfg.persistence_lookback_windows, cfg.persistence_min_positive
        )
        far = float(np.mean(confirmed))
        episodes = count_boolean_episodes(validation_times, confirmed, cfg)
        episodes_per_30d = float(episodes * 30.0 / duration_days)
        row = {
            "quantile": q,
            "threshold": threshold,
            "confirmed_window_far": far,
            "warning_episodes": float(episodes),
            "warning_episodes_per_30d": episodes_per_30d,
        }
        rows.append(row)
        if (
            chosen is None
            and far <= cfg.target_validation_far
            and episodes_per_30d <= cfg.max_validation_episodes_per_30d
        ):
            chosen = row
    if chosen is None:
        chosen = rows[-1]
    chosen["calibration_candidates"] = rows
    return chosen


def _episode_summary(rows: list[pd.Series]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    return {
        "episode_start": pd.to_datetime(frame["window_end"]).min(),
        "episode_end": pd.to_datetime(frame["window_end"]).max(),
        "n_windows": int(len(frame)),
        "max_score": float(frame["reconstruction_error"].max()),
        "max_score_ratio": float(frame["score_ratio"].max()),
        "dominant_top_sensor": frame["top_sensor"].mode().iloc[0] if len(frame) else "",
        "dominant_sensor_description": frame["top_sensor_description"].mode().iloc[0] if len(frame) else "",
    }


def build_episodes(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    active = scored[scored["confirmed_warning"]].sort_values("window_end").copy()
    if active.empty:
        return pd.DataFrame(columns=[
            "episode_start", "episode_end", "n_windows", "max_score", "max_score_ratio",
            "dominant_top_sensor", "dominant_sensor_description"
        ])
    max_gap = pd.Timedelta(hours=cfg.merge_gap_hours + cfg.test_step * cfg.sampling_minutes / 60.0)
    episodes, current_rows, last_t = [], [], None
    for _, row in active.iterrows():
        t = pd.Timestamp(row["window_end"])
        if last_t is None or t - last_t <= max_gap:
            current_rows.append(row)
        else:
            episodes.append(_episode_summary(current_rows))
            current_rows = [row]
        last_t = t
    if current_rows:
        episodes.append(_episode_summary(current_rows))
    return pd.DataFrame(episodes)


def save_timeline(scored, threshold, interval_start, interval_end, label, event_id, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(scored["window_end"], scored["reconstruction_error"], linewidth=0.8, alpha=0.45, label="Raw reconstruction error")
    if "smoothed_reconstruction_error" in scored:
        ax.plot(scored["window_end"], scored["smoothed_reconstruction_error"], linewidth=1.4, label="Smoothed reconstruction error")
    ax.axhline(threshold, linestyle="--", linewidth=1.2, label="Validation threshold")
    shade_label = "Recorded anomaly interval" if label == "anomaly" else "Recorded normal interval"
    ax.axvspan(interval_start, interval_end, alpha=0.18, label=shade_label)
    warned = scored[scored["confirmed_warning"]]
    if not warned.empty:
        ax.scatter(warned["window_end"], warned["reconstruction_error"], s=18, label="Confirmed warning")
    ax.set_title(f"Event {event_id}: turbine-specific Conv Autoencoder timeline ({label})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean absolute reconstruction error")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def process_event(row: pd.Series, cfg: Config, feature_desc: pd.DataFrame) -> dict[str, object]:
    event_id = str(row["event_id"])
    label = str(row["label"])
    interval_start = pd.Timestamp(row["interval_start"])
    interval_end = pd.Timestamp(row["interval_end"])
    description = str(row["event_description"])
    csv_path = Path(cfg.datasets_dir) / f"{event_id}.csv"
    event_out = Path(cfg.output_dir) / f"event_{event_id}_{label}"
    event_out.mkdir(parents=True, exist_ok=True)

    summary_path = event_out / "summary.json"
    if summary_path.exists() and not cfg.overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    df, asset_id = load_and_prepare(csv_path, cfg)
    exclusion_days = (
        cfg.anomaly_pre_exclusion_days if label == "anomaly" else cfg.normal_pre_exclusion_days
    )
    normal_cutoff = interval_start - pd.Timedelta(days=exclusion_days)
    pre_cut = df[df["__timestamp__"] < normal_cutoff].copy()
    if len(pre_cut) < cfg.window_size * 10:
        raise ValueError(
            f"Insufficient history before cutoff {normal_cutoff}; rows={len(pre_cut)}"
        )

    earliest, latest = pre_cut["__timestamp__"].min(), pre_cut["__timestamp__"].max()
    raw_split = earliest + (latest - earliest) * cfg.train_fraction_before_validation
    gap = pd.Timedelta(hours=cfg.split_gap_hours)
    train_end = raw_split - gap / 2
    val_start = raw_split + gap / 2
    val_end = normal_cutoff

    train_rows = df[(df["__timestamp__"] >= earliest) & (df["__timestamp__"] <= train_end)].copy()
    val_rows = df[(df["__timestamp__"] >= val_start) & (df["__timestamp__"] < val_end)].copy()
    if len(train_rows) < cfg.window_size * 5 or len(val_rows) < cfg.window_size * 2:
        raise ValueError("Train/validation periods too short after chronological split")

    features, ranking = select_features(train_rows, cfg)
    mapping = enrich_features(features, feature_desc)
    mapping_lookup = mapping.set_index("feature")["physical_description"].to_dict()
    ranking.merge(mapping, on="feature", how="left").to_csv(event_out / "feature_ranking_with_descriptions.csv", index=False)
    mapping.to_csv(event_out / "selected_features_with_descriptions.csv", index=False)

    scaler, medians = fit_scaler(train_rows, features)
    (event_out / "scaler.json").write_text(json.dumps({
        "features": features,
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
        "training_medians": medians.to_dict(),
    }, indent=2, default=float), encoding="utf-8")

    x_train, _ = make_windows(df, features, scaler, medians, cfg, earliest, train_end, cfg.train_step)
    x_val, m_val = make_windows(df, features, scaler, medians, cfg, val_start, val_end, cfg.validation_step)
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError(f"No windows generated: train={len(x_train)}, val={len(x_val)}")

    tf = require_tensorflow()
    tf.keras.backend.clear_session()
    set_seed(cfg.seed + int(re.sub(r"\D", "", event_id) or 0))
    model = build_model(cfg, len(features))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience, restore_best_weights=True, min_delta=1e-5
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, cfg.patience // 3), min_lr=1e-6
        ),
    ]
    history = model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(event_out / "conv_autoencoder.keras")
    pd.DataFrame(history.history).to_csv(event_out / "training_history.csv", index=False)

    # Rebuild validation windows at the dense test cadence so calibration uses
    # the same highly overlapping score stream that will be used online.
    x_val_dense, m_val_dense = make_windows(
        df, features, scaler, medians, cfg, val_start, val_end, cfg.test_step
    )
    if len(x_val_dense) == 0:
        raise ValueError("No dense validation windows generated")
    val_global, val_sensor, _ = reconstruction_scores(model, x_val_dense, cfg.batch_size)
    sensor_thresholds = np.quantile(
        val_sensor, min(max(cfg.local_threshold_quantile, 0.50), 0.9999), axis=0
    )
    calibration = calibrate_global_threshold(
        val_global, val_sensor, m_val_dense["window_end"], sensor_thresholds, cfg
    )
    threshold = float(calibration["threshold"])
    val_smoothed = rolling_median(val_global, cfg.score_smoothing_windows)
    val_local_count = (val_sensor > sensor_thresholds[None, :]).sum(axis=1)
    val_candidate = (val_smoothed > threshold) & (
        val_local_count >= cfg.minimum_local_sensor_count
    )
    val_confirmed = apply_k_of_n(
        val_candidate, cfg.persistence_lookback_windows, cfg.persistence_min_positive
    )
    val_far = float(np.mean(val_confirmed))
    val_scores = m_val_dense.copy()
    val_scores["reconstruction_error"] = val_global
    val_scores["smoothed_reconstruction_error"] = val_smoothed
    val_scores["threshold"] = threshold
    val_scores["global_above_threshold"] = val_smoothed > threshold
    val_scores["local_sensor_exceedance_count"] = val_local_count
    val_scores["candidate_warning"] = val_candidate
    val_scores["confirmed_warning"] = val_confirmed
    val_scores.to_csv(event_out / "normal_validation_scores.csv", index=False)
    pd.DataFrame({
        "feature": features,
        "local_reconstruction_threshold": sensor_thresholds,
        "physical_description": [mapping_lookup.get(x, "UNKNOWN") for x in features],
    }).to_csv(event_out / "local_sensor_thresholds.csv", index=False)
    pd.DataFrame(calibration["calibration_candidates"]).to_csv(
        event_out / "threshold_calibration_candidates.csv", index=False
    )

    test_start = max(df["__timestamp__"].min(), interval_start - pd.Timedelta(days=cfg.test_lookback_days))
    test_end = min(df["__timestamp__"].max(), interval_end + pd.Timedelta(days=cfg.post_interval_test_days))
    x_test, m_test = make_windows(df, features, scaler, medians, cfg, test_start, test_end, cfg.test_step)
    if len(x_test) == 0:
        raise ValueError("No test windows generated")

    global_score, sensor_score, time_score = reconstruction_scores(model, x_test, cfg.batch_size)
    top_idx = np.argmax(sensor_score, axis=1)
    top_sensor = [features[i] for i in top_idx]
    top_sensor_error = sensor_score[np.arange(len(sensor_score)), top_idx]

    scored = m_test.copy()
    scored["event_id"] = event_id
    scored["label"] = label
    scored["interval_start"] = interval_start
    scored["interval_end"] = interval_end
    scored["reconstruction_error"] = global_score
    scored["smoothed_reconstruction_error"] = rolling_median(
        global_score, cfg.score_smoothing_windows
    )
    scored["threshold"] = threshold
    scored["score_ratio"] = scored["smoothed_reconstruction_error"] / max(threshold, 1e-12)
    scored["global_above_threshold"] = scored["smoothed_reconstruction_error"] > threshold
    local_exceed = sensor_score > sensor_thresholds[None, :]
    scored["local_sensor_exceedance_count"] = local_exceed.sum(axis=1)
    scored["candidate_warning"] = scored["global_above_threshold"] & (
        scored["local_sensor_exceedance_count"] >= cfg.minimum_local_sensor_count
    )
    scored["confirmed_warning"] = apply_k_of_n(
        scored["candidate_warning"].to_numpy(),
        cfg.persistence_lookback_windows,
        cfg.persistence_min_positive,
    )
    scored["top_sensor"] = top_sensor
    scored["top_sensor_description"] = [mapping_lookup.get(x, "UNKNOWN") for x in top_sensor]
    scored["top_sensor_error"] = top_sensor_error
    scored["max_timepoint_error"] = time_score.max(axis=1)
    scored["phase"] = np.select(
        [
            scored["window_end"] < interval_start,
            scored["window_end"].between(interval_start, interval_end, inclusive="both"),
            scored["window_end"] > interval_end,
        ],
        ["pre_interval", "recorded_interval", "post_interval"],
        default="unknown",
    )
    scored.to_csv(event_out / "window_scores.csv", index=False)

    sensor_df = pd.DataFrame(sensor_score, columns=features)
    sensor_df.insert(0, "window_end", scored["window_end"].to_numpy())
    sensor_df.to_csv(event_out / "sensor_reconstruction_errors.csv", index=False)

    episodes = build_episodes(scored, cfg)
    if not episodes.empty:
        episodes["duration_hours"] = (
            pd.to_datetime(episodes["episode_end"]) - pd.to_datetime(episodes["episode_start"])
        ).dt.total_seconds() / 3600.0 + cfg.test_step * cfg.sampling_minutes / 60.0
        episodes = episodes[episodes["duration_hours"] >= cfg.minimum_episode_hours].reset_index(drop=True)
        # A window is only a final warning when it belongs to a retained episode.
        scored["confirmed_warning"] = False
        for _, ep in episodes.iterrows():
            mask = scored["window_end"].between(ep["episode_start"], ep["episode_end"], inclusive="both")
            scored.loc[mask, "confirmed_warning"] = True
        scored.to_csv(event_out / "window_scores.csv", index=False)
        episodes["episode_phase"] = np.select(
            [
                pd.to_datetime(episodes["episode_start"]) < interval_start,
                pd.to_datetime(episodes["episode_start"]).between(interval_start, interval_end, inclusive="both"),
                pd.to_datetime(episodes["episode_start"]) > interval_end,
            ],
            ["pre_interval", "recorded_interval", "post_interval"],
            default="unknown",
        )
        episodes["lead_time_hours"] = np.where(
            pd.to_datetime(episodes["episode_start"]) < interval_start,
            (interval_start - pd.to_datetime(episodes["episode_start"])).dt.total_seconds() / 3600.0,
            np.nan,
        )
    episodes.to_csv(event_out / "warning_episodes.csv", index=False)

    match_start = interval_start - pd.Timedelta(days=cfg.warning_match_days)
    pre_eligible = episodes[
        (pd.to_datetime(episodes["episode_start"]) >= match_start)
        & (pd.to_datetime(episodes["episode_start"]) < interval_start)
    ] if not episodes.empty else episodes
    first_pre_warning = pd.to_datetime(pre_eligible["episode_start"]).min() if not pre_eligible.empty else pd.NaT
    lead_hours = (
        float((interval_start - first_pre_warning).total_seconds() / 3600.0)
        if not pd.isna(first_pre_warning) else math.nan
    )

    in_interval = scored[scored["window_end"].between(interval_start, interval_end, inclusive="both")]
    pre_interval = scored[(scored["window_end"] >= match_start) & (scored["window_end"] < interval_start)]
    interval_episodes = episodes[episodes.get("episode_phase", pd.Series(dtype=str)) == "recorded_interval"] if not episodes.empty else episodes

    anomaly_early_warning_detected = bool(label == "anomaly" and not pre_eligible.empty)
    early_warning_by_horizon = {}
    for days in (7, 14, 30):
        h_start = interval_start - pd.Timedelta(days=days)
        h_eps = episodes[
            (pd.to_datetime(episodes["episode_start"]) >= h_start)
            & (pd.to_datetime(episodes["episode_start"]) < interval_start)
        ] if not episodes.empty else episodes
        early_warning_by_horizon[f"anomaly_early_warning_{days}d"] = bool(
            label == "anomaly" and not h_eps.empty
        )
    anomaly_interval_detected = bool(label == "anomaly" and in_interval["confirmed_warning"].any())
    normal_interval_clean = bool(label == "normal" and not in_interval["confirmed_warning"].any())

    dominant_interval_sensor = ""
    dominant_interval_description = ""
    if len(in_interval) and in_interval["confirmed_warning"].any():
        warned_interval = in_interval[in_interval["confirmed_warning"]]
        dominant_interval_sensor = warned_interval["top_sensor"].mode().iloc[0]
        dominant_interval_description = mapping_lookup.get(dominant_interval_sensor, "UNKNOWN")

    summary = {
        "event_id": event_id,
        "label": label,
        "event_description": description,
        "asset_id": asset_id,
        "asset_csv": str(csv_path.resolve()),
        "data_start": str(df["__timestamp__"].min()),
        "data_end": str(df["__timestamp__"].max()),
        "interval_start": str(interval_start),
        "interval_end": str(interval_end),
        "normal_training_end": str(train_end),
        "normal_validation_start": str(val_start),
        "normal_validation_end": str(val_end),
        "pre_exclusion_days": exclusion_days,
        "selected_feature_count": len(features),
        "training_windows": int(len(x_train)),
        "validation_windows": int(len(x_val)),
        "test_windows": int(len(x_test)),
        "threshold": threshold,
        "threshold_quantile_selected": float(calibration["quantile"]),
        "validation_false_alarm_rate": val_far,
        "validation_warning_episodes_per_30d": float(calibration["warning_episodes_per_30d"]),
        "score_smoothing_windows": cfg.score_smoothing_windows,
        "persistence_lookback_windows": cfg.persistence_lookback_windows,
        "persistence_min_positive": cfg.persistence_min_positive,
        "local_threshold_quantile": cfg.local_threshold_quantile,
        "minimum_local_sensor_count": cfg.minimum_local_sensor_count,
        "minimum_episode_hours": cfg.minimum_episode_hours,
        "pre_interval_warning_detected": bool(not pre_eligible.empty),
        "first_pre_interval_warning": None if pd.isna(first_pre_warning) else str(first_pre_warning),
        "lead_time_hours": None if not np.isfinite(lead_hours) else lead_hours,
        "lead_time_days": None if not np.isfinite(lead_hours) else lead_hours / 24.0,
        "pre_interval_window_warning_fraction": float(pre_interval["confirmed_warning"].mean()) if len(pre_interval) else None,
        "recorded_interval_window_warning_fraction": float(in_interval["confirmed_warning"].mean()) if len(in_interval) else None,
        "recorded_interval_warning_episode_count": int(len(interval_episodes)),
        "anomaly_early_warning_detected": anomaly_early_warning_detected,
        **early_warning_by_horizon,
        "anomaly_interval_detected": anomaly_interval_detected,
        "normal_interval_clean": normal_interval_clean,
        "dominant_interval_sensor": dominant_interval_sensor,
        "dominant_interval_sensor_description": dominant_interval_description,
        "total_warning_episode_count": int(len(episodes)),
        "interpretation": (
            "Warnings indicate deviation from this turbine's learned normal SCADA behaviour; "
            "they do not by themselves diagnose the exact maintenance-manual failure."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_timeline(
        scored, threshold, interval_start, interval_end, label, event_id,
        event_out / "autoencoder_timeline.png"
    )
    return summary


def build_aggregate_report(results: pd.DataFrame, out_dir: Path) -> None:
    results.to_csv(out_dir / "all_turbines_event_results.csv", index=False)

    anomaly = results[results["label"] == "anomaly"].copy()
    normal = results[results["label"] == "normal"].copy()

    metrics = {
        "processed_events": int(len(results)),
        "anomaly_events": int(len(anomaly)),
        "normal_events": int(len(normal)),
        "anomaly_early_warning_recall": (
            float(anomaly["anomaly_early_warning_detected"].mean()) if len(anomaly) else None
        ),
        "anomaly_early_warning_recall_7d": (
            float(anomaly["anomaly_early_warning_7d"].mean()) if len(anomaly) else None
        ),
        "anomaly_early_warning_recall_14d": (
            float(anomaly["anomaly_early_warning_14d"].mean()) if len(anomaly) else None
        ),
        "anomaly_early_warning_recall_30d": (
            float(anomaly["anomaly_early_warning_30d"].mean()) if len(anomaly) else None
        ),
        "anomaly_interval_detection_recall": (
            float(anomaly["anomaly_interval_detected"].mean()) if len(anomaly) else None
        ),
        "normal_interval_specificity": (
            float(normal["normal_interval_clean"].mean()) if len(normal) else None
        ),
        "normal_interval_false_alarm_event_rate": (
            float((~normal["normal_interval_clean"]).mean()) if len(normal) else None
        ),
        "median_lead_time_days_detected_anomalies": (
            float(anomaly.loc[anomaly["anomaly_early_warning_detected"], "lead_time_days"].median())
            if len(anomaly) and anomaly["anomaly_early_warning_detected"].any() else None
        ),
        "mean_validation_false_alarm_rate": (
            float(results["validation_false_alarm_rate"].mean()) if len(results) else None
        ),
        "mean_validation_warning_episodes_per_30d": (
            float(results["validation_warning_episodes_per_30d"].mean()) if len(results) else None
        ),
    }
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Improved Turbine-specific Convolutional Autoencoder Results",
        "",
        f"Processed events: {metrics['processed_events']}",
        f"Anomaly events: {metrics['anomaly_events']}",
        f"Normal events: {metrics['normal_events']}",
        "",
        "## Aggregate metrics",
        "",
        f"- Anomaly early-warning recall: {metrics['anomaly_early_warning_recall']}",
        f"- Anomaly early-warning recall (7 days): {metrics['anomaly_early_warning_recall_7d']}",
        f"- Anomaly early-warning recall (14 days): {metrics['anomaly_early_warning_recall_14d']}",
        f"- Anomaly early-warning recall (30 days): {metrics['anomaly_early_warning_recall_30d']}",
        f"- Anomaly interval detection recall: {metrics['anomaly_interval_detection_recall']}",
        f"- Normal interval specificity: {metrics['normal_interval_specificity']}",
        f"- Normal interval false-alarm event rate: {metrics['normal_interval_false_alarm_event_rate']}",
        f"- Median lead time (detected anomalies, days): {metrics['median_lead_time_days_detected_anomalies']}",
        f"- Mean validation confirmed-window FAR: {metrics['mean_validation_false_alarm_rate']}",
        f"- Mean validation warning episodes per 30 days: {metrics['mean_validation_warning_episodes_per_30d']}",
        "",
        "## Interpretation",
        "",
        "Each turbine used the same anomaly-detection framework but an independently fitted scaler, Autoencoder and threshold. "
        "A warning now requires a smoothed global reconstruction deviation, at least one locally abnormal sensor, k-of-n temporal support, and a minimum episode duration. "
        "An anomaly event is counted as early-detected when a retained warning episode appears within the configured pre-event matching window. "
        "A normal event is counted as clean when no confirmed warning occurs inside its recorded interval.",
    ]
    (out_dir / "aggregate_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metadata = load_metadata(cfg)
    feature_desc = load_feature_description(Path(cfg.feature_description_file))
    feature_desc.to_csv(out_dir / "parsed_feature_description.csv", index=False)

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    print(f"Events selected: {len(metadata)}")

    for i, row in metadata.iterrows():
        event_id = str(row["event_id"])
        label = str(row["label"])
        print(f"\n[{i + 1}/{len(metadata)}] Event {event_id} ({label})")
        try:
            summary = process_event(row, cfg, feature_desc)
            results.append(summary)
            print(
                f"Done: threshold={summary['threshold']:.4f}, "
                f"interval warning fraction={summary['recorded_interval_window_warning_fraction']}"
            )
        except Exception as exc:
            print(f"FAILED Event {event_id}: {exc}")
            failures.append({
                "event_id": event_id,
                "label": label,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    if results:
        results_df = pd.DataFrame(results)
        build_aggregate_report(results_df, out_dir)
    pd.DataFrame(failures).to_csv(out_dir / "failed_events.csv", index=False)

    print("\n=== COMPLETE ===")
    print(f"Successful events: {len(results)}")
    print(f"Failed events: {len(failures)}")
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
