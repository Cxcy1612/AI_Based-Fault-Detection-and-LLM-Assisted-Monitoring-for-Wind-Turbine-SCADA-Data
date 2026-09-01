from __future__ import annotations

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration: Farm C Event 9
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FARM_ID = "C"
EVENT_ID = "9"

EVENT_LABEL = "anomaly"
EVENT_START = pd.Timestamp("2017-08-05 07:30:00")
EVENT_START_ID = 52992
EVENT_END = pd.Timestamp("2017-08-26 09:30:00")
EVENT_END_ID = 56028
EVENT_DESCRIPTION = "PENDING19_PREV_YAW_Grease pump defective"

CSV_SEPARATOR = ";"
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# Measurement selection:
# "avg_only" -> *_avg
# "avg_std"  -> *_avg and *_std
# "all"      -> *_avg, *_max, *_min and *_std
MEASUREMENT_MODE = "all"

# Before/after context
USE_EQUAL_LENGTH_BEFORE_AFTER = True
FIXED_CONTEXT_DAYS = 7

# Robust anomaly settings
ROBUST_Z_THRESHOLD = 8.0
ROLLING_POINTS = 6                  # 6 × 10 min ≈ 1 hour
REFERENCE_QUANTILE = 0.995          # adaptive threshold from pre-event period
GLOBAL_FRACTION_FLOOR = 0.05        # only used if reference quantile is lower
LOCAL_TOP_K = 10                    # localised-fault detector
LOCAL_SCORE_FLOOR = 8.0             # minimum mean robust-z for top-K signals

# Segment extraction from adaptive flags
GLOBAL_MIN_SEGMENT_POINTS = 3       # ≥30 min
LOCAL_MIN_SEGMENT_POINTS = 3        # ≥30 min
MAX_GAP_POINTS = 1                  # bridge one missing 10-min point

# Output controls
TOP_N_MEASUREMENTS_TO_PLOT = 20
TOP_N_BASE_SIGNALS = 20
TOP_CANDIDATE_ROWS = 30
TOP_CONTRIBUTORS_PER_CANDIDATE = 20
HEATMAP_TOP_SIGNALS = 20
HEATMAP_RESAMPLE = "1h"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "event_analysis"
    / f"farm_{FARM_ID}_event_{EVENT_ID}_yaw_grease_pump_adaptive"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-9
STAT_PATTERN = re.compile(r"_(avg|max|min|std)$", re.IGNORECASE)


# ============================================================
# File finding and loading
# ============================================================

def find_raw_scada_file(
    data_root: Path,
    farm_id: str,
    event_id: str,
) -> Path:
    possible_farm_dirs = [
        data_root / f"Farm_{farm_id}",
        data_root / f"farm_{farm_id}",
        data_root / f"FARM_{farm_id}",
        data_root / farm_id,
        data_root / f"Wind_Farm_{farm_id}",
        data_root / f"Wind Farm {farm_id}",
        data_root / f"Farm {farm_id}",
    ]

    candidates: list[Path] = []

    for farm_dir in possible_farm_dirs:
        if not farm_dir.exists():
            continue

        candidates.extend(farm_dir.rglob(f"{event_id}.csv"))
        candidates.extend(farm_dir.rglob(f"{event_id}.txt"))
        candidates.extend(farm_dir.rglob(f"*{event_id}*.csv"))
        candidates.extend(farm_dir.rglob(f"*{event_id}*.txt"))

    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find raw SCADA file for Farm {farm_id}, "
            f"Event {event_id} under:\n{data_root}"
        )

    exact = [path for path in candidates if path.stem == str(event_id)]
    return exact[0] if exact else candidates[0]


def load_raw_scada(raw_path: Path) -> pd.DataFrame:
    print(f"Loading raw SCADA file:\n{raw_path}")

    df = pd.read_csv(
        raw_path,
        sep=CSV_SEPARATOR,
        low_memory=False,
    )
    df.columns = [str(column).strip() for column in df.columns]

    required = {"time_stamp", "id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw SCADA file is missing required columns: {sorted(missing)}"
        )

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["time_stamp"] = pd.to_datetime(
        df["time_stamp"],
        errors="coerce",
    )

    df = df.dropna(subset=["id", "time_stamp"]).copy()
    df["id"] = df["id"].astype(int)

    return (
        df.sort_values(["id", "time_stamp"])
        .drop_duplicates(subset=["id"], keep="first")
        .reset_index(drop=True)
    )


# ============================================================
# Measurement handling
# ============================================================

def get_measurement_columns(
    df: pd.DataFrame,
    mode: str,
) -> list[str]:
    exclude = {
        "time_stamp",
        "asset_id",
        "id",
        "train_test",
        "status_type_id",
        "event_id",
        "event_label",
    }

    allowed_prefixes = (
        "sensor_",
        "power_",
        "reactive_power_",
        "wind_speed_",
    )

    if mode == "avg_only":
        suffixes = ("_avg",)
    elif mode == "avg_std":
        suffixes = ("_avg", "_std")
    elif mode == "all":
        suffixes = ("_avg", "_max", "_min", "_std")
    else:
        raise ValueError(
            "MEASUREMENT_MODE must be avg_only, avg_std or all."
        )

    selected: list[str] = []

    for column in df.columns:
        if column in exclude:
            continue
        if not column.startswith(allowed_prefixes):
            continue
        if not column.endswith(suffixes):
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() > 0:
            selected.append(column)

    return selected


def get_base_signal_name(column: str) -> str:
    return STAT_PATTERN.sub("", column)


def get_stat_suffix(column: str) -> str:
    match = STAT_PATTERN.search(column)
    return match.group(1).lower() if match else "unknown"


def group_measurements_by_base_signal(
    measurement_columns: list[str],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}

    for column in measurement_columns:
        base = get_base_signal_name(column)
        groups.setdefault(base, []).append(column)

    return groups


# ============================================================
# Segment extraction
# ============================================================

def extract_segments(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    event_length = EVENT_END_ID - EVENT_START_ID + 1

    if event_length <= 0:
        raise ValueError("EVENT_END_ID must be greater than EVENT_START_ID.")

    context_points = (
        event_length
        if USE_EQUAL_LENGTH_BEFORE_AFTER
        else int(FIXED_CONTEXT_DAYS * 24 * 60 / 10)
    )

    before_start_id = EVENT_START_ID - context_points
    before_end_id = EVENT_START_ID - 1
    after_start_id = EVENT_END_ID + 1
    after_end_id = EVENT_END_ID + context_points

    before = raw.loc[
        raw["id"].between(before_start_id, before_end_id)
    ].copy()

    during = raw.loc[
        raw["id"].between(EVENT_START_ID, EVENT_END_ID)
    ].copy()

    after = raw.loc[
        raw["id"].between(after_start_id, after_end_id)
    ].copy()

    combined = (
        pd.concat(
            [
                before.assign(segment="before"),
                during.assign(segment="during"),
                after.assign(segment="after"),
            ],
            ignore_index=True,
        )
        .sort_values("id")
        .reset_index(drop=True)
    )

    info = {
        "event_length_expected": event_length,
        "context_points_requested": context_points,
        "before_start_id": before_start_id,
        "before_end_id": before_end_id,
        "during_start_id": EVENT_START_ID,
        "during_end_id": EVENT_END_ID,
        "after_start_id": after_start_id,
        "after_end_id": after_end_id,
        "before_rows": len(before),
        "during_rows": len(during),
        "after_rows": len(after),
        "combined_rows": len(combined),
    }

    return before, during, after, combined, info


# ============================================================
# Robust statistics
# ============================================================

def robust_scale(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    median = float(values.median())
    mad = float(np.median(np.abs(values - median))) * 1.4826

    if np.isfinite(mad) and mad > EPSILON:
        return mad

    q25, q75 = values.quantile([0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)

    if np.isfinite(iqr_scale) and iqr_scale > EPSILON:
        return iqr_scale

    std = float(values.std())

    if np.isfinite(std) and std > EPSILON:
        return std

    return np.nan


def calculate_slope(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) < 2:
        return np.nan

    x = np.arange(len(values), dtype=float)

    try:
        return float(np.polyfit(x, values.to_numpy(), 1)[0])
    except Exception:
        return np.nan


def calculate_segment_stats(
    segment: pd.DataFrame,
    measurement_columns: list[str],
    segment_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    for column in measurement_columns:
        values = pd.to_numeric(segment[column], errors="coerce")
        valid = values.dropna()

        if valid.empty:
            rows.append(
                {
                    "measurement": column,
                    "base_signal": get_base_signal_name(column),
                    "stat_type": get_stat_suffix(column),
                    "segment": segment_name,
                    "n_rows": len(values),
                    "n_valid": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "range": np.nan,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "slope": np.nan,
                    "missing_ratio": 1.0,
                }
            )
            continue

        rows.append(
            {
                "measurement": column,
                "base_signal": get_base_signal_name(column),
                "stat_type": get_stat_suffix(column),
                "segment": segment_name,
                "n_rows": len(values),
                "n_valid": int(valid.shape[0]),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "range": float(valid.max() - valid.min()),
                "median": float(valid.median()),
                "q25": float(valid.quantile(0.25)),
                "q75": float(valid.quantile(0.75)),
                "slope": calculate_slope(values),
                "missing_ratio": float(values.isna().mean()),
            }
        )

    return pd.DataFrame(rows)


def safe_relative_change(new_value, old_value) -> float:
    if pd.isna(new_value) or pd.isna(old_value):
        return np.nan
    if abs(old_value) <= EPSILON:
        return np.nan
    return float((new_value - old_value) / abs(old_value))


def build_comparison_table(
    all_stats: pd.DataFrame,
    before: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for measurement in sorted(all_stats["measurement"].unique()):
        sub = all_stats.loc[
            all_stats["measurement"] == measurement
        ]

        row = {
            "measurement": measurement,
            "base_signal": get_base_signal_name(measurement),
            "stat_type": get_stat_suffix(measurement),
        }

        for segment_name in ["before", "during", "after"]:
            segment_row = sub.loc[
                sub["segment"] == segment_name
            ]

            for metric in [
                "mean",
                "std",
                "min",
                "max",
                "range",
                "median",
                "q25",
                "q75",
                "slope",
                "missing_ratio",
            ]:
                row[f"{segment_name}_{metric}"] = (
                    segment_row[metric].iloc[0]
                    if not segment_row.empty
                    else np.nan
                )

        row["during_minus_before_mean"] = (
            row["during_mean"] - row["before_mean"]
            if pd.notna(row["during_mean"])
            and pd.notna(row["before_mean"])
            else np.nan
        )
        row["during_vs_before_rel_mean"] = safe_relative_change(
            row["during_mean"],
            row["before_mean"],
        )
        row["during_vs_before_rel_std"] = safe_relative_change(
            row["during_std"],
            row["before_std"],
        )
        row["during_vs_before_rel_range"] = safe_relative_change(
            row["during_range"],
            row["before_range"],
        )

        scale = robust_scale(before[measurement])
        row["before_robust_scale"] = scale

        if (
            np.isfinite(scale)
            and scale > EPSILON
            and pd.notna(row["before_median"])
            and pd.notna(row["during_median"])
        ):
            row["robust_median_shift"] = abs(
                row["during_median"] - row["before_median"]
            ) / scale
        else:
            row["robust_median_shift"] = np.nan

        row["change_score"] = (
            min(
                float(row["robust_median_shift"])
                if pd.notna(row["robust_median_shift"])
                else 0.0,
                100.0,
            )
            + min(
                abs(float(row["during_vs_before_rel_std"]))
                if pd.notna(row["during_vs_before_rel_std"])
                else 0.0,
                20.0,
            )
            + min(
                abs(float(row["during_vs_before_rel_range"]))
                if pd.notna(row["during_vs_before_rel_range"])
                else 0.0,
                20.0,
            )
        )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("change_score", ascending=False)
        .reset_index(drop=True)
    )


def build_base_signal_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for base_signal, sub in comparison.groupby("base_signal"):
        ordered = sub.sort_values(
            "change_score",
            ascending=False,
        )

        rows.append(
            {
                "base_signal": base_signal,
                "n_measurements": len(sub),
                "top_measurement": ordered["measurement"].iloc[0],
                "max_measurement_change_score": float(
                    ordered["change_score"].iloc[0]
                ),
                "mean_measurement_change_score": float(
                    sub["change_score"].mean()
                ),
                "max_robust_median_shift": float(
                    sub["robust_median_shift"].max()
                )
                if sub["robust_median_shift"].notna().any()
                else np.nan,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "max_measurement_change_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# Row-level adaptive anomaly scoring
# ============================================================

def calculate_robust_z_scores(
    combined: pd.DataFrame,
    before: pd.DataFrame,
    measurement_columns: list[str],
) -> pd.DataFrame:
    z_columns: dict[str, pd.Series] = {}

    for measurement in measurement_columns:
        baseline = pd.to_numeric(
            before[measurement],
            errors="coerce",
        ).dropna()

        if baseline.empty:
            continue

        centre = float(baseline.median())
        scale = robust_scale(baseline)

        if not np.isfinite(scale) or scale <= EPSILON:
            continue

        values = pd.to_numeric(
            combined[measurement],
            errors="coerce",
        )

        z_columns[measurement] = (
            (values - centre).abs() / scale
        ).clip(upper=1_000_000.0)

    if not z_columns:
        raise ValueError(
            "No measurements had a usable pre-event robust scale."
        )

    return pd.DataFrame(
        z_columns,
        index=combined.index,
    )


def build_base_signal_z_scores(
    measurement_z: pd.DataFrame,
) -> pd.DataFrame:
    groups = group_measurements_by_base_signal(
        list(measurement_z.columns)
    )

    base_columns: dict[str, pd.Series] = {}

    for base_signal, columns in groups.items():
        available = [
            column
            for column in columns
            if column in measurement_z.columns
        ]

        if available:
            base_columns[base_signal] = (
                measurement_z[available].max(axis=1)
            )

    if not base_columns:
        raise ValueError("No base-signal z-score table could be built.")

    return pd.DataFrame(
        base_columns,
        index=measurement_z.index,
    ).copy()


def top_k_mean(
    values: pd.DataFrame,
    k: int,
) -> pd.Series:
    array = values.to_numpy(dtype=float)
    k = min(k, array.shape[1])

    if k <= 0:
        return pd.Series(0.0, index=values.index)

    partitioned = np.partition(
        array,
        array.shape[1] - k,
        axis=1,
    )[:, -k:]

    return pd.Series(
        np.nanmean(partitioned, axis=1),
        index=values.index,
    )


def bridge_small_gaps(
    flag: pd.Series,
    max_gap_points: int,
) -> pd.Series:
    values = flag.fillna(False).to_numpy(dtype=bool).copy()

    if max_gap_points <= 0:
        return pd.Series(values, index=flag.index)

    n = len(values)
    i = 0

    while i < n:
        if values[i]:
            i += 1
            continue

        gap_start = i

        while i < n and not values[i]:
            i += 1

        gap_end = i - 1
        gap_length = gap_end - gap_start + 1

        left_true = gap_start > 0 and values[gap_start - 1]
        right_true = i < n and values[i]

        if (
            left_true
            and right_true
            and gap_length <= max_gap_points
        ):
            values[gap_start:i] = True

    return pd.Series(values, index=flag.index)


def contiguous_true_ranges(
    flag: pd.Series,
) -> list[tuple[int, int]]:
    values = flag.fillna(False).to_numpy(dtype=bool)
    ranges: list[tuple[int, int]] = []

    start: int | None = None

    for position, value in enumerate(values):
        if value and start is None:
            start = position
        elif not value and start is not None:
            ranges.append((start, position - 1))
            start = None

    if start is not None:
        ranges.append((start, len(values) - 1))

    return ranges


def calculate_adaptive_threshold(
    score: pd.Series,
    reference_mask: pd.Series,
    quantile: float,
    floor: float,
) -> float:
    reference_values = score.loc[reference_mask].dropna()

    if reference_values.empty:
        raise ValueError(
            "Reference score is empty; adaptive threshold cannot be calculated."
        )

    return max(
        float(floor),
        float(reference_values.quantile(quantile)),
    )


def build_row_scores(
    combined: pd.DataFrame,
    base_z: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    row_scores = combined[
        ["time_stamp", "id", "segment"]
    ].copy()

    reference_mask = row_scores["segment"] == "before"
    event_mask = row_scores["segment"] == "during"

    abnormal_fraction = (
        base_z >= ROBUST_Z_THRESHOLD
    ).mean(axis=1)

    smoothed_fraction = abnormal_fraction.rolling(
        window=ROLLING_POINTS,
        min_periods=1,
        center=True,
    ).mean()

    local_top_k_score = top_k_mean(
        base_z,
        k=LOCAL_TOP_K,
    )

    smoothed_local_score = local_top_k_score.rolling(
        window=ROLLING_POINTS,
        min_periods=1,
        center=True,
    ).mean()

    global_threshold = calculate_adaptive_threshold(
        smoothed_fraction,
        reference_mask,
        REFERENCE_QUANTILE,
        GLOBAL_FRACTION_FLOOR,
    )

    local_threshold = calculate_adaptive_threshold(
        smoothed_local_score,
        reference_mask,
        REFERENCE_QUANTILE,
        LOCAL_SCORE_FLOOR,
    )

    global_raw_flag = (
        smoothed_fraction >= global_threshold
    ) & event_mask

    local_raw_flag = (
        smoothed_local_score >= local_threshold
    ) & event_mask

    global_flag = bridge_small_gaps(
        global_raw_flag,
        MAX_GAP_POINTS,
    ) & event_mask

    local_flag = bridge_small_gaps(
        local_raw_flag,
        MAX_GAP_POINTS,
    ) & event_mask

    row_scores["base_signal_abnormal_fraction"] = abnormal_fraction
    row_scores["smoothed_abnormal_fraction"] = smoothed_fraction
    row_scores["adaptive_global_threshold"] = global_threshold
    row_scores["global_abnormal_flag"] = global_flag

    row_scores["top_k_local_score"] = local_top_k_score
    row_scores["smoothed_top_k_local_score"] = smoothed_local_score
    row_scores["adaptive_local_threshold"] = local_threshold
    row_scores["localised_abnormal_flag"] = local_flag

    row_scores["global_rank"] = smoothed_fraction.rank(
        method="average",
        pct=True,
    )
    row_scores["local_rank"] = smoothed_local_score.rank(
        method="average",
        pct=True,
    )

    row_scores["exploratory_composite_rank"] = (
        row_scores["global_rank"]
        + row_scores["local_rank"]
    ) / 2.0

    row_scores.loc[
        ~event_mask,
        "exploratory_composite_rank",
    ] = np.nan

    calibration = {
        "robust_z_threshold": ROBUST_Z_THRESHOLD,
        "rolling_points": ROLLING_POINTS,
        "reference_quantile": REFERENCE_QUANTILE,
        "adaptive_global_threshold": global_threshold,
        "adaptive_local_threshold": local_threshold,
        "global_reference_trigger_rate": float(
            (
                smoothed_fraction.loc[reference_mask]
                >= global_threshold
            ).mean()
        ),
        "local_reference_trigger_rate": float(
            (
                smoothed_local_score.loc[reference_mask]
                >= local_threshold
            ).mean()
        ),
    }

    return row_scores, calibration


def extract_detected_segments(
    row_scores: pd.DataFrame,
) -> pd.DataFrame:
    specifications = [
        (
            "system_wide_multisensor_state",
            "global_abnormal_flag",
            "smoothed_abnormal_fraction",
            GLOBAL_MIN_SEGMENT_POINTS,
        ),
        (
            "localised_topk_state",
            "localised_abnormal_flag",
            "smoothed_top_k_local_score",
            LOCAL_MIN_SEGMENT_POINTS,
        ),
    ]

    segment_rows: list[dict] = []

    for (
        detector_name,
        flag_column,
        score_column,
        min_points,
    ) in specifications:
        ranges = contiguous_true_ranges(
            row_scores[flag_column]
        )

        for start_position, end_position in ranges:
            n_points = end_position - start_position + 1

            if n_points < min_points:
                continue

            segment = row_scores.iloc[
                start_position:end_position + 1
            ].copy()

            strongest_index = segment[
                score_column
            ].idxmax()
            strongest = row_scores.loc[strongest_index]

            sampling_minutes = (
                row_scores["time_stamp"]
                .sort_values()
                .diff()
                .dropna()
                .dt.total_seconds()
                .div(60)
                .median()
            )

            if not np.isfinite(sampling_minutes):
                sampling_minutes = 10.0

            segment_rows.append(
                {
                    "detector": detector_name,
                    "segment_start": segment["time_stamp"].iloc[0],
                    "segment_end": (
                        segment["time_stamp"].iloc[-1]
                        + pd.Timedelta(minutes=float(sampling_minutes))
                    ),
                    "start_id": int(segment["id"].iloc[0]),
                    "end_id": int(segment["id"].iloc[-1]),
                    "n_points": n_points,
                    "duration_minutes": float(
                        n_points * sampling_minutes
                    ),
                    "strongest_candidate_time": strongest["time_stamp"],
                    "strongest_candidate_id": int(strongest["id"]),
                    "peak_score": float(strongest[score_column]),
                    "peak_composite_rank": float(
                        strongest["exploratory_composite_rank"]
                    ),
                }
            )

    return pd.DataFrame(segment_rows).sort_values(
        ["segment_start", "detector"]
    ).reset_index(drop=True)


def build_candidate_contributors(
    row_scores: pd.DataFrame,
    measurement_z: pd.DataFrame,
    base_z: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = (
        row_scores.loc[row_scores["segment"] == "during"]
        .sort_values(
            "exploratory_composite_rank",
            ascending=False,
        )
        .head(TOP_CANDIDATE_ROWS)
        .copy()
    )

    candidates.insert(
        0,
        "candidate_rank",
        range(1, len(candidates) + 1),
    )

    contributor_rows: list[dict] = []

    for source_index, candidate in candidates.iterrows():
        measurement_top = (
            measurement_z.loc[source_index]
            .sort_values(ascending=False)
            .head(TOP_CONTRIBUTORS_PER_CANDIDATE)
        )
        base_top = (
            base_z.loc[source_index]
            .sort_values(ascending=False)
            .head(TOP_CONTRIBUTORS_PER_CANDIDATE)
        )

        for rank, (name, score) in enumerate(
            measurement_top.items(),
            start=1,
        ):
            contributor_rows.append(
                {
                    "candidate_rank": int(
                        candidate["candidate_rank"]
                    ),
                    "candidate_time": candidate["time_stamp"],
                    "candidate_id": int(candidate["id"]),
                    "level": "measurement",
                    "contributor_rank": rank,
                    "contributor": name,
                    "base_signal": get_base_signal_name(name),
                    "robust_z": float(score),
                }
            )

        for rank, (name, score) in enumerate(
            base_top.items(),
            start=1,
        ):
            contributor_rows.append(
                {
                    "candidate_rank": int(
                        candidate["candidate_rank"]
                    ),
                    "candidate_time": candidate["time_stamp"],
                    "candidate_id": int(candidate["id"]),
                    "level": "base_signal",
                    "contributor_rank": rank,
                    "contributor": name,
                    "base_signal": name,
                    "robust_z": float(score),
                }
            )

    return (
        candidates.reset_index(names="source_index"),
        pd.DataFrame(contributor_rows),
    )


# ============================================================
# Plotting
# ============================================================

def add_event_boundaries(ax) -> None:
    ax.axvline(
        EVENT_START,
        linestyle="--",
        label="metadata start",
    )
    ax.axvline(
        EVENT_END,
        linestyle="--",
        label="metadata end",
    )


def plot_adaptive_global_timeline(
    row_scores: pd.DataFrame,
) -> None:
    threshold = float(
        row_scores["adaptive_global_threshold"].iloc[0]
    )

    fig, ax = plt.subplots(figsize=(17, 5))
    ax.plot(
        row_scores["time_stamp"],
        row_scores["smoothed_abnormal_fraction"],
        label="smoothed base-signal abnormal fraction",
    )
    ax.axhline(
        threshold,
        linestyle="--",
        label=f"adaptive 99.5% threshold = {threshold:.4f}",
    )
    add_event_boundaries(ax)

    ax.set_title(
        f"Farm {FARM_ID} Event {EVENT_ID}: "
        "adaptive multisensor abnormal-fraction timeline"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Base-signal abnormal fraction")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "adaptive_multisensor_abnormal_fraction_timeline.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_adaptive_local_timeline(
    row_scores: pd.DataFrame,
) -> None:
    threshold = float(
        row_scores["adaptive_local_threshold"].iloc[0]
    )

    fig, ax = plt.subplots(figsize=(17, 5))
    ax.plot(
        row_scores["time_stamp"],
        row_scores["smoothed_top_k_local_score"],
        label=f"smoothed mean robust-z of top {LOCAL_TOP_K} base signals",
    )
    ax.axhline(
        threshold,
        linestyle="--",
        label=f"adaptive 99.5% threshold = {threshold:.4g}",
    )
    add_event_boundaries(ax)

    ax.set_title(
        f"Farm {FARM_ID} Event {EVENT_ID}: "
        "adaptive localised top-K anomaly timeline"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Top-{LOCAL_TOP_K} mean robust z-score")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "adaptive_localised_topk_timeline.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_daily_adaptive_summary(
    row_scores: pd.DataFrame,
) -> None:
    daily = (
        row_scores.set_index("time_stamp")
        .resample("1D")
        .agg(
            global_median=(
                "smoothed_abnormal_fraction",
                "median",
            ),
            global_max=(
                "smoothed_abnormal_fraction",
                "max",
            ),
            local_median=(
                "smoothed_top_k_local_score",
                "median",
            ),
            local_max=(
                "smoothed_top_k_local_score",
                "max",
            ),
        )
        .dropna(how="all")
    )

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(
        daily.index,
        daily["global_median"],
        label="daily median global fraction",
    )
    ax.plot(
        daily.index,
        daily["global_max"],
        label="daily max global fraction",
    )
    add_event_boundaries(ax)

    ax.set_title(
        f"Farm {FARM_ID} Event {EVENT_ID}: "
        "daily global multisensor abnormality"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Abnormal fraction")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "daily_global_abnormality.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(
        daily.index,
        daily["local_median"],
        label="daily median local top-K score",
    )
    ax.plot(
        daily.index,
        daily["local_max"],
        label="daily max local top-K score",
    )
    add_event_boundaries(ax)

    ax.set_title(
        f"Farm {FARM_ID} Event {EVENT_ID}: "
        "daily localised top-K anomaly"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Top-{LOCAL_TOP_K} mean robust z-score")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "daily_localised_topk_anomaly.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_top_measurement_changes(
    comparison: pd.DataFrame,
) -> None:
    top = comparison.head(20).sort_values(
        "change_score",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        top["measurement"],
        top["change_score"],
    )
    ax.set_title(
        f"Farm {FARM_ID} Event {EVENT_ID}: "
        "top changed measurements"
    )
    ax.set_xlabel("Before-vs-during change score")
    ax.set_ylabel("Measurement")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "top20_changed_measurements_bar.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_top_base_signal_changes(
    base_summary: pd.DataFrame,
) -> None:
    top = (
        base_summary.head(TOP_N_BASE_SIGNALS)
        .sort_values(
            "max_measurement_change_score",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        top["base_signal"],
        top["max_measurement_change_score"],
    )
    ax.set_title(
        f"Farm {FARM_ID} Event {EVENT_ID}: "
        "top changed base signals"
    )
    ax.set_xlabel("Maximum measurement-level change score")
    ax.set_ylabel("Base signal")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "top_base_signal_changes.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_top_signal_heatmap(
    combined: pd.DataFrame,
    base_z: pd.DataFrame,
    base_summary: pd.DataFrame,
) -> None:
    top_signals = [
        signal
        for signal in base_summary[
            "base_signal"
        ].head(HEATMAP_TOP_SIGNALS)
        if signal in base_z.columns
    ]

    if not top_signals:
        return

    heatmap = base_z[top_signals].copy()
    heatmap["time_stamp"] = combined["time_stamp"].values

    heatmap = (
        heatmap.set_index("time_stamp")
        .resample(HEATMAP_RESAMPLE)
        .max()
        .T
    )

    values = np.log1p(
        heatmap.to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(18, 8))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)

    if heatmap.shape[1] > 0:
        tick_positions = np.linspace(
            0,
            heatmap.shape[1] - 1,
            num=min(12, heatmap.shape[1]),
            dtype=int,
        )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [
                heatmap.columns[position].strftime(
                    "%Y-%m-%d\n%H:%M"
                )
                for position in tick_positions
            ],
            rotation=45,
            ha="right",
        )

    ax.set_title(
        f"Farm {FARM_ID} Event {EVENT_ID}: "
        f"top base-signal robust-z heatmap ({HEATMAP_RESAMPLE} max)"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Base signal")

    colour_bar = fig.colorbar(image, ax=ax)
    colour_bar.set_label("log(1 + robust z-score)")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "top_base_signal_z_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_measurement_time_series(
    combined: pd.DataFrame,
    measurement: str,
) -> None:
    values = pd.to_numeric(
        combined[measurement],
        errors="coerce",
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(
        combined["time_stamp"],
        values,
        linewidth=1,
    )
    add_event_boundaries(ax)

    ax.set_title(
        f"Farm {FARM_ID} Event {EVENT_ID}: "
        f"{measurement} before/during/after"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel(measurement)
    ax.legend()

    fig.tight_layout()

    safe_name = measurement.replace("/", "_").replace("\\", "_")
    fig.savefig(
        OUTPUT_DIR / f"{safe_name}_before_during_after.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_candidate_contexts(
    candidates: pd.DataFrame,
    combined: pd.DataFrame,
    contributors: pd.DataFrame,
) -> None:
    context_dir = OUTPUT_DIR / "candidate_signal_contexts"
    context_dir.mkdir(parents=True, exist_ok=True)

    for _, candidate in candidates.head(10).iterrows():
        rank = int(candidate["candidate_rank"])
        candidate_time = pd.Timestamp(candidate["time_stamp"])

        top_base = (
            contributors.loc[
                (contributors["candidate_rank"] == rank)
                & (contributors["level"] == "base_signal")
            ]
            .sort_values("contributor_rank")
            ["base_signal"]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        if not top_base:
            continue

        context = combined.loc[
            combined["time_stamp"].between(
                candidate_time - pd.Timedelta(hours=3),
                candidate_time + pd.Timedelta(hours=3),
            )
        ].copy()

        if context.empty:
            continue

        fig, ax = plt.subplots(figsize=(15, 6))

        for base_signal in top_base:
            avg_column = f"{base_signal}_avg"

            candidate_columns = [
                column
                for column in combined.columns
                if get_base_signal_name(column) == base_signal
            ]

            plot_column = (
                avg_column
                if avg_column in context.columns
                else candidate_columns[0]
                if candidate_columns
                else None
            )

            if plot_column is None:
                continue

            values = pd.to_numeric(
                context[plot_column],
                errors="coerce",
            )

            median = values.median()
            scale = robust_scale(values)

            if not np.isfinite(scale) or scale <= EPSILON:
                continue

            standardised = (values - median) / scale

            ax.plot(
                context["time_stamp"],
                standardised,
                label=plot_column,
            )

        ax.axvline(
            candidate_time,
            linestyle="--",
            label="candidate time",
        )
        ax.set_title(
            f"Event {EVENT_ID} candidate {rank}: "
            f"{candidate_time}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Locally standardised signal value")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            context_dir
            / (
                f"candidate_{rank:02d}_"
                f"{candidate_time:%Y%m%d_%H%M}_context.png"
            ),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


# ============================================================
# Summary output
# ============================================================

def write_markdown_summary(
    segment_info: dict,
    comparison: pd.DataFrame,
    base_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    segments: pd.DataFrame,
    calibration: dict,
) -> None:
    path = OUTPUT_DIR / "event9_adaptive_exploratory_summary.md"

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Farm C Event 9 Adaptive Exploratory Analysis\n\n")

        file.write("## Event information\n\n")
        file.write(f"- Event ID: `{EVENT_ID}`\n")
        file.write(f"- Event label: `{EVENT_LABEL}`\n")
        file.write(f"- Metadata start: `{EVENT_START}`\n")
        file.write(f"- Start ID: `{EVENT_START_ID}`\n")
        file.write(f"- Metadata end: `{EVENT_END}`\n")
        file.write(f"- End ID: `{EVENT_END_ID}`\n")
        file.write(f"- Description: `{EVENT_DESCRIPTION}`\n\n")

        file.write("## Adaptive thresholds\n\n")
        file.write(
            f"- Robust z-score threshold: "
            f"`{ROBUST_Z_THRESHOLD}`\n"
        )
        file.write(
            f"- Reference quantile: "
            f"`{REFERENCE_QUANTILE}`\n"
        )
        file.write(
            f"- Global abnormal-fraction threshold: "
            f"`{calibration['adaptive_global_threshold']:.6g}`\n"
        )
        file.write(
            f"- Local top-{LOCAL_TOP_K} threshold: "
            f"`{calibration['adaptive_local_threshold']:.6g}`\n"
        )
        file.write(
            f"- Global reference trigger rate: "
            f"`{calibration['global_reference_trigger_rate']:.4%}`\n"
        )
        file.write(
            f"- Local reference trigger rate: "
            f"`{calibration['local_reference_trigger_rate']:.4%}`\n\n"
        )

        file.write(
            "The thresholds are calculated from the pre-event period. "
            "They therefore replace the previous fixed exploratory "
            "abnormal-fraction threshold of 0.05.\n\n"
        )

        file.write("## Segment availability\n\n")
        file.write(
            f"- Before rows: `{segment_info['before_rows']}`\n"
        )
        file.write(
            f"- During rows: `{segment_info['during_rows']}`\n"
        )
        file.write(
            f"- After rows: `{segment_info['after_rows']}`\n\n"
        )

        file.write("## Detected exploratory periods\n\n")

        if segments.empty:
            file.write(
                "No periods met the adaptive duration criteria.\n\n"
            )
        else:
            file.write(
                "| Detector | Start | End | Points | "
                "Duration (min) | Strongest candidate | Peak score |\n"
                "|---|---|---|---:|---:|---|---:|\n"
            )

            for _, row in segments.iterrows():
                file.write(
                    f"| `{row['detector']}` | "
                    f"{row['segment_start']} | "
                    f"{row['segment_end']} | "
                    f"{int(row['n_points'])} | "
                    f"{row['duration_minutes']:.1f} | "
                    f"{row['strongest_candidate_time']} | "
                    f"{row['peak_score']:.6g} |\n"
                )

            file.write("\n")

        file.write("## Highest-ranked event rows\n\n")
        file.write(
            "| Rank | Time | ID | Global fraction | "
            f"Top-{LOCAL_TOP_K} local score | Composite rank |\n"
            "|---:|---|---:|---:|---:|---:|\n"
        )

        for _, row in candidates.head(15).iterrows():
            file.write(
                f"| {int(row['candidate_rank'])} | "
                f"{row['time_stamp']} | "
                f"{int(row['id'])} | "
                f"{row['smoothed_abnormal_fraction']:.6g} | "
                f"{row['smoothed_top_k_local_score']:.6g} | "
                f"{row['exploratory_composite_rank']:.6g} |\n"
            )

        file.write("\n## Top changed measurements\n\n")
        file.write(
            "| Rank | Measurement | Base signal | "
            "Before median | During median | Change score |\n"
            "|---:|---|---|---:|---:|---:|\n"
        )

        for rank, (_, row) in enumerate(
            comparison.head(15).iterrows(),
            start=1,
        ):
            file.write(
                f"| {rank} | `{row['measurement']}` | "
                f"`{row['base_signal']}` | "
                f"{row['before_median']:.6g} | "
                f"{row['during_median']:.6g} | "
                f"{row['change_score']:.6g} |\n"
            )

        file.write("\n## Top changed base signals\n\n")
        file.write(
            "| Rank | Base signal | Top measurement | "
            "Maximum change score |\n"
            "|---:|---|---|---:|\n"
        )

        for rank, (_, row) in enumerate(
            base_summary.head(15).iterrows(),
            start=1,
        ):
            file.write(
                f"| {rank} | `{row['base_signal']}` | "
                f"`{row['top_measurement']}` | "
                f"{row['max_measurement_change_score']:.6g} |\n"
            )

        file.write("\n## Interpretation note\n\n")
        file.write(
            "- `system_wide_multisensor_state` means a relatively large "
            "fraction of base signals exceeded the pre-event 99.5th-percentile "
            "threshold.\n"
            "- `localised_topk_state` means a small number of signals became "
            "extremely unusual even if the whole system did not change.\n"
            "- These are exploratory SCADA periods, not confirmed grease-pump "
            "failure times or fault probabilities.\n"
            "- Because the SCADA variables are anonymised, component-level "
            "causality cannot be confirmed from this analysis alone.\n"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("Farm C Event 9 - Adaptive Yaw Grease Pump SCADA Analysis")
    print("=" * 100)

    raw_path = find_raw_scada_file(
        DATA_ROOT,
        FARM_ID,
        EVENT_ID,
    )
    raw = load_raw_scada(raw_path)

    before, during, after, combined, segment_info = (
        extract_segments(raw)
    )

    if before.empty:
        raise ValueError(
            "Before segment is empty; an adaptive baseline cannot be built."
        )

    if during.empty:
        raise ValueError(
            "During segment is empty; check Event 9 IDs."
        )

    if after.empty:
        warnings.warn(
            "After segment is empty. Recovery cannot be assessed, but "
            "before-vs-during and adaptive scoring will still run."
        )

    measurement_columns = get_measurement_columns(
        raw,
        MEASUREMENT_MODE,
    )

    if not measurement_columns:
        raise ValueError(
            "No usable SCADA measurement columns were found."
        )

    print(f"Selected measurement columns: {len(measurement_columns)}")

    pd.DataFrame(
        {"measurement": measurement_columns}
    ).to_csv(
        OUTPUT_DIR / "selected_measurement_columns.csv",
        index=False,
    )

    before_stats = calculate_segment_stats(
        before,
        measurement_columns,
        "before",
    )
    during_stats = calculate_segment_stats(
        during,
        measurement_columns,
        "during",
    )
    after_stats = calculate_segment_stats(
        after,
        measurement_columns,
        "after",
    )

    all_stats = pd.concat(
        [before_stats, during_stats, after_stats],
        ignore_index=True,
    )
    all_stats.to_csv(
        OUTPUT_DIR / "before_during_after_segment_stats.csv",
        index=False,
    )

    comparison = build_comparison_table(
        all_stats,
        before,
    )
    comparison.to_csv(
        OUTPUT_DIR / "sensor_change_comparison.csv",
        index=False,
    )
    comparison.head(20).to_csv(
        OUTPUT_DIR / "top20_changed_measurements.csv",
        index=False,
    )

    base_summary = build_base_signal_summary(comparison)
    base_summary.to_csv(
        OUTPUT_DIR / "base_signal_change_summary.csv",
        index=False,
    )

    measurement_z = calculate_robust_z_scores(
        combined,
        before,
        measurement_columns,
    )
    base_z = build_base_signal_z_scores(
        measurement_z,
    )

    row_scores, calibration = build_row_scores(
        combined,
        base_z,
    )
    row_scores.to_csv(
        OUTPUT_DIR / "event9_adaptive_row_scores.csv",
        index=False,
    )

    segments = extract_detected_segments(
        row_scores,
    )
    segments.to_csv(
        OUTPUT_DIR / "event9_adaptive_detected_segments.csv",
        index=False,
    )

    candidates, contributors = build_candidate_contributors(
        row_scores,
        measurement_z,
        base_z,
    )
    candidates.to_csv(
        OUTPUT_DIR / "event9_adaptive_top_candidates.csv",
        index=False,
    )
    contributors.to_csv(
        OUTPUT_DIR / "event9_adaptive_candidate_contributors.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "event9_adaptive_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "farm_id": FARM_ID,
                "event_id": EVENT_ID,
                "event_start": str(EVENT_START),
                "event_end": str(EVENT_END),
                "event_start_id": EVENT_START_ID,
                "event_end_id": EVENT_END_ID,
                "event_description": EVENT_DESCRIPTION,
                "measurement_mode": MEASUREMENT_MODE,
                "measurement_columns": len(measurement_columns),
                "base_signals": int(base_z.shape[1]),
                "segment_info": segment_info,
                "calibration": calibration,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Saving figures...")

    plot_adaptive_global_timeline(row_scores)
    plot_adaptive_local_timeline(row_scores)
    plot_daily_adaptive_summary(row_scores)
    plot_top_measurement_changes(comparison)
    plot_top_base_signal_changes(base_summary)
    plot_top_signal_heatmap(
        combined,
        base_z,
        base_summary,
    )
    plot_candidate_contexts(
        candidates,
        combined,
        contributors,
    )

    for measurement in comparison[
        "measurement"
    ].head(TOP_N_MEASUREMENTS_TO_PLOT):
        try:
            plot_measurement_time_series(
                combined,
                measurement,
            )
        except Exception as exc:
            print(
                f"Could not plot {measurement}: {exc}"
            )

    write_markdown_summary(
        segment_info,
        comparison,
        base_summary,
        candidates,
        segments,
        calibration,
    )

    print("\nAdaptive thresholds:")
    print(
        f"Global abnormal fraction: "
        f"{calibration['adaptive_global_threshold']:.6g}"
    )
    print(
        f"Local top-{LOCAL_TOP_K} score: "
        f"{calibration['adaptive_local_threshold']:.6g}"
    )

    print("\nDetected exploratory periods:")
    if segments.empty:
        print("None")
    else:
        print(
            segments.to_string(index=False)
        )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nMain figures:")
    print("- adaptive_multisensor_abnormal_fraction_timeline.png")
    print("- adaptive_localised_topk_timeline.png")
    print("- daily_global_abnormality.png")
    print("- daily_localised_topk_anomaly.png")
    print("- top20_changed_measurements_bar.png")
    print("- top_base_signal_changes.png")
    print("- top_base_signal_z_heatmap.png")
    print("- candidate_signal_contexts/*.png")
    print("- top measurement before/during/after timelines")


if __name__ == "__main__":
    main()