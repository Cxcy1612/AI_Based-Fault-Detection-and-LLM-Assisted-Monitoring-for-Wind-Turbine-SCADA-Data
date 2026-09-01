from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 改成你的真实 farm
FARM_ID = "C"

# 这个值要对应 raw SCADA 文件名，例如 55.csv / 55.txt
# 这里使用 Event 47，对应原始文件 47.csv / 47.txt
ASSET_ID_OR_EVENT_KEY = "47"

EVENT_LABEL = "anomaly"
EVENT_START = "2018-12-23 15:00:00"
EVENT_START_ID = 52416
EVENT_END = "2018-12-28 13:40:00"
EVENT_END_ID = 53128
EVENT_DESCRIPTION = "Failure due to Rotorbrake and Hydraulic problems - Hydraulic pump A disabled; 2h later turbine back in production"

CSV_SEPARATOR = ";"

# raw data 根目录
# 你的项目大概率是：
# wind_farm_fault_detection/data/...

DATA_ROOT = PROJECT_ROOT / "data" / "raw" 

# 选择分析哪些列：
# "avg_only"  = 只分析 *_avg，最适合报告解释
# "avg_std"   = 分析 *_avg 和 *_std，推荐
# "all"       = 分析 avg/max/min/std，列很多
MEASUREMENT_MODE = "all"

# before / after 是否取和 during 一样长
USE_EQUAL_LENGTH_BEFORE_AFTER = True

# 只保存变化最大的前多少个 sensor 图
TOP_N_TO_PLOT = 15

# 额外输出：全局多传感器异常比例时间线
# 10分钟采样下，6个点约等于1小时
GLOBAL_ROLLING_POINTS = 6
GLOBAL_ROBUST_Z_THRESHOLD = 8.0
GLOBAL_THRESHOLD_QUANTILE = 0.995
GLOBAL_THRESHOLD_FLOOR = 0.05

# 结果输出目录
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "event_analysis"
    / f"farm_{FARM_ID}_asset_{ASSET_ID_OR_EVENT_KEY}_event_{EVENT_START_ID}_{EVENT_END_ID}_{MEASUREMENT_MODE}_top{TOP_N_TO_PLOT}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# File finding and loading
# ============================================================

def find_raw_scada_file(
    data_root: Path,
    farm_id: str,
    asset_id_or_event_key: str,
) -> Path:
    """
    Try to find the raw SCADA file for a given farm and asset/event key.

    The dataset structure may be slightly different, so this function searches
    common folder names and file patterns.
    """
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

        # exact file names first
        candidates.extend(farm_dir.rglob(f"{asset_id_or_event_key}.csv"))
        candidates.extend(farm_dir.rglob(f"{asset_id_or_event_key}.txt"))

        # then broader matching
        candidates.extend(farm_dir.rglob(f"*{asset_id_or_event_key}*.csv"))
        candidates.extend(farm_dir.rglob(f"*{asset_id_or_event_key}*.txt"))

    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find raw SCADA file for Farm {farm_id}, "
            f"asset/event key {asset_id_or_event_key}.\n\n"
            f"Checked under: {data_root}\n\n"
            "Please check:\n"
            "1. DATA_ROOT is correct\n"
            "2. FARM_ID is correct\n"
            "3. ASSET_ID_OR_EVENT_KEY matches the raw file name\n"
        )

    exact_matches = [
        p for p in candidates
        if p.stem == str(asset_id_or_event_key)
    ]

    if exact_matches:
        return exact_matches[0]

    return candidates[0]


def load_raw_scada(raw_path: Path) -> pd.DataFrame:
    """
    Load raw SCADA data using semicolon separator.
    """
    print(f"Loading raw SCADA file:\n{raw_path}")

    df = pd.read_csv(raw_path, sep=CSV_SEPARATOR, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"time_stamp", "asset_id", "id"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Raw SCADA file is missing required columns: {missing}"
        )

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")

    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    df = df.sort_values("id").reset_index(drop=True)

    return df


# ============================================================
# Column selection
# ============================================================

def get_measurement_columns(
    df: pd.DataFrame,
    mode: str = "avg_std",
) -> list[str]:
    """
    Select SCADA measurement columns.

    Excluded:
    - time_stamp
    - asset_id
    - id
    - train_test
    - status_type_id

    Included prefixes:
    - sensor_
    - power_
    - reactive_power_
    - wind_speed_

    mode:
    - avg_only: use only *_avg columns
    - avg_std: use *_avg and *_std columns
    - all: use *_avg, *_max, *_min and *_std columns
    """
    exclude_cols = {
        "time_stamp",
        "asset_id",
        "id",
        "train_test",
        "status_type_id",
    }

    allowed_prefixes = (
        "sensor_",
        "power_",
        "reactive_power_",
        "wind_speed_",
    )

    if mode == "avg_only":
        allowed_suffixes = ("_avg",)
    elif mode == "avg_std":
        allowed_suffixes = ("_avg", "_std")
    elif mode == "all":
        allowed_suffixes = ("_avg", "_max", "_min", "_std")
    else:
        raise ValueError("mode must be one of: avg_only, avg_std, all")

    selected_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if not col.startswith(allowed_prefixes):
            continue

        if not col.endswith(allowed_suffixes):
            continue

        numeric_col = pd.to_numeric(df[col], errors="coerce")

        # keep only columns with at least some numeric values
        if numeric_col.notna().sum() > 0:
            selected_cols.append(col)

    return selected_cols


def get_base_signal_name(column_name: str) -> str:
    """
    Convert sensor_0_avg -> sensor_0
    Convert power_2_std -> power_2
    Convert wind_speed_236_avg -> wind_speed_236
    """
    return re.sub(r"_(avg|max|min|std)$", "", column_name)


def get_stat_suffix(column_name: str) -> str:
    """
    Extract avg / max / min / std from column name.
    """
    match = re.search(r"_(avg|max|min|std)$", column_name)
    if match:
        return match.group(1)
    return "unknown"


# ============================================================
# Segment extraction
# ============================================================

def extract_before_during_after_segments(
    raw: pd.DataFrame,
    event_start_id: int,
    event_end_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Extract before, during and after segments.

    before and after are equal length to the event interval.
    """
    event_length = event_end_id - event_start_id + 1

    if event_length <= 0:
        raise ValueError("EVENT_END_ID must be greater than EVENT_START_ID.")

    before_start_id = event_start_id - event_length
    before_end_id = event_start_id - 1

    after_start_id = event_end_id + 1
    after_end_id = event_end_id + event_length

    before = raw[
        (raw["id"] >= before_start_id)
        & (raw["id"] <= before_end_id)
    ].copy()

    during = raw[
        (raw["id"] >= event_start_id)
        & (raw["id"] <= event_end_id)
    ].copy()

    after = raw[
        (raw["id"] >= after_start_id)
        & (raw["id"] <= after_end_id)
    ].copy()

    info = {
        "event_length_expected": event_length,
        "before_start_id": before_start_id,
        "before_end_id": before_end_id,
        "during_start_id": event_start_id,
        "during_end_id": event_end_id,
        "after_start_id": after_start_id,
        "after_end_id": after_end_id,
        "before_rows": len(before),
        "during_rows": len(during),
        "after_rows": len(after),
    }

    return before, during, after, info


def get_time_period(segment: pd.DataFrame) -> tuple[str, str]:
    """
    Get first and last timestamp for a segment.
    """
    if segment.empty:
        return "NA", "NA"

    start_time = str(segment["time_stamp"].iloc[0])
    end_time = str(segment["time_stamp"].iloc[-1])

    return start_time, end_time


# ============================================================
# Statistics
# ============================================================

def calculate_slope(values: pd.Series) -> float:
    """
    Calculate simple linear slope over row order.
    """
    x = pd.to_numeric(values, errors="coerce").dropna()

    if len(x) < 2:
        return np.nan

    y = x.values
    t = np.arange(len(y))

    try:
        return float(np.polyfit(t, y, 1)[0])
    except Exception:
        return np.nan


def calculate_segment_stats(
    segment: pd.DataFrame,
    measurement_cols: list[str],
    segment_name: str,
) -> pd.DataFrame:
    """
    Calculate statistics for each measurement column in one segment.
    """
    rows = []

    for col in measurement_cols:
        x = pd.to_numeric(segment[col], errors="coerce")

        if len(x) == 0 or x.notna().sum() == 0:
            rows.append({
                "measurement": col,
                "base_signal": get_base_signal_name(col),
                "stat_type": get_stat_suffix(col),
                "segment": segment_name,
                "n_rows": len(x),
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
            })
            continue

        valid = x.dropna()

        rows.append({
            "measurement": col,
            "base_signal": get_base_signal_name(col),
            "stat_type": get_stat_suffix(col),
            "segment": segment_name,
            "n_rows": len(x),
            "n_valid": int(valid.shape[0]),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "range": float(valid.max() - valid.min()),
            "median": float(valid.median()),
            "q25": float(valid.quantile(0.25)),
            "q75": float(valid.quantile(0.75)),
            "slope": calculate_slope(x),
            "missing_ratio": float(x.isna().mean()),
        })

    return pd.DataFrame(rows)


def safe_diff(a, b) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a - b)


def safe_relative_change(new_value, old_value) -> float:
    """
    Relative change = (new - old) / abs(old)
    """
    if pd.isna(new_value) or pd.isna(old_value):
        return np.nan

    if abs(old_value) < 1e-9:
        return np.nan

    return float((new_value - old_value) / abs(old_value))


def build_comparison_table(all_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build before/during/after comparison table for each measurement.
    """
    metrics = [
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
    ]

    comparison_rows = []

    measurements = sorted(all_stats["measurement"].unique())

    for measurement in measurements:
        sub = all_stats[all_stats["measurement"] == measurement].copy()

        row = {
            "measurement": measurement,
            "base_signal": get_base_signal_name(measurement),
            "stat_type": get_stat_suffix(measurement),
        }

        segment_stats = {
            segment: sub[sub["segment"] == segment]
            for segment in ["before", "during", "after"]
        }

        for metric in metrics:
            values = {}

            for segment in ["before", "during", "after"]:
                seg_df = segment_stats[segment]

                if seg_df.empty:
                    values[segment] = np.nan
                else:
                    values[segment] = seg_df[metric].iloc[0]

                row[f"{segment}_{metric}"] = values[segment]

            before_value = values["before"]
            during_value = values["during"]
            after_value = values["after"]

            row[f"during_minus_before_{metric}"] = safe_diff(
                during_value,
                before_value,
            )
            row[f"after_minus_during_{metric}"] = safe_diff(
                after_value,
                during_value,
            )
            row[f"after_minus_before_{metric}"] = safe_diff(
                after_value,
                before_value,
            )

            row[f"during_vs_before_rel_{metric}"] = safe_relative_change(
                during_value,
                before_value,
            )
            row[f"after_vs_before_rel_{metric}"] = safe_relative_change(
                after_value,
                before_value,
            )

        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)

    # A simple ranking score:
    # mean change + variability change + range change
    comparison["change_score"] = (
        comparison["during_vs_before_rel_mean"].abs().fillna(0)
        + comparison["during_vs_before_rel_std"].abs().fillna(0)
        + comparison["during_vs_before_rel_range"].abs().fillna(0)
    )

    comparison = comparison.sort_values(
        "change_score",
        ascending=False,
    ).reset_index(drop=True)

    return comparison


# ============================================================
# Group-level summaries
# ============================================================

def build_base_signal_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise changes at base signal level.

    Example:
    sensor_0_avg and sensor_0_std belong to base_signal sensor_0.
    """
    rows = []

    for base_signal, sub in comparison.groupby("base_signal"):
        avg_rows = sub[sub["stat_type"] == "avg"]
        std_rows = sub[sub["stat_type"] == "std"]
        max_rows = sub[sub["stat_type"] == "max"]
        min_rows = sub[sub["stat_type"] == "min"]

        def get_first_value(df: pd.DataFrame, col: str):
            if df.empty or col not in df.columns:
                return np.nan
            return df[col].iloc[0]

        rows.append({
            "base_signal": base_signal,
            "max_measurement_change_score": sub["change_score"].max(),
            "mean_measurement_change_score": sub["change_score"].mean(),

            "avg_before_mean": get_first_value(avg_rows, "before_mean"),
            "avg_during_mean": get_first_value(avg_rows, "during_mean"),
            "avg_after_mean": get_first_value(avg_rows, "after_mean"),
            "avg_during_minus_before_mean": get_first_value(
                avg_rows,
                "during_minus_before_mean",
            ),
            "avg_during_vs_before_rel_mean": get_first_value(
                avg_rows,
                "during_vs_before_rel_mean",
            ),

            "std_before_mean": get_first_value(std_rows, "before_mean"),
            "std_during_mean": get_first_value(std_rows, "during_mean"),
            "std_after_mean": get_first_value(std_rows, "after_mean"),
            "std_during_minus_before_mean": get_first_value(
                std_rows,
                "during_minus_before_mean",
            ),
            "std_during_vs_before_rel_mean": get_first_value(
                std_rows,
                "during_vs_before_rel_mean",
            ),

            "has_avg": not avg_rows.empty,
            "has_std": not std_rows.empty,
            "has_max": not max_rows.empty,
            "has_min": not min_rows.empty,
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "max_measurement_change_score",
        ascending=False,
    ).reset_index(drop=True)

    return summary


def summarise_by_signal_family(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise changes by sensor family:
    sensor / power / reactive_power / wind_speed.
    """
    def family_name(measurement: str) -> str:
        if measurement.startswith("reactive_power_"):
            return "reactive_power"
        if measurement.startswith("wind_speed_"):
            return "wind_speed"
        if measurement.startswith("power_"):
            return "power"
        if measurement.startswith("sensor_"):
            return "sensor"
        return "other"

    temp = comparison.copy()
    temp["family"] = temp["measurement"].apply(family_name)

    summary = (
        temp.groupby("family")
        .agg(
            n_measurements=("measurement", "count"),
            mean_change_score=("change_score", "mean"),
            max_change_score=("change_score", "max"),
            median_change_score=("change_score", "median"),
        )
        .reset_index()
        .sort_values("max_change_score", ascending=False)
    )

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_measurement_time_series(
    before: pd.DataFrame,
    during: pd.DataFrame,
    after: pd.DataFrame,
    measurement: str,
    output_dir: Path,
) -> None:
    """
    Plot one measurement across before, during and after periods.
    """
    plot_df = pd.concat(
        [
            before.assign(segment="before"),
            during.assign(segment="during"),
            after.assign(segment="after"),
        ],
        ignore_index=True,
    ).copy()

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("id")
    plot_df[measurement] = pd.to_numeric(
        plot_df[measurement],
        errors="coerce",
    )

    plt.figure(figsize=(14, 5))
    plt.plot(
        plot_df["time_stamp"],
        plot_df[measurement],
        linewidth=1,
    )

    if not during.empty:
        plt.axvline(
            during["time_stamp"].iloc[0],
            linestyle="--",
            label="event start",
        )
        plt.axvline(
            during["time_stamp"].iloc[-1],
            linestyle="--",
            label="event end",
        )

    plt.title(f"{measurement}: before / during / after")
    plt.xlabel("Time")
    plt.ylabel(measurement)
    plt.legend()
    plt.tight_layout()

    safe_name = measurement.replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{safe_name}_before_during_after.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_changes_bar(
    comparison: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Save a bar plot for top changed measurements.
    """
    top = comparison.head(top_n).copy()

    if top.empty:
        return

    plt.figure(figsize=(12, 7))
    plt.barh(
        top["measurement"][::-1],
        top["change_score"][::-1],
    )
    plt.xlabel("Change score")
    plt.ylabel("Measurement")
    plt.title(f"Top {top_n} changed SCADA measurements")
    plt.tight_layout()

    output_path = output_dir / f"top{top_n}_changed_measurements_bar.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# Additional global multisensor abnormal-fraction timeline
# ============================================================

def calculate_robust_scale(values: pd.Series) -> float:
    """
    Calculate a robust scale using MAD.

    The factor 1.4826 makes MAD comparable to standard deviation
    for approximately normal data.
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna()

    if numeric.empty:
        return np.nan

    median = float(numeric.median())
    mad = float((numeric - median).abs().median() * 1.4826)

    if not np.isfinite(mad) or mad <= 1e-9:
        mad = float(numeric.std())

    if not np.isfinite(mad) or mad <= 1e-9:
        return np.nan

    return mad


def plot_global_multisensor_abnormal_fraction_timeline(
    before: pd.DataFrame,
    during: pd.DataFrame,
    after: pd.DataFrame,
    measurement_cols: list[str],
    output_dir: Path,
) -> None:
    """
    Create one additional output without changing the original analysis.

    Method:
    1. Use the equal-length Before segment as the reference baseline.
    2. Calculate an absolute robust z-score for every measurement.
    3. Group avg/max/min/std columns into their base signal and retain
       the largest z-score within each base signal at each timestamp.
    4. Calculate the fraction of base signals with robust z >= 8.
    5. Smooth the fraction over six 10-minute rows, approximately one hour.
    6. Use the Before segment's 99.5th percentile as the adaptive threshold.
    """
    combined = pd.concat(
        [
            before.assign(segment="before"),
            during.assign(segment="during"),
            after.assign(segment="after"),
        ],
        ignore_index=True,
    ).copy()

    if combined.empty or before.empty:
        warnings.warn(
            "Global multisensor timeline was not generated because "
            "the combined data or Before reference segment is empty."
        )
        return

    combined = combined.sort_values(
        ["time_stamp", "id"]
    ).reset_index(drop=True)

    measurement_z: dict[str, pd.Series] = {}

    for measurement in measurement_cols:
        baseline = pd.to_numeric(
            before[measurement],
            errors="coerce",
        ).dropna()

        if baseline.empty:
            continue

        baseline_median = float(baseline.median())
        baseline_scale = calculate_robust_scale(baseline)

        if (
            not np.isfinite(baseline_scale)
            or baseline_scale <= 1e-9
        ):
            continue

        values = pd.to_numeric(
            combined[measurement],
            errors="coerce",
        )

        measurement_z[measurement] = (
            (values - baseline_median).abs()
            / baseline_scale
        ).clip(upper=1_000_000.0)

    if not measurement_z:
        warnings.warn(
            "Global multisensor timeline was not generated because "
            "no measurements had a usable Before-segment robust scale."
        )
        return

    measurement_z_df = pd.DataFrame(
        measurement_z,
        index=combined.index,
    )

    # Combine avg/max/min/std measurements into one score per base signal.
    base_signal_groups: dict[str, list[str]] = {}

    for measurement in measurement_z_df.columns:
        base_signal = get_base_signal_name(measurement)
        base_signal_groups.setdefault(
            base_signal,
            [],
        ).append(measurement)

    base_signal_z: dict[str, pd.Series] = {}

    for base_signal, columns in base_signal_groups.items():
        available_columns = [
            column
            for column in columns
            if column in measurement_z_df.columns
        ]

        if available_columns:
            base_signal_z[base_signal] = (
                measurement_z_df[available_columns]
                .max(axis=1)
            )

    if not base_signal_z:
        warnings.warn(
            "Global multisensor timeline was not generated because "
            "no base-signal z-score table could be built."
        )
        return

    base_signal_z_df = pd.DataFrame(
        base_signal_z,
        index=combined.index,
    )

    abnormal_fraction = (
        base_signal_z_df >= GLOBAL_ROBUST_Z_THRESHOLD
    ).mean(axis=1)

    smoothed_abnormal_fraction = abnormal_fraction.rolling(
        window=GLOBAL_ROLLING_POINTS,
        min_periods=1,
        center=True,
    ).mean()

    before_mask = combined["segment"].eq("before")
    before_smoothed = smoothed_abnormal_fraction.loc[
        before_mask
    ].dropna()

    if before_smoothed.empty:
        warnings.warn(
            "Global multisensor timeline was not generated because "
            "the Before-period abnormal-fraction series is empty."
        )
        return

    adaptive_threshold = max(
        GLOBAL_THRESHOLD_FLOOR,
        float(
            before_smoothed.quantile(
                GLOBAL_THRESHOLD_QUANTILE
            )
        ),
    )

    fig, ax = plt.subplots(figsize=(17, 5))

    ax.plot(
        combined["time_stamp"],
        smoothed_abnormal_fraction,
        linewidth=1,
        label="smoothed base-signal abnormal fraction",
    )

    ax.axhline(
        adaptive_threshold,
        linestyle="--",
        label=(
            "adaptive 99.5% threshold = "
            f"{adaptive_threshold:.4f}"
        ),
    )

    ax.axvline(
        pd.Timestamp(EVENT_START),
        linestyle="--",
        label="metadata start",
    )

    ax.axvline(
        pd.Timestamp(EVENT_END),
        linestyle="--",
        label="metadata end",
    )

    ax.set_title(
        "Farm C Event 47: Global Multisensor Abnormal-Fraction "
        "Timeline Using the Equal-Length Pre-Event Baseline "
        "and One-Hour Smoothing"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Base-signal abnormal fraction")
    ax.legend()
    fig.tight_layout()

    output_path = (
        output_dir
        / "global_multisensor_abnormal_fraction_timeline.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Markdown summary
# ============================================================

def write_markdown_summary(
    output_dir: Path,
    segment_info: dict,
    before: pd.DataFrame,
    during: pd.DataFrame,
    after: pd.DataFrame,
    comparison: pd.DataFrame,
    base_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    """
    Write a human-readable markdown summary for report drafting.
    """
    before_start, before_end = get_time_period(before)
    during_start, during_end = get_time_period(during)
    after_start, after_end = get_time_period(after)

    top_measurements = comparison.head(10)
    top_base_signals = base_summary.head(10)

    md_path = output_dir / "event_analysis_summary.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Before–During–After Fault Analysis Summary\n\n")

        f.write("## Event information\n\n")
        f.write(f"- Farm ID: `{FARM_ID}`\n")
        f.write(f"- Asset/Event key: `{ASSET_ID_OR_EVENT_KEY}`\n")
        f.write(f"- Event label: `{EVENT_LABEL}`\n")
        f.write(f"- Event start: `{EVENT_START}`\n")
        f.write(f"- Event start ID: `{EVENT_START_ID}`\n")
        f.write(f"- Event end: `{EVENT_END}`\n")
        f.write(f"- Event end ID: `{EVENT_END_ID}`\n")
        f.write(f"- Event description: `{EVENT_DESCRIPTION}`\n\n")

        f.write("## Segment definition\n\n")
        f.write(
            "| Segment | ID range | Time period | Rows |\n"
            "|---|---:|---|---:|\n"
        )
        f.write(
            f"| Before | {segment_info['before_start_id']}–{segment_info['before_end_id']} "
            f"| {before_start} to {before_end} | {segment_info['before_rows']} |\n"
        )
        f.write(
            f"| During | {segment_info['during_start_id']}–{segment_info['during_end_id']} "
            f"| {during_start} to {during_end} | {segment_info['during_rows']} |\n"
        )
        f.write(
            f"| After | {segment_info['after_start_id']}–{segment_info['after_end_id']} "
            f"| {after_start} to {after_end} | {segment_info['after_rows']} |\n\n"
        )

        f.write("## Top changed measurements\n\n")
        f.write(
            "| Rank | Measurement | Base signal | Type | "
            "Before mean | During mean | During - Before | Relative change | Change score |\n"
            "|---:|---|---|---|---:|---:|---:|---:|---:|\n"
        )

        for i, row in top_measurements.iterrows():
            f.write(
                f"| {i + 1} "
                f"| `{row['measurement']}` "
                f"| `{row['base_signal']}` "
                f"| `{row['stat_type']}` "
                f"| {row['before_mean']:.4g} "
                f"| {row['during_mean']:.4g} "
                f"| {row['during_minus_before_mean']:.4g} "
                f"| {row['during_vs_before_rel_mean']:.4g} "
                f"| {row['change_score']:.4g} |\n"
            )

        f.write("\n## Top changed base signals\n\n")
        f.write(
            "| Rank | Base signal | Max change score | Avg during-before mean change | "
            "Std during-before mean change |\n"
            "|---:|---|---:|---:|---:|\n"
        )

        for i, row in top_base_signals.iterrows():
            f.write(
                f"| {i + 1} "
                f"| `{row['base_signal']}` "
                f"| {row['max_measurement_change_score']:.4g} "
                f"| {row['avg_during_minus_before_mean']:.4g} "
                f"| {row['std_during_minus_before_mean']:.4g} |\n"
            )

        f.write("\n## Signal family summary\n\n")
        f.write(
            "| Family | Number of measurements | Mean change score | Max change score |\n"
            "|---|---:|---:|---:|\n"
        )

        for _, row in family_summary.iterrows():
            f.write(
                f"| `{row['family']}` "
                f"| {int(row['n_measurements'])} "
                f"| {row['mean_change_score']:.4g} "
                f"| {row['max_change_score']:.4g} |\n"
            )

        f.write("\n## Interpretation note\n\n")
        f.write(
            "Large positive `during_minus_before_mean` means the measurement increased "
            "during the fault-related interval compared with the preceding period. "
            "Large negative values mean it decreased. Increased standard deviation "
            "or range suggests more unstable behaviour during the fault-related period. "
            "Because the SCADA variables are anonymised, the analysis should be described "
            "as changes in anonymised SCADA measurements rather than confirmed physical "
            "component-level causes.\n"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("Before–During–After SCADA Sensor Change Analysis")
    print("=" * 100)

    print("\nEvent metadata:")
    print(f"Farm ID: {FARM_ID}")
    print(f"Asset/Event key: {ASSET_ID_OR_EVENT_KEY}")
    print(f"Event label: {EVENT_LABEL}")
    print(f"Event start: {EVENT_START}")
    print(f"Event start ID: {EVENT_START_ID}")
    print(f"Event end: {EVENT_END}")
    print(f"Event end ID: {EVENT_END_ID}")
    print(f"Description: {EVENT_DESCRIPTION}")

    raw_path = find_raw_scada_file(
        data_root=DATA_ROOT,
        farm_id=FARM_ID,
        asset_id_or_event_key=ASSET_ID_OR_EVENT_KEY,
    )

    raw = load_raw_scada(raw_path)

    print("\nRaw SCADA shape:")
    print(raw.shape)

    before, during, after, segment_info = extract_before_during_after_segments(
        raw=raw,
        event_start_id=EVENT_START_ID,
        event_end_id=EVENT_END_ID,
    )

    print("\nSegment information:")
    for key, value in segment_info.items():
        print(f"{key}: {value}")

    if during.empty:
        raise ValueError(
            "During segment is empty. Check EVENT_START_ID and EVENT_END_ID."
        )

    if before.empty:
        warnings.warn(
            "Before segment is empty. The event may be too close to the start of the raw file."
        )

    if after.empty:
        warnings.warn(
            "After segment is empty. The event may be too close to the end of the raw file."
        )

    before_start, before_end = get_time_period(before)
    during_start, during_end = get_time_period(during)
    after_start, after_end = get_time_period(after)

    print("\nTime periods:")
    print(f"Before: {before_start} to {before_end}")
    print(f"During: {during_start} to {during_end}")
    print(f"After:  {after_start} to {after_end}")

    measurement_cols = get_measurement_columns(
        raw,
        mode=MEASUREMENT_MODE,
    )

    print(f"\nMeasurement mode: {MEASUREMENT_MODE}")
    print(f"Number of selected measurement columns: {len(measurement_cols)}")

    if not measurement_cols:
        raise ValueError("No measurement columns selected.")

    selected_cols_path = OUTPUT_DIR / "selected_measurement_columns.csv"
    pd.DataFrame({"measurement": measurement_cols}).to_csv(
        selected_cols_path,
        index=False,
    )

    print("\nCalculating segment statistics...")

    before_stats = calculate_segment_stats(
        before,
        measurement_cols,
        "before",
    )

    during_stats = calculate_segment_stats(
        during,
        measurement_cols,
        "during",
    )

    after_stats = calculate_segment_stats(
        after,
        measurement_cols,
        "after",
    )

    all_stats = pd.concat(
        [before_stats, during_stats, after_stats],
        ignore_index=True,
    )

    all_stats_path = OUTPUT_DIR / "before_during_after_segment_stats.csv"
    all_stats.to_csv(all_stats_path, index=False)

    print("Building comparison table...")

    comparison = build_comparison_table(all_stats)

    comparison_path = OUTPUT_DIR / "sensor_change_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    top20 = comparison.head(20).copy()
    top20_path = OUTPUT_DIR / "top20_changed_measurements.csv"
    top20.to_csv(top20_path, index=False)

    base_summary = build_base_signal_summary(comparison)
    base_summary_path = OUTPUT_DIR / "base_signal_change_summary.csv"
    base_summary.to_csv(base_summary_path, index=False)

    family_summary = summarise_by_signal_family(comparison)
    family_summary_path = OUTPUT_DIR / "signal_family_change_summary.csv"
    family_summary.to_csv(family_summary_path, index=False)

    print("\nTop 20 changed measurements:")
    display_cols = [
        "measurement",
        "base_signal",
        "stat_type",
        "before_mean",
        "during_mean",
        "during_minus_before_mean",
        "during_vs_before_rel_mean",
        "before_std",
        "during_std",
        "during_minus_before_std",
        "change_score",
    ]
    print(top20[display_cols].to_string(index=False))

    print("\nSignal family summary:")
    print(family_summary.to_string(index=False))

    print("\nSaving plots...")

    plot_top_changes_bar(
        comparison,
        OUTPUT_DIR,
        top_n=20,
    )

    # Additional output only; the original Before–During–After
    # analysis and all existing outputs remain unchanged.
    plot_global_multisensor_abnormal_fraction_timeline(
        before=before,
        during=during,
        after=after,
        measurement_cols=measurement_cols,
        output_dir=OUTPUT_DIR,
    )

    for measurement in comparison["measurement"].head(TOP_N_TO_PLOT):
        try:
            plot_measurement_time_series(
                before=before,
                during=during,
                after=after,
                measurement=measurement,
                output_dir=OUTPUT_DIR,
            )
        except Exception as exc:
            print(f"Could not plot {measurement}: {exc}")

    write_markdown_summary(
        output_dir=OUTPUT_DIR,
        segment_info=segment_info,
        before=before,
        during=during,
        after=after,
        comparison=comparison,
        base_summary=base_summary,
        family_summary=family_summary,
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nMain output files:")
    print(f"- {selected_cols_path.name}")
    print(f"- {all_stats_path.name}")
    print(f"- {comparison_path.name}")
    print(f"- {top20_path.name}")
    print(f"- {base_summary_path.name}")
    print(f"- {family_summary_path.name}")
    print("- event_analysis_summary.md")
    print("- top20_changed_measurements_bar.png")
    print("- global_multisensor_abnormal_fraction_timeline.png")
    print(f"- top {TOP_N_TO_PLOT} measurement time-series plots")

    print("\nInterpretation:")
    print(
        "Use top20_changed_measurements.csv to identify which anonymised SCADA "
        "measurements changed most strongly during the fault-related interval. "
        "Positive during_minus_before_mean means the signal increased during "
        "the event; negative means it decreased. Increased std/range suggests "
        "more unstable behaviour."
    )


if __name__ == "__main__":
    main()