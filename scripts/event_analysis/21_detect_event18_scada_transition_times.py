from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Wind Farm C"
    / "datasets"
    / "18.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "event_analysis"
    / "farm_C_asset_18_scada_inferred_transition_times"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Full window for plotting and context
ANALYSIS_START_TIME = "2025-09-11 00:00:00"
ANALYSIS_END_TIME = "2025-09-18 23:50:00"

# Baseline reference day
BASELINE_START_TIME = "2025-09-11 00:00:00"
BASELINE_END_TIME = "2025-09-11 23:50:00"

# Start automatic segment detection after baseline day
DETECTION_START_TIME = "2025-09-12 00:00:00"

# Use all measurement types: avg, max, min, std
MEASUREMENT_MODE = "all"

# Stricter thresholds than the previous version
Z_THRESHOLD = 12.0

# 10% of 952 measurements = about 95 measurement columns
ABNORMAL_FRACTION_THRESHOLD = 0.10

# 6 points = about 1 hour because SCADA interval is 10 minutes
ROLLING_POINTS = 6

# Minimum abnormal segment length
MIN_SEGMENT_POINTS = 12

# Merge gaps shorter than 30 minutes
MAX_GAP_POINTS = 3


# ============================================================
# Load data
# ============================================================

def load_raw_scada(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw SCADA file not found:\n{path}")

    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "time_stamp" not in df.columns:
        raise ValueError("Missing column: time_stamp")

    if "id" not in df.columns:
        raise ValueError("Missing column: id")

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df["id"] = pd.to_numeric(df["id"], errors="coerce")

    df = df.dropna(subset=["time_stamp", "id"]).copy()
    df["id"] = df["id"].astype(int)
    df = df.sort_values("time_stamp").reset_index(drop=True)

    return df


def get_measurement_columns(df: pd.DataFrame, mode: str = "all") -> list[str]:
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
        raise ValueError("MEASUREMENT_MODE must be avg_only, avg_std or all")

    cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if not col.startswith(allowed_prefixes):
            continue

        if not col.endswith(allowed_suffixes):
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        if numeric.notna().sum() > 0:
            cols.append(col)

    return cols


# ============================================================
# Robust anomaly score
# ============================================================

def robust_scale(series: pd.Series) -> float:
    """
    Robust scale based on MAD.
    If MAD is zero, fall back to std.
    If std is also zero, use a small scale value.
    """
    x = pd.to_numeric(series, errors="coerce").dropna()

    if x.empty:
        return np.nan

    median = x.median()
    mad = np.median(np.abs(x - median))

    scale = 1.4826 * mad

    if pd.isna(scale) or scale < 1e-9:
        scale = x.std()

    if pd.isna(scale) or scale < 1e-9:
        scale = max(abs(median) * 0.01, 1e-3)

    return float(scale)


def compute_scada_anomaly_score(
    window: pd.DataFrame,
    baseline: pd.DataFrame,
    measurement_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each timestamp:
    - calculate robust z-score against Sep11 baseline
    - calculate fraction of abnormal measurement columns
    - calculate mean of top 20 z-scores
    """

    baseline_median = {}
    baseline_scale = {}

    for col in measurement_cols:
        b = pd.to_numeric(baseline[col], errors="coerce")
        baseline_median[col] = b.median()
        baseline_scale[col] = robust_scale(b)

    baseline_table = pd.DataFrame({
        "measurement": measurement_cols,
        "baseline_median": [baseline_median[c] for c in measurement_cols],
        "baseline_scale": [baseline_scale[c] for c in measurement_cols],
    })

    values = window[measurement_cols].apply(pd.to_numeric, errors="coerce")

    medians = pd.Series(baseline_median)
    scales = pd.Series(baseline_scale).replace(0, np.nan)

    z = (values - medians).abs().divide(scales, axis=1)
    z = z.replace([np.inf, -np.inf], np.nan)

    abnormal_mask = z > Z_THRESHOLD
    abnormal_fraction = abnormal_mask.mean(axis=1)

    top20_mean_z = z.apply(
        lambda row: row.dropna().nlargest(20).mean()
        if row.dropna().shape[0] > 0 else np.nan,
        axis=1,
    )

    median_z = z.median(axis=1, skipna=True)
    p95_z = z.quantile(0.95, axis=1)

    score_df = window[["time_stamp", "id"]].copy()
    score_df["abnormal_fraction"] = abnormal_fraction
    score_df["top20_mean_z"] = top20_mean_z
    score_df["median_z"] = median_z
    score_df["p95_z"] = p95_z

    score_df["raw_state_score"] = (
        score_df["abnormal_fraction"]
        + 0.01 * score_df["top20_mean_z"].fillna(0)
    )

    score_df["smooth_state_score"] = (
        score_df["raw_state_score"]
        .rolling(ROLLING_POINTS, min_periods=1, center=True)
        .median()
    )

    score_df["smooth_abnormal_fraction"] = (
        score_df["abnormal_fraction"]
        .rolling(ROLLING_POINTS, min_periods=1, center=True)
        .median()
    )

    return score_df, baseline_table


# ============================================================
# Segment detection
# ============================================================

def merge_small_gaps(mask: pd.Series, max_gap_points: int) -> pd.Series:
    """
    If abnormal segments are separated by a very short normal gap,
    merge them.

    Important:
    .copy() is needed here, otherwise numpy may return a read-only array.
    """
    arr = mask.astype(bool).to_numpy().copy()
    n = len(arr)

    i = 0

    while i < n:
        if arr[i]:
            i += 1
            continue

        gap_start = i

        while i < n and not arr[i]:
            i += 1

        gap_end = i - 1

        left_abnormal = gap_start > 0 and arr[gap_start - 1]
        right_abnormal = gap_end < n - 1 and arr[gap_end + 1]

        gap_len = gap_end - gap_start + 1

        if left_abnormal and right_abnormal and gap_len <= max_gap_points:
            arr[gap_start:gap_end + 1] = True

    return pd.Series(arr, index=mask.index)


def find_segments_from_mask(
    df: pd.DataFrame,
    mask: pd.Series,
    label: str,
    min_points: int = MIN_SEGMENT_POINTS,
) -> pd.DataFrame:
    """
    Convert True/False mask into start/end time segments.
    """

    mask = mask.astype(bool).reset_index(drop=True)
    df = df.reset_index(drop=True)

    segments = []
    n = len(mask)

    i = 0

    while i < n:
        if not mask.iloc[i]:
            i += 1
            continue

        start_i = i

        while i < n and mask.iloc[i]:
            i += 1

        end_i = i - 1
        length = end_i - start_i + 1

        if length >= min_points:
            start_time = df.loc[start_i, "time_stamp"]
            end_time = df.loc[end_i, "time_stamp"]

            segment = {
                "label": label,
                "start_time": start_time,
                "end_time": end_time,
                "start_id": int(df.loc[start_i, "id"]),
                "end_id": int(df.loc[end_i, "id"]),
                "n_points": int(length),
                "duration_hours": float(length * 10 / 60),
            }

            if "smooth_state_score" in df.columns:
                segment["mean_smooth_state_score"] = float(
                    df.loc[start_i:end_i, "smooth_state_score"].mean()
                )
            else:
                segment["mean_smooth_state_score"] = np.nan

            if "abnormal_fraction" in df.columns:
                segment["mean_abnormal_fraction"] = float(
                    df.loc[start_i:end_i, "abnormal_fraction"].mean()
                )
            else:
                segment["mean_abnormal_fraction"] = np.nan

            segments.append(segment)

    return pd.DataFrame(segments)


# ============================================================
# Standstill / control-state detection
# ============================================================

def add_standstill_control_signal(window: pd.DataFrame) -> pd.DataFrame:
    """
    Detect the state switch seen in Event 18:
    - sensor_100-105 avg: around 0 -> around 90
    - sensor_143 avg: around 50 -> around 2

    These are SCADA-based signal rules, not metadata times.
    """

    out = window[["time_stamp", "id"]].copy()

    high_state_cols = [
        col for col in [
            "sensor_100_avg",
            "sensor_101_avg",
            "sensor_102_avg",
            "sensor_103_avg",
            "sensor_104_avg",
            "sensor_105_avg",
        ]
        if col in window.columns
    ]

    if high_state_cols:
        high_state_values = window[high_state_cols].apply(pd.to_numeric, errors="coerce")
        out["sensor_100_105_avg_mean"] = high_state_values.mean(axis=1)

        # This threshold is based on observed transition:
        # reference state around 0, standstill/control state around 90.
        out["sensor_100_105_high_state"] = out["sensor_100_105_avg_mean"] > 50
    else:
        out["sensor_100_105_avg_mean"] = np.nan
        out["sensor_100_105_high_state"] = False

    if "sensor_143_avg" in window.columns:
        out["sensor_143_avg"] = pd.to_numeric(window["sensor_143_avg"], errors="coerce")

        # This threshold is based on observed transition:
        # reference state around 50, standstill/control state around 2.
        out["sensor_143_low_state"] = out["sensor_143_avg"] < 25
    else:
        out["sensor_143_avg"] = np.nan
        out["sensor_143_low_state"] = False

    out["standstill_control_state_mask"] = (
        out["sensor_100_105_high_state"]
        | out["sensor_143_low_state"]
    )

    out["standstill_control_state_mask"] = merge_small_gaps(
        out["standstill_control_state_mask"],
        max_gap_points=MAX_GAP_POINTS,
    )

    return out


# ============================================================
# Transition candidates
# ============================================================

def find_transition_candidates(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find sharp changes in the smooth state score.
    """

    out = score_df.copy()

    out["score_diff"] = out["smooth_state_score"].diff().abs()
    out["abnormal_fraction_diff"] = out["smooth_abnormal_fraction"].diff().abs()

    candidates = out.sort_values("score_diff", ascending=False).head(30).copy()

    return candidates[
        [
            "time_stamp",
            "id",
            "smooth_state_score",
            "smooth_abnormal_fraction",
            "abnormal_fraction",
            "top20_mean_z",
            "score_diff",
            "abnormal_fraction_diff",
        ]
    ]


# ============================================================
# Plotting
# ============================================================

def plot_state_score(score_df: pd.DataFrame, segments: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(16, 6))

    plt.plot(
        score_df["time_stamp"],
        score_df["smooth_state_score"],
        linewidth=1.5,
        label="SCADA state score",
    )

    plt.plot(
        score_df["time_stamp"],
        score_df["smooth_abnormal_fraction"],
        linewidth=1,
        label="abnormal sensor fraction",
    )

    # Draw detected segments
    if segments is not None and not segments.empty:
        for _, row in segments.iterrows():
            plt.axvspan(
                row["start_time"],
                row["end_time"],
                alpha=0.15,
            )

            plt.axvline(
                row["start_time"],
                linestyle="--",
                linewidth=1,
            )

            plt.axvline(
                row["end_time"],
                linestyle="--",
                linewidth=1,
            )

    # Reference labels
    plt.axvline(
        pd.to_datetime("2025-09-12 00:00:00"),
        linestyle=":",
        linewidth=1.5,
        label="metadata interval begins",
    )

    plt.axvline(
        pd.to_datetime("2025-09-16 12:00:00"),
        linestyle=":",
        linewidth=1.5,
        label="16th afternoon mentioned",
    )

    plt.axvline(
        pd.to_datetime("2025-09-17 00:00:00"),
        linestyle=":",
        linewidth=1.5,
        label="17th onwards mentioned",
    )

    plt.title("Event 18 SCADA-inferred state changes, Sep 11 to Sep 18")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "event18_scada_inferred_state_score.png", dpi=200)
    plt.close()


def plot_control_state(
    control_df: pd.DataFrame,
    control_segments: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(16, 6))

    if "sensor_100_105_avg_mean" in control_df.columns:
        plt.plot(
            control_df["time_stamp"],
            control_df["sensor_100_105_avg_mean"],
            linewidth=1.5,
            label="mean(sensor_100_avg to sensor_105_avg)",
        )

    if "sensor_143_avg" in control_df.columns:
        plt.plot(
            control_df["time_stamp"],
            control_df["sensor_143_avg"],
            linewidth=1.5,
            label="sensor_143_avg",
        )

    if control_segments is not None and not control_segments.empty:
        for _, row in control_segments.iterrows():
            plt.axvspan(
                row["start_time"],
                row["end_time"],
                alpha=0.15,
            )

            plt.axvline(
                row["start_time"],
                linestyle="--",
                linewidth=1,
            )

            plt.axvline(
                row["end_time"],
                linestyle="--",
                linewidth=1,
            )

    plt.axvline(
        pd.to_datetime("2025-09-16 12:00:00"),
        linestyle=":",
        linewidth=1.5,
        label="16th afternoon mentioned",
    )

    plt.axvline(
        pd.to_datetime("2025-09-17 00:00:00"),
        linestyle=":",
        linewidth=1.5,
        label="17th onwards mentioned",
    )

    plt.title("Event 18 SCADA-inferred standstill/control state")
    plt.xlabel("Time")
    plt.ylabel("Signal value")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "event18_control_state_signals.png", dpi=200)
    plt.close()


# ============================================================
# Markdown summary
# ============================================================

def write_summary(
    score_segments: pd.DataFrame,
    control_segments: pd.DataFrame,
    transition_candidates: pd.DataFrame,
    output_dir: Path,
) -> None:
    out_path = output_dir / "event18_scada_inferred_transition_summary.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Event 18 SCADA-inferred transition time analysis\n\n")

        f.write("## Method\n\n")
        f.write(
            "Sep 11 was used as the SCADA reference baseline. "
            "For each 10-minute SCADA record from Sep 11 to Sep 18, "
            "a robust deviation score was calculated against the Sep 11 baseline. "
            "Automatic abnormal segment detection was only applied from Sep 12 onwards, "
            "so that the reference day itself would not be treated as a detected abnormal segment.\n\n"
        )

        f.write("## Multisensor abnormal segments\n\n")
        if score_segments.empty:
            f.write("No multisensor abnormal segment was detected under the current thresholds.\n\n")
        else:
            f.write(score_segments.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Standstill/control-state segments\n\n")
        if control_segments.empty:
            f.write("No standstill/control-state segment was detected.\n\n")
        else:
            f.write(control_segments.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Top transition candidate timestamps\n\n")
        if transition_candidates.empty:
            f.write("No transition candidates found.\n\n")
        else:
            f.write(transition_candidates.head(15).to_markdown(index=False))
            f.write("\n\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("Event 18: SCADA-inferred transition time detection")
    print("=" * 100)

    print("\nLoading raw SCADA file:")
    print(RAW_FILE_PATH)

    raw = load_raw_scada(RAW_FILE_PATH)

    print("\nRaw SCADA shape:")
    print(raw.shape)

    analysis_start = pd.to_datetime(ANALYSIS_START_TIME)
    analysis_end = pd.to_datetime(ANALYSIS_END_TIME)

    window = raw[
        (raw["time_stamp"] >= analysis_start)
        & (raw["time_stamp"] <= analysis_end)
    ].copy()

    if window.empty:
        raise ValueError("Analysis window is empty.")

    baseline = raw[
        (raw["time_stamp"] >= pd.to_datetime(BASELINE_START_TIME))
        & (raw["time_stamp"] <= pd.to_datetime(BASELINE_END_TIME))
    ].copy()

    if baseline.empty:
        raise ValueError("Baseline window is empty.")

    print("\nAnalysis window:")
    print(f"{window['time_stamp'].min()} to {window['time_stamp'].max()}")
    print(f"Rows: {len(window)}")

    print("\nBaseline window:")
    print(f"{baseline['time_stamp'].min()} to {baseline['time_stamp'].max()}")
    print(f"Rows: {len(baseline)}")

    measurement_cols = get_measurement_columns(raw, mode=MEASUREMENT_MODE)

    print(f"\nMeasurement mode: {MEASUREMENT_MODE}")
    print(f"Selected measurement columns: {len(measurement_cols)}")

    if not measurement_cols:
        raise ValueError("No measurement columns selected.")

    print("\nComputing SCADA anomaly score from baseline...")

    score_df, baseline_table = compute_scada_anomaly_score(
        window=window,
        baseline=baseline,
        measurement_cols=measurement_cols,
    )

    baseline_table.to_csv(
        OUTPUT_DIR / "baseline_signal_statistics_sep11.csv",
        index=False,
    )

    score_df.to_csv(
        OUTPUT_DIR / "event18_scada_state_score_timeline.csv",
        index=False,
    )

    # Only detect abnormal segments after Sep 12.
    detection_score_df = score_df[
        score_df["time_stamp"] >= pd.to_datetime(DETECTION_START_TIME)
    ].copy()

    abnormal_mask = (
        detection_score_df["smooth_abnormal_fraction"]
        >= ABNORMAL_FRACTION_THRESHOLD
    )

    abnormal_mask = merge_small_gaps(
        abnormal_mask,
        max_gap_points=MAX_GAP_POINTS,
    )

    score_segments = find_segments_from_mask(
        df=detection_score_df,
        mask=abnormal_mask,
        label="multisensor_abnormal_state_from_scada",
        min_points=MIN_SEGMENT_POINTS,
    )

    print("\nSCADA-inferred multisensor abnormal segments:")
    if score_segments.empty:
        print("No segments found. Try lowering thresholds if needed.")
    else:
        print(score_segments.to_string(index=False))

    # Detect standstill/control-state from specific SCADA signal transitions.
    print("\nDetecting standstill/control-state from sensor_100-105 and sensor_143...")

    control_df = add_standstill_control_signal(window)

    detection_control_df = control_df[
        control_df["time_stamp"] >= pd.to_datetime(DETECTION_START_TIME)
    ].copy()

    control_segments = find_segments_from_mask(
        df=detection_control_df,
        mask=detection_control_df["standstill_control_state_mask"],
        label="standstill_or_control_state_from_scada_signals",
        min_points=MIN_SEGMENT_POINTS,
    )

    print("\nSCADA-inferred standstill/control-state segments:")
    if control_segments.empty:
        print("No control-state segments found.")
    else:
        print(control_segments.to_string(index=False))

    transition_candidates = find_transition_candidates(score_df)

    # Also remove Sep11 candidates from automatic candidate table
    transition_candidates_after_detection_start = transition_candidates[
        transition_candidates["time_stamp"] >= pd.to_datetime(DETECTION_START_TIME)
    ].copy()

    print("\nTop transition candidate timestamps based on state-score jumps:")
    if transition_candidates_after_detection_start.empty:
        print("No transition candidates found after detection start.")
    else:
        print(
            transition_candidates_after_detection_start
            .head(15)
            .to_string(index=False)
        )

    # Save outputs
    score_segments.to_csv(
        OUTPUT_DIR / "scada_inferred_multisensor_abnormal_segments.csv",
        index=False,
    )

    control_segments.to_csv(
        OUTPUT_DIR / "scada_inferred_standstill_control_segments.csv",
        index=False,
    )

    transition_candidates.to_csv(
        OUTPUT_DIR / "scada_transition_candidate_timestamps_all.csv",
        index=False,
    )

    transition_candidates_after_detection_start.to_csv(
        OUTPUT_DIR / "scada_transition_candidate_timestamps_after_sep12.csv",
        index=False,
    )

    control_df.to_csv(
        OUTPUT_DIR / "event18_control_state_signal_timeline.csv",
        index=False,
    )

    all_segments = pd.concat(
        [score_segments, control_segments],
        ignore_index=True,
    )

    if not all_segments.empty:
        all_segments = all_segments.sort_values("start_time").reset_index(drop=True)

    all_segments.to_csv(
        OUTPUT_DIR / "event18_all_scada_inferred_segments.csv",
        index=False,
    )

    # Plots
    plot_state_score(score_df, all_segments, OUTPUT_DIR)
    plot_control_state(control_df, control_segments, OUTPUT_DIR)

    write_summary(
        score_segments=score_segments,
        control_segments=control_segments,
        transition_candidates=transition_candidates_after_detection_start,
        output_dir=OUTPUT_DIR,
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nMain files:")
    print("- baseline_signal_statistics_sep11.csv")
    print("- event18_scada_state_score_timeline.csv")
    print("- scada_inferred_multisensor_abnormal_segments.csv")
    print("- scada_inferred_standstill_control_segments.csv")
    print("- scada_transition_candidate_timestamps_after_sep12.csv")
    print("- event18_all_scada_inferred_segments.csv")
    print("- event18_control_state_signal_timeline.csv")
    print("- event18_scada_inferred_state_score.png")
    print("- event18_control_state_signals.png")
    print("- event18_scada_inferred_transition_summary.md")

    print("\nDone.")


if __name__ == "__main__":
    main()