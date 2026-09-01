from __future__ import annotations

from pathlib import Path
import re
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
    / "55.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "event_analysis"
    / "farm_C_asset_55_scada_inferred_transition_times"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_ID = "55"
FARM_ID = "C"

EVENT_DESCRIPTION = (
    "Harting plug Nacelle/HUB damaged + "
    "NCR20_HUB: Wiring blade control system"
)

# Metadata interval: only used for reference lines and broad window selection.
# It is NOT used to decide the detected transition times.
METADATA_START_ID = 52848
METADATA_END_ID = 55320
METADATA_START_TIME = "2018-10-29 11:30:00"
METADATA_END_TIME = "2018-11-15 15:30:00"

# Use one event-length period before and one event-length period after as broad analysis window.
EVENT_LENGTH = METADATA_END_ID - METADATA_START_ID + 1

ANALYSIS_START_ID = METADATA_START_ID - EVENT_LENGTH
ANALYSIS_END_ID = METADATA_END_ID + EVENT_LENGTH

# Baseline is selected from the early part of the analysis window.
# This avoids directly using the metadata start time as the detection boundary.
BASELINE_DAYS = 7
MEASUREMENT_MODE = "all"
Z_THRESHOLD = 12.0
ABNORMAL_FRACTION_THRESHOLD = 0.10
ROLLING_POINTS = 6

MIN_SEGMENT_POINTS = 12
MAX_GAP_POINTS = 3

# Output controls
TOP_TRANSITION_CANDIDATES = 30
TOP_CONTRIBUTORS_PER_CANDIDATE = 20
PLOT_TOP_CONTRIBUTOR_SIGNALS = 20


# ============================================================
# Basic helpers
# ============================================================

def load_raw_scada(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw SCADA file not found:\n{path}")

    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"time_stamp", "id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df["id"] = pd.to_numeric(df["id"], errors="coerce")

    df = df.dropna(subset=["time_stamp", "id"]).copy()
    df["id"] = df["id"].astype(int)

    df = df.sort_values("id").reset_index(drop=True)

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


def get_base_signal(col: str) -> str:
    return re.sub(r"_(avg|max|min|std)$", "", col)


def get_stat_type(col: str) -> str:
    match = re.search(r"_(avg|max|min|std)$", col)
    if match:
        return match.group(1)
    return "unknown"


def get_family(col: str) -> str:
    if col.startswith("reactive_power_"):
        return "reactive_power"
    if col.startswith("wind_speed_"):
        return "wind_speed"
    if col.startswith("power_"):
        return "power"
    if col.startswith("sensor_"):
        return "sensor"
    return "other"


# ============================================================
# Robust deviation score
# ============================================================

def robust_scale(series: pd.Series) -> float:
    """
    Robust scale using MAD.
    If MAD is zero, fall back to standard deviation.
    If std is also zero, use a small value.
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


def compute_scada_state_score(
    window: pd.DataFrame,
    baseline: pd.DataFrame,
    measurement_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each timestamp:
    - compute robust z-score against baseline
    - compute fraction of abnormal measurement columns
    - compute top20 mean z-score
    """

    baseline_median = {}
    baseline_scale = {}

    for col in measurement_cols:
        b = pd.to_numeric(baseline[col], errors="coerce")
        baseline_median[col] = b.median()
        baseline_scale[col] = robust_scale(b)

    baseline_table = pd.DataFrame({
        "measurement": measurement_cols,
        "base_signal": [get_base_signal(c) for c in measurement_cols],
        "stat_type": [get_stat_type(c) for c in measurement_cols],
        "family": [get_family(c) for c in measurement_cols],
        "baseline_median": [baseline_median[c] for c in measurement_cols],
        "baseline_scale": [baseline_scale[c] for c in measurement_cols],
    })

    values = window[measurement_cols].apply(pd.to_numeric, errors="coerce")

    medians = pd.Series(baseline_median)
    scales = pd.Series(baseline_scale).replace(0, np.nan)

    z = (values - medians).abs().divide(scales, axis=1)
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.reset_index(drop=True)

    abnormal_mask = z > Z_THRESHOLD
    abnormal_fraction = abnormal_mask.mean(axis=1)

    top20_mean_z = z.apply(
        lambda row: row.dropna().nlargest(20).mean()
        if row.dropna().shape[0] > 0 else np.nan,
        axis=1,
    )

    score_df = window[["time_stamp", "id"]].reset_index(drop=True).copy()
    score_df["row_index"] = score_df.index
    score_df["abnormal_fraction"] = abnormal_fraction
    score_df["top20_mean_z"] = top20_mean_z
    score_df["median_z"] = z.median(axis=1, skipna=True)
    score_df["p95_z"] = z.quantile(0.95, axis=1)

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

    return score_df, baseline_table, z


# ============================================================
# Segment detection
# ============================================================

def merge_small_gaps(mask: pd.Series, max_gap_points: int) -> pd.Series:
    """
    Merge short normal gaps between abnormal periods.
    .copy() avoids read-only numpy assignment error.
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
            segment = {
                "label": label,
                "start_time": df.loc[start_i, "time_stamp"],
                "end_time": df.loc[end_i, "time_stamp"],
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
# Transition candidates and contributors
# ============================================================

def find_transition_candidates(score_df: pd.DataFrame) -> pd.DataFrame:
    out = score_df.copy()

    out["score_diff"] = out["smooth_state_score"].diff().abs()
    out["abnormal_fraction_diff"] = out["smooth_abnormal_fraction"].diff().abs()

    candidates = out.sort_values("score_diff", ascending=False).head(
        TOP_TRANSITION_CANDIDATES
    ).copy()

    return candidates[
        [
            "row_index",
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


def build_top_contributors_table(
    candidates: pd.DataFrame,
    z_df: pd.DataFrame,
    window: pd.DataFrame,
    measurement_cols: list[str],
) -> pd.DataFrame:
    """
    For each transition candidate timestamp, identify the top contributing sensors.
    """
    rows = []

    window_reset = window.reset_index(drop=True)

    for _, cand in candidates.iterrows():
        idx = int(cand["row_index"])

        if idx < 0 or idx >= len(z_df):
            continue

        z_row = z_df.iloc[idx].dropna().sort_values(ascending=False)

        for rank, measurement in enumerate(
            z_row.head(TOP_CONTRIBUTORS_PER_CANDIDATE).index,
            start=1,
        ):
            raw_value = pd.to_numeric(
                pd.Series([window_reset.loc[idx, measurement]]),
                errors="coerce",
            ).iloc[0]

            rows.append({
                "candidate_time": cand["time_stamp"],
                "candidate_id": cand["id"],
                "candidate_score_diff": cand["score_diff"],
                "rank": rank,
                "measurement": measurement,
                "base_signal": get_base_signal(measurement),
                "stat_type": get_stat_type(measurement),
                "family": get_family(measurement),
                "raw_value_at_candidate": raw_value,
                "robust_z_score": z_row.loc[measurement],
            })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_state_score(
    score_df: pd.DataFrame,
    segments: pd.DataFrame,
    output_dir: Path,
) -> None:
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
        linewidth=1.0,
        label="abnormal measurement fraction",
    )

    if segments is not None and not segments.empty:
        for _, row in segments.iterrows():
            plt.axvspan(row["start_time"], row["end_time"], alpha=0.15)
            plt.axvline(row["start_time"], linestyle="--", linewidth=1)
            plt.axvline(row["end_time"], linestyle="--", linewidth=1)

    # Metadata is shown only as reference lines.
    plt.axvline(
        pd.to_datetime(METADATA_START_TIME),
        linestyle=":",
        linewidth=1.5,
        label="metadata start, reference only",
    )

    plt.axvline(
        pd.to_datetime(METADATA_END_TIME),
        linestyle=":",
        linewidth=1.5,
        label="metadata end, reference only",
    )

    plt.title("Event 55 SCADA-inferred state changes")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "event55_scada_inferred_state_score.png", dpi=200)
    plt.close()


def plot_measurement(
    window: pd.DataFrame,
    measurement: str,
    output_dir: Path,
    segments: pd.DataFrame | None = None,
) -> None:
    if measurement not in window.columns:
        return

    plot_df = window[["time_stamp", "id", measurement]].copy()
    plot_df[measurement] = pd.to_numeric(plot_df[measurement], errors="coerce")

    plt.figure(figsize=(16, 5))

    plt.plot(
        plot_df["time_stamp"],
        plot_df[measurement],
        linewidth=1,
        label=measurement,
    )

    if segments is not None and not segments.empty:
        for _, row in segments.iterrows():
            plt.axvspan(row["start_time"], row["end_time"], alpha=0.10)

    plt.axvline(
        pd.to_datetime(METADATA_START_TIME),
        linestyle=":",
        linewidth=1.5,
        label="metadata start, reference only",
    )

    plt.axvline(
        pd.to_datetime(METADATA_END_TIME),
        linestyle=":",
        linewidth=1.5,
        label="metadata end, reference only",
    )

    plt.title(f"{measurement}: SCADA signal over analysis window")
    plt.xlabel("Time")
    plt.ylabel(measurement)
    plt.legend()
    plt.tight_layout()

    safe_name = measurement.replace("/", "_").replace("\\", "_")
    plt.savefig(output_dir / f"{safe_name}_event55_scada_transition_plot.png", dpi=200)
    plt.close()


def plot_transition_candidate_markers(
    score_df: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(16, 6))

    plt.plot(
        score_df["time_stamp"],
        score_df["smooth_state_score"],
        linewidth=1.5,
        label="SCADA state score",
    )

    for _, row in candidates.head(10).iterrows():
        plt.axvline(
            row["time_stamp"],
            linestyle="--",
            linewidth=1,
        )

    plt.axvline(
        pd.to_datetime(METADATA_START_TIME),
        linestyle=":",
        linewidth=1.5,
        label="metadata start, reference only",
    )

    plt.axvline(
        pd.to_datetime(METADATA_END_TIME),
        linestyle=":",
        linewidth=1.5,
        label="metadata end, reference only",
    )

    plt.title("Event 55 top SCADA transition candidate timestamps")
    plt.xlabel("Time")
    plt.ylabel("SCADA state score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "event55_transition_candidate_markers.png", dpi=200)
    plt.close()


# ============================================================
# Summary
# ============================================================

def write_summary(
    raw: pd.DataFrame,
    window: pd.DataFrame,
    baseline: pd.DataFrame,
    segments: pd.DataFrame,
    candidates: pd.DataFrame,
    contributors: pd.DataFrame,
    output_dir: Path,
) -> None:
    path = output_dir / "event55_scada_inferred_transition_summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("Event 55 SCADA-inferred transition time analysis\n")
        f.write("=" * 80 + "\n\n")

        f.write("Event information\n")
        f.write("-" * 80 + "\n")
        f.write(f"Farm ID: {FARM_ID}\n")
        f.write(f"Event / asset key: {EVENT_ID}\n")
        f.write(f"Description: {EVENT_DESCRIPTION}\n")
        f.write(f"Metadata start: {METADATA_START_TIME} / id {METADATA_START_ID}\n")
        f.write(f"Metadata end: {METADATA_END_TIME} / id {METADATA_END_ID}\n")
        f.write("\n")

        f.write("Important note\n")
        f.write("-" * 80 + "\n")
        f.write(
            "Metadata timestamps are used only for reference and broad window selection. "
            "The detected segments below are inferred from SCADA signal deviation "
            "against the baseline period.\n\n"
        )

        f.write("Raw and window information\n")
        f.write("-" * 80 + "\n")
        f.write(f"Raw shape: {raw.shape}\n")
        f.write(
            f"Analysis window: {window['time_stamp'].min()} "
            f"to {window['time_stamp'].max()}\n"
        )
        f.write(f"Analysis rows: {len(window)}\n")
        f.write(
            f"Baseline window: {baseline['time_stamp'].min()} "
            f"to {baseline['time_stamp'].max()}\n"
        )
        f.write(f"Baseline rows: {len(baseline)}\n\n")

        f.write("SCADA-inferred abnormal segments\n")
        f.write("-" * 80 + "\n")
        if segments.empty:
            f.write("No abnormal segment detected under current thresholds.\n")
        else:
            f.write(segments.to_string(index=False))
        f.write("\n\n")

        f.write("Top transition candidate timestamps\n")
        f.write("-" * 80 + "\n")
        if candidates.empty:
            f.write("No transition candidates found.\n")
        else:
            f.write(candidates.head(15).to_string(index=False))
        f.write("\n\n")

        f.write("Top contributors for transition candidates\n")
        f.write("-" * 80 + "\n")
        if contributors.empty:
            f.write("No contributors found.\n")
        else:
            f.write(contributors.head(50).to_string(index=False))
        f.write("\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("Event 55: SCADA-inferred transition time detection")
    print("=" * 100)

    print("\nLoading raw SCADA file:")
    print(RAW_FILE_PATH)

    raw = load_raw_scada(RAW_FILE_PATH)

    print("\nRaw SCADA shape:")
    print(raw.shape)

    raw_min_id = int(raw["id"].min())
    raw_max_id = int(raw["id"].max())

    analysis_start_id = max(ANALYSIS_START_ID, raw_min_id)
    analysis_end_id = min(ANALYSIS_END_ID, raw_max_id)

    window = raw[
        (raw["id"] >= analysis_start_id)
        & (raw["id"] <= analysis_end_id)
    ].copy()

    if window.empty:
        raise ValueError("Analysis window is empty. Check IDs.")

    print("\nAnalysis window:")
    print(f"ID: {analysis_start_id} to {analysis_end_id}")
    print(f"Time: {window['time_stamp'].min()} to {window['time_stamp'].max()}")
    print(f"Rows: {len(window)}")

    # Baseline = first BASELINE_DAYS from the analysis window.
    baseline_start_time = window["time_stamp"].min()
    baseline_end_time = baseline_start_time + pd.Timedelta(days=BASELINE_DAYS)

    baseline = window[
        (window["time_stamp"] >= baseline_start_time)
        & (window["time_stamp"] < baseline_end_time)
    ].copy()

    if baseline.empty:
        raise ValueError("Baseline window is empty.")

    detection_start_time = baseline["time_stamp"].max() + pd.Timedelta(minutes=10)

    print("\nBaseline window:")
    print(f"{baseline['time_stamp'].min()} to {baseline['time_stamp'].max()}")
    print(f"Rows: {len(baseline)}")

    print("\nDetection starts after baseline:")
    print(detection_start_time)

    measurement_cols = get_measurement_columns(raw, mode=MEASUREMENT_MODE)

    print(f"\nMeasurement mode: {MEASUREMENT_MODE}")
    print(f"Selected measurement columns: {len(measurement_cols)}")

    if not measurement_cols:
        raise ValueError("No measurement columns selected.")

    pd.DataFrame({"measurement": measurement_cols}).to_csv(
        OUTPUT_DIR / "selected_measurement_columns_event55.csv",
        index=False,
    )

    print("\nComputing SCADA state score from baseline...")

    score_df, baseline_table, z_df = compute_scada_state_score(
        window=window,
        baseline=baseline,
        measurement_cols=measurement_cols,
    )

    baseline_table.to_csv(
        OUTPUT_DIR / "baseline_signal_statistics_event55.csv",
        index=False,
    )

    score_df.to_csv(
        OUTPUT_DIR / "event55_scada_state_score_timeline.csv",
        index=False,
    )

    # Detect only after baseline period.
    detection_score_df = score_df[
        score_df["time_stamp"] >= detection_start_time
    ].copy()

    abnormal_mask = (
        detection_score_df["smooth_abnormal_fraction"]
        >= ABNORMAL_FRACTION_THRESHOLD
    )

    abnormal_mask = merge_small_gaps(
        abnormal_mask,
        max_gap_points=MAX_GAP_POINTS,
    )

    segments = find_segments_from_mask(
        df=detection_score_df,
        mask=abnormal_mask,
        label="multisensor_abnormal_state_from_scada",
        min_points=MIN_SEGMENT_POINTS,
    )

    print("\nSCADA-inferred abnormal segments:")
    if segments.empty:
        print("No segments found. Try lowering thresholds.")
    else:
        print(segments.to_string(index=False))

    # Transition candidates
    candidates_all = find_transition_candidates(score_df)

    candidates = candidates_all[
        candidates_all["time_stamp"] >= detection_start_time
    ].copy()

    candidates.to_csv(
        OUTPUT_DIR / "scada_transition_candidate_timestamps_event55.csv",
        index=False,
    )

    print("\nTop transition candidate timestamps:")
    if candidates.empty:
        print("No candidates found.")
    else:
        print(candidates.head(15).to_string(index=False))

    # Top contributors
    contributors = build_top_contributors_table(
        candidates=candidates.head(15),
        z_df=z_df,
        window=window,
        measurement_cols=measurement_cols,
    )

    contributors.to_csv(
        OUTPUT_DIR / "top_contributors_for_transition_candidates_event55.csv",
        index=False,
    )

    segments.to_csv(
        OUTPUT_DIR / "scada_inferred_abnormal_segments_event55.csv",
        index=False,
    )

    segments.to_csv(
        OUTPUT_DIR / "event55_all_scada_inferred_segments.csv",
        index=False,
    )

    # Plots
    print("\nSaving plots...")

    plot_state_score(score_df, segments, OUTPUT_DIR)
    plot_transition_candidate_markers(score_df, candidates, OUTPUT_DIR)

    # Plot unique top contributor signals
    if not contributors.empty:
        top_measurements = (
            contributors["measurement"]
            .drop_duplicates()
            .head(PLOT_TOP_CONTRIBUTOR_SIGNALS)
            .tolist()
        )

        for measurement in top_measurements:
            try:
                plot_measurement(window, measurement, OUTPUT_DIR, segments)
            except Exception as exc:
                print(f"Could not plot {measurement}: {exc}")

    write_summary(
        raw=raw,
        window=window,
        baseline=baseline,
        segments=segments,
        candidates=candidates,
        contributors=contributors,
        output_dir=OUTPUT_DIR,
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nMain output files:")
    print("- event55_all_scada_inferred_segments.csv")
    print("- scada_inferred_abnormal_segments_event55.csv")
    print("- scada_transition_candidate_timestamps_event55.csv")
    print("- top_contributors_for_transition_candidates_event55.csv")
    print("- event55_scada_state_score_timeline.csv")
    print("- event55_scada_inferred_state_score.png")
    print("- event55_transition_candidate_markers.png")
    print("- event55_scada_inferred_transition_summary.txt")
    print("- top contributor signal plots")

    print("\nDone.")


if __name__ == "__main__":
    main()