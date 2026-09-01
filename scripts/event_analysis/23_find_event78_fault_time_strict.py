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

FARM_ID = "C"
EVENT_ID = "78"

RAW_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Wind Farm C"
    / "datasets"
    / "78.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "event_analysis"
    / "farm_C_asset_78_fault_time_detection_strict_after_to_aug06"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Event 78 metadata
# ============================================================

EVENT_START_TIME = "2022-07-29 13:00:00"
EVENT_END_TIME = "2022-07-31 14:30:00"

EVENT_START_ID = 52560
EVENT_END_ID = 52857

AFTER_END_TIME = "2022-08-06 23:50:00"

EVENT_DESCRIPTION = (
    "P20_Grounding role brake disc + "
    "P20_cover-lightning-main-cabinet-hub"
)


# ============================================================
# Strict transition-detection settings
# ============================================================

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
TOP_CONTRIBUTOR_BAR_COUNT = 20

# Segment detection is only applied to these periods.
# before_reference is used only as baseline, not as detected fault output.
DETECTION_PERIODS = [
    "during_metadata_interval",
    "after_period",
]


# ============================================================
# Basic helper functions
# ============================================================

def load_raw_scada(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw SCADA file not found:\n{path}")

    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"time_stamp", "asset_id", "id"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df["id"] = pd.to_numeric(df["id"], errors="coerce")

    df = df.dropna(subset=["time_stamp", "id"]).copy()
    df["id"] = df["id"].astype(int)

    df = (
        df.sort_values(["time_stamp", "id"])
        .reset_index(drop=True)
    )

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

    selected_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if not col.startswith(allowed_prefixes):
            continue

        if not col.endswith(allowed_suffixes):
            continue

        numeric_col = pd.to_numeric(df[col], errors="coerce")

        if numeric_col.notna().sum() > 0:
            selected_cols.append(col)

    return selected_cols


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


def get_time_by_id(df: pd.DataFrame, target_id: int):
    value = df.loc[df["id"] == target_id, "time_stamp"]

    if value.empty:
        return None

    return value.iloc[0]


# ============================================================
# Build before / during / after periods
# ============================================================

def build_analysis_periods(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Event 78 period definition:

    - before_reference:
      An equal-length period immediately before the metadata interval.

    - during_metadata_interval:
      The official metadata-labelled anomaly interval.

    - after_period:
      From the first timestamp after metadata end until 2022-08-06 23:50.

    - analysis_window:
      before + during + extended after period.
    """

    event_len = EVENT_END_ID - EVENT_START_ID + 1

    before_start_id = EVENT_START_ID - event_len
    before_end_id = EVENT_START_ID - 1

    before = df[
        (df["id"] >= before_start_id)
        & (df["id"] <= before_end_id)
    ].copy()

    during = df[
        (df["id"] >= EVENT_START_ID)
        & (df["id"] <= EVENT_END_ID)
    ].copy()

    metadata_end_time = pd.to_datetime(EVENT_END_TIME)
    extended_after_end_time = pd.to_datetime(AFTER_END_TIME)

    after = df[
        (df["time_stamp"] > metadata_end_time)
        & (df["time_stamp"] <= extended_after_end_time)
    ].copy()

    analysis_window = pd.concat(
        [before, during, after],
        ignore_index=True,
    )

    analysis_window = (
        analysis_window
        .drop_duplicates(subset=["time_stamp", "id"])
        .sort_values(["time_stamp", "id"])
        .reset_index(drop=True)
    )

    return {
        "before_reference": before.reset_index(drop=True),
        "during_metadata_interval": during.reset_index(drop=True),
        "after_period": after.reset_index(drop=True),
        "analysis_window": analysis_window,
    }

def assign_period_label(score_df: pd.DataFrame) -> pd.DataFrame:
    df = score_df.copy()

    conditions = [
        df["id"] < EVENT_START_ID,
        (df["id"] >= EVENT_START_ID) & (df["id"] <= EVENT_END_ID),
        df["id"] > EVENT_END_ID,
    ]

    labels = [
        "before_reference",
        "during_metadata_interval",
        "after_period",
    ]

    df["period"] = np.select(
        conditions,
        labels,
        default="unknown",
    )

    return df


# ============================================================
# Reference statistics
# ============================================================

def calculate_reference_stats(
    reference_df: pd.DataFrame,
    measurement_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for col in measurement_cols:
        x = pd.to_numeric(reference_df[col], errors="coerce").dropna()

        if x.empty:
            row = {
                "measurement": col,
                "base_signal": get_base_signal(col),
                "stat_type": get_stat_type(col),
                "family": get_family(col),
                "reference_mean": np.nan,
                "reference_std": np.nan,
                "reference_median": np.nan,
                "reference_mad": np.nan,
                "reference_min": np.nan,
                "reference_max": np.nan,
            }
        else:
            median = float(x.median())
            mad = float((x - median).abs().median())

            row = {
                "measurement": col,
                "base_signal": get_base_signal(col),
                "stat_type": get_stat_type(col),
                "family": get_family(col),
                "reference_mean": float(x.mean()),
                "reference_std": float(x.std()),
                "reference_median": median,
                "reference_mad": mad,
                "reference_min": float(x.min()),
                "reference_max": float(x.max()),
            }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Z-score and abnormal-fraction timeline
# ============================================================

def build_z_score_table(
    window_df: pd.DataFrame,
    reference_stats: pd.DataFrame,
    measurement_cols: list[str],
) -> pd.DataFrame:
    z_records = []

    reference_lookup = reference_stats.set_index("measurement").to_dict(orient="index")

    for col in measurement_cols:
        x = pd.to_numeric(window_df[col], errors="coerce")

        ref = reference_lookup[col]

        ref_median = ref["reference_median"]
        ref_mad = ref["reference_mad"]
        ref_mean = ref["reference_mean"]
        ref_std = ref["reference_std"]

        if pd.notna(ref_median) and pd.notna(ref_mad) and ref_mad >= 1e-9:
            robust_scale = 1.4826 * ref_mad
            abs_z = ((x - ref_median).abs() / robust_scale).fillna(0)

        elif pd.notna(ref_mean) and pd.notna(ref_std) and ref_std >= 1e-9:
            abs_z = ((x - ref_mean).abs() / ref_std).fillna(0)

        else:
            abs_z = pd.Series(
                np.zeros(len(window_df)),
                index=window_df.index,
            )

        z_records.append(abs_z.rename(col))

    z_df = pd.concat(z_records, axis=1)

    return z_df


def build_transition_score_timeline(
    window_df: pd.DataFrame,
    z_df: pd.DataFrame,
) -> pd.DataFrame:
    score_df = window_df[["time_stamp", "asset_id", "id"]].copy()

    abnormal_bool = z_df > Z_THRESHOLD

    abnormal_count = abnormal_bool.sum(axis=1)
    abnormal_fraction = abnormal_count / z_df.shape[1]

    sorted_abs_z = np.sort(z_df.values, axis=1)

    if z_df.shape[1] >= 10:
        mean_top10_abs_z = sorted_abs_z[:, -10:].mean(axis=1)
    else:
        mean_top10_abs_z = sorted_abs_z.mean(axis=1)

    score_df["n_measurements"] = z_df.shape[1]
    score_df["abnormal_count"] = abnormal_count
    score_df["abnormal_fraction"] = abnormal_fraction

    score_df["rolling_abnormal_fraction"] = (
        score_df["abnormal_fraction"]
        .rolling(window=ROLLING_POINTS, min_periods=1)
        .mean()
    )

    score_df["mean_top10_abs_z"] = mean_top10_abs_z
    score_df["max_abs_z"] = z_df.max(axis=1)

    # Used only for ranking candidate points.
    # Segment detection itself is based on rolling_abnormal_fraction.
    score_df["transition_score"] = (
        score_df["rolling_abnormal_fraction"]
        * (1.0 + np.log1p(score_df["mean_top10_abs_z"]))
    )

    score_df["raw_transition_flag"] = (
        score_df["rolling_abnormal_fraction"] >= ABNORMAL_FRACTION_THRESHOLD
    )

    score_df = assign_period_label(score_df)

    # Only detect fault segments inside Event 78 interval and after period.
    # before_reference is used only as baseline.
    score_df["event_related_transition_flag"] = (
        score_df["raw_transition_flag"]
        & score_df["period"].isin(DETECTION_PERIODS)
    )

    return score_df


# ============================================================
# Segment detection
# ============================================================

def find_raw_segments(flag: pd.Series) -> list[dict]:
    flag_values = flag.fillna(False).astype(bool).to_numpy()

    segments = []
    in_segment = False
    start_pos = None

    for pos, value in enumerate(flag_values):
        if value and not in_segment:
            in_segment = True
            start_pos = pos

        if in_segment and not value:
            end_pos = pos - 1
            segments.append({
                "start_pos": start_pos,
                "end_pos": end_pos,
            })
            in_segment = False
            start_pos = None

    if in_segment:
        segments.append({
            "start_pos": start_pos,
            "end_pos": len(flag_values) - 1,
        })

    return segments


def merge_close_segments(
    segments: list[dict],
    max_gap_points: int,
) -> list[dict]:
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        current = merged[-1]

        gap = seg["start_pos"] - current["end_pos"] - 1

        if gap <= max_gap_points:
            current["end_pos"] = seg["end_pos"]
        else:
            merged.append(seg.copy())

    return merged


def build_segment_table(
    score_df: pd.DataFrame,
    segments: list[dict],
    min_segment_points: int,
) -> pd.DataFrame:
    rows = []

    for seg in segments:
        start_pos = seg["start_pos"]
        end_pos = seg["end_pos"]

        n_points = end_pos - start_pos + 1

        if n_points < min_segment_points:
            continue

        seg_df = score_df.iloc[start_pos:end_pos + 1].copy()

        peak_idx = seg_df["transition_score"].idxmax()
        peak_row = score_df.loc[peak_idx]

        rows.append({
            "segment_id": len(rows) + 1,
            "start_time": seg_df["time_stamp"].iloc[0],
            "end_time": seg_df["time_stamp"].iloc[-1],
            "start_id": int(seg_df["id"].iloc[0]),
            "end_id": int(seg_df["id"].iloc[-1]),
            "start_period": seg_df["period"].iloc[0],
            "end_period": seg_df["period"].iloc[-1],
            "n_points": n_points,
            "duration_minutes": n_points * 10,
            "peak_time": peak_row["time_stamp"],
            "peak_id": int(peak_row["id"]),
            "peak_period": peak_row["period"],
            "peak_transition_score": float(peak_row["transition_score"]),
            "peak_abnormal_fraction": float(peak_row["abnormal_fraction"]),
            "peak_rolling_abnormal_fraction": float(peak_row["rolling_abnormal_fraction"]),
            "peak_abnormal_count": int(peak_row["abnormal_count"]),
            "peak_max_abs_z": float(peak_row["max_abs_z"]),
            "peak_mean_top10_abs_z": float(peak_row["mean_top10_abs_z"]),
        })

    return pd.DataFrame(rows)


def detect_transition_segments(score_df: pd.DataFrame) -> pd.DataFrame:
    raw_segments = find_raw_segments(score_df["event_related_transition_flag"])

    merged_segments = merge_close_segments(
        raw_segments,
        max_gap_points=MAX_GAP_POINTS,
    )

    segment_table = build_segment_table(
        score_df,
        merged_segments,
        min_segment_points=MIN_SEGMENT_POINTS,
    )

    return segment_table


# ============================================================
# Top candidate points and contributors
# ============================================================

def select_top_transition_candidates(score_df: pd.DataFrame) -> pd.DataFrame:
    candidates = score_df.copy()

    # Only rank candidates inside Event 78 interval and after period.
    candidates = candidates[candidates["period"].isin(DETECTION_PERIODS)].copy()

    candidates = candidates.sort_values(
        "transition_score",
        ascending=False,
    ).head(TOP_TRANSITION_CANDIDATES)

    candidates = candidates.reset_index().rename(columns={"index": "row_position"})

    return candidates


def get_top_contributors_for_candidates(
    candidates: pd.DataFrame,
    z_df: pd.DataFrame,
    window_df: pd.DataFrame,
    reference_stats: pd.DataFrame,
) -> pd.DataFrame:
    reference_lookup = reference_stats.set_index("measurement")

    records = []

    for candidate_rank, candidate in candidates.iterrows():
        row_position = int(candidate["row_position"])

        row_z = z_df.iloc[row_position].sort_values(ascending=False)
        top_z = row_z.head(TOP_CONTRIBUTORS_PER_CANDIDATE)

        for contributor_rank, (measurement, abs_z) in enumerate(top_z.items(), start=1):
            value_at_candidate = pd.to_numeric(
                window_df[measurement],
                errors="coerce",
            ).iloc[row_position]

            ref_row = reference_lookup.loc[measurement]

            records.append({
                "candidate_rank": candidate_rank + 1,
                "candidate_time": candidate["time_stamp"],
                "candidate_id": int(candidate["id"]),
                "candidate_period": candidate["period"],
                "candidate_transition_score": float(candidate["transition_score"]),
                "candidate_abnormal_fraction": float(candidate["abnormal_fraction"]),
                "candidate_rolling_abnormal_fraction": float(candidate["rolling_abnormal_fraction"]),
                "candidate_abnormal_count": int(candidate["abnormal_count"]),
                "contributor_rank": contributor_rank,
                "measurement": measurement,
                "base_signal": get_base_signal(measurement),
                "stat_type": get_stat_type(measurement),
                "family": get_family(measurement),
                "value_at_candidate": value_at_candidate,
                "reference_mean": ref_row["reference_mean"],
                "reference_median": ref_row["reference_median"],
                "reference_std": ref_row["reference_std"],
                "reference_mad": ref_row["reference_mad"],
                "abs_z_at_candidate": float(abs_z),
            })

    return pd.DataFrame(records)


def get_unique_top_contributor_measurements(
    contributors: pd.DataFrame,
    top_n: int,
) -> list[str]:
    if contributors.empty:
        return []

    ranked = (
        contributors.groupby("measurement")
        .agg(max_abs_z=("abs_z_at_candidate", "max"))
        .reset_index()
        .sort_values("max_abs_z", ascending=False)
    )

    return ranked["measurement"].head(top_n).tolist()


# ============================================================
# Plotting
# ============================================================

def plot_transition_timeline(
    score_df: pd.DataFrame,
    segments: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(16, 6))

    plt.plot(
        score_df["time_stamp"],
        score_df["abnormal_fraction"],
        linewidth=1.0,
        label="abnormal_fraction",
    )

    plt.plot(
        score_df["time_stamp"],
        score_df["rolling_abnormal_fraction"],
        linewidth=1.5,
        label=f"rolling_abnormal_fraction ({ROLLING_POINTS} points)",
    )

    plt.axhline(
        ABNORMAL_FRACTION_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=f"threshold = {ABNORMAL_FRACTION_THRESHOLD}",
    )

    metadata_start_time = get_time_by_id(score_df, EVENT_START_ID)
    metadata_end_time = get_time_by_id(score_df, EVENT_END_ID)

    if metadata_start_time is not None:
        plt.axvline(
            metadata_start_time,
            linestyle=":",
            linewidth=2,
            label="metadata start",
        )

    if metadata_end_time is not None:
        plt.axvline(
            metadata_end_time,
            linestyle=":",
            linewidth=2,
            label="metadata end",
        )

    if not segments.empty:
        for _, row in segments.iterrows():
            plt.axvspan(
                row["start_time"],
                row["end_time"],
                alpha=0.2,
            )

    plt.title("Event 78 strict multi-sensor transition detection")
    plt.xlabel("Time")
    plt.ylabel("Fraction of abnormal measurements")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "event78_strict_transition_timeline.png",
        dpi=200,
    )

    plt.close()


def plot_transition_score(
    score_df: pd.DataFrame,
    segments: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(16, 5))

    plt.plot(
        score_df["time_stamp"],
        score_df["transition_score"],
        linewidth=1.2,
        label="transition_score",
    )

    metadata_start_time = get_time_by_id(score_df, EVENT_START_ID)
    metadata_end_time = get_time_by_id(score_df, EVENT_END_ID)

    if metadata_start_time is not None:
        plt.axvline(
            metadata_start_time,
            linestyle=":",
            linewidth=2,
            label="metadata start",
        )

    if metadata_end_time is not None:
        plt.axvline(
            metadata_end_time,
            linestyle=":",
            linewidth=2,
            label="metadata end",
        )

    if not segments.empty:
        for _, row in segments.iterrows():
            plt.axvspan(
                row["start_time"],
                row["end_time"],
                alpha=0.2,
            )

    plt.title("Event 78 transition score")
    plt.xlabel("Time")
    plt.ylabel("Transition score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "event78_transition_score.png",
        dpi=200,
    )

    plt.close()


def plot_contributor_signals(
    window_df: pd.DataFrame,
    measurements: list[str],
    output_dir: Path,
) -> None:
    metadata_start_time = get_time_by_id(window_df, EVENT_START_ID)
    metadata_end_time = get_time_by_id(window_df, EVENT_END_ID)

    for measurement in measurements:
        if measurement not in window_df.columns:
            continue

        y = pd.to_numeric(window_df[measurement], errors="coerce")

        plt.figure(figsize=(16, 5))

        plt.plot(
            window_df["time_stamp"],
            y,
            linewidth=1.1,
            label=measurement,
        )

        if metadata_start_time is not None:
            plt.axvline(
                metadata_start_time,
                linestyle="--",
                linewidth=2,
                label="metadata start",
            )

        if metadata_end_time is not None:
            plt.axvline(
                metadata_end_time,
                linestyle="--",
                linewidth=2,
                label="metadata end",
            )

        plt.title(f"{measurement}: Event 78 strict transition contributor")
        plt.xlabel("Time")
        plt.ylabel(measurement)
        plt.legend()
        plt.tight_layout()

        safe_name = measurement.replace("/", "_").replace("\\", "_")

        plt.savefig(
            output_dir / f"{safe_name}_event78_strict_contributor.png",
            dpi=200,
        )

        plt.close()


def plot_top_contributors_bar(
    contributors: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Plot the strongest candidate's top contributing measurements.

    The bar length is the absolute robust z-score at the strongest
    transition candidate, relative to the before-reference period.
    """

    if contributors.empty:
        print(
            "Top contributor bar chart skipped: "
            "contributors table is empty."
        )
        return

    strongest_candidate = contributors[
        contributors["candidate_rank"] == 1
    ].copy()

    if strongest_candidate.empty:
        print(
            "Top contributor bar chart skipped: "
            "candidate rank 1 was not found."
        )
        return

    top = (
        strongest_candidate
        .sort_values("abs_z_at_candidate", ascending=False)
        .head(top_n)
        .copy()
    )

    if top.empty:
        print("Top contributor bar chart skipped: no rows to plot.")
        return

    top.to_csv(
        output_dir
        / f"event78_top{top_n}_contributors_strongest_candidate.csv",
        index=False,
    )

    plot_df = top.sort_values(
        "abs_z_at_candidate",
        ascending=True,
    )

    candidate_time = pd.to_datetime(
        strongest_candidate["candidate_time"].iloc[0]
    )

    plt.figure(figsize=(13, 8))

    plt.barh(
        plot_df["measurement"],
        plot_df["abs_z_at_candidate"],
    )

    plt.xlabel("Absolute robust z-score vs before reference")
    plt.ylabel("Measurement")
    plt.title(
        f"Event 78 Top {top_n} transition contributors\n"
        f"Strongest candidate: {candidate_time:%Y-%m-%d %H:%M}"
    )
    plt.tight_layout()

    plt.savefig(
        output_dir
        / f"event78_top{top_n}_contributors_strongest_candidate.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Markdown report
# ============================================================

def write_markdown_report(
    periods: dict[str, pd.DataFrame],
    score_df: pd.DataFrame,
    segments: pd.DataFrame,
    candidates: pd.DataFrame,
    contributors: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path = output_dir / "event78_strict_fault_time_summary.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Event 78 Strict Fault-Time Detection Summary\n\n")

        f.write("## Event information\n\n")
        f.write(f"- Farm ID: `{FARM_ID}`\n")
        f.write(f"- Event / asset key: `{EVENT_ID}`\n")
        f.write("- Label: `anomaly`\n")
        f.write(f"- Metadata start: `{EVENT_START_TIME}` / id `{EVENT_START_ID}`\n")
        f.write(f"- Metadata end: `{EVENT_END_TIME}` / id `{EVENT_END_ID}`\n")
        f.write(f"- Description: `{EVENT_DESCRIPTION}`\n\n")

        f.write("## Detection rule\n\n")
        f.write(f"- Measurement mode: `{MEASUREMENT_MODE}`\n")
        f.write(f"- Individual measurement z-score threshold: `{Z_THRESHOLD}`\n")
        f.write(f"- Abnormal fraction threshold: `{ABNORMAL_FRACTION_THRESHOLD}`\n")
        f.write(f"- Rolling points: `{ROLLING_POINTS}` 10-minute rows\n")
        f.write(f"- Minimum segment length: `{MIN_SEGMENT_POINTS}` 10-minute rows\n")
        f.write(f"- Maximum gap merged: `{MAX_GAP_POINTS}` 10-minute rows\n\n")

        f.write(
            "A timestamp is treated as system-level abnormal only when at least "
            "10% of all measurement columns are abnormal after rolling smoothing. "
            "A detected fault-like segment must last for at least 12 points, or about "
            "two hours. Short gaps of up to 3 points, or about 30 minutes, are merged. "
            "The before period is used as reference only; detected segments are reported "
            "only inside the metadata interval and after period.\n\n"
        )

        f.write("## Periods\n\n")
        f.write("| Period | Start | End | Rows |\n")
        f.write("|---|---|---|---:|\n")

        for name, df in periods.items():
            if df.empty:
                f.write(f"| `{name}` | NA | NA | 0 |\n")
            else:
                f.write(
                    f"| `{name}` | {df['time_stamp'].min()} "
                    f"| {df['time_stamp'].max()} "
                    f"| {len(df)} |\n"
                )

        f.write("\n## Detected transition segments\n\n")

        if segments.empty:
            f.write("No confirmed strict multi-sensor transition segment was detected.\n\n")
        else:
            f.write(
                "| Segment | Start time | End time | Period | Duration min | Peak time | "
                "Peak abnormal fraction | Peak abnormal count | Peak score |\n"
            )
            f.write("|---:|---|---|---|---:|---|---:|---:|---:|\n")

            for _, row in segments.iterrows():
                f.write(
                    f"| {int(row['segment_id'])} "
                    f"| {row['start_time']} "
                    f"| {row['end_time']} "
                    f"| `{row['start_period']}` "
                    f"| {int(row['duration_minutes'])} "
                    f"| {row['peak_time']} "
                    f"| {row['peak_abnormal_fraction']:.4f} "
                    f"| {int(row['peak_abnormal_count'])} "
                    f"| {row['peak_transition_score']:.4f} |\n"
                )

        f.write("\n## Top transition candidates\n\n")

        if candidates.empty:
            f.write("No transition candidate was selected.\n\n")
        else:
            f.write(
                "| Rank | Time | ID | Period | Transition score | "
                "Abnormal fraction | Rolling abnormal fraction | Abnormal count |\n"
            )
            f.write("|---:|---|---:|---|---:|---:|---:|---:|\n")

            for i, row in candidates.head(10).iterrows():
                f.write(
                    f"| {i + 1} "
                    f"| {row['time_stamp']} "
                    f"| {int(row['id'])} "
                    f"| `{row['period']}` "
                    f"| {row['transition_score']:.4f} "
                    f"| {row['abnormal_fraction']:.4f} "
                    f"| {row['rolling_abnormal_fraction']:.4f} "
                    f"| {int(row['abnormal_count'])} |\n"
                )

        f.write("\n## Top contributors for strongest candidate\n\n")

        if contributors.empty:
            f.write("No contributor was found.\n\n")
        else:
            first_candidate = contributors[
                contributors["candidate_rank"] == 1
            ].copy()

            f.write(
                "| Rank | Measurement | Family | Type | Value | "
                "Reference median | Abs z-score |\n"
            )
            f.write("|---:|---|---|---|---:|---:|---:|\n")

            for _, row in first_candidate.head(20).iterrows():
                f.write(
                    f"| {int(row['contributor_rank'])} "
                    f"| `{row['measurement']}` "
                    f"| `{row['family']}` "
                    f"| `{row['stat_type']}` "
                    f"| {row['value_at_candidate']:.4g} "
                    f"| {row['reference_median']:.4g} "
                    f"| {row['abs_z_at_candidate']:.4g} |\n"
                )

        f.write("\n## Interpretation note\n\n")
        f.write(
            "The detected start and end times are estimated from SCADA signal transitions. "
            "They should be interpreted as data-driven fault-related state transition points "
            "at 10-minute resolution, rather than exact maintenance-log timestamps. "
            "This stricter method focuses on persistent multi-sensor abnormal behaviour and "
            "filters out isolated short sensor spikes.\n"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("Event 78 Strict Multi-Sensor Fault-Time Detection")
    print("=" * 100)

    print("\nEvent metadata:")
    print(f"Farm ID: {FARM_ID}")
    print(f"Event ID / asset key: {EVENT_ID}")
    print("Label: anomaly")
    print(f"Event start: {EVENT_START_TIME}")
    print(f"Event start ID: {EVENT_START_ID}")
    print(f"Event end: {EVENT_END_TIME}")
    print(f"Event end ID: {EVENT_END_ID}")
    print(f"Description: {EVENT_DESCRIPTION}")

    print("\nDetection settings:")
    print(f"MEASUREMENT_MODE = {MEASUREMENT_MODE}")
    print(f"Z_THRESHOLD = {Z_THRESHOLD}")
    print(f"ABNORMAL_FRACTION_THRESHOLD = {ABNORMAL_FRACTION_THRESHOLD}")
    print(f"ROLLING_POINTS = {ROLLING_POINTS}")
    print(f"MIN_SEGMENT_POINTS = {MIN_SEGMENT_POINTS}")
    print(f"MAX_GAP_POINTS = {MAX_GAP_POINTS}")

    print("\nLoading raw SCADA file:")
    print(RAW_FILE_PATH)

    raw = load_raw_scada(RAW_FILE_PATH)

    print("\nRaw SCADA shape:")
    print(raw.shape)

    requested_after_end = pd.to_datetime(AFTER_END_TIME)
    raw_start_time = raw["time_stamp"].min()
    raw_end_time = raw["time_stamp"].max()

    print("\nRaw file time coverage:")
    print(f"Raw start:           {raw_start_time}")
    print(f"Raw end:             {raw_end_time}")
    print(f"Requested after end: {requested_after_end}")

    if raw_end_time < requested_after_end:
        print(
            "\n[WARNING] 78.csv does not contain data up to the "
            f"requested after end. Plots can only extend to {raw_end_time}."
        )

    periods = build_analysis_periods(raw)

    reference_df = periods["before_reference"]
    after_df = periods["after_period"]
    window_df = periods["analysis_window"]

    print("\nExtended after-period check:")

    if after_df.empty:
        print("after_period is empty.")
    else:
        actual_after_start = after_df["time_stamp"].min()
        actual_after_end = after_df["time_stamp"].max()

        print(f"After start:         {actual_after_start}")
        print(f"After end:           {actual_after_end}")
        print(f"After rows:          {len(after_df)}")
        print(f"Analysis window end: {window_df['time_stamp'].max()}")

        if actual_after_end < requested_after_end:
            print(
                "\n[WARNING] The actual after period ends earlier than "
                "requested. The raw file may not contain later usable rows, "
                "or timestamps may be missing."
            )

    if reference_df.empty:
        raise ValueError("before_reference period is empty. Check Event 78 metadata IDs.")

    if window_df.empty:
        raise ValueError("analysis_window is empty.")

    print("\nAnalysis periods:")

    for name, df in periods.items():
        if df.empty:
            print(f"{name}: 0 rows")
        else:
            print(
                f"{name}: {len(df)} rows, "
                f"{df['time_stamp'].min()} to {df['time_stamp'].max()}"
            )

    measurement_cols = get_measurement_columns(raw, mode=MEASUREMENT_MODE)

    if not measurement_cols:
        raise ValueError("No measurement columns selected.")

    print(f"\nSelected measurement columns: {len(measurement_cols)}")

    pd.DataFrame({"measurement": measurement_cols}).to_csv(
        OUTPUT_DIR / "selected_measurement_columns.csv",
        index=False,
    )

    print("\nCalculating reference statistics...")

    reference_stats = calculate_reference_stats(
        reference_df=reference_df,
        measurement_cols=measurement_cols,
    )

    reference_stats.to_csv(
        OUTPUT_DIR / "event78_reference_stats.csv",
        index=False,
    )

    print("Building z-score table...")

    z_df = build_z_score_table(
        window_df=window_df,
        reference_stats=reference_stats,
        measurement_cols=measurement_cols,
    )

    z_df.to_csv(
        OUTPUT_DIR / "event78_abs_z_scores_by_sensor.csv",
        index=False,
    )

    print("Building transition score timeline...")

    score_df = build_transition_score_timeline(
        window_df=window_df,
        z_df=z_df,
    )

    score_df.to_csv(
        OUTPUT_DIR / "event78_strict_transition_timeline.csv",
        index=False,
    )

    print("Detecting transition segments...")

    segments = detect_transition_segments(score_df)

    segments.to_csv(
        OUTPUT_DIR / "event78_strict_detected_segments.csv",
        index=False,
    )

    print("Selecting top transition candidates...")

    candidates = select_top_transition_candidates(score_df)

    candidates.to_csv(
        OUTPUT_DIR / "event78_top_transition_candidates.csv",
        index=False,
    )

    print("Finding top contributors for candidates...")

    contributors = get_top_contributors_for_candidates(
        candidates=candidates,
        z_df=z_df,
        window_df=window_df,
        reference_stats=reference_stats,
    )

    contributors.to_csv(
        OUTPUT_DIR / "event78_top_contributors_by_candidate.csv",
        index=False,
    )

    print("Plotting Top contributor bar chart...")

    plot_top_contributors_bar(
        contributors=contributors,
        output_dir=OUTPUT_DIR,
        top_n=TOP_CONTRIBUTOR_BAR_COUNT,
    )

    print("\nPlotting timelines and contributor signals...")

    plot_transition_timeline(
        score_df=score_df,
        segments=segments,
        output_dir=OUTPUT_DIR,
    )

    plot_transition_score(
        score_df=score_df,
        segments=segments,
        output_dir=OUTPUT_DIR,
    )

    top_measurements = get_unique_top_contributor_measurements(
        contributors,
        top_n=PLOT_TOP_CONTRIBUTOR_SIGNALS,
    )

    plot_contributor_signals(
        window_df=window_df,
        measurements=top_measurements,
        output_dir=OUTPUT_DIR,
    )

    write_markdown_report(
        periods=periods,
        score_df=score_df,
        segments=segments,
        candidates=candidates,
        contributors=contributors,
        output_dir=OUTPUT_DIR,
    )

    print("\nDetected strict transition segments:")

    if segments.empty:
        print("No strict multi-sensor transition segment detected.")
    else:
        print(segments.to_string(index=False))

    print("\nTop transition candidates:")
    display_candidate_cols = [
        "time_stamp",
        "id",
        "period",
        "transition_score",
        "abnormal_fraction",
        "rolling_abnormal_fraction",
        "abnormal_count",
        "mean_top10_abs_z",
        "max_abs_z",
    ]

    if candidates.empty:
        print("No candidates found.")
    else:
        print(candidates[display_candidate_cols].head(10).to_string(index=False))

    print("\nTop contributors for strongest candidate:")

    if contributors.empty:
        print("No contributors found.")
    else:
        first_candidate = contributors[
            contributors["candidate_rank"] == 1
        ].copy()

        display_contributor_cols = [
            "measurement",
            "family",
            "stat_type",
            "value_at_candidate",
            "reference_median",
            "reference_mean",
            "abs_z_at_candidate",
        ]

        print(first_candidate[display_contributor_cols].head(20).to_string(index=False))

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nMain output files:")
    print("- selected_measurement_columns.csv")
    print("- event78_reference_stats.csv")
    print("- event78_abs_z_scores_by_sensor.csv")
    print("- event78_strict_transition_timeline.csv")
    print("- event78_strict_detected_segments.csv")
    print("- event78_top_transition_candidates.csv")
    print("- event78_top_contributors_by_candidate.csv")
    print(
        f"- event78_top{TOP_CONTRIBUTOR_BAR_COUNT}_contributors_"
        "strongest_candidate.csv"
    )
    print(
        f"- event78_top{TOP_CONTRIBUTOR_BAR_COUNT}_contributors_"
        "strongest_candidate.png"
    )
    print("- event78_strict_fault_time_summary.md")
    print("- event78_strict_transition_timeline.png")
    print("- event78_transition_score.png")
    print("- top contributor signal plots")

    print("\nDone.")


if __name__ == "__main__":
    main()