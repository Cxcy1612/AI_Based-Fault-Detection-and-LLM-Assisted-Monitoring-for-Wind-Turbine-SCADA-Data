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
    / "18.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "event_analysis"
    / "farm_C_asset_18_sep11_to_sep18_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_ID = "18"
FARM_ID = "C"

EVENT_DESCRIPTION = (
    "We had some failures (störung 24VAC Versorgung Rotorbremse) "
    "on the 16th in the afternoon. From 17th onwards a longer standstill "
    "where we don't know the root cause to."
)

# Metadata-labelled interval
EVENT_START_TIME = "2025-09-12 00:00:00"
EVENT_END_TIME = "2025-09-15 23:50:00"
EVENT_START_ID = 51408
EVENT_END_ID = 51983

# Wider analysis window
ANALYSIS_START_TIME = "2025-09-11 00:00:00"
ANALYSIS_END_TIME = "2025-09-18 23:50:00"

# "avg_only" = only *_avg
# "avg_std"  = *_avg and *_std
# "all"      = *_avg, *_max, *_min, *_std
MEASUREMENT_MODE = "all"

TOP_N_TO_PLOT = 20


# ============================================================
# Loading and column selection
# ============================================================

def load_raw_scada(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw SCADA file not found:\n{path}")

    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"time_stamp", "asset_id", "id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw file: {missing}")

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
        raise ValueError("MEASUREMENT_MODE must be one of: avg_only, avg_std, all")

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


# ============================================================
# Period extraction
# ============================================================

def slice_period(
    df: pd.DataFrame,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)

    return df[
        (df["time_stamp"] >= start)
        & (df["time_stamp"] <= end)
    ].copy()


def define_periods(window: pd.DataFrame) -> dict[str, pd.DataFrame]:
    periods = {
        "sep11_reference_day": slice_period(
            window,
            "2025-09-11 00:00:00",
            "2025-09-11 23:50:00",
        ),
        "metadata_interval_12_15": slice_period(
            window,
            "2025-09-12 00:00:00",
            "2025-09-15 23:50:00",
        ),
        "sep16_failure_day": slice_period(
            window,
            "2025-09-16 00:00:00",
            "2025-09-16 23:50:00",
        ),
        "sep17_18_standstill": slice_period(
            window,
            "2025-09-17 00:00:00",
            "2025-09-18 23:50:00",
        ),
    }

    return periods


# ============================================================
# Statistics
# ============================================================

def calculate_period_stats(
    period_df: pd.DataFrame,
    measurement_cols: list[str],
    period_name: str,
) -> pd.DataFrame:
    rows = []

    for col in measurement_cols:
        x = pd.to_numeric(period_df[col], errors="coerce")
        valid = x.dropna()

        if valid.empty:
            rows.append({
                "period": period_name,
                "measurement": col,
                "base_signal": get_base_signal(col),
                "stat_type": get_stat_type(col),
                "family": get_family(col),
                "n_rows": len(period_df),
                "n_valid": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
                "median": np.nan,
            })
            continue

        rows.append({
            "period": period_name,
            "measurement": col,
            "base_signal": get_base_signal(col),
            "stat_type": get_stat_type(col),
            "family": get_family(col),
            "n_rows": len(period_df),
            "n_valid": int(valid.shape[0]),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "range": float(valid.max() - valid.min()),
            "median": float(valid.median()),
        })

    return pd.DataFrame(rows)


def safe_diff(new_value, old_value) -> float:
    if pd.isna(new_value) or pd.isna(old_value):
        return np.nan
    return float(new_value - old_value)


def safe_relative_change(new_value, old_value) -> float:
    if pd.isna(new_value) or pd.isna(old_value):
        return np.nan
    if abs(old_value) < 1e-9:
        return np.nan
    return float((new_value - old_value) / abs(old_value))


def safe_relative_abs_diff(new_value, old_value) -> float:
    value = safe_relative_change(new_value, old_value)
    if pd.isna(value):
        return 0.0
    return abs(value)


def build_comparison_table(period_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per measurement.

    It compares:
    - Sep16 vs Sep11
    - Sep17-18 vs Sep11
    - Metadata interval 12-15 vs Sep11
    - Sep17-18 vs metadata interval 12-15
    """

    periods = {
        name: period_stats[period_stats["period"] == name].set_index("measurement")
        for name in [
            "sep11_reference_day",
            "metadata_interval_12_15",
            "sep16_failure_day",
            "sep17_18_standstill",
        ]
    }

    reference = periods["sep11_reference_day"]

    rows = []

    for measurement, ref_row in reference.iterrows():
        row = {
            "measurement": measurement,
            "base_signal": ref_row["base_signal"],
            "stat_type": ref_row["stat_type"],
            "family": ref_row["family"],

            "sep11_mean": ref_row["mean"],
            "sep11_std": ref_row["std"],
            "sep11_min": ref_row["min"],
            "sep11_max": ref_row["max"],
            "sep11_range": ref_row["range"],
        }

        for period_name, short_name in [
            ("metadata_interval_12_15", "metadata_12_15"),
            ("sep16_failure_day", "sep16"),
            ("sep17_18_standstill", "standstill_17_18"),
        ]:
            table = periods[period_name]

            if measurement not in table.index:
                for metric in ["mean", "std", "min", "max", "range"]:
                    row[f"{short_name}_{metric}"] = np.nan
                    row[f"{short_name}_minus_sep11_{metric}"] = np.nan
                    row[f"{short_name}_vs_sep11_rel_{metric}"] = np.nan
                continue

            current = table.loc[measurement]

            for metric in ["mean", "std", "min", "max", "range"]:
                current_value = current[metric]
                ref_value = ref_row[metric]

                row[f"{short_name}_{metric}"] = current_value
                row[f"{short_name}_minus_sep11_{metric}"] = safe_diff(
                    current_value,
                    ref_value,
                )
                row[f"{short_name}_vs_sep11_rel_{metric}"] = safe_relative_change(
                    current_value,
                    ref_value,
                )

        # Also compare standstill against metadata interval
        metadata_table = periods["metadata_interval_12_15"]
        standstill_table = periods["sep17_18_standstill"]

        if measurement in metadata_table.index and measurement in standstill_table.index:
            meta = metadata_table.loc[measurement]
            st = standstill_table.loc[measurement]

            for metric in ["mean", "std", "min", "max", "range"]:
                row[f"standstill_minus_metadata_{metric}"] = safe_diff(
                    st[metric],
                    meta[metric],
                )
                row[f"standstill_vs_metadata_rel_{metric}"] = safe_relative_change(
                    st[metric],
                    meta[metric],
                )
        else:
            for metric in ["mean", "std", "min", "max", "range"]:
                row[f"standstill_minus_metadata_{metric}"] = np.nan
                row[f"standstill_vs_metadata_rel_{metric}"] = np.nan

        # Change scores
        # 1. Sep16 compared with Sep11
        row["sep16_change_score_vs_sep11"] = (
            safe_relative_abs_diff(row.get("sep16_mean"), row.get("sep11_mean"))
            + safe_relative_abs_diff(row.get("sep16_std"), row.get("sep11_std"))
            + safe_relative_abs_diff(row.get("sep16_range"), row.get("sep11_range"))
        )

        # 2. Sep17-18 compared with Sep11
        row["standstill_change_score_vs_sep11"] = (
            safe_relative_abs_diff(row.get("standstill_17_18_mean"), row.get("sep11_mean"))
            + safe_relative_abs_diff(row.get("standstill_17_18_std"), row.get("sep11_std"))
            + safe_relative_abs_diff(row.get("standstill_17_18_range"), row.get("sep11_range"))
        )

        # 3. Sep17-18 compared with metadata 12-15
        row["standstill_change_score_vs_metadata"] = (
            abs(row.get("standstill_vs_metadata_rel_mean", 0))
            if pd.notna(row.get("standstill_vs_metadata_rel_mean", np.nan))
            else 0
        ) + (
            abs(row.get("standstill_vs_metadata_rel_std", 0))
            if pd.notna(row.get("standstill_vs_metadata_rel_std", np.nan))
            else 0
        ) + (
            abs(row.get("standstill_vs_metadata_rel_range", 0))
            if pd.notna(row.get("standstill_vs_metadata_rel_range", np.nan))
            else 0
        )

        rows.append(row)

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        "standstill_change_score_vs_sep11",
        ascending=False,
    ).reset_index(drop=True)

    return comparison


def build_signal_family_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    summary = (
        comparison.groupby("family")
        .agg(
            n_measurements=("measurement", "count"),
            mean_standstill_change_score_vs_sep11=(
                "standstill_change_score_vs_sep11",
                "mean",
            ),
            max_standstill_change_score_vs_sep11=(
                "standstill_change_score_vs_sep11",
                "max",
            ),
            median_standstill_change_score_vs_sep11=(
                "standstill_change_score_vs_sep11",
                "median",
            ),
            mean_sep16_change_score_vs_sep11=(
                "sep16_change_score_vs_sep11",
                "mean",
            ),
            max_sep16_change_score_vs_sep11=(
                "sep16_change_score_vs_sep11",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "max_standstill_change_score_vs_sep11",
            ascending=False,
        )
    )

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_measurement_time_series(
    window: pd.DataFrame,
    measurement: str,
    output_dir: Path,
) -> None:
    plot_df = window.copy()
    plot_df[measurement] = pd.to_numeric(plot_df[measurement], errors="coerce")

    plt.figure(figsize=(15, 5))

    plt.plot(
        plot_df["time_stamp"],
        plot_df[measurement],
        linewidth=1,
    )

    # Metadata interval 12-15
    plt.axvline(
        pd.to_datetime("2025-09-12 00:00:00"),
        linestyle="--",
        linewidth=2,
        label="metadata start",
    )
    plt.axvline(
        pd.to_datetime("2025-09-15 23:50:00"),
        linestyle="--",
        linewidth=2,
        label="metadata end",
    )

    # Description-based markers
    plt.axvline(
        pd.to_datetime("2025-09-16 12:00:00"),
        linestyle=":",
        linewidth=2,
        label="16th afternoon fault mentioned",
    )
    plt.axvline(
        pd.to_datetime("2025-09-17 00:00:00"),
        linestyle=":",
        linewidth=2,
        label="17th standstill starts",
    )

    plt.title(f"{measurement}: Sep 11 to Sep 18")
    plt.xlabel("Time")
    plt.ylabel(measurement)
    plt.legend()
    plt.tight_layout()

    safe_name = measurement.replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{safe_name}_sep11_to_sep18.png"

    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top20_bar(
    comparison: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> None:
    top = comparison.head(top_n).copy()

    if top.empty:
        return

    plt.figure(figsize=(12, 7))
    plt.barh(
        top["measurement"][::-1],
        top["standstill_change_score_vs_sep11"][::-1],
    )

    plt.xlabel("Standstill change score vs Sep 11")
    plt.ylabel("Measurement")
    plt.title(f"Top {top_n} changed measurements: Sep17-18 vs Sep11")
    plt.tight_layout()

    output_path = output_dir / f"top{top_n}_changed_measurements_bar_sep17_18_vs_sep11.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_signal_family_bar(
    family_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(9, 5))

    plt.bar(
        family_summary["family"],
        family_summary["mean_standstill_change_score_vs_sep11"],
    )

    plt.xlabel("Signal family")
    plt.ylabel("Mean change score")
    plt.title("Mean standstill change score by signal family")
    plt.tight_layout()

    output_path = output_dir / "signal_family_mean_change_score_sep17_18_vs_sep11.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# Markdown summary
# ============================================================

def write_markdown_summary(
    window: pd.DataFrame,
    periods: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    family_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    md_path = output_dir / "event18_sep11_to_sep18_summary.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Event 18 Sep 11 to Sep 18 SCADA Change Analysis\n\n")

        f.write("## Event information\n\n")
        f.write(f"- Farm ID: `{FARM_ID}`\n")
        f.write(f"- Event / asset key: `{EVENT_ID}`\n")
        f.write("- Label: `anomaly`\n")
        f.write(f"- Metadata start: `{EVENT_START_TIME}` / id `{EVENT_START_ID}`\n")
        f.write(f"- Metadata end: `{EVENT_END_TIME}` / id `{EVENT_END_ID}`\n")
        f.write(f"- Description: `{EVENT_DESCRIPTION}`\n\n")

        f.write("## Period definition\n\n")
        f.write("| Period | Time range | Rows | Meaning |\n")
        f.write("|---|---|---:|---|\n")

        meanings = {
            "sep11_reference_day": "Earlier reference day",
            "metadata_interval_12_15": "Metadata-labelled anomaly interval",
            "sep16_failure_day": "Reported 24VAC rotor brake fault day",
            "sep17_18_standstill": "Longer standstill period",
        }

        for name, df in periods.items():
            if df.empty:
                time_range = "NA"
            else:
                time_range = f"{df['time_stamp'].min()} to {df['time_stamp'].max()}"

            f.write(
                f"| `{name}` | {time_range} | {len(df)} | {meanings.get(name, '')} |\n"
            )

        f.write("\n## Top changed measurements\n\n")
        f.write(
            "| Rank | Measurement | Family | Type | Sep11 mean | Standstill mean | "
            "Difference | Relative change | Change score |\n"
        )
        f.write("|---:|---|---|---|---:|---:|---:|---:|---:|\n")

        top20 = comparison.head(20)

        for i, row in top20.iterrows():
            f.write(
                f"| {i + 1} "
                f"| `{row['measurement']}` "
                f"| `{row['family']}` "
                f"| `{row['stat_type']}` "
                f"| {row['sep11_mean']:.4g} "
                f"| {row['standstill_17_18_mean']:.4g} "
                f"| {row['standstill_17_18_minus_sep11_mean']:.4g} "
                f"| {row['standstill_17_18_vs_sep11_rel_mean']:.4g} "
                f"| {row['standstill_change_score_vs_sep11']:.4g} |\n"
            )

        f.write("\n## Signal family summary\n\n")
        f.write(
            "| Family | Measurements | Mean standstill change score | "
            "Max standstill change score |\n"
        )
        f.write("|---|---:|---:|---:|\n")

        for _, row in family_summary.iterrows():
            f.write(
                f"| `{row['family']}` "
                f"| {int(row['n_measurements'])} "
                f"| {row['mean_standstill_change_score_vs_sep11']:.4g} "
                f"| {row['max_standstill_change_score_vs_sep11']:.4g} |\n"
            )

        f.write("\n## Interpretation note\n\n")
        f.write(
            "This analysis extends the window beyond the metadata interval because the "
            "event description mentions failures on 16 September and a longer standstill "
            "from 17 September onwards. Sep 11 is used as an earlier reference day. "
            "The comparison focuses on how the Sep17-18 standstill period differs from "
            "the Sep11 reference state. Average values show overall level changes, "
            "maximum and minimum values show extreme peaks or drops, and standard deviation "
            "shows whether the signal became more or less variable.\n"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("Event 18: Sep 11 to Sep 18 SCADA Change Analysis")
    print("=" * 100)

    print("\nEvent description:")
    print(EVENT_DESCRIPTION)

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
        raise ValueError(
            "Analysis window is empty. Check ANALYSIS_START_TIME, "
            "ANALYSIS_END_TIME and the raw file timestamps."
        )

    print("\nAnalysis window:")
    print(f"{window['time_stamp'].min()} to {window['time_stamp'].max()}")
    print(f"Rows: {len(window)}")

    periods = define_periods(window)

    print("\nPeriod rows:")
    for name, df in periods.items():
        if df.empty:
            print(f"{name}: 0 rows")
        else:
            print(
                f"{name}: {len(df)} rows, "
                f"{df['time_stamp'].min()} to {df['time_stamp'].max()}"
            )

    measurement_cols = get_measurement_columns(raw, mode=MEASUREMENT_MODE)

    print(f"\nMeasurement mode: {MEASUREMENT_MODE}")
    print(f"Selected measurement columns: {len(measurement_cols)}")

    if not measurement_cols:
        raise ValueError("No measurement columns selected.")

    pd.DataFrame({"measurement": measurement_cols}).to_csv(
        OUTPUT_DIR / "selected_measurement_columns.csv",
        index=False,
    )

    print("\nCalculating period statistics...")

    period_stats_list = []

    for period_name, period_df in periods.items():
        period_stats = calculate_period_stats(
            period_df=period_df,
            measurement_cols=measurement_cols,
            period_name=period_name,
        )
        period_stats_list.append(period_stats)

    period_stats_all = pd.concat(period_stats_list, ignore_index=True)

    period_stats_path = OUTPUT_DIR / "period_stats_sep11_to_sep18.csv"
    period_stats_all.to_csv(period_stats_path, index=False)

    print("Building comparison table...")

    comparison = build_comparison_table(period_stats_all)

    comparison_path = OUTPUT_DIR / "comparison_sep11_to_sep18.csv"
    comparison.to_csv(comparison_path, index=False)

    top20 = comparison.head(20).copy()
    top20_path = OUTPUT_DIR / "top20_changed_measurements_sep17_18_vs_sep11.csv"
    top20.to_csv(top20_path, index=False)

    family_summary = build_signal_family_summary(comparison)
    family_summary_path = OUTPUT_DIR / "signal_family_summary_sep11_to_sep18.csv"
    family_summary.to_csv(family_summary_path, index=False)

    print("\nTop 20 changed measurements: Sep17-18 standstill vs Sep11")
    display_cols = [
        "measurement",
        "base_signal",
        "stat_type",
        "family",
        "sep11_mean",
        "standstill_17_18_mean",
        "standstill_17_18_minus_sep11_mean",
        "standstill_17_18_vs_sep11_rel_mean",
        "standstill_change_score_vs_sep11",
    ]

    print(top20[display_cols].to_string(index=False))

    print("\nSignal family summary:")
    print(family_summary.to_string(index=False))

    print("\nSaving plots...")

    plot_top20_bar(comparison, OUTPUT_DIR, top_n=20)
    plot_signal_family_bar(family_summary, OUTPUT_DIR)

    for measurement in top20["measurement"].head(TOP_N_TO_PLOT):
        try:
            plot_measurement_time_series(window, measurement, OUTPUT_DIR)
        except Exception as exc:
            print(f"Could not plot {measurement}: {exc}")

    write_markdown_summary(
        window=window,
        periods=periods,
        comparison=comparison,
        family_summary=family_summary,
        output_dir=OUTPUT_DIR,
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nMain output files:")
    print("- selected_measurement_columns.csv")
    print("- period_stats_sep11_to_sep18.csv")
    print("- comparison_sep11_to_sep18.csv")
    print("- top20_changed_measurements_sep17_18_vs_sep11.csv")
    print("- signal_family_summary_sep11_to_sep18.csv")
    print("- event18_sep11_to_sep18_summary.md")
    print("- top20 changed measurement plots")
    print("- signal_family_mean_change_score_sep17_18_vs_sep11.png")
    print("- top20_changed_measurements_bar_sep17_18_vs_sep11.png")

    print("\nDone.")


if __name__ == "__main__":
    main()