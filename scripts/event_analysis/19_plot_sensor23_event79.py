from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 配置区
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Wind Farm C"
    / "datasets"
    / "79.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "event_analysis"
    / "farm_C_asset_79_event_52704_52992_sensor23_plots"
)

EVENT_START_ID = 52704
EVENT_END_ID = 52992

EVENT_START_TIME = "2025-07-24 09:00:00"
EVENT_END_TIME = "2025-07-26 09:00:00"

MEASUREMENTS = [
    "sensor_23_avg",
    "sensor_23_max",
    "sensor_23_min",
    "sensor_23_std",
]

USE_EQUAL_LENGTH_BEFORE_AFTER = True


# =========================================================
# 工具函数
# =========================================================
def load_raw_scada(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=False)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    return df


def build_segments(df: pd.DataFrame,
                   event_start_id: int,
                   event_end_id: int,
                   equal_length: bool = True):
    event_len = event_end_id - event_start_id + 1

    if equal_length:
        before_start_id = event_start_id - event_len
        before_end_id = event_start_id - 1
        after_start_id = event_end_id + 1
        after_end_id = event_end_id + event_len
    else:
        before_start_id = event_start_id - event_len
        before_end_id = event_start_id - 1
        after_start_id = event_end_id + 1
        after_end_id = df["id"].max()

    before = df[(df["id"] >= before_start_id) & (df["id"] <= before_end_id)].copy()
    during = df[(df["id"] >= event_start_id) & (df["id"] <= event_end_id)].copy()
    after = df[(df["id"] >= after_start_id) & (df["id"] <= after_end_id)].copy()

    return before, during, after, {
        "event_len": event_len,
        "before_start_id": before_start_id,
        "before_end_id": before_end_id,
        "after_start_id": after_start_id,
        "after_end_id": after_end_id,
        "before_rows": len(before),
        "during_rows": len(during),
        "after_rows": len(after),
    }


def plot_one_measurement(before: pd.DataFrame,
                         during: pd.DataFrame,
                         after: pd.DataFrame,
                         measurement: str,
                         output_dir: Path):
    plot_df = pd.concat(
        [
            before.assign(segment="before"),
            during.assign(segment="during"),
            after.assign(segment="after"),
        ],
        ignore_index=True,
    ).copy()

    plot_df = plot_df.sort_values("id")
    plot_df[measurement] = pd.to_numeric(plot_df[measurement], errors="coerce")

    plt.figure(figsize=(14, 5))
    plt.plot(plot_df["time_stamp"], plot_df[measurement], linewidth=1)

    if not during.empty:
        plt.axvline(
            during["time_stamp"].iloc[0],
            linestyle="--",
            linewidth=2,
            label="event start",
        )
        plt.axvline(
            during["time_stamp"].iloc[-1],
            linestyle="--",
            linewidth=2,
            label="event end",
        )

    plt.title(f"{measurement}: before / during / after")
    plt.xlabel("Time")
    plt.ylabel(measurement)
    plt.legend()
    plt.tight_layout()

    out_path = output_dir / f"{measurement}_before_during_after.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_summary_table(before: pd.DataFrame,
                       during: pd.DataFrame,
                       after: pd.DataFrame,
                       measurements: list,
                       output_dir: Path):
    rows = []

    for col in measurements:
        b = pd.to_numeric(before[col], errors="coerce")
        d = pd.to_numeric(during[col], errors="coerce")
        a = pd.to_numeric(after[col], errors="coerce")

        rows.append({
            "measurement": col,
            "before_mean": b.mean(),
            "during_mean": d.mean(),
            "after_mean": a.mean(),
            "before_std": b.std(),
            "during_std": d.std(),
            "after_std": a.std(),
            "before_min": b.min(),
            "during_min": d.min(),
            "after_min": a.min(),
            "before_max": b.max(),
            "during_max": d.max(),
            "after_max": a.max(),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "sensor23_summary_stats.csv", index=False)
    return summary


# =========================================================
# 主程序
# =========================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Plot sensor_23 avg / max / min / std for Event 79")
    print("=" * 80)

    print(f"\nLoading raw file:\n{RAW_FILE_PATH}")
    df = load_raw_scada(RAW_FILE_PATH)

    print(f"\nRaw SCADA shape: {df.shape}")

    missing_cols = [c for c in MEASUREMENTS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in raw file: {missing_cols}")

    before, during, after, info = build_segments(
        df=df,
        event_start_id=EVENT_START_ID,
        event_end_id=EVENT_END_ID,
        equal_length=USE_EQUAL_LENGTH_BEFORE_AFTER,
    )

    print("\nSegment info:")
    for k, v in info.items():
        print(f"{k}: {v}")

    print("\nTime periods:")
    if not before.empty:
        print(f"Before: {before['time_stamp'].min()} to {before['time_stamp'].max()}")
    if not during.empty:
        print(f"During: {during['time_stamp'].min()} to {during['time_stamp'].max()}")
    if not after.empty:
        print(f"After : {after['time_stamp'].min()} to {after['time_stamp'].max()}")

    print("\nGenerating plots...")
    for col in MEASUREMENTS:
        plot_one_measurement(before, during, after, col, OUTPUT_DIR)
        print(f"Saved plot: {col}_before_during_after.png")

    print("\nSaving summary table...")
    summary = save_summary_table(before, during, after, MEASUREMENTS, OUTPUT_DIR)
    print(summary.to_string(index=False))

    print(f"\nSaved outputs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()