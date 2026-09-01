from __future__ import annotations

from pathlib import Path
import gc
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FARM_ID = "C"
EVENT_ID = "35"

RAW_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Wind Farm C"
    / "datasets"
    / "35.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "event_analysis"
    / "farm_C_event_35_short_standstill_exploration"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_DESCRIPTION = (
    'Turbine had several short standstills (max 8min) with failure '
    '"Schwingungen Umrichter Drehmomenten Level 1"'
)

# Metadata-labelled interval
EVENT_START_TIME = "2024-09-23 00:00:00"
EVENT_END_TIME = "2024-09-29 09:00:00"
EVENT_START_ID = 51696
EVENT_END_ID = 52614

# Context around the event.
# These periods are not assumed to be verified normal or faulty baselines.
ANALYSIS_START_TIME = "2024-09-21 00:00:00"
ANALYSIS_END_TIME = "2024-10-01 23:50:00"

# Analyse all available *_avg, *_max, *_min and *_std measurements.
MEASUREMENT_MODE = "all"

CSV_SEPARATOR = ";"

# Memory controls
CHUNK_SIZE = 250
CONVERSION_BLOCK_SIZE = 64
LOCAL_SCORE_BLOCK_SIZE = 64

# Local comparison uses the previous and next three 10-minute rows.
# This gives a surrounding context of approximately one hour.
LOCAL_NEIGHBOUR_POINTS = 3

# Number of ranked windows and measurements saved for inspection.
TOP_CANDIDATE_WINDOWS = 30
TOP_CONTRIBUTORS_PER_WINDOW = 20
TOP_MEASUREMENTS_TO_PLOT = 20
TOP_CANDIDATE_CONTEXT_PLOTS = 10

# Context shown around each candidate timestamp.
CANDIDATE_CONTEXT_HOURS = 2


# =============================================================================
# Column definitions
# =============================================================================

ALLOWED_PREFIXES = (
    "sensor_",
    "power_",
    "reactive_power_",
    "wind_speed_",
)

ALLOWED_SUFFIXES = (
    "_avg",
    "_max",
    "_min",
    "_std",
)

METADATA_COLUMNS = (
    "time_stamp",
    "asset_id",
    "id",
    "train_test",
    "status_type_id",
)


def clean_column_name(column: object) -> str:
    return str(column).strip()


def is_measurement_column(column: str) -> bool:
    return (
        column.startswith(ALLOWED_PREFIXES)
        and column.endswith(ALLOWED_SUFFIXES)
    )


def get_base_signal(column: str) -> str:
    return re.sub(r"_(avg|max|min|std)$", "", column)


def get_stat_type(column: str) -> str:
    match = re.search(r"_(avg|max|min|std)$", column)
    return match.group(1) if match else "unknown"


def get_family(column: str) -> str:
    if column.startswith("reactive_power_"):
        return "reactive_power"
    if column.startswith("wind_speed_"):
        return "wind_speed"
    if column.startswith("power_"):
        return "power"
    if column.startswith("sensor_"):
        return "sensor"
    return "other"


# =============================================================================
# Memory-efficient loading
# =============================================================================

def inspect_csv_header(
    path: Path,
) -> tuple[list[str], dict[str, str], list[str], list[str]]:
    """
    Read only the CSV header and select all required metadata and
    measurement columns.
    """

    header = pd.read_csv(
        path,
        sep=CSV_SEPARATOR,
        nrows=0,
    )

    original_columns = list(header.columns)

    rename_map = {
        original: clean_column_name(original)
        for original in original_columns
    }

    cleaned_columns = [
        rename_map[column]
        for column in original_columns
    ]

    required_columns = {
        "time_stamp",
        "asset_id",
        "id",
    }

    missing = required_columns - set(cleaned_columns)

    if missing:
        raise ValueError(
            f"Missing required columns in raw file: {missing}"
        )

    metadata_columns = [
        column
        for column in cleaned_columns
        if column in METADATA_COLUMNS
    ]

    measurement_columns = [
        column
        for column in cleaned_columns
        if is_measurement_column(column)
    ]

    if not measurement_columns:
        raise ValueError(
            "No *_avg, *_max, *_min or *_std measurement columns were found."
        )

    selected_clean_columns = set(
        metadata_columns + measurement_columns
    )

    usecols_original = [
        original
        for original in original_columns
        if rename_map[original] in selected_clean_columns
    ]

    return (
        usecols_original,
        rename_map,
        metadata_columns,
        measurement_columns,
    )


def convert_measurements_to_float32(
    chunk: pd.DataFrame,
    measurement_columns: list[str],
) -> pd.DataFrame:
    """
    Convert measurement columns in small blocks to reduce temporary memory.
    """

    available_columns = [
        column
        for column in measurement_columns
        if column in chunk.columns
    ]

    for start in range(
        0,
        len(available_columns),
        CONVERSION_BLOCK_SIZE,
    ):
        block_columns = available_columns[
            start:start + CONVERSION_BLOCK_SIZE
        ]

        numeric_block = chunk[
            block_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        chunk[block_columns] = numeric_block.astype(
            np.float32
        )

        del numeric_block
        gc.collect()

    return chunk


def load_analysis_window(
    path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """
    Read 35.csv in small chunks and retain only the requested context window.

    No fixed baseline, anomaly threshold, abnormal-fraction threshold,
    rolling detection rule or minimum anomaly duration is applied.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Raw SCADA file not found:\n{path}"
        )

    analysis_start = pd.to_datetime(
        ANALYSIS_START_TIME
    )
    analysis_end = pd.to_datetime(
        ANALYSIS_END_TIME
    )

    print("\nInspecting CSV header...")

    (
        usecols_original,
        rename_map,
        metadata_columns,
        measurement_columns,
    ) = inspect_csv_header(path)

    print(
        f"Selected measurement columns (all): "
        f"{len(measurement_columns)}"
    )
    print(
        f"Requested analysis range: "
        f"{analysis_start} to {analysis_end}"
    )
    print(f"Chunk size: {CHUNK_SIZE}")

    retained_chunks: list[pd.DataFrame] = []

    total_rows_read = 0
    total_rows_retained = 0
    last_timestamp_seen = pd.NaT

    reader = pd.read_csv(
        path,
        sep=CSV_SEPARATOR,
        usecols=usecols_original,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        total_rows_read += len(chunk)

        chunk = chunk.rename(
            columns=rename_map
        )

        chunk["time_stamp"] = pd.to_datetime(
            chunk["time_stamp"],
            errors="coerce",
        )

        valid_times = chunk[
            "time_stamp"
        ].dropna()

        if not valid_times.empty:
            chunk_min = valid_times.min()
            chunk_max = valid_times.max()
            last_timestamp_seen = chunk_max

            # Event files are expected to be chronological.
            if chunk_min > analysis_end:
                del chunk
                break

        chunk = chunk[
            (chunk["time_stamp"] >= analysis_start)
            & (chunk["time_stamp"] <= analysis_end)
        ].copy()

        if chunk.empty:
            if chunk_number % 25 == 0:
                print(
                    f"Chunk {chunk_number}: "
                    f"read={total_rows_read:,}, "
                    f"retained={total_rows_retained:,}"
                )

            del chunk
            continue

        chunk["id"] = pd.to_numeric(
            chunk["id"],
            errors="coerce",
            downcast="integer",
        )

        chunk = chunk.dropna(
            subset=["time_stamp", "id"]
        ).copy()

        if chunk.empty:
            del chunk
            continue

        chunk["id"] = chunk[
            "id"
        ].astype(np.int32)

        chunk = convert_measurements_to_float32(
            chunk=chunk,
            measurement_columns=measurement_columns,
        )

        ordered_columns = [
            column
            for column in (
                metadata_columns
                + measurement_columns
            )
            if column in chunk.columns
        ]

        retained_chunks.append(
            chunk[ordered_columns].copy()
        )

        total_rows_retained += len(chunk)

        print(
            f"Chunk {chunk_number}: "
            f"read={total_rows_read:,}, "
            f"retained={total_rows_retained:,}"
        )

        del chunk
        gc.collect()

    if not retained_chunks:
        raise ValueError(
            "No rows were found inside the requested analysis window. "
            "Check Event 35 timestamps and raw-file coverage."
        )

    print("\nCombining retained chunks...")

    window = pd.concat(
        retained_chunks,
        ignore_index=True,
        copy=False,
    )

    del retained_chunks
    gc.collect()

    window = (
        window.drop_duplicates(
            subset=["time_stamp", "id"]
        )
        .sort_values(["time_stamp", "id"])
        .reset_index(drop=True)
    )

    usable_measurement_columns = [
        column
        for column in measurement_columns
        if (
            column in window.columns
            and window[column].notna().any()
        )
    ]

    load_info = {
        "total_rows_read": total_rows_read,
        "total_rows_retained": len(window),
        "last_timestamp_seen": last_timestamp_seen,
        "window_start": window["time_stamp"].min(),
        "window_end": window["time_stamp"].max(),
        "n_selected_measurements": len(
            usable_measurement_columns
        ),
    }

    print("\nLoading completed.")
    print(f"Rows read: {total_rows_read:,}")
    print(f"Rows retained: {len(window):,}")
    print(
        f"Actual window: "
        f"{window['time_stamp'].min()} "
        f"to {window['time_stamp'].max()}"
    )
    print(
        f"Usable measurement columns: "
        f"{len(usable_measurement_columns)}"
    )

    if window["time_stamp"].max() < analysis_end:
        print(
            "\n[WARNING] The raw file ends before the requested "
            "analysis end time."
        )

    return (
        window,
        usable_measurement_columns,
        load_info,
    )


# =============================================================================
# Period labels and sampling interval
# =============================================================================

def assign_period_label(
    timestamps: pd.Series,
) -> pd.Series:
    event_start = pd.to_datetime(
        EVENT_START_TIME
    )
    event_end = pd.to_datetime(
        EVENT_END_TIME
    )

    labels = np.where(
        timestamps < event_start,
        "before_event_context",
        np.where(
            timestamps <= event_end,
            "metadata_event_interval",
            "after_event_context",
        ),
    )

    return pd.Series(
        labels,
        index=timestamps.index,
    )


def infer_sample_minutes(
    window: pd.DataFrame,
) -> float:
    differences = (
        window["time_stamp"]
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(60)
    )

    differences = differences[
        (differences > 0)
        & (differences <= 60)
    ]

    if differences.empty:
        return 10.0

    return float(
        differences.median()
    )


# =============================================================================
# Local-neighbour comparison
# =============================================================================

def build_neighbour_median(
    values: np.ndarray,
    neighbour_points: int,
) -> np.ndarray:
    """
    Calculate the median of surrounding rows, excluding the current row.

    For neighbour_points=3, each row is compared with up to:
      - three previous 10-minute rows;
      - three following 10-minute rows.

    This is a local contextual comparison, not a fixed normal baseline.
    """

    n_rows, n_columns = values.shape

    neighbours = np.full(
        (
            neighbour_points * 2,
            n_rows,
            n_columns,
        ),
        np.nan,
        dtype=np.float32,
    )

    neighbour_index = 0

    for offset in range(
        1,
        neighbour_points + 1,
    ):
        # Previous rows
        neighbours[
            neighbour_index,
            offset:,
            :,
        ] = values[
            :-offset,
            :,
        ]

        neighbour_index += 1

        # Following rows
        neighbours[
            neighbour_index,
            :-offset,
            :,
        ] = values[
            offset:,
            :,
        ]

        neighbour_index += 1

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        neighbour_median = np.nanmedian(
            neighbours,
            axis=0,
        )

    del neighbours
    gc.collect()

    return neighbour_median.astype(
        np.float32,
        copy=False,
    )


def calculate_measurement_scale(
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Create a data-driven scale for each measurement.

    The scale is used only to make measurements with different units
    comparable in a ranking. It does not define normal or abnormal.

    Scale = max(
        median absolute row-to-row change,
        1% of the 5th-to-95th percentile span,
        a very small magnitude floor
    )
    """

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        q05 = np.nanpercentile(
            values,
            5,
            axis=0,
        )

        q50 = np.nanpercentile(
            values,
            50,
            axis=0,
        )

        q95 = np.nanpercentile(
            values,
            95,
            axis=0,
        )

        absolute_differences = np.abs(
            np.diff(
                values,
                axis=0,
            )
        )

        median_abs_difference = np.nanmedian(
            absolute_differences,
            axis=0,
        )

    percentile_span = q95 - q05

    magnitude_floor = (
        1e-6
        * np.maximum(
            np.abs(q50),
            1.0,
        )
    )

    percentile_floor = (
        0.01
        * np.maximum(
            percentile_span,
            0.0,
        )
    )

    scale = np.maximum(
        median_abs_difference,
        percentile_floor,
    )

    scale = np.maximum(
        scale,
        magnitude_floor,
    )

    invalid = (
        ~np.isfinite(scale)
        | (scale <= 0)
    )

    scale[invalid] = 1.0

    diagnostics = {
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "percentile_span": percentile_span,
        "median_abs_difference": median_abs_difference,
        "scale": scale,
    }

    del absolute_differences
    gc.collect()

    return (
        scale.astype(
            np.float32,
            copy=False,
        ),
        diagnostics,
    )


def calculate_local_measurement_scores(
    window: pd.DataFrame,
    measurement_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each measurement and timestamp calculate:

      local neighbour median
      absolute local difference
      normalised local-change score

    The normalised score is a relative ranking quantity:
        abs(current - neighbour_median) / measurement_scale

    No cut-off is applied.
    """

    n_rows = len(window)
    n_columns = len(measurement_columns)

    score_values = np.zeros(
        (n_rows, n_columns),
        dtype=np.float32,
    )

    centre_values = np.full(
        (n_rows, n_columns),
        np.nan,
        dtype=np.float32,
    )

    scale_rows: list[dict[str, object]] = []

    for start in range(
        0,
        n_columns,
        LOCAL_SCORE_BLOCK_SIZE,
    ):
        end = min(
            start + LOCAL_SCORE_BLOCK_SIZE,
            n_columns,
        )

        block_columns = measurement_columns[
            start:end
        ]

        values = window[
            block_columns
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        neighbour_median = build_neighbour_median(
            values=values,
            neighbour_points=LOCAL_NEIGHBOUR_POINTS,
        )

        (
            measurement_scale,
            diagnostics,
        ) = calculate_measurement_scale(
            values
        )

        local_difference = np.abs(
            values
            - neighbour_median
        )

        block_scores = (
            local_difference
            / measurement_scale
        )

        block_scores[
            ~np.isfinite(block_scores)
        ] = 0.0

        # Prevent a single numerical edge case from dominating all rankings.
        block_scores = np.clip(
            block_scores,
            0.0,
            1_000_000.0,
        )

        score_values[
            :,
            start:end,
        ] = block_scores.astype(
            np.float32,
            copy=False,
        )

        centre_values[
            :,
            start:end,
        ] = neighbour_median

        for column_index, column in enumerate(
            block_columns
        ):
            scale_rows.append({
                "measurement": column,
                "base_signal": get_base_signal(
                    column
                ),
                "stat_type": get_stat_type(
                    column
                ),
                "family": get_family(
                    column
                ),
                "q05": float(
                    diagnostics[
                        "q05"
                    ][column_index]
                ),
                "median": float(
                    diagnostics[
                        "q50"
                    ][column_index]
                ),
                "q95": float(
                    diagnostics[
                        "q95"
                    ][column_index]
                ),
                "percentile_span": float(
                    diagnostics[
                        "percentile_span"
                    ][column_index]
                ),
                "median_abs_row_change": float(
                    diagnostics[
                        "median_abs_difference"
                    ][column_index]
                ),
                "comparison_scale": float(
                    diagnostics[
                        "scale"
                    ][column_index]
                ),
            })

        print(
            f"Local scores processed: "
            f"{end}/{n_columns} measurements"
        )

        del values
        del neighbour_median
        del local_difference
        del block_scores
        gc.collect()

    score_df = pd.DataFrame(
        score_values,
        columns=measurement_columns,
    )

    centre_df = pd.DataFrame(
        centre_values,
        columns=measurement_columns,
    )

    scale_df = pd.DataFrame(
        scale_rows
    )

    return (
        score_df,
        centre_df,
        scale_df,
    )


# =============================================================================
# General row-level exploration metrics
# =============================================================================

def safe_row_percentile(
    values: np.ndarray,
    percentile: float,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        result = np.nanpercentile(
            values,
            percentile,
            axis=1,
        )

    result[
        ~np.isfinite(result)
    ] = 0.0

    return result.astype(
        np.float32,
        copy=False,
    )


def mean_top_k_per_row(
    values: np.ndarray,
    top_k: int,
) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(
            values.shape[0],
            dtype=np.float32,
        )

    actual_k = min(
        top_k,
        values.shape[1],
    )

    partition_index = (
        values.shape[1]
        - actual_k
    )

    top_values = np.partition(
        values,
        partition_index,
        axis=1,
    )[
        :,
        -actual_k:,
    ]

    result = np.nanmean(
        top_values,
        axis=1,
    )

    result[
        ~np.isfinite(result)
    ] = 0.0

    del top_values
    gc.collect()

    return result.astype(
        np.float32,
        copy=False,
    )


def build_general_window_metrics(
    window: pd.DataFrame,
    score_df: pd.DataFrame,
    measurement_columns: list[str],
) -> pd.DataFrame:
    score_values = score_df.to_numpy(
        dtype=np.float32,
        copy=False,
    )

    available_metadata = [
        column
        for column in (
            "time_stamp",
            "asset_id",
            "id",
            "status_type_id",
        )
        if column in window.columns
    ]

    metrics = window[
        available_metadata
    ].copy()

    metrics[
        "row_position"
    ] = np.arange(
        len(window),
        dtype=np.int32,
    )

    metrics[
        "period"
    ] = assign_period_label(
        metrics["time_stamp"]
    )

    metrics[
        "all_local_change_p95"
    ] = safe_row_percentile(
        score_values,
        95,
    )

    metrics[
        "all_local_change_p99"
    ] = safe_row_percentile(
        score_values,
        99,
    )

    metrics[
        "all_local_change_max"
    ] = np.nanmax(
        score_values,
        axis=1,
    )

    metrics[
        "all_local_change_top20_mean"
    ] = mean_top_k_per_row(
        score_values,
        top_k=20,
    )

    families = [
        "sensor",
        "power",
        "reactive_power",
        "wind_speed",
    ]

    for family in families:
        family_indices = [
            index
            for index, column in enumerate(
                measurement_columns
            )
            if get_family(column) == family
        ]

        if not family_indices:
            metrics[
                f"{family}_local_change_p95"
            ] = 0.0

            metrics[
                f"{family}_local_change_max"
            ] = 0.0

            continue

        family_values = score_values[
            :,
            family_indices,
        ]

        metrics[
            f"{family}_local_change_p95"
        ] = safe_row_percentile(
            family_values,
            95,
        )

        metrics[
            f"{family}_local_change_max"
        ] = np.nanmax(
            family_values,
            axis=1,
        )

    return metrics


# =============================================================================
# Power-specific short-standstill signatures
# =============================================================================

def get_base_measurement_map(
    measurement_columns: list[str],
    family: str,
) -> dict[str, dict[str, str]]:
    mapping: dict[
        str,
        dict[str, str],
    ] = {}

    for column in measurement_columns:
        if get_family(column) != family:
            continue

        base_signal = get_base_signal(
            column
        )

        stat_type = get_stat_type(
            column
        )

        mapping.setdefault(
            base_signal,
            {},
        )[stat_type] = column

    return mapping


def local_score_for_derived_series(
    values: np.ndarray,
) -> np.ndarray:
    """
    Calculate the same local-neighbour ranking score for a derived 1D series,
    such as max-min or the reported within-window standard deviation.
    """

    two_dimensional = values.reshape(
        -1,
        1,
    ).astype(
        np.float32,
        copy=False,
    )

    neighbour_median = build_neighbour_median(
        values=two_dimensional,
        neighbour_points=LOCAL_NEIGHBOUR_POINTS,
    )[
        :,
        0,
    ]

    (
        scale,
        _,
    ) = calculate_measurement_scale(
        two_dimensional
    )

    result = np.abs(
        values
        - neighbour_median
    ) / float(
        scale[0]
    )

    result[
        ~np.isfinite(result)
    ] = 0.0

    return np.clip(
        result,
        0.0,
        1_000_000.0,
    ).astype(
        np.float32,
        copy=False,
    )


def safe_aggregate(
    matrix: np.ndarray,
    method: str,
) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(
            matrix.shape[0],
            dtype=np.float32,
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        if method == "max":
            result = np.nanmax(
                matrix,
                axis=1,
            )
        elif method == "median":
            result = np.nanmedian(
                matrix,
                axis=1,
            )
        elif method == "mean":
            result = np.nanmean(
                matrix,
                axis=1,
            )
        else:
            raise ValueError(
                f"Unknown aggregation method: {method}"
            )

    result[
        ~np.isfinite(result)
    ] = 0.0

    return result.astype(
        np.float32,
        copy=False,
    )


def add_power_signature_metrics(
    metrics: pd.DataFrame,
    window: pd.DataFrame,
    centre_df: pd.DataFrame,
    scale_df: pd.DataFrame,
    measurement_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build power-related exploratory signatures.

    The signatures are intended to surface 10-minute windows that may contain:
      - a temporary reduction in average power;
      - a low minimum with a higher maximum;
      - increased within-window variability;
      - rapid recovery in the following row.

    No decision threshold is applied.
    """

    power_map = get_base_measurement_map(
        measurement_columns,
        family="power",
    )

    scale_lookup = (
        scale_df.set_index(
            "measurement"
        )[
            "comparison_scale"
        ]
        .to_dict()
    )

    per_base_rows: list[
        dict[str, object]
    ] = []

    dip_columns = []
    previous_drop_columns = []
    next_recovery_columns = []
    range_change_columns = []
    std_change_columns = []
    average_columns = []
    minimum_columns = []
    maximum_columns = []

    for base_signal, stats in power_map.items():
        average_column = stats.get(
            "avg"
        )

        minimum_column = stats.get(
            "min"
        )

        maximum_column = stats.get(
            "max"
        )

        standard_deviation_column = stats.get(
            "std"
        )

        if average_column is None:
            continue

        average_values = pd.to_numeric(
            window[average_column],
            errors="coerce",
        ).to_numpy(
            dtype=np.float32,
        )

        local_average = centre_df[
            average_column
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        average_scale = float(
            scale_lookup.get(
                average_column,
                1.0,
            )
        )

        if (
            not np.isfinite(
                average_scale
            )
            or average_scale <= 0
        ):
            average_scale = 1.0

        local_dip = np.maximum(
            local_average
            - average_values,
            0.0,
        ) / average_scale

        previous_values = np.roll(
            average_values,
            1,
        )

        previous_values[0] = np.nan

        next_values = np.roll(
            average_values,
            -1,
        )

        next_values[-1] = np.nan

        previous_drop = np.maximum(
            previous_values
            - average_values,
            0.0,
        ) / average_scale

        next_recovery = np.maximum(
            next_values
            - average_values,
            0.0,
        ) / average_scale

        local_dip[
            ~np.isfinite(local_dip)
        ] = 0.0

        previous_drop[
            ~np.isfinite(previous_drop)
        ] = 0.0

        next_recovery[
            ~np.isfinite(next_recovery)
        ] = 0.0

        dip_columns.append(
            local_dip.astype(
                np.float32,
                copy=False,
            )
        )

        previous_drop_columns.append(
            previous_drop.astype(
                np.float32,
                copy=False,
            )
        )

        next_recovery_columns.append(
            next_recovery.astype(
                np.float32,
                copy=False,
            )
        )

        average_columns.append(
            average_values
        )

        if (
            minimum_column is not None
            and maximum_column is not None
        ):
            minimum_values = pd.to_numeric(
                window[minimum_column],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32,
            )

            maximum_values = pd.to_numeric(
                window[maximum_column],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32,
            )

            within_window_range = (
                maximum_values
                - minimum_values
            )

            range_local_score = (
                local_score_for_derived_series(
                    within_window_range
                )
            )

            range_change_columns.append(
                range_local_score
            )

            minimum_columns.append(
                minimum_values
            )

            maximum_columns.append(
                maximum_values
            )
        else:
            within_window_range = np.full(
                len(window),
                np.nan,
                dtype=np.float32,
            )

        if standard_deviation_column is not None:
            standard_deviation_values = pd.to_numeric(
                window[
                    standard_deviation_column
                ],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32,
            )

            std_local_score = (
                local_score_for_derived_series(
                    standard_deviation_values
                )
            )

            std_change_columns.append(
                std_local_score
            )
        else:
            standard_deviation_values = np.full(
                len(window),
                np.nan,
                dtype=np.float32,
            )

        per_base_rows.append({
            "base_signal": base_signal,
            "avg_column": average_column,
            "min_column": minimum_column,
            "max_column": maximum_column,
            "std_column": standard_deviation_column,
            "average_scale": average_scale,
            "n_rows": len(window),
            "mean_local_dip_score": float(
                np.nanmean(local_dip)
            ),
            "max_local_dip_score": float(
                np.nanmax(local_dip)
            ),
            "mean_previous_drop_score": float(
                np.nanmean(previous_drop)
            ),
            "max_previous_drop_score": float(
                np.nanmax(previous_drop)
            ),
            "mean_next_recovery_score": float(
                np.nanmean(next_recovery)
            ),
            "max_next_recovery_score": float(
                np.nanmax(next_recovery)
            ),
            "mean_within_window_range": float(
                np.nanmean(
                    within_window_range
                )
            ),
            "max_within_window_range": float(
                np.nanmax(
                    within_window_range
                )
            ),
            "mean_reported_std": float(
                np.nanmean(
                    standard_deviation_values
                )
            ),
            "max_reported_std": float(
                np.nanmax(
                    standard_deviation_values
                )
            ),
        })

    n_rows = len(window)

    def stack_or_empty(
        columns: list[np.ndarray],
    ) -> np.ndarray:
        if not columns:
            return np.empty(
                (n_rows, 0),
                dtype=np.float32,
            )

        return np.column_stack(
            columns
        ).astype(
            np.float32,
            copy=False,
        )

    dip_matrix = stack_or_empty(
        dip_columns
    )

    previous_drop_matrix = stack_or_empty(
        previous_drop_columns
    )

    next_recovery_matrix = stack_or_empty(
        next_recovery_columns
    )

    range_change_matrix = stack_or_empty(
        range_change_columns
    )

    std_change_matrix = stack_or_empty(
        std_change_columns
    )

    average_matrix = stack_or_empty(
        average_columns
    )

    minimum_matrix = stack_or_empty(
        minimum_columns
    )

    maximum_matrix = stack_or_empty(
        maximum_columns
    )

    metrics[
        "power_local_dip_max"
    ] = safe_aggregate(
        dip_matrix,
        "max",
    )

    metrics[
        "power_local_dip_median"
    ] = safe_aggregate(
        dip_matrix,
        "median",
    )

    metrics[
        "power_drop_from_previous_max"
    ] = safe_aggregate(
        previous_drop_matrix,
        "max",
    )

    metrics[
        "power_next_row_recovery_max"
    ] = safe_aggregate(
        next_recovery_matrix,
        "max",
    )

    metrics[
        "power_range_local_change_max"
    ] = safe_aggregate(
        range_change_matrix,
        "max",
    )

    metrics[
        "power_std_local_change_max"
    ] = safe_aggregate(
        std_change_matrix,
        "max",
    )

    metrics[
        "power_avg_median"
    ] = safe_aggregate(
        average_matrix,
        "median",
    )

    metrics[
        "power_min_median"
    ] = safe_aggregate(
        minimum_matrix,
        "median",
    )

    metrics[
        "power_max_median"
    ] = safe_aggregate(
        maximum_matrix,
        "median",
    )

    return (
        metrics,
        pd.DataFrame(
            per_base_rows
        ),
    )


# =============================================================================
# Wind context
# =============================================================================

def add_wind_context(
    metrics: pd.DataFrame,
    window: pd.DataFrame,
    measurement_columns: list[str],
) -> pd.DataFrame:
    wind_average_columns = [
        column
        for column in measurement_columns
        if (
            get_family(column)
            == "wind_speed"
            and get_stat_type(column)
            == "avg"
        )
    ]

    if not wind_average_columns:
        metrics[
            "wind_speed_avg_median"
        ] = np.nan

        return metrics

    wind_values = window[
        wind_average_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        wind_median = np.nanmedian(
            wind_values,
            axis=1,
        )

    metrics[
        "wind_speed_avg_median"
    ] = wind_median

    del wind_values
    gc.collect()

    return metrics


# =============================================================================
# Relative ranking without a decision threshold
# =============================================================================

def add_exploratory_rank_score(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine percentile ranks of complementary exploratory quantities.

    The result is only a relative browsing/ranking score from 0 to 1.
    It is not a fault probability and has no anomaly cut-off.
    """

    ranking_columns = [
        "all_local_change_top20_mean",
        "all_local_change_p99",
        "power_local_dip_max",
        "power_drop_from_previous_max",
        "power_next_row_recovery_max",
        "power_range_local_change_max",
        "power_std_local_change_max",
    ]

    rank_columns = []

    for column in ranking_columns:
        if column not in metrics.columns:
            continue

        rank_column = (
            f"rank_{column}"
        )

        metrics[
            rank_column
        ] = metrics[
            column
        ].rank(
            method="average",
            pct=True,
        )

        rank_columns.append(
            rank_column
        )

    if not rank_columns:
        metrics[
            "exploratory_composite_rank"
        ] = 0.0
    else:
        metrics[
            "exploratory_composite_rank"
        ] = metrics[
            rank_columns
        ].mean(
            axis=1
        )

    event_mask = (
        metrics["period"]
        == "metadata_event_interval"
    )

    metrics[
        "event_only_rank"
    ] = np.nan

    if event_mask.any():
        metrics.loc[
            event_mask,
            "event_only_rank",
        ] = metrics.loc[
            event_mask,
            "exploratory_composite_rank",
        ].rank(
            method="first",
            ascending=False,
        )

    return metrics


# =============================================================================
# Candidate windows and contributors
# =============================================================================

def select_top_candidate_windows(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows = metrics[
        metrics["period"]
        == "metadata_event_interval"
    ].copy()

    top_composite = (
        event_rows.sort_values(
            "exploratory_composite_rank",
            ascending=False,
        )
        .head(TOP_CANDIDATE_WINDOWS)
        .reset_index(drop=True)
    )

    top_composite[
        "candidate_rank"
    ] = np.arange(
        1,
        len(top_composite) + 1,
    )

    top_power_dip = (
        event_rows.sort_values(
            "power_local_dip_max",
            ascending=False,
        )
        .head(TOP_CANDIDATE_WINDOWS)
        .reset_index(drop=True)
    )

    top_power_dip[
        "candidate_rank"
    ] = np.arange(
        1,
        len(top_power_dip) + 1,
    )

    top_local_change = (
        event_rows.sort_values(
            "all_local_change_top20_mean",
            ascending=False,
        )
        .head(TOP_CANDIDATE_WINDOWS)
        .reset_index(drop=True)
    )

    top_local_change[
        "candidate_rank"
    ] = np.arange(
        1,
        len(top_local_change) + 1,
    )

    return (
        top_composite,
        top_power_dip,
        top_local_change,
    )


def build_candidate_contributors(
    candidates: pd.DataFrame,
    window: pd.DataFrame,
    score_df: pd.DataFrame,
    centre_df: pd.DataFrame,
    scale_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    scale_lookup = (
        scale_df.set_index(
            "measurement"
        )[
            "comparison_scale"
        ]
        .to_dict()
    )

    measurement_records: list[
        dict[str, object]
    ] = []

    base_records: list[
        dict[str, object]
    ] = []

    for _, candidate in candidates.iterrows():
        row_position = int(
            candidate["row_position"]
        )

        ranked_measurements = (
            score_df.iloc[
                row_position
            ]
            .sort_values(
                ascending=False
            )
            .head(
                TOP_CONTRIBUTORS_PER_WINDOW
            )
        )

        candidate_measurement_rows = []

        for contributor_rank, (
            measurement,
            local_score,
        ) in enumerate(
            ranked_measurements.items(),
            start=1,
        ):
            current_value = pd.to_numeric(
                window[measurement],
                errors="coerce",
            ).iloc[
                row_position
            ]

            local_centre = centre_df[
                measurement
            ].iloc[
                row_position
            ]

            absolute_difference = (
                abs(
                    current_value
                    - local_centre
                )
                if (
                    pd.notna(current_value)
                    and pd.notna(local_centre)
                )
                else np.nan
            )

            record = {
                "candidate_rank": int(
                    candidate[
                        "candidate_rank"
                    ]
                ),
                "candidate_time": candidate[
                    "time_stamp"
                ],
                "candidate_id": int(
                    candidate["id"]
                ),
                "candidate_row_position": row_position,
                "exploratory_composite_rank": float(
                    candidate[
                        "exploratory_composite_rank"
                    ]
                ),
                "power_local_dip_max": float(
                    candidate[
                        "power_local_dip_max"
                    ]
                ),
                "power_range_local_change_max": float(
                    candidate[
                        "power_range_local_change_max"
                    ]
                ),
                "power_std_local_change_max": float(
                    candidate[
                        "power_std_local_change_max"
                    ]
                ),
                "contributor_rank": contributor_rank,
                "measurement": measurement,
                "base_signal": get_base_signal(
                    measurement
                ),
                "stat_type": get_stat_type(
                    measurement
                ),
                "family": get_family(
                    measurement
                ),
                "current_value": current_value,
                "local_neighbour_median": local_centre,
                "absolute_local_difference": absolute_difference,
                "comparison_scale": float(
                    scale_lookup.get(
                        measurement,
                        np.nan,
                    )
                ),
                "local_change_score": float(
                    local_score
                ),
            }

            measurement_records.append(
                record
            )

            candidate_measurement_rows.append(
                record
            )

        candidate_measurement_df = pd.DataFrame(
            candidate_measurement_rows
        )

        if not candidate_measurement_df.empty:
            candidate_base_df = (
                candidate_measurement_df.sort_values(
                    "local_change_score",
                    ascending=False,
                )
                .drop_duplicates(
                    subset=["base_signal"],
                    keep="first",
                )
                .head(
                    TOP_CONTRIBUTORS_PER_WINDOW
                )
                .copy()
            )

            candidate_base_df[
                "base_contributor_rank"
            ] = np.arange(
                1,
                len(candidate_base_df) + 1,
            )

            base_records.extend(
                candidate_base_df.to_dict(
                    orient="records"
                )
            )

    return (
        pd.DataFrame(
            measurement_records
        ),
        pd.DataFrame(
            base_records
        ),
    )


# =============================================================================
# Measurement and base-signal summaries
# =============================================================================

def build_event_measurement_summary(
    score_df: pd.DataFrame,
    window_metrics: pd.DataFrame,
    measurement_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_positions = window_metrics.loc[
        window_metrics["period"]
        == "metadata_event_interval",
        "row_position",
    ].astype(
        int
    ).to_numpy()

    if len(event_positions) == 0:
        raise ValueError(
            "No rows are present inside the metadata event interval."
        )

    event_scores = score_df.iloc[
        event_positions
    ].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        maximum_scores = np.nanmax(
            event_scores,
            axis=0,
        )

        p95_scores = np.nanpercentile(
            event_scores,
            95,
            axis=0,
        )

        mean_scores = np.nanmean(
            event_scores,
            axis=0,
        )

    measurement_summary = pd.DataFrame({
        "measurement": measurement_columns,
        "base_signal": [
            get_base_signal(column)
            for column in measurement_columns
        ],
        "stat_type": [
            get_stat_type(column)
            for column in measurement_columns
        ],
        "family": [
            get_family(column)
            for column in measurement_columns
        ],
        "event_max_local_change_score": maximum_scores,
        "event_p95_local_change_score": p95_scores,
        "event_mean_local_change_score": mean_scores,
    })

    measurement_summary = (
        measurement_summary.sort_values(
            "event_max_local_change_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    base_summary = (
        measurement_summary.groupby(
            [
                "base_signal",
                "family",
            ]
        )
        .agg(
            n_statistics=(
                "measurement",
                "count",
            ),
            max_local_change_score=(
                "event_max_local_change_score",
                "max",
            ),
            max_p95_local_change_score=(
                "event_p95_local_change_score",
                "max",
            ),
            mean_of_statistic_scores=(
                "event_mean_local_change_score",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "max_local_change_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        measurement_summary,
        base_summary,
    )


# =============================================================================
# Daily candidate summary
# =============================================================================

def build_daily_candidate_summary(
    top_candidates: pd.DataFrame,
) -> pd.DataFrame:
    if top_candidates.empty:
        return pd.DataFrame()

    daily = top_candidates.copy()

    daily[
        "date"
    ] = daily[
        "time_stamp"
    ].dt.date

    return (
        daily.groupby(
            "date"
        )
        .agg(
            n_top_ranked_windows=(
                "time_stamp",
                "count",
            ),
            highest_composite_rank=(
                "exploratory_composite_rank",
                "max",
            ),
            highest_power_dip_score=(
                "power_local_dip_max",
                "max",
            ),
            highest_local_change_score=(
                "all_local_change_top20_mean",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "date"
        )
    )


# =============================================================================
# Plotting
# =============================================================================

def add_event_markers() -> None:
    plt.axvline(
        pd.to_datetime(
            EVENT_START_TIME
        ),
        linestyle="--",
        linewidth=2,
        label="metadata start",
    )

    plt.axvline(
        pd.to_datetime(
            EVENT_END_TIME
        ),
        linestyle="--",
        linewidth=2,
        label="metadata end",
    )


def plot_metric_timeline(
    metrics: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    filename: str,
    candidates: pd.DataFrame | None = None,
) -> None:
    plt.figure(
        figsize=(16, 5)
    )

    plt.plot(
        metrics["time_stamp"],
        metrics[metric_column],
        linewidth=1,
        label=metric_column,
    )

    add_event_markers()

    if (
        candidates is not None
        and not candidates.empty
    ):
        plt.scatter(
            candidates["time_stamp"],
            candidates[metric_column],
            s=30,
            label="top ranked event windows",
        )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_top_measurement_bar(
    measurement_summary: pd.DataFrame,
) -> None:
    top = measurement_summary.head(
        20
    ).copy()

    if top.empty:
        return

    plot_df = top.sort_values(
        "event_max_local_change_score",
        ascending=True,
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        plot_df["measurement"],
        plot_df[
            "event_max_local_change_score"
        ],
    )

    plt.xlabel(
        "Maximum local-change score during Event 35"
    )
    plt.ylabel("Measurement")
    plt.title(
        "Event 35 Top 20 measurements by local change"
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "event35_top20_measurements_local_change.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_top_base_signal_bar(
    base_summary: pd.DataFrame,
) -> None:
    top = base_summary.head(
        20
    ).copy()

    if top.empty:
        return

    plot_df = top.sort_values(
        "max_local_change_score",
        ascending=True,
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        plot_df["base_signal"],
        plot_df[
            "max_local_change_score"
        ],
    )

    plt.xlabel(
        "Maximum local-change score during Event 35"
    )
    plt.ylabel("Base signal")
    plt.title(
        "Event 35 Top 20 base signals "
        "(avg/max/min/std deduplicated)"
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "event35_top20_base_signals_local_change.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_measurement_time_series(
    window: pd.DataFrame,
    measurement: str,
    candidates: pd.DataFrame,
) -> None:
    if measurement not in window.columns:
        return

    values = pd.to_numeric(
        window[measurement],
        errors="coerce",
    )

    plt.figure(
        figsize=(16, 5)
    )

    plt.plot(
        window["time_stamp"],
        values,
        linewidth=1,
        label=measurement,
    )

    add_event_markers()

    if not candidates.empty:
        candidate_positions = candidates[
            "row_position"
        ].astype(
            int
        ).to_numpy()

        valid_positions = candidate_positions[
            (
                candidate_positions >= 0
            )
            & (
                candidate_positions
                < len(window)
            )
        ]

        if len(valid_positions) > 0:
            plt.scatter(
                window.iloc[
                    valid_positions
                ][
                    "time_stamp"
                ],
                values.iloc[
                    valid_positions
                ],
                s=25,
                label="top ranked windows",
            )

    plt.title(
        f"{measurement}: Event 35 exploratory timeline"
    )
    plt.xlabel("Time")
    plt.ylabel(measurement)
    plt.legend()
    plt.tight_layout()

    safe_name = (
        measurement.replace("/", "_")
        .replace("\\", "_")
    )

    plt.savefig(
        OUTPUT_DIR
        / f"{safe_name}_event35_timeline.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_candidate_context(
    metrics: pd.DataFrame,
    candidate: pd.Series,
) -> None:
    candidate_time = pd.to_datetime(
        candidate[
            "time_stamp"
        ]
    )

    context_start = (
        candidate_time
        - pd.Timedelta(
            hours=CANDIDATE_CONTEXT_HOURS
        )
    )

    context_end = (
        candidate_time
        + pd.Timedelta(
            hours=CANDIDATE_CONTEXT_HOURS
        )
    )

    context = metrics[
        (
            metrics["time_stamp"]
            >= context_start
        )
        & (
            metrics["time_stamp"]
            <= context_end
        )
    ].copy()

    if context.empty:
        return

    plt.figure(
        figsize=(14, 5)
    )

    columns_to_plot = [
        "all_local_change_top20_mean",
        "power_local_dip_max",
        "power_range_local_change_max",
        "power_std_local_change_max",
        "power_next_row_recovery_max",
    ]

    for column in columns_to_plot:
        if column in context.columns:
            plt.plot(
                context["time_stamp"],
                context[column],
                linewidth=1,
                label=column,
            )

    plt.axvline(
        candidate_time,
        linestyle="--",
        linewidth=2,
        label="candidate timestamp",
    )

    plt.title(
        "Event 35 candidate context: "
        f"{candidate_time:%Y-%m-%d %H:%M}"
    )
    plt.xlabel("Time")
    plt.ylabel("Exploratory local ranking quantity")
    plt.legend()
    plt.tight_layout()

    rank = int(
        candidate[
            "candidate_rank"
        ]
    )

    plt.savefig(
        OUTPUT_DIR
        / (
            f"event35_candidate_{rank:02d}_"
            f"{candidate_time:%Y%m%d_%H%M}_context.png"
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# Markdown summary
# =============================================================================

def format_number(
    value: object,
) -> str:
    if pd.isna(value):
        return "NA"

    try:
        return f"{float(value):.4g}"
    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def write_markdown_summary(
    window: pd.DataFrame,
    metrics: pd.DataFrame,
    top_candidates: pd.DataFrame,
    measurement_summary: pd.DataFrame,
    base_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    load_info: dict[str, object],
    sample_minutes: float,
) -> None:
    summary_path = (
        OUTPUT_DIR
        / "event35_exploratory_summary.md"
    )

    event_rows = metrics[
        metrics["period"]
        == "metadata_event_interval"
    ]

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# Event 35 Short-Standstill Exploratory Analysis\n\n"
        )

        file.write("## Event information\n\n")
        file.write(f"- Farm ID: `{FARM_ID}`\n")
        file.write(f"- Event ID: `{EVENT_ID}`\n")
        file.write("- Label: `anomaly`\n")
        file.write(
            f"- Metadata start: `{EVENT_START_TIME}` "
            f"/ id `{EVENT_START_ID}`\n"
        )
        file.write(
            f"- Metadata end: `{EVENT_END_TIME}` "
            f"/ id `{EVENT_END_ID}`\n"
        )
        file.write(
            f"- Description: `{EVENT_DESCRIPTION}`\n"
        )
        file.write(
            f"- Measurement mode: `{MEASUREMENT_MODE}`\n"
        )
        file.write(
            f"- Inferred sampling interval: "
            f"`{sample_minutes:.2f} minutes`\n\n"
        )

        file.write("## Important method note\n\n")
        file.write(
            "This script performs exploratory ranking only. It does not "
            "define a fixed baseline, z-score threshold, abnormal-fraction "
            "threshold, rolling alarm rule or minimum anomaly duration. "
            "Each 10-minute row is compared with nearby rows to surface "
            "short local changes. A high ranking is not a confirmed fault "
            "or confirmed standstill.\n\n"
        )

        file.write("## Data coverage\n\n")
        file.write(
            f"- Rows read from CSV: "
            f"`{load_info['total_rows_read']}`\n"
        )
        file.write(
            f"- Rows retained: "
            f"`{load_info['total_rows_retained']}`\n"
        )
        file.write(
            f"- Actual retained start: "
            f"`{load_info['window_start']}`\n"
        )
        file.write(
            f"- Actual retained end: "
            f"`{load_info['window_end']}`\n"
        )
        file.write(
            f"- Selected measurements: "
            f"`{load_info['n_selected_measurements']}`\n"
        )
        file.write(
            f"- Rows inside metadata interval: "
            f"`{len(event_rows)}`\n\n"
        )

        file.write(
            "## Top ranked 10-minute windows inside the metadata interval\n\n"
        )

        file.write(
            "| Rank | Time | ID | Composite rank | "
            "Top-20 local change | Power dip | Power range change | "
            "Power std change | Next-row recovery | Wind-speed median |\n"
        )

        file.write(
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        for _, row in top_candidates.head(
            20
        ).iterrows():
            file.write(
                f"| {int(row['candidate_rank'])} "
                f"| {row['time_stamp']} "
                f"| {int(row['id'])} "
                f"| {format_number(row['exploratory_composite_rank'])} "
                f"| {format_number(row['all_local_change_top20_mean'])} "
                f"| {format_number(row['power_local_dip_max'])} "
                f"| {format_number(row['power_range_local_change_max'])} "
                f"| {format_number(row['power_std_local_change_max'])} "
                f"| {format_number(row['power_next_row_recovery_max'])} "
                f"| {format_number(row['wind_speed_avg_median'])} |\n"
            )

        file.write(
            "\n## Top changed measurements during Event 35\n\n"
        )

        file.write(
            "| Rank | Measurement | Base signal | Family | Type | "
            "Maximum local score | P95 local score | Mean local score |\n"
        )

        file.write(
            "|---:|---|---|---|---|---:|---:|---:|\n"
        )

        for rank, (
            _,
            row,
        ) in enumerate(
            measurement_summary.head(
                20
            ).iterrows(),
            start=1,
        ):
            file.write(
                f"| {rank} "
                f"| `{row['measurement']}` "
                f"| `{row['base_signal']}` "
                f"| `{row['family']}` "
                f"| `{row['stat_type']}` "
                f"| {format_number(row['event_max_local_change_score'])} "
                f"| {format_number(row['event_p95_local_change_score'])} "
                f"| {format_number(row['event_mean_local_change_score'])} |\n"
            )

        file.write(
            "\n## Top base signals during Event 35\n\n"
        )

        file.write(
            "| Rank | Base signal | Family | Statistics available | "
            "Maximum local score | Maximum P95 score |\n"
        )

        file.write(
            "|---:|---|---|---:|---:|---:|\n"
        )

        for rank, (
            _,
            row,
        ) in enumerate(
            base_summary.head(
                20
            ).iterrows(),
            start=1,
        ):
            file.write(
                f"| {rank} "
                f"| `{row['base_signal']}` "
                f"| `{row['family']}` "
                f"| {int(row['n_statistics'])} "
                f"| {format_number(row['max_local_change_score'])} "
                f"| {format_number(row['max_p95_local_change_score'])} |\n"
            )

        file.write(
            "\n## Distribution of the top-ranked windows by day\n\n"
        )

        if daily_summary.empty:
            file.write(
                "No daily candidate summary was generated.\n"
            )
        else:
            file.write(
                "| Date | Top-ranked windows | Highest composite rank | "
                "Highest power dip | Highest local change |\n"
            )

            file.write(
                "|---|---:|---:|---:|---:|\n"
            )

            for _, row in daily_summary.iterrows():
                file.write(
                    f"| {row['date']} "
                    f"| {int(row['n_top_ranked_windows'])} "
                    f"| {format_number(row['highest_composite_rank'])} "
                    f"| {format_number(row['highest_power_dip_score'])} "
                    f"| {format_number(row['highest_local_change_score'])} |\n"
                )

        file.write(
            "\n## How to interpret the outputs\n\n"
        )

        file.write(
            "Because the reported standstills lasted no more than eight "
            "minutes while the SCADA sampling interval is approximately "
            "ten minutes, the exact start and end of a stop cannot be "
            "recovered. The main targets are isolated 10-minute rows where "
            "average power drops relative to neighbouring rows, minimum "
            "power becomes low while maximum power remains higher, "
            "within-window variability increases, and the following row "
            "shows recovery. Wind-speed context must be checked before "
            "interpreting a power reduction as fault-related.\n"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print(
        "Event 35 Exploratory Analysis for Short Standstill Windows"
    )
    print("=" * 100)

    print("\nEvent description:")
    print(EVENT_DESCRIPTION)

    print("\nNo fixed anomaly thresholds are used.")
    print(
        "Each 10-minute row is ranked relative to nearby rows."
    )

    print("\nLoading raw SCADA file:")
    print(RAW_FILE_PATH)

    (
        window,
        measurement_columns,
        load_info,
    ) = load_analysis_window(
        RAW_FILE_PATH
    )

    sample_minutes = infer_sample_minutes(
        window
    )

    print(
        f"\nInferred sampling interval: "
        f"{sample_minutes:.2f} minutes"
    )

    event_start = pd.to_datetime(
        EVENT_START_TIME
    )

    event_end = pd.to_datetime(
        EVENT_END_TIME
    )

    event_window = window[
        (
            window["time_stamp"]
            >= event_start
        )
        & (
            window["time_stamp"]
            <= event_end
        )
    ]

    if event_window.empty:
        raise ValueError(
            "The metadata event interval is empty in the retained data."
        )

    print(
        f"Rows inside metadata interval: "
        f"{len(event_window):,}"
    )

    pd.DataFrame({
        "measurement": measurement_columns,
        "base_signal": [
            get_base_signal(column)
            for column in measurement_columns
        ],
        "stat_type": [
            get_stat_type(column)
            for column in measurement_columns
        ],
        "family": [
            get_family(column)
            for column in measurement_columns
        ],
    }).to_csv(
        OUTPUT_DIR
        / "event35_selected_measurements_all.csv",
        index=False,
    )

    print(
        "\nCalculating local-neighbour measurement scores..."
    )

    (
        score_df,
        centre_df,
        scale_df,
    ) = calculate_local_measurement_scores(
        window=window,
        measurement_columns=measurement_columns,
    )

    scale_df.to_csv(
        OUTPUT_DIR
        / "event35_measurement_comparison_scales.csv",
        index=False,
    )

    print(
        "\nBuilding general row-level metrics..."
    )

    metrics = build_general_window_metrics(
        window=window,
        score_df=score_df,
        measurement_columns=measurement_columns,
    )

    print(
        "Adding power-specific short-standstill signatures..."
    )

    (
        metrics,
        power_base_summary,
    ) = add_power_signature_metrics(
        metrics=metrics,
        window=window,
        centre_df=centre_df,
        scale_df=scale_df,
        measurement_columns=measurement_columns,
    )

    power_base_summary.to_csv(
        OUTPUT_DIR
        / "event35_power_base_signal_summary.csv",
        index=False,
    )

    metrics = add_wind_context(
        metrics=metrics,
        window=window,
        measurement_columns=measurement_columns,
    )

    metrics = add_exploratory_rank_score(
        metrics
    )

    metrics.to_csv(
        OUTPUT_DIR
        / "event35_all_10min_window_metrics.csv",
        index=False,
    )

    print(
        "Selecting top-ranked windows inside the metadata interval..."
    )

    (
        top_composite,
        top_power_dip,
        top_local_change,
    ) = select_top_candidate_windows(
        metrics
    )

    top_composite.to_csv(
        OUTPUT_DIR
        / "event35_top30_composite_candidate_windows.csv",
        index=False,
    )

    top_power_dip.to_csv(
        OUTPUT_DIR
        / "event35_top30_power_dip_windows.csv",
        index=False,
    )

    top_local_change.to_csv(
        OUTPUT_DIR
        / "event35_top30_local_change_windows.csv",
        index=False,
    )

    print(
        "Building candidate contributor tables..."
    )

    (
        candidate_contributors,
        candidate_base_contributors,
    ) = build_candidate_contributors(
        candidates=top_composite,
        window=window,
        score_df=score_df,
        centre_df=centre_df,
        scale_df=scale_df,
    )

    candidate_contributors.to_csv(
        OUTPUT_DIR
        / "event35_candidate_measurement_contributors.csv",
        index=False,
    )

    candidate_base_contributors.to_csv(
        OUTPUT_DIR
        / "event35_candidate_base_signal_contributors.csv",
        index=False,
    )

    print(
        "Building event measurement summaries..."
    )

    (
        measurement_summary,
        base_summary,
    ) = build_event_measurement_summary(
        score_df=score_df,
        window_metrics=metrics,
        measurement_columns=measurement_columns,
    )

    measurement_summary.to_csv(
        OUTPUT_DIR
        / "event35_measurement_local_change_summary.csv",
        index=False,
    )

    base_summary.to_csv(
        OUTPUT_DIR
        / "event35_base_signal_local_change_summary.csv",
        index=False,
    )

    daily_summary = build_daily_candidate_summary(
        top_composite
    )

    daily_summary.to_csv(
        OUTPUT_DIR
        / "event35_daily_top_candidate_summary.csv",
        index=False,
    )

    print(
        "\nTop 20 exploratory candidate windows:"
    )

    display_columns = [
        "candidate_rank",
        "time_stamp",
        "id",
        "exploratory_composite_rank",
        "all_local_change_top20_mean",
        "power_local_dip_max",
        "power_drop_from_previous_max",
        "power_range_local_change_max",
        "power_std_local_change_max",
        "power_next_row_recovery_max",
        "wind_speed_avg_median",
    ]

    print(
        top_composite[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print(
        "\nTop 20 base signals by maximum local change:"
    )

    print(
        base_summary.head(
            20
        ).to_string(
            index=False
        )
    )

    print("\nSaving timeline plots...")

    plot_metric_timeline(
        metrics=metrics,
        metric_column=(
            "exploratory_composite_rank"
        ),
        title=(
            "Event 35 exploratory composite rank "
            "for each 10-minute row"
        ),
        ylabel=(
            "Relative rank from 0 to 1"
        ),
        filename=(
            "event35_composite_rank_timeline.png"
        ),
        candidates=top_composite,
    )

    plot_metric_timeline(
        metrics=metrics,
        metric_column=(
            "all_local_change_top20_mean"
        ),
        title=(
            "Event 35 mean of the 20 largest "
            "measurement-level local changes"
        ),
        ylabel=(
            "Top-20 mean local-change score"
        ),
        filename=(
            "event35_top20_local_change_timeline.png"
        ),
        candidates=top_composite,
    )

    plot_metric_timeline(
        metrics=metrics,
        metric_column=(
            "power_local_dip_max"
        ),
        title=(
            "Event 35 maximum local power-dip signature"
        ),
        ylabel=(
            "Local power-dip ranking quantity"
        ),
        filename=(
            "event35_power_dip_timeline.png"
        ),
        candidates=top_power_dip,
    )

    plot_metric_timeline(
        metrics=metrics,
        metric_column=(
            "power_range_local_change_max"
        ),
        title=(
            "Event 35 maximum local change "
            "in within-window power range"
        ),
        ylabel=(
            "Power range local-change score"
        ),
        filename=(
            "event35_power_range_change_timeline.png"
        ),
        candidates=top_composite,
    )

    plot_metric_timeline(
        metrics=metrics,
        metric_column=(
            "power_std_local_change_max"
        ),
        title=(
            "Event 35 maximum local change "
            "in reported power standard deviation"
        ),
        ylabel=(
            "Power std local-change score"
        ),
        filename=(
            "event35_power_std_change_timeline.png"
        ),
        candidates=top_composite,
    )

    plot_metric_timeline(
        metrics=metrics,
        metric_column=(
            "power_next_row_recovery_max"
        ),
        title=(
            "Event 35 next-row power recovery signature"
        ),
        ylabel=(
            "Next-row recovery ranking quantity"
        ),
        filename=(
            "event35_next_row_recovery_timeline.png"
        ),
        candidates=top_composite,
    )

    plot_top_measurement_bar(
        measurement_summary
    )

    plot_top_base_signal_bar(
        base_summary
    )

    top_measurements = (
        measurement_summary[
            "measurement"
        ]
        .head(
            TOP_MEASUREMENTS_TO_PLOT
        )
        .tolist()
    )

    for measurement in top_measurements:
        try:
            plot_measurement_time_series(
                window=window,
                measurement=measurement,
                candidates=top_composite,
            )
        except Exception as error:
            print(
                f"Could not plot {measurement}: {error}"
            )

    for _, candidate in top_composite.head(
        TOP_CANDIDATE_CONTEXT_PLOTS
    ).iterrows():
        try:
            plot_candidate_context(
                metrics=metrics,
                candidate=candidate,
            )
        except Exception as error:
            print(
                "Could not plot candidate context "
                f"for {candidate['time_stamp']}: "
                f"{error}"
            )

    write_markdown_summary(
        window=window,
        metrics=metrics,
        top_candidates=top_composite,
        measurement_summary=measurement_summary,
        base_summary=base_summary,
        daily_summary=daily_summary,
        load_info=load_info,
        sample_minutes=sample_minutes,
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nMain output files:")
    print("- event35_all_10min_window_metrics.csv")
    print("- event35_top30_composite_candidate_windows.csv")
    print("- event35_top30_power_dip_windows.csv")
    print("- event35_top30_local_change_windows.csv")
    print("- event35_candidate_measurement_contributors.csv")
    print("- event35_candidate_base_signal_contributors.csv")
    print("- event35_measurement_local_change_summary.csv")
    print("- event35_base_signal_local_change_summary.csv")
    print("- event35_daily_top_candidate_summary.csv")
    print("- event35_exploratory_summary.md")
    print("- composite, power-dip, power-range and power-std timelines")
    print("- Top 20 measurement and base-signal bar charts")
    print("- Top measurement time-series plots")
    print("- Top candidate context plots")

    print("\nImportant:")
    print(
        "These are exploratory rankings, not confirmed standstill detections."
    )
    print(
        "Inspect the top candidate rows and their surrounding power/wind "
        "patterns before selecting any thresholds."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()