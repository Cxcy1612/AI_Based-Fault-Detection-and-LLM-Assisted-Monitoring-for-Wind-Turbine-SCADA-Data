#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate a high-sensitivity streaming alert classifier.

The input table must be produced by ``build_streaming_window_dataset.py``.
Windows from the same group are never split between development and holdout
sets when enough groups are available.  The default operating objective is
recall-first: among probability thresholds whose normal false-positive rate is
at most ``--max-normal-fpr``, choose the threshold with the highest anomaly
recall, then F2, then balanced accuracy.

Two model artefacts are distinguished:
1. Honest chronological holdout evaluation: latest groups are untouched during
   model/threshold selection.
2. Deployment bundle: after evaluation, the selected model is refit on all
   available snapshots and saved with an all-data grouped-OOF threshold.

Example
-------
python train_streaming_alert_model.py ^
  --input-file "outputs/farmC_streaming_dataset/streaming_window_features.csv" ^
  --output-dir "outputs/farmC_streaming_model" ^
  --group-column group_id ^
  --cv-splits 5 ^
  --test-fraction 0.20 ^
  --max-normal-fpr 0.35
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

TRUE_TEXT = {"true", "1", "yes", "y", "t"}
TARGET_COLUMN = "target_anomaly"

NON_FEATURE_COLUMNS = {
    "farm_id",
    "source_id",
    "source_file",
    "asset_id",
    "asset_group_id",
    "source_group_id",
    "group_id",
    "snapshot_id",
    "timestamp",
    "row_id",
    "metadata_event_id",
    "metadata_label",
    "metadata_start",
    "metadata_end",
    "inside_metadata_interval",
    "target_anomaly",
    "target_anomaly_current",
    "prediction_horizon_hours",
    "source_row_score_file",
    "sampling_minutes",
    # These are outputs of the alert state machine and would copy the rule
    # decision rather than learn from detector evidence.
    "stream_state",
    "review_flag",
    "active_alert_flag",
}


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    recall: float
    precision: float
    f2: float
    f1: float
    balanced_accuracy: float
    specificity: float
    normal_false_positive_rate: float
    feasible: bool


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline
    param_grid: dict[str, list[Any]]


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(TRUE_TEXT)
    )


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    probabilities: Optional[np.ndarray | pd.Series] = None,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result: dict[str, Any] = {
        "n": int(len(truth)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "precision_anomaly": float(precision_score(truth, predicted, zero_division=0)),
        "recall_anomaly": float(recall_score(truth, predicted, zero_division=0)),
        "f1_anomaly": float(f1_score(truth, predicted, zero_division=0)),
        "f2_anomaly": float(fbeta_score(truth, predicted, beta=2, zero_division=0)),
        "specificity_normal": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "normal_false_positive_rate": float(fp / (tn + fp)) if (tn + fp) else np.nan,
    }
    if probabilities is not None and len(np.unique(truth)) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(truth, np.asarray(probabilities, dtype=float)))
        except ValueError:
            result["roc_auc"] = np.nan
    else:
        result["roc_auc"] = np.nan
    return result


def select_recall_first_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    max_normal_fpr: float,
) -> ThresholdResult:
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    candidates: list[ThresholdResult] = []

    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        prediction = (probability >= threshold).astype(int)
        metrics = calculate_metrics(truth, prediction, probability)
        candidates.append(
            ThresholdResult(
                threshold=float(threshold),
                recall=float(metrics["recall_anomaly"]),
                precision=float(metrics["precision_anomaly"]),
                f2=float(metrics["f2_anomaly"]),
                f1=float(metrics["f1_anomaly"]),
                balanced_accuracy=float(metrics["balanced_accuracy"]),
                specificity=float(metrics["specificity_normal"]),
                normal_false_positive_rate=float(metrics["normal_false_positive_rate"]),
                feasible=float(metrics["normal_false_positive_rate"]) <= max_normal_fpr,
            )
        )

    feasible = [candidate for candidate in candidates if candidate.feasible]
    pool = feasible if feasible else candidates
    return max(
        pool,
        key=lambda item: (
            item.recall,
            item.f2,
            item.balanced_accuracy,
            item.precision,
            -abs(item.threshold - 0.5),
        ),
    )


def build_model_specs(random_state: int) -> list[ModelSpec]:
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    tree = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return [
        ModelSpec(
            "logistic_regression",
            logistic,
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        ModelSpec(
            "shallow_decision_tree",
            tree,
            {
                "model__max_depth": [2, 3, 4, 5],
                "model__min_samples_leaf": [10, 25, 50],
            },
        ),
        ModelSpec(
            "restricted_random_forest",
            forest,
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [10, 25, 50],
                "model__max_features": ["sqrt", 0.30],
            },
        ),
    ]


def load_dataset(
    path: Path,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    data = pd.read_csv(path, low_memory=False)
    required = {TARGET_COLUMN, "timestamp", group_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    data[group_column] = data[group_column].fillna("").astype(str)
    empty_group = data[group_column].str.len() == 0
    data.loc[empty_group, group_column] = "row_group__" + data.index[empty_group].astype(str)

    feature_columns: list[str] = []
    numeric_columns: dict[str, pd.Series] = {}
    for column in data.columns:
        if column in NON_FEATURE_COLUMNS or column == group_column:
            continue
        series = data[column]
        if pd.api.types.is_bool_dtype(series):
            numeric = series.astype(int)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            # Text-like columns become all missing and are excluded.
            if numeric.notna().sum() == 0:
                continue
        if numeric.nunique(dropna=True) <= 1:
            continue
        feature_columns.append(column)
        numeric_columns[column] = numeric.astype(float)

    if not feature_columns:
        raise ValueError("No usable numeric feature columns were found.")

    X = pd.DataFrame(numeric_columns, index=data.index)
    y = data[TARGET_COLUMN].astype(int)
    groups = data[group_column].astype(str)
    return data, X, y, groups, feature_columns


def group_sort_time(group: pd.DataFrame) -> pd.Timestamp:
    if "metadata_start" in group:
        values = pd.to_datetime(group["metadata_start"], errors="coerce").dropna()
        if not values.empty:
            return pd.Timestamp(values.min())
    return pd.Timestamp(group["timestamp"].min())


def chronological_group_holdout(
    metadata: pd.DataFrame,
    groups: pd.Series,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    unique_groups = groups.drop_duplicates().tolist()
    if len(unique_groups) < 4:
        raise ValueError("At least four groups are required for group holdout.")

    group_table = []
    for group_id in unique_groups:
        subset = metadata.loc[groups == group_id]
        group_table.append(
            {
                "group_id": group_id,
                "sort_time": group_sort_time(subset),
                "has_anomaly": bool((subset[TARGET_COLUMN] == 1).any()),
                "rows": int(len(subset)),
            }
        )
    summary = pd.DataFrame(group_table).sort_values(["sort_time", "group_id"])
    test_count = max(1, int(math.ceil(len(summary) * test_fraction)))
    test_count = min(test_count, len(summary) - 2)
    test_groups = summary.tail(test_count)["group_id"].tolist()

    # Ensure the holdout contains at least one labelled anomaly group when one
    # exists in the full dataset.
    test_summary = summary.loc[summary["group_id"].isin(test_groups)]
    if summary["has_anomaly"].any() and not test_summary["has_anomaly"].any():
        latest_anomaly = summary.loc[summary["has_anomaly"]].iloc[-1]["group_id"]
        test_groups[-1] = latest_anomaly
        test_groups = list(dict.fromkeys(test_groups))

    test_mask = groups.isin(test_groups).to_numpy()
    train_mask = ~test_mask
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Chronological group holdout produced an empty split.")

    details = {
        "strategy": "chronological_group_holdout",
        "development_groups": int(groups.iloc[train_idx].nunique()),
        "test_groups": int(groups.iloc[test_idx].nunique()),
        "test_group_ids": sorted(groups.iloc[test_idx].unique().tolist()),
        "development_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "development_end": str(metadata.iloc[train_idx]["timestamp"].max()),
        "test_start": str(metadata.iloc[test_idx]["timestamp"].min()),
    }
    return train_idx, test_idx, details


def chronological_row_holdout(
    metadata: pd.DataFrame,
    test_fraction: float,
    purge_hours: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ordered = metadata.sort_values("timestamp")
    split_position = max(1, int(math.floor(len(ordered) * (1.0 - test_fraction))))
    split_position = min(split_position, len(ordered) - 1)
    test_start = pd.Timestamp(ordered.iloc[split_position]["timestamp"])
    train_end = test_start - pd.Timedelta(hours=purge_hours)
    train_idx = metadata.index[metadata["timestamp"] < train_end].to_numpy()
    test_idx = metadata.index[metadata["timestamp"] >= test_start].to_numpy()
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Chronological row holdout produced an empty split.")
    details = {
        "strategy": "chronological_row_holdout_with_purge",
        "purge_hours": purge_hours,
        "development_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "development_end": str(metadata.loc[train_idx, "timestamp"].max()),
        "test_start": str(metadata.loc[test_idx, "timestamp"].min()),
    }
    return train_idx, test_idx, details


def make_cv(
    y: pd.Series,
    groups: pd.Series,
    requested_splits: int,
    random_state: int,
):
    n_groups = groups.nunique()
    class_count = int(y.value_counts().min()) if y.nunique() == 2 else 0
    n_splits = max(2, min(requested_splits, n_groups, class_count))
    if n_groups >= n_splits:
        try:
            return StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            ), True
        except TypeError:
            return GroupKFold(n_splits=n_splits), True
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), False


def tune_and_oof(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv_splits: int,
    random_state: int,
    max_normal_fpr: float,
) -> tuple[BaseEstimator, dict[str, Any], np.ndarray, ThresholdResult, dict[str, Any]]:
    cv, uses_groups = make_cv(y, groups, cv_splits, random_state)
    search = GridSearchCV(
        estimator=clone(spec.estimator),
        param_grid=spec.param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    fit_kwargs = {"groups": groups} if uses_groups else {}
    search.fit(X, y, **fit_kwargs)
    best = search.best_estimator_

    predict_kwargs = {"groups": groups} if uses_groups else {}
    oof_probability = cross_val_predict(
        clone(best),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
        **predict_kwargs,
    )[:, 1]
    threshold = select_recall_first_threshold(y, oof_probability, max_normal_fpr)
    oof_prediction = (oof_probability >= threshold.threshold).astype(int)
    metrics = calculate_metrics(y, oof_prediction, oof_probability)
    best.fit(X, y)
    return best, search.best_params_, oof_probability, threshold, metrics


def model_selection_key(row: pd.Series, max_normal_fpr: float) -> tuple[float, ...]:
    feasible = float(row["normal_false_positive_rate"]) <= max_normal_fpr
    return (
        1.0 if feasible else 0.0,
        float(row["recall_anomaly"]),
        float(row["f2_anomaly"]),
        float(row["balanced_accuracy"]),
        float(row["roc_auc"]) if np.isfinite(row["roc_auc"]) else -1.0,
    )


def extract_feature_importance(
    estimator: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "tree_feature_importance"
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_[0], dtype=float)
        kind = "logistic_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "absolute_importance", "importance_type"])
    table = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
            "absolute_importance": np.abs(values),
            "importance_type": kind,
        }
    )
    return table.sort_values("absolute_importance", ascending=False).reset_index(drop=True)


def event_detection_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    positive = predictions.loc[
        (predictions["target_anomaly_current"] == 1)
        & predictions.get("metadata_event_id", "").fillna("").astype(str).ne("")
    ].copy()
    event_rows: list[dict[str, Any]] = []
    grouping = ["source_id", "metadata_event_id"]
    if not positive.empty:
        for keys, group in positive.groupby(grouping, dropna=False):
            group = group.sort_values("timestamp")
            predicted = group.loc[group["predicted_anomaly"] == 1]
            event_start_values = pd.to_datetime(group.get("metadata_start"), errors="coerce").dropna()
            event_end_values = pd.to_datetime(group.get("metadata_end"), errors="coerce").dropna()
            event_start = event_start_values.min() if not event_start_values.empty else group["timestamp"].min()
            event_end = event_end_values.max() if not event_end_values.empty else group["timestamp"].max()
            detected = not predicted.empty
            first_alert = predicted["timestamp"].min() if detected else pd.NaT
            delay = (
                max(0.0, (first_alert - event_start).total_seconds() / 3600.0)
                if detected else np.nan
            )
            event_rows.append(
                {
                    "source_id": keys[0],
                    "metadata_event_id": keys[1],
                    "event_start": event_start,
                    "event_end": event_end,
                    "detected": detected,
                    "first_alert_time": first_alert,
                    "detection_delay_hours": delay,
                    "positive_snapshots": int(len(group)),
                    "alerted_positive_snapshots": int((group["predicted_anomaly"] == 1).sum()),
                    "alert_coverage_fraction": float((group["predicted_anomaly"] == 1).mean()),
                }
            )
    event_table = pd.DataFrame(event_rows)
    if event_table.empty:
        summary = {
            "anomaly_events": 0,
            "detected_events": 0,
            "missed_events": 0,
            "event_detection_recall": np.nan,
            "median_detection_delay_hours": np.nan,
            "mean_detection_delay_hours": np.nan,
        }
    else:
        delays = pd.to_numeric(event_table.loc[event_table["detected"], "detection_delay_hours"], errors="coerce")
        summary = {
            "anomaly_events": int(len(event_table)),
            "detected_events": int(event_table["detected"].sum()),
            "missed_events": int((~event_table["detected"]).sum()),
            "event_detection_recall": float(event_table["detected"].mean()),
            "median_detection_delay_hours": float(delays.median()) if not delays.empty else np.nan,
            "mean_detection_delay_hours": float(delays.mean()) if not delays.empty else np.nan,
        }
    return event_table, summary


def false_alarm_episode_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id, group in predictions.groupby("evaluation_group", dropna=False):
        group = group.sort_values("timestamp").copy()
        false = group.loc[
            (group["target_anomaly_current"] == 0)
            & (group["predicted_anomaly"] == 1)
        ]
        if false.empty:
            continue
        all_diffs = group["timestamp"].diff().dt.total_seconds().div(60.0)
        typical_minutes = float(all_diffs[all_diffs > 0].median()) if bool((all_diffs > 0).any()) else 60.0
        maximum_gap = max(typical_minutes * 2.5, 30.0)
        episode_number = (false["timestamp"].diff().dt.total_seconds().div(60.0) > maximum_gap).cumsum()
        for _, episode in false.groupby(episode_number):
            start = episode["timestamp"].min()
            end = episode["timestamp"].max()
            rows.append(
                {
                    "evaluation_group": group_id,
                    "false_alarm_start": start,
                    "false_alarm_end": end,
                    "duration_hours": max(typical_minutes / 60.0, (end - start).total_seconds() / 3600.0),
                    "snapshots": int(len(episode)),
                    "max_probability": float(episode["anomaly_probability"].max()),
                }
            )
    table = pd.DataFrame(rows)
    span_hours = 0.0
    for _, group in predictions.groupby("evaluation_group"):
        span_hours += max(0.0, (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 3600.0)
    turbine_months = span_hours / (24.0 * 30.4375)
    summary = {
        "false_alarm_episodes": int(len(table)),
        "evaluated_turbine_months": float(turbine_months),
        "false_alarm_episodes_per_turbine_month": float(len(table) / turbine_months) if turbine_months > 0 else np.nan,
        "median_false_alarm_duration_hours": float(table["duration_hours"].median()) if not table.empty else 0.0,
    }
    return table, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--max-normal-fpr", type=float, default=0.35)
    parser.add_argument("--purge-hours", type=float, default=24.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata, X, y, groups, feature_columns = load_dataset(args.input_file, args.group_column)

    if y.nunique() < 2:
        raise ValueError("The training dataset must contain both normal and anomaly targets.")

    if groups.nunique() >= 4:
        dev_idx, test_idx, split_details = chronological_group_holdout(
            metadata, groups, args.test_fraction
        )
    else:
        warnings.warn(
            "Fewer than four independent groups are available. Falling back to "
            "a chronological row holdout with a purge gap.",
            RuntimeWarning,
        )
        dev_idx, test_idx, split_details = chronological_row_holdout(
            metadata, args.test_fraction, args.purge_hours
        )

    X_dev = X.iloc[dev_idx].reset_index(drop=True)
    y_dev = y.iloc[dev_idx].reset_index(drop=True)
    groups_dev = groups.iloc[dev_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    comparison_rows: list[dict[str, Any]] = []
    fitted_by_name: dict[str, BaseEstimator] = {}
    params_by_name: dict[str, dict[str, Any]] = {}
    threshold_by_name: dict[str, ThresholdResult] = {}

    for spec in build_model_specs(args.random_state):
        print(f"[model] {spec.name}")
        fitted, params, oof_probability, threshold, metrics = tune_and_oof(
            spec=spec,
            X=X_dev,
            y=y_dev,
            groups=groups_dev,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
            max_normal_fpr=args.max_normal_fpr,
        )
        row = {
            "model": spec.name,
            **metrics,
            "selected_threshold": threshold.threshold,
            "threshold_feasible": threshold.feasible,
            "best_params": json.dumps(params, sort_keys=True),
        }
        comparison_rows.append(row)
        fitted_by_name[spec.name] = fitted
        params_by_name[spec.name] = params
        threshold_by_name[spec.name] = threshold

    comparison = pd.DataFrame(comparison_rows)
    comparison["selection_key"] = comparison.apply(
        lambda row: str(model_selection_key(row, args.max_normal_fpr)), axis=1
    )
    selected_name = max(
        comparison_rows,
        key=lambda row: model_selection_key(pd.Series(row), args.max_normal_fpr),
    )["model"]
    selected_model = fitted_by_name[selected_name]
    selected_threshold = threshold_by_name[selected_name]

    test_probability = selected_model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= selected_threshold.threshold).astype(int)
    test_metrics = calculate_metrics(y_test, test_prediction, test_probability)

    predictions = metadata.iloc[test_idx].copy().reset_index(drop=True)
    predictions["evaluation_group"] = groups.iloc[test_idx].reset_index(drop=True)
    predictions["actual_anomaly"] = y_test.to_numpy()
    predictions["anomaly_probability"] = test_probability
    predictions["threshold"] = selected_threshold.threshold
    predictions["predicted_anomaly"] = test_prediction
    predictions["prediction_correct"] = predictions["actual_anomaly"] == predictions["predicted_anomaly"]

    event_table, event_summary = event_detection_metrics(predictions)
    false_alarm_table, false_alarm_summary = false_alarm_episode_metrics(predictions)

    # Final deployment model: choose a threshold from all-data grouped OOF
    # predictions, then fit the selected model family on all snapshots.
    selected_spec = next(spec for spec in build_model_specs(args.random_state) if spec.name == selected_name)
    final_model, final_params, all_oof_probability, final_threshold, all_oof_metrics = tune_and_oof(
        spec=selected_spec,
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        cv_splits=args.cv_splits,
        random_state=args.random_state + 1000,
        max_normal_fpr=args.max_normal_fpr,
    )

    bundle = {
        "model": final_model,
        "model_name": selected_name,
        "feature_columns": feature_columns,
        "probability_threshold": final_threshold.threshold,
        "threshold_policy": "maximise recall, then F2, under normal-FPR constraint",
        "max_normal_fpr": args.max_normal_fpr,
        "group_column": args.group_column,
        "best_params": final_params,
        "training_rows": int(len(X)),
        "training_groups": int(groups.nunique()),
    }
    joblib.dump(bundle, args.output_dir / "streaming_alert_model.joblib")

    importance = extract_feature_importance(final_model, feature_columns)
    importance.to_csv(args.output_dir / "streaming_feature_importance.csv", index=False)
    comparison.to_csv(args.output_dir / "streaming_model_comparison.csv", index=False)
    predictions.to_csv(args.output_dir / "streaming_holdout_predictions.csv", index=False)
    event_table.to_csv(args.output_dir / "streaming_event_detection_details.csv", index=False)
    false_alarm_table.to_csv(args.output_dir / "streaming_false_alarm_episodes.csv", index=False)

    cm = confusion_matrix(y_test, test_prediction, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["actual_normal", "actual_anomaly"],
        columns=["predicted_normal", "predicted_anomaly"],
    ).to_csv(args.output_dir / "streaming_holdout_confusion_matrix.csv")

    metrics_payload = {
        "selected_model": selected_name,
        "development_selected_threshold": asdict(selected_threshold),
        "holdout_row_metrics": test_metrics,
        "holdout_event_metrics": event_summary,
        "holdout_false_alarm_metrics": false_alarm_summary,
        "split": split_details,
        "development_best_params": params_by_name[selected_name],
        "final_all_data_threshold": asdict(final_threshold),
        "final_all_data_grouped_oof_metrics": all_oof_metrics,
        "final_best_params": final_params,
    }
    with (args.output_dir / "streaming_holdout_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, default=str)

    manifest = {
        "input_file": str(args.input_file),
        "output_dir": str(args.output_dir),
        "rows": int(len(metadata)),
        "features": int(len(feature_columns)),
        "groups": int(groups.nunique()),
        "positive_rows": int((y == 1).sum()),
        "negative_rows": int((y == 0).sum()),
        "selected_model": selected_name,
        "saved_model": str(args.output_dir / "streaming_alert_model.joblib"),
        "important_note": (
            "Holdout metrics are the evaluation result. The saved deployment "
            "model is refit on all available data and must not be evaluated on "
            "its own training predictions."
        ),
    }
    with (args.output_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"Selected model: {selected_name}")
    print(f"Holdout recall: {test_metrics['recall_anomaly']:.3f}")
    print(f"Holdout normal FPR: {test_metrics['normal_false_positive_rate']:.3f}")
    print(f"Event detection recall: {event_summary['event_detection_recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
