"""DeepChecks-based data quality checks and privacy-safe reports."""

import json
from pathlib import Path
import re
from typing import Any

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    FeatureDrift,
    MixedDataTypes,
    PercentOfNulls,
)
from loguru import logger
import pandas as pd

from mibici_mlops.config import DATA_QUALITY_DIR

TRIP_REQUIRED_COLUMNS = [
    "viaje_id",
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
]
NOMENCLATURE_REQUIRED_COLUMNS = [
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "capacity",
]
MAX_TRIP_DURATION_HOURS = 24
MAX_DRIFT_SAMPLE_SIZE = 100_000
DRIFT_SCORE_THRESHOLD = 0.2
LATITUDE_RANGE = (20.5, 20.9)
LONGITUDE_RANGE = (-103.5, -103.2)


def _condition_summaries(result: Any) -> list[dict[str, str]]:
    summaries = []
    for condition in result.conditions_results:
        category = condition.category
        summaries.append(
            {
                "name": str(condition.name),
                "status": str(getattr(category, "value", category)).lower(),
                "details": str(condition.details),
            }
        )
    return summaries


def _check_summary(result: Any) -> dict[str, Any]:
    return {
        "passed": bool(result.passed_conditions()),
        "conditions": _condition_summaries(result),
    }


def save_quality_report(report: dict[str, Any], report_name: str) -> Path:
    """Write a summary-only JSON report; dataset rows are never serialized."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(report_name).stem)
    if not safe_name:
        raise ValueError("A non-empty quality report name is required.")

    DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DATA_QUALITY_DIR / f"{safe_name}.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
    logger.info("Data quality report saved: {}", report_path)
    return report_path


def _schema_failure_report(
    source_name: str,
    dataset_type: str,
    row_count: int,
    missing_columns: list[str],
) -> dict[str, Any]:
    return {
        "source": source_name,
        "dataset_type": dataset_type,
        "status": "failed",
        "metrics": {"rows": row_count},
        "checks": {
            "required_columns": {
                "passed": False,
                "details": {"missing_columns": missing_columns},
            }
        },
        "actions": [],
    }


def _check_missing_values(
    dataset: Dataset,
    columns: list[str],
) -> dict[str, Any]:
    check = PercentOfNulls(columns=columns, n_samples=max(1, dataset.n_samples))
    check.add_condition_percent_of_nulls_not_greater_than(threshold=0.0)
    return _check_summary(check.run(dataset))


def _check_mixed_types(
    dataset: Dataset,
    columns: list[str],
) -> dict[str, Any]:
    if not columns:
        return {"passed": True, "status": "skipped", "conditions": []}

    check = MixedDataTypes(
        columns=columns,
        n_samples=max(1, dataset.n_samples),
    )
    check.add_condition_rare_type_ratio_not_in_range()
    return _check_summary(check.run(dataset))


def _report_status(checks: dict[str, dict[str, Any]]) -> str:
    if any(not check.get("passed", True) for check in checks.values()):
        return "warning"
    return "passed"


def run_trip_quality_checks(
    trips: pd.DataFrame,
    source_name: str,
) -> dict[str, Any]:
    """Run exhaustive schema, null, duplicate, and mixed-type checks on trips."""
    missing_columns = sorted(set(TRIP_REQUIRED_COLUMNS) - set(trips.columns))
    if missing_columns or trips.empty:
        report = _schema_failure_report(
            source_name,
            "trips",
            len(trips),
            missing_columns,
        )
        if trips.empty:
            report["checks"]["non_empty_dataset"] = {
                "passed": False,
                "details": "No trip rows were found.",
            }
        save_quality_report(report, source_name)
        raise ValueError(
            f"Trip quality check failed; missing columns={missing_columns}, rows={len(trips)}"
        )

    categorical_columns = [column for column in ["genero"] if column in trips.columns]
    dataset = Dataset(trips, cat_features=categorical_columns)
    null_counts = {column: int(trips[column].isna().sum()) for column in trips.columns}
    critical_null_counts = {column: null_counts[column] for column in TRIP_REQUIRED_COLUMNS}
    duplicate_rows = int(trips.duplicated().sum())
    numeric_columns = [
        column
        for column in [
            "viaje_id",
            "origen_id",
            "destino_id",
            "ano_de_nacimiento",
        ]
        if column in trips.columns
    ]

    null_check = _check_missing_values(dataset, list(trips.columns))
    duplicate_check = DataDuplicates(
        n_samples=max(1, len(trips))
    ).add_condition_ratio_less_or_equal(0)
    duplicate_result = _check_summary(duplicate_check.run(dataset))
    mixed_check = _check_mixed_types(dataset, numeric_columns)

    duration_seconds = (trips["fin_del_viaje"] - trips["inicio_del_viaje"]).dt.total_seconds()
    invalid_duration_rows = int(
        (
            duration_seconds.notna()
            & ((duration_seconds <= 0) | (duration_seconds > MAX_TRIP_DURATION_HOURS * 3600))
        ).sum()
    )

    checks = {
        "required_columns": {
            "passed": True,
            "details": {"columns_checked": TRIP_REQUIRED_COLUMNS},
        },
        "missing_values_all_columns": null_check,
        "critical_missing_values": {
            "passed": not any(critical_null_counts.values()),
            "details": {"null_counts": critical_null_counts},
        },
        "exact_duplicates": duplicate_result,
        "mixed_data_types": mixed_check,
        "valid_trip_duration": {
            "passed": invalid_duration_rows == 0,
            "details": {
                "invalid_rows": invalid_duration_rows,
                "maximum_hours": MAX_TRIP_DURATION_HOURS,
            },
        },
    }
    status = _report_status(checks)
    if any(critical_null_counts.values()):
        status = "failed"

    report: dict[str, Any] = {
        "source": source_name,
        "dataset_type": "trips",
        "status": status,
        "metrics": {
            "rows_before_cleaning": len(trips),
            "columns": len(trips.columns),
            "null_counts_by_column": null_counts,
            "critical_null_counts": critical_null_counts,
            "exact_duplicate_rows": duplicate_rows,
            "invalid_duration_rows": invalid_duration_rows,
        },
        "checks": checks,
        "actions": [],
    }
    report["report_path"] = str(save_quality_report(report, source_name))

    if any(critical_null_counts.values()):
        raise ValueError(
            "Trip quality check failed; null values in critical columns: "
            f"{ {key: value for key, value in critical_null_counts.items() if value} }"
        )

    return report


def remove_invalid_trip_durations(
    trips: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Drop trips with non-positive or over-24-hour durations."""
    duration_seconds = (trips["fin_del_viaje"] - trips["inicio_del_viaje"]).dt.total_seconds()
    valid_duration = duration_seconds.gt(0) & duration_seconds.le(MAX_TRIP_DURATION_HOURS * 3600)
    filtered = trips.loc[valid_duration].copy().reset_index(drop=True)
    return filtered, len(trips) - len(filtered)


def run_nomenclature_quality_checks(
    stations: pd.DataFrame,
    source_name: str,
) -> dict[str, Any]:
    """Run DeepChecks and integrity checks on station nomenclature."""
    missing_columns = sorted(set(NOMENCLATURE_REQUIRED_COLUMNS) - set(stations.columns))
    if missing_columns or stations.empty:
        report = _schema_failure_report(
            source_name,
            "nomenclature",
            len(stations),
            missing_columns,
        )
        if stations.empty:
            report["checks"]["non_empty_dataset"] = {
                "passed": False,
                "details": "No station rows were found.",
            }
        save_quality_report(report, source_name)
        raise ValueError(
            "Nomenclature quality check failed; "
            f"missing columns={missing_columns}, rows={len(stations)}"
        )

    dataset = Dataset(stations, cat_features=["station_name"])
    null_counts = {column: int(stations[column].isna().sum()) for column in stations.columns}
    required_null_counts = {
        column: null_counts[column] for column in NOMENCLATURE_REQUIRED_COLUMNS
    }
    duplicate_station_ids = int(stations["station_id"].duplicated().sum())
    negative_capacities = int(stations["capacity"].lt(0).sum())
    zero_capacities = int(stations["capacity"].eq(0).sum())
    invalid_coordinates = ~stations["latitude"].between(*LATITUDE_RANGE) | ~stations[
        "longitude"
    ].between(*LONGITUDE_RANGE)
    invalid_coordinate_rows = int(invalid_coordinates.sum())
    numeric_columns = ["station_id", "latitude", "longitude", "capacity"]

    null_check = _check_missing_values(dataset, list(stations.columns))
    duplicate_check = DataDuplicates(
        columns=["station_id"],
        n_samples=max(1, len(stations)),
    ).add_condition_ratio_less_or_equal(0)
    duplicate_result = _check_summary(duplicate_check.run(dataset))
    mixed_check = _check_mixed_types(dataset, numeric_columns)

    checks = {
        "required_columns": {
            "passed": True,
            "details": {"columns_checked": NOMENCLATURE_REQUIRED_COLUMNS},
        },
        "missing_values_all_columns": null_check,
        "critical_missing_values": {
            "passed": not any(required_null_counts.values()),
            "details": {"null_counts": required_null_counts},
        },
        "duplicate_station_ids": duplicate_result,
        "mixed_data_types": mixed_check,
        "non_negative_capacity": {
            "passed": negative_capacities == 0,
            "details": {"negative_capacity_rows": negative_capacities},
        },
        "positive_capacity": {
            "passed": zero_capacities == 0,
            "status": "warning" if zero_capacities else "passed",
            "details": {"zero_capacity_rows": zero_capacities},
        },
        "coordinates_in_service_area": {
            "passed": invalid_coordinate_rows == 0,
            "details": {
                "invalid_rows": invalid_coordinate_rows,
                "latitude_range": LATITUDE_RANGE,
                "longitude_range": LONGITUDE_RANGE,
            },
        },
    }
    status = _report_status(checks)
    if (
        any(required_null_counts.values())
        or duplicate_station_ids
        or negative_capacities
        or invalid_coordinate_rows
    ):
        status = "failed"
    elif zero_capacities:
        status = "warning"

    report: dict[str, Any] = {
        "source": source_name,
        "dataset_type": "nomenclature",
        "status": status,
        "metrics": {
            "rows": len(stations),
            "critical_null_counts": null_counts,
            "required_null_counts": required_null_counts,
            "duplicate_station_ids": duplicate_station_ids,
            "negative_capacity_rows": negative_capacities,
            "zero_capacity_rows": zero_capacities,
            "invalid_coordinate_rows": invalid_coordinate_rows,
        },
        "checks": checks,
        "actions": [],
    }
    report["report_path"] = str(save_quality_report(report, source_name))

    if (
        any(required_null_counts.values())
        or duplicate_station_ids
        or negative_capacities
        or invalid_coordinate_rows
    ):
        raise ValueError(
            "Nomenclature quality check failed: "
            f"Duplicate station IDs={duplicate_station_ids}, "
            f"{negative_capacities} negative capacities, "
            f"{invalid_coordinate_rows} out-of-area coordinates; see "
            f"{report['report_path']}"
        )

    return report


def run_feature_drift_check(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    report_name: str,
) -> dict[str, Any]:
    """Compare training and test feature distributions and persist summary."""
    missing_columns = sorted(
        set(feature_columns) - set(train_features.columns)
        | (set(feature_columns) - set(test_features.columns))
    )
    if missing_columns:
        report = {
            "dataset_type": "feature_drift",
            "status": "failed",
            "metrics": {},
            "checks": {
                "required_features": {
                    "passed": False,
                    "details": {"missing_features": missing_columns},
                }
            },
        }
        report["report_path"] = str(save_quality_report(report, report_name))
        raise ValueError(f"Feature drift check is missing features: {missing_columns}")

    sample_size = min(
        MAX_DRIFT_SAMPLE_SIZE,
        len(train_features),
        len(test_features),
    )
    if sample_size < 10 or not feature_columns:
        report = {
            "dataset_type": "feature_drift",
            "status": "skipped",
            "metrics": {
                "train_rows": len(train_features),
                "test_rows": len(test_features),
                "sample_size_per_split": sample_size,
            },
            "checks": {
                "feature_drift": {
                    "passed": True,
                    "status": "skipped",
                    "details": {"reason": "At least 10 rows per split are required."},
                }
            },
        }
        report["report_path"] = str(save_quality_report(report, report_name))
        return report

    train_dataset = Dataset(
        train_features[feature_columns],
        cat_features=[column for column in categorical_columns if column in feature_columns],
    )
    test_dataset = Dataset(
        test_features[feature_columns],
        cat_features=[column for column in categorical_columns if column in feature_columns],
    )
    drift_check = FeatureDrift(
        columns=feature_columns,
        min_samples=10,
        n_samples=sample_size,
        random_state=42,
    ).add_condition_drift_score_less_than(
        max_allowed_categorical_score=DRIFT_SCORE_THRESHOLD,
        max_allowed_numeric_score=DRIFT_SCORE_THRESHOLD,
        allowed_num_features_exceeding_threshold=0,
    )
    drift_result = _check_summary(drift_check.run(train_dataset, test_dataset))
    report = {
        "dataset_type": "feature_drift",
        "status": "passed" if drift_result["passed"] else "warning",
        "metrics": {
            "train_rows": len(train_features),
            "test_rows": len(test_features),
            "sample_size_per_split": sample_size,
            "features_checked": feature_columns,
        },
        "checks": {"feature_drift": drift_result},
    }
    report["report_path"] = str(save_quality_report(report, report_name))

    if report["status"] == "warning":
        logger.warning(
            "Feature drift exceeded configured thresholds; see {}",
            report["report_path"],
        )
    return report
