import typer
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger
from catboost import CatBoostRegressor
from mibici_mlops.config import (
    P1_DATA_END,
    P1_DATA_START,
    P1_TEST_START,
    P1_VALIDATION_START,
    PROCESSED_DATA_DIR,
)
from mibici_mlops.features import (
    build_p1_features,
    build_p1_hourly_demand,
)


app = typer.Typer()


@app.command()
def main():
    """Train the model for P1: hourly trip demand forecasting."""

    trips_dir = PROCESSED_DATA_DIR / "trips"

    parquet_files = sorted(
        trips_dir.glob("datos_abiertos_*.parquet")
    )

    logger.info(
        f"Found {len(parquet_files)} processed trip files."
    )

    logger.info("Building P1 hourly demand table...")

    hourly_model = build_p1_hourly_demand(
        parquet_files=parquet_files,
        data_start=P1_DATA_START,
        data_end=P1_DATA_END,
    )

    logger.info(
        f"Hourly demand table: "
        f"{len(hourly_model):,} rows | "
        f"{hourly_model['station_id'].nunique()} stations | "
        f"{hourly_model['trips_started'].sum():,} trips"
    )

    logger.success("P1 hourly demand table built successfully.")

    logger.info("Building P1 features...")

    features_p1 = build_p1_features(hourly_model)

    logger.info(
        f"P1 feature table: "
        f"{len(features_p1):,} rows | "
        f"{features_p1.shape[1]} columns"
    )

    logger.success("P1 features built successfully.")

    logger.info("Creating temporal train/validation/test split...")

    train_mask = features_p1["hour"] < P1_VALIDATION_START

    validation_mask = (
        (features_p1["hour"] >= P1_VALIDATION_START)
        & (features_p1["hour"] < P1_TEST_START)
    )

    test_mask = features_p1["hour"] >= P1_TEST_START

    train_data = features_p1.loc[train_mask].copy()
    validation_data = features_p1.loc[validation_mask].copy()
    test_data = features_p1.loc[test_mask].copy()

    logger.info(
        f"Train: {len(train_data):,} rows | "
        f"Validation: {len(validation_data):,} rows | "
        f"Test: {len(test_data):,} rows"
    )

    logger.success("Temporal split created successfully.")

    target = "trips_started"

    feature_columns = [
        "station_id",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "month",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_3h",
        "rolling_mean_24h",
        "history_hours",
    ]

    X_train = train_data[feature_columns].copy()
    y_train = train_data[target].copy()

    X_validation = validation_data[feature_columns].copy()
    y_validation = validation_data[target].copy()

    X_test = test_data[feature_columns].copy()
    y_test = test_data[target].copy()

    logger.info(
        f"Model matrices prepared | "
        f"X_train: {X_train.shape} | "
        f"X_validation: {X_validation.shape} | "
        f"X_test: {X_test.shape}"
    )

    logger.success("P1 model matrices prepared successfully.")

    categorical_features = [
        "station_id",
        "hour_of_day",
        "day_of_week",
        "month",
    ]

    for column in categorical_features:
        X_train[column] = X_train[column].astype(str)
        X_validation[column] = X_validation[column].astype(str)
        X_test[column] = X_test[column].astype(str)

    logger.info(
        f"Categorical features prepared for CatBoost: "
        f"{categorical_features}"
    )

    logger.success("CatBoost input data prepared successfully.")

    logger.info("Training final P1 CatBoost model...")

    model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=50,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
    )

    logger.success("Final P1 CatBoost model trained successfully.")

    logger.info("Evaluating P1 model on validation data...")

    validation_predictions = model.predict(X_validation)

    # Trip demand cannot be negative.
    validation_predictions = np.clip(validation_predictions, 0, None)

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions,
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_predictions,
        )
    )

    positive_mask = y_validation > 0

    validation_mae_positive = mean_absolute_error(
        y_validation[positive_mask],
        validation_predictions[positive_mask],
    )

    logger.info(
        f"Validation metrics | "
        f"MAE: {validation_mae:.6f} | "
        f"RMSE: {validation_rmse:.6f} | "
        f"MAE positive demand: {validation_mae_positive:.6f}"
    )

    logger.success("P1 validation evaluation completed successfully.")

    logger.info("Evaluating P1 model on test data...")

    test_predictions = model.predict(X_test)

    # Trip demand cannot be negative.
    test_predictions = np.clip(test_predictions, 0, None)

    test_mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions,
        )
    )

    positive_test_mask = y_test > 0

    test_mae_positive = mean_absolute_error(
        y_test[positive_test_mask],
        test_predictions[positive_test_mask],
    )

    logger.info(
        f"Test metrics | "
        f"MAE: {test_mae:.6f} | "
        f"RMSE: {test_rmse:.6f} | "
        f"MAE positive demand: {test_mae_positive:.6f}"
    )

    logger.success("P1 test evaluation completed successfully.")


if __name__ == "__main__":
    app()
