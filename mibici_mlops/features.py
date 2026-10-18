import pandas as pd
import typer
from pathlib import Path
from loguru import logger
from mibici_mlops.config import PROCESSED_DATA_DIR


app = typer.Typer()


# ---------------------------------------------------------------------
# Shared feature utilities
# ---------------------------------------------------------------------

# Add reusable transformations shared by P1, P2, and P3 here.


# ---------------------------------------------------------------------
# P1 - Hourly trip demand forecasting
# ---------------------------------------------------------------------
def build_p1_hourly_demand(parquet_files, data_start, data_end):
    """
    Build the hourly station-level demand table for P1.

    For each station, the active period is estimated from any observed
    activity, either as trip origin or destination. A complete hourly grid
    is then created within that active period, and hours without trip
    departures are represented with zero demand.

    Parameters
    ----------
    parquet_files : iterable of pathlib.Path
        Processed monthly trip Parquet files.
    data_start : str or pandas.Timestamp
        Beginning of the study period.
    data_end : str or pandas.Timestamp
        End of the study period.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - station_id
        - hour
        - trips_started
    """
    hourly_parts = []
    activity_parts = []

    for file in parquet_files:
        df_month = pd.read_parquet(
            file,
            columns=[
                "inicio_del_viaje",
                "fin_del_viaje",
                "origen_id",
                "destino_id",
            ],
        )

        # Hourly trip departures
        df_month["hour"] = df_month["inicio_del_viaje"].dt.floor("h")

        hourly_month = (
            df_month
            .groupby(["origen_id", "hour"])
            .size()
            .reset_index(name="trips_started")
            .rename(columns={"origen_id": "station_id"})
        )

        hourly_parts.append(hourly_month)

        # Activity observed as origin
        origin_activity = (
            df_month
            .groupby("origen_id")
            .agg(
                first_activity=("inicio_del_viaje", "min"),
                last_activity=("inicio_del_viaje", "max"),
            )
            .rename_axis("station_id")
            .reset_index()
        )

        # Activity observed as destination
        destination_activity = (
            df_month
            .groupby("destino_id")
            .agg(
                first_activity=("fin_del_viaje", "min"),
                last_activity=("fin_del_viaje", "max"),
            )
            .rename_axis("station_id")
            .reset_index()
        )

        activity_parts.extend(
            [origin_activity, destination_activity]
        )

    hourly_demand = pd.concat(
        hourly_parts,
        ignore_index=True,
    )

    station_activity = (
        pd.concat(activity_parts, ignore_index=True)
        .groupby("station_id", as_index=False)
        .agg(
            first_activity=("first_activity", "min"),
            last_activity=("last_activity", "max"),
        )
    )

    data_start = pd.Timestamp(data_start)
    data_end = pd.Timestamp(data_end)

    complete_parts = []

    for row in station_activity.itertuples(index=False):
        station_id = row.station_id

        start_hour = max(
            row.first_activity.floor("h"),
            data_start,
        )

        end_hour = min(
            row.last_activity.floor("h"),
            data_end,
        )

        full_hours = pd.date_range(
            start=start_hour,
            end=end_hour,
            freq="h",
        )

        station_complete = pd.DataFrame({
            "station_id": station_id,
            "hour": full_hours,
        })

        station_demand = hourly_demand.loc[
            hourly_demand["station_id"] == station_id,
            ["hour", "trips_started"],
        ]

        station_complete = station_complete.merge(
            station_demand,
            on="hour",
            how="left",
        )

        station_complete["trips_started"] = (
            station_complete["trips_started"]
            .fillna(0)
            .astype("int32")
        )

        complete_parts.append(station_complete)

    hourly_model = (
        pd.concat(complete_parts, ignore_index=True)
        .sort_values(["station_id", "hour"])
        .reset_index(drop=True)
    )

    return hourly_model


def build_p1_features(hourly_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for P1: hourly trip demand forecasting by station.

    Each row represents one station-hour observation. Historical features
    use only information available before the forecast hour to avoid
    target leakage.

    Parameters
    ----------
    hourly_data : pd.DataFrame
        Hourly station-level dataset containing at least:
        station_id, hour, and trips_started.

    Returns
    -------
    pd.DataFrame
        Dataset with the features required for P1.
    """
    features = (
        hourly_data
        .sort_values(["station_id", "hour"])
        .copy()
    )

    # Calendar features
    features["hour_of_day"] = features["hour"].dt.hour
    features["day_of_week"] = features["hour"].dt.dayofweek
    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype("int8")
    features["month"] = features["hour"].dt.month

    # Historical lag features
    features["lag_1"] = (
        features
        .groupby("station_id")["trips_started"]
        .shift(1)
    )

    features["lag_24"] = (
        features
        .groupby("station_id")["trips_started"]
        .shift(24)
    )

    features["lag_168"] = (
        features
        .groupby("station_id")["trips_started"]
        .shift(168)
    )

    # Rolling historical features
    features["rolling_mean_3h"] = (
        features
        .groupby("station_id")["trips_started"]
        .transform(
            lambda x: x.shift(1).rolling(window=3).mean()
        )
    )

    features["rolling_mean_24h"] = (
        features
        .groupby("station_id")["trips_started"]
        .transform(
            lambda x: x.shift(1).rolling(window=24).mean()
        )
    )

    # Amount of historical information available for each station
    features["history_hours"] = (
        features
        .groupby("station_id")
        .cumcount()
    )

    return features


# ---------------------------------------------------------------------
# P2 - Availability risk classification
# ---------------------------------------------------------------------

# Add P2-specific feature functions here.


# ---------------------------------------------------------------------
# P3 - Station clustering
# ---------------------------------------------------------------------

# Add P3-specific feature functions here.


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "features.parquet",
):
    """Generate reusable features for the MiBici project."""
    logger.info("Generating features...")

    # The final orchestration will be added once the feature pipelines
    # for P1, P2, and P3 are defined.

    logger.success("Feature generation complete.")


if __name__ == "__main__":
    app()
