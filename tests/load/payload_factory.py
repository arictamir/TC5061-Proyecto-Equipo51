
"""Generación de datos sintéticos para las pruebas de carga de MiBici."""

import random
from typing import Any

SAMPLE_STATIONS: list[str] = [
    "2", "3", "5", "8", "15", "24", "32",
    "50", "78", "100", "150", "200", "310", "420",
]


def generate_sample_prediction_payload() -> dict[str, Any]:
    """Genera un payload válido representativo para el endpoint POST /predict."""
    station_id = random.choice(SAMPLE_STATIONS)
    hour = random.randint(0, 23)
    day = random.randint(0, 6)
    is_weekend = 1 if day in (5, 6) else 0
    month = random.randint(1, 12)

    lag_1 = round(random.uniform(0.0, 15.0), 2)
    lag_24 = round(random.uniform(0.0, 20.0), 2)
    lag_168 = round(random.uniform(0.0, 22.0), 2)
    rolling_3h = round((lag_1 + random.uniform(0.0, 5.0)) / 2.0, 2)
    rolling_24h = round((lag_24 + random.uniform(0.0, 8.0)) / 2.0, 2)
    history_hours = random.randint(24, 720)

    return {
        "station_id": station_id,
        "hour_of_day": hour,
        "day_of_week": day,
        "is_weekend": is_weekend,
        "month": month,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "rolling_mean_3h": rolling_3h,
        "rolling_mean_24h": rolling_24h,
        "history_hours": history_hours,
    }
