"""Fixtures compartidos para la suite de pruebas de MLOps - MiBici Equipo 51.

Provee datos sintéticos mínimos, reproducibles y rápidos para evitar dependencias
de archivos Parquet pesados o llamadas a servicios externos en pruebas unitarias.
"""

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici_mlops.config import SEMILLA_ALEATORIA
from mibici_mlops.features import COLUMNAS_CARACTERISTICAS_P2

# ============================================================================
# SECCIÓN 1: FIXTURES DE INGESTA Y CALIDAD DE DATOS
# ============================================================================


@pytest.fixture
def sample_raw_trips_df() -> pd.DataFrame:
    """Fixture de DataFrame sintético representativo de viajes crudos de MiBici.

    Contiene la estructura original de columnas antes de la limpieza:
    Viaje_Id, Usuario_Id, Genero, Ano_de_nacimiento, Inicio_del_viaje,
    Fin_del_viaje, Origen_Id, Destino_Id.
    """
    return pd.DataFrame(
        {
            "Viaje_Id": [101, 102, 103, 104, 105, 106],
            "Usuario_Id": [1001, 1002, 1003, 1004, 1005, 1006],
            "Genero": ["M", "F", "M", "F", "M", "F"],
            "Ano_de_nacimiento": [1990, 1985, 1995, 2000, 1992, 1988],
            "Inicio_del_viaje": [
                "2024-01-01 08:00:00",
                "2024-01-01 08:15:00",
                "2024-01-01 09:00:00",
                "2024-01-01 09:30:00",
                "2024-01-01 10:00:00",
                "2024-01-01 10:15:00",
            ],
            "Fin_del_viaje": [
                "2024-01-01 08:20:00",
                "2024-01-01 08:35:00",
                "2024-01-01 09:25:00",
                "2024-01-01 09:45:00",
                "2024-01-01 10:18:00",
                "2024-01-01 10:30:00",
            ],
            "Origen_Id": [2, 3, 2, 4, 3, 4],
            "Destino_Id": [3, 2, 4, 2, 2, 3],
        }
    )


@pytest.fixture
def sample_nomenclatura_df() -> pd.DataFrame:
    """Fixture de catálogo de estaciones con coordenadas válidas del AMG."""
    return pd.DataFrame(
        {
            "id": [2, 3, 4],
            "name": ["Estacion 2", "Estacion 3", "Estacion 4"],
            "latitude": [20.6750, 20.6780, 20.6720],
            "longitude": [-103.3550, -103.3600, -103.3500],
            "capacity": [16, 20, 14],
        }
    )


@pytest.fixture
def sample_corrupt_csv_content() -> str:
    """Fixture que emula un archivo CSV corrupto con campos truncados o delimitadores erróneos."""
    return "Viaje_Id,Usuario_Id,Genero\n101,1001,M\n102,CORRUPT_ROW_MISSING_FIELDS\n"


@pytest.fixture
def sample_trips_with_anomalies_df() -> pd.DataFrame:
    """Fixture con viajes anómalos (duración negativa, duraciones extremas o nulos)."""
    return pd.DataFrame(
        {
            "Viaje_Id": [201, 202, 203],
            "Usuario_Id": [2001, 2002, 2003],
            "Inicio_del_viaje": [
                "2024-01-01 08:30:00",
                "2024-01-01 09:00:00",
                "2024-01-01 10:00:00",
            ],
            "Fin_del_viaje": [
                 "2024-01-01 08:10:00",  # Duración negativa: -20 min
                "2024-01-01 09:00:00",  # Duración cero: 0 min
                "2024-01-03 12:00:00",  # Duración extrema: >48 horas
            ],
            "Origen_Id": [2, 3, 4],
            "Destino_Id": [3, 2, 2],
        }
    )


# ============================================================================
# SECCIÓN 2: FIXTURES DE FEATURE ENGINEERING Y DATA LEAKAGE (Gabriela)
# ============================================================================


@pytest.fixture
def sample_p1_hourly_demand_df() -> pd.DataFrame:
    """Fixture de demanda horaria continua (200 horas) para probar lags de P1."""
    fechas = pd.date_range("2024-01-01 00:00:00", periods=200, freq="h")
    return pd.DataFrame(
        {
            "station_id": [2] * len(fechas),
            "hour": fechas,
            "trips_started": [(i * 3 + 1) % 11 for i in range(len(fechas))],
        }
    )


@pytest.fixture
def sample_p2_station_hourly_df() -> pd.DataFrame:
    """Fixture de entradas, salidas y capacidad horaria para probar objetivo de 2h de P2."""
    fechas = pd.date_range("2024-01-01 00:00:00", periods=10, freq="h")
    return pd.DataFrame(
        {
            "estacion_id": [2] * len(fechas),
            "fecha_hora": fechas,
            "entradas": [1, 2, 3, 0, 5, 2, 1, 4, 0, 2],
            "salidas": [2, 1, 0, 4, 1, 3, 2, 1, 3, 1],
            "capacidad": [10] * len(fechas),
        }
    )


@pytest.fixture
def sample_p3_station_aggregated_df() -> pd.DataFrame:
    """Fixture de métricas consolidadas por estación para construir las 7 features de P3."""
    return pd.DataFrame(
        {
            "latitude": [20.6750, 20.6780, 20.6720, 20.6800],
            "longitude": [-103.3550, -103.3600, -103.3500, -103.3650],
            "ratio_am": [1.25, 0.80, 1.10, 0.95],
            "ratio_pm": [0.90, 1.30, 0.85, 1.15],
            "pct_fin_semana": [0.28, 0.22, 0.25, 0.30],
            "pct_domingo": [0.14, 0.10, 0.12, 0.15],
            "rotacion_capacidad": [2.10, 1.50, 3.20, 1.80],
        },
        index=[2, 3, 4, 5],
    )


# ============================================================================
# SECCIÓN 3: FIXTURES DE CONTRATOS, FIRMAS E INFERENCIA (Ingrid)
# ============================================================================

@pytest.fixture
def sample_p1_inference_payload() -> pd.DataFrame:
    """Fixture con las 11 features requeridas por el contrato de entrada de P1 (CatBoost)."""
    return pd.DataFrame(
        {
            "station_id": [2, 3],
            "hour_of_day": [8, 18],
            "day_of_week": [1, 4],
            "is_weekend": [0, 0],
            "month": [3, 3],
            "lag_1": [2.0, 5.0],
            "lag_24": [3.0, 4.0],
            "lag_168": [1.0, 6.0],
            "rolling_mean_3h": [2.33, 4.67],
            "rolling_mean_24h": [1.80, 3.20],
            "history_hours": [200, 200],
        }
    )


@pytest.fixture
def sample_p2_inference_payload() -> pd.DataFrame:
    """Fixture con las features de entrada para el modelo Champion de P2 (Regresión Logística)."""
    filas = 5
    datos = {col: [i * 1.5 for i in range(filas)] for col in COLUMNAS_CARACTERISTICAS_P2}
    datos["capacidad"] = [16, 20, 14, 18, 22]
    datos["latitud"] = [20.6750, 20.6780, 20.6720, 20.6800, 20.6710]
    datos["longitud"] = [-103.3550, -103.3600, -103.3500, -103.3650, -103.3580]
    datos["hora"] = [8, 12, 14, 18, 20]
    datos["dia_semana"] = [0, 1, 2, 3, 4]
    datos["mes"] = [3, 3, 3, 3, 3]
    datos["es_fin_semana"] = [0, 0, 0, 0, 0]
    return pd.DataFrame(datos)


@pytest.fixture
def sample_p3_inference_payload(sample_p3_station_aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture con las 7 features para el modelo Champion P3 (K-Means)."""
    return sample_p3_station_aggregated_df.copy()


@pytest.fixture
def mock_champion_p1_model(sample_p1_inference_payload: pd.DataFrame) -> CatBoostRegressor:
    """Modelo sintético Champion P1 (CatBoost) entrenado en memoria para pruebas de inferencia."""
    X = sample_p1_inference_payload.copy()
    y = np.array([3.0, 7.0])
    cb = CatBoostRegressor(
        iterations=5,
        depth=2,
        learning_rate=0.1,
        loss_function="RMSE",
        random_seed=SEMILLA_ALEATORIA,
        verbose=0,
    )
    cb.fit(X, y)
    return cb


@pytest.fixture
def mock_champion_p2_model(sample_p2_inference_payload: pd.DataFrame) -> Pipeline:
    """Pipeline sintético Champion P2 (Logistic Regression) entrenado en memoria para contratos."""
    X = sample_p2_inference_payload.copy()
    # Generar al menos un ejemplo para cada una de las 3 clases operativas
    y = np.array(
        [
            "riesgo_vaciado",
            "normal",
            "riesgo_saturacion",
            "normal",
            "riesgo_vaciado",
        ]
    )
    pipe = Pipeline(
        [
            ("escalador", StandardScaler()),
            (
                "clasificador",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=100,
                    class_weight="balanced",
                    random_state=SEMILLA_ALEATORIA,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def mock_champion_p3_model(sample_p3_inference_payload: pd.DataFrame) -> Pipeline:
    """Pipeline sintético Champion P3 (K-Means k=4) entrenado en memoria para clustering."""
    X = sample_p3_inference_payload.copy()
    pipe = Pipeline(
        [
            ("escalador", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=4,
                    random_state=SEMILLA_ALEATORIA,
                    n_init=5,
                ),
            ),
        ]
    )
    pipe.fit(X)
    return pipe