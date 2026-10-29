"""Fixtures compartidos para la suite de pruebas de MLOps - MiBici Equipo 51.

Provee datos sintéticos mínimos, reproducibles y rápidos para evitar dependencias
de archivos Parquet pesados o llamadas a servicios externos en pruebas unitarias.
"""

import pandas as pd
import pytest

# ============================================================================
# SECCIÓN 1: FIXTURES DE INGESTA Y CALIDAD DE DATOS (Ariadna)
# ============================================================================


@pytest.fixture
def sample_raw_trips_df() -> pd.DataFrame:
    """Fixture de DataFrame sintético representativo de viajes crudos de MiBici.

    Contiene la estructura original de columnas antes de la limpieza:
    Viaje_Id, Usuario_Id, Genero, Ano_de_nacimiento, Inicio_del_viaje,
    Fin_del_viaje, Origen_Id, Destino_Id.
    """
    # TODO - Ariadna - Generar DataFrame sintético de viajes con columnas originales y tipos válidos
    return pd.DataFrame(
        {
            "Viaje_Id": [101, 102, 103, 104, 105],
            "Usuario_Id": [1001, 1002, 1003, 1004, 1005],
            "Genero": ["M", "F", "M", "F", "M"],
            "Ano_de_nacimiento": [1990, 1985, 1995, 2000, 1992],
            "Inicio_del_viaje": [
                "2024-01-01 08:00:00",
                "2024-01-01 08:15:00",
                "2024-01-01 09:00:00",
                "2024-01-01 09:30:00",
                "2024-01-01 10:00:00",
            ],
            "Fin_del_viaje": [
                "2024-01-01 08:20:00",
                "2024-01-01 08:35:00",
                "2024-01-01 09:25:00",
                "2024-01-01 09:45:00",
                "2024-01-01 10:18:00",
            ],
            "Origen_Id": [2, 3, 2, 4, 3],
            "Destino_Id": [3, 2, 4, 2, 2],
        }
    )


@pytest.fixture
def sample_nomenclatura_df() -> pd.DataFrame:
    """Fixture de catálogo de estaciones con coordenadas válidas del AMG."""
    # TODO - Ariadna - Crear DataFrame sintético de nomenclatura con identificadores y coordenadas
    return pd.DataFrame(
        {
            "id": [2, 3, 4],
            "name": ["Estacion 2", "Estacion 3", "Estacion 4"],
            "obcn": ["GDL-002", "GDL-003", "GDL-004"],
            "latitude": [20.6750, 20.6780, 20.6720],
            "longitude": [-103.3550, -103.3600, -103.3500],
            "slots": [16, 20, 14],
        }
    )


@pytest.fixture
def sample_corrupt_csv_content() -> str:
    """Fixture que emula un archivo CSV corrupto con campos truncados o delimitadores erróneos."""
    # TODO - Ariadna - Definir contenido de texto malformado para pruebas de tolerancia a fallos
    return "Viaje_Id,Usuario_Id,Genero\n101,1001,M\n102,CORRUPT_ROW_MISSING_FIELDS\n"


@pytest.fixture
def sample_trips_with_anomalies_df() -> pd.DataFrame:
    """Fixture con viajes anómalos (duración negativa, duraciones extremas o nulos)."""
    # TODO - Ariadna - Crear registros con anomalías temporales para probar filtros de limpieza
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
                "2024-01-01 08:10:00",
                "2024-01-01 09:00:00",
                "2024-01-03 12:00:00",
            ],  # Negativo, cero y >48h
            "Origen_Id": [2, 3, 4],
            "Destino_Id": [3, 2, 2],
        }
    )


# ============================================================================
# SECCIÓN 2: FIXTURES DE FEATURE ENGINEERING Y DATA LEAKAGE (Gabriela)
# ============================================================================


@pytest.fixture
def sample_p1_hourly_demand_df() -> pd.DataFrame:
    """Fixture de demanda horaria continua (al menos 200 horas) para probar lags de P1."""
    # TODO - Gabriela - Generar serie temporal horaria sintética para estación única con >=200 horas
    fechas = pd.date_range("2024-01-01 00:00:00", periods=200, freq="h")
    return pd.DataFrame(
        {
            "station_id": [2] * len(fechas),
            "hour": fechas,
            "demand_trips": [i % 7 for i in range(len(fechas))],
        }
    )


@pytest.fixture
def sample_p2_station_hourly_df() -> pd.DataFrame:
    """Fixture de entradas, salidas y capacidad horaria para probar objetivo de 2h de P2."""
    # TODO - Gabriela - Crear serie horaria de balance y flujos netos para clasificación de presión
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
    # TODO - Gabriela - Generar DataFrame sintético de perfiles operativos agregados para clustering P3
    return pd.DataFrame(
        {
            "estacion_id": [2, 3, 4],
            "latitude": [20.6750, 20.6780, 20.6720],
            "longitude": [-103.3550, -103.3600, -103.3500],
            "prop_manana": [0.35, 0.20, 0.45],
            "prop_tarde": [0.40, 0.50, 0.30],
            "prop_fin_semana": [0.25, 0.30, 0.25],
            "prop_domingo": [0.12, 0.15, 0.10],
            "rotacion_diaria": [2.5, 1.8, 3.2],
        }
    )


# ============================================================================
# SECCIÓN 3: FIXTURES DE CONTRATOS, FIRMAS E INFERENCIA (Ingrid)
# ============================================================================


@pytest.fixture
def sample_p1_inference_payload() -> pd.DataFrame:
    """Fixture con las 11 features requeridas por el contrato de entrada de P1 (CatBoost)."""
    # TODO - Ingrid - Definir DataFrame con tipos exactos de las 11 features del modelo P1
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
    # TODO - Ingrid - Definir DataFrame con features numéricas y categóricas para P2
    return pd.DataFrame(
        {
            "capacidad": [16, 20],
            "latitude": [20.6750, 20.6780],
            "longitude": [-103.3550, -103.3600],
            "flujo_neto_t_1": [-2, 3],
            "flujo_neto_t_2": [1, 2],
            "flujo_neto_media_3h": [-0.33, 1.67],
            "entradas_media_3h": [2.0, 4.0],
            "salidas_media_3h": [2.33, 2.33],
            "hora_sin": [0.866, -0.965],
            "hora_cos": [-0.5, -0.258],
            "dia_sin": [0.781, -0.433],
            "dia_cos": [0.623, -0.900],
            "mes_sin": [0.866, 0.866],
            "mes_cos": [0.5, 0.5],
        }
    )


@pytest.fixture
def sample_p3_inference_payload() -> pd.DataFrame:
    """Fixture con las 7 features normalizadas para el modelo Champion P3 (K-Means)."""
    # TODO - Ingrid - Definir DataFrame con las 7 variables escaladas de P3
    return pd.DataFrame(
        {
            "latitude_scaled": [0.2, 0.5],
            "longitude_scaled": [-0.1, 0.3],
            "ratio_salidas_entradas_manana": [1.2, 0.8],
            "ratio_salidas_entradas_tarde": [0.9, 1.4],
            "prop_fin_semana": [0.28, 0.22],
            "prop_domingo": [0.14, 0.10],
            "rotacion_capacidad": [2.1, 1.5],
        }
    )
