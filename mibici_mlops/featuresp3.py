"""Módulo de ingeniería de características para la Pregunta 3 (P3 - Clustering).

Este módulo procesa los datos consolidados de viajes y nomenclatura de MiBici
para construir un vector multidimensional por estación que captura patrones
temporales, pendulares y de rotación para tareas de agrupamiento.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def compute_hourly_profiles(trips_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula la distribución de salidas y entradas por franjas horarias y días de la semana.

    Args:
        trips_df: DataFrame procesado de viajes con columnas 'Inicio_del_viaje',
                  'Fin_del_viaje', 'Origen_Id' y 'Destino_Id'.

    Returns:
        Tupla con (salidas_agg, entradas_agg) agrupadas por ID de estación.
    """
    df = trips_df.copy()

    # Asegurar formato datetime si aún no está tipado
    if not np.issubdtype(df["Inicio_del_viaje"].dtype, np.datetime64):
        df["Inicio_del_viaje"] = pd.to_datetime(df["Inicio_del_viaje"])

    df["hora"] = df["Inicio_del_viaje"].dt.hour
    df["dia_semana"] = df["Inicio_del_viaje"].dt.dayofweek
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_domingo"] = (df["dia_semana"] == 6).astype(int)

    # Agregaciones de salida por estación de origen
    salidas_agg = df.groupby("Origen_Id").agg(
        salidas_totales=("Viaje_Id", "count"),
        salidas_am=("hora", lambda x: ((x >= 7) & (x <= 10)).sum()),
        salidas_pm=("hora", lambda x: ((x >= 18) & (x <= 21)).sum()),
        salidas_fin_semana=("es_fin_de_semana", "sum"),
        salidas_domingo=("es_domingo", "sum"),
    )

    # Agregaciones de entrada por estación de destino
    entradas_agg = df.groupby("Destino_Id").agg(
        entradas_totales=("Viaje_Id", "count"),
        entradas_am=("hora", lambda x: ((x >= 7) & (x <= 10)).sum()),
        entradas_pm=("hora", lambda x: ((x >= 18) & (x <= 21)).sum()),
    )

    return salidas_agg, entradas_agg


def generate_p3_station_features(
    trips_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    features_list: List[str] = None,
) -> pd.DataFrame:
    """Construye la matriz de características para el clustering de estaciones.

    Variables construidas:
    - ratio_am: Salidas / Entradas en horario pico matutino (identifica zonas origen/residenciales).
    - ratio_pm: Salidas / Entradas en horario pico vespertino (identifica zonas destino/laborales).
    - pct_fin_semana: Proporción de viajes en sábado y domingo frente al total.
    - pct_domingo: Proporción de viajes en domingo (captura dinamismo de Vía RecreActiva).
    - rotacion_capacidad: Intensidad de uso total (entradas + salidas) normalizada por capacidad.
    - latitude, longitude: Contexto espacial de la estación.

    Args:
        trips_df: DataFrame con el historial consolidado de viajes.
        stations_df: Catálogo de estaciones (nomenclatura) con 'id', 'latitude', 'longitude' y 'capacidad'.
        features_list: Lista opcional de columnas a retornar. Si es None, retorna el set estándar.

    Returns:
        DataFrame indexado por 'station_id' con las variables numéricas listas para escalar.
    """
    salidas, entradas = compute_hourly_profiles(trips_df)

    stations = stations_df.copy()
    if "id" in stations.columns:
        stations = stations.set_index("id")

    # Unir datos asegurando que no se pierdan estaciones del catálogo
    merged = stations.join([salidas, entradas], how="inner")

    # Reemplazar valores nulos si una estación no tuvo viajes en algún periodo
    merged = merged.fillna(0)

    # Pequeña constante epsilon para evitar división entre cero
    eps = 1e-5

    # Construcción de variables relativas (normalizadas por escala)
    merged["ratio_am"] = merged["salidas_am"] / (merged["entradas_am"] + eps)
    merged["ratio_pm"] = merged["salidas_pm"] / (merged["entradas_pm"] + eps)
    merged["pct_fin_semana"] = merged["salidas_fin_semana"] / (merged["salidas_totales"] + eps)
    merged["pct_domingo"] = merged["salidas_domingo"] / (merged["salidas_totales"] + eps)
    
    # Rotación relativa a la infraestructura instalada
    capacidad_col = "capacidad" if "capacidad" in merged.columns else "obcn"
    merged["rotacion_capacidad"] = (
        merged["salidas_totales"] + merged["entradas_totales"]
    ) / (merged[capacidad_col] + eps)

    if features_list is None:
        features_list = [
            "latitude",
            "longitude",
            "ratio_am",
            "ratio_pm",
            "pct_fin_semana",
            "pct_domingo",
            "rotacion_capacidad",
        ]

    return merged[features_list].copy()