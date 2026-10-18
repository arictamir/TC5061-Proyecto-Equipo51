"""Módulo de ingeniería de características para la Pregunta 3 (P3 - Clustering).

Este módulo procesa los datos consolidados de viajes y nomenclatura de MiBici
para construir un vector multidimensional por estación que captura patrones
temporales, pendulares y de rotación para tareas de agrupamiento.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def _resolve_column_names(df: pd.DataFrame) -> dict:
    """Mapea dinámicamente las columnas del DataFrame sin importar la convención de nombres."""
    col_map = {}
    cols_lower = {col.lower(): col for col in df.columns}

    # Buscar columna de inicio del viaje
    for candidate in [
        "inicio_del_viaje",
        "iniciodelviaje",
        "inicio",
        "start_time",
        "started_at",
        "trip_start",
    ]:
        if candidate in cols_lower:
            col_map["inicio"] = cols_lower[candidate]
            break

    # Buscar columna de ID de viaje
    for candidate in ["viaje_id", "viajeid", "id", "trip_id"]:
        if candidate in cols_lower:
            col_map["viaje_id"] = cols_lower[candidate]
            break

    # Buscar columna de origen
    for candidate in [
        "origen_id",
        "origenid",
        "origen",
        "start_station_id",
        "origin_id",
    ]:
        if candidate in cols_lower:
            col_map["origen"] = cols_lower[candidate]
            break

    # Buscar columna de destino
    for candidate in [
        "destino_id",
        "destinoid",
        "destino",
        "end_station_id",
        "destination_id",
    ]:
        if candidate in cols_lower:
            col_map["destino"] = cols_lower[candidate]
            break

    return col_map


def compute_hourly_profiles(
    trips_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula la distribución de salidas y entradas por franjas horarias y días de la semana."""
    df = trips_df.copy()
    col_map = _resolve_column_names(df)

    inicio_col = col_map.get("inicio", "Inicio_del_viaje")
    viaje_id_col = col_map.get(
        "viaje_id", df.columns[0]
    )  # Si no está, usa la primera columna como contador
    origen_col = col_map.get("origen", "Origen_Id")
    destino_col = col_map.get("destino", "Destino_Id")

    # Asegurar formato datetime si aún no está tipado
    if not np.issubdtype(df[inicio_col].dtype, np.datetime64):
        df[inicio_col] = pd.to_datetime(df[inicio_col])

    df["hora"] = df[inicio_col].dt.hour
    df["dia_semana"] = df[inicio_col].dt.dayofweek
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_domingo"] = (df["dia_semana"] == 6).astype(int)

    # Agregaciones de salida por estación de origen
    salidas_agg = df.groupby(origen_col).agg(
        salidas_totales=(viaje_id_col, "count"),
        salidas_am=("hora", lambda x: ((x >= 7) & (x <= 10)).sum()),
        salidas_pm=("hora", lambda x: ((x >= 18) & (x <= 21)).sum()),
        salidas_fin_semana=("es_fin_de_semana", "sum"),
        salidas_domingo=("es_domingo", "sum"),
    )
    salidas_agg.index.name = "station_id"

    # Agregaciones de entrada por estación de destino
    entradas_agg = df.groupby(destino_col).agg(
        entradas_totales=(viaje_id_col, "count"),
        entradas_am=("hora", lambda x: ((x >= 7) & (x <= 10)).sum()),
        entradas_pm=("hora", lambda x: ((x >= 18) & (x <= 21)).sum()),
    )
    entradas_agg.index.name = "station_id"

    return salidas_agg, entradas_agg


def generate_p3_station_features(
    trips_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    features_list: List[str] = None,
) -> pd.DataFrame:
    """Construye la matriz de características para el clustering de estaciones."""
    salidas, entradas = compute_hourly_profiles(trips_df)

    stations = stations_df.copy()

    # Normalizar index o columna ID de estaciones
    station_id_candidates = ["id", "station_id", "id_estacion", "codigo"]
    found_id = None
    for cand in station_id_candidates:
        if cand in [c.lower() for c in stations.columns]:
            found_id = [
                c for c in stations.columns if c.lower() == cand
            ][0]
            break

    if found_id:
        stations = stations.set_index(found_id)

    stations.index.name = "station_id"

    # Unir asegurando que coincidan los IDs
    merged = stations.join([salidas, entradas], how="inner")
    merged = merged.fillna(0)

    eps = 1e-5

    # Ratios y normalizaciones
    merged["ratio_am"] = merged["salidas_am"] / (merged["entradas_am"] + eps)
    merged["ratio_pm"] = merged["salidas_pm"] / (merged["entradas_pm"] + eps)
    merged["pct_fin_semana"] = merged["salidas_fin_semana"] / (
        merged["salidas_totales"] + eps
    )
    merged["pct_domingo"] = merged["salidas_domingo"] / (
        merged["salidas_totales"] + eps
    )

    # Identificar columna de capacidad
    cap_candidates = ["capacidad", "capacity", "obcn", "slots"]
    cap_col = None
    for cand in cap_candidates:
        matches = [c for c in merged.columns if c.lower() == cand]
        if matches:
            cap_col = matches[0]
            break

    if cap_col:
        merged["rotacion_capacidad"] = (
            merged["salidas_totales"] + merged["entradas_totales"]
        ) / (merged[cap_col] + eps)
    else:
        merged["rotacion_capacidad"] = (
            merged["salidas_totales"] + merged["entradas_totales"]
        )

    # Resolver columnas de coordenadas
    lat_col = [
        c
        for c in merged.columns
        if c.lower() in ["latitude", "latitud", "lat"]
    ][0]
    lon_col = [
        c
        for c in merged.columns
        if c.lower() in ["longitude", "longitud", "lon", "lng"]
    ][0]

    # Renombrar a nombres estándar si difieren
    merged = merged.rename(columns={lat_col: "latitude", lon_col: "longitude"})

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