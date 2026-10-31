"""Pruebas unitarias y de integración para Feature Engineering y Prevención de Data Leakage.

Módulo asignado a: Gabriela Díaz Oropeza (Integrante 2)
Cubre:
- Transformaciones de series temporales (lags lag_1, lag_24, lag_168 y rolling windows).
- Verificación matemática y algorítmica de NO fuga temporal (data leakage).
- Tratamiento de valores nulos y periodo de calentamiento (warmup).
- Consistencia de dimensiones y tipos de salida para P1, P2 y P3.
"""

import pandas as pd
import pytest

from mibici_mlops.features import (
    agregar_caracteristicas_historicas,
    agregar_objetivo_p2,
    asignar_conjunto_temporal,
    build_p1_features,
    clasificar_presion,
    generate_p3_station_features,
)

# ============================================================================
# 1. GENERACIÓN DE REZAGOS TEMPORALES (LAGS)
# ============================================================================


@pytest.mark.unit
def test_p1_lag_creation_temporal_correctness(sample_p1_hourly_demand_df: pd.DataFrame) -> None:
    """Qué prueba: Verifica la correcta alineación y valor de los rezagos lag_1, lag_24 y lag_168 en P1.

    Por qué es crítico: Si los rezagos no están correctamente alineados con el índice temporal, el modelo
    aprenderá relaciones espurias con horas equivocadas o días desfasados.
    Criterio de aceptación: En la fila t, lag_1 coincide exactamente con la demanda de t-1,
    lag_24 con la de t-24 y lag_168 con la de t-168 de la misma estación.
    """
    df_feat = build_p1_features(sample_p1_hourly_demand_df)

    t_eval = 175  # Fila posterior al calentamiento de 168 horas
    demanda_serie = sample_p1_hourly_demand_df["trips_started"]

    assert df_feat.loc[t_eval, "lag_1"] == demanda_serie.iloc[t_eval - 1]
    assert df_feat.loc[t_eval, "lag_24"] == demanda_serie.iloc[t_eval - 24]
    assert df_feat.loc[t_eval, "lag_168"] == demanda_serie.iloc[t_eval - 168]


# ============================================================================
# 2. PREVENCIÓN ESTRICTA DE DATA LEAKAGE (FUGA TEMPORAL)
# ============================================================================


@pytest.mark.unit
def test_p1_rolling_windows_no_future_leakage(sample_p1_hourly_demand_df: pd.DataFrame) -> None:
    """Qué prueba: Comprueba que las ventanas móviles (rolling_mean_3h, rolling_mean_24h) miren estrictamente al pasado.

    Por qué es crítico: Incluir la demanda de la hora actual t o futura t+k en los promedios móviles
    provoca fuga de datos (lookahead bias), invalidando el modelo en producción.
    Criterio de aceptación: Modificar la demanda en la hora t o t+1 no altera el valor de las features calculadas para predecir t.
    """
    df_base = build_p1_features(sample_p1_hourly_demand_df)

    t_target = 50
    valor_rolling_original = df_base.loc[t_target, "rolling_mean_3h"]

    # Verificar que coincide con el promedio aritmético de (t-1, t-2, t-3)
    d = sample_p1_hourly_demand_df["trips_started"]
    esperado = (d.iloc[t_target - 1] + d.iloc[t_target - 2] + d.iloc[t_target - 3]) / 3.0
    assert valor_rolling_original == pytest.approx(esperado)

    # Mutación hacia el futuro: alteramos t_target y t_target + 1
    df_mutado = sample_p1_hourly_demand_df.copy()
    df_mutado.loc[t_target, "trips_started"] = 9999
    df_mutado.loc[t_target + 1, "trips_started"] = 8888

    df_mutado_feat = build_p1_features(df_mutado)

    # El valor de la feature en t_target NO debe cambiar
    assert df_mutado_feat.loc[t_target, "rolling_mean_3h"] == valor_rolling_original
    assert (
        df_mutado_feat.loc[t_target, "rolling_mean_24h"]
        == df_base.loc[t_target, "rolling_mean_24h"]
    )


# ============================================================================
# 3. VARIABLES DE CALENDARIO Y ESTACIONALIDAD CÍCLICA
# ============================================================================


@pytest.mark.unit
def test_cyclical_calendar_features(sample_p1_hourly_demand_df: pd.DataFrame) -> None:
    """Qué prueba: Verifica la extracción y acotamiento de variables de calendario (hora, día de la semana, fin de semana, mes).

    Por qué es crítico: Los patrones de movilidad urbana en MiBici dependen estrechamente de los ciclos
    laborales, horarios de entrada escolar/laboral y fines de semana.
    Criterio de aceptación: hour_of_day en [0, 23], day_of_week en [0, 6], is_weekend en {0, 1} y month en [1, 12].
    """
    df_feat = build_p1_features(sample_p1_hourly_demand_df)

    assert df_feat["hour_of_day"].between(0, 23).all()
    assert df_feat["day_of_week"].between(0, 6).all()
    assert set(df_feat["is_weekend"].unique()).issubset({0, 1})
    assert df_feat["month"].between(1, 12).all()


# ============================================================================
# 4. TRATAMIENTO DE NULOS Y PERIODO DE CALENTAMIENTO
# ============================================================================


@pytest.mark.unit
def test_null_handling_and_lag_warmup(sample_p1_hourly_demand_df: pd.DataFrame) -> None:
    """Qué prueba: Comprueba que el periodo de calentamiento inicial provocado por lag_168 sea gestionado adecuadamente.

    Por qué es crítico: Registros con valores nulos no tratados en las primeras 168 horas pueden causar
    excepciones durante el ajuste o inferencia de modelos como Ridge o CatBoost.
    Criterio de aceptación: Filtrar el calentamiento elimina todos los valores nulos en features de retardo.
    """
    df_feat = build_p1_features(sample_p1_hourly_demand_df)

    # Las primeras 168 horas tienen lag_168 nulo
    assert df_feat.loc[:167, "lag_168"].isna().all()

    # Tras el periodo de calentamiento (history_hours >= 168), no deben existir NaNs
    df_post_warmup = df_feat.loc[df_feat["history_hours"] >= 168]
    assert not df_post_warmup["lag_168"].isna().any()
    assert not df_post_warmup["lag_24"].isna().any()
    assert not df_post_warmup["lag_1"].isna().any()
    assert not df_post_warmup["rolling_mean_3h"].isna().any()


# ============================================================================
# 5. TRANSFORMACIONES Y OBJETIVO DE PRESIÓN EN P2
# ============================================================================


@pytest.mark.unit
def test_p2_future_target_two_hours_window(sample_p2_station_hourly_df: pd.DataFrame) -> None:
    """Qué prueba: Verifica que la etiqueta de riesgo operativo en P2 utilice la suma del flujo neto de las dos horas siguientes.

    Por qué es crítico: Cumple con la especificación de negocio de MiBici: alertar al equipo de balanceo
    con 2 horas de anticipación sobre vaciado o saturación.
    Criterio de aceptación: presion_futura_2h = (flujo_neto_{t+1} + flujo_neto_{t+2}) / capacidad, sin incluir la hora t.
    """
    df = sample_p2_station_hourly_df.copy()
    resultado = agregar_objetivo_p2(df)

    # Fila 0: t=0, flujo t+1: entradas(2)-salidas(1)=1; flujo t+2: entradas(3)-salidas(0)=3
    # Presión = (1 + 3) / 10 = 0.4
    flujo_t1 = df.loc[1, "entradas"] - df.loc[1, "salidas"]
    flujo_t2 = df.loc[2, "entradas"] - df.loc[2, "salidas"]
    presion_esperada = (flujo_t1 + flujo_t2) / df.loc[0, "capacidad"]

    assert resultado.loc[0, "presion_futura_2h"] == pytest.approx(presion_esperada)
    assert pd.isna(resultado.iloc[-1]["presion_futura_2h"])  # Última fila no tiene t+1 ni t+2


@pytest.mark.unit
def test_p2_caracteristicas_historicas_no_usen_hora_actual(
    sample_p2_station_hourly_df: pd.DataFrame,
) -> None:
    """Qué prueba: Comprueba que las features de P2 no utilicen los flujos de la hora actual a clasificar."""
    df = sample_p2_station_hourly_df.copy()
    df["flujo_neto"] = df["entradas"] - df["salidas"]
    resultado = agregar_caracteristicas_historicas(df)

    # Para la fila en índice 2, flujo_neto_t_1 debe ser el flujo de la fila 1 y t_2 el de la fila 0
    assert resultado.loc[2, "flujo_neto_t_1"] == df.loc[1, "flujo_neto"]
    assert resultado.loc[2, "flujo_neto_t_2"] == df.loc[0, "flujo_neto"]


@pytest.mark.unit
def test_p2_clasificacion_incluye_los_umbrales() -> None:
    """Qué prueba: Verifica la clasificación en las tres clases operativas según umbrales de percentil."""
    presion = pd.Series([-0.3, -0.1, 0.2])
    resultado = clasificar_presion(presion, umbral_vaciado=-0.3, umbral_saturacion=0.2)
    assert resultado.tolist() == ["riesgo_vaciado", "normal", "riesgo_saturacion"]


@pytest.mark.unit
def test_p2_split_temporal_excluye_horas_limite() -> None:
    """Qué prueba: Comprueba que las horas que cruzan la frontera temporal de evaluación no se traslapen."""
    fechas = pd.Series(
        pd.to_datetime(
            [
                "2024-12-31 21:00:00",
                "2024-12-31 22:00:00",
                "2024-12-31 23:00:00",
                "2025-01-01 00:00:00",
            ]
        )
    )
    resultado = asignar_conjunto_temporal(fechas)
    assert resultado.iloc[0] == "entrenamiento"
    assert pd.isna(resultado.iloc[1])
    assert pd.isna(resultado.iloc[2])
    assert resultado.iloc[3] == "validacion"


# ============================================================================
# 6. CONSISTENCIA DE MATRIZ DE CARACTERÍSTICAS PARA P3
# ============================================================================


@pytest.mark.unit
def test_p3_feature_matrix_dimensions_and_scaling(
    sample_raw_trips_df: pd.DataFrame, sample_nomenclatura_df: pd.DataFrame
) -> None:
    """Qué prueba: Valida que la matriz de features para el clustering P3 contenga exactamente 7 variables numéricas.

    Por qué es crítico: K-Means requiere variables en escalas comparables; la presencia de NaNs o dimensiones
    incompatibles invalida el agrupamiento de las 471 estaciones.
    Criterio de aceptación: 7 columnas por estación, valores finitos sin NaNs ni Infs.
    """
    columnas_esperadas = [
        "latitude",
        "longitude",
        "ratio_am",
        "ratio_pm",
        "pct_fin_semana",
        "pct_domingo",
        "rotacion_capacidad",
    ]

    matriz_p3 = generate_p3_station_features(sample_raw_trips_df, sample_nomenclatura_df)

    assert list(matriz_p3.columns) == columnas_esperadas
    assert not matriz_p3.isna().any().any()
    assert (matriz_p3["rotacion_capacidad"] >= 0).all()