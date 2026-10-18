"""Pruebas pequeñas para las transformaciones de P2."""

import pandas as pd
import pytest

from mibici_mlops.features import (
    agregar_caracteristicas_historicas,
    agregar_objetivo_p2,
    asignar_conjunto_temporal,
    clasificar_presion,
)


def crear_ejemplo_estacion_hora(flujos: list[int]) -> pd.DataFrame:
    """Crea una serie horaria sencilla para una estación."""
    return pd.DataFrame(
        {
            "estacion_id": 1,
            "fecha_hora": pd.date_range("2024-01-01", periods=len(flujos), freq="h"),
            "entradas": [max(flujo, 0) for flujo in flujos],
            "salidas": [max(-flujo, 0) for flujo in flujos],
            "capacidad": 10,
        }
    )


def test_objetivo_usa_las_dos_horas_siguientes() -> None:
    datos = crear_ejemplo_estacion_hora([1, -1, -2, 4])

    resultado = agregar_objetivo_p2(datos)

    assert resultado.loc[0, "presion_futura_2h"] == pytest.approx(-0.3)
    assert resultado.loc[1, "presion_futura_2h"] == pytest.approx(0.2)
    assert pd.isna(resultado.loc[2, "presion_futura_2h"])


def test_caracteristicas_historicas_no_usen_hora_actual() -> None:
    datos = crear_ejemplo_estacion_hora([1, 2, 9])
    datos["flujo_neto"] = datos["entradas"] - datos["salidas"]

    resultado = agregar_caracteristicas_historicas(datos)

    assert resultado.loc[2, "flujo_neto_t_1"] == 2
    assert resultado.loc[2, "flujo_neto_t_2"] == 1
    assert resultado.loc[2, "flujo_neto_media_3h"] == pytest.approx(1.5)


def test_clasificacion_incluye_los_umbrales() -> None:
    presion = pd.Series([-0.3, -0.1, 0.2])

    resultado = clasificar_presion(
        presion,
        umbral_vaciado=-0.3,
        umbral_saturacion=0.2,
    )

    assert resultado.tolist() == [
        "riesgo_vaciado",
        "normal",
        "riesgo_saturacion",
    ]


def test_split_temporal_excluye_horas_que_cruzan_el_limite() -> None:
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
