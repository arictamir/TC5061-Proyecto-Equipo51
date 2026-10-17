"""Construcción reproducible del conjunto estación-hora para P2."""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici_mlops.config import (
    END_MONTH,
    END_YEAR,
    NOMENCLATURE_FILE,
    P2_DIR,
    P2_FIN_ENTRENAMIENTO,
    P2_FIN_PRUEBA,
    P2_FIN_VALIDACION,
    P2_HORIZONTE_HORAS,
    P2_INICIO_ENTRENAMIENTO,
    P2_INICIO_PRUEBA,
    P2_INICIO_VALIDACION,
    P2_UMBRAL_SATURACION,
    P2_UMBRAL_VACIADO,
    PROCESSED_DATA_DIR,
    START_MONTH,
    START_YEAR,
)

app = typer.Typer()

COLUMNAS_CARACTERISTICAS_P2 = [
    "estacion_id",
    "capacidad",
    "latitud",
    "longitud",
    "hora",
    "dia_semana",
    "mes",
    "es_fin_semana",
    "entradas_t_1",
    "salidas_t_1",
    "flujo_neto_t_1",
    "flujo_neto_t_2",
    "flujo_neto_t_24",
    "entradas_media_24h",
    "salidas_media_24h",
    "flujo_neto_media_3h",
    "flujo_neto_media_24h",
]


def construir_nombres_mensuales(
    ano_inicial: int,
    mes_inicial: int,
    ano_final: int,
    mes_final: int,
) -> list[str]:
    """Construye los nombres esperados de los Parquet mensuales."""
    periodos = pd.period_range(
        start=f"{ano_inicial}-{mes_inicial:02d}",
        end=f"{ano_final}-{mes_final:02d}",
        freq="M",
    )
    return [f"datos_abiertos_{periodo.year}_{periodo.month:02d}.parquet" for periodo in periodos]


def obtener_rutas_viajes(directorio: Path) -> list[Path]:
    """Devuelve las rutas mensuales configuradas y verifica que existan."""
    nombres = construir_nombres_mensuales(
        START_YEAR,
        START_MONTH,
        END_YEAR,
        END_MONTH,
    )
    rutas = [directorio / nombre for nombre in nombres]
    faltantes = [ruta.name for ruta in rutas if not ruta.exists()]

    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos de viajes procesados: " + ", ".join(faltantes)
        )

    return rutas


def agrupar_viajes_por_estacion_hora(ruta: Path) -> pd.DataFrame:
    """Cuenta entradas y salidas por estación-hora para un mes."""
    viajes = pd.read_parquet(
        ruta,
        columns=[
            "inicio_del_viaje",
            "fin_del_viaje",
            "origen_id",
            "destino_id",
        ],
    )

    salidas = (
        viajes.assign(
            estacion_id=viajes["origen_id"],
            fecha_hora=viajes["inicio_del_viaje"].dt.floor("h"),
        )[["estacion_id", "fecha_hora"]]
        .groupby(["estacion_id", "fecha_hora"])
        .size()
        .rename("salidas")
        .reset_index()
    )
    entradas = (
        viajes.assign(
            estacion_id=viajes["destino_id"],
            fecha_hora=viajes["fin_del_viaje"].dt.floor("h"),
        )[["estacion_id", "fecha_hora"]]
        .groupby(["estacion_id", "fecha_hora"])
        .size()
        .rename("entradas")
        .reset_index()
    )

    return salidas.merge(
        entradas,
        how="outer",
        on=["estacion_id", "fecha_hora"],
    )


def crear_flujos_estacion_hora(rutas: list[Path]) -> pd.DataFrame:
    """Agrupa todos los meses en una tabla de movimientos estación-hora."""
    movimientos = []
    total = len(rutas)

    for numero, ruta in enumerate(rutas, start=1):
        movimientos.append(agrupar_viajes_por_estacion_hora(ruta))
        logger.info(f"Mes procesado {numero:02d}/{total}: {ruta.name}")

    flujos = (
        pd.concat(movimientos, ignore_index=True)
        .groupby(["estacion_id", "fecha_hora"], as_index=False)[["entradas", "salidas"]]
        .sum()
    )
    flujos[["entradas", "salidas"]] = (
        flujos[["entradas", "salidas"]].fillna(0).astype("int32")
    )

    inicio = pd.Timestamp(P2_INICIO_ENTRENAMIENTO)
    fin = pd.Timestamp(P2_FIN_PRUEBA)
    return flujos[flujos["fecha_hora"].between(inicio, fin, inclusive="left")].copy()


def cargar_nomenclatura(ruta: Path) -> pd.DataFrame:
    """Carga las columnas de estación necesarias para P2."""
    nomenclatura = pd.read_parquet(
        ruta,
        columns=["station_id", "latitude", "longitude", "capacity"],
    ).rename(
        columns={
            "station_id": "estacion_id",
            "latitude": "latitud",
            "longitude": "longitud",
            "capacity": "capacidad",
        }
    )

    if nomenclatura["estacion_id"].duplicated().any():
        raise ValueError("La nomenclatura contiene identificadores de estación duplicados.")

    return nomenclatura


def filtrar_estaciones_sin_capacidad(
    flujos: pd.DataFrame,
    nomenclatura: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reporta y excluye estaciones sin una capacidad positiva."""
    estaciones_validas = nomenclatura.loc[
        nomenclatura["capacidad"].gt(0), "estacion_id"
    ]
    ids_validos = set(estaciones_validas.tolist())
    ids_observados = set(flujos["estacion_id"].unique().tolist())
    ids_excluidos = sorted(ids_observados - ids_validos)

    if ids_excluidos:
        logger.warning(
            "Se excluirán estaciones sin capacidad válida: "
            + ", ".join(str(valor) for valor in ids_excluidos)
        )

    flujos_validos = flujos[flujos["estacion_id"].isin(ids_validos)].copy()
    nomenclatura_valida = nomenclatura[
        nomenclatura["estacion_id"].isin(ids_validos)
    ].copy()
    return flujos_validos, nomenclatura_valida


def crear_rejilla_horas_activas(flujos: pd.DataFrame) -> pd.DataFrame:
    """Crea una fila por hora entre la primera y última actividad de cada estación."""
    intervalos = (
        flujos.groupby("estacion_id")["fecha_hora"]
        .agg(inicio="min", fin="max")
        .reset_index()
    )
    partes = [
        pd.DataFrame(
            {
                "estacion_id": fila.estacion_id,
                "fecha_hora": pd.date_range(fila.inicio, fila.fin, freq="h"),
            }
        )
        for fila in intervalos.itertuples(index=False)
    ]
    return pd.concat(partes, ignore_index=True)


def completar_horas_y_estaciones(
    flujos: pd.DataFrame,
    nomenclatura: pd.DataFrame,
) -> pd.DataFrame:
    """Completa horas sin viajes y agrega capacidad y ubicación."""
    rejilla = crear_rejilla_horas_activas(flujos)
    estacion_hora = rejilla.merge(
        flujos,
        how="left",
        on=["estacion_id", "fecha_hora"],
    )
    estacion_hora[["entradas", "salidas"]] = (
        estacion_hora[["entradas", "salidas"]].fillna(0).astype("int32")
    )
    estacion_hora = estacion_hora.merge(
        nomenclatura,
        how="left",
        on="estacion_id",
        validate="many_to_one",
    )

    if estacion_hora[["capacidad", "latitud", "longitud"]].isna().any().any():
        raise ValueError("Quedaron estaciones sin información después de aplicar el filtro.")

    return estacion_hora.sort_values(["estacion_id", "fecha_hora"]).reset_index(drop=True)


def agregar_objetivo_p2(estacion_hora: pd.DataFrame) -> pd.DataFrame:
    """Calcula la presión acumulada de las dos horas siguientes."""
    resultado = estacion_hora.copy()
    resultado["flujo_neto"] = resultado["entradas"] - resultado["salidas"]
    flujo_por_estacion = resultado.groupby("estacion_id", sort=False)["flujo_neto"]
    resultado["presion_futura_2h"] = (
        flujo_por_estacion.shift(-1) + flujo_por_estacion.shift(-2)
    ) / resultado["capacidad"]
    return resultado


def agregar_caracteristicas_historicas(estacion_hora: pd.DataFrame) -> pd.DataFrame:
    """Crea variables disponibles antes de la hora que se quiere predecir."""
    resultado = estacion_hora.copy()
    resultado["hora"] = resultado["fecha_hora"].dt.hour.astype("int8")
    resultado["dia_semana"] = resultado["fecha_hora"].dt.dayofweek.astype("int8")
    resultado["mes"] = resultado["fecha_hora"].dt.month.astype("int8")
    resultado["es_fin_semana"] = resultado["dia_semana"].isin([5, 6]).astype("int8")

    agrupado = resultado.groupby("estacion_id", sort=False)
    resultado["entradas_t_1"] = agrupado["entradas"].shift(1)
    resultado["salidas_t_1"] = agrupado["salidas"].shift(1)
    resultado["flujo_neto_t_1"] = agrupado["flujo_neto"].shift(1)
    resultado["flujo_neto_t_2"] = agrupado["flujo_neto"].shift(2)
    resultado["flujo_neto_t_24"] = agrupado["flujo_neto"].shift(24)

    resultado["entradas_media_24h"] = agrupado["entradas"].transform(
        lambda serie: serie.shift(1).rolling(24, min_periods=1).mean()
    )
    resultado["salidas_media_24h"] = agrupado["salidas"].transform(
        lambda serie: serie.shift(1).rolling(24, min_periods=1).mean()
    )
    resultado["flujo_neto_media_3h"] = agrupado["flujo_neto"].transform(
        lambda serie: serie.shift(1).rolling(3, min_periods=1).mean()
    )
    resultado["flujo_neto_media_24h"] = agrupado["flujo_neto"].transform(
        lambda serie: serie.shift(1).rolling(24, min_periods=1).mean()
    )
    return resultado


def clasificar_presion(
    presion: pd.Series,
    umbral_vaciado: float = P2_UMBRAL_VACIADO,
    umbral_saturacion: float = P2_UMBRAL_SATURACION,
) -> pd.Series:
    """Convierte la presión futura en las tres clases de P2."""
    clase = pd.Series("normal", index=presion.index, dtype="string")
    clase.loc[presion <= umbral_vaciado] = "riesgo_vaciado"
    clase.loc[presion >= umbral_saturacion] = "riesgo_saturacion"
    return clase


def asignar_conjunto_temporal(fecha_hora: pd.Series) -> pd.Series:
    """Asigna train, validación o prueba sin cruzar el horizonte futuro."""
    conjunto = pd.Series(pd.NA, index=fecha_hora.index, dtype="string")
    periodos = [
        ("entrenamiento", P2_INICIO_ENTRENAMIENTO, P2_FIN_ENTRENAMIENTO),
        ("validacion", P2_INICIO_VALIDACION, P2_FIN_VALIDACION),
        ("prueba", P2_INICIO_PRUEBA, P2_FIN_PRUEBA),
    ]

    for nombre, inicio, fin in periodos:
        fin_seguro = pd.Timestamp(fin) - pd.Timedelta(hours=P2_HORIZONTE_HORAS)
        mascara = fecha_hora.between(
            pd.Timestamp(inicio),
            fin_seguro,
            inclusive="left",
        )
        conjunto.loc[mascara] = nombre

    return conjunto


def preparar_dataset_p2(
    rutas_viajes: list[Path],
    ruta_nomenclatura: Path,
) -> pd.DataFrame:
    """Ejecuta todas las transformaciones reproducibles de P2."""
    flujos = crear_flujos_estacion_hora(rutas_viajes)
    nomenclatura = cargar_nomenclatura(ruta_nomenclatura)
    flujos, nomenclatura = filtrar_estaciones_sin_capacidad(flujos, nomenclatura)
    estacion_hora = completar_horas_y_estaciones(flujos, nomenclatura)
    estacion_hora = agregar_objetivo_p2(estacion_hora)
    estacion_hora = agregar_caracteristicas_historicas(estacion_hora)
    estacion_hora["clase_riesgo"] = clasificar_presion(
        estacion_hora["presion_futura_2h"]
    )
    estacion_hora["conjunto"] = asignar_conjunto_temporal(estacion_hora["fecha_hora"])

    columnas_salida = [
        "fecha_hora",
        *COLUMNAS_CARACTERISTICAS_P2,
        "presion_futura_2h",
        "clase_riesgo",
        "conjunto",
    ]
    dataset = estacion_hora[columnas_salida].dropna().copy()
    dataset["estacion_id"] = dataset["estacion_id"].astype("int32")
    dataset["capacidad"] = dataset["capacidad"].astype("int16")
    dataset[["latitud", "longitud"]] = dataset[["latitud", "longitud"]].astype("float32")
    dataset["clase_riesgo"] = dataset["clase_riesgo"].astype("category")
    dataset["conjunto"] = dataset["conjunto"].astype("category")
    validar_dataset_p2(dataset)
    return dataset


def validar_dataset_p2(dataset: pd.DataFrame) -> None:
    """Valida esquema, valores faltantes y unicidad del resultado."""
    columnas_requeridas = {
        "fecha_hora",
        *COLUMNAS_CARACTERISTICAS_P2,
        "presion_futura_2h",
        "clase_riesgo",
        "conjunto",
    }
    faltantes = columnas_requeridas - set(dataset.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    if dataset[list(columnas_requeridas)].isna().any().any():
        raise ValueError("El conjunto P2 contiene valores faltantes.")
    if dataset.duplicated(["estacion_id", "fecha_hora"]).any():
        raise ValueError("El conjunto P2 contiene filas estación-hora duplicadas.")
    if not set(dataset["clase_riesgo"].astype("string").unique()).issubset(
        {"riesgo_vaciado", "normal", "riesgo_saturacion"}
    ):
        raise ValueError("El conjunto P2 contiene clases desconocidas.")


@app.command()
def main(
    output_path: Path = P2_DIR / "dataset_p2.parquet",
) -> None:
    """Genera el conjunto reproducible de características y objetivo de P2."""
    directorio_viajes = PROCESSED_DATA_DIR / "trips"
    ruta_nomenclatura = (
        PROCESSED_DATA_DIR
        / "nomenclature"
        / NOMENCLATURE_FILE.replace(".csv", ".parquet")
    )
    rutas_viajes = obtener_rutas_viajes(directorio_viajes)

    if not ruta_nomenclatura.exists():
        raise FileNotFoundError(f"No existe la nomenclatura: {ruta_nomenclatura}")

    logger.info("Construyendo el conjunto estación-hora de P2.")
    dataset = preparar_dataset_p2(rutas_viajes, ruta_nomenclatura)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False, compression="snappy")

    logger.success(
        f"Dataset P2 creado: {output_path} | "
        f"filas={len(dataset):,} | estaciones={dataset['estacion_id'].nunique()}"
    )
    logger.info(
        "Filas por conjunto: "
        + str(dataset["conjunto"].value_counts().sort_index().to_dict())
    )


if __name__ == "__main__":
    app()
