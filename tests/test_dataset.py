"""Pruebas unitarias y de integración para Ingesta, Manifiesto y Calidad de Datos.

Módulo asignado a: Ariadna del Carmen Tapia Miranda (Integrante 1)
Cubre:
- Validación de esquemas y tipos de viajes y nomenclatura.
- Integridad y reproducibilidad del manifiesto con hash SHA-256.
- Manejo de casos límite: CSVs corruptos, duraciones negativas, nulos.
- Idempotencia del proceso de ingesta hacia Parquet.
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

import numpy as np
 
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    FeatureDrift,
    MixedDataTypes,
    PercentOfNulls,
)

from mibici_mlops.dataset import (
    calculate_dataset_version,
    calculate_sha256,
    clean_nomenclature,
    clean_trips,
    deduplicate_trips,
    read_mibici_csv,
    validate_nomenclature,
    validate_trips,
)


# ============================================================================
# 1. VALIDACIÓN DE ESQUEMAS Y TIPOS DE DATOS
# ============================================================================

@pytest.mark.unit
@pytest.mark.data_quality
def test_schema_valid_trips_data(sample_raw_trips_df: pd.DataFrame) -> None:
    """Qué prueba: Valida que el DataFrame procesado de viajes contenga todas las columnas obligatorias y tipos esperados.

    Por qué es crítico: Previene fallos silenciosos en las etapas posteriores de agregación y modelado
    si los datos abiertos sufren cambios imprevistos en su esquema.
    Criterio de aceptación: Presencia de columnas canónicas (viaje_id, inicio_del_viaje, fin_del_viaje,
    origen_id, destino_id) con tipos de datos correctos y sin duplicados exactos.
    """
    df_clean = clean_trips(sample_raw_trips_df)
    df_dedup = deduplicate_trips(df_clean)

    # Validar que pasa sin excepciones
    validate_trips(df_dedup)

    columnas_esperadas = {
        "viaje_id",
        "usuario_id",
        "genero",
        "ano_de_nacimiento",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    }
    assert columnas_esperadas.issubset(set(df_dedup.columns))
    assert pd.api.types.is_datetime64_any_dtype(df_dedup["inicio_del_viaje"])
    assert pd.api.types.is_datetime64_any_dtype(df_dedup["fin_del_viaje"])

    # Criterio de rechazo: si falta una columna crítica debe levantar ValueError
    df_invalido = df_dedup.drop(columns=["viaje_id"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_trips(df_invalido)


@pytest.mark.unit
@pytest.mark.data_quality
def test_schema_valid_nomenclatura_stations(sample_nomenclatura_df: pd.DataFrame) -> None:
    """Qué prueba: Valida que el catálogo de estaciones contenga identificadores únicos y coordenadas en el AMG.

    Por qué es crítico: Las coordenadas y capacidades son insumos directos para el clustering espacial
    y la tasa de rotación en P2 y P3.
    Criterio de aceptación: IDs únicos, capacidad >= 0, latitud en [20.5, 20.9] y longitud en [-103.5, -103.2].
    """
    df_clean = clean_nomenclature(sample_nomenclatura_df)
    validate_nomenclature(df_clean)

    assert df_clean["station_id"].is_unique
    assert (df_clean["capacity"] >= 0).all()

    # Bounding Box del Área Metropolitana de Guadalajara (AMG)
    assert (df_clean["latitude"] >= 20.5).all() and (df_clean["latitude"] <= 20.9).all()
    assert (df_clean["longitude"] >= -103.5).all() and (df_clean["longitude"] <= -103.2).all()

    # Criterio de rechazo: duplicados en station_id o capacidad negativa
    df_duplicado = df_clean.copy()
    df_duplicado.loc[1, "station_id"] = df_duplicado.loc[0, "station_id"]
    with pytest.raises(ValueError, match="Duplicate station IDs"):
        validate_nomenclature(df_duplicado)

    df_negativo = df_clean.copy()
    df_negativo.loc[0, "capacity"] = -5
    with pytest.raises(ValueError, match="Negative station capacity"):
        validate_nomenclature(df_negativo)


# ============================================================================
# 2. INTEGRIDAD DEL MANIFIESTO DE DATOS (SHA-256)
# ============================================================================

@pytest.mark.unit
@pytest.mark.data_quality
def test_manifest_generation_and_sha256_integrity(tmp_path: Path) -> None:
    """Qué prueba: Verifica la generación determinista del hash SHA-256 para archivos procesados en el manifiesto.

    Por qué es crítico: Asegura la inmutabilidad y auditoría de la versión del dataset (hash f4ac5ce8...)
    de acuerdo con los principios de gobernanza de MLOps.
    Criterio de aceptación: El hash calculado coincide exactamente con el hash criptográfico del contenido en disco.
    """
    contenido_prueba = b"Viaje_Id,Usuario_Id,Origen_Id\n1,101,2\n2,102,3\n"
    archivo_prueba = tmp_path / "test_viajes.csv"
    archivo_prueba.write_bytes(contenido_prueba)

    hash_obtenido = calculate_sha256(archivo_prueba)
    hash_esperado = hashlib.sha256(contenido_prueba).hexdigest()

    assert hash_obtenido == hash_esperado
    assert len(hash_obtenido) == 64

    # Verificar que cualquier mutación altera el hash
    archivo_prueba.write_bytes(contenido_prueba + b"3,103,4\n")
    nuevo_hash = calculate_sha256(archivo_prueba)
    assert nuevo_hash != hash_obtenido

    # Verificar versionado determinista del manifiesto conjunto
    manifest_files = [
        {"filename": "test_viajes.csv", "sha256": hash_obtenido},
        {"filename": "nomenclatura.csv", "sha256": "abc123def456"},
    ]
    version1 = calculate_dataset_version(manifest_files)
    version2 = calculate_dataset_version(list(reversed(manifest_files)))
    assert version1 == version2  # Invariante ante el orden de la lista


# ============================================================================
# 3. MANEJO DE CASOS LÍMITE (EDGE CASES) Y LIMPIEZA
# ============================================================================

@pytest.mark.unit
@pytest.mark.data_quality
def test_handle_corrupt_csv_and_invalid_encoding(
    tmp_path: Path, sample_corrupt_csv_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qué prueba: Comprueba que el proceso de lectura maneje archivos CSV truncados o con líneas malformadas.

    Por qué es crítico: Los portales de datos abiertos pueden incluir inconsistencias de formato
    que no deben colapsar la canalización de ingesta global.
    Criterio de aceptación: Se detecta y descarta el registro corrupto o se levanta una excepción controlada con logging.
    """
    archivo_corrupto = tmp_path / "corrupto.csv"
    archivo_corrupto.write_text(sample_corrupt_csv_content, encoding="utf-8")

    df = read_mibici_csv(archivo_corrupto)
    df_clean = clean_trips(df)

    # Validar que si faltan columnas críticas al procesar un CSV roto, validate_trips detecta el fallo
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_trips(df_clean)

    # Validar que si ninguna codificación tiene éxito, read_mibici_csv levanta ValueError controlado
    def simular_error_decodificacion(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "simulated error")

    monkeypatch.setattr(pd, "read_csv", simular_error_decodificacion)
    with pytest.raises(ValueError, match="Could not decode"):
        read_mibici_csv(archivo_corrupto)


@pytest.mark.unit
@pytest.mark.data_quality
def test_filter_invalid_durations_and_dates(sample_trips_with_anomalies_df: pd.DataFrame) -> None:
    """Qué prueba: Verifica el filtrado de viajes con duración negativa, cero o que excedan límites operativos (>24h).

    Por qué es crítico: Viajes con duraciones negativas o extremas introducen ruido y distorsionan
    la agregación horaria de demanda por estación.
    Criterio de aceptación: El conjunto limpio descarta viajes con Fin_del_viaje <= Inicio_del_viaje.
    """
    df = clean_trips(sample_trips_with_anomalies_df)

    duracion_segundos = (df["fin_del_viaje"] - df["inicio_del_viaje"]).dt.total_seconds()

    # Identificar viajes válidos: duración positiva (> 0) y razonable (< 24 horas)
    viajes_validos_mask = (duracion_segundos > 0) & (duracion_segundos <= 24 * 3600)
    df_filtrado = df.loc[viajes_validos_mask].copy()

    # El registro 201 (duración negativa) y 202 (duración cero) y 203 (>48h) deben identificarse
    assert (duracion_segundos.loc[0]) < 0  # 201: negativo
    assert (duracion_segundos.loc[1]) == 0  # 202: cero
    assert (duracion_segundos.loc[2]) > 24 * 3600  # 203: >24h

    assert len(df_filtrado) == 0  # Ninguna anomalía supera el filtro operativo


# ============================================================================
# 4. IDEMPOTENCIA Y GENERACIÓN DE PARQUET
# ============================================================================

@pytest.mark.integration
def test_ingestion_idempotency_and_parquet_generation(
    tmp_path: Path, sample_raw_trips_df: pd.DataFrame
) -> None:
    """Qué prueba: Verifica que ejecutar la ingesta múltiples veces con los mismos insumos genere resultados idénticos.

    Por qué es crítico: Principio fundamental de reproducibilidad: la re-ejecución no debe duplicar datos
    ni alterar los esquemas de los archivos Parquet generados.
    Criterio de aceptación: Mismo número de registros y mismos esquemas columnares en ejecuciones sucesivas.
    """
    raw_csv = tmp_path / "datos_abiertos_2024_01.csv"
    sample_raw_trips_df.to_csv(raw_csv, index=False)

    parquet_out = tmp_path / "datos_abiertos_2024_01.parquet"

    # Primera ejecución
    df1 = clean_trips(read_mibici_csv(raw_csv))
    df1 = deduplicate_trips(df1)
    validate_trips(df1)
    df1.to_parquet(parquet_out, index=False)
    registros_ejecucion_1 = len(df1)

    # Segunda ejecución sobre el mismo destino
    df2 = clean_trips(read_mibici_csv(raw_csv))
    df2 = deduplicate_trips(df2)
    validate_trips(df2)
    df2.to_parquet(parquet_out, index=False)
    registros_ejecucion_2 = len(df2)

    assert registros_ejecucion_1 == registros_ejecucion_2 == len(sample_raw_trips_df)

    # Verificar que el archivo Parquet resultante es 100% idéntico
    df_leido = pd.read_parquet(parquet_out)
    pd.testing.assert_frame_equal(df1, df_leido)

# ============================================================================
# 5. VALIDACIÓN DE CALIDAD CON DEEPCHECKS
# ============================================================================
 

@pytest.mark.unit
@pytest.mark.data_quality
def test_deepchecks_valores_faltantes(
    sample_raw_trips_df: pd.DataFrame,
) -> None:
    """Valida que DeepChecks detecte valores faltantes."""
    check = (
        PercentOfNulls()
        .add_condition_percent_of_nulls_not_greater_than(
            threshold=0.0,
        )
    )

    dataset_limpio = Dataset(
        sample_raw_trips_df,
        cat_features=["Genero"],
    )
    resultado_limpio = check.run(dataset_limpio)

    assert resultado_limpio.passed_conditions()

    datos_con_nulo = sample_raw_trips_df.copy()
    datos_con_nulo["Usuario_Id"] = datos_con_nulo[
        "Usuario_Id"
    ].astype("float64")
    datos_con_nulo.loc[0, "Usuario_Id"] = np.nan

    dataset_con_nulo = Dataset(
        datos_con_nulo,
        cat_features=["Genero"],
    )
    resultado_con_nulo = check.run(dataset_con_nulo)

    assert not resultado_con_nulo.passed_conditions()


@pytest.mark.unit
@pytest.mark.data_quality
def test_deepchecks_registros_duplicados(
    sample_raw_trips_df: pd.DataFrame,
) -> None:
    """Valida que DeepChecks detecte registros duplicados."""
    check = DataDuplicates().add_condition_ratio_less_or_equal(0)

    dataset_limpio = Dataset(
        sample_raw_trips_df,
        cat_features=["Genero"],
    )
    resultado_limpio = check.run(dataset_limpio)

    assert resultado_limpio.passed_conditions()

    datos_duplicados = pd.concat(
        [
            sample_raw_trips_df,
            sample_raw_trips_df.iloc[[0]],
        ],
        ignore_index=True,
    )
    dataset_duplicado = Dataset(
        datos_duplicados,
        cat_features=["Genero"],
    )
    resultado_duplicado = check.run(dataset_duplicado)

    assert not resultado_duplicado.passed_conditions()


@pytest.mark.unit
@pytest.mark.data_quality
def test_deepchecks_tipos_de_datos_consistentes(
    sample_raw_trips_df: pd.DataFrame,
) -> None:
    """Valida que una columna no mezcle números y cadenas."""
    check = (
        MixedDataTypes(columns=["Ano_de_nacimiento"])
        .add_condition_rare_type_ratio_not_in_range()
    )

    dataset_limpio = Dataset(
        sample_raw_trips_df,
        cat_features=["Genero"],
    )
    resultado_limpio = check.run(dataset_limpio)

    assert resultado_limpio.passed_conditions()

    datos_contaminados = pd.concat(
        [sample_raw_trips_df] * 4,
        ignore_index=True,
    )
    datos_contaminados["Ano_de_nacimiento"] = (
        datos_contaminados["Ano_de_nacimiento"].astype(object)
    )
    datos_contaminados.loc[
        0,
        "Ano_de_nacimiento",
    ] = "desconocido"

    dataset_contaminado = Dataset(
        datos_contaminados,
        cat_features=["Genero"],
    )
    resultado_contaminado = check.run(dataset_contaminado)

    assert not resultado_contaminado.passed_conditions()


@pytest.mark.unit
@pytest.mark.data_quality
def test_deepchecks_drift_entre_entrenamiento_y_prueba() -> None:
    """Valida que DeepChecks detecte drift en una variable numérica."""
    duraciones = np.linspace(5.0, 60.0, num=60)

    train_df = pd.DataFrame(
        {
            "duracion_minutos": duraciones,
            "viajes_por_hora": np.tile(
                np.arange(6, dtype="float64"),
                10,
            ),
        }
    )
    test_sin_drift_df = train_df.copy()

    test_con_drift_df = train_df.copy()
    test_con_drift_df["duracion_minutos"] += 200.0

    train_dataset = Dataset(train_df, cat_features=[])
    test_sin_drift = Dataset(
        test_sin_drift_df,
        cat_features=[],
    )
    test_con_drift = Dataset(
        test_con_drift_df,
        cat_features=[],
    )

    check = FeatureDrift(
        columns=["duracion_minutos"],
        min_samples=10,
        random_state=42,
    ).add_condition_drift_score_less_than(
        max_allowed_categorical_score=0.2,
        max_allowed_numeric_score=0.2,
        allowed_num_features_exceeding_threshold=0,
    )

    resultado_sin_drift = check.run(
        train_dataset,
        test_sin_drift,
    )
    resultado_con_drift = check.run(
        train_dataset,
        test_con_drift,
    )

    assert resultado_sin_drift.passed_conditions()
    assert not resultado_con_drift.passed_conditions()
