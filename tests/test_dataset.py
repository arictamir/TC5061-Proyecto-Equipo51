"""Pruebas unitarias y de integración para Ingesta, Manifiesto y Calidad de Datos.

Módulo asignado a: Ariadna del Carmen Tapia Miranda (Integrante 1)
Cubre:
- Validación de esquemas y tipos de viajes y nomenclatura.
- Integridad y reproducibilidad del manifiesto con hash SHA-256.
- Manejo de casos límite: CSVs corruptos, duraciones negativas, nulos.
- Idempotencia del proceso de ingesta hacia Parquet.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

# Compatibilidad de NumPy 2.x con DeepChecks
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    MixedDataTypes,
    PercentOfNulls,
)
import pandas as pd
import pytest

from mibici_mlops import dataset as dataset_module
from mibici_mlops import quality as quality_module
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
from mibici_mlops.quality import (
    remove_invalid_trip_durations,
    run_feature_drift_check,
    run_nomenclature_quality_checks,
    run_trip_quality_checks,
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
    df_filtrado, filas_invalidas = remove_invalid_trip_durations(df)

    assert filas_invalidas == 3
    assert df_filtrado.empty


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


@pytest.mark.integration
@pytest.mark.data_quality
def test_ingestion_runs_checks_and_documents_cleaning_actions(
    tmp_path: Path,
    sample_raw_trips_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La ingesta reporta problemas antes de deduplicar y persiste solo datos limpios."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    report_dir = tmp_path / "quality"
    raw_file = raw_dir / "sample_trips.csv"

    duplicate = sample_raw_trips_df.iloc[[0]]
    invalid_duration = sample_raw_trips_df.iloc[[1]].copy()
    invalid_duration.loc[:, "Viaje_Id"] = 999
    invalid_duration.loc[:, "Fin_del_viaje"] = "2024-01-01 08:00:00"
    source_df = pd.concat(
        [sample_raw_trips_df, duplicate, invalid_duration],
        ignore_index=True,
    )
    source_df.to_csv(raw_file, index=False)

    monkeypatch.setattr(
        dataset_module,
        "build_trip_filenames",
        lambda *_args: [raw_file.name],
    )
    monkeypatch.setattr(dataset_module, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(dataset_module, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(quality_module, "DATA_QUALITY_DIR", report_dir)

    output_files = dataset_module.process_trip_files()

    assert len(output_files) == 1
    processed_df = pd.read_parquet(output_files[0])
    assert len(processed_df) == len(sample_raw_trips_df)

    report = json.loads((report_dir / "sample_trips.json").read_text("utf-8"))
    assert report["status"] == "warning"
    assert report["metrics"]["exact_duplicate_rows"] == 1
    assert report["metrics"]["invalid_duration_rows"] == 1
    assert report["metrics"]["rows_after_cleaning"] == len(sample_raw_trips_df)
    assert report["actions"][0]["removed_rows"] == 1
    assert report["actions"][1]["removed_rows"] == 1
    assert "1001" not in json.dumps(report)


@pytest.mark.unit
@pytest.mark.data_quality
def test_trip_quality_blocks_critical_nulls_and_writes_report(
    tmp_path: Path,
    sample_raw_trips_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Los nulos críticos detienen el flujo y dejan evidencia resumida."""
    monkeypatch.setattr(quality_module, "DATA_QUALITY_DIR", tmp_path)
    trips = clean_trips(sample_raw_trips_df)
    trips.loc[0, "viaje_id"] = np.nan

    with pytest.raises(ValueError, match="null values in critical columns"):
        run_trip_quality_checks(trips, "invalid_trips.csv")

    report = json.loads((tmp_path / "invalid_trips.json").read_text("utf-8"))
    assert report["status"] == "failed"
    assert report["metrics"]["critical_null_counts"]["viaje_id"] == 1
    assert "1001" not in json.dumps(report)


@pytest.mark.unit
@pytest.mark.data_quality
def test_nomenclature_quality_blocks_duplicate_station_ids(
    tmp_path: Path,
    sample_nomenclatura_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepChecks detects duplicate station identifiers in the live ingestion check."""
    monkeypatch.setattr(quality_module, "DATA_QUALITY_DIR", tmp_path)
    stations = clean_nomenclature(sample_nomenclatura_df)
    stations.loc[1, "station_id"] = stations.loc[0, "station_id"]

    with pytest.raises(ValueError, match="Duplicate station IDs"):
        run_nomenclature_quality_checks(stations, "stations.csv")

    report = json.loads((tmp_path / "stations.json").read_text("utf-8"))
    assert report["status"] == "failed"
    assert report["metrics"]["duplicate_station_ids"] == 1


@pytest.mark.integration
@pytest.mark.data_quality
def test_nomenclature_ingestion_generates_quality_report(
    tmp_path: Path,
    sample_nomenclatura_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    report_dir = tmp_path / "quality"
    filename = "sample_stations.csv"
    sample_nomenclatura_df.to_csv(raw_dir / filename, index=False)

    monkeypatch.setattr(dataset_module, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(dataset_module, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(dataset_module, "NOMENCLATURE_FILE", filename)
    monkeypatch.setattr(quality_module, "DATA_QUALITY_DIR", report_dir)

    output_file = dataset_module.process_nomenclature()

    assert output_file.exists()
    report = json.loads((report_dir / "sample_stations.json").read_text("utf-8"))
    assert report["status"] == "passed"
    assert report["metrics"]["rows"] == len(sample_nomenclatura_df)


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
def test_deepchecks_drift_entre_entrenamiento_y_prueba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valida drift entre splits y registra sus resultados resumidos."""
    monkeypatch.setattr(quality_module, "DATA_QUALITY_DIR", tmp_path)
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

    report_sin_drift = run_feature_drift_check(
        train_df,
        test_sin_drift_df,
        ["duracion_minutos", "viajes_por_hora"],
        [],
        "no_drift",
    )
    report_con_drift = run_feature_drift_check(
        train_df,
        test_con_drift_df,
        ["duracion_minutos", "viajes_por_hora"],
        [],
        "with_drift",
    )
    report_json = json.loads((tmp_path / "with_drift.json").read_text("utf-8"))

    assert report_sin_drift["status"] == "passed"
    assert report_con_drift["status"] == "warning"
    assert report_json["metrics"]["sample_size_per_split"] == 60
    assert not report_json["checks"]["feature_drift"]["passed"]
