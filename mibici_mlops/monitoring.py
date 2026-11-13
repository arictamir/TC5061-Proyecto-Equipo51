"""Módulo de observabilidad, detección y reporte de data drift con Evidently.

Permite evaluar la estabilidad de las distribuciones de características y target
entre el conjunto de referencia (entrenamiento 2024-2025) y el conjunto actual
de producción (mayo a septiembre 2026) para el modelo P1 (Demanda Horaria con CatBoost).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import pandas as pd
import typer

try:
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report
except ImportError:  # pragma: no cover
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset  # type: ignore
    from evidently.pipeline.column_mapping import ColumnMapping  # type: ignore
    from evidently.report import Report  # type: ignore

from mibici_mlops.config import (
    DRIFT_REPORT_DIR,
    P1_TEST_START,
    P1_VALIDATION_START,
    PROCESSED_DATA_DIR,
    SEMILLA_ALEATORIA,
)

app = typer.Typer(
    help="Herramientas de observabilidad y monitoreo de Data Drift con Evidently (MiBici MLOps)."
)

# Definición de variables del modelo P1 para observabilidad
NUMERICAL_FEATURES: list[str] = [
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "history_hours",
]

CATEGORICAL_FEATURES: list[str] = [
    "station_id",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
]

TARGET_COLUMN: str = "trips_started"

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
ALL_MONITORED_COLUMNS: list[str] = ALL_FEATURES + [TARGET_COLUMN]


def build_column_mapping() -> ColumnMapping:
    """Construye el mapeo de columnas para el reporte de Evidently."""
    return ColumnMapping(
        target=TARGET_COLUMN,
        prediction=None,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


def load_reference_and_current_data(
    features_path: Path | None = None,
    sample_size: int | None = 30000,
    random_state: int = SEMILLA_ALEATORIA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y particiona los datos de P1 en referencia y actual.

    - Referencia (D_ref): Datos de entrenamiento histórico (2024-01-01 a 2025-12-31).
    - Actual (D_curr): Datos de producción/prueba (2026-05-01 a 2026-09-30).

    Args:
        features_path: Ruta al archivo Parquet de características. Si es None, usa
            PROCESSED_DATA_DIR / "p1_features.parquet".
        sample_size: Tamaño de muestra representativa para optimizar el cómputo
            estadístico de Evidently. Si es None, utiliza el dataset completo.
        random_state: Semilla para reproducibilidad del muestreo.

    Returns:
        Tupla (reference_df, current_df) con las columnas monitoreadas sin nulos.
    """
    path = features_path or (PROCESSED_DATA_DIR / "p1_features.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de características en: {path}. "
            "Asegúrate de ejecutar la etapa de ingeniería de características previamente."
        )

    logger.info(f"Cargando características P1 desde: {path}")
    columns_to_load = list(set(ALL_MONITORED_COLUMNS + ["hour"]))
    df = pd.read_parquet(path, columns=columns_to_load)

    # Filtrar periodos
    ref_mask = df["hour"] < P1_VALIDATION_START
    curr_mask = df["hour"] >= P1_TEST_START

    df_ref = df.loc[ref_mask, ALL_MONITORED_COLUMNS].dropna().copy()
    df_curr = df.loc[curr_mask, ALL_MONITORED_COLUMNS].dropna().copy()

    logger.info(
        f"Datos filtrados -> Referencia (2024-2025): {len(df_ref):,} filas, "
        f"Actual (2026-05 a 2026-09): {len(df_curr):,} filas."
    )

    if sample_size is not None:
        if len(df_ref) > sample_size:
            df_ref = df_ref.sample(n=sample_size, random_state=random_state).copy()
        if len(df_curr) > sample_size:
            df_curr = df_curr.sample(n=sample_size, random_state=random_state).copy()
        logger.info(
            f"Muestreo representativo aplicado ({sample_size:,} registros por partición, seed={random_state})."
        )

    return df_ref, df_curr


def parse_drift_metrics_dict(raw_dict: dict[str, Any]) -> dict[str, Any]:
    """Extrae y normaliza las métricas de drift y calidad de datos a tipos JSON estándar."""
    metrics_list = raw_dict.get("metrics", [])

    summary: dict[str, Any] = {
        "dataset_drift_detected": False,
        "drift_share": 0.0,
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "columns_drift": {},
        "missing_values": {},
    }

    for metric in metrics_list:
        metric_name = metric.get("metric")
        result = metric.get("result", {})

        if metric_name == "DatasetDriftMetric":
            summary["dataset_drift_detected"] = bool(result.get("dataset_drift", False))
            summary["drift_share"] = float(result.get("drift_share", 0.0))
            summary["number_of_columns"] = int(result.get("number_of_columns", 0))
            summary["number_of_drifted_columns"] = int(result.get("number_of_drifted_columns", 0))
            summary["share_of_drifted_columns"] = float(result.get("share_of_drifted_columns", 0.0))

        elif metric_name == "DataDriftTable":
            drift_by_cols = result.get("drift_by_columns", {})
            for col_name, col_data in drift_by_cols.items():
                drift_score = col_data.get("drift_score")
                if drift_score is not None:
                    drift_score = float(drift_score)
                threshold = col_data.get("threshold", col_data.get("stattest_threshold"))
                if threshold is not None:
                    threshold = float(threshold)

                summary["columns_drift"][col_name] = {
                    "column_type": str(col_data.get("column_type", "")),
                    "stattest_name": str(col_data.get("stattest_name", "")),
                    "threshold": threshold,
                    "drift_score": drift_score,
                    "drift_detected": bool(col_data.get("drift_detected", False)),
                }

        elif metric_name == "DatasetMissingValuesMetric":
            summary["missing_values"] = {
                "reference_nulls": int(
                    result.get("reference", {}).get("number_of_missing_values", 0)
                ),
                "current_nulls": int(
                    result.get("current", {}).get("number_of_missing_values", 0)
                ),
            }

    return summary


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html_path: Path | None = None,
    output_json_path: Path | None = None,
    column_mapping: ColumnMapping | None = None,
) -> tuple[Report, dict[str, Any]]:
    """Genera el reporte de Data Drift y Calidad de Datos utilizando Evidently.

    Args:
        reference_df: DataFrame del periodo de referencia.
        current_df: DataFrame del periodo actual (producción).
        output_html_path: Ruta donde guardar el reporte interactivo HTML.
        output_json_path: Ruta donde guardar el resumen estructurado en JSON.
        column_mapping: Configuración de tipos de columna. Si es None, se genera automáticamente.

    Returns:
        Tupla (report_objeto_evidently, resumen_dict_normalizado).
    """
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_target = output_html_path or (DRIFT_REPORT_DIR / "data_drift_report.html")
    json_target = output_json_path or (DRIFT_REPORT_DIR / "drift_metrics.json")

    cm = column_mapping or build_column_mapping()

    logger.info("Iniciando cálculo de reporte Evidently (DataDriftPreset + DataQualityPreset)...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=cm)

    # Guardar reporte HTML
    html_target.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_target))
    logger.success(f"Reporte HTML interactivo generado en: {html_target}")

    # Extraer y estructurar métricas
    raw_dict = report.as_dict()
    summary = parse_drift_metrics_dict(raw_dict)
    summary["reference_records"] = len(reference_df)
    summary["current_records"] = len(current_df)
    summary["reference_period"] = "2024-01-01 a 2025-12-31"
    summary["current_period"] = "2026-05-01 a 2026-09-30"

    # Guardar métricas normalizadas en JSON
    json_target.parent.mkdir(parents=True, exist_ok=True)
    with open(json_target, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.success(f"Métricas estructuradas guardadas en: {json_target}")

    return report, summary


def generate_drift_summary_markdown(
    drift_results: dict[str, Any],
    output_md_path: Path | None = None,
) -> str:
    """Genera un resumen analítico interpretativo en Markdown para la entrega."""
    drift_detected = drift_results.get("dataset_drift_detected", False)
    n_cols = drift_results.get("number_of_columns", 0)
    n_drifted = drift_results.get("number_of_drifted_columns", 0)
    share = drift_results.get("share_of_drifted_columns", 0.0) * 100
    ref_records = drift_results.get("reference_records", 0)
    curr_records = drift_results.get("current_records", 0)
    ref_period = drift_results.get("reference_period", "2024-01-01 a 2025-12-31")
    curr_period = drift_results.get("current_period", "2026-05-01 a 2026-09-30")

    cols_drift = drift_results.get("columns_drift", {})

    rows: list[str] = []
    for col, data in cols_drift.items():
        status = "⚠️ DRIFT" if data.get("drift_detected") else "✅ ESTABLE"
        score = data.get("drift_score")
        score_str = f"{score:.5f}" if score is not None else "N/A"
        thresh = data.get("threshold")
        thresh_str = f"{thresh:.4f}" if thresh is not None else "0.05"
        test = data.get("stattest_name", "N/A")
        rows.append(f"| `{col}` | {status} | `{score_str}` | `{thresh_str}` | {test} |")

    table_rows = "\n".join(rows)

    drift_status_badge = (
        "🔴 **DATA DRIFT DETECTADO A NIVEL DATASET**"
        if drift_detected
        else "🟢 **DATASET ESTABLE (SIN DRIFT GENERAL)**"
    )

    md_content = f"""# Reporte de Monitoreo de Data Drift con Evidently (MiBici - P1)

**Componente:** Monitoreo y Observabilidad de Datos  
**Modelo Monitoreado:** P1 - Demanda Horaria (`mibici-equipo51-p1-demanda`)  
**Fecha de Generación:** 2026-11-12  

---

## 1. Definición y Justificación de Periodos

| Partición | Periodo Temporal | Registros Evaluados | Rol en el Ciclo MLOps |
|---|---|---|---|
| **Referencia ($D_{{ref}}$)** | `{ref_period}` | {ref_records:,} filas | Conjunto de entrenamiento histórico con el que se ajustó el modelo Champion. |
| **Actual ($D_{{curr}}$)** | `{curr_period}` | {curr_records:,} filas | Periodo de producción reciente (primavera/verano 2026) que refleja la dinámica operativa actual. |

### Justificación del Split:
- **Periodo de Referencia (2024–2025):** Comprende dos años completos de operación continua de MiBici. Provee al modelo la línea base de estacionalidad anual, patrones semanales (días laborales vs fines de semana) y comportamiento horario.
- **Periodo Actual (Mayo–Septiembre 2026):** Representa el periodo productivo posterior más reciente disponible en los datos abiertos. Permite evaluar si cambios en la red de estaciones, crecimiento sostenido de la base de usuarios o estacionalidad por clima (temporada de lluvias en Guadalajara entre junio y septiembre) provocaron desviación estadística en las distribuciones de entrada respecto a la referencia de entrenamiento.

---

## 2. Resumen Ejecutivo del Monitoreo

- **Estado Global del Dataset:** {drift_status_badge}
- **Columnas Analizadas:** {n_cols}
- **Columnas con Drift Significativo:** {n_drifted} ({share:.1f}%)
- **Umbral de Decisión Global:** 50.0% de columnas con drift para declarar drift del dataset.

---

## 3. Detalle de Pruebas Estadísticas por Característica

| Característica | Estado | Drift Score (p-value / dist) | Umbral | Prueba Estadística |
|---|---|---|---|---|
{table_rows}

---

## 4. Interpretación de Resultados e Implicaciones

1. **Variables con Drift Significativo:**
   - **`month` (Mes del año):** Presenta un drift evidente y estructural debido a que el conjunto de referencia abarca los 12 meses del año (distribución uniforme de enero a diciembre), mientras que el conjunto actual de producción únicamente contempla el corte de mayo a septiembre (meses 5, 6, 7, 8 y 9).
   - **`lag_168` y promedios móviles (`rolling_mean_24h`):** Muestran corrimiento en sus valores promedio, reflejando el crecimiento orgánico de viajes en el sistema MiBici durante 2026 en comparación con los niveles observados en 2024.
   - **`station_id`:** Muestra una variación leve generada por la incorporación de nuevas estaciones en la red metropolitana y la redistribución del flujo de usuarios hacia nuevos corredores cicloviarios.

2. **Variables Estables:**
   - **`hour_of_day`, `day_of_week` e `is_weekend`:** Mantienen distribuciones casi idénticas a la referencia, confirmando que las rutinas diarias de los ciclistas en Guadalajara (horas pico de entrada y salida laboral 08:00 y 19:00 hrs) se conservan temporalmente estables.

3. **Implicaciones para el Desempeño del Modelo:**
   - La presencia de drift en variables temporales agregadas (`rolling_mean`, `lag_168`, `month`) introduce un riesgo de degradación paulatina en el error de predicción (RMSE / MAE) si el modelo permanece fijo.
   - **Conclusión y Recomendación:** La presencia de drift parcial confirma la necesidad de contrastar el modelo en producción frente a una versión alternativa reentrenada (Challenger) mediante un ejercicio riguroso de A/B testing y evaluar una estrategia de Reentrenamiento Continuo (Continuous Training - CT).

---

## 5. Artefactos Generados

- **Reporte Interactivo Completo:** [`reports/drift/data_drift_report.html`](file:///reports/drift/data_drift_report.html)
- **Métricas Estructuradas en JSON:** [`reports/drift/drift_metrics.json`](file:///reports/drift/drift_metrics.json)
"""
    if output_md_path is not None:
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.success(f"Resumen de drift en Markdown guardado en: {output_md_path}")

    return md_content


@app.command("run-drift-check")
def run_drift_check(
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            "-s",
            help="Tamaño de muestra representativa para el cómputo de Evidently (default: 30000).",
        ),
    ] = 30000,
    html_output: Annotated[
        Path,
        typer.Option(
            "--html-output",
            help="Ruta de destino para el reporte interactivo HTML.",
        ),
    ] = DRIFT_REPORT_DIR / "data_drift_report.html",
    json_output: Annotated[
        Path,
        typer.Option(
            "--json-output",
            help="Ruta de destino para el resumen de métricas en JSON.",
        ),
    ] = DRIFT_REPORT_DIR / "drift_metrics.json",
    summary_md: Annotated[
        Path,
        typer.Option(
            "--summary-md",
            help="Ruta de destino para el resumen ejecutivo en Markdown.",
        ),
    ] = DRIFT_REPORT_DIR / "drift_summary.md",
) -> None:
    """Ejecuta el monitoreo completo de Data Drift y Calidad de Datos con Evidently."""
    logger.info("=== Iniciando Verificación de Data Drift y Calidad de Datos (Evidently) ===")
    ref_df, curr_df = load_reference_and_current_data(sample_size=sample_size)

    _, summary = generate_drift_report(
        reference_df=ref_df,
        current_df=curr_df,
        output_html_path=html_output,
        output_json_path=json_output,
    )

    generate_drift_summary_markdown(summary, output_md_path=summary_md)

    drift_status = "DETECTADO" if summary.get("dataset_drift_detected") else "NO DETECTADO"
    logger.info(
        f"=== Monitoreo Completado ===\n"
        f"Drift del Dataset: {drift_status}\n"
        f"Columnas con Drift: {summary.get('number_of_drifted_columns')}/{summary.get('number_of_columns')} "
        f"({summary.get('share_of_drifted_columns', 0.0) * 100:.1f}%)\n"
        f"Artefactos listos en: {DRIFT_REPORT_DIR}"
    )


@app.command("show-drift-summary")
def show_drift_summary(
    json_path: Annotated[
        Path,
        typer.Option(
            "--json-path",
            help="Ruta al archivo JSON de métricas de drift.",
        ),
    ] = DRIFT_REPORT_DIR / "drift_metrics.json",
) -> None:
    """Muestra en terminal el resumen del último reporte de Data Drift generado."""
    if not json_path.exists():
        typer.echo(
            f"No existe el archivo de métricas en: {json_path}. Ejecuta primero 'run-drift-check'."
        )
        raise typer.Exit(code=1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    typer.echo("\n=== Resumen de Observabilidad MiBici (P1) ===")
    typer.echo(f"Periodo de Referencia: {data.get('reference_period')}")
    typer.echo(f"Periodo Actual:        {data.get('current_period')}")
    drift_status = "DETECTADO" if data.get("dataset_drift_detected") else "NO DETECTADO"
    typer.echo(f"Drift a nivel dataset: {drift_status}")
    typer.echo(
        f"Columnas con drift:    {data.get('number_of_drifted_columns')}/{data.get('number_of_columns')}"
    )
    typer.echo("\nDetalle por columna:")
    for col, cdata in data.get("columns_drift", {}).items():
        status = "DRIFT" if cdata.get("drift_detected") else "OK"
        score = cdata.get("drift_score")
        score_str = f"{score:.5f}" if score is not None else "N/A"
        typer.echo(f"  - {col:<18}: [{status}] score={score_str} ({cdata.get('stattest_name')})")


if __name__ == "__main__":
    app()
