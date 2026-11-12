"""Pruebas unitarias para el módulo de monitoreo de data drift y observabilidad."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici_mlops.config import PROCESSED_DATA_DIR
from mibici_mlops.monitoring import (
    ALL_MONITORED_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    app,
    build_column_mapping,
    generate_drift_report,
    generate_drift_summary_markdown,
    load_reference_and_current_data,
)

runner = CliRunner()


def test_build_column_mapping() -> None:
    """Verifica que el mapeo de columnas contenga las variables correctas."""
    cm = build_column_mapping()
    assert cm.target == TARGET_COLUMN
    assert cm.numerical_features == NUMERICAL_FEATURES
    assert cm.categorical_features == CATEGORICAL_FEATURES


def test_load_reference_and_current_data_missing_file(tmp_path: Path) -> None:
    """Verifica que lance FileNotFoundError si no existe el parquet."""
    fake_path = tmp_path / "non_existent.parquet"
    with pytest.raises(FileNotFoundError):
        load_reference_and_current_data(features_path=fake_path)


def test_load_reference_and_current_data_sampling() -> None:
    """Verifica la carga y partición de datos con muestreo representativo."""
    parquet_path = PROCESSED_DATA_DIR / "p1_features.parquet"
    if not parquet_path.exists():
        pytest.skip("p1_features.parquet no disponible para la prueba.")

    sample_size = 50
    ref_df, curr_df = load_reference_and_current_data(
        features_path=parquet_path,
        sample_size=sample_size,
        random_state=42,
    )

    assert len(ref_df) == sample_size
    assert len(curr_df) == sample_size
    assert set(ref_df.columns) == set(ALL_MONITORED_COLUMNS)
    assert set(curr_df.columns) == set(ALL_MONITORED_COLUMNS)
    assert ref_df.isna().sum().sum() == 0
    assert curr_df.isna().sum().sum() == 0


def test_generate_drift_report_synthetic(tmp_path: Path) -> None:
    """Verifica la generación de reportes HTML y JSON con datos sintéticos rápidos."""
    n = 60
    np.random.seed(42)

    # Crear datos de referencia
    ref_data = {col: np.random.randn(n) for col in NUMERICAL_FEATURES}
    for col in CATEGORICAL_FEATURES:
        ref_data[col] = np.random.randint(1, 10, size=n)
    ref_data[TARGET_COLUMN] = np.random.randint(0, 5, size=n)
    ref_df = pd.DataFrame(ref_data)

    # Crear datos actuales con desplazamiento para forzar drift en alguna columna
    curr_data = {col: np.random.randn(n) for col in NUMERICAL_FEATURES}
    curr_data["rolling_mean_24h"] = np.random.randn(n) + 15.0  # Drift claro
    for col in CATEGORICAL_FEATURES:
        curr_data[col] = np.random.randint(1, 10, size=n)
    curr_data["month"] = np.random.randint(5, 10, size=n)
    curr_data[TARGET_COLUMN] = np.random.randint(0, 5, size=n)
    curr_df = pd.DataFrame(curr_data)

    html_file = tmp_path / "test_report.html"
    json_file = tmp_path / "test_report.json"

    _report, summary = generate_drift_report(
        reference_df=ref_df,
        current_df=curr_df,
        output_html_path=html_file,
        output_json_path=json_file,
    )

    assert html_file.exists()
    assert html_file.stat().st_size > 1000  # Reporte HTML con contenido
    assert json_file.exists()

    with open(json_file, encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert "number_of_columns" in loaded_json
    assert loaded_json["number_of_columns"] == len(ALL_MONITORED_COLUMNS)
    assert "columns_drift" in loaded_json
    assert "rolling_mean_24h" in loaded_json["columns_drift"]
    assert isinstance(summary["dataset_drift_detected"], bool)


def test_generate_drift_summary_markdown(tmp_path: Path) -> None:
    """Verifica que el resumen ejecutivo en markdown se genere con la estructura correcta."""
    sample_summary = {
        "dataset_drift_detected": False,
        "number_of_columns": 12,
        "number_of_drifted_columns": 4,
        "share_of_drifted_columns": 0.333,
        "reference_records": 30000,
        "current_records": 30000,
        "reference_period": "2024-01-01 a 2025-12-31",
        "current_period": "2026-05-01 a 2026-09-30",
        "columns_drift": {
            "month": {
                "drift_detected": True,
                "drift_score": 0.511,
                "threshold": 0.1,
                "stattest_name": "Jensen-Shannon",
            },
            "hour_of_day": {
                "drift_detected": False,
                "drift_score": 0.012,
                "threshold": 0.1,
                "stattest_name": "Jensen-Shannon",
            },
        },
    }

    out_md = tmp_path / "summary.md"
    content = generate_drift_summary_markdown(sample_summary, output_md_path=out_md)

    assert out_md.exists()
    assert "# Reporte de Monitoreo de Data Drift con Evidently" in content
    assert "Monitoreo y Observabilidad de Datos" in content
    assert "2024-01-01 a 2025-12-31" in content
    assert "2026-05-01 a 2026-09-30" in content
    assert "`month`" in content
    assert "⚠️ DRIFT" in content
    assert "✅ ESTABLE" in content


def test_cli_monitoring_help() -> None:
    """Verifica que el comando CLI muestre la ayuda sin errores."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-drift-check" in result.stdout
    assert "show-drift-summary" in result.stdout


def test_cli_show_drift_summary_not_found(tmp_path: Path) -> None:
    """Verifica el manejo de error si el JSON no existe."""
    fake_json = tmp_path / "missing.json"
    result = runner.invoke(app, ["show-drift-summary", "--json-path", str(fake_json)])
    assert result.exit_code != 0
    assert "No existe el archivo de métricas" in result.stdout


def test_cli_show_drift_summary_success(tmp_path: Path) -> None:
    """Verifica que show-drift-summary formatee correctamente el resumen."""
    fake_json = tmp_path / "metrics.json"
    data = {
        "reference_period": "2024 a 2025",
        "current_period": "2026-05 a 2026-09",
        "dataset_drift_detected": False,
        "number_of_columns": 12,
        "number_of_drifted_columns": 2,
        "columns_drift": {
            "month": {
                "drift_detected": True,
                "drift_score": 0.55,
                "stattest_name": "JS",
            }
        },
    }
    with open(fake_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    result = runner.invoke(app, ["show-drift-summary", "--json-path", str(fake_json)])
    assert result.exit_code == 0
    assert "Resumen de Observabilidad MiBici" in result.stdout
    assert "month" in result.stdout

