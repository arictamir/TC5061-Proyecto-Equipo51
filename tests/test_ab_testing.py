"""Pruebas unitarias para el módulo de evaluación y A/B Testing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici_mlops.config import PROCESSED_DATA_DIR
from mibici_mlops.modeling.ab_testing import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    app,
    compute_model_metrics,
    compute_statistical_test,
    generate_ab_testing_plots,
    load_ab_evaluation_data,
)

runner = CliRunner()


def test_load_ab_evaluation_data_missing_file(tmp_path: Path) -> None:
    """Verifica que lance FileNotFoundError si no existe el archivo de características."""
    fake_path = tmp_path / "non_existent.parquet"
    with pytest.raises(FileNotFoundError):
        load_ab_evaluation_data(features_path=fake_path)


def test_load_ab_evaluation_data_structure() -> None:
    """Verifica que los datos cargados para A/B testing tengan la estructura esperada."""
    parquet_path = PROCESSED_DATA_DIR / "p1_features.parquet"
    if not parquet_path.exists():
        pytest.skip("p1_features.parquet no disponible para la prueba.")

    X_eval, y_eval, full_eval_df = load_ab_evaluation_data(
        features_path=parquet_path,
        sample_size=60,
        random_state=42,
    )

    assert len(X_eval) == 60
    assert len(y_eval) == 60
    assert len(full_eval_df) == 60
    assert list(X_eval.columns) == FEATURE_COLUMNS
    for col in CATEGORICAL_COLUMNS:
        assert full_eval_df[col].dtype == object or str(full_eval_df[col].dtype) == "string"


def test_compute_model_metrics() -> None:
    """Verifica el cálculo de métricas de precisión y latencia."""
    y_true = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([0.1, 1.2, 1.8, 3.1, 3.9, 5.2], dtype=float)
    metrics = compute_model_metrics(y_true, y_pred, inference_time_s=0.01)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mae_positive" in metrics
    assert "latency_ms_per_1k" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["r2"] > 0.9


def test_compute_statistical_test() -> None:
    """Verifica la prueba t pareada de significancia estadística."""
    np.random.seed(42)
    err_champ = np.random.normal(loc=1.0, scale=0.2, size=100)
    err_chall = np.random.normal(loc=1.5, scale=0.2, size=100)

    stat = compute_statistical_test(err_champ, err_chall)
    assert "paired_t_statistic" in stat
    assert "p_value" in stat
    assert "is_statistically_significant" in stat
    assert stat["is_statistically_significant"] is True
    assert stat["mean_absolute_error_difference"] < 0


def test_generate_ab_testing_plots(tmp_path: Path) -> None:
    """Verifica que se generen los 3 gráficos comparativos de A/B testing."""
    champ_metrics = {
        "rmse": 1.42,
        "mae": 0.78,
        "mae_positive": 1.32,
        "residual_mean": 0.05,
        "residual_std": 1.41,
    }
    chall_metrics = {
        "rmse": 1.43,
        "mae": 0.81,
        "mae_positive": 1.30,
        "residual_mean": 0.06,
        "residual_std": 1.42,
    }
    y_true = np.array([1, 2, 3, 4, 5], dtype=float)
    champ_preds = np.array([1.1, 1.9, 3.2, 3.8, 5.1], dtype=float)
    chall_preds = np.array([1.2, 2.1, 3.0, 3.9, 4.9], dtype=float)

    eval_df = pd.DataFrame(
        {
            "hour_of_day": [8, 9, 10, 11, 12],
        }
    )

    plots = generate_ab_testing_plots(
        champ_metrics=champ_metrics,
        chall_metrics=chall_metrics,
        y_true=y_true,
        champ_preds=champ_preds,
        chall_preds=chall_preds,
        full_eval_df=eval_df,
        output_dir=tmp_path,
    )

    assert len(plots) == 3
    for p in plots:
        assert p.exists()
        assert p.stat().st_size > 1000


def test_cli_ab_help() -> None:
    """Verifica que el comando CLI muestre la ayuda sin errores."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-ab-test" in result.stdout
    assert "show-ab-summary" in result.stdout


def test_cli_show_ab_summary(tmp_path: Path) -> None:
    """Verifica que el comando show-ab-summary formatee el resultado adecuadamente."""
    test_json = tmp_path / "test_ab.json"
    data = {
        "model_name": "mibici-equipo51-p1-demanda",
        "evaluation_period": "2026-05 a 2026-09",
        "champion_info": {"version": 3, "metrics": {"rmse": 1.42, "mae": 0.78}},
        "challenger_info": {"version": 5, "metrics": {"rmse": 1.43, "mae": 0.81}},
        "decision": {
            "active_version_number": 3,
            "decision_justification": "Champion mantiene menor error global.",
        },
    }
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    result = runner.invoke(app, ["show-ab-summary", "--json-path", str(test_json)])
    assert result.exit_code == 0
    assert "Resumen de A/B Testing" in result.stdout
    assert "Versión 3" in result.stdout

