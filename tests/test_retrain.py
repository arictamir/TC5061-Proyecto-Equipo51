"""Pruebas unitarias para el módulo de reentrenamiento y registro de la versión alternativa."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mibici_mlops.config import PROCESSED_DATA_DIR
from mibici_mlops.modeling.retrain import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    app,
    load_retraining_data,
    train_and_register_challenger,
)

runner = CliRunner()


def test_load_retraining_data_missing_file(tmp_path: Path) -> None:
    """Verifica que lance FileNotFoundError si no existe el archivo de características."""
    fake_path = tmp_path / "non_existent.parquet"
    with pytest.raises(FileNotFoundError):
        load_retraining_data(features_path=fake_path)


def test_load_retraining_data_structure() -> None:
    """Verifica que los datos cargados tengan las columnas y tipos requeridos para CatBoost."""
    parquet_path = PROCESSED_DATA_DIR / "p1_features.parquet"
    if not parquet_path.exists():
        pytest.skip("p1_features.parquet no disponible para la prueba.")

    train_df, eval_df, feat_cols, cat_cols, target = load_retraining_data(
        features_path=parquet_path,
        include_recent_history=True,
        max_train_samples=50,
        max_eval_samples=50,
        random_state=42,
    )

    assert len(train_df) == 50
    assert len(eval_df) == 50
    assert feat_cols == FEATURE_COLUMNS
    assert cat_cols == CATEGORICAL_COLUMNS
    assert target == TARGET_COLUMN

    for col in cat_cols:
        assert train_df[col].dtype == object or str(train_df[col].dtype) == "string"
        assert eval_df[col].dtype == object or str(eval_df[col].dtype) == "string"

    assert train_df.isna().sum().sum() == 0
    assert eval_df.isna().sum().sum() == 0


def test_train_challenger_fast_execution() -> None:
    """Verifica la ejecución rápida de entrenamiento y cálculo de métricas."""
    parquet_path = PROCESSED_DATA_DIR / "p1_features.parquet"
    if not parquet_path.exists():
        pytest.skip("p1_features.parquet no disponible para la prueba.")

    resultado = train_and_register_challenger(
        depth=3,
        learning_rate=0.1,
        iterations=5,
        l2_leaf_reg=1.0,
        max_train_samples=60,
        max_eval_samples=30,
        include_recent_history=True,
        register_in_registry=False,
    )

    assert "run_id" in resultado
    assert "metrics" in resultado
    assert "params" in resultado

    metrics = resultado["metrics"]
    assert "prueba_mae" in metrics
    assert "prueba_rmse" in metrics
    assert "prueba_r2" in metrics
    assert "latencia_ms_por_1k" in metrics
    assert metrics["prueba_mae"] >= 0.0
    assert metrics["prueba_rmse"] >= 0.0


def test_cli_retrain_help() -> None:
    """Verifica que el comando CLI muestre la ayuda sin errores."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "retrain-p1-challenger" in result.stdout
    assert "show-registry-status" in result.stdout


def test_cli_show_registry_status() -> None:
    """Verifica que la consulta del estado del registro en MLflow funcione correctamente."""
    result = runner.invoke(app, ["show-registry-status"])
    assert result.exit_code == 0
    assert "Estado de Model Registry" in result.stdout
    assert "mibici-equipo51-p1-demanda" in result.stdout
    assert "@champion" in result.stdout
    assert "@challenger" in result.stdout
