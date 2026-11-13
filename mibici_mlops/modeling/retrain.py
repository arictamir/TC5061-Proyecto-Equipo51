"""Módulo para el entrenamiento y registro de la versión alternativa (Challenger) de P1.

Permite generar una nueva versión del modelo de demanda horaria con CatBoost,
incorporando hiperparámetros optimizados y una ventana de entrenamiento adaptada
tras la detección de data drift en las distribuciones de entrada, y registrarla
en el Model Registry de MLflow con el alias @challenger.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Annotated, Any

from catboost import CatBoostRegressor
from loguru import logger
import mlflow
import mlflow.catboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import typer

from mibici_mlops.config import (
    MODEL_REGISTRY_P1,
    P1_TEST_START,
    P1_VALIDATION_START,
    PROCESSED_DATA_DIR,
    SEMILLA_ALEATORIA,
    TAG_QUESTION_P1,
    TAG_STAGE_CANDIDATE,
)
from mibici_mlops.modeling.train import (
    asignar_model_registry_con_alias,
    configurar_mlflow,
    crear_etiquetas_mlflow,
    obtener_commit_git,
)

app = typer.Typer(
    help="Entrenamiento y registro de versión alternativa Challenger para MiBici P1."
)

FEATURE_COLUMNS: list[str] = [
    "station_id",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "history_hours",
]

CATEGORICAL_COLUMNS: list[str] = [
    "station_id",
    "hour_of_day",
    "day_of_week",
    "month",
]

TARGET_COLUMN: str = "trips_started"


def load_retraining_data(
    features_path: Path | None = None,
    include_recent_history: bool = True,
    max_train_samples: int | None = 200000,
    max_eval_samples: int | None = 50000,
    random_state: int = SEMILLA_ALEATORIA,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], str]:
    """Carga y prepara los conjuntos de entrenamiento y evaluación para la versión alternativa.

    - Entrenamiento: Si include_recent_history=True, amplía la ventana hasta antes de P1_TEST_START
      (2026-04-30), permitiendo al modelo aprender de los patrones más recientes observados.
    - Evaluación: Conjunto actual de prueba (a partir de P1_TEST_START: 2026-05 a 2026-09), idéntico
      al utilizado para evaluar la versión Champion y consistente con el periodo actual de drift.
    """
    path = features_path or (PROCESSED_DATA_DIR / "p1_features.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de características en: {path}. "
            "Asegúrate de ejecutar la etapa de ingeniería de características previamente."
        )

    logger.info(f"Cargando características para reentrenamiento desde: {path}")
    cols_to_load = list(set(FEATURE_COLUMNS + [TARGET_COLUMN, "hour"]))
    df = pd.read_parquet(path, columns=cols_to_load)

    # Definir máscaras de entrenamiento y evaluación
    if include_recent_history:
        # Ventana extendida que incluye datos históricos hasta el inicio del periodo actual
        train_mask = df["hour"] < P1_TEST_START
        strategy_desc = "ventana_extendida_pre_test (2024 a 2026-04)"
    else:
        # Ventana estándar previa a validación
        train_mask = df["hour"] < P1_VALIDATION_START
        strategy_desc = "ventana_estandar_historica (2024 a 2025)"

    eval_mask = df["hour"] >= P1_TEST_START

    train_df = df.loc[train_mask, FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    eval_df = df.loc[eval_mask, FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()

    logger.info(
        f"Datos filtrados -> Entrenamiento ({strategy_desc}): {len(train_df):,} filas, "
        f"Evaluación (2026-05 a 2026-09): {len(eval_df):,} filas."
    )

    # Muestreo representativo para entrenamiento ágil y eficiente
    if max_train_samples is not None and len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=random_state).copy()
        logger.info(f"Muestreo de entrenamiento aplicado: {max_train_samples:,} observaciones.")

    if max_eval_samples is not None and len(eval_df) > max_eval_samples:
        eval_df = eval_df.sample(n=max_eval_samples, random_state=random_state).copy()
        logger.info(f"Muestreo de evaluación aplicado: {max_eval_samples:,} observaciones.")

    # Formatear columnas categóricas como cadenas de texto para CatBoost
    for col in CATEGORICAL_COLUMNS:
        train_df[col] = train_df[col].astype(str)
        eval_df[col] = eval_df[col].astype(str)

    return train_df, eval_df, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN


def train_and_register_challenger(
    depth: int = 7,
    learning_rate: float = 0.08,
    iterations: int = 220,
    l2_leaf_reg: float = 4.0,
    max_train_samples: int = 200000,
    max_eval_samples: int = 50000,
    include_recent_history: bool = True,
    register_in_registry: bool = True,
    alias: str = "challenger",
) -> dict[str, Any]:
    """Entrena la versión alternativa Challenger y la registra en el Model Registry de MLflow.

    Args:
        depth: Profundidad de los árboles de decisión en CatBoost.
        learning_rate: Tasa de aprendizaje.
        iterations: Número de árboles/iteraciones de boosting.
        l2_leaf_reg: Coeficiente de regularización L2 para evitar sobreajuste.
        max_train_samples: Muestra máxima de entrenamiento.
        max_eval_samples: Muestra máxima de evaluación en test set.
        include_recent_history: Si True, incluye datos de 2026 previos al periodo de test.
        register_in_registry: Si True, registra la versión en MLflow Model Registry.
        alias: Alias para asignar en el registro (por defecto: 'challenger').

    Returns:
        Diccionario con run_id, métricas evaluadas, parámetros y número de versión en el Registry.
    """
    configurar_mlflow()
    commit_hash = obtener_commit_git()

    train_df, eval_df, feat_cols, cat_cols, target = load_retraining_data(
        include_recent_history=include_recent_history,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    X_train, y_train = train_df[feat_cols], train_df[target]
    X_eval, y_eval = eval_df[feat_cols], eval_df[target]

    run_name = (
        f"P1 - CatBoost - Version Alternativa Challenger (depth={depth}, lr={learning_rate})"
    )

    logger.info(f"Iniciando run de MLflow: '{run_name}'...")
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Etiquetas de gobernanza y trazabilidad
        tags = crear_etiquetas_mlflow(
            question=TAG_QUESTION_P1,
            model_family="catboost",
            stage_of_work=TAG_STAGE_CANDIDATE,
            commit_git=commit_hash,
            rol_modelo="challenger",
            candidate_type="challenger_drift_adapted",
            data_strategy="extended_window" if include_recent_history else "standard_window",
        )
        mlflow.set_tags(tags)

        # Ajuste del modelo CatBoost
        logger.info(
            f"Entrenando CatBoostRegressor (depth={depth}, lr={learning_rate}, "
            f"iter={iterations}, l2={l2_leaf_reg})..."
        )
        t_start = time.time()
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="RMSE",
            random_seed=SEMILLA_ALEATORIA,
            thread_count=-1,
            verbose=0,
        )
        model.fit(X_train, y_train, cat_features=cat_cols)
        fit_time_s = time.time() - t_start
        logger.success(f"Entrenamiento completado en {fit_time_s:.2f} s.")

        # Inferencia y evaluación sobre el conjunto actual (producción / test)
        t_pred = time.time()
        preds = np.clip(model.predict(X_eval), 0, None)
        eval_time_s = time.time() - t_pred
        latency_ms_per_1k = (eval_time_s / len(X_eval)) * 1000.0 * 1000.0

        mae = float(mean_absolute_error(y_eval, preds))
        rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
        r2 = float(r2_score(y_eval, preds))

        pos_mask = y_eval > 0
        mae_positive = (
            float(mean_absolute_error(y_eval[pos_mask], preds[pos_mask]))
            if pos_mask.sum() > 0
            else mae
        )

        logger.info(
            f"Métricas en periodo actual ($D_{{curr}}$):\n"
            f"  - RMSE: {rmse:.4f}\n"
            f"  - MAE:  {mae:.4f}\n"
            f"  - R²:   {r2:.4f}\n"
            f"  - MAE en demanda positiva: {mae_positive:.4f}\n"
            f"  - Latencia aprox por 1k peticiones: {latency_ms_per_1k:.2f} ms"
        )

        # Registro de hiperparámetros
        params_dict = {
            "modelo": "catboost_regressor",
            "depth": depth,
            "learning_rate": learning_rate,
            "iterations": iterations,
            "l2_leaf_reg": l2_leaf_reg,
            "train_samples": len(train_df),
            "eval_samples": len(eval_df),
            "include_recent_history": include_recent_history,
            "random_seed": SEMILLA_ALEATORIA,
        }
        mlflow.log_params(params_dict)

        # Registro de métricas
        metrics_dict = {
            "prueba_mae": mae,
            "prueba_rmse": rmse,
            "prueba_r2": r2,
            "prueba_mae_positiva": mae_positive,
            "tiempo_entrenamiento_s": fit_time_s,
            "latencia_ms_por_1k": latency_ms_per_1k,
        }
        mlflow.log_metrics(metrics_dict)

        # Firma e inferencia de ejemplo para empaquetado del modelo
        input_example = X_eval.head(5)
        signature = infer_signature(input_example, model.predict(input_example))

        logger.info("Registrando artefacto del modelo en MLflow...")
        mlflow.catboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    # Registro en Model Registry y asignación de alias
    registered_version: int | None = None
    if register_in_registry:
        logger.info(f"Registrando versión en Model Registry '{MODEL_REGISTRY_P1}'...")
        version_desc = (
            f"Challenger: CatBoost depth={depth}, lr={learning_rate}, iter={iterations}, "
            f"l2={l2_leaf_reg}. Adaptado ante data drift con ventana extendida. Commit: {commit_hash}"
        )
        model_desc = (
            "Modelo de regresión para predicción de demanda horaria por estación en MiBici."
        )

        client = MlflowClient()
        asignar_model_registry_con_alias(
            run_id=run_id,
            artifact_path="model",
            model_name=MODEL_REGISTRY_P1,
            alias=alias,
            model_description=model_desc,
            version_description=version_desc,
        )

        # Obtener el número de versión asignado
        try:
            alias_version_info = client.get_model_version_by_alias(MODEL_REGISTRY_P1, alias)
            registered_version = int(alias_version_info.version)
            logger.success(
                f"Model Registry actualizado con éxito: '{MODEL_REGISTRY_P1}' v{registered_version} "
                f"asignado con alias @{alias}."
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"No se pudo consultar la versión recién registrada con alias @{alias}: {e}")

    return {
        "run_id": run_id,
        "registered_version": registered_version,
        "alias": alias,
        "model_name": MODEL_REGISTRY_P1,
        "metrics": metrics_dict,
        "params": params_dict,
    }


@app.command("retrain-p1-challenger")
def retrain_p1_challenger_cli(
    depth: Annotated[
        int,
        typer.Option("--depth", "-d", help="Profundidad de árboles CatBoost."),
    ] = 7,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", "-lr", help="Tasa de aprendizaje."),
    ] = 0.08,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-i", help="Número de iteraciones / árboles."),
    ] = 220,
    l2_leaf_reg: Annotated[
        float,
        typer.Option("--l2-leaf-reg", "-l2", help="Regularización L2."),
    ] = 4.0,
    max_train_samples: Annotated[
        int,
        typer.Option("--max-train-samples", help="Número de muestras de entrenamiento."),
    ] = 200000,
    max_eval_samples: Annotated[
        int,
        typer.Option("--max-eval-samples", help="Número de muestras de evaluación."),
    ] = 50000,
    alias: Annotated[
        str,
        typer.Option("--alias", help="Alias para asignar en Model Registry."),
    ] = "challenger",
) -> None:
    """Ejecuta el entrenamiento de la versión alternativa Challenger y la registra en MLflow."""
    logger.info("=== Iniciando Entrenamiento de Versión Alternativa Challenger ===")
    resultado = train_and_register_challenger(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        register_in_registry=True,
        alias=alias,
    )

    typer.echo("\n=== Resultado del Reentrenamiento Challenger ===")
    typer.echo(f"Run ID:            {resultado.get('run_id')}")
    typer.echo(f"Modelo:            {resultado.get('model_name')}")
    typer.echo(f"Versión asignada:  v{resultado.get('registered_version')}")
    typer.echo(f"Alias asignado:    @{resultado.get('alias')}")
    typer.echo("Métricas obtenidas en periodo actual ($D_{curr}$):")
    for k, v in resultado.get("metrics", {}).items():
        typer.echo(f"  - {k:<25}: {v:.4f}")


@app.command("show-registry-status")
def show_registry_status(
    model_name: Annotated[
        str,
        typer.Option("--model-name", help="Nombre del modelo registrado en MLflow."),
    ] = MODEL_REGISTRY_P1,
) -> None:
    """Muestra el estado actual de las versiones registradas y sus alias en MLflow."""
    configurar_mlflow()
    client = MlflowClient()

    try:
        model = client.get_registered_model(model_name)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Error al consultar el modelo '{model_name}': {e}")
        raise typer.Exit(code=1)

    typer.echo(f"\n=== Estado de Model Registry: '{model.name}' ===")
    typer.echo(f"Descripción: {model.description}")
    typer.echo("Aliases configurados:")
    for alias_name, ver in model.aliases.items():
        typer.echo(f"  - @{alias_name:<12} -> Versión {ver}")

    typer.echo("\nVersiones disponibles:")
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in sorted(versions, key=lambda x: int(x.version)):
        typer.echo(
            f"  - v{v.version} (run_id: {v.run_id[:8]}...): status={v.status} | {v.description}"
        )


if __name__ == "__main__":
    app()
