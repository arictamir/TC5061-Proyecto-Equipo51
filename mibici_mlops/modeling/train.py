"""Entrenamiento, experimentación avanzada y registro en MLflow para P1, P2 y P3.

Soporta:
- Baselines determinísticos y estadísticos
- Búsqueda de hiperparámetros con Nested Runs (Parent/Child)
- Modelos candidatos con registro completo de artefactos (plots/, data/, config/)
- Registro formal de datasets con mlflow.log_input
- Firmas inferidas e input examples
- Model Registry con gestión de ciclo de vida mediante Aliases (@champion / @challenger)
- Orquestación CLI compatible con MLproject.
"""

from enum import Enum
import json
from math import prod
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Annotated, Any

from catboost import CatBoostRegressor
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skops.io as sio
import typer

from mibici_mlops.config import (
    FIGURES_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_P1,
    MODEL_REGISTRY_P2,
    MODEL_REGISTRY_P3,
    MODELS_DIR,
    NOMBRE_EXPERIMENTO,
    NOMENCLATURE_FILE,
    P1_DATA_END,
    P1_DATA_START,
    P1_TEST_START,
    P1_VALIDATION_START,
    P2_CUANTIL_SATURACION,
    P2_CUANTIL_VACIADO,
    P2_DIR,
    P2_FIGURAS_DIR,
    P2_FIN_ENTRENAMIENTO,
    P2_FIN_PRUEBA,
    P2_FIN_VALIDACION,
    P2_HORIZONTE_HORAS,
    P2_INICIO_ENTRENAMIENTO,
    P2_INICIO_PRUEBA,
    P2_INICIO_VALIDACION,
    P2_UMBRAL_SATURACION,
    P2_UMBRAL_VACIADO,
    P3_DEFAULT_K,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    SEMILLA_ALEATORIA,
    TAG_QUESTION_P1,
    TAG_QUESTION_P2,
    TAG_QUESTION_P3,
    TAG_STAGE_BASELINE,
    TAG_STAGE_CANDIDATE,
    TAG_STAGE_SEARCH,
    TAG_TEAM,
)
from mibici_mlops.features import (
    COLUMNAS_CARACTERISTICAS_P2,
    build_p1_features,
    build_p1_hourly_demand,
    generate_p3_station_features,
    preparar_dataset_p2,
)
from mibici_mlops.plots import (
    plot_p1_real_vs_pred,
    plot_p1_residuals,
    plot_p2_confusion_matrix,
    plot_p2_roc_curves,
    plot_p3_cluster_distribution,
    plot_p3_spatial_clusters,
)
from mibici_mlops.quality import run_feature_drift_check

plt.switch_backend("Agg")

app = typer.Typer(help="Pipeline de entrenamiento y tracking avanzado para MiBici.")

RUTA_MANIFIESTO = PROJ_ROOT / "data_manifest.json"
COLUMNA_OBJETIVO = "clase_riesgo"
COLUMNA_CONJUNTO = "conjunto"
ORDEN_CLASES = ["riesgo_vaciado", "normal", "riesgo_saturacion"]


class NombreModelo(str, Enum):
    DUMMY = "dummy_stratified"
    LOGISTICA = "logistic_regression"


class SolverRegresion(str, Enum):
    SAGA = "saga"
    LBFGS = "lbfgs"


ETIQUETA_EQUIPO = "51"
CONJUNTOS_DESARROLLO = ["entrenamiento", "validacion"]
CONJUNTOS_FINALES = ["entrenamiento", "validacion", "prueba"]
NOMBRE_RUN_DUMMY = "P2 - Dummy estratificado - desarrollo"
NOMBRE_RUN_LOGISTICA = "P2 - Regresión logística - desarrollo"
NOMBRE_RUN_FINAL = "P2 - Regresión logística - evaluación final"


# =============================================================================
# UTILIDADES COMUNES DE TRAZABILIDAD Y GOBERNANZA
# =============================================================================


def obtener_commit_git() -> str:
    """Obtiene el hash del commit actual de Git."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return resultado.stdout.strip()
    except Exception:  # noqa: BLE001
        return "sin_commit_local"


def cargar_version_datos(ruta_manifiesto: Path = RUTA_MANIFIESTO) -> str:
    """Lee la versión SHA-256 reproducible de los datos."""
    if not ruta_manifiesto.exists():
        return "desconocida"
    try:
        manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
        return str(manifiesto.get("dataset_version", "desconocida"))
    except Exception:  # noqa: BLE001
        return "desconocida"


def configurar_mlflow() -> str:
    """Configura el backend store y asegura el experimento unificado."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experimento = client.get_experiment_by_name(NOMBRE_EXPERIMENTO)
    if experimento is None:
        exp_id = client.create_experiment(NOMBRE_EXPERIMENTO)
        logger.info(f"Experimento '{NOMBRE_EXPERIMENTO}' creado con ID: {exp_id}")
    else:
        exp_id = experimento.experiment_id
    mlflow.set_experiment(NOMBRE_EXPERIMENTO)
    return exp_id


def crear_etiquetas_mlflow(
    question: str | None = None,
    model_family: str | None = None,
    stage_of_work: str | None = None,
    commit_git: str = "sin_commit",
    rol_modelo: str = "candidato",
    pregunta: str | None = None,
    tipo_tarea: str | None = None,
    etapa: str | None = None,
    **extra_tags: Any,
) -> dict[str, str]:
    """Crea esquema estricto de etiquetas requerido por la rúbrica y soporta tests legacy."""
    if pregunta is not None and question is None:
        return {
            "equipo": ETIQUETA_EQUIPO,
            "pregunta": pregunta,
            "tipo_tarea": tipo_tarea or "clasificacion",
            "etapa": etapa or "desarrollo",
            "rol_modelo": rol_modelo,
            "git_commit": commit_git,
        }
    q = (question or "p1").lower()
    tags = {
        "question": q,
        "model_family": model_family or "desconocido",
        "stage_of_work": stage_of_work or "candidate",
        "team": TAG_TEAM,
        "git_commit": commit_git,
        "equipo": ETIQUETA_EQUIPO,
        "pregunta": (pregunta or q).upper(),
        "tipo_tarea": tipo_tarea or "modelado",
        "etapa": etapa or (stage_of_work or "candidate"),
        "rol_modelo": rol_modelo,
    }
    for k, v in extra_tags.items():
        tags[k] = str(v)
    return tags


def registrar_artefactos_en_subcarpetas(
    config_dict: dict[str, Any],
    plots_dict: dict[str, Path] | None = None,
    dataset_df: pd.DataFrame | None = None,
    dataset_target: str | None = None,
    dataset_name: str | None = None,
) -> None:
    """Registra artefactos organizados en subcarpetas: config/, data/ y plots/."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Subcarpeta config/
        config_file = temp_path / "run_config.json"
        with config_file.open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)
        mlflow.log_artifact(str(config_file), artifact_path="config")

        # 2. Subcarpeta data/
        if RUTA_MANIFIESTO.exists():
            mlflow.log_artifact(str(RUTA_MANIFIESTO), artifact_path="data")

        # Registro formal del dataset en MLflow
        if dataset_df is not None and dataset_name is not None:
            try:
                sample_size = min(len(dataset_df), 1000)
                df_sample = dataset_df.head(sample_size).copy()
                # Limpiar columnas temporales que puedan dar problema de serialización
                for col in df_sample.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_sample[col]):
                        df_sample[col] = df_sample[col].astype(str)
                mlflow_dataset = mlflow.data.from_pandas(
                    df_sample,
                    source=str(RUTA_MANIFIESTO),
                    targets=dataset_target,
                    name=dataset_name,
                )
                mlflow.log_input(mlflow_dataset, context="training")
            except Exception as error:  # noqa: BLE001
                logger.warning(f"No se pudo registrar mlflow.log_input: {error}")

        # 3. Subcarpeta plots/
        if plots_dict:
            for plot_path in plots_dict.values():
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")


def asignar_model_registry_con_alias(
    run_id: str,
    artifact_path: str,
    model_name: str,
    alias: str,  # "champion" o "challenger"
    model_description: str,
    version_description: str,
) -> None:
    """Registra el modelo en Model Registry y asigna el alias correspondiente."""
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        model_version = mlflow.register_model(model_uri, model_name)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error al registrar {model_name}: {e}")
        return

    try:
        client.update_registered_model(name=model_name, description=model_description)
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=version_description,
        )
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=model_version.version,
        )
        logger.success(
            f"Model Registry: '{model_name}' v{model_version.version} asignado con alias @{alias}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"No se pudo asignar alias @{alias} a '{model_name}': {e}")


# =============================================================================
# PREGUNTA 1 (P1): DEMANDA HORARIA DE VIAJES (REGRESIÓN)
# =============================================================================


def cargar_datos_p1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Carga y prepara los conjuntos de P1 (Train, Validation, Test) con caché en Parquet."""
    cache_path = PROCESSED_DATA_DIR / "p1_features.parquet"

    if cache_path.exists():
        logger.info(f"Cargando características P1 desde caché: {cache_path}")
        features_p1 = pd.read_parquet(cache_path)
    else:
        logger.info("Construyendo características P1 desde archivos Parquet de viajes...")
        trips_dir = PROCESSED_DATA_DIR / "trips"
        parquet_files = sorted(trips_dir.glob("datos_abiertos_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No se encontraron viajes procesados en data/processed/trips")
        hourly_model = build_p1_hourly_demand(
            parquet_files=parquet_files,
            data_start=P1_DATA_START,
            data_end=P1_DATA_END,
        )
        features_p1 = build_p1_features(hourly_model)
        features_p1.to_parquet(cache_path, index=False)
        logger.success(f"Características P1 cacheadas en: {cache_path}")

    train_mask = features_p1["hour"] < P1_VALIDATION_START
    val_mask = (features_p1["hour"] >= P1_VALIDATION_START) & (features_p1["hour"] < P1_TEST_START)
    test_mask = features_p1["hour"] >= P1_TEST_START

    feature_columns = [
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

    return (
        features_p1.loc[train_mask].copy(),
        features_p1.loc[val_mask].copy(),
        features_p1.loc[test_mask].copy(),
        feature_columns,
    )


def ejecutar_p1(
    stage: str = "all",
    register_models: bool = True,
) -> None:
    """Ejecuta baseline, búsqueda nested y candidatos para P1."""
    configurar_mlflow()
    commit_hash = obtener_commit_git()
    version_datos = cargar_version_datos()

    logger.info("Cargando datos de P1...")
    train_df, val_df, test_df, feature_cols = cargar_datos_p1()
    target = "trips_started"

    cat_cols = ["station_id", "hour_of_day", "day_of_week", "month"]
    for c in cat_cols:
        train_df[c] = train_df[c].astype(str)
        val_df[c] = val_df[c].astype(str)
        test_df[c] = test_df[c].astype(str)

    run_feature_drift_check(
        train_df[feature_cols],
        test_df[feature_cols],
        feature_cols,
        cat_cols,
        "p1_feature_drift",
    )

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_val, y_val = val_df[feature_cols], val_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    # 1. Baseline Run: Seasonal Naive (Lag 168)
    if stage in ["all", "baseline"]:
        logger.info("=== P1: Ejecutando Baseline (Seasonal Naive) ===")
        val_pred_base = np.clip(val_df["lag_168"].to_numpy(), 0, None)
        base_mae = float(mean_absolute_error(y_val, val_pred_base))
        base_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred_base)))

        with mlflow.start_run(run_name="P1 - Baseline estacional - desarrollo"):
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P1,
                    model_family="seasonal_naive",
                    stage_of_work=TAG_STAGE_BASELINE,
                    commit_git=commit_hash,
                    rol_modelo="baseline",
                )
            )
            mlflow.log_params(
                {
                    "modelo": "seasonal_naive_lag_168",
                    "baseline_feature": "lag_168",
                    "version_datos": version_datos,
                    "validation_start": P1_VALIDATION_START,
                    "test_start": P1_TEST_START,
                }
            )
            mlflow.log_metrics(
                {
                    "validacion_mae": base_mae,
                    "validacion_rmse": base_rmse,
                }
            )
            registrar_artefactos_en_subcarpetas(
                config_dict={"baseline": "seasonal_naive", "lag": 168},
                dataset_df=val_df[feature_cols + [target]],
                dataset_target=target,
                dataset_name="p1_demand_baseline",
            )
        logger.info(f"P1 Baseline: MAE={base_mae:.4f}, RMSE={base_rmse:.4f}")

    # 2. Nested Run: Búsqueda de hiperparámetros (Grid Search)
    if stage in ["all", "search"]:
        logger.info("=== P1: Ejecutando Búsqueda de Hiperparámetros (Nested Runs) ===")
        grid_p1 = [
            {"depth": 4, "lr": 0.05, "iter": 150, "nombre": "CatBoost_depth4_lr0.05"},
            {"depth": 6, "lr": 0.10, "iter": 200, "nombre": "CatBoost_depth6_lr0.10"},
            {"depth": 8, "lr": 0.10, "iter": 200, "nombre": "CatBoost_depth8_lr0.10"},
        ]

        # Muestreo representativo para agilizar la búsqueda
        sample_train = train_df.sample(
            n=min(len(train_df), 200000), random_state=SEMILLA_ALEATORIA
        )
        X_tr_s = sample_train[feature_cols]
        y_tr_s = sample_train[target]

        with mlflow.start_run(
            run_name="P1 - CatBoost - Búsqueda de Hiperparámetros (Grid Search)"
        ):
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P1,
                    model_family="catboost",
                    stage_of_work=TAG_STAGE_SEARCH,
                    commit_git=commit_hash,
                    rol_modelo="exploracion",
                )
            )
            mlflow.log_params({"grid_size": len(grid_p1), "sample_size": len(sample_train)})

            for config in grid_p1:
                run_child_name = f"P1 - {config['nombre']}"
                with mlflow.start_run(run_name=run_child_name, nested=True):
                    mlflow.set_tags(
                        crear_etiquetas_mlflow(
                            question=TAG_QUESTION_P1,
                            model_family="catboost",
                            stage_of_work=TAG_STAGE_SEARCH,
                            commit_git=commit_hash,
                            rol_modelo="hijo_busqueda",
                        )
                    )
                    cb = CatBoostRegressor(
                        iterations=config["iter"],
                        learning_rate=config["lr"],
                        depth=config["depth"],
                        loss_function="RMSE",
                        random_seed=SEMILLA_ALEATORIA,
                        verbose=0,
                    )
                    cb.fit(X_tr_s, y_tr_s, cat_features=cat_cols)
                    val_pred = np.clip(cb.predict(X_val), 0, None)
                    v_mae = float(mean_absolute_error(y_val, val_pred))
                    v_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

                    mlflow.log_params(config)
                    mlflow.log_metrics({"validacion_mae": v_mae, "validacion_rmse": v_rmse})

            # Añadir modelo alternativo en búsqueda: Ridge
            with mlflow.start_run(run_name="P1 - Ridge (alpha=1.0)", nested=True):
                mlflow.set_tags(
                    crear_etiquetas_mlflow(
                        question=TAG_QUESTION_P1,
                        model_family="ridge",
                        stage_of_work=TAG_STAGE_SEARCH,
                        commit_git=commit_hash,
                    )
                )
                num_cols = [c for c in feature_cols if c not in cat_cols]
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_tr_s[num_cols].fillna(0), y_tr_s)
                val_pred_r = np.clip(ridge.predict(X_val[num_cols].fillna(0)), 0, None)
                mlflow.log_params({"alpha": 1.0, "family": "ridge"})
                mlflow.log_metrics(
                    {
                        "validacion_mae": float(mean_absolute_error(y_val, val_pred_r)),
                        "validacion_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred_r))),
                    }
                )

    # 3. Modelos Candidatos (Champion y Challenger con Test Set y Artefactos)
    if stage in ["all", "candidate"]:
        logger.info("=== P1: Evaluando Candidatos Finales y Artefactos ===")

        # Champion: depth=6, lr=0.10, iter=200
        champ_run_id = None
        challenger_run_id = None

        with mlflow.start_run(
            run_name="P1 - CatBoost - Candidato Champion (depth=6, lr=0.10)"
        ) as r_champ:
            champ_run_id = r_champ.info.run_id
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P1,
                    model_family="catboost",
                    stage_of_work=TAG_STAGE_CANDIDATE,
                    commit_git=commit_hash,
                    rol_modelo="champion",
                )
            )
            cb_champ = CatBoostRegressor(
                iterations=200,
                learning_rate=0.10,
                depth=6,
                loss_function="RMSE",
                random_seed=SEMILLA_ALEATORIA,
                verbose=0,
            )
            cb_champ.fit(X_train, y_train, cat_features=cat_cols)

            val_p = np.clip(cb_champ.predict(X_val), 0, None)
            test_p = np.clip(cb_champ.predict(X_test), 0, None)

            c_val_mae = float(mean_absolute_error(y_val, val_p))
            c_val_rmse = float(np.sqrt(mean_squared_error(y_val, val_p)))
            c_test_mae = float(mean_absolute_error(y_test, test_p))
            c_test_rmse = float(np.sqrt(mean_squared_error(y_test, test_p)))
            pos_mask = y_test > 0
            c_test_mae_pos = float(mean_absolute_error(y_test[pos_mask], test_p[pos_mask]))

            mlflow.log_params(
                {
                    "modelo": "catboost_regressor",
                    "depth": 6,
                    "learning_rate": 0.10,
                    "iterations": 200,
                    "semilla": SEMILLA_ALEATORIA,
                    "version_datos": version_datos,
                }
            )
            mlflow.log_metrics(
                {
                    "validacion_mae": c_val_mae,
                    "validacion_rmse": c_val_rmse,
                    "prueba_mae": c_test_mae,
                    "prueba_rmse": c_test_rmse,
                    "prueba_mae_positiva": c_test_mae_pos,
                }
            )

            # Generar y registrar gráficas de desempeño
            plot_dir = FIGURES_DIR / "p1_champion"
            plot_rvp = plot_p1_real_vs_pred(
                y_test.to_numpy(),
                test_p,
                plot_dir / "real_vs_predicho.png",
                title="P1 Champion: Demanda Real vs. Predicha (Test)",
            )
            plot_res = plot_p1_residuals(
                y_test.to_numpy(),
                test_p,
                plot_dir / "grafico_residuales.png",
                title="P1 Champion: Residuales vs. Predicho (Test)",
            )

            registrar_artefactos_en_subcarpetas(
                config_dict={"model": "CatBoost", "depth": 6, "lr": 0.1, "iterations": 200},
                plots_dict={"real_vs_pred": plot_rvp, "residuales": plot_res},
                dataset_df=test_df[feature_cols + [target]],
                dataset_target=target,
                dataset_name="p1_demand_test",
            )

            # Firma e input example
            input_ex = X_test.head(5)
            sig = infer_signature(input_ex, cb_champ.predict(input_ex))
            mlflow.catboost.log_model(
                cb_champ,
                artifact_path="model",
                signature=sig,
                input_example=input_ex,
            )

        # Challenger: depth=4, lr=0.05, iter=150
        with mlflow.start_run(
            run_name="P1 - CatBoost - Candidato Challenger (depth=4, lr=0.05)"
        ) as r_chall:
            challenger_run_id = r_chall.info.run_id
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P1,
                    model_family="catboost",
                    stage_of_work=TAG_STAGE_CANDIDATE,
                    commit_git=commit_hash,
                    rol_modelo="challenger",
                )
            )
            cb_chall = CatBoostRegressor(
                iterations=150,
                learning_rate=0.05,
                depth=4,
                loss_function="RMSE",
                random_seed=SEMILLA_ALEATORIA,
                verbose=0,
            )
            cb_chall.fit(X_train, y_train, cat_features=cat_cols)

            val_p_ch = np.clip(cb_chall.predict(X_val), 0, None)
            test_p_ch = np.clip(cb_chall.predict(X_test), 0, None)

            mlflow.log_params(
                {
                    "modelo": "catboost_regressor",
                    "depth": 4,
                    "learning_rate": 0.05,
                    "iterations": 150,
                    "semilla": SEMILLA_ALEATORIA,
                    "version_datos": version_datos,
                }
            )
            mlflow.log_metrics(
                {
                    "validacion_mae": float(mean_absolute_error(y_val, val_p_ch)),
                    "validacion_rmse": float(np.sqrt(mean_squared_error(y_val, val_p_ch))),
                    "prueba_mae": float(mean_absolute_error(y_test, test_p_ch)),
                    "prueba_rmse": float(np.sqrt(mean_squared_error(y_test, test_p_ch))),
                }
            )

            plot_dir_ch = FIGURES_DIR / "p1_challenger"
            plot_rvp_ch = plot_p1_real_vs_pred(
                y_test.to_numpy(),
                test_p_ch,
                plot_dir_ch / "real_vs_predicho.png",
                title="P1 Challenger: Demanda Real vs. Predicha",
            )
            registrar_artefactos_en_subcarpetas(
                config_dict={"model": "CatBoost", "depth": 4, "lr": 0.05, "iterations": 150},
                plots_dict={"real_vs_pred": plot_rvp_ch},
            )
            sig_ch = infer_signature(input_ex, cb_chall.predict(input_ex))
            mlflow.catboost.log_model(
                cb_chall,
                artifact_path="model",
                signature=sig_ch,
                input_example=input_ex,
            )

        # Registro en Model Registry con Aliases
        if register_models and champ_run_id and challenger_run_id:
            logger.info("Registrando modelos de P1 en Model Registry...")
            asignar_model_registry_con_alias(
                run_id=champ_run_id,
                artifact_path="model",
                model_name=MODEL_REGISTRY_P1,
                alias="champion",
                model_description="Modelo de regresión para predicción de demanda horaria en MiBici.",
                version_description=f"Champion: CatBoost depth=6, lr=0.10. Commit: {commit_hash}",
            )
            asignar_model_registry_con_alias(
                run_id=challenger_run_id,
                artifact_path="model",
                model_name=MODEL_REGISTRY_P1,
                alias="challenger",
                model_description="Modelo de regresión para predicción de demanda horaria en MiBici.",
                version_description=f"Challenger: CatBoost depth=4, lr=0.05. Commit: {commit_hash}",
            )


# =============================================================================
# =============================================================================
# PREGUNTA 2 (P2): CLASIFICACIÓN DE RIESGO OPERATIVO (MULTICLASE)
# =============================================================================


def cargar_dataset(
    ruta: Path,
    conjuntos: list[str],
) -> pd.DataFrame:
    """Carga únicamente las características, el objetivo y los conjuntos solicitados."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el dataset de P2: {ruta}. "
            "Ejecuta primero python -m mibici_mlops.features p2."
        )

    columnas = [*COLUMNAS_CARACTERISTICAS_P2, COLUMNA_OBJETIVO, COLUMNA_CONJUNTO]
    dataset = pd.read_parquet(
        ruta,
        columns=columnas,
        filters=[(COLUMNA_CONJUNTO, "in", conjuntos)],
    )
    dataset[COLUMNA_CONJUNTO] = dataset[COLUMNA_CONJUNTO].astype("string")
    encontrados = set(dataset[COLUMNA_CONJUNTO].unique())
    faltantes = set(conjuntos) - encontrados

    if faltantes:
        raise ValueError(f"No se encontraron los conjuntos: {sorted(faltantes)}")
    if not encontrados.issubset(set(conjuntos)):
        raise ValueError("Se cargaron conjuntos que no fueron solicitados.")

    return dataset


def separar_variables(dataset: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Separa X e y para cada conjunto sin incluir columnas del futuro."""
    columnas_requeridas = {
        *COLUMNAS_CARACTERISTICAS_P2,
        COLUMNA_OBJETIVO,
        COLUMNA_CONJUNTO,
    }
    faltantes = columnas_requeridas - set(dataset.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas para entrenar P2: {sorted(faltantes)}")

    separados = {}
    for conjunto in dataset[COLUMNA_CONJUNTO].unique():
        filas = dataset[dataset[COLUMNA_CONJUNTO] == conjunto]
        caracteristicas = filas[COLUMNAS_CARACTERISTICAS_P2].copy()
        objetivo = filas[COLUMNA_OBJETIVO].astype("string").copy()
        separados[str(conjunto)] = (caracteristicas, objetivo)
    return separados


def cargar_datos_p2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y particiona el dataset de P2 reutilizando cargar_dataset."""
    p2_path = P2_DIR / "dataset_p2.parquet"
    if not p2_path.exists():
        logger.info("Dataset P2 no existe, construyendo con preparar_dataset_p2...")
        trips_dir = PROCESSED_DATA_DIR / "trips"
        nom_file = (
            PROCESSED_DATA_DIR / "nomenclature" / NOMENCLATURE_FILE.replace(".csv", ".parquet")
        )
        rutas_viajes = sorted(trips_dir.glob("datos_abiertos_*.parquet"))
        dataset_p2 = preparar_dataset_p2(rutas_viajes, nom_file)
        p2_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_p2.to_parquet(p2_path, index=False)

    df = cargar_dataset(p2_path, CONJUNTOS_FINALES)
    train_df = df[df[COLUMNA_CONJUNTO] == "entrenamiento"].copy()
    val_df = df[df[COLUMNA_CONJUNTO] == "validacion"].copy()
    test_df = df[df[COLUMNA_CONJUNTO] == "prueba"].copy()

    return train_df, val_df, test_df


def crear_dummy(semilla: int = SEMILLA_ALEATORIA) -> DummyClassifier:
    """Crea el clasificador de referencia más simple y reproducible."""
    return DummyClassifier(strategy="stratified", random_state=semilla)


def crear_regresion_logistica(
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> Pipeline:
    """Crea el clasificador candidato con preprocesamiento completo para P2."""
    columnas_categoricas = ["estacion_id"]
    columnas_numericas = [
        columna for columna in COLUMNAS_CARACTERISTICAS_P2 if columna not in columnas_categoricas
    ]
    preprocesamiento = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), columnas_numericas),
            (
                "estacion",
                OneHotEncoder(handle_unknown="ignore"),
                columnas_categoricas,
            ),
        ],
        sparse_threshold=1.0,
    )
    clasificador = LogisticRegression(
        C=c,
        solver=solver.value if isinstance(solver, SolverRegresion) else solver,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=semilla,
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("clasificador", clasificador),
        ]
    )


def crear_pipeline_p2(
    c: float = 1.0,
    solver: str | SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> Pipeline:
    """Crea el pipeline scikit-learn para P2 reutilizando crear_regresion_logistica."""
    solver_enum = SolverRegresion(solver) if isinstance(solver, str) else solver
    return crear_regresion_logistica(
        semilla=SEMILLA_ALEATORIA,
        c=c,
        solver=solver_enum,
        max_iter=max_iter,
    )


def crear_modelo(
    nombre: NombreModelo,
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> DummyClassifier | Pipeline:
    """Fábrica para instanciar el modelo solicitado."""
    if nombre == NombreModelo.DUMMY:
        return crear_dummy(semilla)
    if nombre == NombreModelo.LOGISTICA:
        return crear_regresion_logistica(semilla, c, solver, max_iter)
    raise ValueError(f"Modelo desconocido: {nombre}")


def calcular_metricas(y_real: pd.Series, y_predicha: pd.Series) -> dict[str, float]:
    """Calcula métricas multiclase, incluida la media geométrica del recall."""
    recalls = recall_score(
        y_real,
        y_predicha,
        labels=ORDEN_CLASES,
        average=None,
        zero_division=0,
    )
    recalls_por_clase = dict(zip(ORDEN_CLASES, recalls, strict=True))
    gmean = float(prod(float(valor) for valor in recalls) ** (1 / len(recalls)))
    return {
        "accuracy": float(accuracy_score(y_real, y_predicha)),
        "f1_macro": float(
            f1_score(
                y_real,
                y_predicha,
                labels=ORDEN_CLASES,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_vaciado": float(recalls_por_clase["riesgo_vaciado"]),
        "recall_saturacion": float(recalls_por_clase["riesgo_saturacion"]),
        "gmean": gmean,
    }


def calcular_metricas_p2(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula las métricas multiclase de P2 reutilizando calcular_metricas."""
    return calcular_metricas(y_true, y_pred)


def evaluar_modelo(
    modelo: DummyClassifier | Pipeline,
    caracteristicas: pd.DataFrame,
    objetivo: pd.Series,
) -> tuple[dict[str, float], pd.Series]:
    """Genera predicciones y calcula métricas en un único conjunto."""
    prediccion = pd.Series(modelo.predict(caracteristicas), index=caracteristicas.index)
    return calcular_metricas(objetivo, prediccion), prediccion


def guardar_matriz_confusion(
    objetivo: pd.Series,
    prediccion: pd.Series,
    titulo: str,
    ruta: Path,
) -> None:
    """Guarda una matriz normalizada utilizando plot_p2_confusion_matrix de plots.py."""
    plot_p2_confusion_matrix(objetivo, prediccion, ruta, titulo, normalize="true")


def convertir_metricas_en_fila(
    modelo: str,
    conjunto: str,
    metricas: dict[str, float],
) -> dict[str, str | float]:
    """Agrega los identificadores del modelo y conjunto a sus métricas."""
    return {
        "modelo": modelo,
        "conjunto": conjunto,
        **metricas,
    }


def seleccionar_mejor_modelo(metricas: pd.DataFrame) -> str:
    """Selecciona el modelo con mejor F1-macro en validación."""
    validacion = metricas[metricas["conjunto"] == "validacion"]
    mejor_fila = validacion.sort_values("f1_macro", ascending=False).iloc[0]
    return str(mejor_fila["modelo"])


def mostrar_metricas(metricas: pd.DataFrame) -> None:
    """Muestra en consola una tabla corta de resultados."""
    columnas_metricas = [
        "accuracy",
        "f1_macro",
        "recall_vaciado",
        "recall_saturacion",
        "gmean",
    ]
    tabla = metricas.copy()
    tabla[columnas_metricas] = tabla[columnas_metricas].round(4)
    typer.echo(tabla.to_string(index=False))


def prefijar_metricas_mlflow(
    conjunto: str,
    metricas: dict[str, float],
) -> dict[str, float]:
    """Agrega el conjunto al nombre de cada métrica registrada."""
    return {f"{conjunto}_{nombre}": valor for nombre, valor in metricas.items()}


def crear_parametros_mlflow(
    nombre_modelo: NombreModelo,
    version_datos: str,
    conteos: dict[str, int],
    semilla: int,
    c: float,
    solver: SolverRegresion,
    max_iter: int,
) -> dict[str, str | int | float]:
    """Crea un diccionario con los parámetros que permiten reproducir el entrenamiento."""
    parametros: dict[str, str | int | float] = {
        "version_datos": version_datos,
        "semilla": semilla,
        "modelo": nombre_modelo.value,
        "filas_entrenamiento": conteos["entrenamiento"],
        "filas_validacion": conteos["validacion"],
        "inicio_entrenamiento": P2_INICIO_ENTRENAMIENTO,
        "fin_entrenamiento_exclusivo": P2_FIN_ENTRENAMIENTO,
        "inicio_validacion": P2_INICIO_VALIDACION,
        "fin_validacion_exclusivo": P2_FIN_VALIDACION,
        "inicio_prueba": P2_INICIO_PRUEBA,
        "fin_prueba_exclusivo": P2_FIN_PRUEBA,
        "horizonte_horas": P2_HORIZONTE_HORAS,
        "cuantil_vaciado": P2_CUANTIL_VACIADO,
        "cuantil_saturacion": P2_CUANTIL_SATURACION,
        "umbral_vaciado": P2_UMBRAL_VACIADO,
        "umbral_saturacion": P2_UMBRAL_SATURACION,
    }

    if nombre_modelo == NombreModelo.DUMMY:
        parametros["estrategia"] = "stratified"
    elif nombre_modelo == NombreModelo.LOGISTICA:
        parametros["c"] = c
        parametros["solver"] = solver.value if isinstance(solver, SolverRegresion) else solver
        parametros["max_iter"] = max_iter
        parametros["class_weight"] = "balanced"

    return parametros


def entrenar_y_evaluar_desarrollo(
    dataset: pd.DataFrame,
    directorio_figuras: Path,
    version_datos: str,
    commit_git: str,
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> tuple[pd.DataFrame, str]:
    """Compara Dummy y regresión logística sin consultar prueba."""
    conjuntos = separar_variables(dataset)
    x_entrenamiento, y_entrenamiento = conjuntos["entrenamiento"]
    conteos = {conjunto: len(objetivo) for conjunto, (_, objetivo) in conjuntos.items()}
    resultados = []
    nombres_run = {
        NombreModelo.DUMMY: NOMBRE_RUN_DUMMY,
        NombreModelo.LOGISTICA: NOMBRE_RUN_LOGISTICA,
    }

    for nombre in [NombreModelo.DUMMY, NombreModelo.LOGISTICA]:
        with mlflow.start_run(run_name=nombres_run[nombre]):
            rol_modelo = "baseline" if nombre == NombreModelo.DUMMY else "candidato"
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    pregunta="P2",
                    tipo_tarea="clasificacion",
                    etapa="desarrollo",
                    rol_modelo=rol_modelo,
                    commit_git=commit_git,
                )
            )
            mlflow.log_params(
                crear_parametros_mlflow(
                    nombre,
                    version_datos,
                    conteos,
                    semilla,
                    c,
                    solver,
                    max_iter,
                )
            )
            modelo = crear_modelo(nombre, semilla, c, solver, max_iter)
            modelo.fit(x_entrenamiento, y_entrenamiento)

            for conjunto in CONJUNTOS_DESARROLLO:
                caracteristicas, objetivo = conjuntos[conjunto]
                metricas, prediccion = evaluar_modelo(modelo, caracteristicas, objetivo)
                mlflow.log_metrics(prefijar_metricas_mlflow(conjunto, metricas))
                resultados.append(convertir_metricas_en_fila(nombre.value, conjunto, metricas))

                ruta_matriz = directorio_figuras / f"matriz_{nombre.value}_{conjunto}.png"
                guardar_matriz_confusion(
                    objetivo,
                    prediccion,
                    f"Matriz {nombre.value} - {conjunto}",
                    ruta_matriz,
                )
                mlflow.log_artifact(
                    str(ruta_matriz),
                    artifact_path="matrices_confusion",
                )

    tabla_metricas = pd.DataFrame(resultados)
    mejor_modelo = seleccionar_mejor_modelo(tabla_metricas)
    return tabla_metricas, mejor_modelo


def evaluar_modelo_final(
    dataset: pd.DataFrame,
    nombre: NombreModelo,
    directorio_figuras: Path,
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> tuple[DummyClassifier | Pipeline, pd.DataFrame]:
    """Entrena el modelo confirmado y lo evalúa en los tres conjuntos."""
    conjuntos = separar_variables(dataset)
    x_entrenamiento, y_entrenamiento = conjuntos["entrenamiento"]
    modelo = crear_modelo(nombre, semilla, c, solver, max_iter)
    modelo.fit(x_entrenamiento, y_entrenamiento)
    resultados = []

    for conjunto in CONJUNTOS_FINALES:
        caracteristicas, objetivo = conjuntos[conjunto]
        metricas, prediccion = evaluar_modelo(modelo, caracteristicas, objetivo)
        resultados.append(convertir_metricas_en_fila(nombre.value, conjunto, metricas))
        guardar_matriz_confusion(
            objetivo,
            prediccion,
            f"Modelo final {nombre.value} - {conjunto}",
            directorio_figuras / f"matriz_final_{conjunto}.png",
        )

    return modelo, pd.DataFrame(resultados)


def ejecutar_p2(stage: str = "all", register_models: bool = True) -> None:
    """Ejecuta baseline, búsqueda nested y candidatos para P2."""
    configurar_mlflow()
    commit_hash = obtener_commit_git()
    version_datos = cargar_version_datos()

    logger.info("Cargando datos de P2...")
    train_df, val_df, test_df = cargar_datos_p2()
    X_train, y_train = (
        train_df[COLUMNAS_CARACTERISTICAS_P2],
        train_df[COLUMNA_OBJETIVO].astype(str),
    )
    X_val, y_val = val_df[COLUMNAS_CARACTERISTICAS_P2], val_df[COLUMNA_OBJETIVO].astype(str)
    X_test, y_test = test_df[COLUMNAS_CARACTERISTICAS_P2], test_df[COLUMNA_OBJETIVO].astype(str)

    run_feature_drift_check(
        X_train,
        X_test,
        COLUMNAS_CARACTERISTICAS_P2,
        ["estacion_id"],
        "p2_feature_drift",
    )

    # 1. Baseline: DummyClassifier
    if stage in ["all", "baseline"]:
        logger.info("=== P2: Ejecutando Baseline (Dummy Estratificado) ===")
        dummy = DummyClassifier(strategy="stratified", random_state=SEMILLA_ALEATORIA)
        dummy.fit(X_train, y_train)
        pred_val = dummy.predict(X_val)
        m_val = calcular_metricas_p2(y_val, pd.Series(pred_val, index=y_val.index))

        with mlflow.start_run(run_name="P2 - Dummy estratificado - desarrollo"):
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P2,
                    model_family="dummy",
                    stage_of_work=TAG_STAGE_BASELINE,
                    commit_git=commit_hash,
                    rol_modelo="baseline",
                )
            )
            mlflow.log_params(
                {
                    "modelo": "dummy_stratified",
                    "strategy": "stratified",
                    "version_datos": version_datos,
                    "semilla": SEMILLA_ALEATORIA,
                }
            )
            for k, v in m_val.items():
                mlflow.log_metric(f"validacion_{k}", v)

            p_base_cm = plot_p2_confusion_matrix(
                y_val,
                pred_val,
                ORDEN_CLASES,
                FIGURES_DIR / "p2_baseline_cm.png",
                "P2 Baseline Matriz de Confusión",
            )
            registrar_artefactos_en_subcarpetas(
                config_dict={"modelo": "dummy_stratified"},
                plots_dict={"confusion_matrix": p_base_cm},
            )
        logger.info(f"P2 Baseline: F1={m_val['f1_macro']:.4f}, Acc={m_val['accuracy']:.4f}")

    # 2. Nested Run: Búsqueda de regularización C
    if stage in ["all", "search"]:
        logger.info("=== P2: Ejecutando Búsqueda de Regularización C (Nested Runs) ===")
        c_values = [0.01, 0.1, 1.0, 10.0]

        # Muestreo estratificado para exploración rápida
        sample_size = min(len(train_df), 150000)
        idx_sample = (
            train_df.groupby(COLUMNA_OBJETIVO, group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=int(sample_size * len(x) / len(train_df)), random_state=SEMILLA_ALEATORIA
                )
            )
            .index
        )
        X_tr_s = X_train.loc[idx_sample]
        y_tr_s = y_train.loc[idx_sample]

        with mlflow.start_run(
            run_name="P2 - Logistic Regression - Búsqueda Regularización C (Grid Search)"
        ):
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P2,
                    model_family="logistic_regression",
                    stage_of_work=TAG_STAGE_SEARCH,
                    commit_git=commit_hash,
                    rol_modelo="exploracion",
                )
            )
            mlflow.log_params({"grid_c_values": c_values, "search_sample_size": len(X_tr_s)})

            for c_val in c_values:
                with mlflow.start_run(
                    run_name=f"P2 - Logistic Regression (C={c_val})", nested=True
                ):
                    mlflow.set_tags(
                        crear_etiquetas_mlflow(
                            question=TAG_QUESTION_P2,
                            model_family="logistic_regression",
                            stage_of_work=TAG_STAGE_SEARCH,
                            commit_git=commit_hash,
                            rol_modelo="hijo_busqueda",
                        )
                    )
                    pipe_s = crear_pipeline_p2(c=c_val, max_iter=40)
                    pipe_s.fit(X_tr_s, y_tr_s)
                    p_val_s = pipe_s.predict(X_val)
                    m_s = calcular_metricas_p2(y_val, pd.Series(p_val_s, index=y_val.index))

                    mlflow.log_params({"C": c_val, "solver": "saga", "max_iter": 40})
                    for k, v in m_s.items():
                        mlflow.log_metric(f"validacion_{k}", v)

    # 3. Modelos Candidatos (Champion y Challenger con Test Set y Artefactos)
    if stage in ["all", "candidate"]:
        logger.info("=== P2: Evaluando Candidatos Finales y Artefactos ===")
        champ_run_id = None
        challenger_run_id = None

        # Champion: C=1.0 (entrenamiento con conjunto completo)
        with mlflow.start_run(
            run_name="P2 - Regresión Logística - Candidato Champion (C=1.0)"
        ) as r_champ:
            champ_run_id = r_champ.info.run_id
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P2,
                    model_family="logistic_regression",
                    stage_of_work=TAG_STAGE_CANDIDATE,
                    commit_git=commit_hash,
                    rol_modelo="champion",
                )
            )
            pipe_champ = crear_pipeline_p2(c=1.0, max_iter=100)
            pipe_champ.fit(X_train, y_train)

            pred_test_ch = pipe_champ.predict(X_test)
            proba_test_ch = pipe_champ.predict_proba(X_test)
            m_test_ch = calcular_metricas_p2(y_test, pd.Series(pred_test_ch, index=y_test.index))

            mlflow.log_params(
                {
                    "modelo": "logistic_regression",
                    "C": 1.0,
                    "solver": "saga",
                    "max_iter": 100,
                    "class_weight": "balanced",
                    "version_datos": version_datos,
                    "semilla": SEMILLA_ALEATORIA,
                }
            )
            for k, v in m_test_ch.items():
                mlflow.log_metric(f"prueba_{k}", v)

            # Artefactos: plots/
            p2_dir_champ = FIGURES_DIR / "p2_champion"
            plot_cm = plot_p2_confusion_matrix(
                y_test,
                pred_test_ch,
                ORDEN_CLASES,
                p2_dir_champ / "matriz_confusion.png",
                "P2 Champion: Matriz de Confusión (Test)",
            )
            plot_roc = plot_p2_roc_curves(
                y_test,
                proba_test_ch,
                ORDEN_CLASES,
                p2_dir_champ / "curva_roc.png",
                "P2 Champion: Curvas ROC (Test)",
            )

            # Guardar modelo skops localmente
            model_skops_path = MODELS_DIR / "p2" / "modelo_p2.skops"
            model_skops_path.parent.mkdir(parents=True, exist_ok=True)
            sio.dump(pipe_champ, model_skops_path)

            registrar_artefactos_en_subcarpetas(
                config_dict={"modelo": "logistic_regression", "C": 1.0, "solver": "saga"},
                plots_dict={"confusion_matrix": plot_cm, "roc_curve": plot_roc},
                dataset_df=test_df[COLUMNAS_CARACTERISTICAS_P2 + [COLUMNA_OBJETIVO]],
                dataset_target=COLUMNA_OBJETIVO,
                dataset_name="p2_risk_test",
            )

            input_ex_p2 = X_test.head(5)
            sig_p2 = infer_signature(input_ex_p2, pipe_champ.predict(input_ex_p2))
            mlflow.sklearn.log_model(
                pipe_champ,
                artifact_path="model",
                signature=sig_p2,
                input_example=input_ex_p2,
            )

        # Challenger: C=0.1
        with mlflow.start_run(
            run_name="P2 - Regresión Logística - Candidato Challenger (C=0.1)"
        ) as r_chall:
            challenger_run_id = r_chall.info.run_id
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P2,
                    model_family="logistic_regression",
                    stage_of_work=TAG_STAGE_CANDIDATE,
                    commit_git=commit_hash,
                    rol_modelo="challenger",
                )
            )
            pipe_chall = crear_pipeline_p2(c=0.1, max_iter=80)
            pipe_chall.fit(X_train, y_train)

            pred_test_cl = pipe_chall.predict(X_test)
            m_test_cl = calcular_metricas_p2(y_test, pd.Series(pred_test_cl, index=y_test.index))

            mlflow.log_params(
                {
                    "modelo": "logistic_regression",
                    "C": 0.1,
                    "solver": "saga",
                    "max_iter": 80,
                    "class_weight": "balanced",
                    "version_datos": version_datos,
                    "semilla": SEMILLA_ALEATORIA,
                }
            )
            for k, v in m_test_cl.items():
                mlflow.log_metric(f"prueba_{k}", v)

            p2_dir_chall = FIGURES_DIR / "p2_challenger"
            plot_cm_cl = plot_p2_confusion_matrix(
                y_test,
                pred_test_cl,
                ORDEN_CLASES,
                p2_dir_chall / "matriz_confusion.png",
                "P2 Challenger: Matriz de Confusión",
            )
            registrar_artefactos_en_subcarpetas(
                config_dict={"modelo": "logistic_regression", "C": 0.1, "solver": "saga"},
                plots_dict={"confusion_matrix": plot_cm_cl},
            )
            sig_p2_cl = infer_signature(input_ex_p2, pipe_chall.predict(input_ex_p2))
            mlflow.sklearn.log_model(
                pipe_chall,
                artifact_path="model",
                signature=sig_p2_cl,
                input_example=input_ex_p2,
            )

        # Registro en Model Registry
        if register_models and champ_run_id and challenger_run_id:
            logger.info("Registrando modelos de P2 en Model Registry...")
            asignar_model_registry_con_alias(
                run_id=champ_run_id,
                artifact_path="model",
                model_name=MODEL_REGISTRY_P2,
                alias="champion",
                model_description="Modelo de clasificación multiclase de riesgo operativo (vaciado/normal/saturación) para MiBici.",
                version_description=f"Champion: LogisticRegression C=1.0 balanced. Commit: {commit_hash}",
            )
            asignar_model_registry_con_alias(
                run_id=challenger_run_id,
                artifact_path="model",
                model_name=MODEL_REGISTRY_P2,
                alias="challenger",
                model_description="Modelo de clasificación multiclase de riesgo operativo para MiBici.",
                version_description=f"Challenger: LogisticRegression C=0.1 balanced. Commit: {commit_hash}",
            )


# =============================================================================


# PREGUNTA 3 (P3): AGRUPAMIENTO DE ESTACIONES (CLUSTERING)
# =============================================================================


def cargar_datos_p3() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga viajes y nomenclatura para generar las 7 características de estación de P3."""
    trips_dir = PROCESSED_DATA_DIR / "trips"
    nom_dir = PROCESSED_DATA_DIR / "nomenclature"

    trips_df = pd.read_parquet(trips_dir)
    nom_files = list(nom_dir.glob("*.parquet"))
    if not nom_files:
        raise FileNotFoundError(f"No se encontró nomenclatura en {nom_dir}")
    stations_df = pd.read_parquet(nom_files[0])

    return trips_df, stations_df


def ejecutar_p3(stage: str = "all", register_models: bool = True) -> None:
    """Ejecuta baseline, búsqueda nested y candidatos para P3."""
    configurar_mlflow()
    commit_hash = obtener_commit_git()
    version_datos = cargar_version_datos()

    logger.info("Cargando y procesando características P3...")
    trips_df, stations_df = cargar_datos_p3()
    features_df = generate_p3_station_features(trips_df, stations_df)
    n_stations, n_features = features_df.shape
    logger.info(f"Matriz P3 construida: {n_stations} estaciones x {n_features} variables.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # 1. Baseline: K-Means (k=4)
    if stage in ["all", "baseline"]:
        logger.info("=== P3: Ejecutando Baseline (K-Means k=4) ===")
        km_base = KMeans(n_clusters=P3_DEFAULT_K, random_state=SEMILLA_ALEATORIA, n_init=10)
        labels_base = km_base.fit_predict(X_scaled)

        sil_base = float(silhouette_score(X_scaled, labels_base))
        dbi_base = float(davies_bouldin_score(X_scaled, labels_base))
        ine_base = float(km_base.inertia_)

        with mlflow.start_run(run_name="P3 - KMeans - desarrollo (k=4)"):
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P3,
                    model_family="kmeans",
                    stage_of_work=TAG_STAGE_BASELINE,
                    commit_git=commit_hash,
                    rol_modelo="baseline",
                )
            )
            mlflow.log_params(
                {
                    "modelo": "kmeans",
                    "n_clusters": P3_DEFAULT_K,
                    "semilla": SEMILLA_ALEATORIA,
                    "version_datos": version_datos,
                    "n_stations": n_stations,
                }
            )
            mlflow.log_metrics(
                {
                    "silhouette_score": sil_base,
                    "davies_bouldin_index": dbi_base,
                    "inercia": ine_base,
                }
            )
            p_base_map = plot_p3_spatial_clusters(
                features_df,
                labels_base,
                FIGURES_DIR / "p3_baseline_spatial.png",
                f"P3 Baseline: Clusters Espaciales (k={P3_DEFAULT_K})",
            )
            registrar_artefactos_en_subcarpetas(
                config_dict={"modelo": "kmeans", "k": P3_DEFAULT_K},
                plots_dict={"clusters_spatial": p_base_map},
            )
        logger.info(f"P3 Baseline: Silhouette={sil_base:.4f}, DBI={dbi_base:.4f}")

    # 2. Nested Run: Exploración de K y Algoritmos
    if stage in ["all", "search"]:
        logger.info("=== P3: Ejecutando Exploración de K y Algoritmos (Nested Runs) ===")
        k_values = [3, 4, 5, 6]

        with mlflow.start_run(
            run_name="P3 - Clustering - Exploración de K y Algoritmos (Grid Search)"
        ):
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P3,
                    model_family="kmeans",
                    stage_of_work=TAG_STAGE_SEARCH,
                    commit_git=commit_hash,
                    rol_modelo="exploracion",
                )
            )
            mlflow.log_params({"k_explored": k_values})

            for k in k_values:
                with mlflow.start_run(run_name=f"P3 - KMeans (k={k})", nested=True):
                    mlflow.set_tags(
                        crear_etiquetas_mlflow(
                            question=TAG_QUESTION_P3,
                            model_family="kmeans",
                            stage_of_work=TAG_STAGE_SEARCH,
                            commit_git=commit_hash,
                            rol_modelo="hijo_busqueda",
                        )
                    )
                    km_k = KMeans(n_clusters=k, random_state=SEMILLA_ALEATORIA, n_init=10)
                    lab_k = km_k.fit_predict(X_scaled)
                    mlflow.log_params({"n_clusters": k, "modelo": "kmeans"})
                    mlflow.log_metrics(
                        {
                            "silhouette_score": float(silhouette_score(X_scaled, lab_k)),
                            "davies_bouldin_index": float(davies_bouldin_score(X_scaled, lab_k)),
                            "inercia": float(km_k.inertia_),
                        }
                    )

            # Algoritmo Alternativo: Agglomerative Clustering
            with mlflow.start_run(
                run_name="P3 - Agglomerative Clustering (k=4, ward)", nested=True
            ):
                mlflow.set_tags(
                    crear_etiquetas_mlflow(
                        question=TAG_QUESTION_P3,
                        model_family="agglomerative",
                        stage_of_work=TAG_STAGE_SEARCH,
                        commit_git=commit_hash,
                        rol_modelo="hijo_busqueda",
                    )
                )
                agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
                lab_agg = agg.fit_predict(X_scaled)
                mlflow.log_params({"n_clusters": 4, "modelo": "agglomerative", "linkage": "ward"})
                mlflow.log_metrics(
                    {
                        "silhouette_score": float(silhouette_score(X_scaled, lab_agg)),
                        "davies_bouldin_index": float(davies_bouldin_score(X_scaled, lab_agg)),
                    }
                )

    # 3. Modelos Candidatos (Champion y Challenger con Artefactos)
    if stage in ["all", "candidate"]:
        logger.info("=== P3: Evaluando Candidatos Finales y Artefactos ===")
        champ_run_id = None
        challenger_run_id = None

        # Champion: K-Means con k=4 (mejor balance de Silhouette y separación interpretable)
        with mlflow.start_run(run_name="P3 - KMeans - Candidato Champion (k=4)") as r_champ:
            champ_run_id = r_champ.info.run_id
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P3,
                    model_family="kmeans",
                    stage_of_work=TAG_STAGE_CANDIDATE,
                    commit_git=commit_hash,
                    rol_modelo="champion",
                )
            )
            # Empaquetar pipeline con escalador para inferencia out-of-sample
            pipe_p3_champ = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("kmeans", KMeans(n_clusters=4, random_state=SEMILLA_ALEATORIA, n_init=10)),
                ]
            )
            pipe_p3_champ.fit(features_df)
            labels_champ = pipe_p3_champ.named_steps["kmeans"].labels_

            sil_ch = float(silhouette_score(X_scaled, labels_champ))
            dbi_ch = float(davies_bouldin_score(X_scaled, labels_champ))
            ine_ch = float(pipe_p3_champ.named_steps["kmeans"].inertia_)

            mlflow.log_params(
                {
                    "modelo": "kmeans",
                    "n_clusters": 4,
                    "semilla": SEMILLA_ALEATORIA,
                    "scaler": "StandardScaler",
                    "n_stations": n_stations,
                    "version_datos": version_datos,
                    "features_list": list(features_df.columns),
                }
            )
            mlflow.log_metrics(
                {
                    "silhouette_score": sil_ch,
                    "davies_bouldin_index": dbi_ch,
                    "inercia": ine_ch,
                }
            )

            p3_dir_champ = FIGURES_DIR / "p3_champion"
            p_map_ch = plot_p3_spatial_clusters(
                features_df,
                labels_champ,
                p3_dir_champ / "p3_clusters_spatial.png",
                "P3 Champion: Clusters Espaciales (k=4)",
            )
            p_dist_ch = plot_p3_cluster_distribution(
                features_df,
                labels_champ,
                p3_dir_champ / "p3_clusters_distribucion.png",
                "P3 Champion: Distribución de Estaciones",
            )

            registrar_artefactos_en_subcarpetas(
                config_dict={"modelo": "kmeans", "k": 4, "scaler": "StandardScaler"},
                plots_dict={"clusters_spatial": p_map_ch, "clusters_distribution": p_dist_ch},
                dataset_df=features_df,
                dataset_name="p3_station_profiles",
            )

            input_ex_p3 = features_df.head(5)
            sig_p3 = infer_signature(input_ex_p3, pipe_p3_champ.predict(input_ex_p3))
            mlflow.sklearn.log_model(
                pipe_p3_champ,
                artifact_path="model",
                signature=sig_p3,
                input_example=input_ex_p3,
            )

        # Challenger: K-Means con k=5
        with mlflow.start_run(run_name="P3 - KMeans - Candidato Challenger (k=5)") as r_chall:
            challenger_run_id = r_chall.info.run_id
            mlflow.set_tags(
                crear_etiquetas_mlflow(
                    question=TAG_QUESTION_P3,
                    model_family="kmeans",
                    stage_of_work=TAG_STAGE_CANDIDATE,
                    commit_git=commit_hash,
                    rol_modelo="challenger",
                )
            )
            pipe_p3_chall = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("kmeans", KMeans(n_clusters=5, random_state=SEMILLA_ALEATORIA, n_init=10)),
                ]
            )
            pipe_p3_chall.fit(features_df)
            labels_chall = pipe_p3_chall.named_steps["kmeans"].labels_

            mlflow.log_params(
                {
                    "modelo": "kmeans",
                    "n_clusters": 5,
                    "semilla": SEMILLA_ALEATORIA,
                    "scaler": "StandardScaler",
                    "version_datos": version_datos,
                }
            )
            mlflow.log_metrics(
                {
                    "silhouette_score": float(silhouette_score(X_scaled, labels_chall)),
                    "davies_bouldin_index": float(davies_bouldin_score(X_scaled, labels_chall)),
                    "inercia": float(pipe_p3_chall.named_steps["kmeans"].inertia_),
                }
            )

            p3_dir_chall = FIGURES_DIR / "p3_challenger"
            p_map_cl = plot_p3_spatial_clusters(
                features_df,
                labels_chall,
                p3_dir_chall / "p3_clusters_spatial.png",
                "P3 Challenger: Clusters Espaciales (k=5)",
            )
            registrar_artefactos_en_subcarpetas(
                config_dict={"modelo": "kmeans", "k": 5, "scaler": "StandardScaler"},
                plots_dict={"clusters_spatial": p_map_cl},
            )
            sig_p3_cl = infer_signature(input_ex_p3, pipe_p3_chall.predict(input_ex_p3))
            mlflow.sklearn.log_model(
                pipe_p3_chall,
                artifact_path="model",
                signature=sig_p3_cl,
                input_example=input_ex_p3,
            )

        # Registro en Model Registry
        if register_models and champ_run_id and challenger_run_id:
            logger.info("Registrando modelos de P3 en Model Registry...")
            asignar_model_registry_con_alias(
                run_id=champ_run_id,
                artifact_path="model",
                model_name=MODEL_REGISTRY_P3,
                alias="champion",
                model_description="Modelo de agrupamiento no supervisado de estaciones MiBici por patrón de movilidad y perfil horario.",
                version_description=f"Champion: KMeans k=4 + StandardScaler. Commit: {commit_hash}",
            )
            asignar_model_registry_con_alias(
                run_id=challenger_run_id,
                artifact_path="model",
                model_name=MODEL_REGISTRY_P3,
                alias="challenger",
                model_description="Modelo de agrupamiento no supervisado de estaciones MiBici.",
                version_description=f"Challenger: KMeans k=5 + StandardScaler. Commit: {commit_hash}",
            )


# =============================================================================


# =============================================================================
# COMANDOS CLI (MLPROJECT Y RETROCOMPATIBILIDAD)
# =============================================================================


@app.command("run-mlproject")
def run_mlproject(
    question: Annotated[str, typer.Option(help="Pregunta a entrenar: p1, p2, p3 o all")] = "all",
    model: Annotated[str, typer.Option(help="Modelo específico o default")] = "default",
    stage: Annotated[str, typer.Option(help="Etapa: all, baseline, search o candidate")] = "all",
    register_models: Annotated[
        str, typer.Option(help="Registrar en Model Registry (true/false)")
    ] = "true",
) -> None:
    """Punto de entrada estándar para ejecutar desde MLproject."""
    reg = register_models.strip().lower() in ["true", "1", "yes"]
    q = question.strip().lower()

    logger.info(f"Iniciando run-mlproject: question={q}, stage={stage}, register={reg}")

    if q in ["p1", "all"]:
        ejecutar_p1(stage=stage, register_models=reg)

    if q in ["p2", "all"]:
        ejecutar_p2(stage=stage, register_models=reg)
    if q in ["p3", "all"]:
        ejecutar_p3(stage=stage, register_models=reg)

    logger.success("run-mlproject finalizado con éxito.")


@app.command("main")
def main_p1() -> None:
    """Comando retrocompatible para P1."""
    ejecutar_p1(stage="all", register_models=True)


@app.command("comparar")
def comparar(
    data_path: Path = P2_DIR / "dataset_p2.parquet",
    metrics_path: Path = P2_DIR / "metricas_desarrollo.csv",
    figures_dir: Path = P2_FIGURAS_DIR,
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> None:
    """Compara los modelos usando solamente entrenamiento y validación."""
    configurar_mlflow()
    version_datos = cargar_version_datos()
    commit_git = obtener_commit_git()
    logger.info(
        f"Parámetros de regresión logística: C={c}, solver={solver.value}, "
        f"max_iter={max_iter}, semilla={semilla}."
    )
    dataset = cargar_dataset(data_path, CONJUNTOS_DESARROLLO)
    metricas, mejor_modelo = entrenar_y_evaluar_desarrollo(
        dataset,
        figures_dir,
        version_datos,
        commit_git,
        semilla,
        c,
        solver,
        max_iter,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(metrics_path, index=False)
    mostrar_metricas(metricas)
    logger.success(
        f"Comparación terminada. Mejor modelo en validación: {mejor_modelo}. "
        f"Métricas: {metrics_path}"
    )


@app.command("evaluar-final")
def evaluar_final(
    modelo: Annotated[NombreModelo, typer.Option("--modelo")],
    data_path: Path = P2_DIR / "dataset_p2.parquet",
    metrics_path: Path = P2_DIR / "metricas_finales.csv",
    figures_dir: Path = P2_FIGURAS_DIR,
    model_path: Path = MODELS_DIR / "p2" / "modelo_p2.skops",
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> None:
    """Evalúa en prueba un modelo ya seleccionado mediante validación."""
    configurar_mlflow()
    version_datos = cargar_version_datos()
    commit_git = obtener_commit_git()
    logger.info(
        f"Parámetros de regresión logística: C={c}, solver={solver.value}, "
        f"max_iter={max_iter}, semilla={semilla}."
    )
    dataset = cargar_dataset(data_path, CONJUNTOS_FINALES)
    conteos = {conjunto: len(df) for conjunto, df in dataset.groupby(COLUMNA_CONJUNTO)}
    with mlflow.start_run(run_name=NOMBRE_RUN_FINAL):
        mlflow.set_tags(
            crear_etiquetas_mlflow(
                pregunta="P2",
                tipo_tarea="clasificacion",
                etapa="evaluacion_final",
                rol_modelo="final",
                commit_git=commit_git,
            )
        )
        mlflow.log_params(
            crear_parametros_mlflow(
                modelo,
                version_datos,
                conteos,
                semilla,
                c,
                solver,
                max_iter,
            )
        )
        modelo_final, metricas = evaluar_modelo_final(
            dataset,
            modelo,
            figures_dir,
            semilla,
            c,
            solver,
            max_iter,
        )
        for _, fila in metricas.iterrows():
            conjunto = str(fila["conjunto"])
            valores = {
                clave: float(fila[clave])
                for clave in [
                    "accuracy",
                    "f1_macro",
                    "recall_vaciado",
                    "recall_saturacion",
                    "gmean",
                ]
            }
            mlflow.log_metrics(prefijar_metricas_mlflow(conjunto, valores))

        for conjunto in CONJUNTOS_FINALES:
            ruta_matriz = figures_dir / f"matriz_final_{conjunto}.png"
            mlflow.log_artifact(
                str(ruta_matriz),
                artifact_path="matrices_confusion",
            )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(modelo_final, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="modelos")
        logger.info(f"Modelo persistido en: {model_path}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(metrics_path, index=False)
    mostrar_metricas(metricas)
    logger.success(f"Evaluación final guardada en: {metrics_path}")


@app.command("p3")
def train_p3_baseline() -> None:
    """Comando retrocompatible para P3."""
    ejecutar_p3(stage="all", register_models=True)


PASOS_EXPERIMENTO_COMPLETO = [
    (
        "P1: regresión de demanda",
        [sys.executable, "-m", "mibici_mlops.modeling.train", "main"],
    ),
    (
        "P2: construcción del conjunto estación-hora",
        [sys.executable, "-m", "mibici_mlops.features", "p2"],
    ),
    (
        "P2: comparación de modelos",
        [
            sys.executable,
            "-m",
            "mibici_mlops.modeling.train",
            "comparar",
            "--c",
            "1.0",
            "--solver",
            SolverRegresion.SAGA.value,
            "--max-iter",
            "100",
            "--semilla",
            str(SEMILLA_ALEATORIA),
            "--metrics-path",
            "data/processed/p2/metricas_desarrollo.csv",
        ],
    ),
    (
        "P2: evaluación final",
        [
            sys.executable,
            "-m",
            "mibici_mlops.modeling.train",
            "evaluar-final",
            "--modelo",
            NombreModelo.LOGISTICA.value,
            "--c",
            "1.0",
            "--solver",
            SolverRegresion.SAGA.value,
            "--max-iter",
            "100",
            "--semilla",
            str(SEMILLA_ALEATORIA),
            "--metrics-path",
            "data/processed/p2/metricas_finales.csv",
            "--model-path",
            "models/p2/modelo_p2.skops",
        ],
    ),
    (
        "P3: agrupamiento de estaciones",
        [sys.executable, "-m", "mibici_mlops.modeling.train", "p3"],
    ),
]


def ejecutar_paso_experimento(nombre: str, comando: list[str]) -> None:
    """Ejecuta una etapa aislada y detiene el pipeline si esta falla."""
    logger.info(f"Iniciando {nombre}.")
    try:
        subprocess.run(comando, cwd=PROJ_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        logger.error(f"Falló {nombre}. El experimento se detendrá.")
        raise typer.Exit(code=error.returncode) from error
    logger.success(f"Etapa terminada: {nombre}.")


@app.command("experimento-completo")
def experimento_completo() -> None:
    """Ejecuta el ciclo de P1, P2 y P3 completo paso a paso tras la ingesta."""
    logger.info(
        "Ejecutando el experimento completo. La ingesta debe haberse ejecutado previamente."
    )
    for nombre, comando in PASOS_EXPERIMENTO_COMPLETO:
        ejecutar_paso_experimento(nombre, comando)
    logger.success("Experimento completo finalizado correctamente.")


if __name__ == "__main__":
    app()
