"""Inferencia y verificación programática de modelos registrados en MLflow Model Registry."""

from pathlib import Path
import sys

from loguru import logger
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import typer

from mibici_mlops.config import (
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_P1,
    MODEL_REGISTRY_P2,
    MODEL_REGISTRY_P3,
)

app = typer.Typer(help="Inferencia y verificación de modelos en MLflow Model Registry.")


def cargar_modelo_registrado(nombre_modelo: str, alias: str = "champion"):
    """Carga un modelo registrado utilizando su URI con alias (@champion / @challenger)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    uri = f"models:/{nombre_modelo}@{alias}"
    logger.info(f"Cargando modelo desde URI: {uri}")
    try:
        model = mlflow.pyfunc.load_model(uri)
        logger.success(f"Modelo cargado exitosamente: {nombre_modelo}@{alias}")
        return model
    except Exception as error:
        logger.error(f"Error al cargar {uri}: {error}")
        raise


def _obtener_datos_sinteticos(nombre_modelo: str) -> pd.DataFrame:
    """Genera datos sintéticos con tipos estrictos para verificar la inferencia."""
    if nombre_modelo == MODEL_REGISTRY_P1:
        return pd.DataFrame(
            {
                "station_id": ["2", "3"],
                "hour_of_day": ["8", "19"],
                "day_of_week": ["1", "5"],
                "is_weekend": np.array([0, 1], dtype=np.int32),
                "month": ["6", "6"],
                "lag_1": np.array([1.0, 4.0], dtype=np.float64),
                "lag_24": np.array([2.0, 5.0], dtype=np.float64),
                "lag_168": np.array([2.0, 6.0], dtype=np.float64),
                "rolling_mean_3h": np.array([1.5, 4.2], dtype=np.float64),
                "rolling_mean_24h": np.array([2.1, 3.8], dtype=np.float64),
                "history_hours": np.array([500, 1200], dtype=np.int64),
            }
        )
    if nombre_modelo == MODEL_REGISTRY_P2:
        return pd.DataFrame(
            {
                "estacion_id": np.array([2, 3], dtype=np.int32),
                "capacidad": np.array([16, 24], dtype=np.int32),
                "latitud": np.array([20.67, 20.68], dtype=np.float32),
                "longitud": np.array([-103.35, -103.36], dtype=np.float32),
                "hora": np.array([8, 19], dtype=np.int32),
                "dia_semana": np.array([1, 5], dtype=np.int32),
                "mes": np.array([6, 6], dtype=np.int32),
                "es_fin_semana": np.array([0, 1], dtype=np.int32),
                "entradas_t_1": np.array([1.0, 3.0], dtype=np.float64),
                "salidas_t_1": np.array([2.0, 1.0], dtype=np.float64),
                "flujo_neto_t_1": np.array([-1.0, 2.0], dtype=np.float64),
                "flujo_neto_t_2": np.array([0.0, 1.0], dtype=np.float64),
                "flujo_neto_t_24": np.array([-1.0, 0.0], dtype=np.float64),
                "entradas_media_24h": np.array([1.2, 2.5], dtype=np.float64),
                "salidas_media_24h": np.array([1.8, 1.1], dtype=np.float64),
                "flujo_neto_media_3h": np.array([-0.6, 1.3], dtype=np.float64),
                "flujo_neto_media_24h": np.array([-0.6, 1.4], dtype=np.float64),
            }
        )
    if nombre_modelo == MODEL_REGISTRY_P3:
        return pd.DataFrame(
            {
                "latitude": [20.675, 20.685],
                "longitude": [-103.355, -103.365],
                "ratio_am": [1.8, 0.4],
                "ratio_pm": [0.5, 2.1],
                "pct_fin_semana": [0.22, 0.35],
                "pct_domingo": [0.10, 0.20],
                "rotacion_capacidad": [150.0, 85.0],
            }
        )
    raise ValueError(f"Modelo no reconocido para datos sintéticos: {nombre_modelo}")


@app.command("verify-registry")
def verify_registry() -> None:
    """Verifica programáticamente que los 3 modelos Champion son cargables y ejecutan inferencia."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("=== INICIANDO VERIFICACIÓN DE MODEL REGISTRY (@champion) ===")
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

    modelos = [
        (MODEL_REGISTRY_P1, "P1 - Demanda Horaria"),
        (MODEL_REGISTRY_P2, "P2 - Riesgo Operativo"),
        (MODEL_REGISTRY_P3, "P3 - Grupos de Estaciones"),
    ]

    resultados = []

    for nombre, etiqueta in modelos:
        logger.info(f"\n--- Verificando {etiqueta} ({nombre}) ---")
        try:
            model = cargar_modelo_registrado(nombre, alias="champion")
            ejemplo = _obtener_datos_sinteticos(nombre)
            predicciones = model.predict(ejemplo)
            pred_list = (
                np.array(predicciones).flatten().tolist()
                if hasattr(predicciones, "flatten")
                else list(predicciones)
            )
            logger.info(f"Predicciones de prueba para {nombre}: {pred_list}")
            assert len(pred_list) == len(ejemplo)
            resultados.append((nombre, "EXITOSO", f"Predicciones: {pred_list[:2]}"))

        except Exception as error:  # noqa: BLE001
            logger.error(f"Fallo en verificación de {nombre}: {error}")
            resultados.append((nombre, "FALLIDO", str(error)))

    logger.info("\n=== RESUMEN DE VERIFICACIÓN DE MODEL REGISTRY ===")
    todo_exitoso = True
    for nombre, estado, detalle in resultados:
        logger.info(f"  • {nombre} -> {estado} ({detalle})")
        if estado != "EXITOSO":
            todo_exitoso = False

    if todo_exitoso:
        logger.success("Todos los modelos @champion fueron verificados con éxito.")
    else:
        logger.error("Al menos un modelo en el registry presentó error.")
        sys.exit(1)


@app.command("predict")
def predict_cmd(
    model_name: str = typer.Option(MODEL_REGISTRY_P1, help="Nombre del modelo en Model Registry"),
    alias: str = typer.Option("champion", help="Alias del modelo (champion / challenger)"),
    input_path: str = typer.Option("", help="Ruta opcional a archivo CSV o Parquet con datos"),
    output_path: str = typer.Option("", help="Ruta opcional para guardar predicciones"),
) -> None:
    """Ejecuta inferencia con un modelo del Model Registry."""
    model = cargar_modelo_registrado(model_name, alias=alias)

    if input_path and input_path.strip():
        in_p = Path(input_path.strip())
        if not in_p.exists():
            logger.error(f"Archivo de entrada no existe: {in_p}")
            raise typer.Exit(code=1)

        logger.info(f"Cargando datos de entrada desde: {in_p}")
        df = pd.read_parquet(in_p) if in_p.suffix == ".parquet" else pd.read_csv(in_p)
    else:
        logger.info(f"Sin archivo de entrada proporcionado. Usando muestra para {model_name}...")
        df = _obtener_datos_sinteticos(model_name)

    logger.info(f"Ejecutando inferencia sobre {len(df)} registro(s)...")
    predicciones = model.predict(df)
    preds = (
        np.array(predicciones).flatten().tolist()
        if hasattr(predicciones, "flatten")
        else list(predicciones)
    )

    df_out = df.copy()
    df_out["prediccion"] = preds

    logger.success(
        f"Inferencia completada. Muestra de predicciones:\n{df_out[['prediccion']].head(10)}"
    )

    if output_path and output_path.strip():
        out_p = Path(output_path.strip())
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix == ".parquet":
            df_out.to_parquet(out_p, index=False)
        else:
            df_out.to_csv(out_p, index=False)
        logger.success(f"Resultados guardados en: {out_p}")


if __name__ == "__main__":
    app()
