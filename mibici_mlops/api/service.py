"""Capa de servicio para carga de modelos desde MLflow Model Registry e inferencia.

Gestiona la conexión con MLflow Model Registry (o fallback local empaquetado),
el preprocesamiento estricto de las 11 características de CatBoost y la inferencia
tanto en tiempo real (unitaria) como en lote (batch).
"""

from collections.abc import Sequence
import os
from pathlib import Path
from typing import Any

from loguru import logger
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from mibici_mlops.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY_P1, MODELS_DIR

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


class ModelService:
    """Servicio singleton para gestionar el ciclo de vida del modelo de MLflow."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_name: str = MODEL_REGISTRY_P1
        self.model_alias: str = "champion"
        self.tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
        self.model_version: str | None = None

    def load_model(self, model_name: str | None = None, alias: str = "champion") -> None:
        """Carga el modelo Champion de P1 desde MLflow Model Registry o desde artefactos locales.

        Estrategia de carga dual:
        1. Intento primario: Carga desde el Model Registry de MLflow utilizando la URI configurada
           (ej. `sqlite:///mlflow.db` local o `http://host.containers.internal:5000` en contenedor).
        2. Intento secundario (fallback): Carga desde artefactos locales empaquetados en `models/p1_champion`.
        """
        self.model_name = model_name or MODEL_REGISTRY_P1
        self.model_alias = alias
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
        self.model = None

        logger.info(
            f"[ModelService] Conectando a MLflow tracking URI='{self.tracking_uri}' "
            f"para modelo='{self.model_name}' alias='@{self.model_alias}'..."
        )

        # 1. Intentar carga desde MLflow Model Registry
        model_uri = f"models:/{self.model_name}@{self.model_alias}"
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.model = mlflow.pyfunc.load_model(model_uri)
            logger.success(
                f"[ModelService] Modelo cargado exitosamente desde Model Registry ({model_uri})"
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[ModelService] No fue posible conectar con Model Registry ({model_uri}): {exc}. "
                "Iniciando verificación de artefactos locales empaquetados..."
            )

        # 2. Fallback a modelo local empaquetado (offline / contenedor standalone)
        local_candidates: list[Path] = [
            MODELS_DIR / "p1_champion",
            MODELS_DIR / "p1",
            Path("models/p1_champion"),
            Path("models/p1"),
        ]
        for candidate in local_candidates:
            if candidate.exists() and (candidate / "MLmodel").exists():
                try:
                    logger.info(
                        f"[ModelService] Cargando modelo local empaquetado desde: {candidate.resolve()}"
                    )
                    self.model = mlflow.pyfunc.load_model(str(candidate))
                    logger.success(
                        f"[ModelService] Modelo empaquetado cargado exitosamente desde {candidate}"
                    )
                    return
                except Exception as local_exc:  # noqa: BLE001
                    logger.warning(
                        f"[ModelService] Falló carga de candidato local '{candidate}': {local_exc}"
                    )

        logger.warning(
            "[ModelService] No se pudo cargar el modelo en memoria. La API continuará en modo degradado."
        )

    def prepare_features(self, payload: Any) -> pd.DataFrame:
        """Transforma datos del esquema Pydantic en DataFrame con los tipos exactos de CatBoost.

        Garantiza los tipos según la firma inferida registrada en MLflow:
        - station_id, hour_of_day, day_of_week, month -> string
        - is_weekend -> int32
        - history_hours -> int64
        - lag_1, lag_24, lag_168, rolling_mean_3h, rolling_mean_24h -> float64
        """
        if isinstance(payload, (list, tuple)):
            records = [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in payload]
        elif hasattr(payload, "model_dump"):
            records = [payload.model_dump()]
        elif isinstance(payload, dict):
            records = [payload]
        else:
            records = [dict(payload)]

        df = pd.DataFrame(records)

        missing = set(FEATURE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Faltan características requeridas para inferencia de P1: {sorted(missing)}"
            )

        # Conversión estricta de tipos de datos para coincidir con la firma de MLflow
        for col in CATEGORICAL_COLUMNS:
            df[col] = df[col].astype(str)

        df["is_weekend"] = df["is_weekend"].astype(np.int32)
        df["history_hours"] = df["history_hours"].astype(np.int64)

        float_cols = ["lag_1", "lag_24", "lag_168", "rolling_mean_3h", "rolling_mean_24h"]
        for col in float_cols:
            df[col] = df[col].astype(np.float64)

        return df[FEATURE_COLUMNS]

    def predict(self, payload: Any) -> float:
        """Ejecuta inferencia para una observación individual y asegura demanda no negativa (>= 0)."""
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise RuntimeError(
                    "El modelo no está disponible en memoria. Verifique la conexión a MLflow o los artefactos locales."
                )

        features_df = self.prepare_features(payload)
        raw_pred = self.model.predict(features_df)

        if hasattr(raw_pred, "item"):
            pred_value = float(raw_pred.item())
        elif isinstance(raw_pred, (list, np.ndarray, pd.Series, Sequence)):
            pred_value = float(raw_pred[0])
        else:
            pred_value = float(raw_pred)

        # Regla de negocio: la demanda de viajes no puede ser negativa
        return max(0.0, round(pred_value, 4))

    def predict_batch(self, payload_list: list[Any]) -> list[float]:
        """Ejecuta inferencia en lote (batch) optimizada sobre múltiples observaciones."""
        if not payload_list:
            return []

        if self.model is None:
            self.load_model()
            if self.model is None:
                raise RuntimeError(
                    "El modelo no está disponible en memoria. Verifique la conexión a MLflow o los artefactos locales."
                )

        features_df = self.prepare_features(payload_list)
        raw_preds = self.model.predict(features_df)

        if isinstance(raw_preds, (np.ndarray, pd.Series, list)):
            return [max(0.0, round(float(val), 4)) for val in raw_preds]

        return [max(0.0, round(float(raw_preds), 4))]


# Instancia singleton del servicio
model_service = ModelService()
