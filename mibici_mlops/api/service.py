"""Capa de servicio para carga de modelos desde MLflow Model Registry e inferencia.

Responsable: Gabriela (Gaby)
"""

from typing import Any

from loguru import logger
import pandas as pd


class ModelService:
    """Servicio singleton para gestionar el ciclo de vida del modelo de MLflow."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_name: str = ""
        self.model_alias: str = "champion"
        self.tracking_uri: str = ""

    # TODO - Gaby - Implementar método para cargar el modelo desde MLflow Model Registry
    # Configurar MLFLOW_TRACKING_URI desde variable de entorno (ej. http://host.containers.internal:5000)
    # Cargar usando uri: f"models:/{model_name}@{alias}" mediante mlflow.pyfunc.load_model
    # Manejar fallback seguro en caso de que el tracking server no esté disponible de inmediato.
    def load_model(self, model_name: str, alias: str = "champion") -> None:
        logger.info(f"Cargando modelo {model_name} con alias {alias}...")

    # TODO - Gaby - Implementar método para transformar el esquema Pydantic en DataFrame
    # con los tipos de datos exactos que espera el modelo CatBoost de P1.
    def prepare_features(self, payload: Any) -> pd.DataFrame:
        return pd.DataFrame()

    # TODO - Gaby - Implementar método de predicción individual y acotamiento no negativo (>= 0)
    def predict(self, payload: Any) -> float:
        return 0.0

    # TODO - Gaby - Implementar método de predicción en lote (batch)
    def predict_batch(self, payload_list: list[Any]) -> list[float]:
        return []


# Instancia global del servicio
model_service = ModelService()
