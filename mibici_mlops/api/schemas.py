"""Esquemas de validación de datos con Pydantic para la API de inferencia.

Responsable: Ariadna (Ari)
"""

from pydantic import BaseModel, Field


# TODO - Ari - Definir el modelo Pydantic para los datos de entrada individuales de P1 (Demanda)
# Debe validar las 11 variables requeridas: station_id, hour_of_day, day_of_week,
# is_weekend, month, lag_1, lag_24, lag_168, rolling_mean_3h, rolling_mean_24h, history_hours.
# Incluir validaciones de rango (e.g. Field(ge=0, le=23) para horas) y ejemplos para Swagger/OpenAPI.
class DemandaInput(BaseModel):
    # TODO - Ari - Agregar los campos con sus tipos y restricciones (usar Field)
    station_id: str = Field(..., description="Identificador único de la estación", examples=["2"])


# TODO - Ari - Definir el modelo Pydantic para petición en lote (batch) de predicciones
class DemandaBatchInput(BaseModel):
    # TODO - Ari - Lista de observaciones DemandaInput
    observations: list[DemandaInput] = Field(..., description="Lista de registros a predecir")


# TODO - Ari - Definir el modelo Pydantic para la salida de predicción
# Debe devolver al menos: predicted_demand (float), station_id, model_name, model_alias
class DemandaOutput(BaseModel):
    # TODO - Ari - Agregar campos de respuesta
    predicted_demand: float = Field(..., ge=0.0, description="Demanda horaria estimada")


# TODO - Ari - Definir modelo para respuesta de estado de salud (/health)
class HealthResponse(BaseModel):
    # TODO - Ari - status: str, model_loaded: bool, tracking_uri: str
    status: str = Field("ok", description="Estado operativo del servicio")
