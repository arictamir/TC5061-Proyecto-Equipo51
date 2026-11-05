"""Esquemas de validación de datos con Pydantic para la API de inferencia.

Define contratos de entrada y salida con tipado estricto, validación de rangos
y ejemplos interactivos para OpenAPI / Swagger UI.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class DemandaInput(BaseModel):
    """Esquema de entrada para estimación de demanda horaria de una estación (P1 - CatBoost)."""

    station_id: str = Field(
        ...,
        description="Identificador único de la estación MiBici.",
        examples=["2"],
    )
    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hora del día (formato 24h, rango 0 a 23).",
        examples=[8],
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Día de la semana (0 = Lunes, 6 = Domingo).",
        examples=[1],
    )
    is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indicador binario de fin de semana (1 = Fin de semana, 0 = Día hábil).",
        examples=[0],
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Mes del año (1 a 12).",
        examples=[6],
    )
    lag_1: float = Field(
        ...,
        ge=0.0,
        description="Demanda observada hace 1 hora (t-1).",
        examples=[1.0],
    )
    lag_24: float = Field(
        ...,
        ge=0.0,
        description="Demanda observada hace 24 horas (t-24 / mismo horario día anterior).",
        examples=[2.0],
    )
    lag_168: float = Field(
        ...,
        ge=0.0,
        description="Demanda observada hace 168 horas (t-168 / misma hora semana previa).",
        examples=[2.0],
    )
    rolling_mean_3h: float = Field(
        ...,
        ge=0.0,
        description="Promedio móvil de demanda en la ventana de las últimas 3 horas.",
        examples=[1.5],
    )
    rolling_mean_24h: float = Field(
        ...,
        ge=0.0,
        description="Promedio móvil de demanda en la ventana de las últimas 24 horas.",
        examples=[2.1],
    )
    history_hours: int = Field(
        ...,
        ge=0,
        description="Horas acumuladas de historial operativo registradas para la estación.",
        examples=[500],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "station_id": "2",
                "hour_of_day": 8,
                "day_of_week": 1,
                "is_weekend": 0,
                "month": 6,
                "lag_1": 1.0,
                "lag_24": 2.0,
                "lag_168": 2.0,
                "rolling_mean_3h": 1.5,
                "rolling_mean_24h": 2.1,
                "history_hours": 500,
            }
        }
    )


class DemandaBatchInput(BaseModel):
    """Esquema de entrada para predicción de demanda en lote (múltiples estaciones u horas)."""

    observations: list[DemandaInput] = Field(
        ...,
        min_length=1,
        description="Lista de observaciones con las características de demanda a predecir.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "observations": [
                    {
                        "station_id": "2",
                        "hour_of_day": 8,
                        "day_of_week": 1,
                        "is_weekend": 0,
                        "month": 6,
                        "lag_1": 1.0,
                        "lag_24": 2.0,
                        "lag_168": 2.0,
                        "rolling_mean_3h": 1.5,
                        "rolling_mean_24h": 2.1,
                        "history_hours": 500,
                    },
                    {
                        "station_id": "3",
                        "hour_of_day": 19,
                        "day_of_week": 5,
                        "is_weekend": 1,
                        "month": 6,
                        "lag_1": 4.0,
                        "lag_24": 5.0,
                        "lag_168": 6.0,
                        "rolling_mean_3h": 4.2,
                        "rolling_mean_24h": 3.8,
                        "history_hours": 1200,
                    },
                ]
            }
        }
    )


class DemandaOutput(BaseModel):
    """Esquema de respuesta para la estimación de demanda horaria."""

    station_id: str = Field(
        ...,
        description="Identificador único de la estación consultada.",
        examples=["2"],
    )
    predicted_demand: float = Field(
        ...,
        ge=0.0,
        description="Demanda horaria predicha (número estimado de viajes iniciados, >= 0.0).",
        examples=[3.42],
    )
    model_name: str = Field(
        default="mibici-equipo51-p1-demanda",
        description="Nombre del modelo registrado en MLflow Model Registry.",
    )
    model_alias: str = Field(
        default="champion",
        description="Alias del modelo utilizado para servir la inferencia.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Estampa de tiempo ISO 8601 en UTC del momento de la inferencia.",
    )


class DemandaBatchOutput(BaseModel):
    """Esquema de respuesta para predicciones de demanda en lote."""

    predictions: list[DemandaOutput] = Field(
        ...,
        description="Lista con las predicciones individuales generadas.",
    )
    total_observations: int = Field(
        ...,
        description="Número total de registros procesados.",
        examples=[2],
    )


class HealthResponse(BaseModel):
    """Esquema de respuesta para la verificación de salud del servicio (/health)."""

    status: str = Field(
        ...,
        description="Estado operativo general de la API (ej. 'ok', 'degraded').",
        examples=["ok"],
    )
    model_loaded: bool = Field(
        ...,
        description="Indica si el modelo Champion de MLflow está cargado y listo.",
        examples=[True],
    )
    model_name: str = Field(
        default="mibici-equipo51-p1-demanda",
        description="Nombre del modelo principal registrado.",
    )
    model_alias: str = Field(
        default="champion",
        description="Alias o versión activa del modelo servido.",
    )
    tracking_uri: str = Field(
        ...,
        description="URI del servidor de MLflow Tracking configurada.",
        examples=["http://host.containers.internal:5000"],
    )
    version: str = Field(
        default="1.0.0",
        description="Versión del microservicio de inferencia.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Estampa de tiempo actual en UTC.",
    )
