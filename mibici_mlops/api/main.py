"""Aplicación principal de FastAPI para el servicio de inferencia de MiBici.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from loguru import logger

from mibici_mlops.api.schemas import (
    DemandaBatchInput,
    DemandaBatchOutput,
    DemandaInput,
    DemandaOutput,
    HealthResponse,
)
from mibici_mlops.api.service import model_service
from mibici_mlops.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY_P1


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ciclo de vida de la aplicación para inicializar el modelo al arranque y liberar recursos al cierre."""
    logger.info("Iniciando API de inferencia MiBici (FastAPI)...")
    try:
        # Intento de precargar el modelo Champion de P1
        model_service.load_model(model_name=MODEL_REGISTRY_P1, alias="champion")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"El modelo no pudo precargarse durante el inicio (se reintentará en la primera petición): {exc}"
        )
    yield
    logger.info("Cerrando API de inferencia MiBici...")


app = FastAPI(
    title="MiBici MLOps - API de Inferencia de Demanda",
    description=(
        "Microservicio REST desarrollado con FastAPI para servir predicciones en tiempo real "
        "y por lotes del modelo Champion de MiBici (P1: Regresión de Demanda Horaria con CatBoost). "
        "Diseñado para ejecutarse en contenedores compatibles con el estándar OCI (Docker / Podman)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get(
    "/",
    tags=["General"],
    summary="Información raíz del servicio",
    description="Devuelve el estado general del microservicio y los enlaces a la documentación interactiva.",
)
async def root() -> dict[str, str]:
    return {
        "service": "MiBici MLOps Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "docs_url": "/docs",
        "health": "/health",
        "health_check": "/health",
        "predict_endpoint": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoreo"],
    summary="Verificación de estado y salud operativa",
    description="Comprueba si el microservicio está activo, si el modelo está cargado y la URI de MLflow configurada.",
)
async def health_check() -> HealthResponse:
    is_loaded = model_service.model is not None
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=model_service.model_name or MODEL_REGISTRY_P1,
        model_alias=model_service.model_alias or "champion",
        tracking_uri=model_service.tracking_uri or MLFLOW_TRACKING_URI,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=DemandaOutput,
    status_code=status.HTTP_200_OK,
    tags=["Inferencia"],
    summary="Predicción de demanda horaria individual",
    description="Recibe las 11 características de una estación y retorna el pronóstico de demanda horaria estimada.",
)
async def predict_demand(payload: DemandaInput) -> DemandaOutput:
    try:
        prediction = model_service.predict(payload)
        return DemandaOutput(
            station_id=str(payload.station_id),
            predicted_demand=prediction,
            model_name=model_service.model_name or MODEL_REGISTRY_P1,
            model_alias=model_service.model_alias or "champion",
        )
    except Exception as exc:
        logger.error(f"Error durante la inferencia: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo interno en el motor de inferencia: {exc}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=DemandaBatchOutput,
    status_code=status.HTTP_200_OK,
    tags=["Inferencia"],
    summary="Predicción de demanda horaria en lote",
    description="Recibe una lista de observaciones y devuelve las predicciones correspondientes.",
)
async def predict_demand_batch(payload: DemandaBatchInput) -> DemandaBatchOutput:
    try:
        predictions = model_service.predict_batch(payload.observations)
        results = [
            DemandaOutput(
                station_id=str(obs.station_id),
                predicted_demand=pred,
                model_name=model_service.model_name or MODEL_REGISTRY_P1,
                model_alias=model_service.model_alias or "champion",
            )
            for obs, pred in zip(payload.observations, predictions, strict=False)
        ]
        return DemandaBatchOutput(
            predictions=results,
            total_observations=len(results),
        )
    except Exception as exc:
        logger.error(f"Error durante la inferencia por lote: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo interno en la inferencia por lote: {exc}",
        ) from exc
