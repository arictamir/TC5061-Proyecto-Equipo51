"""Aplicación principal de FastAPI para el servicio de inferencia de MiBici.

Responsables conjuntos: Ariadna (Ari) y Gabriela (Gaby)
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from mibici_mlops.api.schemas import DemandaInput, DemandaOutput, HealthResponse
from mibici_mlops.api.service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ciclo de vida de la aplicación para inicializar y liberar recursos."""
    logger.info("Iniciando API de inferencia MiBici...")
    # TODO - Gaby - Cargar el modelo Champion en el startup de la aplicación
    # model_service.load_model(model_name="mibici-equipo51-p1-demanda", alias="champion")
    yield
    logger.info("Cerrando API de inferencia MiBici...")


# TODO - Ari - Configurar metadatos de la aplicación FastAPI (title, description, version, docs_url)
app = FastAPI(
    title="MiBici MLOps Inference API",
    description="API REST para servir predicciones del modelo Champion de MiBici empaquetado en contenedor OCI.",
    version="1.0.0",
    lifespan=lifespan,
)


# TODO - Ari - Implementar endpoint raíz con información general y enlaces a la documentación
@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    return {
        "message": "MiBici MLOps Inference API",
        "docs": "/docs",
        "health": "/health",
    }


# TODO - Ari - Implementar endpoint GET /health para verificar estado y modelo cargado
@app.get("/health", response_model=HealthResponse, tags=["Monitoreo"])
async def health_check() -> HealthResponse:
    # TODO - Ari - Devolver estado del servicio, estado del modelo y tracking URI configurada
    is_loaded = model_service.model is not None
    return HealthResponse(status="ok" if is_loaded else "waiting_model")


# TODO - Gaby - Implementar endpoint POST /predict para inferencia en tiempo real
@app.post("/predict", response_model=DemandaOutput, tags=["Inferencia"])
async def predict_demand(payload: DemandaInput) -> DemandaOutput:
    # TODO - Gaby - Validar payload con Pydantic, invocar model_service.predict y devolver DemandaOutput
    prediction = model_service.predict(payload)
    return DemandaOutput(predicted_demand=prediction)


# TODO - Gaby - Implementar endpoint POST /predict/batch para inferencia en lote
@app.post("/predict/batch", response_model=list[DemandaOutput], tags=["Inferencia"])
async def predict_demand_batch(payload_list: list[DemandaInput]) -> list[DemandaOutput]:
    # TODO - Gaby - Procesar lista de observaciones y devolver lista de predicciones
    predictions = model_service.predict_batch(payload_list)
    return [DemandaOutput(predicted_demand=p) for p in predictions]
