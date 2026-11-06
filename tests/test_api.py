"""Pruebas unitarias y de integración para la API REST de inferencia de FastAPI.
"""

from collections.abc import Generator

from fastapi.testclient import TestClient
import numpy as np
import pytest

from mibici_mlops.api.main import app
from mibici_mlops.api.schemas import DemandaInput
from mibici_mlops.api.service import ModelService, model_service


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Cliente de pruebas con soporte completo de ciclo de vida (lifespan) de FastAPI."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint_returns_welcome_and_docs_link(client: TestClient):
    """Prueba endpoint GET / para verificar metadatos de la API."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "docs_url" in data
    assert "service" in data
    assert "version" in data


def test_health_check_endpoint(client: TestClient):
    """Prueba endpoint GET /health para verificar estado operativo y modelo cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "mibici-equipo51-p1-demanda"
    assert data["model_alias"] == "champion"


def test_prepare_features_dtypes_and_structure():
    """Prueba unitaria para verificar que prepare_features genere tipos compatibles con MLflow."""
    payload = DemandaInput(
        station_id="2",
        hour_of_day=8,
        day_of_week=1,
        is_weekend=0,
        month=6,
        lag_1=1.0,
        lag_24=2.0,
        lag_168=3.0,
        rolling_mean_3h=1.5,
        rolling_mean_24h=2.2,
        history_hours=168,
    )
    df = model_service.prepare_features(payload)
    assert df.shape == (1, 11)
    assert df["station_id"].dtype == object
    assert df["is_weekend"].dtype == np.int32
    assert df["history_hours"].dtype == np.int64
    assert df["lag_1"].dtype == np.float64


def test_prepare_features_rejects_missing_columns():
    """Comprueba que el servicio rechace entradas incompletas fuera de Pydantic."""
    service = ModelService()

    with pytest.raises(ValueError, match="Faltan características"):
        service.prepare_features({"station_id": "2"})


def test_predict_batch_empty_returns_empty_list():
    """Comprueba que un lote vacío no invoque innecesariamente el modelo."""
    assert ModelService().predict_batch([]) == []


def test_predict_endpoint_valid_payload(client: TestClient):
    """Prueba endpoint POST /predict con payload válido y respuesta de inferencia."""
    payload = {
        "station_id": "2",
        "hour_of_day": 8,
        "day_of_week": 1,
        "is_weekend": 0,
        "month": 6,
        "lag_1": 1.0,
        "lag_24": 2.0,
        "lag_168": 3.0,
        "rolling_mean_3h": 1.5,
        "rolling_mean_24h": 2.2,
        "history_hours": 168.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_demand" in data
    assert data["predicted_demand"] >= 0.0
    assert data["station_id"] == "2"
    assert data["model_alias"] == "champion"


def test_predict_batch_endpoint_valid_payload(client: TestClient):
    """Prueba endpoint POST /predict/batch con lote de observaciones."""
    payload = {
        "observations": [
            {
                "station_id": "2",
                "hour_of_day": 8,
                "day_of_week": 1,
                "is_weekend": 0,
                "month": 6,
                "lag_1": 1.0,
                "lag_24": 2.0,
                "lag_168": 3.0,
                "rolling_mean_3h": 1.5,
                "rolling_mean_24h": 2.2,
                "history_hours": 168.0,
            },
            {
                "station_id": "3",
                "hour_of_day": 18,
                "day_of_week": 4,
                "is_weekend": 0,
                "month": 6,
                "lag_1": 5.0,
                "lag_24": 4.0,
                "lag_168": 6.0,
                "rolling_mean_3h": 4.5,
                "rolling_mean_24h": 3.8,
                "history_hours": 168.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_observations"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["predicted_demand"] >= 0.0


# TODO - Ingrid - Probar endpoint POST /predict ante payload inválido (error de validación Pydantic 422)
def test_predict_endpoint_invalid_payload_returns_422(client: TestClient):
    payload = {"campo_inexistente": 123}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
