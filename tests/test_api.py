"""Pruebas unitarias y de integración para la API REST de inferencia de FastAPI.

Responsables: Gabriela (Gaby) e Ingrid
"""

from fastapi.testclient import TestClient

from mibici_mlops.api.main import app

client = TestClient(app)


# TODO - Gaby - Probar endpoint GET / para verificar metadatos de la API
def test_root_endpoint_returns_welcome_and_docs_link():
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


# TODO - Gaby - Probar endpoint GET /health para verificar estado saludable
def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


# TODO - Gaby - Probar endpoint POST /predict con payload válido y respuesta esperada
def test_predict_endpoint_valid_payload():
    payload = {"station_id": "2"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_demand" in response.json()


# TODO - Ingrid - Probar endpoint POST /predict ante payload inválido (error de validación Pydantic 422)
def test_predict_endpoint_invalid_payload_returns_422():
    payload = {"campo_inexistente": 123}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
