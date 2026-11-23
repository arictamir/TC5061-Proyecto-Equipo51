"""Escenario de pruebas de carga con Locust para la API de inferencia de MiBici.

Simula tráfico concurrente realista de usuarios y estaciones IoT de MiBici:
1. Healthcheck / Keep-alive sintético (10% del tráfico total).
2. Inferencia de demanda unitaria en tiempo real (75% del tráfico total).
3. Inferencia de demanda en lote para sincronización de racks (15% del tráfico total).
"""


import random

from locust import HttpUser, between, task
from tests.load.payload_factory import generate_sample_prediction_payload


class MiBiciLoadTestUser(HttpUser):
    """Usuario simulado que genera tráfico contra el microservicio de inferencia."""

    wait_time = between(0.1, 0.5)

    @task(1)
    def check_health(self) -> None:
        """Monitoreo sintético y verificación de salud del microservicio."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") in ("ok", "degraded"):
                    response.success()
                else:
                    response.failure(f"Estado de salud inesperado: {data}")
            else:
                response.failure(f"HTTP {response.status_code}: Fallo en healthcheck")

    @task(8)
    def predict_single_demand(self) -> None:
        """Inferencia unitaria de demanda horaria para una estación."""
        payload = generate_sample_prediction_payload()
        headers = {"Content-Type": "application/json"}
        with self.client.post("/predict", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "predicted_demand" in data and isinstance(data["predicted_demand"], (int, float)):
                    response.success()
                else:
                    response.failure(f"Respuesta sin predicted_demand válido: {data}")
            else:
                response.failure(f"HTTP {response.status_code}: Error en /predict")

    @task(2)
    def predict_batch_demand(self) -> None:
        """Inferencia por lotes (5 a 10 estaciones) para centros y hubs de distribución."""
        batch_size = random.randint(5, 10)
        observations = [generate_sample_prediction_payload() for _ in range(batch_size)]
        payload = {"observations": observations}
        headers = {"Content-Type": "application/json"}
        with self.client.post("/predict/batch", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("total_observations") == batch_size and "predictions" in data:
                    response.success()
                else:
                    response.failure(f"Respuesta de batch inconsistente: {data}")
            else:
                response.failure(f"HTTP {response.status_code}: Error en /predict/batch")
