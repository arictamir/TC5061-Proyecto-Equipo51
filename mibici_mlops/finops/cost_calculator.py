"""Módulo de modelado y estimación de costos FinOps para inferencia en la nube.

Entregable: Microservicios, CI/CD y FinOps
Responsable: Gaby
"""

from typing import Any

# TODO (Gaby): Definir precios de referencia para instancias / servicios en la nube (ej. AWS, GCP)


def calculate_cost_per_inference(*args: Any, **kwargs: Any) -> float:
    """Calcula el costo estimado por inferencia individual.

    TODO (Gaby): Implementar cálculo a partir de costos de infraestructura y peticiones atendidas.
    """
    return 0.0


def calculate_cost_per_million_requests(*args: Any, **kwargs: Any) -> float:
    """Calcula el costo estimado por millón de inferencias.

    TODO (Gaby): Implementar cálculo extrapolado por 1M de peticiones.
    """
    return 0.0


def estimate_cloud_costs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Estima y compara costos entre diferentes configuraciones de infraestructura.

    TODO (Gaby): Implementar comparativa y potenciales ahorros por optimizaciones.
    """
    return {}
