"""Pruebas unitarias básicas para el módulo de FinOps.

Entregable: Microservicios, CI/CD y FinOps
Responsable: Gaby
"""

from mibici_mlops.finops.cost_calculator import (
    calculate_cost_per_inference,
    calculate_cost_per_million_requests,
    estimate_cloud_costs,
)


def test_finops_functions_defined() -> None:
    """Verifica que las funciones principales de FinOps estén definidas.

    TODO (Gaby): Agregar pruebas unitarias específicas con valores numéricos y casos de prueba.
    """
    assert callable(calculate_cost_per_inference)
    assert callable(calculate_cost_per_million_requests)
    assert callable(estimate_cloud_costs)
