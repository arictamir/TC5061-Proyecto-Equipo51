"""Pruebas unitarias para el modelo de costos FinOps y escenarios de Locust.
"""

import subprocess
import sys

import pytest
from tests.load.payload_factory import generate_sample_prediction_payload

from mibici_mlops.finops.cost_calculator import (
    CLOUD_PRICING_CATALOG,
    calculate_cost_per_inference,
    calculate_cost_per_inference_from_throughput,
    calculate_cost_per_million_requests,
    compare_cloud_providers,
    compare_cloud_providers_from_throughput,
    estimate_inference_count,
    simulate_finops_optimizations,
)


def test_calculate_cost_per_inference_exact_values() -> None:
    """Verifica el cálculo de costo unitario por inferencia con valores analíticos conocidos."""
    # Instancia de $0.05 / hr atendiendo 50 RPS al 100% de utilización
    # Infe / hora = 50 * 3600 = 180,000
    # Costo = 0.05 / 180,000 = 2.777777777777778e-07
    cost = calculate_cost_per_inference(
        hourly_rate_usd=0.05,
        requests_per_second=50.0,
        system_utilization_ratio=1.0,
    )
    expected = 0.05 / (50.0 * 3600.0)
    assert pytest.approx(cost, rel=1e-6) == expected


def test_calculate_cost_per_inference_validation_errors() -> None:
    """Verifica que se lancen excepciones con argumentos fuera de rango."""
    with pytest.raises(ValueError, match="tarifa horaria"):
        calculate_cost_per_inference(hourly_rate_usd=-0.01, requests_per_second=10.0)

    with pytest.raises(ValueError, match="peticiones por segundo"):
        calculate_cost_per_inference(hourly_rate_usd=0.05, requests_per_second=0.0)

    with pytest.raises(ValueError, match="ratio de utilización"):
        calculate_cost_per_inference(hourly_rate_usd=0.05, requests_per_second=10.0, system_utilization_ratio=1.5)


def test_calculate_cost_per_million_requests() -> None:
    """Verifica la extrapolación de costo unitario a 1 millón de peticiones."""
    cost_per_inf = 0.0000004
    cost_1m = calculate_cost_per_million_requests(cost_per_inf)
    assert pytest.approx(cost_1m, rel=1e-6) == 0.40

    with pytest.raises(ValueError, match="no puede ser negativo"):
        calculate_cost_per_million_requests(-1.0)


def test_compare_cloud_providers() -> None:
    """Valida la tabla comparativa de costos entre proveedores nube."""
    comparison = compare_cloud_providers(requests_per_second=60.0, system_utilization_ratio=0.75)
    assert len(comparison) == len(CLOUD_PRICING_CATALOG)

    for item in comparison:
        assert "provider" in item
        assert "instance_type" in item
        assert "cost_per_inference_usd" in item
        assert "cost_per_million_usd" in item
        assert item["cost_per_million_usd"] > 0.0

    # Debe estar ordenado de menor a mayor costo por millón
    costs = [item["cost_per_million_usd"] for item in comparison]
    assert costs == sorted(costs)


def test_compare_cloud_providers_from_throughput() -> None:
    """Verifica la comparación de costos usando inferencias por segundo."""
    throughput = 569.05
    utilization = 0.70
    hours_per_month = 730.0

    comparison = compare_cloud_providers_from_throughput(
        inferences_per_second=throughput,
        system_utilization_ratio=utilization,
        hours_per_month=hours_per_month,
    )

    assert len(comparison) == len(CLOUD_PRICING_CATALOG)

    for item in comparison:
        expected_unit_cost = item["hourly_rate_usd"] / (
            throughput * 3600.0 * utilization
        )

        assert item["cost_per_inference_usd"] == pytest.approx(
            expected_unit_cost
        )
        assert item["cost_per_million_usd"] == pytest.approx(
            expected_unit_cost * 1_000_000
        )
        assert item["monthly_cost_usd"] == pytest.approx(
            round(item["hourly_rate_usd"] * hours_per_month, 2)
        )

    costs = [item["cost_per_million_usd"] for item in comparison]
    assert costs == sorted(costs)

    with pytest.raises(ValueError, match="horas mensuales"):
        compare_cloud_providers_from_throughput(
            inferences_per_second=throughput,
            hours_per_month=0,
        )


def test_simulate_finops_optimizations() -> None:
    """Valida la simulación de ahorros proyectados por optimizaciones técnicas."""
    sim = simulate_finops_optimizations(
        baseline_hourly_rate=0.0416,
        baseline_rps=40.0,
        system_utilization_ratio=0.70,
        annual_requests=10_000_000.0,
    )

    assert sim["baseline_annual_cost_usd"] > sim["optimized_annual_cost_usd"]
    assert sim["annual_savings_usd"] > 0.0
    assert 40.0 <= sim["savings_percentage"] <= 90.0
    assert sim["optimized_cost_per_million_usd"] < sim["baseline_cost_per_million_usd"]


def test_locust_payload_generator() -> None:
    """Valida que el generador de payload para Locust cumpla estrictamente el esquema de FastAPI."""
    payload = generate_sample_prediction_payload()

    # Campos requeridos por DemandaInput
    required_keys = [
        "station_id",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "month",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_3h",
        "rolling_mean_24h",
        "history_hours",
    ]
    for key in required_keys:
        assert key in payload, f"Falta clave {key} en payload generado"

    assert isinstance(payload["station_id"], str)
    assert 0 <= payload["hour_of_day"] <= 23
    assert 0 <= payload["day_of_week"] <= 6
    assert payload["is_weekend"] in (0, 1)
    assert 1 <= payload["month"] <= 12
    assert payload["history_hours"] >= 24



def test_locust_user_class_attributes() -> None:
    """Valida las tareas de Locust en un proceso independiente."""
    script = """
from tests.load.locustfile import MiBiciLoadTestUser

user = MiBiciLoadTestUser
assert hasattr(user, "tasks")
assert len(user.tasks) >= 3
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"Falló la validación de Locust:\n{result.stdout}\n{result.stderr}"
    )


def test_estimate_inference_count() -> None:
    """Verifica la estimación de inferencias individuales y por lotes."""
    estimated = estimate_inference_count(
        single_requests=11745,
        batch_requests=2924,
        average_batch_size=7.5,
    )

    assert estimated == 33675.0

    # Sin solicitudes por lote, cada petición equivale a una inferencia.
    assert estimate_inference_count(100, 0) == 100.0

    # Verifica que se rechacen entradas inválidas.
    with pytest.raises(ValueError, match="solicitudes"):
        estimate_inference_count(-1, 10)

    with pytest.raises(ValueError, match="tamaño promedio"):
        estimate_inference_count(100, 10, average_batch_size=0)


def test_calculate_cost_per_inference_from_throughput() -> None:
    """Verifica el costo estimado utilizando inferencias por segundo."""
    hourly_rate = 0.0416
    throughput = 569.05
    utilization = 0.70

    cost = calculate_cost_per_inference_from_throughput(
        hourly_rate_usd=hourly_rate,
        inferences_per_second=throughput,
        system_utilization_ratio=utilization,
    )

    expected = hourly_rate / (throughput * 3600 * utilization)

    assert cost == pytest.approx(expected)
    assert calculate_cost_per_million_requests(cost) == pytest.approx(
        expected * 1_000_000
    )

    with pytest.raises(ValueError, match="tarifa horaria"):
        calculate_cost_per_inference_from_throughput(-0.01, 100.0)

    with pytest.raises(ValueError, match="inferencias por segundo"):
        calculate_cost_per_inference_from_throughput(0.05, 0.0)

    with pytest.raises(ValueError, match="ratio de utilización"):
        calculate_cost_per_inference_from_throughput(0.05, 100.0, 1.5)
