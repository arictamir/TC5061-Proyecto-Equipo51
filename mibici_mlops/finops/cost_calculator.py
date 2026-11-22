"""Módulo FinOps para modelado y estimación de costos de inferencia en la nube.

Proporciona herramientas matemáticas y comparativas para:
- Estimar el costo unitario por inferencia ($/inferencia).
- Estimar el costo por millón de peticiones ($/1M requests).
- Comparar costos entre proveedores de infraestructura (AWS EC2, AWS Fargate, GCP Compute Engine, GCP Cloud Run).
- Evaluar el impacto financiero de optimizaciones técnicas (batching, rightsizing, caching, spot instances).
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CloudInstancePricing:
    """Estructura de precios de referencia para cómputo en la nube."""

    provider: str
    service_name: str
    instance_type: str
    vcpus: int
    memory_gb: float
    hourly_rate_usd: float
    pricing_model: str = "On-Demand"


# Catálogo de tarifas públicas estándar de referencia (Región us-east-1 / us-central1)
CLOUD_PRICING_CATALOG: list[CloudInstancePricing] = [
    CloudInstancePricing(
        provider="AWS",
        service_name="EC2",
        instance_type="t3.medium",
        vcpus=2,
        memory_gb=4.0,
        hourly_rate_usd=0.0416,
        pricing_model="On-Demand",
    ),
    CloudInstancePricing(
        provider="AWS",
        service_name="EC2",
        instance_type="c6i.large",
        vcpus=2,
        memory_gb=4.0,
        hourly_rate_usd=0.0850,
        pricing_model="Compute-Optimized",
    ),
    CloudInstancePricing(
        provider="AWS",
        service_name="ECS Fargate",
        instance_type="1 vCPU / 1GB RAM",
        vcpus=1,
        memory_gb=1.0,
        hourly_rate_usd=0.0449,  # $0.04048/vCPU-h + $0.004445/GB-h
        pricing_model="Serverless Container",
    ),
    CloudInstancePricing(
        provider="GCP",
        service_name="Compute Engine",
        instance_type="e2-standard-2",
        vcpus=2,
        memory_gb=8.0,
        hourly_rate_usd=0.0670,
        pricing_model="On-Demand",
    ),
    CloudInstancePricing(
        provider="GCP",
        service_name="Cloud Run",
        instance_type="1 vCPU / 1GB RAM",
        vcpus=1,
        memory_gb=1.0,
        hourly_rate_usd=0.0520,  # Tarifa prorrateada por uso activo + invocaciones
        pricing_model="Serverless Pay-per-Request",
    ),
]


def estimate_inference_count(
    single_requests: int,
    batch_requests: int,
    average_batch_size: float = 7.5,
) -> float:
    """Estima inferencias a partir de solicitudes individuales y por lote."""
    if single_requests < 0 or batch_requests < 0:
        raise ValueError("El número de solicitudes no puede ser negativo.")

    if average_batch_size <= 0:
        raise ValueError("El tamaño promedio del lote debe ser positivo.")

    return single_requests + batch_requests * average_batch_size


def calculate_cost_per_inference(
    hourly_rate_usd: float,
    requests_per_second: float,
    system_utilization_ratio: float = 0.70,
) -> float:
    """Calcula el costo individual estimado por inferencia servida.

    Fórmula:
        Infe_por_hora = RPS * 3600 * system_utilization_ratio
        Costo_por_inferencia = hourly_rate_usd / Infe_por_hora

    Args:
        hourly_rate_usd: Tarifa por hora de la instancia o recurso ($ USD).
        requests_per_second: Tasa promedio de peticiones atendidas por segundo (RPS).
        system_utilization_ratio: Factor de utilización real del sistema (por defecto 0.70 o 70%,
            considerando variaciones estacionales de tráfico diurno/nocturno de MiBici).

    Returns:
        Costo unitario por inferencia en dólares USD.
    """
    if hourly_rate_usd <= 0:
        msg = "La tarifa horaria debe ser mayor a 0."
        raise ValueError(msg)
    if requests_per_second <= 0:
        msg = "Las peticiones por segundo (RPS) deben ser mayores a 0."
        raise ValueError(msg)
    if not (0 < system_utilization_ratio <= 1.0):
        msg = "El ratio de utilización debe encontrarse en el rango (0.0, 1.0]."
        raise ValueError(msg)

    effective_inferences_per_hour = requests_per_second * 3600.0 * system_utilization_ratio
    return hourly_rate_usd / effective_inferences_per_hour


def calculate_cost_per_inference_from_throughput(
    hourly_rate_usd: float,
    inferences_per_second: float,
    system_utilization_ratio: float = 0.70,
) -> float:
    """Estima el costo por inferencia a partir del throughput de inferencias.

    El factor de utilización representa la fracción promedio de la
    capacidad de referencia utilizada durante la operación.
    """
    if hourly_rate_usd <= 0:
        raise ValueError("La tarifa horaria debe ser mayor a 0.")

    if inferences_per_second <= 0:
        raise ValueError("Las inferencias por segundo deben ser mayores a 0.")

    if not (0 < system_utilization_ratio <= 1.0):
        raise ValueError("El ratio de utilización debe estar en el rango (0, 1].")

    effective_inferences_per_hour = (
        inferences_per_second * 3600.0 * system_utilization_ratio
    )
    return hourly_rate_usd / effective_inferences_per_hour


def calculate_cost_per_million_requests(cost_per_inference: float) -> float:
    """Calcula el costo monetario por un millón de peticiones de inferencia.

    Args:
        cost_per_inference: Costo unitario por inferencia ($ USD).

    Returns:
        Costo en USD por 1,000,000 de inferencias servidas.
    """
    if cost_per_inference < 0:
        msg = "El costo por inferencia no puede ser negativo."
        raise ValueError(msg)
    return cost_per_inference * 1_000_000.0


def compare_cloud_providers(
    requests_per_second: float = 50.0,
    system_utilization_ratio: float = 0.70,
    hours_per_month: float = 730.0,
) -> list[dict[str, Any]]:
    """Compara los costos operativos de inferencia entre distintos proveedores de nube.

    Args:
        requests_per_second: Throughput sostenido de inferencia del microservicio.
        system_utilization_ratio: Porcentaje promedio de utilización de la capacidad contratada.
        hours_per_month: Horas promedio mensuales de operación (730 horas estándar).

    Returns:
        Lista de diccionarios con el análisis financiero detallado por proveedor e instancia.
    """
    results: list[dict[str, Any]] = []
    for instance in CLOUD_PRICING_CATALOG:
        unit_cost = calculate_cost_per_inference(
            hourly_rate_usd=instance.hourly_rate_usd,
            requests_per_second=requests_per_second,
            system_utilization_ratio=system_utilization_ratio,
        )
        cost_1m = calculate_cost_per_million_requests(unit_cost)
        monthly_cost = round(instance.hourly_rate_usd * hours_per_month, 2)

        record = asdict(instance)
        record.update(
            {
                "monthly_cost_usd": monthly_cost,
                "cost_per_inference_usd": round(unit_cost, 8),
                "cost_per_million_usd": round(cost_1m, 4),
            }
        )
        results.append(record)

    return sorted(results, key=lambda x: x["cost_per_million_usd"])


def compare_cloud_providers_from_throughput(
    inferences_per_second: float,
    system_utilization_ratio: float = 0.70,
    hours_per_month: float = 730.0,
) -> list[dict[str, Any]]:
    """Proyecta costos de nube usando un throughput estimado de inferencias.

    Se supone el mismo throughput para todas las instancias.
    No se valida que puedan sostener esa capacidad.
    """
    if hours_per_month <= 0:
        raise ValueError("Las horas mensuales deben ser mayores a 0.")

    results: list[dict[str, Any]] = []

    for instance in CLOUD_PRICING_CATALOG:
        unit_cost = calculate_cost_per_inference_from_throughput(
            hourly_rate_usd=instance.hourly_rate_usd,
            inferences_per_second=inferences_per_second,
            system_utilization_ratio=system_utilization_ratio,
        )

        record = asdict(instance)
        record.update(
            {
                "monthly_cost_usd": round(
                    instance.hourly_rate_usd * hours_per_month, 2
                ),
                "cost_per_inference_usd": unit_cost,
                "cost_per_million_usd": calculate_cost_per_million_requests(
                    unit_cost
                ),
            }
        )
        results.append(record)

    return sorted(results, key=lambda item: item["cost_per_million_usd"])


def simulate_finops_optimizations(
    baseline_hourly_rate: float = 0.0416,
    baseline_rps: float = 45.0,
    system_utilization_ratio: float = 0.70,
    annual_requests: float = 50_000_000.0,
) -> dict[str, Any]:
    
    """Simula escenarios hipotéticos de optimización FinOps.

        Estrategias evaluadas:
        1. Batching dinámico: aumento hipotético del throughput.
        2. Rightsizing: reducción hipotética de la tarifa de cómputo.
        3. Caché: reducción de inferencias procesadas.
        4. Estrategia integral: combinación de optimizaciones y descuentos.

        Los factores de mejora son supuestos ilustrativos y no
        representan resultados validados mediante pruebas de carga.

        El modelo estima costos de cómputo proporcionales al volumen
        procesado. No representa el costo total anual de infraestructura
        ni incluye almacenamiento, red o servicios auxiliares.

        Args:
            baseline_hourly_rate: Tarifa horaria de referencia en USD.
            baseline_rps: Throughput hipotético en solicitudes por segundo.
            system_utilization_ratio: Factor supuesto de utilización.
            annual_requests: Volumen anual hipotético de solicitudes.

        Returns:
            Diccionario con costos simulados, ahorro estimado y
            porcentaje de reducción bajo los supuestos definidos.
    """

    base_unit_cost = calculate_cost_per_inference(
        baseline_hourly_rate, baseline_rps, system_utilization_ratio
    )
    base_annual_cost = base_unit_cost * annual_requests

    # 1. Optimización Batching: incrementa throughput efectivo x3.2
    batching_rps = baseline_rps * 3.2
    batching_unit_cost = calculate_cost_per_inference(
        baseline_hourly_rate, batching_rps, system_utilization_ratio
    )
    batching_annual_cost = batching_unit_cost * annual_requests

    # 2. Rightsizing: reduce tamaño de instancia de t3.medium ($0.0416) a Fargate ajustado ($0.0225)
    rightsized_rate = baseline_hourly_rate * 0.55
    rightsized_unit_cost = calculate_cost_per_inference(
        rightsized_rate, baseline_rps, system_utilization_ratio
    )
    rightsized_annual_cost = rightsized_unit_cost * annual_requests

    # 3. Caching: atiende 20% de peticiones directamente desde memoria/Redis sin costo de inferencia
    cache_hit_ratio = 0.20
    caching_annual_cost = base_annual_cost * (1.0 - cache_hit_ratio)

    # 4. Estrategia Integral Combinada (Rightsizing + Batching + Caching + Savings Plans):
    combined_hourly_rate = baseline_hourly_rate * 0.55 * 0.65  # Rightsizing + 35% Savings Plan
    combined_rps = baseline_rps * 2.5
    combined_unit_cost = calculate_cost_per_inference(
        combined_hourly_rate, combined_rps, system_utilization_ratio
    )
    combined_annual_cost = (combined_unit_cost * annual_requests) * (1.0 - cache_hit_ratio)

    total_annual_savings = base_annual_cost - combined_annual_cost
    savings_percentage = (total_annual_savings / base_annual_cost) * 100.0

    return {
        "baseline_annual_cost_usd": round(base_annual_cost, 2),
        "baseline_cost_per_million_usd": round(calculate_cost_per_million_requests(base_unit_cost), 4),
        "batching_annual_cost_usd": round(batching_annual_cost, 2),
        "rightsized_annual_cost_usd": round(rightsized_annual_cost, 2),
        "caching_annual_cost_usd": round(caching_annual_cost, 2),
        "optimized_annual_cost_usd": round(combined_annual_cost, 2),
        "optimized_cost_per_million_usd": round(calculate_cost_per_million_requests(combined_unit_cost * (1.0 - cache_hit_ratio)), 4),
        "annual_savings_usd": round(total_annual_savings, 2),
        "savings_percentage": round(savings_percentage, 2),
    }
