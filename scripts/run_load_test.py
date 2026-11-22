"""Script de automatización para pruebas de carga con Locust, monitoreo de recursos y análisis FinOps.

Genera métricas de latencia, throughput (RPS), consumo de CPU y memoria en contenedor,
así como gráficas de alto impacto para el reporte técnico del equipo.
"""

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from mibici_mlops.finops.cost_calculator import (
    compare_cloud_providers_from_throughput,
    estimate_inference_count,
    simulate_finops_optimizations,
)

# Configuración de estilo visual
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

REPORTS_DIR = Path("reports/performance")
FIGURES_PERF_DIR = Path("reports/figures/performance")
FIGURES_FINOPS_DIR = Path("reports/figures/finops")


def ensure_directories() -> None:
    """Crea los directorios de salida necesarios si no existen."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_PERF_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_FINOPS_DIR.mkdir(parents=True, exist_ok=True)


def load_locust_metrics(csv_path: Path, users: int) -> dict[str, Any]:
    """Lee las métricas agregadas de una ejecución real de Locust."""
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        aggregated = next(
            (row for row in reader if row["Name"] == "Aggregated"),
            None,
        )

    if aggregated is None:
        raise ValueError(
            f"No se encontraron métricas agregadas en {csv_path}"
        )

    requests = int(aggregated["Request Count"])
    failures = int(aggregated["Failure Count"])

    return {
        "users": users,
        "requests": requests,
        "rps": float(aggregated["Requests/s"]),
        "p50_ms": float(aggregated["50%"]),
        "p90_ms": float(aggregated["90%"]),
        "p95_ms": float(aggregated["95%"]),
        "p99_ms": float(aggregated["99%"]),
        "fail_ratio": failures / requests * 100 if requests else 0.0,
    }


def load_real_benchmark_dataset() -> list[dict[str, Any]]:
    """Carga las métricas reales de los escenarios de Locust."""
    scenarios = {
        5: "locust_5_users_stats.csv",
        25: "locust_25_users_stats.csv",
        50: "locust_50_users_stats.csv",
        100: "locust_100_users_monitored_stats.csv",
    }

    results = []

    for users, filename in scenarios.items():
        csv_path = REPORTS_DIR / "locust_real" / filename
        metrics = load_locust_metrics(csv_path, users)
        results.append(metrics)

    return results


def load_docker_metrics(
    csv_path: Path,
    start_time: str,
    end_time: str,
) -> list[dict[str, Any]]:
    """Calcula métricas de recursos durante una ventana de carga."""
    from datetime import datetime

    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)

    if start > end:
        raise ValueError("La fecha inicial debe ser anterior a la final.")

    samples: dict[str, list[dict[str, float]]] = {}

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            timestamp = datetime.fromisoformat(row["timestamp"])

            if not start <= timestamp <= end:
                continue

            container = row["container"]
            cpu = float(row["cpu_percent"].rstrip("%"))

            memory_value = row["memory_usage"].split("/")[0].strip()

            if memory_value.endswith("MiB"):
                memory_mib = float(memory_value.removesuffix("MiB"))
            elif memory_value.endswith("GiB"):
                memory_mib = float(memory_value.removesuffix("GiB")) * 1024
            else:
                raise ValueError(
                    f"Unidad de memoria no reconocida: {memory_value}"
                )

            samples.setdefault(container, []).append(
                {"cpu": cpu, "memory_mib": memory_mib}
            )

    results = []

    for container, measurements in samples.items():
        results.append(
            {
                "container": container,
                "samples": len(measurements),
                "cpu_avg": sum(m["cpu"] for m in measurements)
                / len(measurements),
                "cpu_max": max(m["cpu"] for m in measurements),
                "memory_avg_mib": sum(
                    m["memory_mib"] for m in measurements
                ) / len(measurements),
                "memory_max_mib": max(
                    m["memory_mib"] for m in measurements
                ),
            }
        )

    return results


def generate_real_docker_resource_plot(
    data: list[dict[str, Any]],
) -> Path:
    """Grafica el consumo promedio y máximo de recursos por contenedor."""
    containers = [d["container"].replace("mibici-", "") for d in data]

    cpu_avg = [d["cpu_avg"] for d in data]
    cpu_max = [d["cpu_max"] for d in data]

    memory_avg = [d["memory_avg_mib"] for d in data]
    memory_max = [d["memory_max_mib"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(containers))
    width = 0.35

    # CPU
    ax1.bar(
        [i - width / 2 for i in x],
        cpu_avg,
        width,
        label="Promedio",
        color="#3b528b",
    )
    ax1.bar(
        [i + width / 2 for i in x],
        cpu_max,
        width,
        label="Máximo registrado",
        color="#5dc863",
    )
    ax1.set_xticks(list(x), containers)
    ax1.set_ylabel("CPU (%)")
    ax1.set_title("Uso de CPU durante la prueba de 100 usuarios")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Memoria
    ax2.bar(
        [i - width / 2 for i in x],
        memory_avg,
        width,
        label="Promedio",
        color="#3b528b",
    )
    ax2.bar(
        [i + width / 2 for i in x],
        memory_max,
        width,
        label="Máximo registrado",
        color="#5dc863",
    )
    ax2.axhline(
        1024,
        color="red",
        linestyle="--",
        label="Límite: 1024 MiB",
    )
    ax2.set_xticks(list(x), containers)
    ax2.set_ylabel("Memoria (MiB)")
    ax2.set_title("Uso de memoria durante la prueba de 100 usuarios")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Monitoreo real de Docker: 16 muestras por contenedor",
        fontweight="bold",
    )

    output_path = FIGURES_PERF_DIR / "container_resource_usage.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def export_performance_csv(data: list[dict[str, Any]]) -> Path:
    """Exporta las métricas de carga y monitoreo a CSV."""
    csv_path = REPORTS_DIR / "locust_benchmark_metrics.csv"
    keys = list(data[0].keys())
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    return csv_path


def generate_locust_latency_rps_plot(data: list[dict[str, Any]]) -> Path:
    """Genera gráfica de throughput (RPS) y percentiles de latencia según concurrencia."""
    users = [d["users"] for d in data]
    rps = [d["rps"] for d in data]
    p50 = [d["p50_ms"] for d in data]
    p95 = [d["p95_ms"] for d in data]
    p99 = [d["p99_ms"] for d in data]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Throughput (RPS) vs Concurrencia
    ax1.plot(users, rps, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="Throughput (RPS)")
    ax1.set_title("Capacidad del Microservicio: Throughput (RPS) vs Concurrencia", fontweight="bold")
    ax1.set_xlabel("Usuarios Concurrentes (Locust)")
    ax1.set_ylabel("Peticiones por Segundo (RPS)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    for u, r in zip(users, rps, strict=False):
        ax1.annotate(f"{r:.1f} RPS", (u, r), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")

    # Panel 2: Latencias p50, p95, p99 vs Concurrencia
    ax2.plot(users, p50, marker="s", color="#2ca02c", linewidth=2, label="Mediana (p50)")
    ax2.plot(users, p95, marker="^", color="#ff7f0e", linewidth=2, label="Percentil 95 (p95)")
    ax2.plot(users, p99, marker="d", color="#d62728", linewidth=2, label="Percentil 99 (p99)")
    ax2.axhline(y=500, color="gray", linestyle=":", label="SLA Máximo (500ms)")
    ax2.set_title("Curvas de Latencia bajo Carga (ms)", fontweight="bold")
    ax2.set_xlabel("Usuarios Concurrentes (Locust)")
    ax2.set_ylabel("Tiempo de Respuesta (ms)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    output_path = FIGURES_PERF_DIR / "locust_latency_rps.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def load_estimated_inference_throughput(csv_path: Path) -> float:
    """Estima inferencias/s a partir de las estadísticas finales de Locust."""
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    def get_request_count(endpoint: str) -> int:
        for row in rows:
            if row["Name"] == endpoint:
                return int(row["Request Count"])
        raise ValueError(f"No se encontró el endpoint {endpoint}.")

    single_requests = get_request_count("/predict")
    batch_requests = get_request_count("/predict/batch")

    aggregated = next(
        (row for row in rows if row["Name"] == "Aggregated"),
        None,
    )
    if aggregated is None:
        raise ValueError("No se encontraron estadísticas agregadas de Locust.")

    total_requests = int(aggregated["Request Count"])
    requests_per_second = float(aggregated["Requests/s"])

    if total_requests <= 0 or requests_per_second <= 0:
        raise ValueError("Las estadísticas de Locust deben ser positivas.")

    estimated_inferences = estimate_inference_count(
        single_requests=single_requests,
        batch_requests=batch_requests,
        average_batch_size=7.5,
    )

    estimated_duration = total_requests / requests_per_second
    return estimated_inferences / estimated_duration


def generate_finops_cost_comparison_plot() -> Path:
    """Genera una comparación de costos proyectados basada en Locust."""
    locust_csv = (
        REPORTS_DIR
        / "locust_real"
        / "locust_100_users_monitored_stats.csv"
    )

    inference_throughput = load_estimated_inference_throughput(locust_csv)

    comparison = compare_cloud_providers_from_throughput(
        inferences_per_second=inference_throughput,
        system_utilization_ratio=0.70,
    )

    labels = [
        f"{c['provider']} {c['service_name']}\n({c['instance_type']})"
        for c in comparison
    ]
    costs_1m = [c["cost_per_million_usd"] for c in comparison]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2b5c8f", "#4682b4", "#5c9ebc", "#2e8b57", "#3cb371"]

    bars = ax.barh(
        labels,
        costs_1m,
        color=colors[: len(labels)],
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_title(
        "FinOps: costo proyectado por millón de inferencias",
        fontweight="bold",
    )
    ax.set_xlabel("USD por millón de inferencias estimadas")
    ax.grid(True, linestyle="--", alpha=0.6, axis="x")

    max_cost = max(costs_1m)
    ax.set_xlim(0, max_cost * 1.25)

    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + max_cost * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"${value:.4f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.01,
        "Proyección con throughput local estimado y utilización supuesta del 70%. "
        "Capacidad de instancias no validada.",
        ha="center",
        fontsize=8,
        color="dimgray",
    )

    output_path = FIGURES_FINOPS_DIR / "cost_per_inference_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_finops_optimizations_plot() -> Path:
    """Genera escenarios hipotéticos de optimización FinOps."""
    sim = simulate_finops_optimizations()

    scenarios = [
        "Base\n(On-Demand)",
        "Batching\ndinámico",
        "Rightsizing\n(vCPU/RAM)",
        "Caché en\nmemoria",
        "Estrategia\nintegral",
    ]

    annual_costs = [
        sim["baseline_annual_cost_usd"],
        sim["batching_annual_cost_usd"],
        sim["rightsized_annual_cost_usd"],
        sim["caching_annual_cost_usd"],
        sim["optimized_annual_cost_usd"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = [
        "#d95f02",
        "#7570b3",
        "#1b9e77",
        "#e7298a",
        "#2ca02c",
    ]

    bars = ax.bar(
        scenarios,
        annual_costs,
        color=palette,
        edgecolor="black",
        width=0.55,
        alpha=0.9,
    )

    ax.set_title(
        "FinOps: escenarios hipotéticos de optimización",
        fontweight="bold",
    )
    ax.set_ylabel("Costo anual proyectado (USD)")
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax.set_axisbelow(True)

    max_cost = max(annual_costs)
    ax.set_ylim(0, max_cost * 1.20)

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max_cost * 0.02,
            f"${value:,.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.01,
        "Simulación de 50 millones de solicitudes anuales. "
        "Ahorros hipotéticos, no validados experimentalmente.",
        ha="center",
        fontsize=8,
        color="dimgray",
    )

    output_path = FIGURES_FINOPS_DIR / "finops_optimizations_savings.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    """Ejecuta la generación completa de métricas, reportes y gráficas."""
    print("Iniciando análisis de desempeño bajo carga y modelo FinOps...")
    ensure_directories()

    data = load_real_benchmark_dataset()
    csv_file = export_performance_csv(data)
    print(f"Métricas exportadas a: {csv_file}")

    p1 = generate_locust_latency_rps_plot(data)
    print(f"Gráfica de latencia/RPS generada: {p1}")

    docker_data = load_docker_metrics(
        REPORTS_DIR / "docker_real" / "docker_stats_100_users.csv",
        start_time="2026-11-22T14:07:04",
        end_time="2026-11-22T14:08:03",
    )

    p2 = generate_real_docker_resource_plot(docker_data)
    print(f"Gráfica de recursos Docker generada: {p2}")

    p3 = generate_finops_cost_comparison_plot()
    print(f"Gráfica comparativa FinOps generada: {p3}")

    p4 = generate_finops_optimizations_plot()
    print(f"Gráfica de optimizaciones FinOps generada: {p4}")

    print("Pipeline de pruebas de carga y FinOps completado exitosamente.")


if __name__ == "__main__":
    main()
