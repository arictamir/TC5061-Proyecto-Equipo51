"""Pruebas unitarias para validar la configuración de CI/CD con GitHub Actions.

Pipeline automatizado de pruebas, análisis estático, construcción OCI y validación de Compose.
"""

from pathlib import Path

import yaml


def _load_workflow() -> dict:
    """Carga y parsea el archivo de workflow de GitHub Actions."""
    workflow_path = Path(".github/workflows/ci-cd.yml")
    assert workflow_path.exists(), "El archivo .github/workflows/ci-cd.yml no existe"
    with open(workflow_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_github_workflow_file_exists() -> None:
    """Verifica que el archivo de workflow de GitHub Actions exista en la ruta estándar."""
    assert Path(".github/workflows/ci-cd.yml").exists()


def test_github_workflow_triggers() -> None:
    """Valida los eventos disparadores ('on') configurados para el pipeline."""
    workflow = _load_workflow()
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers is not None, "El workflow debe tener disparadores definidos"

    # Push en main y develop
    assert "push" in triggers, "El trigger 'push' debe estar presente"
    push_branches = triggers["push"].get("branches", [])
    assert "main" in push_branches, "Push debe vigilar la rama main"
    assert "develop" in push_branches, "Push debe vigilar la rama develop"

    # Pull Request en main y develop
    assert "pull_request" in triggers, "El trigger 'pull_request' debe estar presente"
    pr_branches = triggers["pull_request"].get("branches", [])
    assert "main" in pr_branches, "PR debe vigilar la rama main"
    assert "develop" in pr_branches, "PR debe vigilar la rama develop"

    # Soporte para ejecución manual
    assert "workflow_dispatch" in triggers, "Debe soportar ejecución manual workflow_dispatch"


def test_github_workflow_jobs_declared() -> None:
    """Valida que los 3 jobs principales estén declarados en el workflow."""
    workflow = _load_workflow()
    jobs = workflow.get("jobs", {})

    expected_jobs = [
        "lint-and-security",
        "test-suite",
        "docker-build-and-validate",
    ]
    for job_name in expected_jobs:
        assert job_name in jobs, f"Falta el job requerido '{job_name}' en el pipeline"


def test_github_workflow_job_dependencies() -> None:
    """Valida la secuencia ordenada de ejecución de los jobs mediante 'needs'."""
    workflow = _load_workflow()
    jobs = workflow.get("jobs", {})

    # test-suite debe depender de lint-and-security
    assert "needs" in jobs["test-suite"], "test-suite debe tener dependencias declaradas"
    assert "lint-and-security" in jobs["test-suite"]["needs"]

    # docker-build-and-validate debe depender de test-suite
    assert "needs" in jobs["docker-build-and-validate"], (
        "docker-build-and-validate debe tener dependencias"
    )
    assert "test-suite" in jobs["docker-build-and-validate"]["needs"]


def test_github_workflow_lint_and_security_config() -> None:
    """Valida las herramientas estáticas ejecutadas en el job de calidad y seguridad."""
    workflow = _load_workflow()
    job = workflow["jobs"]["lint-and-security"]

    steps_text = " ".join(step.get("run", "") for step in job.get("steps", []) if "run" in step)
    assert "ruff check" in steps_text, "Debe ejecutar verificación con Ruff"
    assert "bandit" in steps_text, "Debe ejecutar análisis de seguridad con Bandit"


def test_github_workflow_optimizations_and_cache() -> None:
    """Valida las optimizaciones avanzadas: matriz de pruebas, caché y artefactos."""
    workflow = _load_workflow()
    test_job = workflow["jobs"]["test-suite"]

    # Matriz de versiones
    strategy = test_job.get("strategy", {})
    matrix = strategy.get("matrix", {})
    assert "python-version" in matrix, (
        "El job test-suite debe incluir una matriz de versiones de Python"
    )

    # Caché y artefactos
    uses_list = [step.get("uses", "") for step in test_job.get("steps", []) if "uses" in step]
    assert any("cache" in u for u in uses_list), "Debe incluir un paso de caché de dependencias"
    assert any("upload-artifact" in u for u in uses_list), "Debe subir artefactos de cobertura"


def test_github_workflow_docker_and_compose_validation() -> None:
    """Valida la construcción de imagen y smoke test con Docker Compose."""
    workflow = _load_workflow()
    docker_job = workflow["jobs"]["docker-build-and-validate"]

    steps_text = " ".join(
        step.get("run", "") for step in docker_job.get("steps", []) if "run" in step
    )
    assert "docker compose config" in steps_text, "Debe validar la sintaxis de Docker Compose"
    assert "docker compose up" in steps_text, "Debe levantar los servicios con Compose"
    assert "/health" in steps_text, "Debe validar la salud de los servicios levantados"
    assert "/predict" in steps_text, "Debe ejecutar un smoke test al endpoint de inferencia"
    assert "docker compose down" in steps_text, "Debe realizar un desmontaje limpio al finalizar"
