"""Pruebas unitarias para validar la configuración de CI/CD con GitHub Actions.

Entregable: Microservicios, CI/CD y FinOps
Responsable: Ingrid
"""

from pathlib import Path
import yaml


def test_github_workflow_file_exists() -> None:
    """Verifica que el archivo de workflow de GitHub Actions exista."""
    workflow_path = Path(".github/workflows/ci-cd.yml")
    assert workflow_path.exists(), "El archivo .github/workflows/ci-cd.yml no existe"


def test_github_workflow_structure() -> None:
    """Valida los eventos y estructura general del pipeline de CI/CD.

    TODO (Ingrid): Ampliar las aserciones según los jobs y herramientas configuradas.
    """
    workflow_path = Path(".github/workflows/ci-cd.yml")
    with open(workflow_path, encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    assert "name" in workflow, "El workflow debe tener un nombre asignado"
    assert "on" in workflow or True in workflow, "El workflow debe definir disparadores ('on')"
    assert "jobs" in workflow, "El workflow debe definir jobs de ejecución"
