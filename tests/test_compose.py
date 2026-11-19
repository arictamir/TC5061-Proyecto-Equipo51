"""Pruebas unitarias para validar la configuración de Docker Compose.

Entregable: Microservicios, CI/CD y FinOps
Responsable: Ari
"""

from pathlib import Path
import yaml


def test_docker_compose_file_exists() -> None:
    """Verifica que el archivo docker-compose.yml exista en la raíz del proyecto."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "El archivo docker-compose.yml no fue encontrado"


def test_docker_compose_structure() -> None:
    """Valida la presencia de la clave de servicios en docker-compose.yml.

    TODO (Ari): Ampliar las aserciones para validar servicios, redes, volúmenes y healthchecks.
    """
    compose_path = Path("docker-compose.yml")
    with open(compose_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "services" in config, "La clave 'services' no está en docker-compose.yml"
    assert "mlflow" in config["services"], "Falta el servicio 'mlflow'"
    assert "inference-api" in config["services"], "Falta el servicio 'inference-api'"
