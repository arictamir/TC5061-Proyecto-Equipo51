"""Pruebas unitarias para validar la configuración de Docker Compose.

Orquestación de microservicios MiBici MLOps (FastAPI + MLflow Tracking Server).
"""

from pathlib import Path
import yaml


def _load_compose() -> dict:
    """Carga y parsea el archivo docker-compose.yml."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "El archivo docker-compose.yml no fue encontrado"
    with open(compose_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_docker_compose_file_exists() -> None:
    """Verifica que el archivo docker-compose.yml exista en la raíz del proyecto."""
    assert Path("docker-compose.yml").exists()


def test_docker_compose_services_declared() -> None:
    """Valida que los dos microservicios requeridos estén presentes."""
    config = _load_compose()
    assert "services" in config, "La clave 'services' no existe en docker-compose.yml"
    assert "mlflow" in config["services"], "Falta el servicio 'mlflow'"
    assert "inference-api" in config["services"], "Falta el servicio 'inference-api'"


def test_docker_compose_mlflow_service_config() -> None:
    """Valida la configuración avanzada del servicio de MLflow."""
    config = _load_compose()
    mlflow_svc = config["services"]["mlflow"]

    # Nombre y política de reinicio
    assert mlflow_svc.get("container_name") == "mibici-mlflow"
    assert mlflow_svc.get("restart") == "unless-stopped"

    # Comando de arranque del servidor de MLflow
    cmd = mlflow_svc.get("command", [])
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    assert "mlflow" in cmd_str
    assert "server" in cmd_str
    assert "--port" in cmd_str
    assert "5000" in cmd_str

    # Puertos expuestos
    ports = mlflow_svc.get("ports", [])
    assert any("5000" in str(p) for p in ports), "El puerto 5000 debe estar expuesto"

    # Red y volúmenes
    assert "mibici-net" in mlflow_svc.get("networks", [])
    volumes = [str(v) for v in mlflow_svc.get("volumes", [])]
    assert any("mlflow.db" in v for v in volumes), "Debe montar el backend store SQLite"

    # Healthcheck
    health = mlflow_svc.get("healthcheck", {})
    assert "test" in health
    test_str = " ".join(health["test"]) if isinstance(health["test"], list) else str(health["test"])
    assert "5000" in test_str or "health" in test_str

    # Límites de recursos (deploy spec)
    deploy = mlflow_svc.get("deploy", {})
    limits = deploy.get("resources", {}).get("limits", {})
    assert "cpus" in limits and "memory" in limits


def test_docker_compose_inference_api_service_config() -> None:
    """Valida la configuración del microservicio de inferencia FastAPI."""
    config = _load_compose()
    api_svc = config["services"]["inference-api"]

    # Nombre y reinicio
    assert api_svc.get("container_name") == "mibici-inference-api"
    assert api_svc.get("restart") == "unless-stopped"

    # Dependencia de orden y salud con MLflow
    depends_on = api_svc.get("depends_on", {})
    assert "mlflow" in depends_on, "inference-api debe depender de mlflow"
    assert depends_on["mlflow"].get("condition") == "service_healthy"

    # Puertos
    ports = api_svc.get("ports", [])
    assert any("8000" in str(p) for p in ports), "El puerto 8000 debe estar expuesto"

    # Variables de entorno
    env = api_svc.get("environment", [])
    env_dict = {}
    for item in env:
        if "=" in item:
            k, v = item.split("=", 1)
            env_dict[k.strip()] = v.strip()
    assert "MLFLOW_TRACKING_URI" in env_dict
    assert "mlflow:5000" in env_dict["MLFLOW_TRACKING_URI"]

    # Red
    assert "mibici-net" in api_svc.get("networks", [])

    # Healthcheck
    health = api_svc.get("healthcheck", {})
    assert "test" in health
    test_str = " ".join(health["test"]) if isinstance(health["test"], list) else str(health["test"])
    assert "8000" in test_str or "health" in test_str

    # Límites de recursos (deploy spec)
    deploy = api_svc.get("deploy", {})
    limits = deploy.get("resources", {}).get("limits", {})
    assert "cpus" in limits and "memory" in limits


def test_docker_compose_networks_and_volumes() -> None:
    """Valida las redes y volúmenes persistentes declarados."""
    config = _load_compose()

    # Red bridge
    assert "networks" in config
    assert "mibici-net" in config["networks"]
    assert config["networks"]["mibici-net"].get("driver") == "bridge"

    # Volúmenes
    assert "volumes" in config
    assert "mlflow-data" in config["volumes"]


def test_env_example_completeness() -> None:
    """Valida que .env.example contenga las variables operativas requeridas."""
    env_path = Path(".env.example")
    assert env_path.exists(), ".env.example debe existir"
    content = env_path.read_text(encoding="utf-8")

    expected_vars = [
        "PORT",
        "MLFLOW_PORT",
        "MLFLOW_TRACKING_URI",
        "MODEL_ALIAS",
        "LOG_LEVEL",
        "API_CPU_LIMIT",
        "API_MEMORY_LIMIT",
    ]
    for var in expected_vars:
        assert var in content, f"Variable {var} no documentada en .env.example"
