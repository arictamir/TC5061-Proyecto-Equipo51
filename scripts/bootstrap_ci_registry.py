"""Construye un Model Registry efímero con los modelos reales de P1 para CI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = "mibici-equipo51-p1-demanda"
EXPERIMENT_NAME = "mibici-ci-bootstrap"
MODEL_PATHS = {
    "champion": Path("models/p1_champion"),
    "challenger": Path("models/p1_challenger"),
}


def _wait_until_ready(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int = 60,
) -> None:
    """Espera hasta que una versión del Registry quede lista."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        model_version = client.get_model_version(model_name, version)
        if model_version.status == "READY":
            return
        if model_version.status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"El registro de {model_name} v{version} terminó con error."
            )
        time.sleep(1)
    raise TimeoutError(
        f"El registro de {model_name} v{version} no quedó listo en "
        f"{timeout_seconds} segundos."
    )


def _ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    """Crea el modelo registrado cuando el Registry efímero está vacío."""
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(
            model_name,
            description="Modelo P1 real preparado para validaciones reproducibles de CI.",
        )


def bootstrap_registry(tracking_uri: str, model_name: str = MODEL_NAME) -> None:
    """Registra Champion y Challenger reales y verifica que puedan cargarse."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)
    _ensure_registered_model(client, model_name)

    for alias, relative_path in MODEL_PATHS.items():
        model_path = relative_path.resolve()
        if not (model_path / "MLmodel").is_file():
            raise FileNotFoundError(
                f"Falta el artefacto MLflow requerido para @{alias}: {model_path}"
            )

        try:
            existing = client.get_model_version_by_alias(model_name, alias)
        except MlflowException:
            existing = None

        if existing is None:
            with mlflow.start_run(run_name=f"ci-bootstrap-{alias}") as run:
                mlflow.set_tag("ci.bootstrap", "true")
                mlflow.set_tag("model.alias", alias)
                mlflow.log_artifacts(str(model_path), artifact_path="model")
                source = mlflow.get_artifact_uri("model")

            created = client.create_model_version(
                name=model_name,
                source=source,
                run_id=run.info.run_id,
                description=f"Modelo real de P1 preparado como @{alias} para CI.",
            )
            _wait_until_ready(client, model_name, created.version)
            client.set_registered_model_alias(model_name, alias, created.version)
            version = created.version
        else:
            version = existing.version

        loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
        if loaded_model is None:
            raise RuntimeError(f"No fue posible cargar {model_name}@{alias}.")
        print(f"Registry CI listo: {model_name} v{version} -> @{alias}")


def main() -> None:
    """Procesa argumentos y construye el Registry utilizado por CI."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        required=True,
        help="URI del servidor MLflow efímero utilizado por CI.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()
    bootstrap_registry(args.tracking_uri, args.model_name)


if __name__ == "__main__":
    main()
