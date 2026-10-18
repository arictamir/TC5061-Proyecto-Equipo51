"""Script ejecutable para entrenar y registrar el baseline de la Pregunta 3 en MLflow.

Pregunta 3: Clustering y segmentación funcional de estaciones de MiBici.
Ventana temporal: Enero 2024 a Septiembre 2026.
"""

from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# Configuración centralizada y módulo de features
from mibici_mlops import config
from mibici_mlops.featuresp3 import generate_p3_station_features


def get_git_commit_hash() -> str:
    """Recupera el hash del commit actual para trazabilidad en MLflow."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "sin_commit_local"


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los conjuntos procesados de viajes y nomenclatura desde data/processed/."""
    trips_path = config.PROCESSED_DATA_DIR / "trips"
    nomenclature_path = config.PROCESSED_DATA_DIR / "nomenclature"

    # Lectura si es carpeta con particiones Parquet o archivo Parquet consolidado
    if trips_path.is_dir():
        trips_df = pd.read_parquet(trips_path)
    elif trips_path.with_suffix(".parquet").exists():
        trips_df = pd.read_parquet(trips_path.with_suffix(".parquet"))
    else:
        parquet_trips = list(config.PROCESSED_DATA_DIR.glob("*trips*.parquet"))
        if not parquet_trips:
            raise FileNotFoundError(
                f"No se encontró el conjunto de viajes en {trips_path}"
            )
        trips_df = pd.read_parquet(parquet_trips[0])

    if nomenclature_path.is_dir():
        stations_df = pd.read_parquet(nomenclature_path)
    elif nomenclature_path.with_suffix(".parquet").exists():
        stations_df = pd.read_parquet(nomenclature_path.with_suffix(".parquet"))
    else:
        parquet_nom = list(config.PROCESSED_DATA_DIR.glob("*nomenclatura*.parquet"))
        if not parquet_nom:
            raise FileNotFoundError(
                f"No se encontró el catálogo de nomenclatura en {nomenclature_path}"
            )
        stations_df = pd.read_parquet(parquet_nom[0])

    return trips_df, stations_df


def train_p3_baseline():
    """Ejecuta el pipeline de clustering y registra parámetros, métricas y artefactos."""
    print("1. Cargando datos desde data/processed/ (Ene 2024 - Sep 2026)...")
    trips_df, stations_df = load_processed_data()

    print("2. Generando features a nivel estación con featuresp3...")
    features_df = generate_p3_station_features(trips_df, stations_df)
    n_stations, n_features = features_df.shape
    print(f"   Matriz construida: {n_stations} estaciones y {n_features} variables.")

    print(f"3. Configurando experimento en MLflow: {config.MLFLOW_EXPERIMENT_NAME}...")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="p3_baseline_kmeans"):
        # Tags de trazabilidad y gobernanza
        commit_hash = get_git_commit_hash()
        mlflow.set_tags(
            {
                "equipo": "equipo51",
                "pregunta": "P3",
                "tipo_tarea": "clustering",
                "git_commit": commit_hash,
            }
        )

        # Escalado estandarizado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)

        # Modelo Baseline: K-Means con hiperparámetros centralizados
        model = KMeans(
            n_clusters=config.P3_DEFAULT_K,
            random_state=config.RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)

        # Métricas no supervisadas
        sil_score = float(silhouette_score(X_scaled, labels))
        db_score = float(davies_bouldin_score(X_scaled, labels))
        inertia_score = float(model.inertia_)

        # Registro de Parámetros
        data_range_str = f"{config.START_YEAR}-{config.START_MONTH:02d}_to_{config.END_YEAR}-{config.END_MONTH:02d}"
        mlflow.log_params(
            {
                "data_range": data_range_str,
                "start_year": config.START_YEAR,
                "start_month": config.START_MONTH,
                "end_year": config.END_YEAR,
                "end_month": config.END_MONTH,
                "nomenclature_file": config.NOMENCLATURE_FILE,
                "model_type": "KMeans",
                "n_clusters": config.P3_DEFAULT_K,
                "random_state": config.RANDOM_STATE,
                "scaler": "StandardScaler",
                "n_stations": n_stations,
                "features_list": list(features_df.columns),
            }
        )

        # Registro de Métricas con prefijo identificador
        mlflow.log_metrics(
            {
                "test_silhouette_score": sil_score,
                "test_davies_bouldin_index": db_score,
                "test_inertia": inertia_score,
            }
        )

        # Generación del artefacto: mapa de clusters
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        figure_path = config.FIGURES_DIR / "p3_clusters_spatial.png"

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            features_df["longitude"],
            features_df["latitude"],
            c=labels,
            cmap="tab10",
            s=45,
            alpha=0.8,
            edgecolors="none",
        )
        ax.set_title(
            f"MiBici - Clusters de Estaciones (k={config.P3_DEFAULT_K})\n"
            f"Periodo: {data_range_str} | Silhouette: {sil_score:.3f} | DBI: {db_score:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()

        # Registro del artefacto visual en MLflow
        mlflow.log_artifact(str(figure_path))

        print(
            f"\nRun P3 registrado en MLflow con éxito:"
            f"\n - Periodo: {data_range_str}"
            f"\n - Silhouette Score: {sil_score:.4f}"
            f"\n - Davies-Bouldin: {db_score:.4f}"
            f"\n - Artefacto: {figure_path}"
        )


if __name__ == "__main__":
    train_p3_baseline()