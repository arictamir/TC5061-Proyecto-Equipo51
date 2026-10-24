import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MiBici dataset configuration
MIBICI_OPEN_DATA_URL = "https://mibici.net/es/datos-abiertos/"

START_YEAR = 2024
START_MONTH = 1

END_YEAR = 2026
END_MONTH = 9

NOMENCLATURE_FILE = "nomenclatura_2026_09.csv"

# ==========================================
# Reproducibilidad y Configuración de MLflow
# ==========================================
NOMBRE_EXPERIMENTO = "mibici-equipo51"
SEMILLA_ALEATORIA = 777

# Tracking URI configurable vía variable de entorno (por defecto SQLite local)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Nombres estandarizados para Model Registry
MODEL_REGISTRY_P1 = "mibici-equipo51-p1-demanda"
MODEL_REGISTRY_P2 = "mibici-equipo51-p2-riesgo"
MODEL_REGISTRY_P3 = "mibici-equipo51-p3-grupos"

# Tags obligatorios de trazabilidad y gobernanza
TAG_TEAM = "equipo51"
TAG_QUESTION_P1 = "p1"
TAG_QUESTION_P2 = "p2"
TAG_QUESTION_P3 = "p3"
TAG_STAGE_BASELINE = "baseline"
TAG_STAGE_SEARCH = "search"
TAG_STAGE_CANDIDATE = "candidate"

# P1 temporal configuration
P1_DATA_START = "2024-01-01 00:00:00"
P1_DATA_END = "2026-09-30 23:00:00"

P1_VALIDATION_START = "2026-01-01 00:00:00"
P1_TEST_START = "2026-05-01 00:00:00"

# Configuración de P2: clasificación de riesgo operativo por estación y hora
P2_HORIZONTE_HORAS = 2

# Umbrales definidos únicamente con los datos de entrenamiento de 2024.
P2_CUANTIL_VACIADO = 0.10
P2_CUANTIL_SATURACION = 0.90
P2_UMBRAL_VACIADO = -0.16129032258064516
P2_UMBRAL_SATURACION = 0.15789473684210525

# Los límites finales son exclusivos para evitar traslapes entre conjuntos.
P2_INICIO_ENTRENAMIENTO = "2024-01-01"
P2_FIN_ENTRENAMIENTO = "2025-01-01"
P2_INICIO_VALIDACION = "2025-01-01"
P2_FIN_VALIDACION = "2026-01-01"
P2_INICIO_PRUEBA = "2026-01-01"
P2_FIN_PRUEBA = "2026-10-01"

P2_DIR = PROCESSED_DATA_DIR / "p2"
P2_FIGURAS_DIR = FIGURES_DIR / "p2"
P2_NOMBRE_RUN_BASE = "P2 - clasificador base"

# ==========================================
# Parámetros Pregunta 3 (Clustering)
# ==========================================
P3_DEFAULT_K = 4

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
