"""Entrenamiento, evaluación y seguimiento de los modelos de P2."""

from enum import Enum
import json
from math import prod
from pathlib import Path
import subprocess
from typing import Annotated

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skops.io as sio
import typer

from mibici_mlops.config import (
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    P2_CUANTIL_SATURACION,
    P2_CUANTIL_VACIADO,
    P2_DIR,
    P2_FIGURAS_DIR,
    P2_FIN_ENTRENAMIENTO,
    P2_FIN_PRUEBA,
    P2_FIN_VALIDACION,
    P2_HORIZONTE_HORAS,
    P2_INICIO_ENTRENAMIENTO,
    P2_INICIO_PRUEBA,
    P2_INICIO_VALIDACION,
    P2_NOMBRE_EXPERIMENTO,
    P2_UMBRAL_SATURACION,
    P2_UMBRAL_VACIADO,
    PROJ_ROOT,
    SEMILLA_ALEATORIA,
)
from mibici_mlops.features_p2 import COLUMNAS_CARACTERISTICAS_P2

plt.switch_backend("Agg")

app = typer.Typer()

COLUMNA_OBJETIVO = "clase_riesgo"
COLUMNA_CONJUNTO = "conjunto"
ORDEN_CLASES = ["riesgo_vaciado", "normal", "riesgo_saturacion"]
CONJUNTOS_DESARROLLO = ["entrenamiento", "validacion"]
CONJUNTOS_FINALES = ["entrenamiento", "validacion", "prueba"]
RUTA_MANIFIESTO = PROJ_ROOT / "data_manifest.json"

NOMBRE_RUN_DUMMY = "P2 - Dummy estratificado - desarrollo"
NOMBRE_RUN_LOGISTICA = "P2 - Regresión logística - desarrollo"
NOMBRE_RUN_FINAL = "P2 - Regresión logística - evaluación final"

ETIQUETA_EQUIPO = "51"
ETIQUETA_PREGUNTA = "P2"


class NombreModelo(str, Enum):
    """Nombres aceptados por el comando de evaluación final."""

    DUMMY = "dummy_stratified"
    LOGISTICA = "logistic_regression"


class SolverRegresion(str, Enum):
    """Solvers permitidos para la regresión logística de P2."""

    SAGA = "saga"
    LBFGS = "lbfgs"


def configurar_mlflow() -> None:
    """Configura el servidor y el experimento compartido del equipo."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(P2_NOMBRE_EXPERIMENTO)


def cargar_version_datos(ruta_manifiesto: Path = RUTA_MANIFIESTO) -> str:
    """Lee la versión reproducible de los archivos fuente."""
    if not ruta_manifiesto.exists():
        raise FileNotFoundError(f"No existe el manifiesto: {ruta_manifiesto}")

    manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    version_datos = manifiesto.get("dataset_version")

    if not version_datos:
        raise ValueError("El manifiesto no contiene dataset_version.")

    return str(version_datos)


def obtener_commit_git() -> str:
    """Obtiene el commit que produjo la corrida de MLflow."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("No fue posible obtener el commit actual de Git.") from error

    commit = resultado.stdout.strip()
    if not commit:
        raise RuntimeError("Git devolvió un commit vacío.")
    return commit


def crear_etiquetas_mlflow(commit_git: str, etapa: str) -> dict[str, str]:
    """Crea las etiquetas exigidas para identificar una corrida."""
    return {
        "equipo": ETIQUETA_EQUIPO,
        "pregunta": ETIQUETA_PREGUNTA,
        "git_commit": commit_git,
        "etapa": etapa,
    }


def crear_parametros_mlflow(
    nombre_modelo: NombreModelo,
    version_datos: str,
    conteos: dict[str, int],
    semilla: int,
    c: float,
    solver: SolverRegresion,
    max_iter: int,
) -> dict[str, str | int | float]:
    """Reúne los parámetros necesarios para reproducir una corrida."""
    parametros: dict[str, str | int | float] = {
        "modelo": nombre_modelo.value,
        "version_datos": version_datos,
        "semilla": semilla,
        "horizonte_horas": P2_HORIZONTE_HORAS,
        "cuantil_vaciado": P2_CUANTIL_VACIADO,
        "cuantil_saturacion": P2_CUANTIL_SATURACION,
        "umbral_vaciado": P2_UMBRAL_VACIADO,
        "umbral_saturacion": P2_UMBRAL_SATURACION,
        "inicio_entrenamiento": P2_INICIO_ENTRENAMIENTO,
        "fin_entrenamiento_exclusivo": P2_FIN_ENTRENAMIENTO,
        "inicio_validacion": P2_INICIO_VALIDACION,
        "fin_validacion_exclusivo": P2_FIN_VALIDACION,
        "inicio_prueba": P2_INICIO_PRUEBA,
        "fin_prueba_exclusivo": P2_FIN_PRUEBA,
        "numero_caracteristicas": len(COLUMNAS_CARACTERISTICAS_P2),
    }
    parametros.update(
        {
            f"filas_{conjunto}": cantidad
            for conjunto, cantidad in conteos.items()
        }
    )

    if nombre_modelo == NombreModelo.DUMMY:
        parametros["strategy"] = "stratified"
    else:
        parametros.update(
            {
                "C": c,
                "solver": solver.value,
                "max_iter": max_iter,
                "class_weight": "balanced",
            }
        )

    return parametros


def prefijar_metricas_mlflow(
    conjunto: str,
    metricas: dict[str, float],
) -> dict[str, float]:
    """Agrega el conjunto al nombre de cada métrica registrada."""
    return {
        f"{conjunto}_{nombre}": valor
        for nombre, valor in metricas.items()
    }


def cargar_dataset(
    ruta: Path,
    conjuntos: list[str],
) -> pd.DataFrame:
    """Carga únicamente las características, el objetivo y los conjuntos solicitados."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el dataset de P2: {ruta}. "
            "Ejecuta primero python -m mibici_mlops.features_p2."
        )

    columnas = [*COLUMNAS_CARACTERISTICAS_P2, COLUMNA_OBJETIVO, COLUMNA_CONJUNTO]
    dataset = pd.read_parquet(
        ruta,
        columns=columnas,
        filters=[(COLUMNA_CONJUNTO, "in", conjuntos)],
    )
    dataset[COLUMNA_CONJUNTO] = dataset[COLUMNA_CONJUNTO].astype("string")
    encontrados = set(dataset[COLUMNA_CONJUNTO].unique())
    faltantes = set(conjuntos) - encontrados

    if faltantes:
        raise ValueError(f"No se encontraron los conjuntos: {sorted(faltantes)}")
    if not encontrados.issubset(set(conjuntos)):
        raise ValueError("Se cargaron conjuntos que no fueron solicitados.")

    return dataset


def separar_variables(dataset: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Separa X e y para cada conjunto sin incluir columnas del futuro."""
    columnas_requeridas = {
        *COLUMNAS_CARACTERISTICAS_P2,
        COLUMNA_OBJETIVO,
        COLUMNA_CONJUNTO,
    }
    faltantes = columnas_requeridas - set(dataset.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas para entrenar P2: {sorted(faltantes)}")

    separados = {}
    for conjunto in dataset[COLUMNA_CONJUNTO].unique():
        filas = dataset[dataset[COLUMNA_CONJUNTO] == conjunto]
        caracteristicas = filas[COLUMNAS_CARACTERISTICAS_P2].copy()
        objetivo = filas[COLUMNA_OBJETIVO].astype("string").copy()
        separados[str(conjunto)] = (caracteristicas, objetivo)
    return separados


def crear_dummy(semilla: int = SEMILLA_ALEATORIA) -> DummyClassifier:
    """Crea el baseline aleatorio estratificado y reproducible."""
    return DummyClassifier(
        strategy="stratified",
        random_state=semilla,
    )


def crear_regresion_logistica(
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> Pipeline:
    """Crea el primer modelo candidato con preprocesamiento incluido."""
    columnas_categoricas = ["estacion_id"]
    columnas_numericas = [
        columna
        for columna in COLUMNAS_CARACTERISTICAS_P2
        if columna not in columnas_categoricas
    ]
    preprocesamiento = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), columnas_numericas),
            (
                "estacion",
                OneHotEncoder(handle_unknown="ignore"),
                columnas_categoricas,
            ),
        ],
        sparse_threshold=1.0,
    )
    clasificador = LogisticRegression(
        C=c,
        solver=solver.value,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=semilla,
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("clasificador", clasificador),
        ]
    )


def crear_modelo(
    nombre: NombreModelo,
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> DummyClassifier | Pipeline:
    """Crea uno de los dos modelos permitidos para P2."""
    if nombre == NombreModelo.DUMMY:
        return crear_dummy(semilla)
    if nombre == NombreModelo.LOGISTICA:
        return crear_regresion_logistica(semilla, c, solver, max_iter)
    raise ValueError(f"Modelo no reconocido: {nombre}")


def calcular_metricas(y_real: pd.Series, y_predicha: pd.Series) -> dict[str, float]:
    """Calcula métricas multiclase, incluida la media geométrica del recall."""
    recalls = recall_score(
        y_real,
        y_predicha,
        labels=ORDEN_CLASES,
        average=None,
        zero_division=0,
    )
    recalls_por_clase = dict(zip(ORDEN_CLASES, recalls, strict=True))
    gmean = float(prod(float(valor) for valor in recalls) ** (1 / len(recalls)))
    return {
        "accuracy": float(accuracy_score(y_real, y_predicha)),
        "f1_macro": float(
            f1_score(
                y_real,
                y_predicha,
                labels=ORDEN_CLASES,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_vaciado": float(recalls_por_clase["riesgo_vaciado"]),
        "recall_saturacion": float(recalls_por_clase["riesgo_saturacion"]),
        "gmean": gmean,
    }


def evaluar_modelo(
    modelo: DummyClassifier | Pipeline,
    caracteristicas: pd.DataFrame,
    objetivo: pd.Series,
) -> tuple[dict[str, float], pd.Series]:
    """Genera predicciones y calcula las métricas para un conjunto."""
    prediccion = pd.Series(modelo.predict(caracteristicas), index=objetivo.index)
    return calcular_metricas(objetivo, prediccion), prediccion


def guardar_matriz_confusion(
    objetivo: pd.Series,
    prediccion: pd.Series,
    titulo: str,
    ruta: Path,
) -> None:
    """Guarda una matriz normalizada para facilitar su lectura."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura, eje = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        objetivo,
        prediccion,
        labels=ORDEN_CLASES,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        xticks_rotation=20,
        ax=eje,
    )
    eje.set_title(titulo)
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def convertir_metricas_en_fila(
    modelo: str,
    conjunto: str,
    metricas: dict[str, float],
) -> dict[str, str | float]:
    """Agrega los identificadores del modelo y conjunto a sus métricas."""
    return {
        "modelo": modelo,
        "conjunto": conjunto,
        **metricas,
    }


def seleccionar_mejor_modelo(metricas: pd.DataFrame) -> str:
    """Selecciona exclusivamente mediante F1 macro de validación."""
    validacion = metricas[metricas["conjunto"] == "validacion"]

    if validacion.empty:
        raise ValueError("No hay métricas de validación para seleccionar un modelo.")

    return str(validacion.loc[validacion["f1_macro"].idxmax(), "modelo"])


def entrenar_y_evaluar_desarrollo(
    dataset: pd.DataFrame,
    directorio_figuras: Path,
    version_datos: str,
    commit_git: str,
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> tuple[pd.DataFrame, str]:
    """Compara Dummy y regresión logística sin consultar prueba."""
    conjuntos = separar_variables(dataset)
    x_entrenamiento, y_entrenamiento = conjuntos["entrenamiento"]
    conteos = {
        conjunto: len(objetivo)
        for conjunto, (_, objetivo) in conjuntos.items()
    }
    resultados = []
    nombres_run = {
        NombreModelo.DUMMY: NOMBRE_RUN_DUMMY,
        NombreModelo.LOGISTICA: NOMBRE_RUN_LOGISTICA,
    }

    for nombre in [NombreModelo.DUMMY, NombreModelo.LOGISTICA]:
        with mlflow.start_run(run_name=nombres_run[nombre]):
            mlflow.set_tags(crear_etiquetas_mlflow(commit_git, "desarrollo"))
            mlflow.log_params(
                crear_parametros_mlflow(
                    nombre,
                    version_datos,
                    conteos,
                    semilla,
                    c,
                    solver,
                    max_iter,
                )
            )
            mlflow.log_artifact(
                str(RUTA_MANIFIESTO),
                artifact_path="reproducibilidad",
            )

            logger.info(f"Entrenando {nombre.value}.")
            modelo = crear_modelo(nombre, semilla, c, solver, max_iter)
            modelo.fit(x_entrenamiento, y_entrenamiento)

            for conjunto in CONJUNTOS_DESARROLLO:
                caracteristicas, objetivo = conjuntos[conjunto]
                metricas, prediccion = evaluar_modelo(
                    modelo,
                    caracteristicas,
                    objetivo,
                )
                resultados.append(
                    convertir_metricas_en_fila(nombre.value, conjunto, metricas)
                )
                mlflow.log_metrics(prefijar_metricas_mlflow(conjunto, metricas))

                ruta_matriz = (
                    directorio_figuras
                    / f"matriz_{nombre.value}_{conjunto}.png"
                )
                guardar_matriz_confusion(
                    objetivo,
                    prediccion,
                    f"{nombre.value} - {conjunto}",
                    ruta_matriz,
                )
                mlflow.log_artifact(
                    str(ruta_matriz),
                    artifact_path="matrices_confusion",
                )

    tabla_metricas = pd.DataFrame(resultados)
    mejor_modelo = seleccionar_mejor_modelo(tabla_metricas)
    return tabla_metricas, mejor_modelo


def evaluar_modelo_final(
    dataset: pd.DataFrame,
    nombre: NombreModelo,
    directorio_figuras: Path,
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> tuple[DummyClassifier | Pipeline, pd.DataFrame]:
    """Entrena el modelo confirmado y lo evalúa en los tres conjuntos."""
    conjuntos = separar_variables(dataset)
    x_entrenamiento, y_entrenamiento = conjuntos["entrenamiento"]
    modelo = crear_modelo(nombre, semilla, c, solver, max_iter)
    modelo.fit(x_entrenamiento, y_entrenamiento)
    resultados = []

    for conjunto in CONJUNTOS_FINALES:
        caracteristicas, objetivo = conjuntos[conjunto]
        metricas, prediccion = evaluar_modelo(modelo, caracteristicas, objetivo)
        resultados.append(convertir_metricas_en_fila(nombre.value, conjunto, metricas))
        guardar_matriz_confusion(
            objetivo,
            prediccion,
            f"Modelo final {nombre.value} - {conjunto}",
            directorio_figuras / f"matriz_final_{conjunto}.png",
        )

    return modelo, pd.DataFrame(resultados)


def mostrar_metricas(metricas: pd.DataFrame) -> None:
    """Muestra en consola una tabla corta de resultados."""
    columnas_metricas = [
        "accuracy",
        "f1_macro",
        "recall_vaciado",
        "recall_saturacion",
        "gmean",
    ]
    tabla = metricas.copy()
    tabla[columnas_metricas] = tabla[columnas_metricas].round(4)
    typer.echo(tabla.to_string(index=False))


@app.command("comparar")
def comparar(
    data_path: Path = P2_DIR / "dataset_p2.parquet",
    metrics_path: Path = P2_DIR / "metricas_desarrollo.csv",
    figures_dir: Path = P2_FIGURAS_DIR,
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> None:
    """Compara los modelos usando solamente entrenamiento y validación."""
    configurar_mlflow()
    version_datos = cargar_version_datos()
    commit_git = obtener_commit_git()
    logger.info(
        f"Parámetros de regresión logística: C={c}, solver={solver.value}, "
        f"max_iter={max_iter}, semilla={semilla}."
    )
    dataset = cargar_dataset(data_path, CONJUNTOS_DESARROLLO)
    metricas, mejor_modelo = entrenar_y_evaluar_desarrollo(
        dataset,
        figures_dir,
        version_datos,
        commit_git,
        semilla,
        c,
        solver,
        max_iter,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(metrics_path, index=False)
    mostrar_metricas(metricas)
    logger.success(
        f"Comparación terminada. Mejor modelo en validación: {mejor_modelo}. "
        f"Métricas: {metrics_path}"
    )


@app.command("evaluar-final")
def evaluar_final(
    modelo: Annotated[NombreModelo, typer.Option("--modelo")],
    data_path: Path = P2_DIR / "dataset_p2.parquet",
    metrics_path: Path = P2_DIR / "metricas_finales.csv",
    figures_dir: Path = P2_FIGURAS_DIR,
    model_path: Path = MODELS_DIR / "p2" / "modelo_p2.skops",
    semilla: int = SEMILLA_ALEATORIA,
    c: float = 1.0,
    solver: SolverRegresion = SolverRegresion.SAGA,
    max_iter: int = 100,
) -> None:
    """Evalúa en prueba un modelo ya seleccionado mediante validación."""
    configurar_mlflow()
    version_datos = cargar_version_datos()
    commit_git = obtener_commit_git()
    logger.info(
        f"Parámetros de regresión logística: C={c}, solver={solver.value}, "
        f"max_iter={max_iter}, semilla={semilla}."
    )
    dataset = cargar_dataset(data_path, CONJUNTOS_FINALES)
    conteos = {
        str(conjunto): int(cantidad)
        for conjunto, cantidad in dataset[COLUMNA_CONJUNTO].value_counts().items()
    }

    with mlflow.start_run(run_name=NOMBRE_RUN_FINAL):
        mlflow.set_tags(crear_etiquetas_mlflow(commit_git, "evaluacion_final"))
        mlflow.log_params(
            crear_parametros_mlflow(
                modelo,
                version_datos,
                conteos,
                semilla,
                c,
                solver,
                max_iter,
            )
        )
        mlflow.log_artifact(
            str(RUTA_MANIFIESTO),
            artifact_path="reproducibilidad",
        )

        modelo_entrenado, metricas = evaluar_modelo_final(
            dataset,
            modelo,
            figures_dir,
            semilla,
            c,
            solver,
            max_iter,
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metricas.to_csv(metrics_path, index=False)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(modelo_entrenado, model_path)

        for fila in metricas.to_dict(orient="records"):
            conjunto = str(fila.pop("conjunto"))
            fila.pop("modelo")
            metricas_conjunto = {
                nombre: float(valor)
                for nombre, valor in fila.items()
            }
            mlflow.log_metrics(
                prefijar_metricas_mlflow(conjunto, metricas_conjunto)
            )

            mlflow.log_artifact(
                str(figures_dir / f"matriz_final_{conjunto}.png"),
                artifact_path="matrices_confusion",
            )

        mlflow.log_artifact(str(metrics_path), artifact_path="metricas")
        mlflow.log_artifact(str(model_path), artifact_path="modelo_serializado")

        ejemplo_entrada = dataset.loc[
            dataset[COLUMNA_CONJUNTO] == "entrenamiento",
            COLUMNAS_CARACTERISTICAS_P2,
        ].head(5)
        ejemplo_salida = modelo_entrenado.predict(ejemplo_entrada)
        firma = infer_signature(ejemplo_entrada, ejemplo_salida)
        mlflow.sklearn.log_model(
            sk_model=modelo_entrenado,
            name="modelo",
            signature=firma,
            input_example=ejemplo_entrada,
        )

    mostrar_metricas(metricas)
    logger.success(
        f"Evaluación final terminada. Métricas: {metrics_path} | Modelo: {model_path}"
    )


if __name__ == "__main__":
    app()
