"""Pruebas pequeñas para el entrenamiento y la evaluación de P2."""

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici_mlops.config import SEMILLA_ALEATORIA
from mibici_mlops.features import COLUMNAS_CARACTERISTICAS_P2
from mibici_mlops.modeling import train
from mibici_mlops.modeling.train import (
    CONJUNTOS_DESARROLLO,
    ETIQUETA_EQUIPO,
    NOMBRE_RUN_DUMMY,
    NOMBRE_RUN_LOGISTICA,
    NombreModelo,
    SolverRegresion,
    app,
    calcular_metricas,
    cargar_dataset,
    cargar_version_datos,
    crear_dummy,
    crear_etiquetas_mlflow,
    crear_parametros_mlflow,
    crear_regresion_logistica,
    entrenar_y_evaluar_desarrollo,
    experimento_completo,
    prefijar_metricas_mlflow,
    seleccionar_mejor_modelo,
    separar_variables,
)

runner = CliRunner()


def crear_dataset_minimo() -> pd.DataFrame:
    """Crea datos pequeños con todas las columnas requeridas."""
    datos = {
        columna: [0, 1, 2]
        for columna in COLUMNAS_CARACTERISTICAS_P2
    }
    datos.update(
        {
            "fecha_hora": pd.to_datetime(
                ["2024-01-01", "2025-01-01", "2026-01-01"]
            ),
            "presion_futura_2h": [-0.3, 0.0, 0.3],
            "clase_riesgo": ["riesgo_vaciado", "normal", "riesgo_saturacion"],
            "conjunto": ["entrenamiento", "validacion", "prueba"],
        }
    )
    return pd.DataFrame(datos)


def crear_dataset_desarrollo_minimo() -> pd.DataFrame:
    """Crea datos balanceados para entrenar ambos modelos sin datos reales."""
    cantidad_filas = 18
    datos = {
        columna: list(range(cantidad_filas))
        for columna in COLUMNAS_CARACTERISTICAS_P2
    }
    datos.update(
        {
            "clase_riesgo": [
                "riesgo_vaciado",
                "normal",
                "riesgo_saturacion",
            ]
            * 6,
            "conjunto": ["entrenamiento"] * 9 + ["validacion"] * 9,
        }
    )
    return pd.DataFrame(datos)


def test_dummy_es_estratificado_y_reproducible() -> None:
    modelo = crear_dummy()

    assert modelo.strategy == "stratified"
    assert modelo.random_state == SEMILLA_ALEATORIA


def test_regresion_logistica_utiliza_solver_seleccionado() -> None:
    modelo = crear_regresion_logistica(solver=SolverRegresion.LBFGS)

    clasificador = modelo.named_steps["clasificador"]

    assert clasificador.solver == "lbfgs"
    assert clasificador.class_weight == "balanced"


def test_comando_rechaza_solver_no_permitido() -> None:
    resultado = runner.invoke(app, ["comparar", "--solver", "invalido"])

    assert resultado.exit_code != 0
    assert "saga" in resultado.output
    assert "lbfgs" in resultado.output


def test_version_de_datos_se_lee_desde_manifiesto(tmp_path: Path) -> None:
    ruta_manifiesto = tmp_path / "data_manifest.json"
    ruta_manifiesto.write_text(
        '{"dataset_version": "version-de-prueba"}',
        encoding="utf-8",
    )

    version = cargar_version_datos(ruta_manifiesto)

    assert version == "version-de-prueba"


def test_etiquetas_mlflow_tienen_un_esquema_comun() -> None:
    etiquetas = crear_etiquetas_mlflow(
        pregunta="P2",
        tipo_tarea="clasificacion",
        etapa="desarrollo",
        rol_modelo="candidato",
        commit_git="abc123",
    )

    assert etiquetas == {
        "equipo": ETIQUETA_EQUIPO,
        "pregunta": "P2",
        "tipo_tarea": "clasificacion",
        "etapa": "desarrollo",
        "rol_modelo": "candidato",
        "git_commit": "abc123",
    }


def test_parametros_mlflow_permiten_reproducir_regresion() -> None:
    parametros = crear_parametros_mlflow(
        NombreModelo.LOGISTICA,
        "version-de-prueba",
        {"entrenamiento": 10, "validacion": 5},
        777,
        1.0,
        SolverRegresion.SAGA,
        100,
    )

    assert parametros["version_datos"] == "version-de-prueba"
    assert parametros["semilla"] == 777
    assert parametros["modelo"] == NombreModelo.LOGISTICA.value
    assert parametros["solver"] == "saga"
    assert parametros["max_iter"] == 100
    assert parametros["class_weight"] == "balanced"
    assert parametros["filas_entrenamiento"] == 10
    assert parametros["filas_validacion"] == 5
    assert "inicio_entrenamiento" in parametros
    assert "fin_prueba_exclusivo" in parametros


def test_metricas_mlflow_incluyen_el_conjunto() -> None:
    metricas = prefijar_metricas_mlflow(
        "validacion",
        {"f1_macro": 0.43, "gmean": 0.55},
    )

    assert metricas == {
        "validacion_f1_macro": 0.43,
        "validacion_gmean": 0.55,
    }


def test_comparacion_abre_un_run_por_modelo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nombres_run = []

    def iniciar_run_simulado(run_name: str) -> nullcontext[None]:
        nombres_run.append(run_name)
        return nullcontext()

    monkeypatch.setattr(train.mlflow, "start_run", iniciar_run_simulado)
    monkeypatch.setattr(train.mlflow, "set_tags", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train,
        "guardar_matriz_confusion",
        lambda *args, **kwargs: None,
    )

    entrenar_y_evaluar_desarrollo(
        crear_dataset_desarrollo_minimo(),
        tmp_path,
        "version-de-prueba",
        "abc123",
        solver=SolverRegresion.LBFGS,
    )

    assert nombres_run == [NOMBRE_RUN_DUMMY, NOMBRE_RUN_LOGISTICA]


def test_variables_no_incluyen_columnas_del_futuro() -> None:
    dataset = crear_dataset_minimo()

    conjuntos = separar_variables(dataset)
    caracteristicas, _ = conjuntos["entrenamiento"]

    assert caracteristicas.columns.tolist() == COLUMNAS_CARACTERISTICAS_P2
    assert "fecha_hora" not in caracteristicas
    assert "presion_futura_2h" not in caracteristicas
    assert "clase_riesgo" not in caracteristicas
    assert "conjunto" not in caracteristicas


def test_metricas_incluyen_gmean() -> None:
    y_real = pd.Series(["riesgo_vaciado", "normal", "riesgo_saturacion"])
    y_predicha = y_real.copy()

    metricas = calcular_metricas(y_real, y_predicha)

    assert set(metricas) == {
        "accuracy",
        "f1_macro",
        "recall_vaciado",
        "recall_saturacion",
        "gmean",
    }
    assert all(valor == pytest.approx(1.0) for valor in metricas.values())


def test_gmean_baja_a_cero_si_una_clase_no_se_reconoce() -> None:
    y_real = pd.Series(["riesgo_vaciado", "normal", "riesgo_saturacion"])
    y_predicha = pd.Series(["normal", "normal", "riesgo_saturacion"])

    metricas = calcular_metricas(y_real, y_predicha)

    assert metricas["gmean"] == pytest.approx(0.0)


def test_seleccion_usa_solamente_f1_de_validacion() -> None:
    metricas = pd.DataFrame(
        [
            {"modelo": "dummy_stratified", "conjunto": "validacion", "f1_macro": 0.30},
            {"modelo": "logistic_regression", "conjunto": "validacion", "f1_macro": 0.60},
            {"modelo": "dummy_stratified", "conjunto": "prueba", "f1_macro": 0.99},
        ]
    )

    seleccionado = seleccionar_mejor_modelo(metricas)

    assert seleccionado == NombreModelo.LOGISTICA.value


def test_comparacion_no_carga_prueba(tmp_path: Path) -> None:
    ruta = tmp_path / "dataset_p2.parquet"
    crear_dataset_minimo().to_parquet(ruta, index=False)

    dataset = cargar_dataset(ruta, CONJUNTOS_DESARROLLO)

    assert set(dataset["conjunto"].unique()) == {"entrenamiento", "validacion"}
    assert "prueba" not in dataset["conjunto"].values


def test_experimento_completo_ejecuta_todas_las_etapas_en_orden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    etapas_ejecutadas = []

    def registrar_etapa(nombre: str, comando: list[str]) -> None:
        etapas_ejecutadas.append((nombre, comando))

    monkeypatch.setattr(train, "ejecutar_paso_experimento", registrar_etapa)

    experimento_completo()

    assert [nombre for nombre, _ in etapas_ejecutadas] == [
        "P1: regresión de demanda",
        "P2: construcción del conjunto estación-hora",
        "P2: comparación de modelos",
        "P2: evaluación final",
        "P3: agrupamiento de estaciones",
    ]


def test_experimento_completo_se_detiene_si_falla_una_etapa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    etapas_ejecutadas = []

    def fallar_en_comparacion(nombre: str, comando: list[str]) -> None:
        etapas_ejecutadas.append(nombre)
        if nombre == "P2: comparación de modelos":
            raise RuntimeError("fallo simulado")

    monkeypatch.setattr(train, "ejecutar_paso_experimento", fallar_en_comparacion)

    with pytest.raises(RuntimeError, match="fallo simulado"):
        experimento_completo()

    assert etapas_ejecutadas == [
        "P1: regresión de demanda",
        "P2: construcción del conjunto estación-hora",
        "P2: comparación de modelos",
    ]
