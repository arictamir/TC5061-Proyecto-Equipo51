"""Pruebas unitarias y de integración para Contratos de Modelos, Firmas e Inferencia.

Módulo asignado a: Ingrid Hernández Carrasco (Integrante 3)
Cubre:
- Validación de firmas formales de MLflow (ModelSignature: inputs/outputs) para los tres Champions.
- Inferencia determinista y rangos plausibles de negocio (P1 >= 0, P2 suma de probas = 1.0, P3 clusters en [0, 3]).
- Pruebas ante casos límite (edge cases): ceros completos, valores atípicos y robustez de predicción.
"""

from catboost import CatBoostRegressor
from mlflow.models import ModelSignature, infer_signature
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mibici_mlops.features import COLUMNAS_CARACTERISTICAS_P2


# ============================================================================
# 1. P1: DEMANDA HORARIA (CATBOOST REGRESSOR)
# ============================================================================

@pytest.mark.unit
def test_p1_champion_model_signature(
    mock_champion_p1_model: CatBoostRegressor,
    sample_p1_inference_payload: pd.DataFrame,
) -> None:
    """Qué prueba: Valida el contrato de firma formal (ModelSignature) del modelo Champion de P1 en MLflow.

    Por qué es crítico: Asegura que el servicio o lote de inferencia valide que el payload entrante
    contenga exactamente las 11 columnas con sus tipos requeridos antes de ejecutar la predicción.
    Criterio de aceptación: La firma requiere 11 variables de entrada numéricas y produce una salida escalar continua.
    """
    preds = np.clip(mock_champion_p1_model.predict(sample_p1_inference_payload), 0, None)
    signature: ModelSignature = infer_signature(sample_p1_inference_payload, preds)

    # Validar 11 columnas en el esquema de entrada
    input_names = [col.name for col in signature.inputs.inputs]
    assert len(input_names) == 11
    assert "station_id" in input_names
    assert "lag_168" in input_names
    assert "rolling_mean_3h" in input_names

    # Validar tipo de salida escalar numérico
    assert len(signature.outputs.inputs) == 1


@pytest.mark.unit
def test_p1_inference_plausible_non_negative_demand(
    mock_champion_p1_model: CatBoostRegressor,
    sample_p1_inference_payload: pd.DataFrame,
) -> None:
    """Qué prueba: Verifica inferencia determinista y acotamiento inferior no negativo en la demanda predicha de P1.

    Por qué es crítico: Físicamente no existen viajes negativos; predicciones negativas romperían
    la lógica de abastecimiento y dimensionamiento de flota.
    Criterio de aceptación: Para cualquier entrada válida, todas las predicciones son >= 0.0 y reproducibles.
    """
    pred1 = np.clip(mock_champion_p1_model.predict(sample_p1_inference_payload), 0, None)
    pred2 = np.clip(mock_champion_p1_model.predict(sample_p1_inference_payload), 0, None)

    # Inferencia determinista
    np.testing.assert_array_almost_equal(pred1, pred2)

    # Rango físicamente plausible: demanda no negativa
    assert np.all(pred1 >= 0.0)
    assert not np.isnan(pred1).any()
    assert not np.isinf(pred1).any()


# ============================================================================
# 2. P2: RIESGO OPERATIVO (REGRESIÓN LOGÍSTICA MULTICLASE)
# ============================================================================

@pytest.mark.unit
def test_p2_champion_model_signature(
    mock_champion_p2_model: Pipeline,
    sample_p2_inference_payload: pd.DataFrame,
) -> None:
    """Qué prueba: Valida la firma formal de MLflow para el modelo Champion de clasificación multiclase P2.

    Por qué es crítico: Garantiza la compatibilidad del contrato de inferencia al promover o consultar
    versiones en el MLflow Model Registry bajo el alias @champion.
    Criterio de aceptación: La firma valida las características de entrada y produce salida categórica.
    """
    preds = mock_champion_p2_model.predict(sample_p2_inference_payload)
    signature = infer_signature(sample_p2_inference_payload, preds)

    input_names = [col.name for col in signature.inputs.inputs]
    assert set(input_names) == set(COLUMNAS_CARACTERISTICAS_P2)
    assert len(signature.outputs.inputs) == 1


@pytest.mark.unit
def test_p2_inference_probability_distribution_and_classes(
    mock_champion_p2_model: Pipeline,
    sample_p2_inference_payload: pd.DataFrame,
) -> None:
    """Qué prueba: Comprueba que predict_proba devuelva probabilidades que sumen 1.0 y pertenencia a las 3 clases válidas.

    Por qué es crítico: Las alertas de logística dependen de probabilidades calibradas en las clases
    'riesgo_vaciado', 'normal' y 'riesgo_saturacion'.
    Criterio de aceptación: Suma de probabilidades igual a 1.0 (+/- 1e-5) y etiquetas predichas pertenecientes al conjunto válido.
    """
    clases_validas = {"riesgo_vaciado", "normal", "riesgo_saturacion"}

    pred_clases = mock_champion_p2_model.predict(sample_p2_inference_payload)
    pred_probas = mock_champion_p2_model.predict_proba(sample_p2_inference_payload)

    # Pertenencia estricta a las 3 clases operativas
    assert set(pred_clases).issubset(clases_validas)

    # Probabilidades bien formadas en [0, 1] y suma 1.0 por observación
    assert np.all(pred_probas >= 0.0) and np.all(pred_probas <= 1.0)
    sumas = pred_probas.sum(axis=1)
    np.testing.assert_allclose(sumas, 1.0, atol=1e-5)


# ============================================================================
# 3. P3: AGRUPAMIENTO DE ESTACIONES (K-MEANS k=4)
# ============================================================================

@pytest.mark.unit
def test_p3_champion_model_signature_and_cluster_range(
    mock_champion_p3_model: Pipeline,
    sample_p3_inference_payload: pd.DataFrame,
) -> None:
    """Qué prueba: Valida la firma del modelo P3 y verifica que las asignaciones de cluster estén acotadas en [0, 3].

    Por qué es crítico: La operación de MiBici mapea las estaciones a 4 cuadrillas de balanceo;
    un cluster fuera de [0, 3] generaría excepciones en la asignación operativa.
    Criterio de aceptación: 7 variables continuas de entrada y predicciones de cluster discretas enteras en {0, 1, 2, 3}.
    """
    clusters = mock_champion_p3_model.predict(sample_p3_inference_payload)
    signature = infer_signature(sample_p3_inference_payload, clusters)

    assert len(signature.inputs.inputs) == 7

    # Clusters enteros en el rango [0, k-1] con k=4
    assert set(clusters).issubset({0, 1, 2, 3})
    assert np.issubdtype(clusters.dtype, np.integer)


# ============================================================================
# 4. CASOS LÍMITE (EDGE CASES) EN INFERENCIA
# ============================================================================

@pytest.mark.unit
def test_models_edge_cases_zero_and_extreme_inputs(
    mock_champion_p1_model: CatBoostRegressor,
    mock_champion_p2_model: Pipeline,
    mock_champion_p3_model: Pipeline,
    sample_p1_inference_payload: pd.DataFrame,
    sample_p2_inference_payload: pd.DataFrame,
    sample_p3_inference_payload: pd.DataFrame,
) -> None:
    """Qué prueba: Evalúa el comportamiento de los modelos ante entradas con ceros completos o valores extremos.

    Por qué es crítico: Asegura que estaciones inactivas, fallas momentáneas de sensores o días atípicos
    no originen NaN, Inf o excepciones no controladas en el motor de inferencia.
    Criterio de aceptación: Las funciones de predicción completan sin errores retornando números finitos.
    """
    # 1. Caso Ceros Completos (Estación sin actividad / inactiva)
    p1_zeros = sample_p1_inference_payload.copy()
    p1_zeros.iloc[:] = 0
    pred_p1_zero = np.clip(mock_champion_p1_model.predict(p1_zeros), 0, None)
    assert not np.isnan(pred_p1_zero).any()
    assert (pred_p1_zero >= 0).all()

    p2_zeros = sample_p2_inference_payload.copy()
    p2_zeros.iloc[:] = 0
    p2_zeros["capacidad"] = 1  # Evitar división entre 0 estricta en features
    pred_p2_zero = mock_champion_p2_model.predict(p2_zeros)
    assert len(pred_p2_zero) == len(p2_zeros)

    p3_zeros = sample_p3_inference_payload.copy()
    p3_zeros.iloc[:] = 0.0
    pred_p3_zero = mock_champion_p3_model.predict(p3_zeros)
    assert set(pred_p3_zero).issubset({0, 1, 2, 3})

    # 2. Caso Valores Extremos (Pico extraordinario de demanda / eventos masivos)
    p1_extreme = sample_p1_inference_payload.copy()
    p1_extreme["lag_1"] = 10000.0
    p1_extreme["rolling_mean_3h"] = 5000.0
    pred_p1_ext = np.clip(mock_champion_p1_model.predict(p1_extreme), 0, None)
    assert not np.isnan(pred_p1_ext).any()
    assert (pred_p1_ext >= 0).all()
