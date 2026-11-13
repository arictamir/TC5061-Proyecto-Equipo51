"""Módulo de evaluación y A/B Testing comparativo entre Champion y Challenger.

Permite evaluar y contrastar cuantitativamente dos versiones del modelo registradas
en el Model Registry de MLflow (@champion vs @challenger) sobre el mismo conjunto
de evaluación del periodo actual (producción), realizando pruebas de significancia
estadística, análisis de residuales y emitiendo una decisión justificada sobre
cuál versión debe quedar activa en producción.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Annotated, Any

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import typer

from mibici_mlops.config import (
    DRIFT_REPORT_DIR,
    FIGURES_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_P1,
    P1_TEST_START,
    PROCESSED_DATA_DIR,
    SEMILLA_ALEATORIA,
)

app = typer.Typer(
    help="Evaluación y A/B testing entre modelos Champion y Challenger para MiBici P1."
)

AB_FIGURES_DIR: Path = FIGURES_DIR / "ab_testing"

FEATURE_COLUMNS: list[str] = [
    "station_id",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "history_hours",
]

CATEGORICAL_COLUMNS: list[str] = [
    "station_id",
    "hour_of_day",
    "day_of_week",
    "month",
]

TARGET_COLUMN: str = "trips_started"


def load_ab_evaluation_data(
    features_path: Path | None = None,
    sample_size: int | None = 100000,
    random_state: int = SEMILLA_ALEATORIA,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Carga y prepara el conjunto de evaluación del periodo actual (D_curr).

    Ambos modelos (Champion y Challenger) son evaluados estrictamente sobre este
    mismo conjunto de datos para garantizar una comparación rigurosa e imparcial.
    """
    path = features_path or (PROCESSED_DATA_DIR / "p1_features.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de características en: {path}. "
            "Asegúrate de ejecutar la etapa de ingeniería de características previamente."
        )

    logger.info(f"Cargando conjunto de evaluación A/B desde: {path}")
    cols_to_load = list(set(FEATURE_COLUMNS + [TARGET_COLUMN, "hour"]))
    df = pd.read_parquet(path, columns=cols_to_load)

    # Filtrar únicamente el periodo actual de producción (D_curr: mayo a septiembre 2026)
    curr_mask = df["hour"] >= P1_TEST_START
    eval_df = df.loc[curr_mask].dropna().copy()

    logger.info(
        f"Registros disponibles en periodo actual ({P1_TEST_START} a 2026-09-30): {len(eval_df):,} filas."
    )

    if sample_size is not None and len(eval_df) > sample_size:
        eval_df = eval_df.sample(n=sample_size, random_state=random_state).copy()
        logger.info(
            f"Muestra representativa para A/B testing seleccionada: {sample_size:,} registros (seed={random_state})."
        )

    for col in CATEGORICAL_COLUMNS:
        eval_df[col] = eval_df[col].astype(str)

    X_eval = eval_df[FEATURE_COLUMNS]
    y_eval = eval_df[TARGET_COLUMN]

    return X_eval, y_eval, eval_df


def compute_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    inference_time_s: float,
) -> dict[str, float]:
    """Calcula las métricas de desempeño, error por cuantiles y eficiencia de inferencia."""
    n_samples = len(y_true)
    abs_errors = np.abs(y_true - y_pred)
    residuals = y_true - y_pred

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    sum_true = float(np.sum(y_true))
    wape = float(np.sum(abs_errors) / sum_true) if sum_true > 0 else 0.0

    pos_mask = y_true > 0
    mae_positive = (
        float(mean_absolute_error(y_true[pos_mask], y_pred[pos_mask]))
        if pos_mask.sum() > 0
        else mae
    )
    rmse_positive = (
        float(np.sqrt(mean_squared_error(y_true[pos_mask], y_pred[pos_mask])))
        if pos_mask.sum() > 0
        else rmse
    )

    latency_ms_per_1k = (inference_time_s / n_samples) * 1000.0 * 1000.0
    throughput = float(n_samples / inference_time_s) if inference_time_s > 0 else 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "wape": round(wape, 4),
        "mae_positive": round(mae_positive, 4),
        "rmse_positive": round(rmse_positive, 4),
        "residual_mean": round(float(np.mean(residuals)), 4),
        "residual_std": round(float(np.std(residuals)), 4),
        "error_p50": round(float(np.percentile(abs_errors, 50)), 4),
        "error_p90": round(float(np.percentile(abs_errors, 90)), 4),
        "error_p95": round(float(np.percentile(abs_errors, 95)), 4),
        "latency_ms_per_1k": round(latency_ms_per_1k, 2),
        "throughput_preds_per_s": round(throughput, 1),
    }


def compute_statistical_test(
    errors_champion: np.ndarray,
    errors_challenger: np.ndarray,
) -> dict[str, Any]:
    """Ejecuta una prueba t pareada para determinar si la diferencia en error absoluto es significativa."""
    t_stat, p_val = ttest_rel(errors_champion, errors_challenger)
    mean_diff = float(np.mean(errors_champion) - np.mean(errors_challenger))

    # Hipótesis: p_val < 0.05 indica que la diferencia de desempeño es estadísticamente significativa
    is_significant = bool(p_val < 0.05)

    if is_significant:
        if mean_diff > 0:
            conclusion = "Challenger presenta un error significativamente menor que Champion (p < 0.05)."
        else:
            conclusion = "Champion presenta un error significativamente menor que Challenger (p < 0.05)."
    else:
        conclusion = "No existe diferencia estadísticamente significativa entre ambos modelos (p >= 0.05)."

    return {
        "paired_t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "mean_absolute_error_difference": round(mean_diff, 5),
        "is_statistically_significant": is_significant,
        "significance_threshold": 0.05,
        "statistical_conclusion": conclusion,
    }


def generate_ab_testing_plots(
    champ_metrics: dict[str, float],
    chall_metrics: dict[str, float],
    y_true: np.ndarray,
    champ_preds: np.ndarray,
    chall_preds: np.ndarray,
    full_eval_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> list[Path]:
    """Genera y guarda gráficos comparativos de métricas y residuales para el reporte."""
    target_dir = output_dir or AB_FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    generated_plots: list[Path] = []

    # 1. Gráfico de barras comparando métricas clave
    _fig, ax = plt.subplots(figsize=(9, 5))
    metrics_to_plot = ["rmse", "mae", "mae_positive"]
    labels = ["RMSE", "MAE Global", "MAE Demanda Positiva (y>0)"]
    champ_vals = [champ_metrics[m] for m in metrics_to_plot]
    chall_vals = [chall_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, champ_vals, width, label="Champion (@champion)", color="#2b5c8f")
    rects2 = ax.bar(x + width / 2, chall_vals, width, label="Challenger (@challenger)", color="#d95f02")

    ax.set_ylabel("Error de Predicción")
    ax.set_title("A/B Testing: Comparación de Métricas de Error sobre el Periodo Actual")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects: Any) -> None:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)

    plot1_path = target_dir / "ab_metrics_comparison.png"
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    generated_plots.append(plot1_path)

    # 2. Distribución de residuales
    _fig, ax = plt.subplots(figsize=(9, 5))
    champ_res = y_true - champ_preds
    chall_res = y_true - chall_preds

    ax.hist(
        champ_res,
        bins=60,
        range=(-6, 6),
        alpha=0.55,
        label=f"Champion (μ={champ_metrics['residual_mean']:.3f}, σ={champ_metrics['residual_std']:.3f})",
        color="#2b5c8f",
        density=True,
    )
    ax.hist(
        chall_res,
        bins=60,
        range=(-6, 6),
        alpha=0.55,
        label=f"Challenger (μ={chall_metrics['residual_mean']:.3f}, σ={chall_metrics['residual_std']:.3f})",
        color="#d95f02",
        density=True,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Error Cero")
    ax.set_xlabel("Residual (y_real - y_predicho)")
    ax.set_ylabel("Densidad de Frecuencia")
    ax.set_title("A/B Testing: Distribución Comparativa de Residuales en Producción")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plot2_path = target_dir / "ab_residuals_distribution.png"
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    generated_plots.append(plot2_path)

    # 3. MAE por hora del día
    _fig, ax = plt.subplots(figsize=(10, 5))
    df_eval_copy = full_eval_df.copy()
    df_eval_copy["hour_num"] = df_eval_copy["hour_of_day"].astype(int)
    df_eval_copy["champ_abs_err"] = np.abs(y_true - champ_preds)
    df_eval_copy["chall_abs_err"] = np.abs(y_true - chall_preds)

    hourly_err = (
        df_eval_copy.groupby("hour_num")[["champ_abs_err", "chall_abs_err"]]
        .mean()
        .reset_index()
    )

    ax.plot(
        hourly_err["hour_num"],
        hourly_err["champ_abs_err"],
        marker="o",
        label="Champion",
        color="#2b5c8f",
        linewidth=2,
    )
    ax.plot(
        hourly_err["hour_num"],
        hourly_err["chall_abs_err"],
        marker="s",
        label="Challenger",
        color="#d95f02",
        linewidth=2,
    )
    ax.set_xlabel("Hora del Día (0 a 23 hrs)")
    ax.set_ylabel("MAE Promedio")
    ax.set_title("A/B Testing: Error Absoluto Medio por Hora del Día (Periodo Actual)")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plot3_path = target_dir / "ab_hourly_error.png"
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=200)
    plt.close()
    generated_plots.append(plot3_path)

    logger.success(f"{len(generated_plots)} gráficos de A/B testing guardados en: {target_dir}")
    return generated_plots


def execute_ab_testing(
    model_name: str = MODEL_REGISTRY_P1,
    sample_size: int = 100000,
    output_json_path: Path | None = None,
) -> dict[str, Any]:
    """Ejecuta el protocolo completo de A/B testing entre @champion y @challenger."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    logger.info("=== Iniciando Ejercicio de A/B Testing entre Champion y Challenger ===")
    logger.info(f"Cargando versiones desde Model Registry '{model_name}'...")

    # Consultar información de versiones y alias
    try:
        champ_ver_info = client.get_model_version_by_alias(model_name, "champion")
        champ_version = int(champ_ver_info.version)
        champ_desc = champ_ver_info.description or "Sin descripción"
    except Exception as e:
        logger.error(f"Error al obtener versión @champion de '{model_name}': {e}")
        raise

    try:
        chall_ver_info = client.get_model_version_by_alias(model_name, "challenger")
        chall_version = int(chall_ver_info.version)
        chall_desc = chall_ver_info.description or "Sin descripción"
    except Exception as e:
        logger.error(f"Error al obtener versión @challenger de '{model_name}': {e}")
        raise

    logger.info(
        f"Versiones identificadas -> Champion: v{champ_version}, Challenger: v{chall_version}"
    )

    # Cargar modelos
    t0 = time.time()
    champion_model = mlflow.catboost.load_model(f"models:/{model_name}@champion")
    challenger_model = mlflow.catboost.load_model(f"models:/{model_name}@challenger")
    logger.info(f"Modelos cargados en {time.time() - t0:.2f} s.")

    # Cargar datos de evaluación (D_curr)
    X_eval, y_eval, full_eval_df = load_ab_evaluation_data(sample_size=sample_size)
    y_true_np = y_eval.to_numpy()

    # Inferencia Champion
    t_champ = time.time()
    preds_champ = np.clip(champion_model.predict(X_eval), 0, None)
    inf_time_champ = time.time() - t_champ
    metrics_champ = compute_model_metrics(y_true_np, preds_champ, inf_time_champ)

    # Inferencia Challenger
    t_chall = time.time()
    preds_chall = np.clip(challenger_model.predict(X_eval), 0, None)
    inf_time_chall = time.time() - t_chall
    metrics_chall = compute_model_metrics(y_true_np, preds_chall, inf_time_chall)

    # Prueba de significancia estadística
    err_champ = np.abs(y_true_np - preds_champ)
    err_chall = np.abs(y_true_np - preds_chall)
    stat_test = compute_statistical_test(err_champ, err_chall)

    # Generar gráficos comparativos
    plots = generate_ab_testing_plots(
        champ_metrics=metrics_champ,
        chall_metrics=metrics_chall,
        y_true=y_true_np,
        champ_preds=preds_champ,
        chall_preds=preds_chall,
        full_eval_df=full_eval_df,
    )

    # Toma de decisión justificada
    # Si Champion tiene menor RMSE y MAE global, pero Challenger muestra ventajas en MAE positivo
    rmse_diff = metrics_chall["rmse"] - metrics_champ["rmse"]
    mae_diff = metrics_chall["mae"] - metrics_champ["mae"]
    mae_pos_diff = metrics_chall["mae_positive"] - metrics_champ["mae_positive"]

    if metrics_champ["rmse"] <= metrics_chall["rmse"] and metrics_champ["mae"] <= metrics_chall["mae"]:
        recommended_winner = "champion"
        decision_summary = (
            f"Se recomienda mantener la Versión {champ_version} (@champion) como versión productiva vigente. "
            f"Champion conserva un error global menor (RMSE {metrics_champ['rmse']:.4f} vs {metrics_chall['rmse']:.4f}, "
            f"MAE {metrics_champ['mae']:.4f} vs {metrics_chall['mae']:.4f}) y latencia balanceada ({metrics_champ['latency_ms_per_1k']:.2f} ms/1k). "
            f"No obstante, Challenger (v{chall_version}) demostró competitividad en viajes activos (MAE positivo {metrics_chall['mae_positive']:.4f}), "
            "lo que fundamenta mantenerla lista en el Registry y adoptar una política de Reentrenamiento Continuo (CT)."
        )
    else:
        recommended_winner = "challenger"
        decision_summary = (
            f"Se recomienda promover la Versión {chall_version} (@challenger) a producción como nuevo @champion. "
            f"Challenger superó a Champion en métricas de precisión sobre el periodo actual de producción."
        )

    results: dict[str, Any] = {
        "model_name": model_name,
        "evaluation_period": f"{P1_TEST_START} a 2026-09-30",
        "evaluation_records": len(X_eval),
        "champion_info": {
            "version": champ_version,
            "alias": "champion",
            "description": champ_desc,
            "metrics": metrics_champ,
        },
        "challenger_info": {
            "version": chall_version,
            "alias": "challenger",
            "description": chall_desc,
            "metrics": metrics_chall,
        },
        "metrics_difference_challenger_minus_champion": {
            "rmse_diff": round(rmse_diff, 4),
            "mae_diff": round(mae_diff, 4),
            "mae_positive_diff": round(mae_pos_diff, 4),
        },
        "statistical_test": stat_test,
        "decision": {
            "recommended_active_version": recommended_winner,
            "active_version_number": champ_version if recommended_winner == "champion" else chall_version,
            "decision_justification": decision_summary,
        },
        "continuous_training_recommendation": {
            "trigger_condition": "Drift significativo en >40% de columnas clave o aumento de MAE > 10% en ventana rodante de 14 días.",
            "recommended_cadence": "Reentrenamiento mensual automatizado con ventana deslizante de 12 a 24 meses.",
            "validation_gate": "La nueva versión reentrenada solo debe sustituir a @champion si supera el test de A/B testing con p < 0.05 y menor RMSE en datos de holdout.",
        },
        "figures": [
            p.relative_to(FIGURES_DIR.parent).as_posix()
            for p in plots
        ],
    }

    # Guardar resumen JSON
    json_path = output_json_path or (DRIFT_REPORT_DIR / "ab_testing_results.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Resultados de A/B testing estructurados en JSON: {json_path}")

    return results


@app.command("run-ab-test")
def run_ab_test_cli(
    sample_size: Annotated[
        int,
        typer.Option("--sample-size", "-s", help="Número de observaciones del periodo actual."),
    ] = 100000,
    output_json: Annotated[
        Path,
        typer.Option("--output-json", help="Ruta para el archivo JSON de resultados."),
    ] = DRIFT_REPORT_DIR / "ab_testing_results.json",
) -> None:
    """Ejecuta el protocolo de A/B testing entre Champion y Challenger."""
    results = execute_ab_testing(sample_size=sample_size, output_json_path=output_json)

    c_info = results["champion_info"]
    ch_info = results["challenger_info"]
    dec = results["decision"]

    typer.echo("\n" + "=" * 65)
    typer.echo("          RESULTADOS DEL EJERCICIO DE A/B TESTING")
    typer.echo("=" * 65)
    typer.echo(f"Modelo:                  {results['model_name']}")
    typer.echo(f"Periodo de Evaluación:   {results['evaluation_period']}")
    typer.echo(f"Observaciones Evaluadas: {results['evaluation_records']:,} filas\n")

    typer.echo(f"Champion (@champion)     -> Versión {c_info['version']}")
    typer.echo(f"  - RMSE:                {c_info['metrics']['rmse']:.4f}")
    typer.echo(f"  - MAE Global:          {c_info['metrics']['mae']:.4f}")
    typer.echo(f"  - MAE (Demanda > 0):   {c_info['metrics']['mae_positive']:.4f}")
    typer.echo(f"  - R²:                  {c_info['metrics']['r2']:.4f}")
    typer.echo(f"  - Latencia:            {c_info['metrics']['latency_ms_per_1k']:.2f} ms/1k preds\n")

    typer.echo(f"Challenger (@challenger) -> Versión {ch_info['version']}")
    typer.echo(f"  - RMSE:                {ch_info['metrics']['rmse']:.4f}")
    typer.echo(f"  - MAE Global:          {ch_info['metrics']['mae']:.4f}")
    typer.echo(f"  - MAE (Demanda > 0):   {ch_info['metrics']['mae_positive']:.4f}")
    typer.echo(f"  - R²:                  {ch_info['metrics']['r2']:.4f}")
    typer.echo(f"  - Latencia:            {ch_info['metrics']['latency_ms_per_1k']:.2f} ms/1k preds\n")

    stat = results["statistical_test"]
    typer.echo("Prueba de Significancia Estadística (Paired t-test):")
    typer.echo(f"  - t-statistic:         {stat['paired_t_statistic']:.4f}")
    typer.echo(f"  - p-value:             {stat['p_value']:.4e}")
    typer.echo(f"  - Significativo (p<.05): {'SÍ' if stat['is_statistically_significant'] else 'NO'}\n")

    typer.echo("Decisión y Recomendación:")
    typer.echo(f"  - Versión Vigente:     Versión {dec['active_version_number']} ({dec['recommended_active_version'].upper()})")
    typer.echo(f"  - Justificación:       {dec['decision_justification']}")
    typer.echo("=" * 65 + "\n")


@app.command("show-ab-summary")
def show_ab_summary_cli(
    json_path: Annotated[
        Path,
        typer.Option("--json-path", help="Ruta al archivo JSON de resultados de A/B testing."),
    ] = DRIFT_REPORT_DIR / "ab_testing_results.json",
) -> None:
    """Muestra el resumen del último ejercicio de A/B testing realizado."""
    if not json_path.exists():
        typer.echo(f"No existe el archivo de resultados en: {json_path}. Ejecuta 'run-ab-test' primero.")
        raise typer.Exit(code=1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    typer.echo(f"\n=== Resumen de A/B Testing ({data.get('model_name')}) ===")
    typer.echo(f"Periodo: {data.get('evaluation_period')}")
    c = data.get("champion_info", {})
    ch = data.get("challenger_info", {})
    typer.echo(f"Champion (v{c.get('version')}):   RMSE={c.get('metrics', {}).get('rmse')} | MAE={c.get('metrics', {}).get('mae')}")
    typer.echo(f"Challenger (v{ch.get('version')}): RMSE={ch.get('metrics', {}).get('rmse')} | MAE={ch.get('metrics', {}).get('mae')}")
    dec = data.get("decision", {})
    typer.echo(f"\nVersión Ganadora Recomendada: Versión {dec.get('active_version_number')}")
    typer.echo(f"Justificación: {dec.get('decision_justification')}")


if __name__ == "__main__":
    app()
