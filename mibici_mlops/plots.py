"""Generación de visualizaciones y artefactos de desempeño para MLflow."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

plt.switch_backend("Agg")


def plot_p1_real_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "P1: Demanda Real vs. Predicha",
    max_points: int = 5000,
) -> Path:
    """Genera gráfica de Real vs. Predicho para regresión."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    if len(y_true) > max_points:
        rng = np.random.default_rng(777)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred

    ax.scatter(yt, yp, alpha=0.3, color="#1f77b4", edgecolors="none", s=20)
    max_val = max(float(np.max(yt)), float(np.max(yp)), 10.0)
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Ideal (y = x)")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Demanda Real (viajes/hora)")
    ax.set_ylabel("Demanda Predicha (viajes/hora)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_p1_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "P1: Gráfico de Residuales",
    max_points: int = 5000,
) -> Path:
    """Genera gráfica de residuales vs. valores predichos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    if len(y_true) > max_points:
        rng = np.random.default_rng(777)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred

    residuals = yt - yp
    ax.scatter(yp, residuals, alpha=0.3, color="#ff7f0e", edgecolors="none", s=20)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Demanda Predicha (viajes/hora)")
    ax.set_ylabel("Residual (Real - Predicho)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_p2_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str],
    output_path: Path,
    title: str = "P2: Matriz de Confusión",
) -> Path:
    """Genera matriz de confusión normalizada."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_p2_roc_curves(
    y_true: Sequence[str],
    y_proba: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str = "P2: Curvas ROC Multiclase (One-vs-Rest)",
) -> Path:
    """Genera curvas ROC para clasificación multiclase."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_bin = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            linewidth=1.8,
            label=f"{label} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_p3_spatial_clusters(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
    title: str = "P3: Agrupamiento Espacial de Estaciones",
) -> Path:
    """Genera mapa espacial de clusters de estaciones."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        features_df["longitude"],
        features_df["latitude"],
        c=labels,
        cmap="tab10",
        s=45,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_p3_cluster_distribution(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
    title: str = "P3: Conteo y Distribución de Estaciones por Cluster",
) -> Path:
    """Genera histograma / perfil de conteo por cluster."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = pd.Series(labels).value_counts().sort_index()
    bars = ax.bar(
        [f"Cluster {c}" for c in counts.index],
        counts.values,
        color="#2b5c8f",
        alpha=0.85,
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Número de Estaciones")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
