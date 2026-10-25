"""Gráficos e tabelas para inspecionar resultados de drift."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike

from src.drift.detect import DriftReport, DriftResult

SUMMARY_COLUMNS = [
    "feature",
    "method",
    "statistic",
    "p_value",
    "threshold",
    "drift_detected",
]


def _numeric_values(values: ArrayLike) -> np.ndarray:
    """Converte a entrada para floats e remove ausentes e infinitos."""
    series = pd.Series(np.asarray(values).ravel())
    array = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    return array[np.isfinite(array)]


def _numeric_pair(
    reference: ArrayLike, current: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Prepara e valida duas amostras numéricas para os gráficos."""
    ref, cur = _numeric_values(reference), _numeric_values(current)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("ambos os períodos precisam ter valores numéricos válidos")
    return ref, cur


def _save_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Cria o diretório pai e salva a figura quando um caminho é informado."""
    if save_path is None:
        return
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight")


def plot_distribution_comparison(
    reference: ArrayLike,
    current: ArrayLike,
    feature_name: str = "feature",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Sobrepõe histogramas de densidade com limites de bins compartilhados."""
    ref, cur = _numeric_pair(reference, current)
    edges = np.histogram_bin_edges(np.concatenate((ref, cur)), bins=50)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ref, bins=edges, alpha=0.5, density=True, label="Referência")
    ax.hist(cur, bins=edges, alpha=0.5, density=True, label="Produção")
    ax.set(xlabel=feature_name, ylabel="Densidade")
    ax.set_title(title or f"Comparação de distribuições — {feature_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_cdf_comparison(
    reference: ArrayLike,
    current: ArrayLike,
    feature_name: str = "feature",
    ks_statistic: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota CDFs empíricas e, se informado, anota a estatística D do KS."""
    ref, cur = _numeric_pair(reference, current)
    ref_sorted, cur_sorted = np.sort(ref), np.sort(cur)
    ref_cdf = np.arange(1, ref.size + 1) / ref.size
    cur_cdf = np.arange(1, cur.size + 1) / cur.size
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ref_sorted, ref_cdf, label="CDF Referência", linewidth=2)
    ax.plot(cur_sorted, cur_cdf, label="CDF Produção", linewidth=2)
    if ks_statistic is not None:
        ax.text(0.65, 0.15, f"KS D = {ks_statistic:.4f}", transform=ax.transAxes)
    ax.set(xlabel=feature_name, ylabel="Probabilidade acumulada")
    ax.set_title(f"CDFs empíricas — {feature_name} (KS)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def _psi_color(value: float) -> str:
    """Escolhe cor de barra de acordo com os intervalos usuais de PSI."""
    if value < 0.10:
        return "forestgreen"
    if value < 0.25:
        return "darkorange"
    return "firebrick"


def plot_psi_bars(
    report: DriftReport,
    title: str = "PSI por feature",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Mostra PSI por feature e os limiares de 0,10 e 0,25."""
    results = [item for item in report.results if item.method == "psi"]
    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.35)))
    if not results:
        ax.text(0.5, 0.5, "Sem resultados PSI", ha="center", va="center")
        _save_figure(fig, save_path)
        return fig
    features = [item.feature_name for item in results]
    values = [item.statistic for item in results]
    ax.barh(features, values, color=[_psi_color(value) for value in values])
    ax.axvline(0.10, color="orange", linestyle="--", label="Moderado (0,10)")
    ax.axvline(0.25, color="red", linestyle="--", label="Severo (0,25)")
    ax.set(xlabel="PSI", title=title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def _metric_value(report_item: DriftResult, metric: str) -> float:
    """Extrai a métrica solicitada sem converter p-valor ausente em zero."""
    if metric == "statistic":
        return float(report_item.statistic)
    if metric == "p_value":
        return np.nan if report_item.p_value is None else report_item.p_value
    if metric == "drift_detected":
        return float(report_item.drift_detected)
    raise ValueError("metric deve ser statistic, p_value ou drift_detected")


def _report_matrix(report: DriftReport, metric: str) -> pd.DataFrame:
    """Organiza feature e método em uma matriz para o heatmap."""
    values: dict[str, dict[str, float]] = {}
    for item in report.results:
        feature = values.setdefault(item.feature_name, {})
        feature[item.method] = _metric_value(item, metric)
    return pd.DataFrame.from_dict(values, orient="index").sort_index()


def plot_drift_heatmap(
    report: DriftReport,
    metric: str = "statistic",
    title: str = "Heatmap de drift por feature e método",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plota estatísticas, p-valores ou decisões em uma matriz de drift."""
    data = _report_matrix(report, metric)
    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.5)))
    if data.empty:
        ax.text(0.5, 0.5, "Sem resultados de drift", ha="center", va="center")
        _save_figure(fig, save_path)
        return fig
    fmt = ".0f" if metric == "drift_detected" else ".4g"
    sns.heatmap(data, annot=True, fmt=fmt, cmap="YlOrRd", mask=data.isna(), ax=ax)
    ax.set(title=title, ylabel="Feature", xlabel="Método")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def generate_drift_summary(report: DriftReport) -> pd.DataFrame:
    """Converte os resultados do relatório em uma tabela por feature/método."""
    rows = [
        {
            "feature": item.feature_name,
            "method": item.method,
            "statistic": item.statistic,
            "p_value": item.p_value,
            "threshold": item.threshold,
            "drift_detected": item.drift_detected,
        }
        for item in report.results
    ]
    return pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)
