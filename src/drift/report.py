"""Gera relatórios HTML e tabelas de detecção para os períodos de produção."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from evidently import Report as EvidentlyReport
from evidently.presets import DataDriftPreset

from src.drift.detect import DriftDetector, DriftReport
from src.drift.evaluation import generate_drift_summary
from src.observability.alerts import Alert, drift_alerts, emit_alerts
from src.observability.drift_metrics import publish_drift, summarize_drift
from src.observability.events import PipelineEvent, current_run_id, emit
from src.observability.stage_tracking import track_stage
from src.utils.config_loader import PROJECT_ROOT, ExperimentConfig, load_config


def _validate_periods(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> None:
    """Valida esquema, dados e colunas de identificação antes da comparação."""
    if reference.columns.tolist() != current.columns.tolist():
        raise ValueError("esquema incompatível entre os períodos")
    required = {id_column, target_column}
    if not required.issubset(reference.columns):
        raise ValueError("colunas de identificação ou alvo ausentes")
    if reference.empty or current.empty:
        raise ValueError("um dos períodos está vazio")
    features = reference.drop(columns=[id_column, target_column]).columns
    if not len(features):
        raise ValueError("não há features para comparar")
    for feature in features:
        ref_numeric = pd.api.types.is_numeric_dtype(reference[feature])
        cur_numeric = pd.api.types.is_numeric_dtype(current[feature])
        if ref_numeric != cur_numeric:
            raise ValueError("tipos incompatíveis entre os períodos")


def load_feature_periods(
    reference_path: Path,
    current_path: Path,
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os períodos e remove ID e alvo para comparar apenas features."""
    if not reference_path.is_file() or not current_path.is_file():
        raise FileNotFoundError("arquivo de período não encontrado")
    reference, current = pd.read_csv(reference_path), pd.read_csv(current_path)
    _validate_periods(reference, current, id_column, target_column)
    excluded = [id_column, target_column]
    return reference.drop(columns=excluded), current.drop(columns=excluded)


def _save_summary(summary: pd.DataFrame, path: Path) -> None:
    """Cria o diretório de destino e grava o resumo sem dados individuais."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)


def _save_evidently_report(
    reference: pd.DataFrame, current: pd.DataFrame, path: Path
) -> None:
    """Executa o preset de drift do Evidently e exporta o snapshot HTML."""
    snapshot = EvidentlyReport([DataDriftPreset()]).run(
        current_data=current, reference_data=reference
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))


def generate_report(
    reference_path: Path,
    current_path: Path,
    html_path: Path,
    summary_path: Path,
    id_column: str,
    target_column: str,
    alpha: float,
    psi_threshold: float,
    n_bins: int,
) -> DriftReport:
    """Gera o HTML Evidently e CSV estatístico para as features dos períodos."""
    reference, current = load_feature_periods(
        reference_path, current_path, id_column, target_column
    )
    detector = DriftDetector(alpha=alpha, psi_threshold=psi_threshold, n_bins=n_bins)
    report = detector.detect_all(
        reference, current, include_columns=reference.columns.tolist()
    )
    summary = generate_drift_summary(report)
    _save_evidently_report(reference, current, html_path)
    _save_summary(summary, summary_path)
    return report


def configured_paths(config: ExperimentConfig) -> tuple[Path, Path, Path, Path]:
    """Resolve entradas processadas e saídas relativas à raiz do projeto."""
    processed = PROJECT_ROOT / config.data.processed_path
    return (
        processed / config.drift.stable_file,
        processed / config.drift.drifted_file,
        PROJECT_ROOT / config.drift.report_html,
        PROJECT_ROOT / config.drift.summary_csv,
    )


def baseline_paths(config: ExperimentConfig) -> tuple[Path, Path, Path, Path]:
    """Resolve a comparação de sanidade: referência de treino x período estável."""
    processed = PROJECT_ROOT / config.data.processed_path
    return (
        processed / config.data.reference_file,
        processed / config.drift.stable_file,
        PROJECT_ROOT / config.drift.baseline_report_html,
        PROJECT_ROOT / config.drift.baseline_summary_csv,
    )


def _observe(
    report: DriftReport, comparison: str, config: ExperimentConfig
) -> list[Alert]:
    """Publica métricas, evento de resultado e alertas de uma comparação."""
    summary = summarize_drift(report)
    alerts = drift_alerts(summary, comparison, config.observability)
    details = {
        "comparison": comparison,
        "n_features": summary.totals.get("psi", 0),
        "n_drifted": summary.drifted.get("psi", 0),
        "share": round(summary.share("psi"), 4),
        "max_psi": round(summary.max_psi, 4),
        "features": report.drifted_features[:50],
    }
    run_id = current_run_id()
    emit(PipelineEvent(run_id, "drift_detection", "drift_result", details=details))
    emit_alerts(alerts)
    publish_drift(comparison, summary, len(alerts))
    return alerts


def run_comparison(
    comparison: str,
    paths: tuple[Path, Path, Path, Path],
    config: ExperimentConfig,
) -> DriftReport:
    """Executa uma comparação sob rastreio da etapa de detecção de drift."""
    reference, current, html_path, summary_path = paths
    with track_stage("drift_detection", comparison) as stage:
        report = generate_report(
            reference,
            current,
            html_path,
            summary_path,
            config.data.id_column,
            config.data.target_column,
            config.drift.alpha,
            config.drift.psi_threshold,
            config.drift.n_bins,
        )
        alerts = _observe(report, comparison, config)
        stage.rows = len(pd.read_csv(current, usecols=[config.data.id_column]))
        stage.metrics["drifted_features"] = float(report.n_features_drifted)
        stage.metrics["alerts"] = float(len(alerts))
        stage.artifacts += [html_path, summary_path]
    return report


def main(argv: list[str] | None = None) -> None:
    """Executa a geração do relatório com os parâmetros do YAML do projeto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        _main_from_config(load_config())
    except Exception:
        raise SystemExit(
            "Falha ao gerar relatório de drift; verifique arquivos e esquema."
        ) from None


def _main_from_config(config: ExperimentConfig) -> None:
    """Gera a comparação de sanidade e a comparação estável x driftado."""
    run_comparison("reference_vs_stable", baseline_paths(config), config)
    run_comparison("stable_vs_drifted", configured_paths(config), config)
    print("Relatórios de drift e resumos CSV gerados.")


if __name__ == "__main__":
    main()
