"""Persistência dos pipelines e das métricas de validação cruzada."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.models.evaluation import Candidate
from src.utils.config_loader import PROJECT_ROOT, ExperimentConfig

METRICS_DIR = PROJECT_ROOT / "metrics"
MODEL_METRICS_FILES = {
    "xgboost": "credit_model_metrics.json",
    "logistic_regression": "logistic_baseline_metrics.json",
}


def persist_pipeline(
    pipeline: Pipeline,
    config: ExperimentConfig,
    filename: str | None = None,
) -> Path:
    """Grava o pipeline no arquivo de artefato solicitado."""
    artifacts = config.artifacts
    target = filename or artifacts.model_file
    path = PROJECT_ROOT / artifacts.model_path / target
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def _evaluation_metadata(config: ExperimentConfig) -> dict[str, object]:
    """Descreve o protocolo usado para calcular as métricas."""
    return {
        "method": "stratified_k_fold",
        "n_splits": config.model.cv_folds,
        "random_state": config.model.random_state,
        "classification_threshold": config.model.prediction_threshold,
    }


def _model_payload(
    candidate: Candidate, artifact_file: str, config: ExperimentConfig
) -> dict[str, object]:
    """Monta o relatório de métricas de um único modelo."""
    return {
        "model": candidate.name,
        "artifact": artifact_file,
        "evaluation": _evaluation_metadata(config),
        "metrics": candidate.summarize_metrics(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Grava um relatório JSON formatado em UTF-8."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_model_metrics(
    candidates: list[Candidate],
    config: ExperimentConfig,
    artifact_files: dict[str, str],
) -> None:
    """Grava um arquivo de métricas correspondente a cada pipeline."""
    for candidate in candidates:
        filename = MODEL_METRICS_FILES.get(candidate.name)
        if filename:
            payload = _model_payload(candidate, artifact_files[candidate.name], config)
            _write_json(METRICS_DIR / filename, payload)


def write_metrics(
    candidates: list[Candidate],
    config: ExperimentConfig,
    artifact_files: dict[str, str],
) -> Path:
    """Grava métricas consolidadas e relatórios JSON por modelo."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    models = {
        candidate.name: {
            "artifact": artifact_files[candidate.name],
            "metrics": candidate.summarize_metrics(),
        }
        for candidate in candidates
    }
    payload = {"evaluation": _evaluation_metadata(config), "models": models}
    path = METRICS_DIR / "training_metrics.json"
    _write_json(path, payload)
    _write_model_metrics(candidates, config, artifact_files)
    return path
