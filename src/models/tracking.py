"""Tracking opcional dos modelos e métricas no MLflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:  # tracking é opcional no ambiente local
    import mlflow
except ImportError:  # pragma: no cover - depende do ambiente de execução
    mlflow = None

from src.models.evaluation import Candidate
from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


def _log_candidate_metrics(candidate: Candidate) -> None:
    """Registra médias e desvios por métrica e modelo."""
    for name, summary in candidate.summarize_metrics().items():
        mlflow.log_metric(f"cv_{name}_mean_{candidate.name}", summary["mean"])
        mlflow.log_metric(f"cv_{name}_std_{candidate.name}", summary["std"])


def _record_run(
    candidates: list[Candidate],
    artifact_paths: dict[str, Path],
    config: ExperimentConfig,
) -> None:
    """Envia configuração, métricas e os dois pipelines em uma run."""
    mlflow.log_params(
        {
            "production_size": config.model.production_size,
            "cv_folds": config.model.cv_folds,
            "random_state": config.model.random_state,
            "classification_threshold": config.model.prediction_threshold,
            "models": ",".join(item.name for item in candidates),
        }
    )
    for candidate in candidates:
        _log_candidate_metrics(candidate)
        mlflow.log_artifact(
            str(artifact_paths[candidate.name]),
            artifact_path=f"models/{candidate.name}",
        )


def log_to_mlflow(
    candidates: list[Candidate],
    artifact_paths: dict[str, Path],
    config: ExperimentConfig,
) -> None:
    """Registra ambos os modelos sem bloquear a persistência local."""
    if mlflow is None:
        logger.warning("mlflow não instalado; tracking desativado")
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.mlflow.experiment)
        with mlflow.start_run(run_name="dual-model-training"):
            _record_run(candidates, artifact_paths, config)
        logger.info("Run de modelos registrada no MLflow em %s", uri)
    except Exception as error:  # noqa: BLE001 - tracking não deve quebrar o treino
        logger.warning("Falha ao registrar no MLflow (%s): %s", uri, error)
