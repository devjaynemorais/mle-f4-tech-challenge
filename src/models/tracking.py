"""Tracking opcional do treino no MLflow."""

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


def _record_run(
    candidates: list[Candidate],
    best: Candidate,
    artifact_path: Path,
    config: ExperimentConfig,
) -> None:
    """Envia parâmetros, métricas e artefato em uma run ativa."""
    mlflow.log_params(
        {
            "sample_size": config.model.sample_size,
            "test_size": config.model.test_size,
            "random_state": config.model.random_state,
            "selected_model": best.name,
        }
    )
    for candidate in candidates:
        mlflow.log_metric(f"roc_auc_{candidate.name}", candidate.roc_auc)
    mlflow.log_metric("roc_auc_best", best.roc_auc)
    mlflow.set_tag("selected_model", best.name)
    mlflow.log_artifact(str(artifact_path), artifact_path="model")


def log_to_mlflow(
    candidates: list[Candidate],
    best: Candidate,
    artifact_path: Path,
    config: ExperimentConfig,
) -> None:
    """Registra o treino sem bloquear a persistência em caso de falha."""
    if mlflow is None:
        logger.warning("mlflow não instalado; tracking desativado")
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.mlflow.experiment)
        with mlflow.start_run(run_name=best.name):
            _record_run(candidates, best, artifact_path, config)
        logger.info("Run registrada no MLflow em %s", uri)
    except Exception as error:  # noqa: BLE001 - tracking não deve quebrar o treino
        logger.warning("Falha ao registrar no MLflow (%s): %s", uri, error)
