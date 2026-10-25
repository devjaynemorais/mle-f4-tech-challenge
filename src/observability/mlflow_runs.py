"""Uma run do MLflow por execução de etapa do pipeline.

Só ativa quando MLFLOW_TRACKING_URI está definido (o compose define), para não
esperar por um servidor inexistente no ambiente local. Qualquer falha vira
aviso: o MLflow nunca derruba o pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:  # o MLflow é opcional fora do container
    import mlflow
except ImportError:  # pragma: no cover - depende do ambiente de execução
    mlflow = None

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def tracking_uri() -> str:
    """URI explícito do servidor de tracking; vazio desativa o registro."""
    return os.environ.get("MLFLOW_TRACKING_URI", "").strip()


def _log_run(
    stage: str,
    params: dict[str, str],
    metrics: dict[str, float],
    artifacts: list[Path],
) -> None:
    """Grava parâmetros, métricas e artefatos na run corrente."""
    mlflow.log_params({"stage": stage, **params})
    mlflow.set_tag("pipeline_status", params.get("status", "unknown"))
    mlflow.log_metrics(metrics)
    for artifact in artifacts:
        if artifact.is_file():
            mlflow.log_artifact(str(artifact), artifact_path=stage)


def log_stage_run(
    stage: str,
    params: dict[str, str],
    metrics: dict[str, float],
    artifacts: list[Path] | None = None,
) -> bool:
    """Registra a etapa no MLflow; devolve False se desativado ou com falha."""
    uri = tracking_uri()
    if mlflow is None or not uri:
        return False
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(load_config().observability.mlflow_experiment)
        with mlflow.start_run(run_name=f"{stage}-{params.get('run_id', '')}"):
            _log_run(stage, params, metrics, artifacts or [])
    except Exception as error:  # noqa: BLE001 - tracking não pode quebrar o pipeline
        logger.warning("Falha ao registrar run no MLflow (%s)", type(error).__name__)
        return False
    return True
