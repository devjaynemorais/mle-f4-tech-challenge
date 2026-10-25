"""Orquestra preparação dos dados, treino e persistência dos modelos."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.ingestion import gate_batch
from src.data.loader import load_raw_dataset
from src.data.processed import prepare_datasets
from src.models.artifacts import persist_pipeline, write_metrics
from src.models.candidates import candidate_estimators
from src.models.dataset import split_features_target
from src.models.evaluation import Candidate, evaluate_candidate, fit_model
from src.models.features import engineer_features
from src.models.preprocessing import reindex_features
from src.models.tracking import log_to_mlflow
from src.observability.stage_tracking import track_stage
from src.utils.config_loader import ExperimentConfig, load_config
from src.utils.logging_config import configure_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)
__all__ = ["engineer_features", "reindex_features", "train", "main"]


def _artifact_files(config: ExperimentConfig) -> dict[str, str]:
    """Relaciona os nomes lógicos dos modelos aos seus arquivos."""
    return {
        "xgboost": config.artifacts.model_file,
        "logistic_regression": config.artifacts.baseline_file,
    }


def _fit_and_persist_models(
    names: list[str],
    features: pd.DataFrame,
    target: pd.Series[int],
    config: ExperimentConfig,
) -> dict[str, Path]:
    """Ajusta cada pipeline na referência e grava os dois modelos."""
    artifacts = _artifact_files(config)
    paths = {}
    for name in names:
        pipeline: Pipeline = fit_model(name, features, target, config)
        paths[name] = persist_pipeline(pipeline, config, artifacts[name])
    return paths


def _evaluate_models(
    names: list[str],
    features: pd.DataFrame,
    target: pd.Series[int],
    config: ExperimentConfig,
) -> list[Candidate]:
    """Avalia os modelos com os mesmos folds estratificados."""
    return [evaluate_candidate(name, features, target, config) for name in names]


def _ingest_and_transform(cfg: ExperimentConfig) -> pd.DataFrame:
    """Lê o CSV bruto, aplica o contrato e gera a referência de treino."""
    with track_stage("ingestion", "raw") as stage:
        raw = load_raw_dataset(config=cfg)
        stage.rows = len(raw)
    gate_batch(raw, cfg, "raw")
    with track_stage("transformation", "raw") as stage:
        reference, _ = prepare_datasets(raw, cfg)
        stage.rows = len(reference)
    return reference


def _train_models(reference: pd.DataFrame, cfg: ExperimentConfig) -> list[Candidate]:
    """Avalia, ajusta e persiste os modelos sob rastreio da etapa de treino."""
    with track_stage("training", "reference") as stage:
        features, target = split_features_target(reference, cfg)
        model_names = list(candidate_estimators(target, cfg))
        evaluations = _evaluate_models(model_names, features, target, cfg)
        paths = _fit_and_persist_models(model_names, features, target, cfg)
        write_metrics(evaluations, cfg, _artifact_files(cfg))
        stage.rows = len(features)
        stage.metrics.update({f"cv_roc_auc_{e.name}": e.roc_auc for e in evaluations})
    log_to_mlflow(evaluations, paths, cfg)
    return evaluations


def train(config: ExperimentConfig | None = None) -> list[Candidate]:
    """Treina e persiste XGBoost e regressão logística na referência."""
    cfg = config or load_config()
    set_seed(cfg.model.random_state)
    return _train_models(_ingest_and_transform(cfg), cfg)


def main() -> None:
    """Ponto de entrada de `make train`: treina e persiste ambos os modelos."""
    configure_logging()
    evaluations = train()
    names = ", ".join(item.name for item in evaluations)
    logger.info("Treino concluído; modelos prontos para servir e comparar: %s", names)


if __name__ == "__main__":
    main()
