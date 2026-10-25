"""Orquestra preparação dos dados, treino e persistência dos modelos."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.loader import load_raw_dataset
from src.data.processed import prepare_datasets
from src.models.artifacts import persist_pipeline, write_metrics
from src.models.candidates import candidate_estimators
from src.models.dataset import split_features_target
from src.models.evaluation import Candidate, evaluate_candidate, fit_model
from src.models.features import engineer_features
from src.models.preprocessing import reindex_features
from src.models.tracking import log_to_mlflow
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


def train(config: ExperimentConfig | None = None) -> list[Candidate]:
    """Treina e persiste XGBoost e regressão logística na referência."""
    cfg = config or load_config()
    set_seed(cfg.model.random_state)
    reference, _ = prepare_datasets(load_raw_dataset(config=cfg), cfg)
    features, target = split_features_target(reference, cfg)
    model_names = list(candidate_estimators(target, cfg))
    evaluations = _evaluate_models(model_names, features, target, cfg)
    artifact_paths = _fit_and_persist_models(model_names, features, target, cfg)
    artifact_files = _artifact_files(cfg)
    write_metrics(evaluations, cfg, artifact_files)
    log_to_mlflow(evaluations, artifact_paths, cfg)
    return evaluations


def main() -> None:
    """Ponto de entrada de `make train`: treina e persiste ambos os modelos."""
    configure_logging()
    evaluations = train()
    names = ", ".join(item.name for item in evaluations)
    logger.info("Treino concluído; modelos prontos para servir e comparar: %s", names)


if __name__ == "__main__":
    main()
