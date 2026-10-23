"""Orquestra geração da referência, validação cruzada e treino final."""

from __future__ import annotations

import logging

from src.data.loader import load_raw_dataset
from src.data.processed import prepare_datasets
from src.models.artifacts import persist_pipeline, write_metrics
from src.models.candidates import candidate_estimators
from src.models.dataset import split_features_target
from src.models.evaluation import (
    Candidate,
    evaluate_candidate,
    fit_selected,
    select_best,
)
from src.models.features import engineer_features
from src.models.preprocessing import reindex_features
from src.models.tracking import log_to_mlflow
from src.utils.config_loader import ExperimentConfig, load_config
from src.utils.logging_config import configure_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)
__all__ = ["engineer_features", "reindex_features", "train", "main"]


def train(config: ExperimentConfig | None = None) -> Candidate:
    """Executa o treino completo e devolve o candidato vencedor."""
    cfg = config or load_config()
    set_seed(cfg.model.random_state)
    reference, _ = prepare_datasets(load_raw_dataset(config=cfg), cfg)
    features, target = split_features_target(reference, cfg)
    estimators = candidate_estimators(target, cfg)
    candidates = [
        evaluate_candidate(name, features, target, cfg) for name in estimators
    ]
    best = select_best(candidates)
    pipeline = fit_selected(best, features, target, cfg)
    artifact_path = persist_pipeline(pipeline, cfg)
    write_metrics(candidates, best)
    log_to_mlflow(candidates, best, artifact_path, cfg)
    return best


def main() -> None:
    """Ponto de entrada de `make train`: treina e persiste o artefato."""
    configure_logging()
    best = train()
    logger.info("Treino concluído; artefato pronto para servir (%s)", best.name)


if __name__ == "__main__":
    main()
