"""Treino do modelo base de risco de crédito (Etapa 1).

Treina candidatos (Regressão Logística e XGBoost) a partir do CSV bruto
enquanto o contrato de referência validada está pendente. Escolhe o candidato
de maior ROC-AUC em validação e persiste o Pipeline
completo (realinhamento de colunas + engenharia de features + pré-processamento
+ classificador) em `models/artifacts/` e grava as métricas de validação em
`metrics/`. Esse é o artefato que a API serve.

A engenharia de features e o pré-processamento ficam dentro do Pipeline para
honrar o contrato de `/predict`: chaves ausentes ou nulas em uma requisição são
realinhadas, derivadas e imputadas sem quebrar a inferência.
"""

from __future__ import annotations

import logging

from src.data.loader import load_raw_dataset
from src.models.artifacts import persist_pipeline, write_metrics
from src.models.candidates import candidate_estimators
from src.models.dataset import make_splits, split_features_target, subsample
from src.models.evaluation import Candidate, evaluate_candidate, select_best
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
    frame = subsample(load_raw_dataset(), cfg)
    features, target = split_features_target(frame, cfg)
    splits = make_splits(features, target, cfg)
    estimators = candidate_estimators(splits[2], cfg)
    candidates = [
        evaluate_candidate(name, estimator, features, splits)
        for name, estimator in estimators.items()
    ]
    best = select_best(candidates)
    artifact_path = persist_pipeline(best.pipeline, cfg)
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
