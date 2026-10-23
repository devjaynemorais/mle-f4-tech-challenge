"""Ajuste, avaliação e seleção do candidato vencedor."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.models.dataset import Splits
from src.models.preprocessing import build_pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """Resultado de validação de um modelo candidato."""

    name: str
    pipeline: Pipeline
    roc_auc: float


def evaluate_candidate(
    name: str, estimator: object, features: pd.DataFrame, splits: Splits
) -> Candidate:
    """Treina um candidato e mede o ROC-AUC de validação."""
    x_train, x_valid, y_train, y_valid = splits
    pipeline = build_pipeline(features, estimator)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_valid)[:, 1]
    roc_auc = float(roc_auc_score(y_valid, probabilities))
    logger.info("Candidato %s: ROC-AUC de validação %.4f", name, roc_auc)
    return Candidate(name=name, pipeline=pipeline, roc_auc=roc_auc)


def select_best(candidates: list[Candidate]) -> Candidate:
    """Seleciona o candidato de maior ROC-AUC de validação."""
    best = max(candidates, key=lambda candidate: candidate.roc_auc)
    logger.info("Modelo escolhido: %s (ROC-AUC %.4f)", best.name, best.roc_auc)
    return best
