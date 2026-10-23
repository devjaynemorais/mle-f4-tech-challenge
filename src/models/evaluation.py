"""Ajuste, avaliação e seleção do candidato vencedor."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models.candidates import candidate_estimators
from src.models.preprocessing import build_pipeline
from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """Resultado da validação cruzada de um modelo candidato."""

    name: str
    roc_auc: float
    roc_auc_std: float
    fold_scores: tuple[float, ...]


def _fold_score(
    name: str,
    features: pd.DataFrame,
    target: pd.Series[int],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    config: ExperimentConfig,
) -> float:
    """Ajusta todo o pipeline somente com o trecho de treino do fold."""
    x_train, x_valid = features.iloc[train_idx], features.iloc[valid_idx]
    y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]
    estimator = candidate_estimators(y_train, config)[name]
    pipeline = build_pipeline(x_train, estimator)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_valid)[:, 1]
    return float(roc_auc_score(y_valid, probabilities))


def evaluate_candidate(
    name: str,
    features: pd.DataFrame,
    target: pd.Series[int],
    config: ExperimentConfig,
) -> Candidate:
    """Compara um candidato por folds estratificados da referência."""
    if target.nunique() != 2:
        raise ValueError("a validação cruzada requer duas classes no alvo")
    if target.value_counts().min() < config.model.cv_folds:
        raise ValueError("classes insuficientes para a validação cruzada")
    splitter = StratifiedKFold(
        n_splits=config.model.cv_folds,
        shuffle=True,
        random_state=config.model.random_state,
    )
    scores = tuple(
        _fold_score(name, features, target, train_idx, valid_idx, config)
        for train_idx, valid_idx in splitter.split(features, target)
    )
    result = Candidate(name, float(np.mean(scores)), float(np.std(scores)), scores)
    logger.info(
        "Candidato %s: ROC-AUC CV %.4f ± %.4f", name, result.roc_auc, result.roc_auc_std
    )
    return result


def fit_selected(
    candidate: Candidate,
    features: pd.DataFrame,
    target: pd.Series[int],
    config: ExperimentConfig,
) -> Pipeline:
    """Ajusta o modelo escolhido com toda a referência."""
    estimator = candidate_estimators(target, config)[candidate.name]
    pipeline = build_pipeline(features, estimator)
    pipeline.fit(features, target)
    return pipeline


def select_best(candidates: list[Candidate]) -> Candidate:
    """Seleciona o candidato de maior ROC-AUC médio na validação cruzada."""
    best = max(candidates, key=lambda candidate: candidate.roc_auc)
    logger.info("Modelo escolhido: %s (ROC-AUC CV %.4f)", best.name, best.roc_auc)
    return best
