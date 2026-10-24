"""Avaliação por validação cruzada e ajuste final dos modelos."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models.candidates import candidate_estimators
from src.models.preprocessing import build_pipeline
from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """Métricas de validação cruzada de um modelo."""

    name: str
    roc_auc: float
    roc_auc_std: float
    fold_scores: tuple[float, ...]
    fold_metrics: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def summarize_metrics(self) -> dict[str, dict[str, float | list[float]]]:
        """Resume as métricas por fold para relatórios e tracking."""
        metrics = self.fold_metrics or {"roc_auc": self.fold_scores}
        return {
            name: {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "folds": [float(score) for score in scores],
            }
            for name, scores in metrics.items()
        }


def _fold_metrics(
    target: pd.Series[int], probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Calcula métricas de classe e de probabilidade em um fold."""
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
    }


def _evaluate_fold(
    name: str,
    features: pd.DataFrame,
    target: pd.Series[int],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Ajusta o pipeline no fold de treino e avalia no fold reservado."""
    x_train, x_valid = features.iloc[train_idx], features.iloc[valid_idx]
    y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]
    estimator = candidate_estimators(y_train, config)[name]
    pipeline = build_pipeline(x_train, estimator)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_valid)[:, 1]
    return _fold_metrics(y_valid, probabilities, config.model.prediction_threshold)


def _validate_target(target: pd.Series[int], folds: int) -> None:
    """Valida as classes e a quantidade mínima para a validação cruzada."""
    if target.nunique() != 2:
        raise ValueError("a validação cruzada requer duas classes no alvo")
    if target.value_counts().min() < folds:
        raise ValueError("classes insuficientes para a validação cruzada")


def _evaluate_folds(
    name: str,
    features: pd.DataFrame,
    target: pd.Series[int],
    config: ExperimentConfig,
) -> list[dict[str, float]]:
    """Executa os mesmos folds estratificados configurados para cada modelo."""
    splitter = StratifiedKFold(
        n_splits=config.model.cv_folds,
        shuffle=True,
        random_state=config.model.random_state,
    )
    return [
        _evaluate_fold(name, features, target, train_idx, valid_idx, config)
        for train_idx, valid_idx in splitter.split(features, target)
    ]


def evaluate_candidate(
    name: str,
    features: pd.DataFrame,
    target: pd.Series[int],
    config: ExperimentConfig,
) -> Candidate:
    """Calcula as métricas de classificação para todos os folds."""
    _validate_target(target, config.model.cv_folds)
    folds = _evaluate_folds(name, features, target, config)
    fold_metrics = {
        metric: tuple(fold[metric] for fold in folds) for metric in folds[0]
    }
    roc_scores = fold_metrics["roc_auc"]
    result = Candidate(
        name,
        float(np.mean(roc_scores)),
        float(np.std(roc_scores)),
        roc_scores,
        fold_metrics,
    )
    logger.info(
        "Modelo %s: ROC-AUC CV %.4f ± %.4f", name, result.roc_auc, result.roc_auc_std
    )
    return result


def fit_model(
    name: str,
    features: pd.DataFrame,
    target: pd.Series[int],
    config: ExperimentConfig,
) -> Pipeline:
    """Ajusta um modelo com toda a referência usando o pipeline comum."""
    estimator = candidate_estimators(target, config)[name]
    pipeline = build_pipeline(features, estimator)
    pipeline.fit(features, target)
    return pipeline
