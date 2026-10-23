"""Preparação dos dados de treino e validação."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)
Splits = tuple[pd.DataFrame, pd.DataFrame, "pd.Series[int]", "pd.Series[int]"]


def subsample(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Aplica a amostra configurada de modo reprodutível."""
    sample_size = config.model.sample_size
    if sample_size is None or sample_size >= len(frame):
        return frame
    logger.info("Amostrando %d de %d linhas para o treino", sample_size, len(frame))
    return frame.sample(n=sample_size, random_state=config.model.random_state)


def split_features_target(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.Series[int]]:
    """Remove identificador e alvo das features de treino."""
    data = config.data
    target = frame[data.target_column].astype(int)
    features = frame.drop(columns=[data.id_column, data.target_column])
    return features, target


def make_splits(
    features: pd.DataFrame, target: pd.Series[int], config: ExperimentConfig
) -> Splits:
    """Divide o conjunto de forma estratificada pelo alvo."""
    return train_test_split(
        features,
        target,
        test_size=config.model.test_size,
        stratify=target,
        random_state=config.model.random_state,
    )
