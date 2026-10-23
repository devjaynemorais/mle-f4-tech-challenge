"""Extrai as entradas do modelo da referência processada."""

from __future__ import annotations

import pandas as pd

from src.models.features import DERIVED_FEATURE_COLUMNS
from src.utils.config_loader import ExperimentConfig


def split_features_target(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.Series[int]]:
    """Remove ID, alvo e derivadas que o pipeline recalcula na inferência."""
    data = config.data
    target = frame[data.target_column].astype(int)
    features = frame.drop(
        columns=[data.id_column, data.target_column, *DERIVED_FEATURE_COLUMNS],
        errors="ignore",
    )
    return features, target
