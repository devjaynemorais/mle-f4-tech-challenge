"""Estimadores candidatos e configuração de desbalanceamento."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.config_loader import ExperimentConfig


def candidate_estimators(
    target: pd.Series[int], config: ExperimentConfig
) -> dict[str, object]:
    """Cria regressão logística e XGBoost com pesos de classe."""
    seed = config.model.random_state
    positives = int(target.sum())
    scale_pos_weight = (len(target) - positives) / max(positives, 1)
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="auc",
            random_state=seed,
        ),
    }
