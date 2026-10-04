"""Carrega o pipeline de crédito treinado e expõe o contrato de predição."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import settings
from src.utils.config_loader import PROJECT_ROOT, ExperimentConfig, load_config


class CreditPredictor(Protocol):
    """Contrato mínimo de um backend de inferência de risco de crédito."""

    backend: str

    def predict(self, features: Mapping[str, object]) -> tuple[float, str]:
        """Retorna a probabilidade de default e o rótulo decidido pelo limiar."""
        ...


class SklearnPredictor:
    """Backend que usa o Pipeline sklearn completo persistido com joblib."""

    backend = "sklearn"

    def __init__(self, pipeline: Pipeline, threshold: float) -> None:
        self._pipeline = pipeline
        self._threshold = threshold

    def predict(self, features: Mapping[str, object]) -> tuple[float, str]:
        """Calcula P(default) para uma linha de features e aplica o limiar."""
        frame = pd.DataFrame([dict(features)])
        probability = float(self._pipeline.predict_proba(frame)[0, 1])
        label = "default" if probability >= self._threshold else "non_default"
        return probability, label


def load_predictor(config: ExperimentConfig | None = None) -> SklearnPredictor:
    """Carrega o artefato configurado; falha se ele ainda não foi treinado."""
    cfg = config or load_config()
    path = PROJECT_ROOT / settings.model_artifacts_path / cfg.artifacts.model_file
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. Execute o treino (Etapa 1) antes."
        )
    pipeline = joblib.load(path)
    return SklearnPredictor(pipeline, cfg.model.prediction_threshold)
