"""Persistência do pipeline vencedor e das métricas de validação."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.models.evaluation import Candidate
from src.utils.config_loader import PROJECT_ROOT, ExperimentConfig

METRICS_DIR = PROJECT_ROOT / "metrics"


def persist_pipeline(pipeline: Pipeline, config: ExperimentConfig) -> Path:
    """Grava o pipeline no caminho configurado para a API."""
    artifacts = config.artifacts
    path = PROJECT_ROOT / artifacts.model_path / artifacts.model_file
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def write_metrics(candidates: list[Candidate], best: Candidate) -> Path:
    """Grava ROC-AUC por candidato e o nome do vencedor."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "training_metrics.json"
    payload = {
        "selected_model": best.name,
        "roc_auc": {candidate.name: candidate.roc_auc for candidate in candidates},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
