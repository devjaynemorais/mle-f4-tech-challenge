"""Carrega e valida a configuração do experimento."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ProjectConfig(BaseModel):
    """Metadados do projeto expostos pela API."""

    name: str
    version: str


class DataConfig(BaseModel):
    """Caminhos e colunas do dataset Home Credit."""

    raw_path: Path
    raw_file: str
    id_column: str
    target_column: str
    processed_path: Path
    reference_file: str
    production_file: str


class ModelConfig(BaseModel):
    """Parâmetros de decisão do modelo em produção."""

    prediction_threshold: float = Field(gt=0, lt=1)


class ArtifactConfig(BaseModel):
    """Diretório e nome do pipeline treinado servido pela API."""

    model_path: Path
    model_file: str


class ExperimentConfig(BaseModel):
    """Configuração raiz tipada do projeto."""

    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    artifacts: ArtifactConfig


@lru_cache(maxsize=8)
def load_config(path: Path = CONFIG_PATH) -> ExperimentConfig:
    """Lê, valida e guarda em cache um YAML de configuração."""
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw_config)
