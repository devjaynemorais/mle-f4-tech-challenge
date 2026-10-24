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
    """Parâmetros de decisão e de treino do modelo de crédito."""

    prediction_threshold: float = Field(gt=0, lt=1)
    production_size: float = Field(default=0.2, gt=0, lt=1)
    cv_folds: int = Field(default=5, ge=2)
    random_state: int = 42


class ArtifactConfig(BaseModel):
    """Diretório e nomes dos pipelines persistidos pelo treino."""

    model_path: Path
    model_file: str
    baseline_file: str = "logistic_baseline.joblib"


class MlflowConfig(BaseModel):
    """Destino do tracking de experimentos (sobrescrito por env)."""

    tracking_uri: str = "http://localhost:5000"
    experiment: str = "credit-scoring"


class DriftConfig(BaseModel):
    """Arquivos e limiares para simulação e detecção de drift."""

    stable_file: str = "prod_periodo1_estavel.csv"
    drifted_file: str = "prod_periodo2_drift.csv"
    manifest_file: str = "data_drift_simulation_manifest.json"
    report_html: Path = Path("reports/data_drift_report.html")
    summary_csv: Path = Path("reports/data_drift_summary.csv")
    alpha: float = Field(default=0.05, gt=0, lt=1)
    psi_threshold: float = Field(default=0.25, gt=0)
    n_bins: int = Field(default=10, ge=2)


class ExperimentConfig(BaseModel):
    """Configuração raiz tipada do projeto."""

    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    artifacts: ArtifactConfig
    mlflow: MlflowConfig = MlflowConfig()
    drift: DriftConfig = DriftConfig()


@lru_cache(maxsize=8)
def load_config(path: Path = CONFIG_PATH) -> ExperimentConfig:
    """Lê, valida e guarda em cache um YAML de configuração."""
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw_config)
