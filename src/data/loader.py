"""Leitura do dataset bruto Home Credit e resumo do alvo.

Esta é a base funcional da Etapa 1. Os contratos de qualidade (Pandera) e a
separação entre referência e produção ficam em `contracts.py` e são
implementados nas próximas etapas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT, ExperimentConfig, load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def raw_dataset_path(config: ExperimentConfig | None = None) -> Path:
    """Caminho absoluto do CSV bruto definido no YAML."""
    data = (config or load_config()).data
    return PROJECT_ROOT / data.raw_path / data.raw_file


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Lê o CSV bruto e confirma as colunas de identificador e alvo."""
    data = load_config().data
    source = path or raw_dataset_path()
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {source}. Baixe o application_train.csv "
            "do Kaggle (Home Credit Default Risk) e coloque nesse caminho."
        )
    frame = pd.read_csv(source)
    missing = {data.id_column, data.target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"colunas obrigatórias ausentes: {sorted(missing)}")
    return frame


def summarize_dataset(frame: pd.DataFrame, target_column: str) -> dict[str, float]:
    """Resumo do dataset: volume, colunas e taxa de default."""
    return {
        "rows": float(len(frame)),
        "columns": float(frame.shape[1]),
        "default_rate": float(frame[target_column].mean()),
    }


def main() -> None:
    """Imprime o resumo do dataset bruto (`make dataset`)."""
    configure_logging()
    data = load_config().data
    frame = load_raw_dataset()
    summary = summarize_dataset(frame, data.target_column)
    logger.info(
        "Dataset bruto: %d linhas, %d colunas, taxa de default %.2f%%",
        int(summary["rows"]),
        int(summary["columns"]),
        summary["default_rate"] * 100,
    )


if __name__ == "__main__":
    main()
