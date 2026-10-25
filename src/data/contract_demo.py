"""Demonstração do contrato: um lote válido passa e um lote com erros é barrado.

Os lotes são sintéticos e pequenos, criados em memória. Nenhum dado real é lido.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.contracts import DataContractError
from src.data.ingestion import gate_batch
from src.utils.config_loader import ExperimentConfig, load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def valid_batch(n: int = 50) -> pd.DataFrame:
    """Lote sintético que respeita todas as regras do contrato."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(900_000, 900_000 + n),
            "TARGET": [0, 1] * (n // 2),
            "AMT_INCOME_TOTAL": [60_000.0 + 1_000 * i for i in range(n)],
            "AMT_CREDIT": [150_000.0 + 2_000 * i for i in range(n)],
            "DAYS_BIRTH": [-(9_000 + 120 * i) for i in range(n)],
        }
    )


def invalid_batch(n: int = 50) -> pd.DataFrame:
    """Lote com erros propositais: duplicata, renda nula, menor de idade, alvo 2."""
    frame = valid_batch(n)
    frame.loc[1, "SK_ID_CURR"] = frame.loc[0, "SK_ID_CURR"]
    frame.loc[2, "AMT_INCOME_TOTAL"] = None
    frame.loc[3, "DAYS_BIRTH"] = -5_000
    frame.loc[4, "TARGET"] = 2
    return frame


def run_demo(config: ExperimentConfig, dataset: str = "demo_invalid") -> bool:
    """Passa os dois lotes pela porta de entrada; True se o inválido foi barrado."""
    gate_batch(valid_batch(), config, "demo_valid")
    logger.info("Lote válido: ingestão liberada")
    try:
        gate_batch(invalid_batch(), config, dataset)
    except DataContractError as error:
        logger.warning("Lote inválido BLOQUEADO: %s", error)
        return True
    return False


def main() -> None:
    """Ponto de entrada de `make contract-demo`."""
    configure_logging()
    if not run_demo(load_config()):
        raise SystemExit("o contrato deveria ter barrado o lote inválido")


if __name__ == "__main__":
    main()
