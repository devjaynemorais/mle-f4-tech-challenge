"""Executa o pipeline completo sob um único `run_id`.

Ordem: treino (ingestão, contrato, transformação), simulação de drift,
predição em lote e relatórios de drift. Cada etapa emite eventos e métricas.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable

from src.drift import data_drift_simulation, predict_batch, report
from src.models import train
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)
Step = tuple[str, Callable[[], None]]


def _steps() -> list[Step]:
    """Lista as etapas na ordem de execução."""
    return [
        ("treino", train.main),
        ("simulação de drift", lambda: data_drift_simulation.main([])),
        ("predição em lote", lambda: predict_batch.main([])),
        ("relatórios de drift", lambda: report.main([])),
    ]


def main() -> None:
    """Ponto de entrada de `make pipeline`."""
    configure_logging()
    os.environ.setdefault("PIPELINE_RUN_ID", uuid.uuid4().hex[:12])
    logger.info("Pipeline iniciado (run_id=%s)", os.environ["PIPELINE_RUN_ID"])
    for name, step in _steps():
        logger.info("Etapa: %s", name)
        step()
    logger.info("Pipeline concluído")


if __name__ == "__main__":
    main()
