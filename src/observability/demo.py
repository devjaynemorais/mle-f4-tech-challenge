"""Cenários reproduzíveis para a demonstração de observabilidade.

Cenários: `stable` (referência x período estável, sem alerta), `drift`
(estável x driftado, gera alertas) e `invalid` (lote barrado pelo contrato).
Exigem `make train` e `make drift` executados antes (exceto `invalid`).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from src.data.contract_demo import run_demo
from src.drift import report
from src.utils.config_loader import ExperimentConfig, load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)
ORDER = ("stable", "drift", "invalid")


def _stable(config: ExperimentConfig) -> None:
    """Comparação sem drift esperado: nenhum alerta deve aparecer."""
    report.run_comparison("reference_vs_stable", report.baseline_paths(config), config)


def _drift(config: ExperimentConfig) -> None:
    """Comparação com drift simulado: alertas de drift devem aparecer."""
    report.run_comparison("stable_vs_drifted", report.configured_paths(config), config)


def _invalid(config: ExperimentConfig) -> None:
    """Lote com erros propositais: o contrato deve barrar a ingestão."""
    run_demo(config)


SCENARIOS: dict[str, Callable[[ExperimentConfig], None]] = {
    "stable": _stable,
    "drift": _drift,
    "invalid": _invalid,
}


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada de `make demo-observability`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[*ORDER, "all"], default="all")
    args = parser.parse_args(argv)
    configure_logging()
    config = load_config()
    for name in ORDER if args.scenario == "all" else (args.scenario,):
        logger.info("Cenário: %s", name)
        SCENARIOS[name](config)


if __name__ == "__main__":
    main()
