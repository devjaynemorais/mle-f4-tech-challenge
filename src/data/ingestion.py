"""Porta de entrada dos lotes: contrato de dados com observabilidade."""

from __future__ import annotations

import pandas as pd

from src.data.contracts import ContractResult, DataContractError, validate_batch
from src.observability.events import PipelineEvent, current_run_id, emit
from src.observability.pipeline_metrics import publish_contract
from src.observability.stage_tracking import track_stage
from src.utils.config_loader import ExperimentConfig

STAGE = "contract_validation"


def _report_violations(result: ContractResult, dataset: str) -> None:
    """Publica métricas e eventos do lote barrado (só regras e contagens)."""
    publish_contract(dataset, result.violations)
    run_id = current_run_id()
    for rule, count in result.violations.items():
        details = {"dataset": dataset, "rule": rule, "violations": count}
        emit(
            PipelineEvent(
                run_id, STAGE, "contract_violation", "failure", details=details
            )
        )
    if result.violations:
        alert = {
            "dataset": dataset,
            "alert": "DataContractViolation",
            "severity": "warning",
        }
        emit(PipelineEvent(run_id, STAGE, "alert_raised", details=alert))


def gate_batch(
    frame: pd.DataFrame, config: ExperimentConfig, dataset: str
) -> ContractResult:
    """Valida o lote; se violar o contrato, registra e bloqueia a ingestão."""
    with track_stage(STAGE, dataset) as stage:
        result = validate_batch(frame, config)
        stage.rows = result.n_rows
        stage.metrics["violated_rules"] = float(len(result.violations))
        if not result.valid:
            _report_violations(result, dataset)
            raise DataContractError(result)
        publish_contract(dataset, {})
    return result
