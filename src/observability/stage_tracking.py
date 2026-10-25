"""Rastreio de execução das etapas do pipeline (eventos, métricas e MLflow)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from src.observability.events import PipelineEvent, current_run_id, emit
from src.observability.mlflow_runs import log_stage_run
from src.observability.pipeline_metrics import publish_stage


@dataclass
class StageHandle:
    """Informações que a etapa preenche enquanto executa."""

    rows: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)


def _finish(
    stage: str,
    dataset: str,
    run_id: str,
    handle: StageHandle,
    started: float,
    error_type: str | None,
) -> None:
    """Emite o evento final e publica métricas e run do MLflow."""
    duration = round(time.perf_counter() - started, 3)
    status = "failure" if error_type else "success"
    details: dict[str, object] = {"dataset": dataset}
    if error_type:
        details["error_type"] = error_type
    emit(
        PipelineEvent(
            run_id, stage, "stage_finished", status, duration, handle.rows, details
        )
    )
    publish_stage(stage, dataset, error_type is None, duration, handle.rows)
    metrics = {"duration_s": duration, **handle.metrics}
    if handle.rows is not None:
        metrics["rows"] = float(handle.rows)
    params = {"run_id": run_id, "dataset": dataset, "status": status}
    log_stage_run(stage, params, metrics, handle.artifacts)


@contextmanager
def track_stage(
    stage: str, dataset: str = "all", run_id: str | None = None
) -> Iterator[StageHandle]:
    """Mede e registra uma etapa; em falha registra só o tipo da exceção."""
    identifier = run_id or current_run_id()
    handle = StageHandle()
    started = time.perf_counter()
    emit(
        PipelineEvent(identifier, stage, "stage_started", details={"dataset": dataset})
    )
    try:
        yield handle
    except Exception as error:
        _finish(stage, dataset, identifier, handle, started, type(error).__name__)
        raise
    _finish(stage, dataset, identifier, handle, started, None)
