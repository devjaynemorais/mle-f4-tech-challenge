"""Eventos estruturados do pipeline, sem dados pessoais, gravados em JSONL.

O schema é uma lista de permissão: só chaves conhecidas, valores escalares e
textos curtos entram no evento. Identificadores, rendas, créditos, linhas de
clientes e mensagens de exceção não cabem nele por construção.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.utils.config_loader import PROJECT_ROOT, ObservabilityConfig, load_config

logger = logging.getLogger(__name__)

EVENT_NAMES = frozenset(
    {
        "stage_started",
        "stage_finished",
        "contract_violation",
        "drift_result",
        "alert_raised",
    }
)
ALLOWED_DETAIL_KEYS = frozenset(
    {
        "dataset",
        "comparison",
        "period",
        "method",
        "severity",
        "alert",
        "error_type",
        "rule",
        "violations",
        "share",
        "threshold",
        "n_features",
        "n_drifted",
        "max_psi",
        "features",
        "score_mean",
        "positive_rate",
        "roc_auc",
    }
)
MAX_TEXT_LENGTH = 64
MAX_LIST_ITEMS = 200
Scalar = str | int | float | bool | None


class EventSchemaError(ValueError):
    """Evento fora do schema permitido."""


@dataclass(frozen=True)
class PipelineEvent:
    """Uma linha do log de execução do pipeline."""

    run_id: str
    stage: str
    event: str
    status: str | None = None
    duration_s: float | None = None
    rows: int | None = None
    details: dict[str, object] = field(default_factory=dict)
    ts: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )


_PROCESS_RUN_ID = uuid.uuid4().hex[:12]


def current_run_id() -> str:
    """Usa PIPELINE_RUN_ID quando o orquestrador define; senão um id por processo."""
    return os.environ.get("PIPELINE_RUN_ID") or _PROCESS_RUN_ID


def _check_scalar(value: object, where: str) -> None:
    """Aceita apenas escalares simples e textos curtos."""
    if value is None or isinstance(value, bool | int | float):
        return
    if isinstance(value, str) and len(value) <= MAX_TEXT_LENGTH:
        return
    raise EventSchemaError(f"valor não permitido em {where}")


def _check_detail(key: str, value: object) -> None:
    """Valida o valor de um detalhe: escalar, lista de escalares ou mapa curto."""
    if isinstance(value, list):
        if len(value) > MAX_LIST_ITEMS:
            raise EventSchemaError(f"lista longa demais em {key}")
        for item in value:
            _check_scalar(item, key)
    elif isinstance(value, dict):
        for name, item in value.items():
            _check_scalar(name, key)
            _check_scalar(item, key)
    else:
        _check_scalar(value, key)


def validate_event(event: PipelineEvent) -> None:
    """Rejeita eventos com nome desconhecido ou detalhes fora da lista."""
    if event.event not in EVENT_NAMES:
        raise EventSchemaError(f"evento desconhecido: {event.event}")
    unknown = set(event.details) - ALLOWED_DETAIL_KEYS
    if unknown:
        raise EventSchemaError(f"detalhes não permitidos: {sorted(unknown)}")
    for key, value in event.details.items():
        _check_detail(key, value)


def _events_path(config: ObservabilityConfig) -> Path:
    """Resolve o arquivo de eventos relativo à raiz do projeto."""
    return PROJECT_ROOT / config.events_file


def _rotate_if_needed(path: Path, max_bytes: int) -> None:
    """Mantém uma geração anterior quando o arquivo passa do limite."""
    if path.exists() and path.stat().st_size >= max_bytes:
        path.replace(path.with_name(path.name + ".1"))


def emit(event: PipelineEvent, config: ObservabilityConfig | None = None) -> None:
    """Valida, registra no logger e acrescenta o evento ao arquivo JSONL."""
    validate_event(event)
    settings = config or load_config().observability
    line = json.dumps(asdict(event), ensure_ascii=False, sort_keys=True)
    logger.info("evento %s", line)
    path = _events_path(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path, settings.events_max_bytes)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as error:
        logger.warning("Falha ao gravar evento (%s)", type(error).__name__)
