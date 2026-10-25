"""Métricas Prometheus dos jobs em lote, enviadas ao Pushgateway.

Jobs curtos não são raspados pelo Prometheus, então cada etapa publica ao
final o seu último estado. Todas as métricas são gauges: o Pushgateway não
acumula contadores entre execuções. O histórico fica no MLflow e no JSONL.
A falha do envio nunca derruba o pipeline.
"""

from __future__ import annotations

import logging
import time

from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

from src.config.settings import settings
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)
PUSH_TIMEOUT_SECONDS = 5.0


def gateway_url() -> str:
    """Endereço do Pushgateway; vazio significa envio desativado."""
    return settings.pushgateway_url.strip()


def push(registry: CollectorRegistry, grouping_key: dict[str, str]) -> bool:
    """Envia o registro ao Pushgateway; devolve False se desativado ou falhar."""
    url = gateway_url()
    if not url:
        return False
    job = load_config().observability.pushgateway_job
    try:
        pushadd_to_gateway(
            url,
            job=job,
            registry=registry,
            grouping_key=grouping_key,
            timeout=PUSH_TIMEOUT_SECONDS,
        )
    except Exception as error:  # noqa: BLE001 - métricas não podem quebrar o pipeline
        logger.warning("Falha ao enviar métricas (%s)", type(error).__name__)
        return False
    return True


def _gauge(registry: CollectorRegistry, name: str, doc: str, value: float) -> None:
    """Cria um gauge sem rótulos já preenchido."""
    Gauge(name, doc, registry=registry).set(value)


def build_stage_registry(
    succeeded: bool, duration_s: float, rows: int | None
) -> CollectorRegistry:
    """Monta as métricas de saúde de uma execução de etapa."""
    registry = CollectorRegistry()
    _gauge(registry, "pipeline_stage_last_status", "1 sucesso, 0 falha", succeeded)
    _gauge(registry, "pipeline_stage_duration_seconds", "Duração", duration_s)
    if succeeded:
        _gauge(
            registry,
            "pipeline_stage_last_success_timestamp_seconds",
            "Instante do último sucesso",
            time.time(),
        )
    if rows is not None:
        _gauge(registry, "pipeline_rows_processed", "Linhas processadas", rows)
    return registry


def publish_stage(
    stage: str, dataset: str, succeeded: bool, duration_s: float, rows: int | None
) -> bool:
    """Publica o último estado de uma etapa (grupo: etapa + dataset)."""
    registry = build_stage_registry(succeeded, duration_s, rows)
    return push(registry, {"stage": stage, "dataset": dataset})


def build_contract_registry(violations: dict[str, int]) -> CollectorRegistry:
    """Monta as métricas do último lote validado pelo contrato."""
    registry = CollectorRegistry()
    blocked = Gauge(
        "data_batch_blocked", "1 se o último lote foi barrado", registry=registry
    )
    blocked.set(1 if violations else 0)
    per_rule = Gauge(
        "data_contract_violations",
        "Linhas que violam a regra no último lote",
        ["rule"],
        registry=registry,
    )
    for rule, count in violations.items():
        per_rule.labels(rule=rule).set(count)
    return registry


def publish_contract(dataset: str, violations: dict[str, int]) -> bool:
    """Publica o resultado do contrato do lote (grupo: contrato + dataset)."""
    registry = build_contract_registry(violations)
    return push(registry, {"stage": "contract_validation", "dataset": dataset})
