"""Alertas de drift derivados dos limiares do config.yaml.

Os nomes e a lógica espelham `monitoring/prometheus/alerts.yml`: o mesmo
limiar gera o evento `alert_raised` no log e o alerta visível no Grafana.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.observability.drift_metrics import DriftSummary
from src.observability.events import PipelineEvent, current_run_id, emit
from src.utils.config_loader import ObservabilityConfig

DRIFT_METHOD = "psi"
WARNING = "DataDriftDetected"
CRITICAL = "DataDriftSevere"


@dataclass(frozen=True)
class Alert:
    """Alerta gerado por uma comparação de períodos."""

    name: str
    severity: str
    comparison: str
    value: float
    threshold: float


def drift_alerts(
    summary: DriftSummary, comparison: str, config: ObservabilityConfig
) -> list[Alert]:
    """Gera no máximo um alerta: o mais grave que o `drift_share` atinge."""
    share = summary.share(DRIFT_METHOD)
    if share >= config.drift_critical_share:
        return [
            Alert(CRITICAL, "critical", comparison, share, config.drift_critical_share)
        ]
    if share >= config.drift_warning_share:
        return [
            Alert(WARNING, "warning", comparison, share, config.drift_warning_share)
        ]
    return []


def emit_alerts(alerts: list[Alert]) -> None:
    """Registra cada alerta como evento `alert_raised`."""
    for alert in alerts:
        details = {
            "alert": alert.name,
            "severity": alert.severity,
            "comparison": alert.comparison,
            "share": round(alert.value, 4),
            "threshold": alert.threshold,
        }
        event = PipelineEvent(
            current_run_id(), "drift_detection", "alert_raised", details=details
        )
        emit(event)
