"""Resultados de drift e de predição em lote como métricas de observabilidade."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Gauge

from src.drift.detect import DriftReport
from src.observability.pipeline_metrics import push

PSI = "psi"
KS = "ks"


@dataclass(frozen=True)
class DriftSummary:
    """Contagens por método e estatísticas por feature, sem dados de clientes."""

    totals: dict[str, int]
    drifted: dict[str, int]
    psi: dict[str, float] = field(default_factory=dict)
    ks: dict[str, float] = field(default_factory=dict)

    @property
    def max_psi(self) -> float:
        """Maior PSI entre as features; 0 quando não há PSI."""
        return max(self.psi.values(), default=0.0)

    def share(self, method: str) -> float:
        """Fração de features com drift no método informado."""
        total = self.totals.get(method, 0)
        return self.drifted.get(method, 0) / total if total else 0.0


def summarize_drift(report: DriftReport) -> DriftSummary:
    """Agrega os resultados do detector por método e por feature."""
    totals: dict[str, int] = defaultdict(int)
    drifted: dict[str, int] = defaultdict(int)
    stats: dict[str, dict[str, float]] = {PSI: {}, KS: {}}
    for item in report.results:
        totals[item.method] += 1
        drifted[item.method] += int(item.drift_detected)
        if item.method in stats and math.isfinite(item.statistic):
            stats[item.method][item.feature_name] = item.statistic
    return DriftSummary(dict(totals), dict(drifted), stats[PSI], stats[KS])


def _by_method(registry: CollectorRegistry, name: str, doc: str) -> Gauge:
    """Gauge com rótulo `method`."""
    return Gauge(name, doc, ["method"], registry=registry)


def _by_feature(
    registry: CollectorRegistry, name: str, values: dict[str, float]
) -> None:
    """Gauge com uma série por feature."""
    gauge = Gauge(
        name, "Estatística de drift por feature", ["feature"], registry=registry
    )
    for feature, value in values.items():
        gauge.labels(feature=feature).set(value)


def build_drift_registry(summary: DriftSummary, n_alerts: int) -> CollectorRegistry:
    """Monta as métricas de drift de uma comparação entre períodos."""
    registry = CollectorRegistry()
    total = _by_method(registry, "drift_features_total", "Features comparadas")
    drifted = _by_method(registry, "drift_features_drifted", "Features com drift")
    share = _by_method(registry, "drift_share", "Fração de features com drift")
    for method, count in summary.totals.items():
        total.labels(method=method).set(count)
        drifted.labels(method=method).set(summary.drifted.get(method, 0))
        share.labels(method=method).set(summary.share(method))
    Gauge("drift_max_psi", "Maior PSI entre as features", registry=registry).set(
        summary.max_psi
    )
    _by_feature(registry, "drift_feature_psi", summary.psi)
    _by_feature(registry, "drift_feature_ks", summary.ks)
    Gauge("drift_alerts_active", "Alertas do último ciclo", registry=registry).set(
        n_alerts
    )
    return registry


def publish_drift(comparison: str, summary: DriftSummary, n_alerts: int) -> bool:
    """Publica o drift de uma comparação (grupo: etapa + comparação)."""
    registry = build_drift_registry(summary, n_alerts)
    return push(registry, {"stage": "drift_detection", "comparison": comparison})


def build_scoring_registry(stats: dict[str, float]) -> CollectorRegistry:
    """Monta as métricas de score de um período; `roc_auc` só quando houver."""
    registry = CollectorRegistry()
    docs = {
        "model_batch_score_mean": "Probabilidade média de default prevista",
        "model_batch_positive_rate": "Fração prevista como default",
        "model_batch_rows": "Linhas pontuadas",
        "model_batch_roc_auc": "AUC no período com alvo real",
    }
    for name, doc in docs.items():
        key = name.removeprefix("model_batch_")
        if key in stats:
            Gauge(name, doc, registry=registry).set(stats[key])
    return registry


def publish_scoring(period: str, stats: dict[str, float]) -> bool:
    """Publica as métricas de score do período (grupo: etapa + período)."""
    registry = build_scoring_registry(stats)
    return push(registry, {"stage": "batch_scoring", "period": period})
