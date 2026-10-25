"""Coerência entre config, regras de alerta, dashboards, compose e catálogo."""

import json
import re
from pathlib import Path

import yaml

from src.observability import drift_metrics, pipeline_metrics
from src.utils.config_loader import load_config

MONITORING = Path("monitoring")
CATALOG_DOC = Path("docs/metricas_saude.md")
API_METRICS = {
    "http_requests_total",
    "http_request_duration_seconds",
    "credit_predictions_total",
    "credit_default_probability",
    "credit_prediction_duration_seconds",
    "credit_prediction_errors_total",
    "credit_model_loaded",
}


def _rules() -> list[dict]:
    groups = yaml.safe_load((MONITORING / "prometheus/alerts.yml").read_text("utf-8"))
    return [rule for group in groups["groups"] for rule in group["rules"]]


PREFIXES = ("pipeline_", "data_", "drift_", "model_", "credit_", "http_")


def _metric_names(text: str) -> set[str]:
    """Nomes de métrica citados numa expressão (ignora rótulos e funções)."""
    names = set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text))
    names = {re.sub(r"_(bucket|sum|count)$", "", name) for name in names}
    return {name for name in names if name.startswith(PREFIXES)}


def _published_metrics() -> set[str]:
    registries = [
        pipeline_metrics.build_stage_registry(True, 1.0, 1),
        pipeline_metrics.build_contract_registry({"r": 1}),
        drift_metrics.build_drift_registry(
            drift_metrics.DriftSummary({"psi": 1}, {"psi": 1}, {"f": 1.0}, {"f": 1.0}),
            1,
        ),
        drift_metrics.build_scoring_registry(
            {"rows": 1.0, "score_mean": 0.1, "positive_rate": 0.1, "roc_auc": 0.7}
        ),
    ]
    return {family.name for registry in registries for family in registry.collect()}


def test_alert_rules_are_well_formed() -> None:
    rules = _rules()

    assert {r["alert"] for r in rules} == {
        "ModelNotLoaded",
        "HighErrorRate",
        "HighInferenceLatency",
        "PipelineStageFailed",
        "PipelineStale",
        "DataContractViolation",
        "DataDriftDetected",
        "DataDriftSevere",
    }
    for rule in rules:
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["summary"]


def test_drift_alert_thresholds_match_the_config() -> None:
    config = load_config().observability
    rules = {r["alert"]: " ".join(r["expr"].split()) for r in _rules()}

    assert f">= {config.drift_critical_share}" in rules["DataDriftSevere"]
    assert f">= {config.drift_warning_share}" in rules["DataDriftDetected"]
    assert f"< {config.drift_critical_share}" in rules["DataDriftDetected"]


def test_alert_rules_only_use_known_metrics() -> None:
    known = _published_metrics() | API_METRICS
    for rule in _rules():
        used = _metric_names(rule["expr"])
        assert used <= known, (rule["alert"], used - known)


def test_pipeline_dashboard_only_queries_known_metrics() -> None:
    dashboard = json.loads(
        (MONITORING / "grafana/dashboards/pipeline-health.json").read_text("utf-8")
    )
    known = _published_metrics() | API_METRICS
    expressions = [t["expr"] for p in dashboard["panels"] for t in p["targets"]]

    assert dashboard["uid"] == "pipeline-health"
    assert expressions
    for expr in expressions:
        used = _metric_names(expr)
        assert used <= known, (expr, used - known)


def test_every_published_metric_is_documented() -> None:
    documented = CATALOG_DOC.read_text("utf-8")

    for name in _published_metrics():
        assert f"`{name}`" in documented, name


def test_prometheus_loads_rules_and_scrapes_both_sources() -> None:
    config = yaml.safe_load(
        (MONITORING / "prometheus/prometheus.yml").read_text("utf-8")
    )
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}

    assert config["rule_files"] == ["/etc/prometheus/alerts.yml"]
    assert jobs["pushgateway"]["honor_labels"] is True
    assert jobs["credit-api"]["static_configs"][0]["targets"] == ["api:8000"]


def test_compose_wires_pushgateway_rules_and_job_environment() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text("utf-8"))
    services = compose["services"]

    assert services["pushgateway"]["ports"][0].startswith("127.0.0.1:")
    assert (
        "./monitoring/prometheus/alerts.yml:/etc/prometheus/alerts.yml:ro"
        in (services["prometheus"]["volumes"])
    )
    for job in ("train", "drift"):
        assert (
            services[job]["environment"]["PUSHGATEWAY_URL"] == "http://pushgateway:9091"
        )
        assert "./logs:/app/logs" in services[job]["volumes"]


def test_original_api_metrics_and_dashboard_are_untouched() -> None:
    dashboard = (MONITORING / "grafana/dashboards/credit-api-overview.json").read_text(
        "utf-8"
    )

    for name in (
        "http_requests_total",
        "http_request_duration_seconds",
        "credit_predictions_total",
    ):
        assert name in dashboard


def test_events_and_logs_are_not_versioned() -> None:
    ignore = Path(".gitignore").read_text("utf-8")

    assert "logs/" in ignore
    assert "*.jsonl" in ignore
