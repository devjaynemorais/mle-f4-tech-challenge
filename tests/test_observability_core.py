"""Eventos, rastreio de etapas e métricas de pipeline."""

import json
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from src.config.settings import settings
from src.observability import events, pipeline_metrics, stage_tracking
from src.observability.events import EventSchemaError, PipelineEvent, emit
from src.observability.stage_tracking import track_stage
from src.utils.config_loader import load_config


def _lines(tmp_path: Path) -> list[dict]:
    path = tmp_path / "events-root/logs/pipeline_events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _value(registry: CollectorRegistry, name: str, labels=None) -> float | None:
    return registry.get_sample_value(name, labels or {})


def test_event_schema_rejects_unknown_names_and_keys() -> None:
    with pytest.raises(EventSchemaError):
        emit(PipelineEvent("r", "training", "mystery"))
    with pytest.raises(EventSchemaError):
        emit(PipelineEvent("r", "ingestion", "stage_started", details={"x": 1}))


@pytest.mark.parametrize("key", ["SK_ID_CURR", "AMT_INCOME_TOTAL", "row", "values"])
def test_event_schema_has_no_room_for_personal_data_keys(key) -> None:
    event = PipelineEvent("r", "ingestion", "stage_started", details={key: 1})

    with pytest.raises(EventSchemaError):
        emit(event)


def test_event_schema_rejects_long_text_and_nested_values() -> None:
    long_text = PipelineEvent("r", "x", "alert_raised", details={"alert": "a" * 65})
    nested = PipelineEvent("r", "x", "alert_raised", details={"features": [{"a": 1}]})

    for event in (long_text, nested):
        with pytest.raises(EventSchemaError):
            emit(event)


def test_emit_appends_one_json_line_per_event(tmp_path) -> None:
    emit(PipelineEvent("run1", "training", "stage_started"))
    emit(PipelineEvent("run1", "training", "stage_finished", "success", 1.5, 10))

    lines = _lines(tmp_path)
    assert [line["event"] for line in lines] == ["stage_started", "stage_finished"]
    assert lines[1]["rows"] == 10
    assert lines[0]["run_id"] == "run1"


def test_emit_rotates_the_file_when_it_grows_too_large(tmp_path) -> None:
    config = load_config().observability.model_copy(update={"events_max_bytes": 100})

    for _ in range(4):
        emit(PipelineEvent("run1", "training", "stage_started"), config)

    folder = tmp_path / "events-root/logs"
    assert (folder / "pipeline_events.jsonl.1").exists()
    assert (folder / "pipeline_events.jsonl").exists()


def test_run_id_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_RUN_ID", "abc123")

    assert events.current_run_id() == "abc123"


def test_track_stage_records_success_and_publishes(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        stage_tracking, "publish_stage", lambda *args: calls.append(args)
    )

    with track_stage("training", "reference") as stage:
        stage.rows = 42

    lines = _lines(tmp_path)
    assert [line["event"] for line in lines] == ["stage_started", "stage_finished"]
    assert lines[1]["status"] == "success"
    assert lines[1]["rows"] == 42
    assert calls[0][:3] == ("training", "reference", True)
    assert calls[0][4] == 42


def test_track_stage_records_only_the_exception_type(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        stage_tracking, "publish_stage", lambda *args: calls.append(args)
    )

    with pytest.raises(ValueError, match="renda 99999"), track_stage("ingestion"):
        raise ValueError("renda 99999 do cliente 4242")

    text = (tmp_path / "events-root/logs/pipeline_events.jsonl").read_text()
    assert "ValueError" in text
    assert "99999" not in text
    assert "4242" not in text
    assert calls[0][2] is False


def test_track_stage_logs_mlflow_run_with_metrics(monkeypatch) -> None:
    runs = []
    monkeypatch.setattr(
        stage_tracking, "log_stage_run", lambda *args: runs.append(args)
    )

    with track_stage("batch_scoring") as stage:
        stage.rows = 5
        stage.metrics["score"] = 0.1

    stage_name, params, metrics, _ = runs[0]
    assert stage_name == "batch_scoring"
    assert params["status"] == "success"
    assert metrics["rows"] == 5.0
    assert metrics["score"] == 0.1


def test_stage_registry_has_last_success_only_on_success() -> None:
    ok = pipeline_metrics.build_stage_registry(True, 2.0, 100)
    failed = pipeline_metrics.build_stage_registry(False, 1.0, None)

    assert _value(ok, "pipeline_stage_last_status") == 1
    assert _value(ok, "pipeline_rows_processed") == 100
    assert _value(ok, "pipeline_stage_last_success_timestamp_seconds") > 0
    assert _value(failed, "pipeline_stage_last_status") == 0
    assert _value(failed, "pipeline_stage_last_success_timestamp_seconds") is None
    assert _value(failed, "pipeline_rows_processed") is None


def test_contract_registry_marks_blocked_batches() -> None:
    blocked = pipeline_metrics.build_contract_registry({"DAYS_BIRTH.idade": 3})
    clean = pipeline_metrics.build_contract_registry({})

    assert _value(blocked, "data_batch_blocked") == 1
    assert (
        _value(blocked, "data_contract_violations", {"rule": "DAYS_BIRTH.idade"}) == 3
    )
    assert _value(clean, "data_batch_blocked") == 0


def test_push_is_disabled_without_gateway_url() -> None:
    assert pipeline_metrics.push(CollectorRegistry(), {"stage": "x"}) is False


def test_push_uses_gateway_job_and_grouping(monkeypatch) -> None:
    sent = {}
    monkeypatch.setattr(settings, "pushgateway_url", "http://gateway:9091")
    monkeypatch.setattr(
        pipeline_metrics,
        "pushadd_to_gateway",
        lambda url, **kwargs: sent.update(url=url, **kwargs),
    )

    assert pipeline_metrics.push(CollectorRegistry(), {"stage": "training"}) is True

    assert sent["url"] == "http://gateway:9091"
    assert sent["job"] == "credit-pipeline"
    assert sent["grouping_key"] == {"stage": "training"}


def test_push_failure_never_breaks_the_pipeline(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise ConnectionError("gateway fora do ar")

    monkeypatch.setattr(settings, "pushgateway_url", "http://gateway:9091")
    monkeypatch.setattr(pipeline_metrics, "pushadd_to_gateway", boom)

    assert pipeline_metrics.push(CollectorRegistry(), {"stage": "x"}) is False


def test_mlflow_runs_are_disabled_without_explicit_uri() -> None:
    from src.observability.mlflow_runs import log_stage_run

    assert log_stage_run("training", {}, {}) is False


def test_mlflow_failure_is_swallowed(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from src.observability import mlflow_runs

    client = MagicMock()
    client.set_tracking_uri.side_effect = ConnectionError("sem servidor")
    monkeypatch.setattr(mlflow_runs, "mlflow", client)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    assert mlflow_runs.log_stage_run("training", {"run_id": "r"}, {"a": 1.0}) is False


def test_mlflow_logs_params_metrics_and_artifacts(monkeypatch, tmp_path) -> None:
    from contextlib import nullcontext
    from unittest.mock import MagicMock

    from src.observability import mlflow_runs

    client = MagicMock()
    client.start_run.return_value = nullcontext()
    monkeypatch.setattr(mlflow_runs, "mlflow", client)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    report = tmp_path / "r.html"
    report.write_text("<html></html>")

    ok = mlflow_runs.log_stage_run(
        "drift_detection", {"run_id": "r"}, {"drifted_features": 3.0}, [report]
    )

    assert ok is True
    client.set_experiment.assert_called_once_with("credit-pipeline-monitoring")
    client.log_metrics.assert_called_once_with({"drifted_features": 3.0})
    client.set_tag.assert_called_once_with("pipeline_status", "unknown")
    client.log_artifact.assert_called_once()
