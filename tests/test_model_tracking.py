"""Falhas de tracking não impedem o resultado local do treino."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

from src.models import tracking
from src.models.evaluation import Candidate
from src.utils.config_loader import load_config


def _candidate() -> Candidate:
    return Candidate("logistic_regression", 0.75, 0.02, (0.73, 0.77))


def test_tracking_can_be_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(tracking, "mlflow", None)
    candidate = _candidate()

    assert (
        tracking.log_to_mlflow(
            [candidate], candidate, Path("model.joblib"), load_config()
        )
        is None
    )


def test_tracking_uses_environment_uri_and_logs_selected_metric(monkeypatch) -> None:
    client = MagicMock()
    client.start_run.return_value = nullcontext()
    monkeypatch.setattr(tracking, "mlflow", client)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    candidate = _candidate()

    tracking.log_to_mlflow([candidate], candidate, Path("model.joblib"), load_config())

    client.set_tracking_uri.assert_called_once_with("http://tracking:5000")
    client.log_metric.assert_any_call("cv_roc_auc_best", 0.75)
    client.log_artifact.assert_called_once_with("model.joblib", artifact_path="model")


def test_tracking_failure_does_not_raise(monkeypatch) -> None:
    client = MagicMock()
    client.set_tracking_uri.side_effect = ConnectionError("offline")
    monkeypatch.setattr(tracking, "mlflow", client)
    candidate = _candidate()

    assert (
        tracking.log_to_mlflow(
            [candidate], candidate, Path("model.joblib"), load_config()
        )
        is None
    )
