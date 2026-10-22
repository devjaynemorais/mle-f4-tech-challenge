import logging

import pytest
from fastapi.testclient import TestClient

import src.serving.api as api_module

SECRET_VALUE = 1234567.891


class _FakePredictor:
    backend = "fake"

    def predict(self, features: dict) -> tuple[float, str]:
        return 0.8, "default"


class _FailingPredictor:
    backend = "fake"

    def predict(self, features: dict) -> tuple[float, str]:
        raise RuntimeError("boom")


def _missing_predictor():
    raise FileNotFoundError("model not found")


def _payload() -> dict:
    return {"features": {"AMT_CREDIT": SECRET_VALUE, "AMT_INCOME_TOTAL": 202500.0}}


def test_root_endpoint_reports_service_name_and_version() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "Credit Scoring Observability API",
        "version": "0.1.0",
    }


def test_health_reports_loaded_backend() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/health")

    assert response.json() == {"status": "ok", "model_loaded": True, "backend": "fake"}


def test_health_is_degraded_when_loading_fails() -> None:
    with TestClient(api_module.create_app(_missing_predictor)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False}


def test_app_instances_keep_predictors_isolated() -> None:
    loaded_app = api_module.create_app(lambda: _FakePredictor())
    degraded_app = api_module.create_app(_missing_predictor)

    with TestClient(loaded_app) as loaded, TestClient(degraded_app) as degraded:
        assert loaded.get("/health").json()["status"] == "ok"
        assert degraded.get("/health").json()["status"] == "degraded"


def test_predict_returns_probability_label_and_backend() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    assert response.json() == {
        "default_probability": 0.8,
        "prediction": "default",
        "backend": "fake",
    }


def test_predict_rejects_empty_feature_payload() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422


def test_predict_returns_503_without_predictor() -> None:
    with TestClient(api_module.create_app(_missing_predictor)) as client:
        response = client.post("/predict", json=_payload())

    assert response.status_code == 503


def test_inference_failure_returns_500_without_logging_client_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)

    with TestClient(api_module.create_app(lambda: _FailingPredictor())) as client:
        response = client.post("/predict", json=_payload())

    assert response.status_code == 500
    assert str(SECRET_VALUE) not in caplog.text
    assert "Falha inesperada na inferência (RuntimeError)" in caplog.text


def test_metrics_endpoint_exposes_preserved_prometheus_names() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        client.get("/")
        client.post("/predict", json=_payload())
        response = client.get("/metrics")

    assert response.status_code == 200
    assert b"http_requests_total" in response.content
    assert b"http_request_duration_seconds" in response.content
    assert b"credit_predictions_total" in response.content
    assert b"credit_default_probability" in response.content
    assert b"credit_prediction_duration_seconds" in response.content
    assert b"credit_model_loaded" in response.content


def test_swagger_docs_are_available() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()
