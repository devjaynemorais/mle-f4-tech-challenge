"""Testes de fumaça: importações, configuração e rotas públicas."""

from src.serving.api import app
from src.utils.config_loader import load_config


def test_config_has_expected_typed_sections() -> None:
    config = load_config()

    assert config.project
    assert config.data
    assert config.model
    assert config.artifacts


def test_target_and_id_columns_match_home_credit_schema() -> None:
    data = load_config().data

    assert data.target_column == "TARGET"
    assert data.id_column == "SK_ID_CURR"
    assert data.raw_file == "application_train.csv"


def test_fastapi_app_loads_with_public_routes() -> None:
    routes = set(app.openapi()["paths"])

    assert {"/", "/health", "/predict", "/metrics"}.issubset(routes)
    assert app.docs_url == "/docs"
