"""Isola a observabilidade dos testes: sem arquivos no repositório nem rede."""

import pytest

from src.config.settings import settings
from src.observability import events


@pytest.fixture(autouse=True)
def isolated_observability(tmp_path, monkeypatch) -> None:
    """Redireciona o log de eventos e desliga Pushgateway e MLflow."""
    monkeypatch.setattr(events, "PROJECT_ROOT", tmp_path / "events-root")
    monkeypatch.setattr(settings, "pushgateway_url", "")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("PIPELINE_RUN_ID", raising=False)
