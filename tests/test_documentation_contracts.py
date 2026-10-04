"""Documentação e artefatos públicos devem refletir o contrato de crédito."""

import json
from pathlib import Path

from src.utils.config_loader import load_config

README = Path("README.md")
AGENTS = Path("AGENTS.md")


def test_readme_documents_home_credit_dataset_and_target() -> None:
    text = README.read_text(encoding="utf-8")

    assert "application_train.csv" in text
    assert "Home Credit" in text
    assert "TARGET" in text


def test_readme_has_governance_section_for_stage_four() -> None:
    text = README.read_text(encoding="utf-8")

    assert "## Governança e LGPD" in text
    assert "Base legal" in text
    assert "Plano de retenção" in text
    assert "Causalidade" in text


def test_readme_maps_every_stage_of_the_challenge() -> None:
    text = README.read_text(encoding="utf-8")

    for stage in (
        "1. Validação",
        "2. Simulação",
        "3. Observabilidade",
        "4. Governança",
    ):
        assert stage in text


def test_challenge_brief_and_checklist_are_versioned() -> None:
    assert Path("docs/MLET - Tech Challenge - Fase 4.pdf").exists()
    assert Path("docs/checklist_requisitos_tc4.md").exists()


def test_agents_guide_points_to_configured_raw_file() -> None:
    guide = AGENTS.read_text(encoding="utf-8")
    data = load_config().data

    assert f"data/raw/{data.raw_file}" in guide
    assert data.target_column in guide
    assert data.id_column in guide


def test_agents_guide_has_no_obsolete_medical_contract() -> None:
    guide = AGENTS.read_text(encoding="utf-8")

    assert "triage" not in guide.lower()
    assert "medical_tc" not in guide


def test_dashboard_uses_credit_prediction_metric_and_title() -> None:
    dashboard = json.loads(
        Path("monitoring/grafana/dashboards/credit-api-overview.json").read_text(
            encoding="utf-8"
        )
    )

    exprs = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if "expr" in target
    ]
    assert any("credit_predictions_total" in expr for expr in exprs)
    assert not any("triage_predictions_total" in expr for expr in exprs)
    assert dashboard["title"].startswith("Credit Scoring API")
