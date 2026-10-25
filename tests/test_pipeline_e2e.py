"""Pipeline de ponta a ponta com dados sintéticos: contrato, treino, drift e alertas."""

import json
import re

import pandas as pd
import pytest

from src.data import loader, processed
from src.data.contracts import DataContractError
from src.drift import data_drift_simulation, predict_batch, report
from src.models import artifacts
from src.models import train as training
from src.observability import events
from tests.synthetic import make_raw_frame


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Projeto isolado em tmp_path, com CSV bruto sintético válido."""
    root = tmp_path / "project"
    for module in (loader, processed, artifacts, data_drift_simulation):
        monkeypatch.setattr(module, "PROJECT_ROOT", root)
    for module in (predict_batch, report, events):
        monkeypatch.setattr(module, "PROJECT_ROOT", root)
    monkeypatch.setattr(artifacts, "METRICS_DIR", root / "metrics")
    monkeypatch.setattr(training, "log_to_mlflow", lambda *args: None)
    raw = root / "data/raw/application_train.csv"
    raw.parent.mkdir(parents=True)
    make_raw_frame().to_csv(raw, index=False)
    return root


def _events(root) -> list[dict]:
    path = root / "logs/pipeline_events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run_all() -> None:
    training.train()
    data_drift_simulation.main(["--n", "600"])
    predict_batch.main([])
    report.main([])


def test_full_pipeline_emits_events_alerts_and_baseline_without_drift(
    workspace,
) -> None:
    _run_all()

    lines = _events(workspace)
    finished = {
        (e["stage"], e["details"].get("dataset"))
        for e in lines
        if e["event"] == "stage_finished" and e["status"] == "success"
    }
    expected = {
        ("ingestion", "raw"),
        ("contract_validation", "raw"),
        ("transformation", "raw"),
        ("training", "reference"),
        ("batch_scoring", "all"),
        ("drift_detection", "reference_vs_stable"),
        ("drift_detection", "stable_vs_drifted"),
    }
    assert expected <= finished
    alerts = [e for e in lines if e["event"] == "alert_raised"]
    assert [a["details"]["comparison"] for a in alerts] == ["stable_vs_drifted"]
    assert alerts[0]["details"]["alert"] == "DataDriftSevere"
    results = {
        e["details"]["comparison"]: e["details"]
        for e in lines
        if e["event"] == "drift_result"
    }
    assert results["reference_vs_stable"]["n_drifted"] == 0
    assert results["stable_vs_drifted"]["max_psi"] > 1.0


def test_pipeline_writes_predictions_reports_and_aggregated_summary(workspace) -> None:
    _run_all()

    processed_dir = workspace / "data/processed"
    stable = pd.read_csv(processed_dir / "predicoes_periodo1_estavel.csv")
    drifted = pd.read_csv(processed_dir / "predicoes_periodo2_drift.csv")
    summary = json.loads((workspace / "metrics/batch_scoring_summary.json").read_text())
    assert list(stable.columns) == ["SK_ID_CURR", "default_probability", "prediction"]
    assert len(stable) == len(drifted) == 300
    assert "roc_auc" in summary["periods"]["stable"]
    assert "roc_auc" not in summary["periods"]["drifted"]
    for stem in ("data_drift_report", "data_drift_summary"):
        assert (
            workspace / f"reports/{stem}.html"
            if stem.endswith("report")
            else workspace / f"reports/{stem}.csv"
        ).exists()
    assert (workspace / "reports/data_drift_reference_vs_stable.html").exists()
    assert (workspace / "reports/data_drift_reference_vs_stable.csv").exists()


def test_events_never_contain_identifiers_or_raw_values(workspace) -> None:
    _run_all()

    text = (workspace / "logs/pipeline_events.jsonl").read_text()
    assert "SK_ID_CURR" not in text
    assert not re.search(r"\b10[0-2]\d{3}\b", text)
    assert "AMT_" not in text


def test_invalid_raw_batch_is_blocked_before_any_training(workspace) -> None:
    raw = workspace / "data/raw/application_train.csv"
    frame = pd.read_csv(raw)
    frame.loc[:9, "DAYS_BIRTH"] = -5_000
    frame.loc[10:14, "AMT_INCOME_TOTAL"] = None
    frame.to_csv(raw, index=False)

    with pytest.raises(DataContractError) as error:
        training.train()

    assert "DAYS_BIRTH.idade_acima_de_18_anos" in error.value.result.violations
    assert not (workspace / "models/artifacts/credit_model.joblib").exists()
    assert not (workspace / "data/processed/ref_data.csv").exists()
    lines = _events(workspace)
    blocked = [e for e in lines if e["event"] == "contract_violation"]
    assert {e["details"]["rule"] for e in blocked} >= {
        "DAYS_BIRTH.idade_acima_de_18_anos",
        "AMT_INCOME_TOTAL.nao_nulo",
    }
    failed = [
        e for e in lines if e["event"] == "stage_finished" and e["status"] == "failure"
    ]
    assert failed[0]["details"]["error_type"] == "DataContractError"
    assert not any(e["stage"] == "training" for e in lines)


def test_scoring_blocks_a_corrupted_production_period(workspace) -> None:
    training.train()
    data_drift_simulation.main(["--n", "600"])
    path = workspace / "data/processed/prod_periodo2_drift.csv"
    frame = pd.read_csv(path)
    frame.loc[:4, "AMT_INCOME_TOTAL"] = -1.0
    frame.to_csv(path, index=False)

    with pytest.raises(DataContractError):
        predict_batch.main([])

    assert not (workspace / "metrics/batch_scoring_summary.json").exists()


def test_pipeline_entrypoint_shares_one_run_id_across_all_stages(
    workspace, monkeypatch
) -> None:
    from src import pipeline

    monkeypatch.delenv("PIPELINE_RUN_ID", raising=False)
    original = data_drift_simulation.main
    monkeypatch.setattr(
        pipeline.data_drift_simulation, "main", lambda argv: original(["--n", "600"])
    )

    pipeline.main()

    lines = _events(workspace)
    assert len({e["run_id"] for e in lines}) == 1
    assert {e["stage"] for e in lines} >= {
        "ingestion",
        "contract_validation",
        "transformation",
        "training",
        "batch_scoring",
        "drift_detection",
    }


def test_demo_scenarios_stable_has_no_alert_and_drift_has_one(workspace) -> None:
    from src.observability import demo

    _run_all()
    before = len([e for e in _events(workspace) if e["event"] == "alert_raised"])
    demo.main(["--scenario", "stable"])
    after_stable = len([e for e in _events(workspace) if e["event"] == "alert_raised"])
    demo.main(["--scenario", "drift"])
    after_drift = len([e for e in _events(workspace) if e["event"] == "alert_raised"])

    assert after_stable == before
    assert after_drift == before + 1


def test_demo_invalid_scenario_blocks_the_bad_batch(workspace) -> None:
    from src.observability import demo

    demo.main(["--scenario", "invalid"])

    lines = _events(workspace)
    rules = {e["details"]["rule"] for e in lines if e["event"] == "contract_violation"}
    assert "DAYS_BIRTH.idade_acima_de_18_anos" in rules
    assert any(
        e["event"] == "alert_raised"
        and e["details"]["alert"] == "DataContractViolation"
        for e in lines
    )


def test_contract_demo_main_succeeds_when_the_bad_batch_is_blocked(workspace) -> None:
    from src.data import contract_demo

    contract_demo.main()

    assert contract_demo.valid_batch().shape[0] == 50
    assert not contract_demo.invalid_batch().equals(contract_demo.valid_batch())
