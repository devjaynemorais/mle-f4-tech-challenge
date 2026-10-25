from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config_loader import load_config


def test_load_config_returns_typed_home_credit_config() -> None:
    config = load_config()

    assert config.project.name == "Credit Scoring Observability API"
    assert config.data.raw_file == "application_train.csv"
    assert config.data.reference_file == "ref_data.csv"
    assert config.data.production_file == "prod_data.csv"
    assert config.model.production_size == 0.2
    assert config.model.cv_folds == 5
    assert config.model.prediction_threshold == 0.5
    assert config.drift.stable_file == "prod_periodo1_estavel.csv"
    assert config.drift.drifted_file == "prod_periodo2_drift.csv"
    assert config.drift.manifest_file == "data_drift_simulation_manifest.json"
    assert config.drift.report_html == Path("reports/data_drift_report.html")
    assert config.drift.summary_csv == Path("reports/data_drift_summary.csv")
    assert config.drift.alpha == 0.05
    assert config.drift.psi_threshold == 0.25
    assert config.drift.n_bins == 10


def test_load_config_rejects_threshold_outside_unit_interval(tmp_path: Path) -> None:
    raw_config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    raw_config["model"]["prediction_threshold"] = 1.5
    invalid_path = tmp_path / "invalid-config.yaml"
    invalid_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    with pytest.raises(ValidationError, match="prediction_threshold"):
        load_config(invalid_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("alpha", 1.0), ("psi_threshold", 0), ("n_bins", 1)],
)
def test_load_config_rejects_invalid_drift_parameters(
    tmp_path: Path, field: str, value: float
) -> None:
    raw_config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    raw_config.setdefault("drift", {})[field] = value
    invalid_path = tmp_path / "invalid-drift-config.yaml"
    invalid_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    with pytest.raises(ValidationError, match=field):
        load_config(invalid_path)


def test_observability_and_contract_sections_have_expected_defaults() -> None:
    config = load_config()

    assert config.contract.min_age_years == 18
    assert config.observability.events_file.name == "pipeline_events.jsonl"
    assert (
        config.observability.drift_warning_share
        <= config.observability.drift_critical_share
    )


def test_observability_rejects_critical_share_below_warning() -> None:
    import pytest
    from pydantic import ValidationError

    from src.utils.config_loader import ObservabilityConfig

    with pytest.raises(ValidationError):
        ObservabilityConfig(drift_warning_share=0.1, drift_critical_share=0.05)
