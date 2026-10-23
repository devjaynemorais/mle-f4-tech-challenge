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
    assert config.data.production_file == "prod_estavel.csv"
    assert config.model.production_size == 0.2
    assert config.model.cv_folds == 5
    assert config.model.prediction_threshold == 0.5


def test_load_config_rejects_threshold_outside_unit_interval(tmp_path: Path) -> None:
    raw_config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    raw_config["model"]["prediction_threshold"] = 1.5
    invalid_path = tmp_path / "invalid-config.yaml"
    invalid_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    with pytest.raises(ValidationError, match="prediction_threshold"):
        load_config(invalid_path)
