"""Contratos dos dois datasets gerados durante o treino."""

import pandas as pd
import pytest

from src.data.processed import prepare_datasets
from src.utils.config_loader import load_config


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(20),
            "TARGET": [0, 1] * 10,
            "AMT_CREDIT": [100.0] * 20,
            "AMT_INCOME_TOTAL": [50.0] * 20,
            "DAYS_EMPLOYED": [365243] * 20,
        }
    )


def test_prepare_datasets_writes_disjoint_stratified_processed_csvs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.data.processed.PROJECT_ROOT", tmp_path)
    config = load_config().model_copy(
        update={
            "data": load_config().data.model_copy(
                update={
                    "reference_file": "ref_data.csv",
                    "production_file": "prod_estavel.csv",
                }
            )
        }
    )

    reference, production = prepare_datasets(_raw_frame(), config)
    saved_reference = pd.read_csv(tmp_path / "data/processed/ref_data.csv")
    saved_production = pd.read_csv(tmp_path / "data/processed/prod_estavel.csv")

    assert len(reference) == len(saved_reference) == 16
    assert len(production) == len(saved_production) == 4
    assert reference["TARGET"].value_counts().to_dict() == {0: 8, 1: 8}
    assert "TARGET" not in production.columns
    assert set(reference.SK_ID_CURR).isdisjoint(set(production.SK_ID_CURR))
    assert "CREDIT_INCOME_PERCENT" in saved_reference.columns
    assert "CREDIT_INCOME_PERCENT" in saved_production.columns
    assert saved_reference.DAYS_EMPLOYED.isna().all()


def test_prepare_datasets_rejects_duplicate_ids_before_writing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.data.processed.PROJECT_ROOT", tmp_path)
    frame = _raw_frame()
    frame.loc[1, "SK_ID_CURR"] = frame.loc[0, "SK_ID_CURR"]

    with pytest.raises(ValueError, match="identificadores duplicados"):
        prepare_datasets(frame, load_config())

    assert not (tmp_path / "data/processed/ref_data.csv").exists()


def test_prepare_datasets_keeps_previous_pair_if_second_write_fails(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.data.processed.PROJECT_ROOT", tmp_path)
    directory = tmp_path / "data/processed"
    directory.mkdir(parents=True)
    reference_path = directory / "ref_data.csv"
    stable_path = directory / "prod_estavel.csv"
    reference_path.write_text("old-reference", encoding="utf-8")
    stable_path.write_text("old-production", encoding="utf-8")
    original = pd.DataFrame.to_csv
    calls = 0

    def fail_second_write(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disco indisponível")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_second_write)

    with pytest.raises(OSError, match="disco indisponível"):
        prepare_datasets(_raw_frame(), load_config())

    assert reference_path.read_text(encoding="utf-8") == "old-reference"
    assert stable_path.read_text(encoding="utf-8") == "old-production"


def test_prepare_datasets_restores_previous_pair_if_publication_fails(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.data.processed.PROJECT_ROOT", tmp_path)
    directory = tmp_path / "data/processed"
    directory.mkdir(parents=True)
    reference_path = directory / "ref_data.csv"
    stable_path = directory / "prod_estavel.csv"
    reference_path.write_text("old-reference", encoding="utf-8")
    stable_path.write_text("old-production", encoding="utf-8")
    original = type(reference_path).replace

    def fail_stable_publication(self, target):
        if target == stable_path and self.suffix == ".csv":
            raise OSError("publicação indisponível")
        return original(self, target)

    monkeypatch.setattr(type(reference_path), "replace", fail_stable_publication)

    with pytest.raises(OSError, match="publicação indisponível"):
        prepare_datasets(_raw_frame(), load_config())

    assert reference_path.read_text(encoding="utf-8") == "old-reference"
    assert stable_path.read_text(encoding="utf-8") == "old-production"
