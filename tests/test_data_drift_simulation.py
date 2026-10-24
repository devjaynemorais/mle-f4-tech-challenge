"""Contratos da simulação dos dois períodos de produção."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.drift.data_drift_simulation import (
    assign_synthetic_target,
    main,
    simulate_periods,
)
from src.models.dataset import split_features_target
from src.models.features import DERIVED_FEATURE_COLUMNS, engineer_features
from src.utils.config_loader import load_config


def _source(rows: int = 100) -> pd.DataFrame:
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": 100000 + index,
            "TARGET": (index % 12 == 0).astype(int),
            "CODE_GENDER": np.where(index % 2 == 0, "F", "M"),
            "DAYS_BIRTH": -365 * (21 + index % 49),
            "DAYS_EMPLOYED": -365 * (1 + index % 20),
            "AMT_CREDIT": 200000.0 + index * 5000,
            "AMT_GOODS_PRICE": 180000.0 + index * 4000,
            "AMT_ANNUITY": 12000.0 + index * 100,
            "AMT_INCOME_TOTAL": 100000.0 + index * 500,
            "CNT_FAM_MEMBERS": 2,
            "EXT_SOURCE_1": 0.2 + index / (rows * 2),
            "EXT_SOURCE_2": 0.3 + index / (rows * 3),
            "EXT_SOURCE_3": 0.4 + index / (rows * 4),
            "ROW_MARKER": index,
        }
    )
    frame.loc[0, "EXT_SOURCE_1"] = np.nan
    frame.loc[1, "AMT_GOODS_PRICE"] = np.nan
    frame.loc[2, "AMT_GOODS_PRICE"] = 0
    return engineer_features(frame)


@pytest.mark.parametrize("n", [20, 21, 140])
def test_simulation_splits_and_preserves_stable_rows(n: int) -> None:
    source = _source()
    stable, drift, manifest = simulate_periods(source, n, 0.12, 42)

    assert len(stable) + len(drift) == n
    assert abs(len(stable) - len(drift)) <= 1
    assert stable.columns.tolist() == drift.columns.tolist() == source.columns.tolist()
    assert pd.concat([stable.SK_ID_CURR, drift.SK_ID_CURR]).is_unique
    assert set(stable.SK_ID_CURR).isdisjoint(set(source.SK_ID_CURR))
    assert min(stable.SK_ID_CURR.min(), drift.SK_ID_CURR.min()) >= 1_000_000_000
    expected = source.set_index("ROW_MARKER").loc[stable.ROW_MARKER]
    actual = stable.set_index("ROW_MARKER")
    assert_frame_equal(
        actual.drop(columns="SK_ID_CURR").reset_index(drop=True),
        expected.drop(columns="SK_ID_CURR").reset_index(drop=True),
        check_dtype=False,
    )
    assert manifest["period_1"]["target_source"] == "original"
    assert manifest["period_2"]["target_source"] == "synthetic"


def test_simulation_mutates_only_selected_drivers_and_recomputes_features() -> None:
    source = _source()
    _, drift, _ = simulate_periods(source, 100, 0.12, 42)
    original = source.set_index("ROW_MARKER").loc[drift.ROW_MARKER].reset_index()

    assert drift.AGE_YEARS.median() == pytest.approx(33.1, abs=0.05)
    assert drift.AGE_YEARS.min() >= 18
    assert drift.DAYS_BIRTH.dtype == original.DAYS_BIRTH.dtype
    for name, factor in (
        ("AMT_CREDIT", 1.20),
        ("AMT_GOODS_PRICE", 1.05),
        ("AMT_ANNUITY", 1.15),
        ("EXT_SOURCE_1", 0.85),
        ("EXT_SOURCE_2", 0.85),
        ("EXT_SOURCE_3", 0.85),
    ):
        np.testing.assert_allclose(drift[name], original[name] * factor, equal_nan=True)
    assert drift.CODE_GENDER.tolist() == original.CODE_GENDER.tolist()
    assert drift.AMT_INCOME_TOTAL.tolist() == original.AMT_INCOME_TOTAL.tolist()
    assert drift.ROW_MARKER.tolist() == original.ROW_MARKER.tolist()
    np.testing.assert_allclose(
        drift.CREDIT_TERM, drift.AMT_ANNUITY / drift.AMT_CREDIT, equal_nan=True
    )
    assert drift.CREDIT_GOODS_PERCENT.isna().sum() >= 1
    assert not np.isinf(drift.select_dtypes(include="number")).any().any()


def test_synthetic_target_depends_on_risk_and_handles_missing_values() -> None:
    drift = pd.DataFrame(
        {
            "AGE_YEARS": [20, 30, 40, 50, 60, 70, 80, 90, 35],
            "EXT_SOURCES_MEAN": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, np.nan],
            "CREDIT_GOODS_PERCENT": [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, np.nan],
            "TARGET": [0] * 9,
        }
    )
    result = assign_synthetic_target(drift, 0.12, 42)

    assert result.TARGET.sum() == 1
    assert result.loc[0, "TARGET"] == 1
    assert pd.isna(result.loc[8, "EXT_SOURCES_MEAN"])
    assert pd.isna(result.loc[8, "CREDIT_GOODS_PERCENT"])


def test_simulation_is_reproducible_and_model_features_are_clean() -> None:
    source = _source()
    first = simulate_periods(source, 60, 0.12, 42)
    second = simulate_periods(source, 60, 0.12, 42)

    assert_frame_equal(first[0], second[0])
    assert_frame_equal(first[1], second[1])
    assert first[2] == second[2]
    features, target = split_features_target(first[1], load_config())
    assert "SK_ID_CURR" not in features
    assert "TARGET" not in features
    assert not set(DERIVED_FEATURE_COLUMNS).intersection(features)
    assert target.sum() == round(0.12 * len(first[1]))


@pytest.mark.parametrize("n,rate", [(15, 0.12), (20, 0.10), (20, 0.14)])
def test_simulation_rejects_invalid_parameters(n: int, rate: float) -> None:
    with pytest.raises(ValueError):
        simulate_periods(_source(), n, rate, 42)


@pytest.mark.parametrize("column", ["TARGET", "DAYS_BIRTH", "AMT_CREDIT"])
def test_simulation_rejects_missing_columns(column: str) -> None:
    with pytest.raises(ValueError, match="colunas obrigatórias"):
        simulate_periods(_source().drop(columns=column), 20, 0.12, 42)


def test_simulation_rejects_invalid_target() -> None:
    source = _source()
    source.loc[0, "TARGET"] = 2
    with pytest.raises(ValueError, match="TARGET"):
        simulate_periods(source, 20, 0.12, 42)


def test_cli_writes_periods_and_manifest(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "prod_data.csv"
    output = tmp_path / "periods"
    _source().to_csv(source_path, index=False)
    config = load_config()
    monkeypatch.setattr(
        "src.drift.data_drift_simulation.load_config",
        lambda: SimpleNamespace(
            data=config.data,
            drift=config.drift.model_copy(
                update={
                    "stable_file": "stable-custom.csv",
                    "drifted_file": "drift-custom.csv",
                    "manifest_file": "manifest-custom.json",
                }
            ),
        ),
    )

    main(["--input", str(source_path), "--output-dir", str(output), "--n", "40"])

    stable = pd.read_csv(output / "stable-custom.csv")
    drift = pd.read_csv(output / "drift-custom.csv")
    manifest = json.loads((output / "manifest-custom.json").read_text(encoding="utf-8"))
    assert len(stable) == len(drift) == 20
    assert drift.TARGET.sum() == round(0.12 * len(drift))
    assert manifest["n"] == 40
    assert manifest["period_2"]["target_source"] == "synthetic"
    assert manifest["period_1"]["file"] == "stable-custom.csv"
    assert manifest["period_2"]["file"] == "drift-custom.csv"
    assert manifest["input_file"] == str(source_path)
    assert not (tmp_path / "prod_periodo1_estavel.csv").exists()


def test_cli_rejects_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="produção-base"):
        main(["--input", str(tmp_path / "missing.csv")])
