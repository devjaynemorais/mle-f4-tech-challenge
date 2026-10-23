"""Comportamentos de engenharia usados no treino e na inferência."""

import numpy as np
import pandas as pd
import pytest

from src.models.features import engineer_features


def test_engineer_features_handles_zero_missing_and_employment_sentinel() -> None:
    frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [0.0, 100.0],
            "AMT_CREDIT": [50.0, 200.0],
            "AMT_ANNUITY": [10.0, 20.0],
            "DAYS_BIRTH": [-7300, -14600],
            "DAYS_EMPLOYED": [365243, -3650],
        }
    )

    result = engineer_features(frame)

    assert np.isnan(result.loc[0, "DAYS_EMPLOYED"])
    assert np.isnan(result.loc[0, "CREDIT_INCOME_PERCENT"])
    assert np.isnan(result.loc[0, "INCOME_PER_PERSON"])
    assert result.loc[1, "CREDIT_INCOME_PERCENT"] == pytest.approx(2.0)
    assert result.loc[1, "DAYS_EMPLOYED_PERCENT"] == pytest.approx(0.25)
    assert result.loc[1, "AGE_YEARS"] == pytest.approx(40.0)
    assert frame.loc[0, "DAYS_EMPLOYED"] == 365243


def test_engineer_features_aggregates_available_external_sources() -> None:
    frame = pd.DataFrame({"EXT_SOURCE_1": [0.2, np.nan], "EXT_SOURCE_2": [0.4, 0.6]})

    result = engineer_features(frame)

    assert result.loc[0, "EXT_SOURCES_MEAN"] == pytest.approx(0.3)
    assert result.loc[0, "EXT_SOURCES_PROD"] == pytest.approx(0.08)
    assert result.loc[1, "EXT_SOURCES_MEAN"] == pytest.approx(0.6)
    assert np.isnan(result.loc[1, "EXT_SOURCES_PROD"])
