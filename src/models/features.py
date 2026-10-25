"""Engenharia de atributos do Home Credit aplicada no treino e na inferência."""

from __future__ import annotations

import numpy as np
import pandas as pd

EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
DAYS_EMPLOYED_ANOMALY = 365243
DERIVED_FEATURE_COLUMNS = (
    "CREDIT_INCOME_PERCENT",
    "ANNUITY_INCOME_PERCENT",
    "CREDIT_TERM",
    "CREDIT_GOODS_PERCENT",
    "INCOME_PER_PERSON",
    "DAYS_EMPLOYED_PERCENT",
    "AGE_YEARS",
    "EXT_SOURCES_MEAN",
    "EXT_SOURCES_STD",
    "EXT_SOURCES_PROD",
)


def _safe_column(frame: pd.DataFrame, name: str) -> pd.Series[float]:
    """Devolve uma coluna numérica ou uma série nula quando ausente."""
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _safe_ratio(
    numerator: pd.Series[float], denominator: pd.Series[float]
) -> pd.Series[float]:
    """Calcula uma razão; zero e valores infinitos se tornam nulos."""
    ratio = numerator / denominator.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _clean_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    """Converte a sentinela de emprego em valor nulo."""
    if "DAYS_EMPLOYED" in frame.columns:
        frame["DAYS_EMPLOYED"] = frame["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_ANOMALY, np.nan
        )
    return frame


def _add_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cria razões financeiras e renda por pessoa."""
    income = _safe_column(frame, "AMT_INCOME_TOTAL")
    credit = _safe_column(frame, "AMT_CREDIT")
    annuity = _safe_column(frame, "AMT_ANNUITY")
    goods = _safe_column(frame, "AMT_GOODS_PRICE")
    family = _safe_column(frame, "CNT_FAM_MEMBERS")
    frame["CREDIT_INCOME_PERCENT"] = _safe_ratio(credit, income)
    frame["ANNUITY_INCOME_PERCENT"] = _safe_ratio(annuity, income)
    frame["CREDIT_TERM"] = _safe_ratio(annuity, credit)
    frame["CREDIT_GOODS_PERCENT"] = _safe_ratio(credit, goods)
    frame["INCOME_PER_PERSON"] = _safe_ratio(income, family)
    return frame


def _add_tenure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cria idade e proporção da vida empregada."""
    days_birth = _safe_column(frame, "DAYS_BIRTH")
    days_employed = _safe_column(frame, "DAYS_EMPLOYED")
    frame["DAYS_EMPLOYED_PERCENT"] = _safe_ratio(days_employed, days_birth)
    frame["AGE_YEARS"] = -days_birth / 365.0
    return frame


def _add_external_source_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega os escores externos disponíveis."""
    available = [name for name in EXT_SOURCE_COLUMNS if name in frame.columns]
    if not available:
        return frame
    sources = frame[available].apply(pd.to_numeric, errors="coerce")
    frame["EXT_SOURCES_MEAN"] = sources.mean(axis=1)
    frame["EXT_SOURCES_STD"] = sources.std(axis=1)
    frame["EXT_SOURCES_PROD"] = sources.prod(axis=1, min_count=len(available))
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adiciona atributos de domínio sem modificar o DataFrame de entrada."""
    result = frame.copy()
    result = _clean_anomalies(result)
    result = _add_ratio_features(result)
    result = _add_tenure_features(result)
    return _add_external_source_features(result)
