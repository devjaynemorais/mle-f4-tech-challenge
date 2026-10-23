"""Construção do pipeline completo serializado para a API."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.models.features import engineer_features


def reindex_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Alinha as colunas de entrada ao conjunto de treino."""
    return frame.reindex(columns=columns)


def column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa colunas numéricas e categóricas pelos tipos de treino."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Monta imputação, escala numérica e codificação categórica."""
    numeric, categorical = column_groups(features)
    numeric_steps = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)]
    )


def build_pipeline(features: pd.DataFrame, estimator: object) -> Pipeline:
    """Inclui alinhamento, engenharia, pré-processamento e classificador."""
    aligner = FunctionTransformer(
        reindex_features,
        kw_args={"columns": features.columns.tolist()},
        validate=False,
    )
    engineer = FunctionTransformer(engineer_features, validate=False)
    engineered = engineer_features(features)
    return Pipeline(
        [
            ("align", aligner),
            ("engineer", engineer),
            ("prep", build_preprocessor(engineered)),
            ("clf", estimator),
        ]
    )
