"""Treino do modelo base de risco de crédito (Etapa 1).

Treina candidatos (Regressão Logística e XGBoost) somente no dataset de
referência, escolhe o de maior ROC-AUC em validação, persiste o Pipeline
completo (realinhamento de colunas + engenharia de features + pré-processamento
+ classificador) em `models/artifacts/` e grava as métricas de validação em
`metrics/`. Esse é o artefato que a API serve.

A engenharia de features e o pré-processamento ficam dentro do Pipeline para
honrar o contrato de `/predict`: chaves ausentes ou nulas em uma requisição são
realinhadas, derivadas e imputadas sem quebrar a inferência.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:  # tracking é opcional: ausência de mlflow não deve quebrar o treino
    import mlflow
except ImportError:  # pragma: no cover - depende do ambiente de execução
    mlflow = None
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data.loader import load_raw_dataset
from src.utils.config_loader import PROJECT_ROOT, ExperimentConfig, load_config
from src.utils.logging_config import configure_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)
METRICS_DIR = PROJECT_ROOT / "metrics"

# Split de validação: (x_train, x_valid, y_train, y_valid)
Splits = tuple[pd.DataFrame, pd.DataFrame, "pd.Series[int]", "pd.Series[int]"]


@dataclass(frozen=True)
class Candidate:
    """Resultado de validação de um modelo candidato."""

    name: str
    pipeline: Pipeline
    roc_auc: float


def reindex_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Alinha as colunas de entrada ao conjunto de features visto no treino."""
    return frame.reindex(columns=columns)


# Colunas EXT_SOURCE_* são os preditores mais fortes do Home Credit.
EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
# Sentinela de "nunca empregado" em DAYS_EMPLOYED (~1000 anos); vira NaN.
DAYS_EMPLOYED_ANOMALY = 365243


def _safe_column(frame: pd.DataFrame, name: str) -> pd.Series[float]:
    """Coluna numérica pedida, ou série de NaN quando ela não existe."""
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _safe_ratio(
    numerator: pd.Series[float], denominator: pd.Series[float]
) -> pd.Series[float]:
    """Razão protegida: divisão por zero ou ausência resulta em NaN."""
    ratio = numerator / denominator.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as features de domínio clássicas do Home Credit.

    Robusto ao contrato de `/predict`: colunas-base ausentes geram derivadas NaN
    (imputadas no pré-processamento) em vez de quebrar a inferência.
    """
    result = frame.copy()

    # DAYS_EMPLOYED usa 365243 como sentinela de "sem emprego"; vira NaN.
    if "DAYS_EMPLOYED" in result.columns:
        result["DAYS_EMPLOYED"] = result["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_ANOMALY, np.nan
        )

    income = _safe_column(result, "AMT_INCOME_TOTAL")
    credit = _safe_column(result, "AMT_CREDIT")
    annuity = _safe_column(result, "AMT_ANNUITY")
    goods = _safe_column(result, "AMT_GOODS_PRICE")
    days_birth = _safe_column(result, "DAYS_BIRTH")
    days_employed = _safe_column(result, "DAYS_EMPLOYED")
    family = _safe_column(result, "CNT_FAM_MEMBERS")

    # Razões de endividamento e capacidade de pagamento.
    result["CREDIT_INCOME_PERCENT"] = _safe_ratio(credit, income)
    result["ANNUITY_INCOME_PERCENT"] = _safe_ratio(annuity, income)
    result["CREDIT_TERM"] = _safe_ratio(annuity, credit)
    result["CREDIT_GOODS_PERCENT"] = _safe_ratio(credit, goods)
    result["INCOME_PER_PERSON"] = _safe_ratio(income, family)

    # Proporção da vida empregada e idade (DAYS_* são dias negativos).
    result["DAYS_EMPLOYED_PERCENT"] = _safe_ratio(days_employed, days_birth)
    result["AGE_YEARS"] = -days_birth / 365.0

    # Agregações dos escores externos (média/desvio/produto).
    available = [name for name in EXT_SOURCE_COLUMNS if name in result.columns]
    if available:
        sources = result[available].apply(pd.to_numeric, errors="coerce")
        result["EXT_SOURCES_MEAN"] = sources.mean(axis=1)
        result["EXT_SOURCES_STD"] = sources.std(axis=1)
        result["EXT_SOURCES_PROD"] = sources.prod(axis=1, min_count=len(available))

    return result


def split_features_target(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.Series[int]]:
    """Separa as features (sem identificador e alvo) do alvo binário."""
    data = config.data
    target = frame[data.target_column].astype(int)
    features = frame.drop(columns=[data.id_column, data.target_column])
    return features, target


def column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Lista as colunas numéricas e categóricas a partir dos dtypes do treino."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Monta imputação/escala numérica e imputação/one-hot categórica."""
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
    """Encadeia realinhamento, engenharia de features, pré-proc. e classificador."""
    aligner = FunctionTransformer(
        reindex_features,
        kw_args={"columns": features.columns.tolist()},
        validate=False,
    )
    engineer = FunctionTransformer(engineer_features, validate=False)
    # Pré-processador dimensionado sobre as colunas já derivadas.
    engineered = engineer_features(features)
    return Pipeline(
        [
            ("align", aligner),
            ("engineer", engineer),
            ("prep", build_preprocessor(engineered)),
            ("clf", estimator),
        ]
    )


def candidate_estimators(
    target: pd.Series[int], config: ExperimentConfig
) -> dict[str, object]:
    """Cria os classificadores candidatos ponderados pelo desbalanceamento."""
    seed = config.model.random_state
    positives = int(target.sum())
    scale_pos_weight = (len(target) - positives) / max(positives, 1)
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="auc",
            random_state=seed,
        ),
    }


def make_splits(
    features: pd.DataFrame, target: pd.Series[int], config: ExperimentConfig
) -> Splits:
    """Divide em treino/validação de forma estratificada pelo alvo."""
    return train_test_split(
        features,
        target,
        test_size=config.model.test_size,
        stratify=target,
        random_state=config.model.random_state,
    )


def evaluate_candidate(
    name: str, estimator: object, features: pd.DataFrame, splits: Splits
) -> Candidate:
    """Treina um candidato e mede o ROC-AUC no conjunto de validação."""
    x_train, x_valid, y_train, y_valid = splits
    pipeline = build_pipeline(features, estimator)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_valid)[:, 1]
    roc_auc = float(roc_auc_score(y_valid, probabilities))
    logger.info("Candidato %s: ROC-AUC de validação %.4f", name, roc_auc)
    return Candidate(name=name, pipeline=pipeline, roc_auc=roc_auc)


def select_best(candidates: list[Candidate]) -> Candidate:
    """Escolhe o candidato de maior ROC-AUC de validação."""
    best = max(candidates, key=lambda candidate: candidate.roc_auc)
    logger.info("Modelo escolhido: %s (ROC-AUC %.4f)", best.name, best.roc_auc)
    return best


def subsample(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Reduz o volume de treino quando `sample_size` está definido no config."""
    sample_size = config.model.sample_size
    if sample_size is None or sample_size >= len(frame):
        return frame
    logger.info("Amostrando %d de %d linhas para o treino", sample_size, len(frame))
    return frame.sample(n=sample_size, random_state=config.model.random_state)


def persist_pipeline(pipeline: Pipeline, config: ExperimentConfig) -> Path:
    """Grava o Pipeline vencedor no caminho de artefatos do config."""
    artifacts = config.artifacts
    path = PROJECT_ROOT / artifacts.model_path / artifacts.model_file
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def write_metrics(candidates: list[Candidate], best: Candidate) -> Path:
    """Registra o ROC-AUC de cada candidato e o modelo escolhido."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "training_metrics.json"
    payload = {
        "selected_model": best.name,
        "roc_auc": {candidate.name: candidate.roc_auc for candidate in candidates},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def log_to_mlflow(
    candidates: list[Candidate],
    best: Candidate,
    artifact_path: Path,
    config: ExperimentConfig,
) -> None:
    """Registra params, ROC-AUC e artefato no MLflow (best-effort).

    Falhas (mlflow ausente ou servidor fora do ar) viram aviso e não
    interrompem o treino. A env MLFLOW_TRACKING_URI tem prioridade sobre o YAML.
    """
    if mlflow is None:
        logger.warning("mlflow não instalado; tracking desativado")
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.mlflow.experiment)
        with mlflow.start_run(run_name=best.name):
            mlflow.log_params(
                {
                    "sample_size": config.model.sample_size,
                    "test_size": config.model.test_size,
                    "random_state": config.model.random_state,
                    "selected_model": best.name,
                }
            )
            for candidate in candidates:
                mlflow.log_metric(f"roc_auc_{candidate.name}", candidate.roc_auc)
            mlflow.log_metric("roc_auc_best", best.roc_auc)
            mlflow.set_tag("selected_model", best.name)
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
        logger.info("Run registrada no MLflow em %s", uri)
    except Exception as error:  # noqa: BLE001 - tracking não deve quebrar o treino
        logger.warning("Falha ao registrar no MLflow (%s): %s", uri, error)


def train(config: ExperimentConfig | None = None) -> Candidate:
    """Executa o treino completo e devolve o candidato vencedor."""
    cfg = config or load_config()
    set_seed(cfg.model.random_state)
    frame = subsample(load_raw_dataset(), cfg)
    features, target = split_features_target(frame, cfg)
    splits = make_splits(features, target, cfg)
    estimators = candidate_estimators(splits[2], cfg)
    candidates = [
        evaluate_candidate(name, estimator, features, splits)
        for name, estimator in estimators.items()
    ]
    best = select_best(candidates)
    artifact_path = persist_pipeline(best.pipeline, cfg)
    write_metrics(candidates, best)
    log_to_mlflow(candidates, best, artifact_path, cfg)
    return best


def main() -> None:
    """Ponto de entrada de `make train`: treina e persiste o artefato."""
    configure_logging()
    best = train()
    logger.info("Treino concluído; artefato pronto para servir (%s)", best.name)


if __name__ == "__main__":
    main()
