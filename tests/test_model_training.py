"""Contratos do preparo e da comparação de candidatos."""

import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models import artifacts, evaluation
from src.models.candidates import candidate_estimators
from src.models.dataset import split_features_target
from src.models.evaluation import Candidate, evaluate_candidate, select_best
from src.models.features import engineer_features
from src.utils.config_loader import load_config


def test_target_split_removes_id_and_precomputed_features() -> None:
    config = load_config()
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": range(4),
            "TARGET": [0, 1, 0, 1],
            "AMT_CREDIT": range(4),
            "CREDIT_INCOME_PERCENT": [0.5] * 4,
        }
    )

    features, target = split_features_target(frame, config)

    assert features.columns.tolist() == ["AMT_CREDIT"]
    assert target.tolist() == [0, 1, 0, 1]


def test_model_inputs_drop_every_precomputed_feature() -> None:
    raw = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "AMT_CREDIT": [100, 200],
            "AMT_INCOME_TOTAL": [50, 100],
            "AMT_ANNUITY": [10, 20],
            "AMT_GOODS_PRICE": [100, 200],
            "CNT_FAM_MEMBERS": [2, 2],
            "DAYS_BIRTH": [-10000, -12000],
            "DAYS_EMPLOYED": [-1000, -2000],
            "EXT_SOURCE_1": [0.1, 0.2],
        }
    )

    inputs, _ = split_features_target(engineer_features(raw), load_config())

    expected = raw.columns.drop(["SK_ID_CURR", "TARGET"]).tolist()
    assert inputs.columns.tolist() == expected


def test_candidate_estimators_keep_class_imbalance_weights() -> None:
    estimators = candidate_estimators(pd.Series([0, 0, 0, 1]), load_config())

    assert estimators["logistic_regression"].class_weight == "balanced"
    assert estimators["xgboost"].scale_pos_weight == 3


def test_evaluation_selects_highest_cross_validated_auc() -> None:
    config = load_config()
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"cv_folds": 3})}
    )
    features = pd.DataFrame(
        {
            "AMT_CREDIT": [1, 10] * 10,
            "AMT_INCOME_TOTAL": [10] * 20,
            "AMT_ANNUITY": [1, 2] * 10,
            "AMT_GOODS_PRICE": [1, 10] * 10,
            "CNT_FAM_MEMBERS": [2] * 20,
            "DAYS_BIRTH": [-10000] * 20,
            "DAYS_EMPLOYED": [-1000] * 20,
        }
    )
    target = pd.Series([0, 1] * 10)

    evaluated = evaluate_candidate("logistic_regression", features, target, config)
    worse = Candidate("worse", 0.1, 0.0, (0.1, 0.1, 0.1))

    assert evaluated.roc_auc == 1.0
    assert len(evaluated.fold_scores) == 3
    assert select_best([worse, evaluated]).name == "logistic_regression"


def test_cv_rejects_reference_with_too_few_positive_rows() -> None:
    config = load_config()
    features = pd.DataFrame({"AMT_CREDIT": range(10)})
    target = pd.Series([0] * 8 + [1] * 2)

    with pytest.raises(ValueError, match="classes insuficientes"):
        evaluate_candidate("logistic_regression", features, target, config)


def test_cv_rejects_reference_with_only_one_class() -> None:
    features = pd.DataFrame({"AMT_CREDIT": range(10)})
    target = pd.Series([0] * 10)

    with pytest.raises(ValueError, match="duas classes"):
        evaluate_candidate("logistic_regression", features, target, load_config())


def test_cv_builds_estimator_using_only_each_training_fold(monkeypatch) -> None:
    config = load_config().model_copy(
        update={"model": load_config().model.model_copy(update={"cv_folds": 3})}
    )
    features = pd.DataFrame(
        {
            "AMT_CREDIT": [1, 2, 3] * 5,
            "AMT_INCOME_TOTAL": [10] * 15,
            "AMT_ANNUITY": [2] * 15,
            "AMT_GOODS_PRICE": [5] * 15,
            "CNT_FAM_MEMBERS": [2] * 15,
            "DAYS_BIRTH": [-10000] * 15,
            "DAYS_EMPLOYED": [-1000] * 15,
        }
    )
    target = pd.Series([0] * 9 + [1] * 6)
    fold_sizes = []

    def fold_estimator(fold_target, _config):
        fold_sizes.append(len(fold_target))
        return {"logistic_regression": LogisticRegression(max_iter=1000)}

    monkeypatch.setattr(evaluation, "candidate_estimators", fold_estimator)

    evaluate_candidate("logistic_regression", features, target, config)

    assert fold_sizes == [10, 10, 10]


def test_artifacts_keep_configured_model_path_and_metrics_shape(
    tmp_path, monkeypatch
) -> None:
    config = load_config()
    monkeypatch.setattr(artifacts, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "METRICS_DIR", tmp_path / "metrics")
    pipeline = Pipeline([("clf", LogisticRegression())])
    best = Candidate("logistic_regression", 0.75, 0.02, (0.73, 0.77))
    other = Candidate("xgboost", 0.7, 0.01, (0.69, 0.71))

    model_path = artifacts.persist_pipeline(pipeline, config)
    metrics_path = artifacts.write_metrics([best, other], best)

    assert model_path == tmp_path / "models/artifacts/credit_model.joblib"
    assert model_path.exists()
    assert metrics_path == tmp_path / "metrics/training_metrics.json"
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "selected_model": "logistic_regression",
        "cv_roc_auc": {
            "logistic_regression": {"mean": 0.75, "std": 0.02, "folds": [0.73, 0.77]},
            "xgboost": {"mean": 0.7, "std": 0.01, "folds": [0.69, 0.71]},
        },
    }
