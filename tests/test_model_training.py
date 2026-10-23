"""Contratos do preparo e da comparação de candidatos."""

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models import artifacts
from src.models.candidates import candidate_estimators
from src.models.dataset import make_splits, split_features_target, subsample
from src.models.evaluation import Candidate, evaluate_candidate, select_best
from src.utils.config_loader import load_config


def test_subsample_is_reproducible_and_target_split_removes_id() -> None:
    config = load_config()
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"sample_size": 6})}
    )
    frame = pd.DataFrame(
        {"SK_ID_CURR": range(20), "TARGET": [0, 1] * 10, "AMT_CREDIT": range(20)}
    )

    sample = subsample(frame, config)
    repeated = subsample(frame, config)
    features, target = split_features_target(sample, config)

    assert len(sample) == 6
    assert sample.index.tolist() == repeated.index.tolist()
    assert features.columns.tolist() == ["AMT_CREDIT"]
    assert target.tolist() == sample["TARGET"].tolist()


def test_make_splits_preserves_each_class_in_validation() -> None:
    config = load_config()
    features = pd.DataFrame({"AMT_CREDIT": range(20)})
    target = pd.Series([0, 1] * 10)

    x_train, x_valid, y_train, y_valid = make_splits(features, target, config)

    assert len(x_train) == 16
    assert len(x_valid) == 4
    assert y_train.value_counts().to_dict() == {0: 8, 1: 8}
    assert y_valid.value_counts().to_dict() == {0: 2, 1: 2}


def test_candidate_estimators_keep_class_imbalance_weights() -> None:
    estimators = candidate_estimators(pd.Series([0, 0, 0, 1]), load_config())

    assert estimators["logistic_regression"].class_weight == "balanced"
    assert estimators["xgboost"].scale_pos_weight == 3


def test_evaluation_selects_highest_validation_auc() -> None:
    config = load_config()
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
    splits = make_splits(features, target, config)

    evaluated = evaluate_candidate(
        "logistic_regression", LogisticRegression(max_iter=1000), features, splits
    )
    worse = Candidate("worse", evaluated.pipeline, 0.1)

    assert evaluated.roc_auc == 1.0
    assert select_best([worse, evaluated]).name == "logistic_regression"


def test_artifacts_keep_configured_model_path_and_metrics_shape(
    tmp_path, monkeypatch
) -> None:
    config = load_config()
    monkeypatch.setattr(artifacts, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "METRICS_DIR", tmp_path / "metrics")
    pipeline = Pipeline([("clf", LogisticRegression())])
    best = Candidate("logistic_regression", pipeline, 0.75)
    other = Candidate("xgboost", pipeline, 0.7)

    model_path = artifacts.persist_pipeline(pipeline, config)
    metrics_path = artifacts.write_metrics([best, other], best)

    assert model_path == tmp_path / "models/artifacts/credit_model.joblib"
    assert model_path.exists()
    assert metrics_path == tmp_path / "metrics/training_metrics.json"
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "selected_model": "logistic_regression",
        "roc_auc": {"logistic_regression": 0.75, "xgboost": 0.7},
    }
