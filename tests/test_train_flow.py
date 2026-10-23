"""Execução integrada de geração de dados, seleção e ajuste final."""

import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import loader, processed
from src.models import artifacts, evaluation
from src.models import train as training
from src.utils.config_loader import load_config


def test_train_generates_two_datasets_and_fits_model_on_reference(
    tmp_path, monkeypatch
) -> None:
    for module in (loader, processed, artifacts):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(training, "log_to_mlflow", lambda *args: None)

    def candidate(*args):
        return {"logistic_regression": LogisticRegression(max_iter=1000)}

    monkeypatch.setattr(training, "candidate_estimators", candidate)
    monkeypatch.setattr(evaluation, "candidate_estimators", candidate)
    raw = tmp_path / "data/raw/application_train.csv"
    raw.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "SK_ID_CURR": range(40),
            "TARGET": [0, 1] * 20,
            "AMT_CREDIT": [100.0, 200.0] * 20,
            "AMT_INCOME_TOTAL": [50.0] * 40,
            "AMT_ANNUITY": [10.0] * 40,
            "AMT_GOODS_PRICE": [100.0] * 40,
            "CNT_FAM_MEMBERS": [2.0] * 40,
            "DAYS_BIRTH": [-10000] * 40,
            "DAYS_EMPLOYED": [-1000] * 40,
        }
    ).to_csv(raw, index=False)
    config = load_config().model_copy(
        update={"model": load_config().model.model_copy(update={"cv_folds": 2})}
    )

    winner = training.train(config)

    reference = pd.read_csv(tmp_path / "data/processed/ref_data.csv")
    stable = pd.read_csv(tmp_path / "data/processed/prod_data.csv")
    model = joblib.load(tmp_path / "models/artifacts/credit_model.joblib")
    metrics = json.loads((tmp_path / "metrics/training_metrics.json").read_text())
    assert len(reference) == 32
    assert len(stable) == 8
    assert stable["TARGET"].isin([0, 1]).all()
    assert winner.name == metrics["selected_model"]
    assert len(metrics["cv_roc_auc"][winner.name]["folds"]) == 2
    assert model.predict_proba(
        reference.drop(columns=["SK_ID_CURR", "TARGET"])
    ).shape == (32, 2)
