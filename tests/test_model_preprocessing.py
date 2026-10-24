"""O artefato completo aceita entradas da API após serialização."""

import os
import subprocess
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config.settings import settings
from src.models.preprocessing import build_pipeline, reindex_features
from src.serving.model_loader import load_predictor
from src.utils.config_loader import load_config


def test_reindex_features_keeps_training_order_and_fills_missing_values() -> None:
    frame = pd.DataFrame({"category": ["new"], "extra": [1]})

    result = reindex_features(frame, ["amount", "category"])

    assert result.columns.tolist() == ["amount", "category"]
    assert pd.isna(result.loc[0, "amount"])
    assert result.loc[0, "category"] == "new"


def test_serialized_pipeline_predicts_with_missing_and_unknown_inputs(
    tmp_path, monkeypatch
) -> None:
    features = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100, 200, 120, 220, 140, 240],
            "AMT_CREDIT": [200, 100, 250, 110, 300, 120],
            "AMT_ANNUITY": [20, 10, 25, 11, 30, 12],
            "AMT_GOODS_PRICE": [180, 90, 230, 100, 280, 110],
            "CNT_FAM_MEMBERS": [2, 2, 3, 3, 4, 4],
            "DAYS_BIRTH": [-7300, -14600, -10950, -18250, -12775, -21900],
            "DAYS_EMPLOYED": [-365, -3650, -730, -4380, -1095, -5475],
            "CATEGORY": ["a", "b", "a", "b", "a", "b"],
        }
    )
    pipeline = build_pipeline(features, LogisticRegression(max_iter=1000))
    pipeline.fit(features, [0, 1, 0, 1, 0, 1])
    path = tmp_path / load_config().artifacts.model_file
    joblib.dump(pipeline, path)

    restored = joblib.load(path)
    request = pd.DataFrame([{"CATEGORY": "unseen", "AMT_CREDIT": 150}])
    probabilities = restored.predict_proba(request)

    assert list(restored.named_steps) == ["align", "engineer", "prep", "clf"]
    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0, 1] <= 1

    monkeypatch.setattr(settings, "model_artifacts_path", str(tmp_path))
    probability, label = load_predictor().predict(request.iloc[0].to_dict())
    assert probability == probabilities[0, 1]
    assert label in {"default", "non_default"}

    isolated = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.serving.model_loader import load_predictor; load_predictor()",
        ],
        env={**os.environ, "MODEL_ARTIFACTS_PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert isolated.returncode == 0, isolated.stderr
