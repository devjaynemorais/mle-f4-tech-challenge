from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.settings import settings
from src.serving.model_loader import load_predictor
from src.utils.config_loader import load_config

FEATURES = ["AMT_CREDIT", "AMT_INCOME_TOTAL"]


def _save_tiny_pipeline(directory: Path) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(60, 2)), columns=FEATURES)
    target = (frame["AMT_CREDIT"] > 0).astype(int)
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(random_state=0))]
    ).fit(frame, target)
    config = load_config()
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, directory / config.artifacts.model_file)


def test_load_predictor_raises_clear_error_when_model_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "model_artifacts_path", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Execute o treino"):
        load_predictor()


def test_predictor_returns_probability_and_label_from_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_tiny_pipeline(tmp_path)
    monkeypatch.setattr(settings, "model_artifacts_path", str(tmp_path))
    config = load_config()

    predictor = load_predictor()
    probability, label = predictor.predict({"AMT_CREDIT": 3.0, "AMT_INCOME_TOTAL": 0.0})

    assert predictor.backend == "sklearn"
    assert 0.0 <= probability <= 1.0
    expected = (
        "default" if probability >= config.model.prediction_threshold else "non_default"
    )
    assert label == expected
