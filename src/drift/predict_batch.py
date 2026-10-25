"""Predições em lote dos períodos de produção (Etapa 2).

Pontua o período estável e o driftado com o modelo servido pela API. O alvo do
período driftado é sintético: por isso só o período estável publica AUC. Os
arquivos de predição ficam em `data/processed/` (fora do Git); o resumo
publicado tem apenas agregados.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config.settings import settings
from src.data.ingestion import gate_batch
from src.observability.drift_metrics import publish_scoring
from src.observability.stage_tracking import track_stage
from src.utils.config_loader import PROJECT_ROOT, ExperimentConfig, load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)
STABLE, DRIFTED = "stable", "drifted"


def score_frame(
    model: object, frame: pd.DataFrame, config: ExperimentConfig
) -> pd.DataFrame:
    """Calcula P(default) e o rótulo pelo limiar configurado."""
    data = config.data
    features = frame.drop(columns=[data.id_column, data.target_column])
    probability = model.predict_proba(features)[:, 1]
    label = [
        "default" if p >= config.model.prediction_threshold else "non_default"
        for p in probability
    ]
    return pd.DataFrame(
        {
            data.id_column: frame[data.id_column].to_numpy(),
            "default_probability": probability,
            "prediction": label,
        }
    )


def summarize_scores(
    scored: pd.DataFrame, target: pd.Series | None = None
) -> dict[str, float]:
    """Agrega o lote; AUC só entra quando o alvo é real."""
    stats = {
        "rows": float(len(scored)),
        "score_mean": float(scored["default_probability"].mean()),
        "positive_rate": float((scored["prediction"] == "default").mean()),
    }
    if target is not None and target.nunique() == 2:
        stats["roc_auc"] = float(roc_auc_score(target, scored["default_probability"]))
    return stats


def _load_model(config: ExperimentConfig) -> object:
    """Carrega o pipeline treinado definido no config."""
    path = PROJECT_ROOT / settings.model_artifacts_path / config.artifacts.model_file
    if not path.is_file():
        raise FileNotFoundError("modelo não encontrado; execute o treino antes")
    return joblib.load(path)


def _score_period(
    model: object, source: Path, output: Path, period: str, config: ExperimentConfig
) -> dict[str, float]:
    """Valida o contrato, pontua um período e grava as predições locais."""
    frame = pd.read_csv(source)
    gate_batch(frame, config, period)
    scored = score_frame(model, frame, config)
    scored.to_csv(output, index=False)
    real_target = frame[config.data.target_column] if period == STABLE else None
    stats = summarize_scores(scored, real_target)
    publish_scoring(period, stats)
    return stats


def run_scoring(config: ExperimentConfig) -> dict[str, dict[str, float]]:
    """Pontua os dois períodos e grava o resumo agregado."""
    directory = PROJECT_ROOT / config.data.processed_path
    drift = config.drift
    model = _load_model(config)
    jobs = {
        STABLE: (drift.stable_file, drift.predictions_stable_file),
        DRIFTED: (drift.drifted_file, drift.predictions_drifted_file),
    }
    with track_stage("batch_scoring") as stage:
        results = {
            period: _score_period(
                model, directory / src, directory / out, period, config
            )
            for period, (src, out) in jobs.items()
        }
        stage.rows = int(sum(item["rows"] for item in results.values()))
        stage.metrics.update(
            {f"{p}_score_mean": r["score_mean"] for p, r in results.items()}
        )
    return results


def _write_summary(
    results: dict[str, dict[str, float]], config: ExperimentConfig
) -> None:
    """Grava o resumo agregado (sem identificadores)."""
    path = PROJECT_ROOT / config.drift.scoring_summary_file
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"periods": results, "target_note": "alvo do período drifted é sintético"}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada de `make predict`."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    configure_logging()
    config = load_config()
    results = run_scoring(config)
    _write_summary(results, config)
    for period, stats in results.items():
        logger.info(
            "Período %s: %d linhas, score médio %.4f",
            period,
            int(stats["rows"]),
            stats["score_mean"],
        )


if __name__ == "__main__":
    main()
