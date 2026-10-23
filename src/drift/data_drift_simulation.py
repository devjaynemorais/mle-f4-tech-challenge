"""Gera períodos estável e driftado a partir da produção-base processada."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.features import EXT_SOURCE_COLUMNS, engineer_features
from src.utils.config_loader import PROJECT_ROOT, load_config

STABLE_FILE = "prod_periodo1_estavel.csv"
DRIFT_FILE = "prod_periodo2_drift.csv"
MANIFEST_FILE = "data_drift_simulation_manifest.json"
SYNTHETIC_ID_START = 1_000_000_000
REQUIRED_COLUMNS = {
    "SK_ID_CURR",
    "TARGET",
    "DAYS_BIRTH",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_ANNUITY",
    "CODE_GENDER",
    "AGE_YEARS",
    "CREDIT_TERM",
    "CREDIT_GOODS_PERCENT",
    "EXT_SOURCES_MEAN",
    *EXT_SOURCE_COLUMNS,
}


@dataclass(frozen=True)
class DriftParameters:
    """Intensidades controladas para o perfil de produção driftado."""

    median_age: float = 33.1
    credit_factor: float = 1.20
    goods_factor: float = 1.05
    annuity_factor: float = 1.15
    external_score_factor: float = 0.85


def _validate_parameters(n: int, rate: float, seed: int, p: DriftParameters) -> None:
    """Rejeita escolhas fora do domínio definido para a simulação."""
    if n < 16:
        raise ValueError("N deve ser pelo menos 16")
    if not 0.11 <= rate <= 0.13:
        raise ValueError("target_rate deve estar entre 0.11 e 0.13")
    if seed < 0:
        raise ValueError("seed deve ser não negativa")
    if p.median_age <= 18 or min(asdict(p).values()) <= 0:
        raise ValueError("idade mediana e fatores devem ser positivos")


def _validate_source(source: pd.DataFrame) -> None:
    """Confere o contrato mínimo da produção-base processada."""
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError("colunas obrigatórias ausentes na produção-base")
    if not source["TARGET"].isin([0, 1]).all():
        raise ValueError("TARGET deve conter apenas 0 e 1, sem nulos")
    if source["TARGET"].nunique() != 2:
        raise ValueError("TARGET precisa conter as duas classes")
    if source["SK_ID_CURR"].isna().any():
        raise ValueError("SK_ID_CURR não pode conter nulos")


def _sample_rows(source: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Amostra as duas classes na proporção original e garante split estratificado."""
    positive = source.loc[source.TARGET == 1]
    negative = source.loc[source.TARGET == 0]
    n_positive = max(2, min(n - 2, round(source.TARGET.mean() * n)))
    pieces = [
        positive.sample(
            n_positive, replace=n_positive > len(positive), random_state=seed
        ),
        negative.sample(
            n - n_positive, replace=n - n_positive > len(negative), random_state=seed
        ),
    ]
    sampled = pd.concat(pieces).sample(frac=1, random_state=seed)
    sampled = sampled.reset_index(drop=True)
    first_id = max(SYNTHETIC_ID_START, int(source.SK_ID_CURR.max()) + 1)
    sampled["SK_ID_CURR"] = np.arange(n, dtype="int64") + first_id
    return sampled


def _shift_age(frame: pd.DataFrame, median_age: float) -> None:
    """Reescala idades acima de 18 para atingir a mediana especificada."""
    age = (-pd.to_numeric(frame.DAYS_BIRTH, errors="coerce") / 365).clip(lower=18)
    original_median = age.median()
    if pd.isna(original_median) or original_median <= 18:
        raise ValueError("DAYS_BIRTH precisa ter mediana de idade acima de 18")
    factor = (median_age - 18) / (original_median - 18)
    shifted = -(18 + (age - 18) * factor).mul(365).round()
    if pd.api.types.is_integer_dtype(frame.DAYS_BIRTH.dtype):
        shifted = shifted.astype(frame.DAYS_BIRTH.dtype)
    frame["DAYS_BIRTH"] = shifted


def _shift_drivers(frame: pd.DataFrame, p: DriftParameters) -> pd.DataFrame:
    """Altera somente as colunas brutas que definem o novo perfil."""
    result = frame.copy()
    _shift_age(result, p.median_age)
    for name, factor in (
        ("AMT_CREDIT", p.credit_factor),
        ("AMT_GOODS_PRICE", p.goods_factor),
        ("AMT_ANNUITY", p.annuity_factor),
    ):
        result[name] = pd.to_numeric(result[name], errors="coerce") * factor
    for name in EXT_SOURCE_COLUMNS:
        source = pd.to_numeric(result[name], errors="coerce")
        result[name] = (source * p.external_score_factor).clip(0, 1)
    return engineer_features(result)[frame.columns]


def _risk_score(frame: pd.DataFrame) -> pd.Series[float]:
    """Combina percentis de idade, score externo e crédito sobre o bem."""
    columns = (
        ("AGE_YEARS", False),
        ("EXT_SOURCES_MEAN", False),
        ("CREDIT_GOODS_PERCENT", True),
    )
    percentiles = [
        pd.to_numeric(frame[name], errors="coerce")
        .rank(pct=True, ascending=ascending)
        .fillna(0.5)
        for name, ascending in columns
    ]
    return pd.concat(percentiles, axis=1).mean(axis=1)


def assign_synthetic_target(
    drift: pd.DataFrame, target_rate: float, random_state: int
) -> pd.DataFrame:
    """Marca como inadimplentes os maiores riscos até a taxa solicitada."""
    result = drift.reset_index(drop=True).copy()
    score = _risk_score(result).to_numpy()
    tie_break = np.random.default_rng(random_state).permutation(len(result))
    order = np.lexsort((tie_break, -score))
    target = np.zeros(len(result), dtype="int8")
    target[order[: round(target_rate * len(result))]] = 1
    result["TARGET"] = target
    return result


def _manifest(
    stable: pd.DataFrame,
    drift: pd.DataFrame,
    n: int,
    rate: float,
    seed: int,
    p: DriftParameters,
) -> dict[str, object]:
    """Registra parâmetros e a origem de cada rótulo, sem dados individuais."""
    return {
        "n": n,
        "seed": seed,
        "target_rate_requested": rate,
        "intensities": asdict(p),
        "risk_features": ["AGE_YEARS", "EXT_SOURCES_MEAN", "CREDIT_GOODS_PERCENT"],
        "risk_rule": "média dos percentis; idade e score menores, razão maior",
        "period_1": {
            "file": STABLE_FILE,
            "rows": len(stable),
            "target_rate": float(stable.TARGET.mean()),
            "target_source": "original",
        },
        "period_2": {
            "file": DRIFT_FILE,
            "rows": len(drift),
            "target_rate": float(drift.TARGET.mean()),
            "target_source": "synthetic",
        },
    }


def simulate_periods(
    source: pd.DataFrame,
    n: int,
    target_rate: float,
    random_state: int,
    parameters: DriftParameters | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Divide a produção-base e simula drift de features e conceito no período 2."""
    p = parameters or DriftParameters()
    _validate_parameters(n, target_rate, random_state, p)
    _validate_source(source)
    sampled = _sample_rows(source, n, random_state)
    stable, drift = train_test_split(
        sampled,
        test_size=n // 2,
        stratify=sampled.TARGET,
        random_state=random_state,
    )
    stable = stable.reset_index(drop=True)
    drift = _shift_drivers(drift.reset_index(drop=True), p)
    drift = assign_synthetic_target(drift, target_rate, random_state)
    return stable, drift, _manifest(stable, drift, n, target_rate, random_state, p)


def _parser() -> argparse.ArgumentParser:
    """Define a interface de linha de comando da simulação."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, help="total de linhas nos dois períodos")
    parser.add_argument("--target-rate", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--median-age", type=float, default=33.1)
    parser.add_argument("--credit-factor", type=float, default=1.20)
    parser.add_argument("--goods-factor", type=float, default=1.05)
    parser.add_argument("--annuity-factor", type=float, default=1.15)
    parser.add_argument("--external-score-factor", type=float, default=0.85)
    return parser


def _write_outputs(
    stable: pd.DataFrame,
    drift: pd.DataFrame,
    manifest: dict[str, object],
    output_dir: Path,
) -> None:
    """Publica os períodos e o manifesto no diretório solicitado."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stable.to_csv(output_dir / STABLE_FILE, index=False)
    drift.to_csv(output_dir / DRIFT_FILE, index=False)
    (output_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    """Lê a produção-base configurada e gera os artefatos da simulação."""
    args = _parser().parse_args(argv)
    config = load_config()
    directory = PROJECT_ROOT / config.data.processed_path
    input_path = args.input or directory / config.data.production_file
    if not input_path.is_file():
        raise FileNotFoundError("arquivo de produção-base não encontrado")
    source = pd.read_csv(input_path)
    parameters = DriftParameters(
        args.median_age,
        args.credit_factor,
        args.goods_factor,
        args.annuity_factor,
        args.external_score_factor,
    )
    stable, drift, manifest = simulate_periods(
        source,
        len(source) if args.n is None else args.n,
        args.target_rate,
        args.seed,
        parameters,
    )
    manifest["input_file"] = str(input_path)
    _write_outputs(stable, drift, manifest, args.output_dir or directory)
    print(f"Períodos gerados: {len(stable)} estável, {len(drift)} driftado")


if __name__ == "__main__":
    main()
