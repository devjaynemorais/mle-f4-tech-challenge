"""Gera a referência e a produção-base a partir do CSV bruto."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.features import engineer_features
from src.utils.config_loader import PROJECT_ROOT, ExperimentConfig


def _validate_raw(frame: pd.DataFrame, config: ExperimentConfig) -> None:
    """Impede sobreposição por ID e alvos inválidos antes da divisão."""
    ids = frame[config.data.id_column]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("identificadores duplicados ou ausentes no CSV bruto")
    if not frame[config.data.target_column].isin([0, 1]).all():
        raise ValueError("alvo deve conter apenas 0 e 1, sem valores ausentes")


def _stage_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Grava um CSV temporário sem alterar o arquivo publicado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".csv")
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_pair(staged: list[Path], destinations: list[Path]) -> None:
    """Publica o par e restaura a versão anterior se houver falha."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for path in destinations:
            if path.exists():
                backup = path.with_name(f".{path.name}.{uuid4().hex}.bak")
                path.replace(backup)
                backups[path] = backup
        for temporary, path in zip(staged, destinations, strict=True):
            temporary.replace(path)
            published.append(path)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        for path, backup in backups.items():
            backup.replace(path)
        raise
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def _write_pair(
    reference: pd.DataFrame, stable: pd.DataFrame, paths: list[Path]
) -> None:
    """Conclui as duas escritas antes de publicar qualquer arquivo."""
    staged: list[Path] = []
    try:
        staged.append(_stage_csv(reference, paths[0]))
        staged.append(_stage_csv(stable, paths[1]))
        _publish_pair(staged, paths)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def prepare_datasets(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa, transforma e salva referência e produção-base."""
    _validate_raw(frame, config)
    reference, stable = train_test_split(
        frame,
        test_size=config.model.production_size,
        stratify=frame[config.data.target_column],
        random_state=config.model.random_state,
    )
    reference = engineer_features(reference).reset_index(drop=True)
    stable = engineer_features(stable).reset_index(drop=True)
    directory = PROJECT_ROOT / config.data.processed_path
    _write_pair(
        reference,
        stable,
        [
            directory / config.data.reference_file,
            directory / config.data.production_file,
        ],
    )
    return reference, stable
