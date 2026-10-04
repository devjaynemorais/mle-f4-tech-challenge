from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_raw_dataset, summarize_dataset
from src.utils.config_loader import load_config


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_load_raw_dataset_raises_clear_error_when_file_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="application_train.csv"):
        load_raw_dataset(tmp_path / "application_train.csv")


def test_load_raw_dataset_requires_id_and_target_columns(tmp_path: Path) -> None:
    source = _write_csv(tmp_path / "raw.csv", pd.DataFrame({"AMT_CREDIT": [1.0]}))

    with pytest.raises(ValueError, match="colunas obrigatórias ausentes"):
        load_raw_dataset(source)


def test_load_raw_dataset_reads_frame_with_required_columns(tmp_path: Path) -> None:
    data = load_config().data
    frame = pd.DataFrame(
        {
            data.id_column: [1, 2, 3],
            data.target_column: [0, 1, 0],
            "AMT_CREDIT": [1, 2, 3],
        }
    )
    source = _write_csv(tmp_path / "raw.csv", frame)

    loaded = load_raw_dataset(source)

    assert list(loaded.columns) == [data.id_column, data.target_column, "AMT_CREDIT"]
    assert len(loaded) == 3


def test_summarize_dataset_reports_rows_columns_and_default_rate() -> None:
    frame = pd.DataFrame({"TARGET": [0, 1, 0, 0], "AMT_CREDIT": [1, 2, 3, 4]})

    summary = summarize_dataset(frame, "TARGET")

    assert summary == {"rows": 4.0, "columns": 2.0, "default_rate": 0.25}
