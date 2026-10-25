"""Contrato de dados: cada regra rígida barra o lote e nada vaza nas mensagens."""

import pandas as pd
import pytest

from src.data.contracts import DataContractError, enforce_contract, validate_batch
from src.data.ingestion import gate_batch
from src.utils.config_loader import load_config


def _batch() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [101, 102, 103, 104],
            "TARGET": [0, 1, 0, 0],
            "AMT_INCOME_TOTAL": [50_000.0, 80_000.0, 120_000.0, 65_000.0],
            "AMT_CREDIT": [100_000.0, 200_000.0, 300_000.0, 150_000.0],
            "DAYS_BIRTH": [-10_000, -12_000, -15_000, -9_000],
        }
    )


def test_valid_batch_passes_the_contract() -> None:
    result = validate_batch(_batch(), load_config())

    assert result.valid
    assert result.n_rows == 4
    assert result.violations == {}


@pytest.mark.parametrize(
    ("column", "row", "value", "rule"),
    [
        ("SK_ID_CURR", 1, 101, "SK_ID_CURR.id_unico"),
        ("TARGET", 1, 2, "TARGET.alvo_binario"),
        ("AMT_INCOME_TOTAL", 0, None, "AMT_INCOME_TOTAL.nao_nulo"),
        ("AMT_INCOME_TOTAL", 0, -5.0, "AMT_INCOME_TOTAL.renda_positiva"),
        ("AMT_INCOME_TOTAL", 0, 5e12, "AMT_INCOME_TOTAL.renda_abaixo_do_teto"),
        ("AMT_CREDIT", 0, 0.0, "AMT_CREDIT.credito_positivo"),
        ("DAYS_BIRTH", 2, -5_000, "DAYS_BIRTH.idade_acima_de_18_anos"),
        ("DAYS_BIRTH", 2, None, "DAYS_BIRTH.nao_nulo"),
    ],
)
def test_each_hard_rule_blocks_the_batch(column, row, value, rule) -> None:
    frame = _batch().astype({column: "float64"})
    frame.loc[row, column] = value

    result = validate_batch(frame, load_config())

    assert not result.valid
    assert rule in result.violations


def test_duplicated_identifier_flags_every_repeated_row() -> None:
    frame = _batch()
    frame.loc[1, "SK_ID_CURR"] = 101

    result = validate_batch(frame, load_config())

    assert result.violations["SK_ID_CURR.id_unico"] == 2


def test_missing_required_column_is_a_violation() -> None:
    result = validate_batch(_batch().drop(columns=["DAYS_BIRTH"]), load_config())

    assert result.violations == {"schema.coluna_ausente": 1}


def test_non_numeric_values_are_reported_as_invalid_type() -> None:
    frame = _batch().astype({"AMT_CREDIT": "object"})
    frame.loc[0, "AMT_CREDIT"] = "abc"

    result = validate_batch(frame, load_config())

    assert "AMT_CREDIT.tipo_invalido" in result.violations


def test_enforce_contract_raises_without_exposing_values() -> None:
    frame = _batch()
    frame.loc[0, "SK_ID_CURR"] = 102
    frame.loc[2, "DAYS_BIRTH"] = -4_321

    with pytest.raises(DataContractError) as error:
        enforce_contract(frame, load_config())

    message = str(error.value)
    assert "SK_ID_CURR.id_unico=2" in message
    for secret in ("102", "4321", "120000"):
        assert secret not in message
    assert error.value.result.violations


def test_gate_batch_blocks_and_records_the_violation(tmp_path, monkeypatch) -> None:
    published = []
    monkeypatch.setattr(
        "src.data.ingestion.publish_contract",
        lambda dataset, violations: published.append((dataset, violations)),
    )
    frame = _batch()
    frame.loc[0, "AMT_INCOME_TOTAL"] = None

    with pytest.raises(DataContractError):
        gate_batch(frame, load_config(), "raw")

    assert published == [("raw", {"AMT_INCOME_TOTAL.nao_nulo": 1})]
    events = (tmp_path / "events-root/logs/pipeline_events.jsonl").read_text()
    assert '"event": "contract_violation"' in events
    assert '"alert": "DataContractViolation"' in events
    assert '"status": "failure"' in events


def test_gate_batch_passes_a_valid_batch_and_clears_the_block(monkeypatch) -> None:
    published = []
    monkeypatch.setattr(
        "src.data.ingestion.publish_contract",
        lambda dataset, violations: published.append((dataset, violations)),
    )

    result = gate_batch(_batch(), load_config(), "raw")

    assert result.valid
    assert published == [("raw", {})]
