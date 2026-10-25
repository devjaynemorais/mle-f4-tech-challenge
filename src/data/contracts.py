"""Contrato de dados Pandera do Home Credit.

Regras rígidas aplicadas a cada lote antes de ele chegar ao modelo:

- `SK_ID_CURR`: obrigatório e sem duplicatas;
- `TARGET`: obrigatório e binário (0 ou 1);
- `AMT_INCOME_TOTAL`: obrigatória, positiva e abaixo de um teto;
- `AMT_CREDIT`: obrigatório e positivo;
- `DAYS_BIRTH`: obrigatório e com idade acima do mínimo (maioridade).

O resultado guarda só nomes de regras e contagens de linhas. Valores,
identificadores e linhas de clientes nunca entram em mensagens ou eventos.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa

from src.utils.config_loader import ContractConfig, ExperimentConfig

DAYS_PER_YEAR = 365.25
SCHEMA_RULE = "schema"


class DataContractError(ValueError):
    """Lote barrado: violou ao menos uma regra do contrato."""

    def __init__(self, result: ContractResult) -> None:
        rules = ", ".join(
            f"{rule}={count}" for rule, count in result.violations.items()
        )
        super().__init__(f"lote barrado pelo contrato de dados: {rules}")
        self.result = result


@dataclass(frozen=True)
class ContractResult:
    """Resumo da validação de um lote: sem valores, só regras e contagens."""

    n_rows: int
    violations: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """Verdadeiro quando nenhuma regra foi violada."""
        return not self.violations


def _check(rule: str, predicate: Callable[[pd.Series], pd.Series]) -> pa.Check:
    """Cria uma checagem vetorizada com nome estável de regra."""
    return pa.Check(predicate, name=rule, error=rule)


def build_schema(
    config: ContractConfig, id_column: str, target_column: str
) -> pa.DataFrameSchema:
    """Monta o schema Pandera com as regras rígidas do contrato."""
    min_days = config.min_age_years * DAYS_PER_YEAR
    return pa.DataFrameSchema(
        {
            id_column: pa.Column(
                int,
                _check("id_unico", lambda s: ~s.duplicated(keep=False)),
                coerce=True,
            ),
            target_column: pa.Column(
                int, _check("alvo_binario", lambda s: s.isin([0, 1])), coerce=True
            ),
            "AMT_INCOME_TOTAL": pa.Column(
                float,
                [
                    _check("renda_positiva", lambda s: s > 0),
                    _check(
                        "renda_abaixo_do_teto", lambda s: s <= config.max_income_total
                    ),
                ],
                coerce=True,
            ),
            "AMT_CREDIT": pa.Column(
                float, _check("credito_positivo", lambda s: s > 0), coerce=True
            ),
            "DAYS_BIRTH": pa.Column(
                float,
                _check(
                    f"idade_acima_de_{config.min_age_years}_anos",
                    lambda s: s < -min_days,
                ),
                coerce=True,
            ),
        },
        strict=False,
    )


def _rule_name(column: object, check: object) -> str | None:
    """Nomeia a violação como `coluna.regra`; ignora o eco de `dtype(...)`."""
    name = str(check)
    if name.startswith("dtype("):
        return None
    name = {"not_nullable": "nao_nulo", "column_in_dataframe": "coluna_ausente"}.get(
        name, name
    )
    if name.startswith("coerce_dtype"):
        name = "tipo_invalido"
    return f"{column if isinstance(column, str) else SCHEMA_RULE}.{name}"


def _violations(error: pa.errors.SchemaErrors) -> dict[str, int]:
    """Conta linhas violadoras por regra a partir dos casos de falha."""
    cases = error.failure_cases
    names = [
        _rule_name(col, chk)
        for col, chk in zip(cases["column"], cases["check"], strict=True)
    ]
    counts = pd.Series([n for n in names if n is not None]).value_counts()
    return {str(rule): int(count) for rule, count in counts.items()}


def validate_batch(frame: pd.DataFrame, config: ExperimentConfig) -> ContractResult:
    """Valida o lote contra o contrato e devolve as violações por regra."""
    schema = build_schema(
        config.contract, config.data.id_column, config.data.target_column
    )
    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as error:
        return ContractResult(len(frame), _violations(error))
    return ContractResult(len(frame))


def enforce_contract(frame: pd.DataFrame, config: ExperimentConfig) -> ContractResult:
    """Valida o lote e bloqueia a ingestão levantando `DataContractError`."""
    result = validate_batch(frame, config)
    if not result.valid:
        raise DataContractError(result)
    return result
