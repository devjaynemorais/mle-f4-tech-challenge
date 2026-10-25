"""Gerador sintético de lotes no formato Home Credit, só para testes.

Os dados são aleatórios e não reproduzem nenhum cliente real.
"""

import numpy as np
import pandas as pd


def make_raw_frame(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Cria um lote bruto válido para o contrato, com sinal fraco no alvo."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(21, 65, n)
    ext = rng.beta(5, 3, (n, 3))
    income = rng.lognormal(12, 0.5, n)
    credit = rng.lognormal(13, 0.6, n)
    risk = 0.4 * (ext.mean(axis=1) < 0.55) + 0.3 * (age < 30)
    target = (rng.random(n) < 0.04 + 0.2 * risk).astype(int)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "TARGET": target,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": credit / rng.uniform(10, 30, n),
            "AMT_GOODS_PRICE": credit * rng.uniform(0.8, 1.0, n),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
            "DAYS_BIRTH": -(age * 365.25).round(),
            "DAYS_EMPLOYED": -rng.integers(100, 9000, n).astype(float),
            "EXT_SOURCE_1": ext[:, 0],
            "EXT_SOURCE_2": ext[:, 1],
            "EXT_SOURCE_3": ext[:, 2],
            "CODE_GENDER": rng.choice(["F", "M"], n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        }
    )
