"""Contratos Pydantic de requisição e resposta da API pública."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FeatureValue = float | int | str | None


class CreditRequest(BaseModel):
    """Features de um cliente, no formato das colunas do Home Credit."""

    features: dict[str, FeatureValue] = Field(
        min_length=1,
        description=(
            "Colunas do application_train (sem TARGET) para um único cliente. "
            "Chaves ausentes ou nulas devem ser tratadas pelo pipeline."
        ),
    )


class CreditResponse(BaseModel):
    """Probabilidade de default, rótulo decidido pelo limiar e backend usado."""

    default_probability: float = Field(ge=0, le=1)
    prediction: Literal["default", "non_default"]
    backend: str


class HealthResponse(BaseModel):
    """Estado de carregamento do modelo exposto pelo health check."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    backend: str | None = None


class ServiceMetadata(BaseModel):
    """Metadados estáveis do serviço, retornados pela rota raiz."""

    service: str
    version: str
