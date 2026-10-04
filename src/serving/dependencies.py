"""Dependências FastAPI apoiadas no estado local da aplicação."""

from typing import cast

from fastapi import HTTPException, Request

from src.serving.model_loader import CreditPredictor


def get_predictor(request: Request) -> CreditPredictor:
    """Devolve o predictor da aplicação ou responde 503 quando não há modelo."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded; train the model (Etapa 1) and retry.",
        )
    return cast(CreditPredictor, predictor)
