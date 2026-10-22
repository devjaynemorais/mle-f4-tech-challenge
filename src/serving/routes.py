"""Rotas públicas: metadados, saúde, predição e métricas."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from src.serving.dependencies import get_predictor
from src.serving.metrics import (
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_PROBABILITY,
    metrics_response,
)
from src.serving.model_loader import CreditPredictor
from src.serving.schemas import (
    CreditRequest,
    CreditResponse,
    HealthResponse,
    ServiceMetadata,
)
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=ServiceMetadata)
def root() -> dict[str, str]:
    """Devolve o nome e a versão configurados do serviço."""
    project = load_config().project
    return {"service": project.name, "version": project.version}


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
def health(request: Request) -> dict[str, str | bool]:
    """Informa o backend carregado ou o estado degradado sem modelo."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return {"status": "degraded", "model_loaded": False}
    return {"status": "ok", "model_loaded": True, "backend": predictor.backend}


@router.post("/predict", response_model=CreditResponse)
def predict(
    request: CreditRequest,
    predictor: Annotated[CreditPredictor, Depends(get_predictor)],
) -> dict[str, object]:
    """Calcula o risco de default de um cliente sem registrar seus dados."""
    started_at = time.perf_counter()
    try:
        probability, label = predictor.predict(request.features)
    except Exception as error:  # noqa: BLE001 - mapeia falhas do backend no contrato
        PREDICTION_ERRORS.labels(error_type=type(error).__name__).inc()
        logger.error("Falha inesperada na inferência (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Model inference failed") from None
    PREDICTION_LATENCY.observe(time.perf_counter() - started_at)
    PREDICTION_PROBABILITY.observe(probability)
    PREDICTION_COUNT.labels(predicted_label=label).inc()
    return {
        "default_probability": probability,
        "prediction": label,
        "backend": predictor.backend,
    }


@router.get("/metrics")
def metrics() -> Response:
    """Expõe as métricas Prometheus no formato texto."""
    return metrics_response()
