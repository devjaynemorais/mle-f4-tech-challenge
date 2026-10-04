"""Fábrica da aplicação FastAPI e entrypoint estável ``src.serving.api:app``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from src.serving.metrics import PrometheusMiddleware
from src.serving.model_loader import CreditPredictor, load_predictor
from src.serving.routes import router
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)
PredictorLoader = Callable[[], CreditPredictor]


def _lifespan(
    predictor_loader: PredictorLoader,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Cria um lifespan vinculado a um carregador de predictor."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.predictor = None
        try:
            application.state.predictor = predictor_loader()
            logger.info(
                "Modelo de crédito carregado (backend=%s)",
                application.state.predictor.backend,
            )
        except Exception:  # noqa: BLE001 - o startup permanece degradado de propósito
            logger.exception("Falha ao carregar o modelo; API em modo degradado")
        yield
        application.state.predictor = None

    return lifespan


def create_app(predictor_loader: PredictorLoader = load_predictor) -> FastAPI:
    """Monta uma instância FastAPI com estado, rotas e middleware isolados."""
    project = load_config().project
    application = FastAPI(
        title=project.name,
        description=(
            "Estima a probabilidade de default de crédito como apoio à decisão. "
            "Não substitui a análise de crédito humana nem a revisão regulatória."
        ),
        version=project.version,
        lifespan=_lifespan(predictor_loader),
    )
    application.add_middleware(PrometheusMiddleware)
    application.include_router(router)
    return application


app = create_app()
