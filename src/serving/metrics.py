"""Métricas Prometheus de requisições, latência e predições de risco."""

from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas pela API",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Duração das requisições HTTP em segundos",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "credit_predictions_total",
    "Predições de risco de crédito retornadas pela API",
    ["predicted_label"],
)


class PrometheusMiddleware:
    """Mede contagem e latência de cada requisição HTTP."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Encaminha a requisição e registra status final e duração."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        started_at = time.perf_counter()
        status = {"code": 500}

        async def capture_status(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            _record_request(request, status["code"], started_at)


def _record_request(request: Request, status_code: int, started_at: float) -> None:
    """Registra contador e latência após uma requisição concluída."""
    path = request.url.path
    REQUEST_COUNT.labels(
        method=request.method, path=path, status_code=status_code
    ).inc()
    duration = time.perf_counter() - started_at
    REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration)


def metrics_response() -> Response:
    """Devolve as métricas no formato de texto do Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
