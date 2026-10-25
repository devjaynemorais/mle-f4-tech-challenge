# Stage 1: builder — instala as dependências de produção com Poetry
FROM python:3.11-slim AS builder

WORKDIR /app

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root && rm -rf "${POETRY_CACHE_DIR}"


# Stage 2: api — serviço FastAPI com predição e métricas Prometheus
FROM python:3.11-slim AS api

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY config/ config/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]


# Stage 3: train — imagem de treino = api + cliente MLflow para tracking.
# Mantém a imagem de serviço enxuta (sem mlflow) e evita mexer no poetry.lock.
FROM api AS train

ENV GIT_PYTHON_REFRESH=quiet

RUN pip install --no-cache-dir "mlflow>=2.16,<3"

CMD ["python", "-m", "src.models.train"]
