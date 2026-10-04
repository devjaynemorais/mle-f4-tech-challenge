.PHONY: env install lint format test test-cov dataset api compose-build compose-up compose-down

# `poetry` nem sempre está no PATH (ex.: Windows Store Python instala o script
# de entrada numa pasta de usuário que não entra no PATH automaticamente).
# `python -m poetry` não depende dessa resolução — só precisa que o pacote
# `poetry` esteja instalado no interpretador `python` corrente.
POETRY := python -m poetry

# ─── Ambiente ─────────────────────────────────────────────────────────────────

env:
	python -m ensurepip --upgrade
	python -m pip install --upgrade pip --quiet
	python -m pip install poetry==1.8.3 --quiet
	$(POETRY) install --with dev

install: env

# ─── Qualidade de Código ──────────────────────────────────────────────────────

lint:
	$(POETRY) run ruff check src/ tests/
	$(POETRY) run ruff format --check src/ tests/

format:
	$(POETRY) run ruff check --fix src/ tests/
	$(POETRY) run ruff format src/ tests/

# ─── Testes ───────────────────────────────────────────────────────────────────

test:
	$(POETRY) run pytest tests/ -v

test-cov:
	$(POETRY) run pytest tests/ --cov=src --cov-report=html

# ─── Dados ────────────────────────────────────────────────────────────────────

# Resumo do CSV bruto Home Credit em data/raw/ (o arquivo fica fora do Git).
dataset:
	$(POETRY) run python -m src.data.loader

# ─── Serviço local ──────────────────────────────────────────────────────────────

API_PORT ?= 8000
api:
	$(POETRY) run uvicorn src.serving.api:app \
		--host 0.0.0.0 --port $(API_PORT) --reload

# ─── Docker: API + Prometheus + Grafana + MLflow ───────────────────────────────

compose-build:
	docker compose build

compose-up:
	docker compose up

compose-down:
	docker compose down
