.PHONY: first rebuild env install lint format test test-cov dataset train api compose-build compose-up compose-down

# `poetry` nem sempre está no PATH (ex.: Windows Store Python instala o script
# de entrada numa pasta de usuário que não entra no PATH automaticamente).
# `python -m poetry` não depende dessa resolução — só precisa que o pacote
# `poetry` esteja instalado no interpretador `python` corrente.
POETRY := python -m poetry

# ─── Primeiro uso (Docker) ──────────────────────────────────────────────────────

# Bootstrap da primeira execução via Docker, de ponta a ponta:
#   1. cria o .env (se faltar);
#   2. builda e treina o modelo (gera models/artifacts/ + metrics/);
#   3. sobe a stack em background: api + prometheus + grafana + mlflow.
# Pré-requisito: data/raw/application_train.csv no host.
# Ao fim: API em :8000, MLflow :5000, Prometheus :9090, Grafana :3000.
first:
	python -c "import os, shutil; shutil.copyfile('.env.example', '.env') if not os.path.exists('.env') else print('.env ja existe')"
	docker compose run --rm --build train
	docker compose up -d --build

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

# Treino do modelo base (Etapa 1): gera models/artifacts/credit_model.joblib.
train:
	$(POETRY) run python -m src.models.train

# ─── Serviço local ──────────────────────────────────────────────────────────────

API_PORT ?= 8000
api:
	$(POETRY) run uvicorn src.serving.api:app \
		--host 0.0.0.0 --port $(API_PORT) --reload

# ─── Docker: API + Prometheus + Grafana + MLflow ───────────────────────────────

compose-build:
	docker compose build

# Rebuild sem cache de todas as imagens, incluindo o serviço `train` (profile).
# Use após mudar Dockerfile, dependências ou o código do treino.
rebuild:
	docker compose --profile train build --no-cache

compose-up:
	docker compose up

compose-down:
	docker compose down
