# Arquitetura

## Visão geral

```text
data/raw/application_train.csv  (Home Credit, fora do Git)
        │
        ▼
src/data/loader.py ──► src/data/contracts.py      Etapa 1: contrato Pandera pendente
        │
        ├──► src/models/train.py (orquestrador)
        │       ├──► dataset.py ──► features.py ──► preprocessing.py
        │       ├──► candidates.py ──► evaluation.py
        │       └──► artifacts.py + tracking.py (MLflow opcional)
        │                         │
        │                         ▼
        │               models/artifacts/credit_model.joblib
        │                                   │
        │                                   ▼
        │                         src/serving/ (FastAPI)  ──► /predict, /health, /metrics
        │                                   │                     │
        ▼                                   ▼                     ▼
data/processed/reference.csv     predições de produção    Prometheus ──► Grafana
data/processed/production.csv   (Etapa 2)                 (Etapa 3: + logs do pipeline)
     (pendentes)
        │
        ▼
src/drift/detect.py ──► Evidently (PSI/KS) ──► reports/*.html
```

## Decisões

- **Referência × produção separadas.** O treino atual lê o CSV bruto por
  `src/data/loader.py`. A geração do arquivo de referência validado e a
  simulação do arquivo de produção ainda são pendências; o treino não lê o
  arquivo de produção. Quando essa etapa for implementada, o treino deverá
  consumir somente a referência validada.
- **Treino modular.** `src/models/train.py` coordena o fluxo. `dataset.py`
  prepara a amostra e o split; `features.py` e `preprocessing.py` compõem o
  pipeline serializado; `candidates.py` e `evaluation.py` treinam e selecionam
  o modelo; `artifacts.py` e `tracking.py` registram os resultados.
- **Configuração tipada.** Parâmetros do experimento ficam em
  `config/config.yaml` e são validados com Pydantic. Variáveis de ambiente
  (`.env`) ficam restritas a runtime: caminhos, portas e senhas locais.
- **API degradada, não quebrada.** A API sobe sem modelo treinado. `/health`
  responde `degraded` e `/predict` responde 503. Isso permite checar a
  infraestrutura antes do primeiro treino.
- **Sem dados pessoais nos logs.** Falhas de inferência registram só o tipo da
  exceção. Features de clientes nunca são logadas (coberto por teste).
- **Métricas com nomes estáveis.** `http_requests_total`,
  `http_request_duration_seconds` e `credit_predictions_total{predicted_label}`
  alimentam o dashboard versionado em `monitoring/grafana/dashboards/`.
- **Métricas de modelo.** Além das de HTTP, a API expõe:
  `credit_default_probability` (histograma da probabilidade prevista, para
  acompanhar drift de score), `credit_prediction_duration_seconds` (latência da
  inferência, sem overhead HTTP), `credit_prediction_errors_total{error_type}`
  (falhas de inferência) e `credit_model_loaded` (gauge 0/1 do estado do
  modelo). Todas em `src/serving/metrics.py` e coletadas pelo mesmo scrape.

## Pendências desta casca

- Schema de features da API: hoje aceita um dicionário livre; a Etapa 1 deve
  restringi-lo ao contrato de dados.
- Persistência de métricas de drift e de logs de pipeline (Etapa 3): definir se
  vão para o Prometheus, para o MLflow ou para um arquivo consumido pelo Grafana.
- MLflow registra o treino de forma opcional; falta definir o rastreamento das
  execuções de contratos e drift da Etapa 3.
