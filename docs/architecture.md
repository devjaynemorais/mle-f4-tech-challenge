# Arquitetura

## Visão geral

```text
data/raw/application_train.csv  (Home Credit, fora do Git)
        │
        ▼
src/data/loader.py ──► src/data/contracts.py      Etapa 1: contrato Pandera bloqueia lotes ruins
        │
        ├──► src/models/train.py ──► models/artifacts/credit_model.joblib
        │                                   │
        │                                   ▼
        │                         src/serving/ (FastAPI)  ──► /predict, /health, /metrics
        │                                   │                     │
        ▼                                   ▼                     ▼
data/processed/reference.csv     predições de produção    Prometheus ──► Grafana
data/processed/production.csv   (Etapa 2)                 (Etapa 3: + logs do pipeline)
        │
        ▼
src/drift/detect.py ──► Evidently (PSI/KS) ──► reports/*.html
```

## Decisões

- **Referência × produção separadas.** O treino só enxerga o arquivo de
  referência. O drift é medido contra o arquivo de produção simulado, que nunca
  entra no treino. Isso evita vazamento entre as duas visões.
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

## Pendências desta casca

- Schema de features da API: hoje aceita um dicionário livre; a Etapa 1 deve
  restringi-lo ao contrato de dados.
- Persistência de métricas de drift e de logs de pipeline (Etapa 3): definir se
  vão para o Prometheus, para o MLflow ou para um arquivo consumido pelo Grafana.
- Escolha de MLflow ou dashboard próprio para rastreamento (Etapa 3).
