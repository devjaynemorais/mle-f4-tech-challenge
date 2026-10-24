# Arquitetura

## Visão geral

```text
data/raw/application_train.csv  (Home Credit, fora do Git)
        │
        ▼
src/data/loader.py ──► src/data/contracts.py      Etapa 1: contrato Pandera pendente
        │
        ▼
src/data/processed.py  (split estratificado 80/20)
        ├──► data/processed/prod_data.csv (TARGET real) ────┐
        └──► data/processed/ref_data.csv                    │
                       │                                    │
                       ▼                                    │
             src/models/train.py                             │
              ├── dataset.py                                 │
              ├── features.py + preprocessing.py por fold    │
              ├── candidates.py + evaluation.py (CV)         │
              └── artifacts.py + tracking.py                 │
                       │                                    │
                       ▼                                    │
             models/artifacts/credit_model.joblib            │
             models/artifacts/logistic_baseline.joblib       │
             metrics/*_metrics.json                          │
                       │                                    │
                       ▼                                    │
             src/serving/ ──► /predict, /health, /metrics     │
                       │                                    │
                       ▼                                    ▼
                Prometheus ──► Grafana        src/drift/data_drift_simulation.py
                                            ├──► prod_periodo1_estavel.csv
                                            ├──► prod_periodo2_drift.csv
                                            └──► data_drift_simulation_manifest.json
```

## Decisões

- **Referência × produção separadas.** O treino lê o CSV bruto, divide 80/20
  com estratificação pelo alvo e salva referência e produção estável simulada.
  A escolha dos candidatos usa validação cruzada apenas na referência; o
  arquivo de produção não entra no treino. A simulação posterior divide
  `prod_data.csv` em período estável com `TARGET` real e período driftado com
  `TARGET` sintético condicionado às features. O contrato Pandera segue pendente.
- **Treino modular.** `src/models/train.py` coordena o fluxo. `dataset.py`
  remove ID, alvo e derivadas previamente salvas; `features.py` e
  `preprocessing.py` compõem os pipelines serializados; `candidates.py` e
  `evaluation.py` comparam XGBoost e regressão logística por validação cruzada.
  Os dois são ajustados na referência completa e persistidos; a API serve o
  XGBoost, enquanto a regressão logística permanece como baseline. `artifacts.py`
  e `tracking.py` registram os pipelines e as métricas de cada modelo.
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
