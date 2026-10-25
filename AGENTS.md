# AGENTS.md

Guia operacional para agentes trabalhando neste repositório.

## Objetivo

Tech Challenge da Fase 4 (FIAP MLE): camada de sustentação e confiabilidade de
um modelo de credit scoring em produção. Cobre contratos de dados, simulação e
detecção de drift, observabilidade e governança (LGPD). Base de dados: Home
Credit Default Risk.

O enunciado completo está em `docs/MLET - Tech Challenge - Fase 4.pdf`. O
mapeamento de requisitos está em `docs/checklist_requisitos_tc4.md`.

## Mapa rápido

- `README.md`: visão geral, execução e seção de governança (LGPD) a preencher.
- `config/config.yaml`: colunas, caminhos, limiar e artefatos do experimento.
- `.env.example`: configuração de runtime (paths, portas, senhas locais).
- `src/data/`: leitura do CSV bruto, contrato Pandera (`contracts.py`) e porta de ingestão (`ingestion.py`).
- `src/models/`: treino do modelo base (Etapa 1).
- `src/drift/`: simulação dos períodos, detecção estatística (`detect.py`), relatórios Evidently e predição em lote (`predict_batch.py`).
- `src/observability/`: eventos estruturados, métricas dos jobs (Pushgateway), alertas e runs MLflow (Etapa 3).
- `docs/adr/`: decisões de arquitetura; `docs/metricas_saude.md` é o catálogo de métricas e alertas.
- `src/serving/`: API FastAPI, schemas, métricas Prometheus e carregamento do modelo.
- `monitoring/`: Prometheus (regras de alerta) e dashboards Grafana versionados.
- `tests/`: unidade, serving e teste arquitetural.
- `Dockerfile`: stages `builder` e `api`.
- `docker-compose.yml`: API, Prometheus, Grafana, MLflow e Pushgateway; serviços `train` e `drift` sob demanda.

## Contratos que devem ser preservados

### Dados

- O CSV bruto é `data/raw/application_train.csv`. Nunca versione esse arquivo:
  contém dados pessoais e é grande.
- Colunas obrigatórias: `SK_ID_CURR` (identificador) e `TARGET` (alvo binário:
  1 = dificuldade de pagamento, 0 = adimplente). Os nomes ficam em
  `config/config.yaml`.
- `make train` grava `ref_data.csv` e `prod_data.csv` em `data/processed/`, ambos
  com `TARGET` real. O treino usa apenas a referência para ajuste e seleção.
- `make drift` lê `prod_data.csv` e grava `prod_periodo1_estavel.csv` com
  `TARGET` original, `prod_periodo2_drift.csv` com `TARGET` sintético dependente
  das features e `data_drift_simulation_manifest.json`. Os CSVs têm o mesmo
  esquema; os IDs sintéticos usam faixa iniciada em 1 bilhão para não coincidir
  com a referência Home Credit. As saídas permanecem fora do Git.
- O `TARGET` sintético serve apenas para avaliação da hipótese simulada; não
  representa inadimplência observada nem desempenho real de produção.
- Não exponha identificadores, valores de renda ou crédito em logs, mensagens de
  erro ou exemplos commitados.

### Modelo e API

- O pipeline servido é `models/artifacts/credit_model.joblib`, nome definido em
  `config/config.yaml`.
- Sem modelo carregado: `/health` retorna `{"status": "degraded", "model_loaded": false}`
  e `/predict` retorna HTTP 503. Não quebre esse comportamento.
- `/predict` recebe `{"features": {...}}` e retorna `default_probability`,
  `prediction` (`default` / `non_default`) e `backend`.
- Falhas de inferência retornam HTTP 500 e registram só o tipo da exceção.

### Observabilidade

Métricas Prometheus que devem continuar existindo:

- `http_requests_total{method,path,status_code}`;
- `http_request_duration_seconds{method,path}`;
- `credit_predictions_total{predicted_label}`.

O dashboard `monitoring/grafana/dashboards/credit-api-overview.json` usa esses
nomes. Se mudar uma métrica, atualize o dashboard e os testes juntos.

Métricas dos jobs em lote (Pushgateway) estão em `docs/metricas_saude.md` e
são gauges do último estado. Ao criar ou renomear uma, atualize esse catálogo,
`monitoring/prometheus/alerts.yml` e `pipeline-health.json`: um teste confere
a coerência entre os três.

Eventos do pipeline (`logs/pipeline_events.jsonl`) só aceitam as chaves da
lista de permissão de `src/observability/events.py`. Nunca inclua
identificadores, rendas, créditos, linhas de clientes nem mensagens de exceção.

### Configuração

- Parâmetros de experimento (colunas, limiar, artefatos) vão em
  `config/config.yaml`, não em variáveis de ambiente.
- Variáveis de ambiente ficam para runtime (`MODEL_ARTIFACTS_PATH`, portas,
  senhas locais do Grafana).

## Comandos de trabalho

```bash
make install
make dataset     # resumo do CSV bruto
make train       # referência, produção-base e artefato do modelo
make drift       # períodos de produção estável e driftada
make lint
make test
make test-cov
make api
make compose-build
make compose-up
make compose-down
make predict          # predições em lote dos períodos de produção
make pipeline         # treino, drift, predição e relatórios sob um run_id
make contract-demo    # lote válido passa, lote com erros é barrado
make demo-observability   # cenários: estável, driftado, lote inválido
make compose-pipeline # o pipeline nos containers (Pushgateway e MLflow)
```

No Windows, se `poetry` não estiver no PATH, use o padrão do `Makefile`
(`python -m poetry`) ou o interpretador `.venv/Scripts/python.exe`.

## Validação antes de finalizar mudanças

- Documentação apenas: revise links, caminhos e aderência ao enunciado.
- Código Python: `make lint` e `make test`.
- Simulação: depois dos testes, rode `make train` e então `make drift`; confira
  esquema, contagens, IDs, distribuições e prevalências registradas no manifesto.
- API: rode testes e confira `/health`, `/predict`, `/metrics` e `/docs`.
- Docker: `make compose-build`; quando possível, `make compose-up`.

Se não for possível rodar uma validação, registre o motivo no resumo final ou
na documentação alterada.

## Regras de desenvolvimento

- Leia os arquivos atuais antes de editar.
- Prefira mudanças pequenas e coerentes com a arquitetura existente.
- Não introduza frameworks novos se o stack atual resolve o requisito.
- Mantenha funções de `src/` com no máximo 20 linhas lógicas (teste arquitetural).
- Ruff com complexidade McCabe máxima 10.
- Não versionar dados, modelos grandes ou segredos. `.env` fica local; mantenha
  `.env.example` sem credenciais reais.

## Commits e entrega

Histórico semântico e organizado. Formato:

```text
tipo(escopo): resumo curto em português

Corpo opcional explicando o porquê.
```

Tipos usados: `feat`, `fix`, `docs`, `test`, `build`, `ci`, `refactor`, `chore`.

## Limites de comunicação

- Não apresente o modelo como decisão automática de crédito. Ele é apoio à
  análise e precisa de revisão humana.
- Claims sobre equidade, causalidade ou conformidade LGPD só entram no README
  depois de análise e evidência nas etapas correspondentes.
