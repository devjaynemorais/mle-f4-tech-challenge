# Credit Scoring Observability API

Projeto do Tech Challenge da Fase 4 de Machine Learning Engineering (FIAP).
Tema: **observabilidade contínua, detecção de drift, qualidade de dados e
governança em produção** para um modelo de credit scoring.

Um modelo de risco de crédito em produção pode degradar em silêncio quando a
economia muda (inflação, novos perfis de clientes). Este repositório constrói a
camada de sustentação e confiabilidade ao redor de um modelo base: contratos
de dados, simulação e detecção de drift, monitoramento e documentação de
governança/LGPD.

> **Status:** contrato de dados, treino base, simulação e detecção de drift,
> predições em lote e observabilidade de pipeline e modelo (métricas, logs,
> alertas e dashboards) implementados. A governança (LGPD) e o vídeo STAR
> (Etapa 4) continuam pendentes.

## Dataset

Base: **Home Credit Default Risk** (Kaggle). Usamos o arquivo
`application_train.csv` (~307 mil clientes, ~120 colunas, alvo binário
`TARGET`: `1` = dificuldade de pagamento, `0` = adimplente).

O CSV não é versionado (contém dados pessoais e é grande). Para obtê-lo:

```bash
kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw
```

ou baixe manualmente em <https://www.kaggle.com/c/home-credit-default-risk/data>
e coloque o arquivo em `data/raw/application_train.csv`.

## Início rápido

```bash
make install          # Poetry + dependências de desenvolvimento
cp .env.example .env
make dataset          # resumo do CSV bruto (valida colunas e taxa de default)
make train            # gera os CSVs processados, treina e grava o pipeline
make drift            # gera os períodos estável e driftado a partir de prod_data.csv
make report           # compara os períodos e gera HTML Evidently + resumo CSV
make lint
make test
make api              # http://localhost:8000/docs
```

Sem modelo treinado, `/health` responde `degraded` e `/predict` responde HTTP 503.
Execute `make train` após disponibilizar o CSV bruto. O comando gera
`data/processed/ref_data.csv`, `data/processed/prod_data.csv` e
`models/artifacts/credit_model.joblib` (XGBoost) e
`models/artifacts/logistic_baseline.joblib` (regressão logística). As métricas
de validação cruzada são gravadas em `metrics/training_metrics.json` e em um
JSON específico para cada modelo.
Depois, `make drift` gera `prod_periodo1_estavel.csv`,
`prod_periodo2_drift.csv` e `data_drift_simulation_manifest.json` em
`data/processed/`. Em seguida, `make report` gera o relatório de drift.

Com Docker (API + Prometheus + Grafana + MLflow):

```bash
make compose-build
make compose-up
```

| Serviço    | URL                     |
|------------|-------------------------|
| API        | http://localhost:8000   |
| Prometheus | http://localhost:9090   |
| Grafana    | http://localhost:3000   |
| MLflow     | http://localhost:5000   |

## API

| Método | Rota       | Contrato |
|--------|------------|----------|
| GET    | `/`        | nome e versão do serviço |
| GET    | `/health`  | `ok` com backend, ou `degraded` sem modelo |
| POST   | `/predict` | recebe `{"features": {...}}` de um cliente e retorna `default_probability`, `prediction` (`default` / `non_default`) e `backend` |
| GET    | `/metrics` | métricas Prometheus |
| GET    | `/docs`    | Swagger UI |

As chaves de `features` são as colunas do `application_train` (sem `TARGET`).
O contrato exato de entrada será definido pelo schema de dados da Etapa 1.

## Mapa de etapas

| Etapa | Disciplina | Status | Onde fica |
|-------|------------|--------|-----------|
| 1. Validação de dados e contratos | Validação de Dados | Contrato Pandera e treino base prontos | `src/data/`, `src/models/` |
| 2. Simulação e detecção de drift | Data Drift | Simulação, relatórios e predições por lote prontos | `src/drift/`, `reports/` |
| 3. Observabilidade de pipelines e modelos | Monitoramento de Pipelines e Infraestrutura | Métricas, logs, alertas e dashboards prontos | `src/observability/`, `monitoring/`, `src/serving/metrics.py` |
| 4. Governança (LGPD) e causalidade | Privacy by Design e by Default | Seção abaixo a preencher | este README |

O checklist completo do enunciado está em
[docs/checklist_requisitos_tc4.md](docs/checklist_requisitos_tc4.md) e a
arquitetura em [docs/architecture.md](docs/architecture.md).

### Separação entre referência e produção

`make train` divide `application_train.csv` de forma estratificada: 80% formam
`ref_data.csv` (referência com `TARGET`) e 20% formam `prod_data.csv`
(produção-base simulada, com `TARGET` real do Home Credit). Ambos recebem limpeza e engenharia
de atributos determinísticas e ficam fora do Git. Regressão Logística e XGBoost
são comparados por validação cruzada estratificada apenas nos 80%; depois, cada
pipeline é ajustado novamente com toda a referência. O XGBoost é o artefato
carregado pela API e a regressão logística permanece como baseline.
Imputação, escala e codificação são ajustadas separadamente em cada fold. A
produção-base é reservada para comparação de drift e não representa dados reais
de produção. `SK_ID_CURR` fica nos CSVs locais
para conferir a separação das amostras, mas não entra no modelo nem deve entrar
nas estatísticas de drift.

### Contratos de dados (Etapa 1)

`src/data/contracts.py` define o contrato Pandera aplicado a cada lote antes
do modelo. Um lote que viola qualquer regra é **barrado** (`DataContractError`):
nada é treinado nem pontuado, e o evento fica registrado.

| Regra | Coluna |
|-------|--------|
| Identificador obrigatório e sem duplicatas | `SK_ID_CURR` |
| Alvo obrigatório e binário (0 ou 1) | `TARGET` |
| Renda obrigatória, positiva e abaixo de um teto (`contract.max_income_total`) | `AMT_INCOME_TOTAL` |
| Crédito obrigatório e positivo | `AMT_CREDIT` |
| Idade acima do mínimo (`contract.min_age_years`, padrão 18) | `DAYS_BIRTH` |

O resultado guarda só nomes de regras e contagens de linhas; valores e
identificadores nunca aparecem em mensagens, eventos ou métricas. O contrato é
aplicado ao CSV bruto no treino e a cada período de produção na predição em
lote. Para ver um lote válido passar e um com erros ser barrado, sem precisar
do CSV: `make contract-demo`.

### Drift (Etapa 2)

`make drift` amostra `N` linhas de `prod_data.csv` e divide a amostra em dois
períodos com tamanhos quase iguais. O primeiro preserva features e `TARGET`
originais. No segundo, o módulo `src/drift/data_drift_simulation.py` desloca
idade, crédito, preço do bem, anuidade e escores externos; depois recalcula as
features derivadas. `CODE_GENDER` permanece como na origem. As saídas têm o
mesmo esquema processado, antes da escala e codificação do modelo.

Por padrão, a idade mediana driftada é 33,1 anos, crédito ×1,20, preço do bem
×1,05, anuidade ×1,15 e escores externos ×0,85. O `TARGET` do segundo período
é **sintético**: os `round(0,12 × n_drift)` maiores riscos recebem 1, segundo a
média dos percentis de idade menor, `EXT_SOURCES_MEAN` menor e
`CREDIT_GOODS_PERCENT` maior. Ausências contam como percentil neutro.
`CODE_GENDER_M` é uma coluna gerada pelo codificador do modelo e a categoria
`CODE_GENDER` não é alterada. No dicionário Home Credit, `AMT_GOODS_PRICE` é o
preço do bem associado ao empréstimo de consumo, quando aplicável; não é o
patrimônio disponível do cliente. A direção de risco atribuída aos escores
externos é uma premissa da simulação. Isso
simula uma mudança na relação entre features e alvo; métricas calculadas contra
esse `TARGET` descrevem somente essa regra sintética, não desempenho real em
produção. O manifesto registra a origem dos rótulos e os parâmetros.

Use `make drift DRIFT_ARGS="--n 50000 --target-rate 0.12 --seed 42"` para
ajustar o volume e a taxa desejada (entre 0,11 e 0,13). A CLI também aceita
`--input`, `--output-dir`, `--median-age`, `--credit-factor`, `--goods-factor`,
`--annuity-factor` e `--external-score-factor`. Sem `--n`, usa o tamanho de
`prod_data.csv`; se `N` exceder a origem, amostra com reposição e gera IDs
sintéticos únicos em uma faixa iniciada em 1 bilhão, separada dos IDs Home Credit.
A taxa do período estável segue os rótulos originais e
pode variar com a amostra.

Execute `make report` depois de `make drift`. O período 1 estável é a
referência e o período 2 driftado é o atual. O script percorre as features
dos dois CSVs e exclui `SK_ID_CURR` e `TARGET`: o ID não é uma feature e o alvo
do período 2 é sintético. Os artefatos ficam em `reports/`:

- `data_drift_report.html`: visão geral gerada pelo preset de drift do Evidently;
- `data_drift_summary.csv`: resultados por feature, com KS e PSI para colunas
  numéricas e qui-quadrado para categóricas, estatística, limiar e decisão.

`make report` também gera a comparação de sanidade **referência de treino ×
período 1 estável** (`reports/data_drift_reference_vs_stable.*`), que não deve
indicar drift e serve de linha de base para calibrar os alertas.

O resumo CSV não inclui linhas de clientes, IDs ou valores individuais. Essa
comparação identifica mudança nas distribuições de entrada (`P(X)`); não
avalia desempenho nem comprova degradação do modelo. O `TARGET` sintético não
entra na análise. Caminhos e parâmetros padrão podem ser ajustados em
`config/config.yaml`.

### Predições em lote (Etapa 2)

`make predict` (depois de `make train` e `make drift`) pontua o período estável
e o driftado com o mesmo modelo da API, grava as predições em
`data/processed/` (fora do Git) e um resumo agregado em
`metrics/batch_scoring_summary.json`. Só o período estável publica AUC: o alvo
do período driftado é sintético e não é tratado como desempenho real.

## Observabilidade de pipelines e modelos (Etapa 3)

Decisão de arquitetura: [docs/adr/0001](docs/adr/0001-observabilidade-pipeline-e-modelo.md).
Catálogo de métricas, alertas e operação: [docs/metricas_saude.md](docs/metricas_saude.md).

```text
etapas do pipeline ──► eventos JSONL (logs/) + Pushgateway ──► Prometheus ──► Grafana + alertas
                   └──► MLflow (uma run por etapa, com o HTML do Evidently)
API FastAPI ──────────► /metrics ──► Prometheus
```

- **Logs centralizados:** cada etapa (ingestão, contrato, transformação,
  treino, predição, drift) emite eventos estruturados sem dados pessoais, com
  um `run_id` por execução.
- **Métricas de saúde:** status, duração, volume, lotes barrados, drift por
  feature, score do modelo e alertas, além das métricas da API.
- **Alertas:** regras Prometheus versionadas em `monitoring/prometheus/alerts.yml`
  (falha de etapa, lote barrado, drift moderado e severo, modelo fora do ar,
  erro e latência da API).
- **Dashboards:** "Credit Pipeline — Saúde, Drift e Alertas" e "Credit Scoring
  API — Overview" no Grafana (`http://localhost:3000`).

```bash
docker compose up -d --build     # API, Prometheus, Grafana, MLflow, Pushgateway
make compose-pipeline            # treino + drift + predição + relatórios nos containers
make demo-observability          # cenários: estável (sem alerta), driftado, lote inválido
```

Sem Docker, rode `PUSHGATEWAY_URL=http://localhost:9091 make pipeline`
(executa tudo sob um único `run_id`). Sem `PUSHGATEWAY_URL` ou
`MLFLOW_TRACKING_URI`, o pipeline funciona normalmente e só grava os eventos.

## Governança e LGPD (Etapa 4)

> Seção a preencher. Cada subseção abaixo lista as perguntas que o enunciado
> exige; o texto definitivo será escrito junto com a implementação.

### Tratamento de dados sensíveis (PII)

- Quais campos do Home Credit são PII direta ou indireta (identificador,
  datas de nascimento e emprego, valores de renda e crédito, contatos e
  documentos quando presentes)?
- Como são tratados: minimização, pseudonimização do `SK_ID_CURR`, e
  exclusão de identificadores dos logs e das respostas da API?
- Como a API evita registrar features de clientes? (Já há teste garantindo
  que o valor de entrada não aparece nos logs de erro.)

### Base legal para a decisão de crédito

- Qual a base legal da LGPD para o tratamento (execução de contrato ou
  legítimo interesse, com as ressalvas da lei)?
- Qual o direito do titular à revisão de decisões automatizadas (art. 20 da
  LGPD) e como o projeto permite isso?

### Plano de retenção

- Por quanto tempo o dataset bruto, os dados de produção simulados, as
  predições e os logs são mantidos, e quando são eliminados?

### Viés e mitigação

- Quais variáveis podem atuar como proxy de características protegidas?
- Como o drift e as métricas de desempenho são segmentados para detectar
  disparidade entre grupos?

### Causalidade

- Análise causal do problema: por que a degradação ocorre, quais variáveis
  são causas e quais são apenas correlações, e qual intervenção o time pode
  fazer (retreinar, recalibrar limiar, revisar política)?

## Vídeo STAR

_Link a adicionar após a gravação (5 minutos, método STAR)._

## Qualidade e CI

```bash
make lint
make test
make test-cov
```

O workflow `.github/workflows/ci.yml` executa lint, testes com cobertura e o
build da imagem `api` em pushes e pull requests para `main`. Um teste
arquitetural (`tests/test_source_function_length.py`) rejeita funções de
`src/` com mais de 20 linhas lógicas.

## Estrutura

```text
config/config.yaml        configuração tipada do experimento
data/raw/                 CSV Home Credit (fora do Git)
data/processed/           referência e produção (Etapas 1 e 2)
src/data/                 leitura do bruto, contrato Pandera e porta de ingestão (Etapa 1)
src/models/               orquestração, features, avaliação e artefatos do treino
src/drift/                simulação e detecção de drift (Etapa 2)
src/observability/        eventos, métricas, alertas e runs MLflow do pipeline (Etapa 3)
src/serving/              API FastAPI, schemas, métricas Prometheus
monitoring/               Prometheus (regras de alerta) e dashboards Grafana versionados
docs/                     enunciado, arquitetura e checklist
reports/                  relatórios HTML (Evidently) — versionados como evidência
logs/                     eventos do pipeline (JSONL, fora do Git)
tests/                    testes de unidade, serving e arquitetura
```

## Troubleshooting

- `poetry: command not found`: use `python -m poetry` (já é o padrão do `Makefile`).
- `FileNotFoundError` em `make dataset`: confira o caminho do CSV em `data/raw/`.
- `/health` degradado: esperado sem modelo treinado; confira `MODEL_ARTIFACTS_PATH` no `.env`.
- `Can't get attribute 'reindex_features' on __main__` ao carregar um artefato
  anterior à modularização: execute `make train` para gerar um novo pipeline.
- portas ocupadas: ajuste `API_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT` ou `MLFLOW_PORT` no `.env`.
