# Credit Scoring Observability API

Projeto do Tech Challenge da Fase 4 de Machine Learning Engineering (FIAP).
Tema: **observabilidade contínua, detecção de drift, qualidade de dados e
governança em produção** para um modelo de credit scoring.

Um modelo de risco de crédito em produção pode degradar em silêncio quando a
economia muda (inflação, novos perfis de clientes). Este repositório constrói a
camada de sustentação e confiabilidade ao redor de um modelo base: contratos
de dados, simulação e detecção de drift, monitoramento e documentação de
governança/LGPD.

> **Status:** casca inicial. A infraestrutura (API, testes, lint, CI, Docker,
> Prometheus/Grafana) está funcional. As quatro etapas do enunciado estão
> mapeadas abaixo e serão implementadas uma a uma.

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
make lint
make test
make api              # http://localhost:8000/docs
```

Sem modelo treinado, `/health` responde `degraded` e `/predict` responde HTTP 503.
Isso é o comportamento esperado até a Etapa 1 gerar `models/artifacts/credit_model.joblib`.

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
| 1. Validação de dados e contratos | Validação de Dados | Estrutura pronta | `src/data/` (`contracts.py` e `loader.py`), `src/models/train.py` |
| 2. Simulação e detecção de drift | Data Drift | Estrutura pronta | `src/drift/` |
| 3. Observabilidade de pipelines e modelos | Monitoramento de Pipelines e Infraestrutura | Infra pronta; métricas de pipeline a implementar | `src/observability/`, `monitoring/`, `src/serving/metrics.py` |
| 4. Governança (LGPD) e causalidade | Privacy by Design e by Default | Seção abaixo a preencher | este README |

O checklist completo do enunciado está em
[docs/checklist_requisitos_tc4.md](docs/checklist_requisitos_tc4.md) e a
arquitetura em [docs/architecture.md](docs/architecture.md).

### Separação entre referência e produção

Boa prática obrigatória do enunciado: o **dataset de referência** (usado no
treino) e o **dataset de produção/análise** (onde o drift acontece) são
arquivos distintos, configurados em `config/config.yaml`
(`reference_file` e `production_file`). O de produção será gerado pela
simulação da Etapa 2 e nunca participa do treino.

### Contratos de dados (Etapa 1)

A regra é que lotes com erros sejam bloqueados antes de chegar ao modelo.
Regras previstas (mínimo de três, exigidas pelo enunciado):

- `TARGET` binário e sem nulos;
- identificador `SK_ID_CURR` sem duplicatas;
- idade do cliente (`DAYS_BIRTH`) implicando maioridade (> 18 anos);
- renda total (`AMT_INCOME_TOTAL`) positiva e não nula.

### Drift (Etapa 2)

Métricas previstas: PSI e Kolmogorov-Smirnov por feature, com relatório HTML
do Evidently AI em `reports/`. Limiares e variáveis alteradas serão
documentados quando a simulação for implementada.

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
src/data/                 leitura do bruto e contratos Pandera (Etapa 1)
src/models/               treino do modelo base (Etapa 1)
src/drift/                simulação e detecção de drift (Etapa 2)
src/observability/        logs e métricas do pipeline (Etapa 3)
src/serving/              API FastAPI, schemas, métricas Prometheus
monitoring/               Prometheus e dashboard Grafana versionado
docs/                     enunciado, arquitetura e checklist
reports/                  relatórios HTML (Evidently) — versionados como evidência
tests/                    testes de unidade, serving e arquitetura
```

## Troubleshooting

- `poetry: command not found`: use `python -m poetry` (já é o padrão do `Makefile`).
- `FileNotFoundError` em `make dataset`: confira o caminho do CSV em `data/raw/`.
- `/health` degradado: esperado sem modelo treinado; confira `MODEL_ARTIFACTS_PATH` no `.env`.
- portas ocupadas: ajuste `API_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT` ou `MLFLOW_PORT` no `.env`.
