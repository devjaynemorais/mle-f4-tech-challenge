# Etapa 1 — Treino do modelo e tracking

Este documento registra as decisões do treino base, iniciadas na branch
`feat/config-train-updates`, e a justificativa de cada decisão. O objetivo da
etapa foi sair de uma casca (stub) para um treino reprodutível que gera o
artefato servido pela API, com feature engineering de domínio e rastreamento
de experimentos no MLflow.

## Resumo das alterações

| Arquivo | Natureza da mudança |
|---|---|
| `src/models/train.py` | Orquestração do treino e ponto de entrada |
| `src/data/processed.py` | Split estratificado e geração dos CSVs processados |
| `src/models/dataset.py` | Seleção das entradas do modelo na referência |
| `src/models/features.py` | Engenharia de atributos do Home Credit |
| `src/models/preprocessing.py` | Pipeline serializado de treino e inferência |
| `src/models/candidates.py` | Estimadores candidatos e pesos de classe |
| `src/models/evaluation.py` | Avaliação e seleção por ROC-AUC |
| `src/models/artifacts.py` | Modelo e métricas locais |
| `src/models/tracking.py` | Tracking opcional no MLflow |
| `config/config.yaml` | Parâmetros de treino (`production_size`, `cv_folds`, `random_state`) e seção `mlflow` |
| `src/utils/config_loader.py` | Validação dos novos campos (`ModelConfig`) e nova `MlflowConfig` |
| `Makefile` | Alvos `train`, `first`, `rebuild` |
| `docker-compose.yml` | Serviço `train` (profile) com tracking apontado ao MLflow |
| `Dockerfile` | Stage `train` (imagem de serviço + cliente MLflow) |

## 1. Treino do modelo (`src/models/`)

O módulo inicial era um stub. O fluxo agora divide o CSV bruto, gera os dois
datasets processados, treina, seleciona e persiste o pipeline que a API serve;
`train.py` coordena as etapas implementadas nos módulos acima. Os contratos
Pandera completos ainda serão implementados.

### Candidatos e seleção

- **Dois candidatos: Regressão Logística e XGBoost.** A Logística é a linha de
  base interpretável e barata; o XGBoost captura não-linearidades e interações,
  típicas de dados tabulares de crédito. Treinar os dois e comparar evita
  escolher um modelo "por fé".
- **Seleção por ROC-AUC em validação cruzada.** Em crédito o alvo é desbalanceado
  (~8% de default), então acurácia engana. ROC-AUC mede separação entre classes
  independente do limiar e é a métrica padrão do desafio Home Credit.
- **Split estratificado 80/20** (`train_test_split(..., stratify=target)`) para
  reservar produção estável simulada. Os 80% de referência passam por cinco
  folds estratificados para a seleção de modelo; a base de 20% não é avaliada
  nem usada no ajuste. O vencedor é reajustado em toda a referência.
- **Tratamento de desbalanceamento.** Logística usa `class_weight="balanced"`;
  XGBoost usa `scale_pos_weight = n_neg / n_pos`, calculado só nas linhas de
  treino de cada fold. Sem isso os modelos tendem a
  prever sempre "adimplente" e ignorar o default, que é justamente o risco a
  detectar.

### Pipeline único (treino = inferência)

O artefato é um `sklearn.Pipeline` com quatro etapas, nesta ordem:

1. `align` — `FunctionTransformer(reindex_features)` realinha as colunas da
   requisição ao conjunto visto no treino.
2. `engineer` — `FunctionTransformer(engineer_features)` cria as features de
   domínio (seção 2).
3. `prep` — `ColumnTransformer` com imputação + escala (numérico) e imputação +
   one-hot (categórico).
4. `clf` — o classificador escolhido.

- **Justificativa:** encapsular alinhamento, engenharia e pré-processamento
  dentro do pipeline garante que a inferência em `/predict` aplique exatamente a
  mesma transformação do treino. Elimina a classe de bug "feature computada no
  treino mas esquecida em produção" (training/serving skew).
- **Robustez ao contrato de `/predict`:** `align` tolera chaves ausentes
  (reindexa para `NaN`) e o `engineer` nunca falha por coluna faltante — a
  derivada correspondente vira `NaN` e é imputada adiante. Uma requisição
  incompleta não derruba a API.

### Saídas

- Artefato: `models/artifacts/credit_model.joblib` (caminho vem do
  `config.yaml`).
- Dados: `data/processed/ref_data.csv` (com `TARGET`) e
  `data/processed/prod_estavel.csv` (sem `TARGET`), ambos fora do Git.
- Métricas: `metrics/training_metrics.json` com média, desvio e resultados por
  fold do ROC-AUC de cada candidato, além do modelo escolhido.

## 2. Feature engineering de domínio (`src/models/features.py`)

São as features clássicas do Home Credit Default Risk. Todas usam divisão
protegida (`_safe_ratio`: divisão por zero ou `inf` vira `NaN`) e acesso
protegido às colunas (`_safe_column`: coluna ausente vira série de `NaN`).

### Razões de endividamento e capacidade de pagamento

| Feature | Fórmula | Por que importa |
|---|---|---|
| `CREDIT_INCOME_PERCENT` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Quanto o crédito pesa sobre a renda — alavancagem |
| `ANNUITY_INCOME_PERCENT` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Comprometimento da renda com a parcela |
| `CREDIT_TERM` | `AMT_ANNUITY / AMT_CREDIT` | Prazo implícito do empréstimo |
| `CREDIT_GOODS_PERCENT` | `AMT_CREDIT / AMT_GOODS_PRICE` | Crédito acima do valor do bem = risco |
| `INCOME_PER_PERSON` | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` | Renda per capita do domicílio |

- **Justificativa:** inadimplência é melhor explicada por *proporções* do que
  por valores absolutos. Uma dívida de R$ 50 mil é trivial para renda alta e
  grave para renda baixa; a razão captura isso, o valor cru não.

### Tempo e idade

| Feature | Fórmula | Por que importa |
|---|---|---|
| `DAYS_EMPLOYED_PERCENT` | `DAYS_EMPLOYED / DAYS_BIRTH` | Estabilidade: fração da vida empregado |
| `AGE_YEARS` | `-DAYS_BIRTH / 365` | Idade em anos, legível (o dado cru é dias negativos) |

### Escores externos

Agrega `EXT_SOURCE_1/2/3` em `EXT_SOURCES_MEAN`, `EXT_SOURCES_STD` e
`EXT_SOURCES_PROD`.

- **Justificativa:** os `EXT_SOURCE_*` são, de longe, os preditores mais fortes
  do dataset. Agregá-los resume o sinal em menos colunas, reduz o impacto de um
  escore ausente (média sobre os disponíveis) e expõe a concordância entre eles
  (desvio/produto), o que modelos lineares não extraem sozinhos.

### Tratamento de anomalia

- **`DAYS_EMPLOYED == 365243` vira `NaN`.** Esse valor (~1000 anos) é a
  sentinela do Home Credit para "nunca empregado". Deixá-lo como número
  distorce escala, imputação e qualquer razão que o use. Convertido para `NaN`,
  é imputado corretamente.

## 3. Configuração (`config/config.yaml` + `config_loader.py`)

Novos parâmetros em `model`:

- `production_size: 0.2` — fração do CSV bruto reservada para produção estável.
- `cv_folds: 5` — número de folds estratificados na referência de 80%.
- `random_state: 42` — seed única para split e modelos, garantindo
  reprodutibilidade.

Nova seção `mlflow` (`tracking_uri`, `experiment`). Todos os campos são
validados por Pydantic (`ModelConfig`, nova `MlflowConfig`) — configuração
inválida falha no load, não no meio do treino. `MlflowConfig` tem defaults, de
modo que configs e testes antigos continuam válidos.

## 4. Tracking com MLflow (`src/models/tracking.py`)

Ao final do treino são registrados: parâmetros (`production_size`, `cv_folds`,
`random_state`, modelo escolhido), as métricas `cv_roc_auc_mean_<candidato>`,
`cv_roc_auc_std_<candidato>` e `cv_roc_auc_best`, a tag `selected_model` e o
artefato `.joblib`.

- **Import opcional + best-effort.** `import mlflow` é protegido por `try`, e
  toda a função é envolvida em `try/except`: se o mlflow não estiver instalado
  (ex.: `make train` no host) ou o servidor estiver fora do ar, o treino emite
  um aviso e termina normalmente. **Justificativa:** tracking é telemetria; uma
  falha de telemetria nunca deve impedir a produção do artefato.
- **`MLFLOW_TRACKING_URI` tem prioridade sobre o YAML.** No Docker o compose
  injeta `http://mlflow:5000`; no host cai no default local. Permite o mesmo
  código logar no servidor em container e num store local fora dele.
- **Logging no fim do treino.** Quando `log_to_mlflow` roda, o fit já terminou
  (~dezenas de segundos) e o servidor MLflow (subido como dependência do
  `docker compose run`) já está de pé.

## 5. Docker (`Dockerfile` + `docker-compose.yml`)

- **Novo stage `train`** no Dockerfile: estende a imagem `api` e instala o
  cliente `mlflow` via `pip`. **Justificativa:** mantém a imagem de serviço
  enxuta (sem mlflow em produção) e evita adicionar `mlflow` ao
  `pyproject.toml`/`poetry.lock` — o que exigiria regenerar o lock e engordar o
  build da API.
- **Novo serviço `train`** no compose, com `profiles: ["train"]` para não subir
  junto do `docker compose up`. Monta `data/raw` como somente-leitura e
  `data/processed`, `models` e `metrics` com escrita. Define
  `MLFLOW_TRACKING_URI` e `depends_on: mlflow`, de modo que `docker compose run`
  sobe o servidor de tracking automaticamente.

## 6. Makefile

- `train` — gera os CSVs e roda o treino via Poetry no host.
- `first` — bootstrap de ponta a ponta via Docker: cria o `.env` (se faltar),
  treina no container e sobe a stack em background (`api`, `prometheus`,
  `grafana`, `mlflow`). **Justificativa:** um único comando leva de repositório
  limpo a ambiente pronto, reduzindo erro de setup manual.
- `rebuild` — `docker compose --profile train build --no-cache`, reconstrói
  todas as imagens (incluindo `train`) do zero após mudança de Dockerfile,
  dependências ou código de treino.

## Pré-requisito

`data/raw/application_train.csv` (Home Credit, Kaggle) presente no host. Fica
fora do Git por tamanho e licença da competição.

## Resultado de referência

Resultado histórico anterior à divisão 80/20 e à validação cruzada, obtido em
amostra de 50 mil linhas:

| Modelo | ROC-AUC (validação) |
|---|---|
| Regressão Logística | 0.7429 |
| XGBoost (escolhido) | 0.7490 |

Esses números não representam o novo protocolo de validação cruzada.
