# Overview do dataset e do pré-processamento — Etapa 1

## Objetivo

A Etapa 1 constrói um modelo baseline de risco de crédito para servir de base às etapas de validação, drift e observabilidade do Tech Challenge da Fase 4. O alvo é binário: `TARGET = 1` indica dificuldade de pagamento segundo a definição do dataset; `TARGET = 0` representa os demais casos. O modelo é apoio analítico, não uma decisão automática de crédito.

## Dataset utilizado

O projeto usa o [Home Credit Default Risk, na página de dados do Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data). O dicionário local [`data/raw/HomeCredit_columns_description.csv`](../data/raw/HomeCredit_columns_description.csv) descreve sete famílias de arquivos: aplicações (`application_train`/`application_test`), histórico de crédito externo (`bureau` e `bureau_balance`), pagamentos anteriores (`previous_application`), parcelas (`installments_payments`), saldo de crédito rotativo (`credit_card_balance`) e histórico de crédito ao consumidor (`POS_CASH_balance`).

Embora o pacote contenha tabelas relacionais, o treinamento atual lê somente `application_train.csv`. Não há junção com as tabelas auxiliares nem uso de `application_test.csv`. Assim, cada linha de treino é uma aplicação e, das 122 colunas originais, 120 ficam como preditores após retirar identificador e alvo. A engenharia acrescenta atributos derivados depois.

Na cópia local analisada, `application_train.csv` tem 307.511 linhas e 122 colunas. A classe positiva corresponde a 24.825 casos (8,07%), contra 282.686 casos da classe 0. O identificador `SK_ID_CURR` não tem duplicatas nesta cópia; essa verificação foi feita para este diagnóstico, mas não é uma regra aplicada pelo loader de treino.

As variáveis cobrem características da solicitação e do cliente: tipo e valor do crédito, renda e prestação; idade e tempo de emprego; composição familiar; moradia e região; indicadores de contato e documentos; escores externos; e informações agregadas de consultas ao bureau. Há campos demográficos, de renda e relacionados a crédito, portanto o CSV bruto fica local e não deve ser versionado nem exposto em logs ou exemplos.

## Como os dados são preparados para treinar

1. **Leitura e seleção do alvo.** [`src/data/loader.py`](../src/data/loader.py) carrega o CSV configurado e verifica apenas que `SK_ID_CURR` e `TARGET` existem. O alvo é convertido para inteiro. Para formar as features, `split_features_target` remove `SK_ID_CURR` e `TARGET`; o identificador não entra no modelo e o alvo não pode atuar como preditor.
2. **Amostragem para desenvolvimento.** [`config/config.yaml`](../config/config.yaml) define `sample_size: 50000` e `random_state: 42`. O código escolhe uma amostra aleatória reprodutível de 50 mil linhas antes do split. Na cópia analisada, essa amostra contém 4.023 positivos (8,05%). `sample_size: null` permite usar o arquivo completo.
3. **Separação treino/validação.** O código reserva 20% para validação e usa `stratify=target`, mantendo a proporção das classes em cada partição. A seed 42 torna a divisão reprodutível.
4. **Correção de sentinela e engenharia de atributos.** A transformação substitui `DAYS_EMPLOYED = 365243` por ausente, pois é uma sentinela do dataset para situação de emprego, e não uma duração plausível. Em seguida cria:
   - razões de crédito/renda, prestação/renda, prestação/crédito, crédito/valor do bem e renda por pessoa;
   - proporção entre dias de emprego e dias de idade, além de idade em anos;
   - média, desvio-padrão e produto de `EXT_SOURCE_1`, `EXT_SOURCE_2` e `EXT_SOURCE_3`.

   Divisões por zero, infinitos, colunas ausentes e valores inválidos convertidos para numérico resultam em `NaN`, que segue para imputação.
5. **Imputação e codificação.** O pré-processador identifica colunas numéricas e categóricas pelos tipos do DataFrame. Nas numéricas, preenche ausentes com a mediana e aplica `StandardScaler`. Nas categóricas, preenche com a categoria mais frequente e aplica one-hot encoding; categorias desconhecidas são ignoradas e categorias com frequência abaixo de 1% são agrupadas conforme `min_frequency=0.01`.
6. **Pipeline reproduzível.** O artefato junta alinhamento de colunas, engenharia, imputação/escala/codificação e classificador num único `sklearn.Pipeline`. Isso faz a API aplicar as mesmas transformações durante a inferência e permite imputar atributos ausentes numa requisição.

## Treino e seleção do baseline

São comparados Regressão Logística e XGBoost. Para lidar com o desbalanceamento, a regressão usa `class_weight="balanced"` e o XGBoost calcula `scale_pos_weight` pela proporção entre negativos e positivos no conjunto de treino. A seleção usa ROC-AUC na validação, em vez de acurácia, que seria pouco informativa com apenas cerca de 8% de positivos.

O documento de implementação [`docs/etapa1-treino.md`](../docs/etapa1-treino.md) registra como resultados de referência ROC-AUC 0,7429 para a Regressão Logística e 0,7490 para XGBoost, que foi selecionado naquele registro. No estado do workspace analisado, `metrics/training_metrics.json` e `models/artifacts/credit_model.joblib` ainda não existem; portanto, esses números são os valores relatados na documentação, não resultados que pude reproduzir nesta análise. Quando executado, o treino grava as métricas em `metrics/training_metrics.json` e o pipeline selecionado em `models/artifacts/credit_model.joblib`.

## Limites da implementação atual frente ao enunciado

O PDF do Tech Challenge pede um dataset de referência limpo, contrato de dados com regras rígidas (por exemplo, idade válida, renda não nula e identificador sem duplicatas) e demonstração de que um lote inválido é bloqueado. No repositório, [`src/data/contracts.py`](../src/data/contracts.py) ainda é um stub. O loader verifica somente a presença das colunas obrigatórias; não rejeita duplicatas, intervalos inválidos ou nulos de negócio. A imputação no pipeline trata ausências para o modelo, mas não substitui o contrato de qualidade exigido.

Também não há limpeza prévia de linhas nem uso das tabelas auxiliares. A preparação atual é principalmente transformação de atributos e imputação dentro do pipeline. Assim, é correto descrever a Etapa 1 como baseline treinável com pré-processamento reproduzível, enquanto validação contratual e bloqueio de dados ruins permanecem como pendências do enunciado/checklist.

## Arquivos de referência

- [`docs/MLET - Tech Challenge - Fase 4.pdf`](../docs/MLET%20-%20Tech%20Challenge%20-%20Fase%204.pdf) — requisitos da Etapa 1 e critérios de avaliação.
- [`docs/checklist_requisitos_tc4.md`](../docs/checklist_requisitos_tc4.md) — estado registrado dos requisitos.
- [`docs/etapa1-treino.md`](../docs/etapa1-treino.md) — decisões e resultados de referência do treino.
- [`src/models/train.py`](../src/models/train.py) — implementação efetiva de split, features, pré-processamento e treino.
