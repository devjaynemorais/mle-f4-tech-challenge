# ADR 0001 — Observabilidade de pipelines e modelos (Etapa 3)

- **Status:** Implementada em 2026-10-25 (fases F0 a F6; F7 opcional não feita). Ver a seção 9 para o que mudou em relação ao plano
- **Data:** 2026-10-25
- **Escopo:** Etapa 3 do Tech Challenge Fase 4
- **Relacionados:** [`etapa3-diagnostico-e-plano.md`](../etapa3-diagnostico-e-plano.md),
  [`checklist_requisitos_tc4.md`](../checklist_requisitos_tc4.md), `AGENTS.md`

## 1. Contexto

O enunciado da Etapa 3 pede três coisas e um entregável:

1. **Centralizar logs e métricas**: logs de execução do pipeline (sucesso e
   falha em ingestão e transformação) integrados aos resultados dos testes de
   drift da Etapa 2.
2. **Rastrear/monitorar**: MLflow para o ciclo de vida, **ou** dashboard
   interativo (Datadog, cloud ou open-source).
3. **Definir e documentar métricas de saúde**: status de execução, latência,
   volume processado, quantidade de alertas de falha ou drift.

Entregável: dashboard ou ambiente de rastreamento que evidencie a saúde do
pipeline de dados, a estabilidade da infraestrutura e os alertas de
degradação do modelo. Peso na nota: **20%** ("apresentação visual das
métricas de saúde do modelo e dos alertas de drift").

### Estado de partida (verificado no repositório)

| Já existe | Falta |
|-----------|-------|
| API com 7 métricas Prometheus (`http_*`, `credit_*`) e 10 painéis Grafana | Nenhuma métrica de **pipeline** (`src/observability/pipeline_metrics.py` é casca) |
| Prometheus, Grafana e MLflow no `docker-compose.yml` | Nenhuma **regra de alerta**; nenhum alerta visível |
| MLflow registra apenas o treino (`src/models/tracking.py`) | Resultados de **drift** não chegam a nenhum backend de observabilidade |
| `configure_logging()` com formato textual | Logs **não estruturados**, sem `run_id`, sem destino único |
| `DriftDetector` + CSV/HTML Evidently | Predições do dataset de produção (pendência da Etapa 2) |

### Restrições do projeto (`AGENTS.md`)

- Não quebrar `/health`, `/predict` e as três métricas Prometheus existentes
  (`http_requests_total`, `http_request_duration_seconds`,
  `credit_predictions_total`); se alterar, atualizar dashboard e testes juntos.
- Funções de `src/` com no máximo 20 linhas lógicas; McCabe ≤ 10.
- Sem frameworks novos se o stack atual resolve; parâmetros de experimento em
  `config/config.yaml`, só runtime em variáveis de ambiente.
- **Nenhum identificador, renda ou crédito em logs, métricas, mensagens de
  erro ou exemplos.** Isso vale para os novos eventos.
- O modelo é apoio à análise, não decisão automática; sem claims de equidade
  ou conformidade LGPD sem evidência.

## 2. Decisões

### D1 — Duas camadas complementares: Prometheus/Grafana + MLflow

**Decisão:** usar **Prometheus + Grafana** como painel operacional e de
alertas, e **MLflow** como registro de execuções (um *run* por execução de
etapa, com métricas de drift e o HTML do Evidently como artefato).

| Opção | Prós | Contras |
|-------|------|---------|
| Só MLflow | Já no compose; simples | Sem série temporal, sem alertas, fraco para latência/infra |
| Só Prometheus/Grafana | Dashboard e alertas nativos | Sem histórico versionado de execuções e artefatos |
| **Ambos (escolhida)** | Cobre "ou" do enunciado com folga; cada ferramenta no que faz melhor | Dois destinos para manter coerentes |

A coerência é garantida por **uma única fonte de eventos** (D3): os dois
destinos são alimentados pelo mesmo evento, nunca por cálculos paralelos.

### D2 — Pushgateway para jobs em lote

`make train`, `make drift` e `make report` são processos curtos; o Prometheus
(modelo *pull*) não os alcançaria. **Decisão:** adicionar o serviço
`prom/pushgateway` ao compose e enviar as métricas de pipeline ao final de
cada etapa (`job="credit-pipeline"`, agrupamento por `stage`). A API continua
sendo raspada diretamente.

Alternativas descartadas:

- *Textfile collector / node_exporter*: exige mais um componente e acesso ao
  disco do host.
- *Mover os jobs para dentro da API*: mistura serving com batch e viola a
  separação de responsabilidades.
- *Airflow* (usado na Fase 3): excesso de infraestrutura para esta fase.

Cuidados conhecidos:

- O Pushgateway mantém o último valor indefinidamente e **não acumula
  contadores entre execuções** (cada processo recomeça do zero e o push
  substitui o grupo). Por isso as métricas de job são **gauges do último
  estado** (D5), nunca `counter` com `increase()`. O histórico de execuções
  fica no MLflow e no JSONL.
- O job do Prometheus para o Pushgateway precisa de `honor_labels: true`.
- O endereço vem de `PUSHGATEWAY_URL` (runtime): `http://localhost:9091` para
  `make` no host e `http://pushgateway:9091` dentro do compose. Publicar a
  porta apenas em `127.0.0.1`; o Pushgateway não tem autenticação.
- O Prometheus só monta `prometheus.yml` hoje: é preciso montar também
  `alerts.yml` e declarar `rule_files`.

### D3 — Log de eventos estruturado como fonte única

**Decisão:** cada etapa emite eventos JSON (uma linha por evento) para stdout
e para `logs/pipeline_events.jsonl` (ignorado pelo Git). O evento é o
contrato interno da camada de observabilidade:

```json
{"ts": "2026-10-25T12:00:00Z", "run_id": "…", "stage": "ingestion",
 "event": "stage_finished", "status": "success", "duration_s": 3.2,
 "rows": 61502, "details": {"rule": null}}
```

- `stage` ∈ `ingestion`, `contract_validation`, `transformation`, `training`,
  `drift_simulation`, `batch_scoring`, `drift_detection`.
- `event` ∈ `stage_started`, `stage_finished`, `contract_violation`,
  `drift_result`, `alert_raised`.
- Campos permitidos: contagens, durações, nomes de regras e de features,
  estatísticas agregadas. **Proibidos:** `SK_ID_CURR`, valores de renda/
  crédito, linhas, exemplos de registros e mensagens de exceção com dados
  (registrar apenas o tipo da exceção, como já faz a API).
- `logs/` e `*.jsonl` **ainda não estão no `.gitignore`** (só `*.log`):
  incluir antes do primeiro evento gravado. O arquivo tem rotação por tamanho
  e entra no plano de retenção da Etapa 4, pois também é tratamento de dados.
- No compose, o serviço `train` precisa montar `./logs:/app/logs` e receber
  `PUSHGATEWAY_URL`.

Loki/Promtail foram avaliados para "centralizar logs", mas ficam como
**fase opcional (F7)**: adicionam dois serviços e não são necessários para
atender ao enunciado, que pede a integração de logs e métricas, não uma pilha
de logs específica. Se houver tempo, entram atrás de um *profile* do compose.

### D4 — Uma API pequena e testável em `src/observability/`

Módulos (cada função ≤ 20 linhas lógicas):

| Módulo | Responsabilidade |
|--------|------------------|
| `events.py` | Schema do evento, `run_id`, escritor JSONL |
| `stage_tracking.py` | `track_stage(stage)`: gerenciador de contexto que emite início/fim/falha, mede duração e delega a métricas (nome distinto de `src/models/tracking.py`, que é do MLflow do treino) |
| `pipeline_metrics.py` | Registro Prometheus dos jobs e push ao Pushgateway |
| `alerts.py` | Avalia limiares sobre o `DriftReport` e emite `alert_raised` |
| `mlflow_runs.py` | Run por etapa com métricas de drift e artefatos |

**Dependência do MLflow fora do container.** `mlflow` não está no
`pyproject.toml`; hoje só é instalado na imagem `train`
(`pip install mlflow`), e `src/models/tracking.py` ignora a ausência. `make
drift` e `make report` rodam no host via Poetry, então sem ação não
registrariam runs. **Decisão:** criar o serviço `drift` no compose (profile
`train`, imagem `train`, que já tem `mlflow`, `evidently` e `matplotlib`,
todos dependências `main`), com `PUSHGATEWAY_URL` e `MLFLOW_TRACKING_URI`
internos. `make drift`/`make report` no host continuam funcionando sem MLflow.
A ausência do MLflow nunca pode derrubar o pipeline.

Pontos de integração (mudança mínima nos módulos existentes):
`src/data/processed.py` (ingestão/transformação), `src/models/train.py`,
`src/drift/data_drift_simulation.py`, `src/drift/report.py` e o novo
`src/drift/predict_batch.py` (pré-requisito, ver D7).

### D5 — Catálogo de métricas de saúde

Prefixos estáveis; rótulos de baixa cardinalidade. **Nenhum rótulo carrega
dado de cliente.**

Todas as métricas de job são **gauges do último estado** (ver D2). Contagens
históricas vêm do MLflow/JSONL, não de `increase()`.

**Pipeline (novas, via Pushgateway):**

| Métrica | Tipo | Rótulos | Responde a |
|---------|------|---------|------------|
| `pipeline_stage_last_status` | gauge (1 sucesso, 0 falha) | `stage` | Status de execução |
| `pipeline_stage_duration_seconds` | gauge | `stage` | Latência do pipeline |
| `pipeline_stage_last_success_timestamp_seconds` | gauge | `stage` | Obsolescência |
| `pipeline_rows_processed` | gauge | `stage`, `dataset` | Volume |
| `data_contract_violations` | gauge (último lote) | `rule` | Qualidade na ingestão |
| `data_batch_blocked` | gauge (1 se o último lote foi barrado) | — | Bloqueio do contrato |

**Drift e modelo (novas):**

| Métrica | Tipo | Rótulos | Responde a |
|---------|------|---------|------------|
| `drift_features_total` / `drift_features_drifted` | gauge | `method` | Quantas features driftaram |
| `drift_share` | gauge | `method` | Fração de features com drift |
| `drift_max_psi` | gauge | — | Severidade da pior feature |
| `drift_feature_psi` / `drift_feature_ks` | gauge | `feature` | Quais features (≈130 séries por métrica) |
| `drift_alerts_active` | gauge | `severity` | Quantidade de alertas do último ciclo |
| `model_batch_score_mean` | gauge | `period` | Deriva de score (proxy de degradação) |
| `model_batch_positive_rate` | gauge | `period` | Taxa de predição `default` |

**API (existentes, mantidas):** `http_requests_total`,
`http_request_duration_seconds`, `credit_predictions_total`,
`credit_default_probability`, `credit_prediction_duration_seconds`,
`credit_prediction_errors_total`, `credit_model_loaded`.

> Desempenho do modelo (AUC etc.) só é publicado para o **período estável**,
> cujo alvo é real. O alvo do período driftado é sintético e **nunca** aparece
> como métrica de desempenho; o painel o trata como proxy de score/predição.

### D6 — Alertas como regras Prometheus versionadas

**Decisão:** `monitoring/prometheus/alerts.yml` carregado pelo Prometheus,
visível no Grafana por meio da série `ALERTS` e de um painel dedicado.
Alertmanager com receptor real fica fora de escopo; o entregável é o alerta
**gerado e visível**, como pede o vídeo STAR ("alertas gerados").

| Alerta | Condição (inicial, a calibrar) | Severidade |
|--------|-------------------------------|------------|
| `ModelNotLoaded` | `max(credit_model_loaded) == 0` por 1 min | critical |
| `HighErrorRate` | taxa 5xx > 5% por 2 min | critical |
| `HighInferenceLatency` | p95 de `credit_prediction_duration_seconds` > 0,5 s por 5 min | warning |
| `PipelineStageFailed` | `pipeline_stage_last_status == 0` | critical |
| `PipelineStale` | `time() - pipeline_stage_last_success_timestamp_seconds > 86400` | warning |
| `DataContractViolation` | `data_batch_blocked == 1` | warning |
| `DataDriftDetected` | `drift_share{method="psi"} >= 0.02` | warning |
| `DataDriftSevere` | `drift_share{method="psi"} >= 0.05` | critical |

Limiares de drift ficam em `config/config.yaml` (seção `observability`) e são
espelhados na regra; os números acima são **pontos de partida** e devem ser
calibrados com a execução estável (que deve ficar abaixo de todos).

> **Calibração com os dados atuais** (`reports/data_drift_summary.csv`): PSI
> ≥ 0,25 em 7 de 114 features numéricas = **6,1%**; KS sinaliza 17 (14,9%) e
> qui-quadrado 0 de 16. Um limiar crítico de 10% **não dispararia** com o
> cenário driftado, por isso o crítico é 5%. A margem é estreita (1 feature) e
> a contagem está inflada por redundância (`AGE_YEARS` e `DAYS_BIRTH` são a
> mesma informação). Alternativa a avaliar: severidade por `drift_max_psi ≥ 1`
> (4 features hoje). Alerta sobre KS é descartado: com ~30 mil linhas por
> período, o teste sinaliza diferenças irrelevantes.

### D7 — Pré-requisitos tratados antes de começar (ver diagnóstico)

1. **Predições em lote do dataset de produção** (pendência da Etapa 2):
   `src/drift/predict_batch.py` + `make predict`. Sem isso, `model_batch_*` e
   o alerta de degradação do modelo não têm de onde vir.
2. **Contrato de dados mínimo** (Etapa 1): a Etapa 3 precisa do evento
   `contract_violation`. Implementar a Etapa 1 completa antes do vídeo; para
   começar a Etapa 3, basta a interface (`validate_batch(df) -> ContractResult`).
3. **Ambiente**: `poetry install` para restaurar `matplotlib`/`seaborn` no
   `.venv`, e o CSV bruto Kaggle em `data/raw/`.
4. **Comparação com a referência de treino.** O `make report` compara
   `prod_periodo1_estavel` × `prod_periodo2_drift`; a referência de treino
   (`ref_data.csv`) não entra. O enunciado pede "Dataset de Referência ×
   Dataset de Produção". Adicionar a comparação `ref_data` × período 1
   (deve dar **sem drift**) cumpre a leitura literal e serve de base para
   calibrar os alertas contra falso positivo.
5. **Cenários de demonstração**: `make demo-observability` com três cenários
   (estável, driftado, lote inválido), para o vídeo mostrar "sem alerta" e
   "alertas gerados" sem edição manual de arquivos.

## 3. Consequências

**Positivas**

- Dashboard único cobre infraestrutura, pipeline, drift e modelo, e atende
  a ambas as opções do enunciado (MLflow **e** dashboard).
- Fonte única de eventos evita divergência entre Grafana e MLflow e dá ao
  vídeo STAR uma trilha reproduzível (`make pipeline`, depois abrir Grafana).
- Privacidade por construção: o schema de evento não comporta dado pessoal.

**Negativas / custos**

- Um serviço novo no compose (Pushgateway) e um arquivo de regras.
- Métricas de pipeline em modo *push* têm semântica de "último valor"; mitigada
  pela métrica de último sucesso.
- Cardinalidade `feature` (~130) é aceitável, mas não deve crescer sem revisão.

**Riscos e mitigação**

| Risco | Mitigação |
|-------|-----------|
| Etapa 1 (contratos) continua vazia e pesa 25% da nota | Priorizar logo após a F1; ver diagnóstico |
| Alertas disparando sempre ou nunca | Calibrar limiares no período estável; testar os dois cenários |
| Vazamento de PII em logs | Teste automatizado que valida o schema do evento e proíbe campos fora da lista |
| Limite de 20 linhas por função | Helpers pequenos; o teste arquitetural já existe |
| Pushgateway fora do ar derruba o pipeline | Push com *timeout* e falha tolerada (log de aviso), como o MLflow já faz |

## 4. Plano de implementação (commits semânticos)

| Fase | Entrega | Commit sugerido |
|------|---------|-----------------|
| F0 | Ambiente, CSV, `predict_batch`, interface de contrato | `feat(drift): predições em lote do período de produção` |
| F1 | Eventos estruturados + `track_stage` + testes de schema/PII | `feat(observability): eventos estruturados de execução do pipeline` |
| F2 | Métricas de pipeline + Pushgateway no compose (porta em `127.0.0.1`) e no Prometheus (`honor_labels`, `rule_files`); `logs/` e `*.jsonl` no `.gitignore`; renomear `job_name: triage-api` | `feat(observability): métricas de pipeline via Pushgateway` |
| F3 | Drift → métricas e alertas (`alerts.py`, seção `observability` no YAML) | `feat(observability): publicar resultados de drift e gerar alertas` |
| F4 | Runs MLflow por etapa com HTML do Evidently como artefato | `feat(observability): runs MLflow de monitoramento` |
| F5 | Dashboard `pipeline-health` + `alerts.yml` | `feat(monitoring): painel de saúde do pipeline e regras de alerta` |
| F6 | `docs/metricas_saude.md`, README, checklist, `docs/architecture.md`, lista de métricas em `AGENTS.md`, `make pipeline` e `make demo-observability` | `docs(observability): documentar métricas de saúde e operação` |
| F7 (opcional) | Loki/Promtail por *profile* | `feat(monitoring): centralização de logs com Loki` |

## 5. Critérios de aceite

- [ ] `make pipeline` executa ingestão → contrato → transformação → treino →
      simulação → predição em lote → drift, emitindo eventos para cada etapa.
- [ ] Um lote inválido é bloqueado, gera `data_batches_blocked_total` e
      aparece como alerta.
- [ ] O período estável **não** dispara alerta de drift; o driftado dispara
      `DataDriftDetected` e `DataDriftSevere`.
- [ ] Grafana mostra: status por etapa, duração, volume, contratos, features
      com drift (PSI/KS), alertas ativos, score do modelo e saúde da API.
- [ ] MLflow lista um run por execução com métricas de drift e o HTML anexo.
- [ ] `docs/metricas_saude.md` documenta cada métrica, rótulo e alerta.
- [ ] Nenhum evento, métrica ou log contém ID, renda, crédito ou linha de
      cliente (teste automatizado).
- [ ] Teste automatizado valida `alerts.yml` (YAML válido) e garante que toda
      métrica usada em regra ou painel existe no catálogo; `promtool check
      rules` quando disponível. A CI hoje só constrói a imagem `api`.
- [ ] A comparação `ref_data` × período 1 não dispara alerta de drift.
- [ ] `make lint` e `make test` passam; `/health`, `/predict` e as três
      métricas originais permanecem intactos.

## 7. Decisões fechadas

| # | Questão | Decisão | Motivo |
|---|---------|---------|--------|
| 1 | Arquitetura | Aprovada: Prometheus/Grafana **e** MLflow, com Pushgateway | Cobre as duas opções do enunciado; peso de 20% |
| 2 | Loki/Promtail | **Fora do caminho crítico.** Só entra se F1 a F6 terminarem e sobrar tempo | Dois serviços a mais sem exigência no enunciado; o tempo rende mais na Etapa 1 |
| 3 | Ordem | **Etapa 1 completa antes de fechar a Etapa 3.** F0 e F1 seguem em paralelo | Maior peso (25%) com entrega zerada; alimenta o evento de falha de ingestão |
| 4 | Commits fora do padrão | **Não reescrever.** Padrão `tipo(escopo)` obrigatório daqui para frente | Os commits já foram publicados e mesclados em PRs (#1, #2); reescrever exigiria *force push* em branch compartilhada, com risco maior que o ganho |
| 5 | Limiares | Aviso a 2% e crítico a 5% de `drift_share{psi}`; `drift_max_psi` apenas exibido no painel. Ajustar após a execução `ref_data` × período 1 | Único par que dispara no cenário driftado (6,1%) com os dados atuais; severidade por `max_psi` fica como evolução |
| 6 | MLflow no host | Serviço `drift` no compose | Sem mexer em `pyproject.toml`/`poetry.lock`, como já fez o stage `train` |
| 7 | Referência do relatório | **Incluir** `ref_data` × período 1 como terceiro artefato (`make report` gera os dois HTML) | Cumpre a leitura literal do enunciado e dá a linha de base sem drift |

## 8. Fora de escopo

Alertmanager com e-mail/Slack, Datadog, tracing distribuído, orquestração com
Airflow, retreino automático e qualquer análise causal ou de equidade (Etapa 4).

## 9. Resultado da implementação

Todas as fases F0 a F6 foram entregues. Diferenças em relação ao plano:

| Plano | Implementado | Motivo |
|-------|--------------|--------|
| Módulo `tracking.py` | `stage_tracking.py` | Evitar confusão com `src/models/tracking.py` |
| Rótulo `dataset` nas métricas de contrato e de etapa | Vem da chave de agrupamento do Pushgateway (`stage`, `dataset`, `comparison`, `period`) | O Pushgateway já anexa essas chaves; repetir como rótulo conflita |
| `drift_alerts_active{severity}` | `drift_alerts_active` sem rótulo, por `comparison` | O alerta mais grave substitui os menores; a severidade está no alerta |
| Alerta de aviso em `>= 2%` | Faixa exclusiva `2% <= share < 5%` | Evita dois alertas para o mesmo evento |
| MLflow sempre ativo se instalado | Só com `MLFLOW_TRACKING_URI` definido | Servidor inexistente atrasaria o pipeline local |
| Relatório referência × estável opcional | Incluído no `make report` | Decisão 7 |
| Loki (F7) | Não implementado | Decisão 2 |

**Validação.** Testes automatizados (unidade, contrato, coerência entre config,
alertas, dashboards e compose, e pipeline de ponta a ponta com dados
sintéticos); `promtool check rules/config`; stack Docker real (API, Prometheus,
Grafana, MLflow, Pushgateway) com `make compose-pipeline` e o cenário de lote
inválido. Com dados sintéticos, a comparação referência × estável ficou sem
drift (`drift_share` 0) e a estável × driftada disparou `DataDriftSevere`
(`drift_share` 0,40). **Não foi executado com o CSV real do Kaggle**, então os
limiares continuam sendo pontos de partida.
