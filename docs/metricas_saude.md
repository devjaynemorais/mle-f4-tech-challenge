# Métricas de saúde, logs e alertas (Etapa 3)

Decisão de arquitetura em [ADR 0001](adr/0001-observabilidade-pipeline-e-modelo.md).
Este documento descreve o que é medido, onde aparece e como operar.

## Visão geral

```text
etapas do pipeline ──► track_stage ──► eventos JSONL   (logs/pipeline_events.jsonl)
 (ingestão, contrato,        │
  transformação, treino,     ├──► Pushgateway ──► Prometheus ──► Grafana + alertas
  drift, predição)           └──► MLflow (uma run por etapa, com o HTML do Evidently)
API FastAPI ───────────────────► /metrics ──► Prometheus
```

Uma única fonte alimenta os três destinos, então o painel, o MLflow e o log
mostram a mesma execução (mesmo `run_id`).

## Como operar

```bash
docker compose up -d --build                 # API, Prometheus, Grafana, MLflow, Pushgateway
make compose-pipeline                        # treino + drift + predição + relatórios nos containers
# ou, no host (com o CSV em data/raw/):
PUSHGATEWAY_URL=http://localhost:9091 make pipeline
make demo-observability                      # cenários: estável, driftado, lote inválido
make contract-demo                           # lote válido passa, lote com erros é barrado
make metrics-reset                           # zera as métricas guardadas no Pushgateway
```

| Onde olhar | Endereço |
|------------|----------|
| Dashboard de pipeline, drift e alertas | `http://localhost:3000` → "Credit Pipeline — Saúde, Drift e Alertas" |
| Dashboard da API | `http://localhost:3000` → "Credit Scoring API — Overview" |
| Alertas e regras | `http://localhost:9090/alerts` |
| Runs por etapa e relatórios HTML | `http://localhost:5000` → experimento `credit-pipeline-monitoring` → run `drift_detection-<run_id>` → aba **Artifacts** → pasta `drift_detection/` (parâmetro `dataset` diz a comparação) |
| Métricas brutas dos jobs | `http://localhost:9091` |

## Logs de execução (eventos estruturados)

Cada etapa grava uma linha JSON por evento em `logs/pipeline_events.jsonl`
(ignorado pelo Git; rotação em `observability.events_max_bytes`, com uma
geração anterior `.1`) e no log padrão.

| Campo | Conteúdo |
|-------|----------|
| `run_id` | Identifica a execução; o `make pipeline` usa o mesmo em todas as etapas |
| `stage` | `ingestion`, `contract_validation`, `transformation`, `training`, `batch_scoring`, `drift_detection` |
| `event` | `stage_started`, `stage_finished`, `contract_violation`, `drift_result`, `alert_raised` |
| `status`, `duration_s`, `rows` | Resultado, duração e volume da etapa |
| `details` | Somente chaves da lista de permissão (regra, contagens, nomes de features, estatísticas agregadas) |

**Privacidade por construção.** O schema só aceita chaves permitidas, textos de
até 64 caracteres e escalares ou listas simples. Identificadores, rendas,
créditos, linhas de clientes e mensagens de exceção não são aceitos; em falhas
registra-se apenas o tipo da exceção. Testes automatizados verificam isso.

## Catálogo de métricas

Todas as métricas dos jobs são **gauges do último estado**: o Pushgateway não
acumula contadores entre execuções. O histórico fica no MLflow e no JSONL.
Os rótulos `stage`, `dataset`, `comparison` e `period` vêm da chave de
agrupamento do envio.

### Pipeline (Pushgateway)

| Métrica | Rótulos | Significado |
|---------|---------|-------------|
| `pipeline_stage_last_status` | `stage`, `dataset` | 1 = última execução com sucesso, 0 = falha |
| `pipeline_stage_duration_seconds` | `stage`, `dataset` | Duração da última execução |
| `pipeline_stage_last_success_timestamp_seconds` | `stage`, `dataset` | Instante do último sucesso (só atualizado em sucesso) |
| `pipeline_rows_processed` | `stage`, `dataset` | Linhas processadas na última execução |
| `data_batch_blocked` | `stage`, `dataset` | 1 = o último lote foi barrado pelo contrato |
| `data_contract_violations` | `rule`, `dataset` | Linhas que violaram a regra no último lote |

### Drift e modelo (Pushgateway)

| Métrica | Rótulos | Significado |
|---------|---------|-------------|
| `drift_features_total` | `method`, `comparison` | Features comparadas (`psi`, `ks`, `chi2`) |
| `drift_features_drifted` | `method`, `comparison` | Features com drift no método |
| `drift_share` | `method`, `comparison` | Fração de features com drift |
| `drift_max_psi` | `comparison` | Maior PSI entre as features |
| `drift_feature_psi` | `feature`, `comparison` | PSI de cada feature numérica |
| `drift_feature_ks` | `feature`, `comparison` | Estatística KS de cada feature numérica |
| `drift_alerts_active` | `comparison` | Alertas de drift do último ciclo |
| `model_batch_score_mean` | `period` | Probabilidade média de default prevista |
| `model_batch_positive_rate` | `period` | Fração prevista como `default` |
| `model_batch_rows` | `period` | Linhas pontuadas |
| `model_batch_roc_auc` | `period` | AUC, **somente no período estável** (alvo real) |

Comparações: `reference_vs_stable` (sanidade, referência de treino × período
estável; deve ficar sem drift) e `stable_vs_drifted` (a degradação simulada).
Períodos: `stable` e `drifted`.

> O alvo do período driftado é sintético. Por isso ele **não** gera métrica de
> desempenho: score médio e taxa prevista são proxies de degradação, não prova.

### API (raspadas de `/metrics`)

`http_requests_total`, `http_request_duration_seconds`,
`credit_predictions_total`, `credit_default_probability`,
`credit_prediction_duration_seconds`, `credit_prediction_errors_total` e
`credit_model_loaded`. As três primeiras são contrato do projeto e não mudam.

## Alertas

Regras em `monitoring/prometheus/alerts.yml`; visíveis em `/alerts` do
Prometheus e no painel "Alertas disparando" do Grafana.

| Alerta | Condição | Severidade |
|--------|----------|------------|
| `ModelNotLoaded` | `max(credit_model_loaded) == 0` por 1 min | critical |
| `HighErrorRate` | mais de 5% de respostas 5xx por 2 min | critical |
| `HighInferenceLatency` | p95 da inferência acima de 0,5 s por 5 min | warning |
| `PipelineStageFailed` | `pipeline_stage_last_status == 0` | critical |
| `PipelineStale` | sem sucesso há mais de 24 h | warning |
| `DataContractViolation` | `data_batch_blocked == 1` | warning |
| `DataDriftDetected` | `drift_share{method="psi"}` entre 2% e 5% | warning |
| `DataDriftSevere` | `drift_share{method="psi"}` ≥ 5% | critical |

Os limiares de drift ficam em `config/config.yaml` (`observability.drift_*`) e
um teste garante que a regra e o config permanecem iguais. A mesma lógica gera
o evento `alert_raised` no log.

**Calibração com o dado real** (Home Credit, 307.511 linhas; execução de
2026-10-25): `reference_vs_stable` deu `drift_share` 0% e `drift_max_psi`
0,0012, muito abaixo do aviso de 2% (zero falsos positivos). `stable_vs_drifted`
deu 7 de 114 features com PSI ≥ 0,25 (6,14%) e `drift_max_psi` 9,28, acima do
crítico de 5%. A margem do crítico é de uma feature (6 já bastam; há 7), e
`AGE_YEARS` e `DAYS_BIRTH` contam a mesma informação duas vezes. Se a simulação
mudar de intensidade, reavalie. Alertar por KS foi descartado: com dezenas de
milhares de linhas o teste marca diferenças irrelevantes (17 de 114 features).

## Limitações

- **Runs antigas do MLflow.** Runs criadas antes de o servidor passar a receber os
  artefatos por proxy (`--artifacts-destination`) guardaram o caminho dentro do
  container do job e **não têm** o HTML. Para começar limpo:
  `docker compose down`, apague a pasta `mlflow-data/` e rode
  `docker compose up -d` e `make compose-pipeline`. O HTML do Evidently também
  fica sempre em `reports/`.
- As métricas dos jobs mostram o **último** estado de cada etapa, não uma série
  histórica; use o MLflow e o JSONL para o histórico.
- O Pushgateway guarda dados em memória: reiniciá-lo (`make metrics-reset`)
  apaga o último estado. O alerta `PipelineStageFailed` do lote inválido da
  demonstração permanece ativo até esse reset ou até uma execução bem-sucedida
  do mesmo conjunto.
- Não há Alertmanager com e-mail ou Slack: o alerta é gerado e visível, mas não
  notificado.
- O modelo é apoio à análise de crédito e exige revisão humana; nenhuma métrica
  aqui indica decisão automática.
