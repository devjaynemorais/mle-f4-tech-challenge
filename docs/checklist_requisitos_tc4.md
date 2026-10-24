# Checklist de requisitos — Tech Challenge Fase 4

Fonte: [MLET - Tech Challenge - Fase 4.pdf](MLET%20-%20Tech%20Challenge%20-%20Fase%204.pdf).

## Requisitos obrigatórios (repositório GitHub)

| Requisito | Status | Evidência |
|-----------|--------|-----------|
| Pipeline de validação de dados (data contracts) configurado | Pendente (Etapa 1) | `src/data/contracts.py` |
| Scripts para simulação e detecção estatística de drift | Implementado: detecção por feature e resumo CSV com `make report` | `src/drift/`, `reports/data_drift_summary.csv` |
| Dashboards ou relatórios HTML comprovando o monitoramento | Implementado: dashboard Grafana e relatório Evidently via `make report` | `monitoring/grafana/`, `reports/data_drift_report.html` |
| Documentação consolidada de Governança no README.md | Pendente (Etapa 4) | `README.md`, seção "Governança e LGPD" |
| Histórico de commits semântico e organizado | Em andamento | `git log` |

## Boas práticas obrigatórias

| Prática | Status |
|---------|--------|
| Separação entre dataset de referência (treino) e de produção (drift) | Implementada em `make train` e `make drift`; arquivos locais fora do Git |
| Pelo menos 3 regras rígidas de qualidade de dados | Pendente (Etapa 1) |
| Relatório estatístico da mudança de distribuição (PSI ou KS) | Implementado; CSV gerado por `make report` |
| Documentação de compliance com plano de privacidade (LGPD) | Pendente (Etapa 4) |

## Etapas de desenvolvimento

### Etapa 1 — Validação de dados e contratos
- [ ] Treinar classificador base com dataset limpo de referência (o treino atual usa o CSV bruto)
- [ ] Criar contrato de dados (Pandera ou Great Expectations) com regras estritas
- [ ] Forçar a passagem de um lote com erros e demonstrar o bloqueio da ingestão
- [x] Base de leitura do CSV bruto com validação de colunas obrigatórias

### Etapa 2 — Simulação e detecção de drift
- [x] Criar dataset de produção com alteração intencional de ao menos duas variáveis
- [ ] Gerar predições para o dataset de produção
- [x] Usar Evidently AI para comparar o período estável e o driftado (`make report`)
- [x] Identificar, com PSI/KS, quais features sofreram drift no CSV resumo
- [x] Entregável: relatório HTML do Evidently gerado em `reports/`

O relatório compara distribuições de features (`P(X)`) e exclui `SK_ID_CURR`
e `TARGET`; o segundo período usa alvo sintético, que não é avaliado como
desempenho observado em produção. A geração de predições por lote continua
pendente.

### Etapa 3 — Observabilidade de pipelines e modelos
- [ ] Centralizar logs de execução do pipeline e resultados dos testes de drift
- [ ] Rastreamento com MLflow ou exportação de métricas para dashboard
- [ ] Documentar métricas de saúde (status de execução, latência, volume, alertas)
- [x] API com métricas Prometheus e dashboard Grafana provisionado
- [x] Serviços de Prometheus, Grafana e MLflow no Docker Compose

### Etapa 4 — Governança (LGPD)
- [ ] Plano de proteção de dados sensíveis (PII) no README
- [ ] Base legal para a decisão de crédito no README
- [ ] Plano de retenção no README
- [ ] Análise causal do problema no README
- [ ] Vídeo STAR de 5 minutos com link no README

## Critérios de avaliação

| Critério | Peso |
|----------|------|
| Validação de dados (contratos interceptando anomalias) | 25% |
| Detecção de drift (simulação correta e relatório Evidently) | 25% |
| Observabilidade e logs (dashboard ou relatórios automatizados) | 20% |
| Governança (LGPD) — clareza sobre privacidade, base legal e viés | 15% |
| Clareza na demonstração técnica (vídeo STAR) | 15% |

## Dataset

Home Credit Default Risk (Kaggle). Requisito mínimo do enunciado: classificação
binária financeira com pelo menos 5.000 amostras. O `application_train.csv` tem
~307 mil linhas.
