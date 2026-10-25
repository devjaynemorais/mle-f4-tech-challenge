# Etapa 3 — Diagnóstico das etapas anteriores e plano

Avaliação feita em 2026-10-25 sobre a `main` atualizada (51 arquivos
incorporados no último `git pull`). Fonte dos requisitos:
[`MLET - Tech Challenge - Fase 4.pdf`](MLET%20-%20Tech%20Challenge%20-%20Fase%204.pdf).
A decisão técnica está na [ADR 0001](adr/0001-observabilidade-pipeline-e-modelo.md).
Nada foi executado nem alterado em código; a verificação foi por leitura do
repositório, `ruff` e `pytest`.

## Legenda de classificação

| Marca | Significado para a Etapa 3 |
|-------|----------------------------|
| 🔴 **BLOCK** | Impede entregar parte da Etapa 3; resolver antes |
| 🟠 **SOFT-BLOCK** | Dá para começar a construir, mas não fechar nem demonstrar |
| 🟡 **ATENÇÃO** | Não bloqueia, mas custa nota ou gera retrabalho |
| 🟢 **OK** | Concluído e verificado |

## 1. Situação por etapa

### Etapa 1 — Validação de dados e contratos (peso 25%)

| Item do enunciado | Estado | Marca |
|-------------------|--------|-------|
| Classificador base treinado (XGBoost + regressão logística) | Código completo (`src/models/`, 54 testes passando). **Artefatos não existem neste ambiente** (`data/raw/` e `models/artifacts/` vazios) | 🟠 |
| Contrato de dados Pandera com regras rígidas | **`src/data/contracts.py` é só docstring.** Pandera está instalado, mas não há schema, nem regra, nem teste | 🔴 |
| ≥ 3 regras rígidas de qualidade | Existe apenas validação de `TARGET` 0/1 e ID único em `processed.py` (2 verificações pontuais, fora de um contrato) | 🔴 |
| Lote com erros bloqueado, com demonstração | Inexistente | 🔴 |

**Impacto na Etapa 3:** o enunciado pede logs de "sucesso/falhas na
**ingestão** e transformação". A falha de ingestão é justamente o contrato
barrando um lote. Sem ele, o painel "lotes bloqueados" fica vazio e a demo
não tem como mostrar uma falha real. **Classificação: 🟠 SOFT-BLOCK para
construir; 🔴 BLOCK para fechar e filmar.** É também o maior risco de nota do
projeto: 25% do peso com entrega zerada.

### Etapa 2 — Simulação e detecção de drift (peso 25%)

| Item | Estado | Marca |
|------|--------|-------|
| Dataset de produção com ≥ 2 variáveis alteradas | Feito (idade, crédito, preço do bem, anuidade, escores externos) | 🟢 |
| Evidently comparando referência × produção | `reports/data_drift_report.html` versionado | 🟢 |
| PSI/KS por feature | `DriftDetector` + `reports/data_drift_summary.csv` (130 features; 17 com drift em ao menos um método; 7 por PSI ≥ 0,25) | 🟢 |
| **Gerar predições para o dataset de produção** | **Pendente** (o próprio checklist e o README admitem) | 🔴 |
| Concept drift | Só `P(X)` é avaliado; o alvo do período 2 é sintético | 🟡 |

**Impacto na Etapa 3:** o enunciado pede "alertas de degradação do modelo".
Sem predições sobre a produção não existe score médio, taxa de `default`
prevista nem comparação entre períodos. **🔴 BLOCK** para a parte de modelo
do dashboard. É uma pendência pequena (um script de predição em lote) e deve
ser a primeira tarefa da F0.

O contexto do enunciado cita Data Drift **e** Concept Drift. Como o rótulo do
período 2 é gerado por regra sintética, desempenho nesse período não é
evidência de degradação real. Tratar como limitação documentada e monitorar
deriva de score e de taxa predita como proxy (ver ADR D5).

### Etapa 3 — Observabilidade (peso 20%): ponto de partida

| Item | Estado | Marca |
|------|--------|-------|
| Métricas da API + dashboard Grafana (10 painéis) | Funciona; cobre só infraestrutura da API | 🟢 |
| Prometheus, Grafana e MLflow no compose | Presentes | 🟢 |
| Métricas de pipeline | `pipeline_metrics.py` é casca | — |
| Resultados de drift no backend de observabilidade | Inexistente | — |
| Regras de alerta | Inexistentes | — |
| Logs centralizados e estruturados | `basicConfig` textual, sem `run_id` | — |
| MLflow | Só o treino é registrado | — |

### Etapa 4 — Governança (peso 15%) e vídeo (15%)

Não bloqueiam a Etapa 3. 🟢 não bloqueante. Dois acoplamentos: (a) os logs da
Etapa 3 são tratamento de dados e precisam respeitar a política de PII que a
Etapa 4 vai descrever; (b) o "Result" do vídeo STAR pede "alertas gerados",
que só existem depois da Etapa 3.

## 2. Pontos de atenção

| # | Ponto | Marca | Ação sugerida |
|---|-------|-------|---------------|
| 1 | `.venv` sem `matplotlib`/`seaborn`: `tests/test_data_drift_simulation.py` e `tests/test_drift_report.py` não coletam (`ModuleNotFoundError`). Com os dois ignorados: 54 testes passam e `ruff check` está limpo. A CI usa `poetry install`, então não é falha do projeto | 🟡 | `make install` |
| 2 | CSV bruto Kaggle ausente em `data/raw/`; sem ele não dá para rodar `make train`, `make drift` e `make report` de ponta a ponta | 🟠 | Baixar o arquivo (não versionar) |
| 3 | Sete commits recentes (fora os dois merges) não seguem `tipo(escopo): resumo` do `AGENTS.md` (ex.: "Adicionar geração de relatório de drift com Evidently"). O enunciado e o critério de avaliação pedem histórico semântico | 🟡 | Seguir o padrão a partir de agora; reescrever histórico só com sua autorização, pois altera commits já publicados |
| 4 | Documentos desatualizados: `checklist_requisitos_tc4.md` e README ainda dizem "Etapa 1 pendente" em partes já feitas; `.env.example` menciona o MLflow "para a Etapa 3" | 🟡 | Atualizar junto da F6 |
| 5 | `monitoring/prometheus/prometheus.yml` usa `job_name: triage-api`, resquício da Fase 3 | 🟡 | Renomear para `credit-api` e conferir o dashboard |
| 6 | O dashboard atual não tem painel de pipeline, drift nem alertas; a nota de 20% depende justamente disso | 🟡 | F5 |
| 7 | Jobs em lote + Prometheus (*pull*) exigem Pushgateway ou equivalente | 🟡 | ADR D2 |
| 8 | `reports/data_drift_report.html` está versionado. Confirmar que o HTML do Evidently contém só agregados (histogramas) e nenhum valor individual ou ID | 🟡 | Inspecionar antes da entrega final |
| 9 | O `TARGET` sintético não pode virar "desempenho em produção" em nenhum painel | 🟡 | Regra no ADR D5 |
| 10 | Limite de 20 linhas por função e McCabe ≤ 10 se aplicam ao código novo | 🟡 | Helpers pequenos; teste arquitetural já cobre |
| 11 | Modelo não é decisão automática; o painel não deve sugerir isso | 🟡 | Texto do dashboard e do README |
| 12 | **A referência do relatório não é a do treino.** `make report` compara `prod_periodo1_estavel` × `prod_periodo2_drift`; `ref_data.csv` não entra. O enunciado pede "Dataset de Referência × Dataset de Produção" | 🟡 | Adicionar `ref_data` × período 1 (esperado: sem drift); serve também de base de calibração dos alertas |
| 13 | **Limiar de alerta inconsistente com os dados.** PSI ≥ 0,25 em 7 de 114 features = 6,1%; um crítico de 10% não dispararia. KS marca 14,9% (excesso de poder estatístico com ~30 mil linhas) e qui-quadrado 0 de 16. `AGE_YEARS` e `DAYS_BIRTH` duplicam a informação | 🟡 | Críticos revistos na ADR (2% e 5%); margem estreita, calibrar com a execução estável |
| 14 | **`mlflow` não está no `pyproject.toml`**; só existe na imagem `train`. `make drift`/`make report` rodam no host e não registrariam runs | 🟠 | Decidir: grupo opcional no Poetry ou serviço `drift` no compose (ADR D4) |
| 15 | **Pushgateway não acumula contadores.** Métricas de job precisam ser gauges do último estado; `increase()` não funciona | 🟡 | Corrigido na ADR D2/D5/D6 |
| 16 | Compose: Prometheus só monta `prometheus.yml` (falta `alerts.yml` e `rule_files`); `train` não monta `logs/` nem recebe `PUSHGATEWAY_URL`; `.gitignore` não cobre `logs/` nem `*.jsonl` | 🟡 | F2 |
| 17 | Pushgateway sem autenticação; Grafana com acesso anônimo e senha padrão `admin` no compose | 🟡 | Publicar portas só em `127.0.0.1`; trocar a senha fora do ambiente local |
| 18 | `reports/xgboost_*_prod1_vs_prod2.png` e `src/drift/evaluation.py` comparam desempenho no período 2 sem nenhuma menção a "sintético" | 🟡 | Rotular os gráficos ou o README antes do vídeo |
| 19 | Os eventos de pipeline são tratamento de dados: precisam de rotação e entrar no plano de retenção | 🟡 | Registrar na Etapa 4 |
| 20 | Nada garante que `alerts.yml`, dashboard e catálogo de métricas fiquem coerentes; a CI só constrói a imagem `api` | 🟡 | Teste que cruza regras, painéis e catálogo |
| 21 | Para o vídeo é preciso reproduzir "sem alerta" e "alertas gerados" em sequência | 🟡 | `make demo-observability` com 3 cenários (ADR D7) |
| 22 | `docs/architecture.md`, a lista de métricas do `AGENTS.md` e o mapa "ainda stub" do `AGENTS.md` ficarão desatualizados | 🟡 | F6 |

## 3. Cobertura dos critérios de avaliação

| Critério (peso) | Situação | O que falta |
|-----------------|----------|-------------|
| Validação de dados (25%) | 🔴 Não atendido | Contrato Pandera com ≥ 3 regras, lote inválido bloqueado, script de demonstração, testes |
| Detecção de drift (25%) | 🟢 Quase completo | Predições em lote; nota explícita sobre concept drift |
| Observabilidade e logs (20%) | 🟠 Parcial (só API) | Toda a Etapa 3 (ADR) |
| Governança LGPD (15%) | 🔴 Pendente | Etapa 4: PII, base legal, retenção, análise causal, viés |
| Vídeo STAR ≤ 5 min (15%) | 🔴 Pendente | Roteiro e gravação, depois das etapas 1 a 3 |

Requisitos obrigatórios do repositório ainda abertos: pipeline de validação
(data contracts), documentação de governança no README, plano de privacidade
(LGPD). Os dois últimos são da Etapa 4.

## 4. Ordem de execução recomendada

1. **F0 (desbloqueio):** `make install`, CSV em `data/raw/`, `make train`,
   `make drift`, `make report`; criar `predict_batch`; definir a interface do
   contrato.
2. **Etapa 1 completa, em paralelo à F1:** é o maior ganho de nota por hora de
   trabalho e alimenta o evento de falha de ingestão.
3. **F1 a F6** conforme a ADR.
4. **Etapa 4 e roteiro do vídeo** usando a trilha produzida na Etapa 3.

## 5. Decisões tomadas (ADR 0001, seção 7)

1. **Arquitetura aprovada:** Prometheus/Grafana e MLflow, com Pushgateway.
2. **Loki/Promtail:** fora do caminho crítico; só se sobrar tempo.
3. **Ordem:** Etapa 1 completa antes de fechar a Etapa 3; F0 e F1 em paralelo.
4. **Commits fora do padrão:** não reescrever (já publicados e mesclados em
   PRs); padrão `tipo(escopo)` obrigatório daqui para frente.
5. **Limiares:** aviso 2% e crítico 5% de `drift_share{psi}`; recalibrar após
   `ref_data` × período 1.
6. **MLflow no host:** serviço `drift` no compose, sem tocar no Poetry.
7. **Referência:** incluir `ref_data` × período 1 no `make report`.

Reabrir qualquer item só se a execução real com o CSV contradisser a premissa.

## 6. Atualização após a implementação (2026-10-25)

| Bloqueio ou ponto | Situação | Como foi resolvido |
|-------------------|----------|--------------------|
| 🔴 Contrato Pandera vazio | Resolvido | `src/data/contracts.py`: 5 colunas, 8 regras nomeadas; barra o lote na ingestão e na predição |
| 🔴 Lote inválido bloqueado | Resolvido | `make contract-demo` e teste de ponta a ponta (treino não roda se o contrato falha) |
| 🔴 Predições em lote | Resolvido | `src/drift/predict_batch.py` + `make predict`; AUC só no período estável |
| 🟠 CSV do Kaggle ausente | **Aberto** | Validado com dados sintéticos (testes e stack Docker); falta rodar com o CSV real |
| 🟠 `mlflow` fora do Poetry | Resolvido | Serviço `drift` no compose reaproveita a imagem `train` |
| 🟡 Referência × estável | Resolvido | `make report` gera os dois relatórios; baseline sem drift nos testes |
| 🟡 Limiar crítico | Resolvido | 2% e 5%; recalibrar com o CSV real |
| 🟡 Pushgateway e contadores | Resolvido | Só gauges do último estado |
| 🟡 Compose, `.gitignore`, portas | Resolvido | Pushgateway em `127.0.0.1`, `alerts.yml` montado, `logs/` e `*.jsonl` ignorados |
| 🟡 Teste de coerência | Resolvido | `tests/test_monitoring_assets.py` |
| 🟡 Cenários para o vídeo | Resolvido | `make demo-observability` (o vídeo em si não faz parte desta entrega) |
| 🟡 Docs desatualizados | Resolvido | README, checklist, `AGENTS.md`, `architecture.md`, `metricas_saude.md` |
| 🟡 Commits fora do padrão | Mantido | Decisão: não reescrever; novos commits seguem o padrão |
| 🟡 Gráficos do período 2 sem aviso de alvo sintético | **Aberto** | Rotular em `reports/` e no README antes da entrega final |
| 🟡 HTML do Evidently só com agregados | **Aberto** | Conferir com o CSV real |
| Etapa 4 (LGPD) e vídeo STAR | **Aberto** | Fora do escopo desta entrega |
