# Roteiro do vídeo STAR (5 minutos)

Roteiro de gravação da demonstração técnica do Tech Challenge Fase 4. Método
STAR, limite de 5 minutos. O link do vídeo gravado vai em
[`docs/governanca-lgpd.md`](governanca-lgpd.md) (seção "Vídeo STAR").

**Orçamento de tempo:** Situation 0:45 · Task 0:45 · Action 2:30 · Result 1:00.

---

## Situation (≈45s)

> Uma fintech de crédito tem um modelo de credit scoring em produção. Com a
> mudança da economia — inflação, novos perfis de clientes — a diretoria
> suspeita que o modelo está errando em silêncio: continua respondendo, mas as
> decisões degradaram sem ninguém perceber. Esse é o risco da **degradação
> silenciosa**: sem monitoramento, só se descobre quando o prejuízo já
> aconteceu.

**Tela:** diagrama de arquitetura (`docs/architecture.md`) ou slide do contexto.

## Task (≈45s)

> Nosso trabalho foi construir a camada de sustentação e confiabilidade em volta
> do modelo. Três objetivos técnicos: **um**, interceptar dados ruins antes de
> entrarem no modelo; **dois**, detectar drift quando a distribuição muda; e
> **três**, garantir compliance e governança sob a LGPD, com análise causal da
> degradação.

**Tela:** checklist das 4 etapas / tabela de requisitos do README.

## Action (≈2min30s)

**1. Validação de dados — contrato (≈50s)**

> Usamos um contrato Pandera com regras rígidas: renda positiva e dentro de um
> teto, idade acima de 18 anos, colunas obrigatórias não-nulas e sem duplicata
> de identificador. Quando passo um lote com erros, a porta de ingestão
> **bloqueia** e levanta `DataContractError` — e, por design, a mensagem só
> contém o nome da regra e a contagem de linhas, nunca o dado do cliente.

**Tela:** rodar `python -m src.data.contract_demo` (ou `make`-alvo); mostrar o
lote válido passando e o inválido sendo barrado.

**2. Simulação e detecção de drift (≈50s)**

> Geramos um dataset de produção alterando de propósito ao menos duas variáveis
> — renda e emprego. Rodamos as predições em lote e comparamos referência vs
> produção com o Evidently, usando PSI e Kolmogorov-Smirnov. O relatório HTML
> aponta exatamente quais features sofreram drift.

**Tela:** abrir `reports/data_drift_report.html`; destacar as features
degradadas e o `data_drift_summary.csv`.

**3. Observabilidade (≈50s)**

> Todo o pipeline é instrumentado: eventos estruturados, métricas de drift
> exportadas via Pushgateway para o Prometheus, e um dashboard de saúde no
> Grafana. Regras de alerta disparam quando o drift passa do limiar — por
> exemplo o `DataDriftSevere`. O ciclo de vida do modelo fica rastreado no
> MLflow sob um único `run_id`.

**Tela:** dashboard Grafana `pipeline-health.json`; alerta em
`monitoring/prometheus/alerts.yml`; run no MLflow.

## Result (≈1min)

> Resultado: conseguimos **flagrar a degradação** que antes era silenciosa. O
> relatório Evidently mostra claramente as variáveis que driftaram, os alertas
> disparam no Grafana, e a decisão permanece auditável e rastreável — o que
> sustenta o direito de revisão da LGPD.
>
> A leitura causal nos diz **qual intervenção** aplicar: retreinar quando há
> data drift genuíno, recalibrar o limiar quando só a taxa-base mudou, ou
> revisar a política quando há concept drift.
>
> Lição aprendida: modelo em produção não é entrega, é operação contínua —
> qualidade de dados, observabilidade e governança precisam andar juntas.

**Tela:** voltar ao dashboard com o alerta aceso; encerrar no resumo das 4
etapas concluídas.

---

## Checklist de gravação

- [ ] Ambiente de pé (`make pipeline` ou serviços do Docker Compose).
- [ ] `reports/data_drift_report.html` gerado e aberto.
- [ ] Grafana acessível com o dashboard de saúde do pipeline.
- [ ] Ensaiar uma vez cronometrando — manter ≤ 5:00.
- [ ] Gravar, exportar e **colar o link em `docs/governanca-lgpd.md`**.
