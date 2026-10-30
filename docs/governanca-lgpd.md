# Governança e LGPD (Etapa 4)

Este documento descreve como o projeto trata dados pessoais sob a Lei Geral de
Proteção de Dados (Lei 13.709/2018), aplicando _Privacy by Design_ e
_by Default_. O dataset Home Credit é anonimizado na origem, mas o tratamento
aqui descrito é o que o projeto adotaria em produção com dados reais de
titulares.

## Tratamento de dados sensíveis (PII)

**Mapeamento dos campos.** No Home Credit, classificamos as colunas em três
grupos:

| Grupo | Campos (exemplos) | Natureza |
|-------|-------------------|----------|
| Identificador | `SK_ID_CURR` | PII direta (chave do titular) |
| PII indireta / quase-identificadores | `DAYS_BIRTH` (idade), `CODE_GENDER`, `DAYS_EMPLOYED`, `AMT_INCOME_TOTAL`, `CNT_FAM_MEMBERS`, `NAME_FAMILY_STATUS`, dados de região | reidentificam o titular em combinação |
| Dados derivados sensíveis | `EXT_SOURCE_1/2/3` (score externo), decisão de crédito | refletem situação financeira |

**Minimização.** O modelo usa apenas o conjunto estritamente necessário à
decisão de risco (`EXT_SOURCE_*`, `AMT_INCOME_TOTAL`, `DAYS_BIRTH`,
`CNT_FAM_MEMBERS` e derivados em `src/models/features.py`). Campos não usados
na inferência não entram no pipeline de serving.

**Pseudonimização.** `SK_ID_CURR` é um identificador substituto, não o CPF ou
nome do titular. Em produção, a correspondência entre `SK_ID_CURR` e a
identidade real ficaria em um cofre separado, fora do escopo do modelo.

**Não exposição em logs e respostas.** O projeto impõe, por construção, que
valores de clientes nunca vazem para a camada de observabilidade:

- O contrato de dados (`src/data/contracts.py`) registra **apenas nomes de
  regras e contagens de linhas** — nunca valores, identificadores ou linhas de
  clientes (ver docstring e `ValidationResult`).
- Os eventos do pipeline (`src/observability/events.py`) usam uma **lista de
  permissão**: só chaves conhecidas, escalares e textos curtos entram no
  evento; rendas, créditos e linhas de titulares são rejeitados.
- Há teste garantindo que o valor de entrada da API **não aparece nos logs de
  erro** (`tests/test_serving.py`).

## Base legal para a decisão de crédito

**Base legal (art. 7º da LGPD).** O tratamento para concessão de crédito
apoia-se na **execução de contrato ou procedimentos preliminares a pedido do
titular** (art. 7º, V) — o titular solicita o crédito e o escore é condição da
análise. Subsidiariamente, aplica-se o **legítimo interesse** (art. 7º, IX)
para prevenção à fraude e gestão de risco, sempre com teste de
proporcionalidade e salvaguarda dos direitos do titular.

**Decisão automatizada (art. 20).** O titular tem direito à **revisão de
decisões tomadas unicamente com base em tratamento automatizado**. O projeto
viabiliza isso porque:

- a decisão é **rastreável** — cada inferência é reproduzível a partir do
  artefato versionado e do `run_id` registrado no MLflow;
- o limiar de decisão é **explícito e configurável** (`config/config.yaml`),
  não uma caixa-preta oculta;
- o pipeline mantém o histórico de métricas e drift, permitindo auditar sob
  quais condições a decisão foi tomada.

Em produção, isso se complementaria com um canal de contestação humana e
fornecimento de informações claras sobre os critérios ao titular que solicitar.

## Plano de retenção

| Artefato | Local | Retenção | Eliminação |
|----------|-------|----------|-----------|
| Dataset bruto (`data/raw/`) | fora do Git | enquanto durar o treino/experimento | descartado ao fim do ciclo; não versionado |
| Dados de produção simulados e processados (`data/processed/`) | fora do Git | vida do experimento | regenerados pelo pipeline; não versionados |
| Predições em lote | efêmeras no run | duração do run de drift | não persistidas com PII |
| Logs de eventos (`logs/`, JSONL) | fora do Git | sem PII por construção | rotação operacional |
| Relatórios (`reports/`) | versionados | evidência do trabalho | contêm só estatísticas agregadas, sem linhas de clientes |

O princípio é a **minimização da retenção**: nenhum dado com PII é versionado
no Git; só estatísticas agregadas e artefatos de modelo persistem.

## Viés e mitigação

**Variáveis proxy de características protegidas.** Mesmo sem usar raça ou
religião, campos como `CODE_GENDER`, região, `NAME_FAMILY_STATUS` e até
`EXT_SOURCE_*` podem atuar como **proxy** de grupos protegidos, propagando
discriminação indireta. `AMT_INCOME_TOTAL` e `DAYS_EMPLOYED` correlacionam com
desigualdades estruturais.

**Mitigação.** A infraestrutura de drift e métricas (Etapa 3) permite
**segmentar** o desempenho e a estabilidade da distribuição por grupo, para
detectar disparidade entre faixas (ex.: taxa de aprovação ou erro por gênero /
faixa etária). A recomendação de governança é monitorar paridade por grupo além
da acurácia global e revisar features de alto risco de proxy antes de promover
um modelo.

## Causalidade

**Por que a degradação ocorre.** A suspeita do contexto (inflação, novos perfis
de clientes) é um caso de **data drift** com possível **concept drift**: a
distribuição das variáveis de entrada muda (renda, emprego) e a relação
entrada→inadimplência também pode mudar, de modo que o modelo treinado no
passado passa a errar silenciosamente.

**Causa vs. correlação.** O relatório Evidently (`reports/data_drift_report.html`,
`data_drift_summary.csv`) aponta **quais features** sofreram desvio via PSI/KS.
É preciso distinguir: um desvio em `AMT_INCOME_TOTAL` por inflação é uma
**causa** plausível de erro; já um desvio em features derivadas
(`EXT_SOURCES_*`) pode ser apenas **correlação/consequência** da mudança nas
fontes. A leitura causal evita retreinar reagindo a sintoma errado.

**Intervenções.** Conforme o diagnóstico, o time pode: (1) **retreinar** com
dados recentes quando há data drift genuíno; (2) **recalibrar o limiar** de
decisão quando a ordenação do risco se mantém mas a taxa-base mudou; (3)
**revisar a política/feature set** quando há concept drift estrutural. Os
alertas (`monitoring/prometheus/alerts.yml`, ex.: `DataDriftSevere`) sinalizam
quando acionar cada intervenção.

## Vídeo STAR

_Link a adicionar após a gravação (5 minutos, método STAR)._

Roteiro previsto:

- **Situation** — risco da degradação silenciosa de um modelo de crédito em
  produção após mudanças econômicas.
- **Task** — interceptar dados ruins, detectar drift e garantir compliance.
- **Action** — contrato Pandera que bloqueia lotes inválidos, simulação de
  drift, relatório Evidently (PSI/KS) e observabilidade (Prometheus/Grafana/
  MLflow), além do plano de governança LGPD acima.
- **Result** — relatório apontando as variáveis degradadas, alertas disparados
  e lições aprendidas.
