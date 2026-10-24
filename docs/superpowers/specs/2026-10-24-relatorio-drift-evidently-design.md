# Especificação: relatório de drift com Evidently

## Intenção

Gerar uma evidência reproduzível que identifique mudanças nas features entre os
dois períodos de produção simulados: `prod_periodo1_estavel.csv` e
`prod_periodo2_drift.csv`. O relatório será acionado por `make report`.

## Abordagens

1. **Somente Evidently:** produzir um HTML com `DataDriftPreset`. Tem menor
   complexidade, mas deixa os resultados e limiares principalmente sob as
   escolhas padrão da biblioteca.
2. **Evidently e detector local (escolhida):** produzir o HTML Evidently e um
   CSV por feature com os resultados já suportados por `DriftDetector` (KS e
   PSI para colunas numéricas; qui-quadrado para categóricas). Mantém a leitura
   visual e torna métodos, estatísticas, limiares e decisões auditáveis.

## Objetivos e limites

- Comparar o período 2 driftado com o período 1 estável, isolando as mudanças
  introduzidas pela simulação.
- Analisar todas as features comuns aos arquivos. Excluir `SK_ID_CURR` e
  `TARGET`: o primeiro identifica clientes e o segundo é real no período
  estável, mas sintético no driftado.
- Salvar `reports/data_drift_report.html` e
  `reports/data_drift_summary.csv`.
- Manter o escopo na detecção de mudanças em `P(X)`. Não inferir degradação do
  modelo, não avaliar o `TARGET` sintético e não adicionar alertas ou operação
  contínua da Etapa 3.

## Decisões de implementação

- Criar `src/drift/report.py` como CLI; `make report` chama
  `python -m src.drift.report` depois de `make drift`.
- Usar `Report([DataDriftPreset()])`, com referência = período estável e atual
  = período driftado; salvar o snapshot com `save_html`.
- Reutilizar `DriftDetector.detect_all` e `generate_drift_summary` para o CSV,
  sem duplicar o cálculo estatístico.
- Validar existência, esquema compatível e conteúdo não vazio antes de
  comparar. Remover ID e alvo dos dois frames antes de passá-los aos detectores.
- Guardar nomes dos arquivos de períodos, saídas e parâmetros estatísticos em
  `config/config.yaml`; estender os modelos tipados em
  `src/utils/config_loader.py` e fazer o simulador usar os nomes configurados.
- Atualizar `README.md` e `docs/checklist_requisitos_tc4.md` para documentar a
  sequência `make train`, `make drift`, `make report` e os dois artefatos.
- Adicionar testes unitários para exclusão de ID/alvo, comparação de todas as
  features, incompatibilidade de esquema, arquivos ausentes e gravação dos
  artefatos com dados sintéticos pequenos.

## Critérios de aceite

- `make report` compara período estável com período driftado e falha com
  mensagem genérica quando os insumos ou esquemas forem inválidos.
- O HTML é criado pelo Evidently e o CSV contém os resultados por feature e
  método, sem linhas de clientes, IDs ou valores individuais de renda/crédito.
- A suíte de testes e `make lint` passam; README e checklist refletem o estado
  implementado.
