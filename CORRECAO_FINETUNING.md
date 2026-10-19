# Correção do fine-tuning e continuidade da campanha — 15/10/2026

## Executar no ambiente Windows já utilizado

Extraia o ZIP em uma pasta nova. No terminal com `perri_dl` ativado, entre na
pasta `experimento_llms` extraída e execute primeiro uma repetição por modelo:

```powershell
python rodar_todos.py --repeticoes 1 --seeds 42 --persona clinico --saida campanha_ft_corrigido
```

Use uma pasta de saída nova: os bancos de demonstrações mudaram e o protocolo
agora é `icl_0_3_5_ft5_personas_v4_ft_separado`. Não misture com resultados antigos.
Para repetir o plano original, preserve suas opções de seeds e repetições.
`--planejar` mostra a ordem sem carregar GPU ou datasets.

## Erro corrigido nos quatro modelos

A chamada ao tokenizer recebia diretamente `formatted['text']`. Em versões do
Datasets que retornam `Column`, essa entrada não é aceita pelo tokenizer.
`textos_para_tokenizacao` materializa `list[str]` e rejeita dados vazios ou de
tipo inválido, indicando o índice. O SFTTrainer continua recebendo o Dataset
formatado completo; a lista é usada somente na contagem de tokens de treino.
Não convertemos valores inválidos para strings silenciosamente.

## Fine-tuning e few-shot são fluxos diferentes

- SFT: 192 registros MedQuAD, cada um com sistema, uma pergunta e sua resposta
  de referência. O formatter não chama a seleção de demonstrações nem depende
  de CURRENT_SHOTS. O modelo parte de um novo carregamento do checkpoint base.
- ICL: 3 ou 5 pares inseridos no prompt, sem atualização de pesos. O banco
  MedQuAD agora contém somente perguntas fora do treino e do teste, após limpeza
  e deduplicação. MedQA/PubMedQA continuam com seus bancos próprios de ICL.
- Avaliação: as mesmas 48 perguntas MedQuAD nas quatro condições. O modelo
  ajustado é avaliado com zero demonstrações. Nenhuma resposta de avaliação
  é usada no SFT ou no banco de demonstrações.
- Os CSVs `registros/dados_train_dataset.csv`, `dados_eval_dataset.csv` e
  `dados_medquad_demo_dataset.csv` permitem auditar essa separação.

A seleção dos 240 registros e o split 192/48 permanecem iguais para a mesma
split-seed; o que mudou foi a origem das demonstrações MedQuAD.

## Relação com o tutorial fornecido

Referência: PDF `Fine-tuning.pdf`, DataCamp, “Fine-tuning do Qwen3.6 em um dataset
de Q&A médico”, etapas 2–8, especialmente páginas 4, 7 e 9–12.
https://www.datacamp.com/pt/tutorial/fine-tuning-qwen3-6

O fluxo seguido é: limpar MedQuAD, separar treino/teste, carregar em 4 bits,
formatar conversa com chat template do modelo, treinar adapters LoRA usando
SFTConfig/SFTTrainer, salvar adapter/tokenizer localmente e avaliar nas mesmas
perguntas antes/depois. O cache é desativado para o treino e a quantidade de
parâmetros treináveis é impressa antes de `trainer.train()`.

Adaptações explícitas ao experimento:

| Configuração | Tutorial | Este experimento |
|---|---|---|
| Modelo/backend | Qwen3.6 via Transformers/PEFT | Quatro modelos existentes via Unsloth |
| Treino/teste | 216/24 | 192/48 |
| Épocas | 1 | 5 |
| Learning rate | 1e-4 | 2e-5 |
| Batch × acumulação | 2 × 2 | 2 × 16 |
| Otimizador | paged_adamw_8bit | adamw_torch |
| LoRA rank/alpha | 8/16 | 8/16 |
| LoRA dropout | 0.05 | 0, preservado do experimento |
| Comprimento SFT | 768 | 768 |

As camadas alvo e o checkpointing usam a configuração Unsloth já existente
para cada família. Não se adiciona um segundo adapter via `peft_config` ao
SFTTrainer: o modelo já foi preparado por `FastModel.get_peft_model`.
O upload ao Hugging Face mostrado no tutorial não faz parte desta execução.

## Tratamento de falhas

Em Python, o equivalente ao try/catch é `try/except`. O executor trata tanto
saída não zero do subprocesso quanto exceções ao executá-lo. Salva o traceback
em `execucao.log` e o estado em `status.json`, encerra o processo ainda ativo,
pula as demais repetições/seeds daquele modelo e passa ao próximo.

Uma falha não é transformada em sucesso: o comando retorna 1 ao final se algum
modelo falhou. Resultados parciais ficam na pasta original e são excluídos da
consolidação final. Ctrl+C continua encerrando a campanha. Erros de configuração,
pasta existente ou consolidação não são mascarados como falhas de modelo.
`--retomar` mantém a regra anterior: pula apenas execuções completas com código
e configuração idênticos; não retoma uma época nem sobrescreve execução falha.

## Verificação realizada

26 testes passaram em CPU, incluindo subprocessos reais com scripts artificiais
para verificar falha/continuidade, exceção de abertura de processo, isolamento
das perguntas, formato SFT independente de ICL e validação da coluna de textos.
As métricas e as regras de seed existentes também passaram nos testes.

Não foi executado treinamento CUDA, download de checkpoints, reload de adapter
ou inferência real. Os testes de tokenização validam o contrato de entrada com
uma coluna iterável simulada; não substituem o piloto com Transformers/Datasets
no ambiente `perri_dl`. O aviso P99 com n=20 permanece, pois é uma limitação
estatística legítima e não a causa do ValueError.

Os documentos em `auditoria/`, os notebooks em `originais/` e o guia PDF antigo
são registros históricos; esta nota e o código atual descrevem a correção.
