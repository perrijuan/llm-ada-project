# Prompts médicos: fontes e desenho do experimento

Consulta em 14/10/2026. Os prompts desta revisão são **adaptações**, não transcrições
integrais e não prompts validados pelos autores para estes quatro modelos.

## Fontes consultadas

1. [MedAgents — prompt_generator.py](https://github.com/gersteinlab/MedAgents/blob/main/prompt_generator.py).
   O código público define papéis de especialistas por domínio médico e instruções
   de saída para questões de múltipla escolha. Inspirou a persona de clínico
   generalista; não incorporamos a arquitetura multiagente nem suas análises longas.

2. [OpenBioLLM — model card oficial](https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B).
   O exemplo de uso atribui expertise em saúde e biomedicina ao assistente.
   Inspirou a persona de pesquisador biomédico; retiramos identidade de modelo,
   alegações de experiência pessoal e pedidos de explicações extensas.

3. [Tutorial solicitado — DataCamp](https://www.datacamp.com/tutorial/fine-tuning-qwen3-6).
   Serve como referência para o fluxo de inferência do checkpoint original,
   formatação de mensagens e treinamento separado. A tarefa é Q&A com MedQuAD.
   Mantivemos a instrução curta de resposta já presente no ZIP, mas não substituímos
   os modelos e hiperparâmetros recebidos pelos do tutorial.

4. [Microsoft — Medprompt / promptbase](https://github.com/microsoft/promptbase).
   Referência conceitual de demonstrações dinâmicas oriundas do treino.
   Nosso código faz sorteio reprodutível; não implementa a seleção semântica,
   geração de raciocínios ou votação do Medprompt completo.

## Templates disponíveis

Os textos exatos estão em `prompts_prontos.json` e são definidos em
`prompts_medicos.py`. Estão em inglês para acompanhar os datasets e o notebook.

- **clinico:** perspectiva de médico generalista, conhecimento clínico pertinente,
  sem inventar informações do paciente; resposta curta ou apenas a alternativa.
- **biomedico:** perspectiva de pesquisador biomédico, distinção entre evidência
  e incerteza, sem inventar estudos; mesmo formato de saída da tarefa.
- **original:** instruções existentes no ZIP, para comparação com a versão recebida.

Não há troca automática de persona entre perguntas ou entre 0/3/5-shot.
Não há três ou cinco personas: **shots contam demonstrações resolvidas**.

## Estrutura das mensagens

```text
system: persona escolhida + instrução da tarefa
user: pergunta do treino 1
assistant: resposta de referência do treino 1
... repetir até completar 3 ou 5 demonstrações ...
user: pergunta do teste (sem sua resposta)
```

No zero-shot e na avaliação fine-tuned, as mensagens intermediárias não existem.
No treino supervisionado, cada conversa contém instrução, pergunta do treino e
sua resposta. Não se inclui o teste no treinamento.

A persona é aplicada igualmente ao zero-shot e aos few-shot, evitando atribuir
ao número de exemplos uma mudança que veio de instruções diferentes.
Logo, esta versão com persona não é numericamente idêntica ao zero-shot antigo.
Para reproduzir suas instruções, selecione `--persona original`.

O banco de demonstrações MedQuAD usa perguntas filtradas externas aos 192
registros de SFT e aos 48 de avaliação; MedQA utiliza seu split oficial de
treino após excluir perguntas do teste; PubMedQA utiliza o restante do conjunto
rotulado após reservar as perguntas de avaliação. Os bancos de benchmark são
demonstrações da tarefa, não dados adicionais de fine-tuning.

O paper citado pelo usuário não foi anexado e sua tabela não foi verificada
independentemente. Seus números não são tratados como resultados deste código.

