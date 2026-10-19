# Experimento de zero shot few shot e fine tuning

Protocolo atualizado em 13 de outubro de 2026

## O que foi implementado

Os quatro scripts separados agora seguem a sequência aprovada: zero-shot, 3-shot, 5-shot e fine-tuning separado por cinco épocas. Não há treinamento antes do zero-shot. Não há multiplicação do dataset por três ou cinco. Cada condição carrega novamente o checkpoint original; somente a última adiciona adapters LoRA e executa o treinamento.

Os parâmetros de treinamento solicitados do trecho foram aplicados: learning rate 2e-5, AdamW, cinco épocas e batch efetivo 32. A seleção de 240 exemplos MedQuAD é dividida em 192 exemplos de treino e 48 de teste, reproduzindo a proporção 80/20. Foram mantidos os quatro modelos, o uso de Unsloth e quantização de quatro bits, rank 8 e alpha 16. Por isso, esta é uma adaptação das estratégias do artigo ao seu projeto, e não uma reprodução integral dos resultados publicados.

| Condição | Exemplos resolvidos no prompt | Treinamento nesta execução | Origem dos pesos |
|---|---:|---|---|
| zero-shot | 0 | Nenhum | Checkpoint original |
| 3-shot | 3 | Nenhum | Checkpoint original recarregado |
| 5-shot | 5 | Nenhum | Checkpoint original recarregado |
| fine-tuned | 0 na avaliação | 192 exemplos por 5 épocas | Checkpoint original e novos adapters |

Cada script contém seu próprio código completo de preparação dos dados, prompts, treinamento e métricas. O módulo `registro.py` só trata de registros e arquivos. Não é necessário usar o executor de todas as famílias para rodar um modelo individual.

## Por que a comparação do artigo faz sentido

O trecho fornecido compara duas formas de aproveitar dados rotulados. O in-context learning coloca exemplos na entrada e observa se o modelo consegue seguir o padrão sem atualizações de gradiente. O fine-tuning usa exemplos para ajustar parâmetros treináveis. O zero-shot é a referência sem essas duas intervenções adicionais.

No zero-shot, a distribuição prevista depende das instruções e da pergunta. No few-shot, ela também depende de pares de entrada e resposta exibidos anteriormente na mesma conversa. Esses pares informam ao modelo qual tarefa deve executar, como as classes são usadas e qual formato de resposta é esperado. Os pesos permanecem iguais: ao retirar os exemplos do prompt, essa informação específica deixa de estar no contexto. Essa distinção é documentada no trabalho sobre few-shot learning de Brown e colaboradores. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

Isso pode funcionar porque o pré-treinamento ensinou o modelo a identificar e continuar muitos padrões textuais. Não é uma garantia de melhora: exemplos irrelevantes, respostas incorretas, ordem desfavorável e prompts muito longos podem atrapalhar. O experimento precisa medir o efeito real em vez de pressupor que cinco exemplos serão melhores que três.

No fine-tuning, os exemplos contribuem para a loss e seus gradientes modificam os parâmetros treináveis. No seu projeto, esses parâmetros pertencem aos adapters LoRA, enquanto a base quantizada permanece congelada. O comportamento aprendido fica salvo no adapter e pode ser utilizado posteriormente sem mostrar exemplos no prompt. [QLoRA](https://arxiv.org/abs/2305.14314), [PEFT e quantização](https://huggingface.co/docs/peft/developer_guides/quantization)

Separar os métodos permite responder: quanto o modelo consegue fazer só com instruções; quanto melhora ao ver poucos exemplos; e quanto melhora com treinamento. Se o fine-tuned também recebesse cinco demonstrações, a comparação misturaria duas intervenções. Por isso, nesta versão ele é avaliado com zero demonstrações.

### O que o trecho afirma e o que é interpretação

O texto afirma que Qwen3 1.7B foi escolhido pelo equilíbrio entre custo e desempenho; que há recortes das cinco, dez e vinte especialidades mais frequentes; que treino e teste usam 80/20; que as demonstrações são amostradas do treino; e que o ajuste utiliza cinco épocas, AdamW, learning rate 2e-5 e batch 32. Ele também afirma que as médias macro procuram lidar com o desbalanceamento.

O trecho não explica uma busca de hiperparâmetros nem demonstra por que exatamente três e cinco exemplos, cinco épocas, batch 32 ou 2e-5 seriam valores ótimos. A interpretação metodológica plausível é que três e cinco testam dois níveis pequenos de contexto, e que uma taxa de aprendizado relativamente baixa limita a intensidade das alterações. Essas são explicações do desenho experimental, não justificativas comprovadas pelos autores no texto fornecido.

O learning rate controla a escala nominal das atualizações. Comparado ao 1e-4 anterior, 2e-5 é cinco vezes menor. Isso pode tornar o ajuste mais conservador, mas AdamW também usa momentos dos gradientes, o scheduler modifica a taxa ao longo do treino e o número de passos importa. Não se pode afirmar que os pesos mudarão exatamente cinco vezes menos nem que a qualidade aumentará.

Cinco épocas significam cinco passagens pelos mesmos exemplos de treino. Mais épocas podem facilitar a adaptação, mas também podem aumentar o sobreajuste. A avaliação em dados reservados verifica o resultado final. Para escolher épocas por desempenho seria necessário um conjunto de validação adicional, sem usar repetidamente o teste para decidir configurações.

AdamW é um otimizador adaptativo com decaimento de pesos desacoplado da atualização principal. O trecho o especifica, mas não fornece betas, epsilon, warm-up ou scheduler. Mantivemos warm-up de cinco passos, scheduler linear e weight decay 0,01 do tutorial; não atribuímos esses valores ao artigo.

### Como ler os resultados copiados

No recorte Top-5, o F1 macro informado passa de 0,60 no zero-shot para 0,86 no 3-shot, 0,87 no 5-shot e 0,98 no fine-tuned. Nos recortes Top-10 e Top-20, o padrão também mostra melhora com demonstrações e uma diferença maior para o modelo ajustado. O ganho adicional de três para cinco exemplos é pequeno nessa tabela.

Isso sugere que exemplos no contexto já ajudam a especificar a tarefa, e que o treinamento explora mais informação do dataset. A tabela sozinha não prova causalmente esse mecanismo, não fornece incerteza entre execuções e não garante significância da diferença de 0,01. Sem título, autores completos ou link do artigo, a referência deste protocolo é o trecho fornecido pelo usuário; não foi inventada uma citação bibliográfica.

Top-5, Top-10 e Top-20 se referem à quantidade de classes de especialidade, não ao número de demonstrações. Recortes com mais classes podem ser mais difíceis, mas também podem mudar a composição dos dados. Não atribua toda a diferença da tabela apenas à quantidade de classes.

## O que foi adaptado e o que não é igual ao artigo

| Elemento | Trecho fornecido | Implementação entregue |
|---|---|---|
| Modelo | Qwen3 1.7B | Quatro modelos dos notebooks, em arquivos separados |
| Tarefa principal | Classificar especialidade | Gerar respostas MedQuAD |
| Recortes | Top-5, Top-10 e Top-20 especialidades | Não implementados: dataset de especialidades não foi fornecido |
| Dados principais | Subdataset de cada recorte | 240 pares MedQuAD filtrados e únicos |
| Split principal | 80/20 | 192 treino e 48 teste |
| ICL | 0, 3 ou 5 demonstrações | 0, 3 ou 5 demonstrações de treino por pergunta |
| Fine-tuning | 5 épocas | 5 épocas sobre os 192 exemplos |
| Learning rate | 2e-5 | 2e-5 |
| Otimizador | AdamW | `adamw_torch` |
| Batch | 32 | Batch efetivo 32: microbatch 2 e acumulação 16 em uma GPU |
| Forma de ajuste | Não esclarecida no trecho | LoRA sobre base em quatro bits com Unsloth |
| Métricas | Precision, recall e F1 macro | Métricas originais e macro nas alternativas de MedQA/PubMedQA |

A acumulação evita carregar 32 exemplos simultaneamente. Em uma GPU, 2 × 16 = 32 exemplos contribuem para cada atualização. Isso preserva o batch efetivo desejado, mas não promete equivalência numérica exata com um batch físico de 32 por causa de arredondamento, normalização da loss e implementação. O executor é de uma GPU por processo. [Trainer e TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer)

Com 192 exemplos e batch efetivo 32, são esperadas seis atualizações por época e trinta em cinco épocas. O CSV registra o `global_step` realmente observado. `max_steps=-1` deixa as épocas controlarem a duração. `warmup_steps=5` e `logging_steps=10` continuam positivos porque controlam outras funções.

Mudar a tarefa principal para especialidades exigiria perguntas com os respectivos rótulos e a definição dos recortes Top-5/10/20. Essa informação não está no material entregue. Portanto, nenhum resultado gerado por estes scripts deve ser apresentado como reprodução da tabela de especialidades.

## Como os dados e demonstrações são separados

O código conserva os filtros básicos do tutorial: tamanho mínimo de pergunta e resposta, limite de 900 caracteres para respostas e exclusão de alguns prefixos genéricos. Agora também remove perguntas exatamente duplicadas após normalização de espaços e caixa antes do split. Isso evita colocar a mesma pergunta normalizada em treino e teste; não detecta todas as paráfrases nem garante separação por doença ou documento.

A seleção continua limitada a 240 pares para manter a escala inicial do projeto. Os 80% são 192 pares, não 80% de todo o repositório MedQuAD. O fine-tuning usa todos os 192 pares dessa seleção, e cada época passa por esse mesmo conjunto. As 48 perguntas de teste são fixas entre as quatro condições e os quatro modelos, desde que a revisão do dataset e o ambiente sejam iguais.

No MedQuAD, as demonstrações são perguntas e respostas dos 192 pares de treino. Para cada pergunta de teste, o código faz uma seleção pseudoaleatória determinística de cinco demonstrações. O 3-shot usa as três primeiras dessa seleção e o 5-shot usa todas as cinco, na mesma ordem. Isso permite atribuir a comparação ao acréscimo de duas demonstrações, em vez de trocar completamente o conjunto de exemplos.

O sorteio usa seed 42, nome da tarefa e pergunta; não depende da família do modelo ou da condição. Assim, há demonstrações diferentes para perguntas diferentes, mas a mesma pergunta recebe exemplos reproduzíveis entre condições, modelos e repetições. O trecho do artigo fala em amostragem dinâmica, mas não informa se há esse pareamento ou uma seed específica. O pareamento é uma decisão explícita desta adaptação.

Os benchmarks suplementares continuam com até 100 perguntas MedQA e 100 PubMedQA, além de até 20 perguntas MedQuAD para velocidade. MedQA recebe demonstrações do split oficial `train`, excluindo perguntas presentes no teste selecionado. PubMedQA usa uma seleção de 100 perguntas de `pqa_labeled/train` para avaliar; as demonstrações vêm dos registros restantes, excluindo perguntas repetidas. Isso mantém o benchmark local anterior e não equivale ao protocolo oficial completo do PubMedQA.

O fine-tuning é feito apenas em MedQuAD. Os bancos de demonstrações MedQA e PubMedQA são usados somente para ICL, não para ajustar pesos. Isso permite avaliar estratégias em cada tarefa, mas implica uma diferença importante: o ICL desses benchmarks vê exemplos do próprio benchmark, enquanto o ajuste é em MedQuAD. Os resultados não isolam apenas o método de adaptação com dados idênticos; são comparações práticas com fontes de supervisão distintas.

`registros/dados_demonstracoes_*.csv` salva os bancos disponíveis. `registros/prompts.csv` registra condição, tarefa, pergunta, IDs das demonstrações e mensagens completas. A última mensagem do usuário contém a pergunta de teste, sem a resposta. A referência só entra separadamente no cálculo supervisionado de NLL, não na geração da resposta.

## Comprimento do contexto e templates

O treino continua limitado a 768 tokens por exemplo. Para a inferência few-shot, `MODEL_CONTEXT_LENGTH=16384` permite incluir demonstrações completas. O script verifica o comprimento tokenizado mais a reserva de geração e interrompe se ultrapassar o limite, sem reduzir silenciosamente o número de exemplos. O limite é um orçamento da implementação, não uma declaração do comprimento nativo máximo de cada modelo.

Maior contexto pode consumir mais VRAM e aumentar TTFT. A capacidade efetiva depende de GPU, arquitetura e versão. Se o piloto apontar falta de memória ou um prompt maior que o limite, a nova configuração deve ser registrada e usada igualmente nas condições comparadas. Mudar só o 5-shot por conveniência prejudica a comparação.

Os templates de chat dos próprios modelos continuam sendo usados. `enable_thinking=False` não é uma opção universal: só tem efeito em templates compatíveis. HuatuoGPT conserva a extração de `## Final Response` e o limite de 1.024 tokens nos três previews, como no original. A avaliação principal e a de velocidade continuam com 160 tokens; podem truncar raciocínio nessa família. [HuatuoGPT-o1-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B)

Para NLL, a tokenização da conversa completa deve começar com os mesmos IDs do prompt. O código agora verifica essa condição e rejeita respostas sem tokens avaliáveis. Também usa o orçamento de inferência, evitando que os exemplos few-shot ocupem os 768 tokens e eliminem a resposta da medição. A NLL ainda pode incluir tokens de encerramento do template; isso deve acompanhar sua descrição metodológica.

## Métricas e interpretação

As métricas de respostas que antes existiam apenas no experimento estendido do Gemma foram aplicadas às quatro famílias, com as mesmas fórmulas de F1 lexical, ROUGE-L e agregação de NLL. As quatro condições recebem todas as avaliações. Não misture medições da campanha antiga com as desta versão: dados, treinamento e prompts mudaram.

| Arquivo ou campo | Significado |
|---|---|
| `benchmark_summary.csv` | Médias e intervalos originais de velocidade e acurácia; macro adicionada |
| `speed_per_question.csv` | TTFT, TTLT, tempo total, tokens/s, prompt_tokens, new_tokens e token_times |
| `precision_per_question.csv` | Alternativa prevista, referência e acerto por questão |
| `answers_per_question.csv` | Resposta MedQuAD, referência, tempos, F1, ROUGE-L, NLL e contagens |
| `metrics_summary.csv` | Agregação MedQuAD e métricas do treino por condição |
| `treino_fine-tuned_steps.csv` | Eventos do Trainer, inclusive loss e learning rate quando emitidos |
| `treino_fine-tuned_metricas.csv` | Métricas finais retornadas pelo Trainer, training_loss e global_step |
| `previews.csv` | Três exemplos completos por condição, sem corte no texto salvo |
| `prompts.csv` | Demonstrações e mensagens efetivamente construídas |

No benchmark com streamer, TTFT é o tempo até o primeiro token observado, TTLT até o último, `tokens_per_s=(N−1)/(TTLT−TTFT)` mede a fase de decodificação e `total_s` inclui a preparação e o pós-processamento. `prompt_tokens` ajuda a interpretar o aumento do contexto few-shot. O warm-up continua separado das observações.

No avaliador MedQuAD herdado do Gemma, TTFT é estimado por outra geração de apenas um token; `tokens_per_s=N/latency_s` inclui a geração completa. A subtração entre duas chamadas usada em `decode_tokens_per_s` é aproximada e pode resultar em NaN. Esses nomes semelhantes têm definições diferentes das medidas do streamer; seus CSVs permanecem separados.

Token F1 usa sobreposição de palavras normalizadas e contagem de multiplicidades. ROUGE-L usa uma maior subsequência comum de palavras. Ambos medem semelhança lexical, não correção médica. Respostas que trocam um termo crucial podem continuar com pontuação alta. [ROUGE](https://aclanthology.org/W04-1013/)

NLL é a soma da perda dos tokens da referência dividida pelo total desses tokens. A perplexidade é exp(NLL). Ela depende do tokenizador e não deve ser usada diretamente como ranking universal entre famílias diferentes. [Perplexidade no Transformers](https://huggingface.co/docs/transformers/perplexity)

`finished_rate` mantém a heurística original de saída menor que o limite de tokens: terminar exatamente no limite pode ser marcado como incompleto. Os picos de VRAM são de memória alocada pelo PyTorch, em GiB, apesar do sufixo `gb`; não representam toda a memória reservada ou todo o uso de `nvidia-smi`. O pico de avaliação inclui também o cálculo de NLL. O throughput de tokens de treino é estimado pela tokenização dos exemplos e não contabiliza necessariamente todo o padding.

### Precision recall e F1 macro

Para uma classe, precision é TP/(TP+FP), recall é TP/(TP+FN) e F1 é a média harmônica dos dois. Primeiro se calcula cada medida por classe; depois se tira a média simples entre classes. F1 macro é a média dos F1 individuais, não a média harmônica da precision macro com recall macro.

Dar o mesmo peso a cada classe reduz a dominância numérica das classes mais frequentes na pontuação. Isso não corrige o desbalanceamento do treinamento nem torna as estimativas das classes raras precisas. No código, denominadores zero recebem valor zero e todas as alternativas esperadas são incluídas na média.

Essas métricas foram acrescentadas a MedQA e PubMedQA porque suas saídas são rótulos discretos. No MedQA as classes são posições A/B/C/D, e sua média macro pode depender da distribuição das posições de resposta; ela não mede desempenho por especialidade. No PubMedQA A/B/C correspondem a yes/no/maybe. Não foi calculada uma macro de classes para texto livre MedQuAD, que não contém esse alvo de classificação.

A categoria antiga `precision` nos CSVs continua significando acurácia para preservar os nomes originais. As novas medidas ficam em `classification_macro`, com nomes explícitos. Os intervalos por pergunta para velocidade e acurácia mantêm a aproximação normal do notebook. Não foi fabricado um intervalo por pergunta para F1 macro: seus campos de intervalo ficam vazios; a variação entre execuções é analisada depois.

O classificador conserva o maior logit entre as variantes de cada letra, não a probabilidade de toda a alternativa. Isso mede escolha imediata sob um formato fixo, não raciocínio livre. A hipótese de token único deve ser conferida por modelo. Para um estudo posterior, pontuar sequências completas ou validar a geração da alternativa seria uma mudança metodológica adicional.

## Dez repetições e comparação justa

Cada repetição executa zero-shot, 3-shot, 5-shot e fine-tuned. São quatro carregamentos do checkpoint e um treinamento por repetição. Dez repetições de uma família produzem quarenta avaliações de condição e dez treinos. As quatro famílias somam quarenta repetições, cento e sessenta avaliações de condição e quarenta treinos; cada avaliação de condição inclui diversos exemplos e benchmarks.

A seed permanece 42 tanto na seleção dos dados como no treinamento e nas demonstrações. Essa decisão preserva a repetibilidade. As métricas de qualidade podem se repetir; as dez execuções não são dez amostras independentes de pacientes nem dez seeds de treinamento. Para estudar aleatoriedade de treino e seleção de demonstrações, essas seeds precisariam ser separadas em um protocolo posterior. Fixar seed também não garante igualdade entre GPUs e versões. [Reprodutibilidade PyTorch](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

`resumir.py` calcula médias e desvios entre execuções e diferenças pareadas de cada condição contra zero-shot. Seu intervalo t de 95% depende das hipóteses do método e descreve a repetibilidade observada, não substitui incerteza entre perguntas. Não trate repetir as mesmas cem questões dez vezes como avaliar mil questões independentes. [NIST sobre intervalos para médias](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)

A ordem entre famílias permanece em blocos de dez, conforme solicitado. Carga concorrente, temperatura e cache podem afetar tempo. Use a mesma GPU, evite atualizar bibliotecas entre repetições e guarde as versões. O registro em disco também tem custo: todos os modelos devem usar a mesma instrumentação.

## Como executar e localizar os resultados

Mantenha a pasta inteira. Instale PyTorch com CUDA e Unsloth de acordo com seu ambiente e depois `python -m pip install -r requirements.txt`. Os scripts não reinstalam bibliotecas em cada execução. Para checkpoints restritos, autentique previamente na Hugging Face ou use a variável de ambiente HF_TOKEN. [Instalação Unsloth](https://unsloth.ai/docs/get-started/install/windows-installation)

Dentro da pasta do pacote, faça primeiro um piloto completo:

```bash
python rodar_todos.py --modelos medgemma --repeticoes 1 --saida piloto_icl
```

Depois, para dez execuções de cada modelo separadamente:

```bash
python rodar_todos.py --modelos gemma4 --repeticoes 10 --saida campanha_icl
python rodar_todos.py --modelos huatuogpt --repeticoes 10 --saida campanha_icl
python rodar_todos.py --modelos medgemma --repeticoes 10 --saida campanha_icl
python rodar_todos.py --modelos qwen35 --repeticoes 10 --saida campanha_icl
```

Para todas as famílias automaticamente: `python rodar_todos.py --repeticoes 10 --saida campanha_icl`. Para rodar uma única vez diretamente: `python modelos/medgemma.py`. Nessa chamada direta os CSVs são salvos em pasta com data e hora; o log completo e a consolidação automática são funções do executor.

Os resultados ficam em `campanha_icl/modelo/run_01/`. Cada repetição tem log, status, ambiente, adapter, configuração de treino, dados e CSVs. Barras tqdm acompanham execuções e perguntas; o Trainer mantém sua barra de treino. A frequência de log é de dez atualizações, não uma linha por exemplo.

Somente execuções completas entram nos CSVs consolidados. Uma falha interrompe a campanha para não repetir um erro de ambiente dez vezes. Resultados parciais permanecem na pasta da execução. `--retomar` pula somente execuções completas com o mesmo código; não retoma o treino no meio do checkpoint. Use uma pasta nova para este protocolo e não misture os resultados antigos.

```bash
python consolidar.py campanha_icl
python resumir.py campanha_icl
python -m pip freeze > ambiente_validado.txt
```

## Validação e limites da entrega

Foram testados a ordem das condições e o treino exclusivo da última condição nos quatro scripts, parâmetros, construção dos prompts sem resposta do teste, exclusão de duplicatas no banco de demonstrações, pareamento 3/5-shot, fórmulas macro, isolamento de execução, CSV, retomada e interrupção por falha. Os testes usam objetos simulados e subprocessos leves; não treinam LLMs.

Os treinamentos e a compatibilidade dos quatro checkpoints com as versões instaladas precisam ser confirmados no piloto NVIDIA. O ambiente desta revisão não possui PyTorch e CUDA preparados para essa validação. Não foram produzidos resultados experimentais nem reproduzidos os números da tabela do artigo.

O relatório antigo registrava erros de importação do PyTorch salvos nos notebooks Gemma e Qwen. Os originais continuam intactos no pacote. Não há notebook de treino Phi-4 no material: o texto de exemplos não permite reconstrução fiel. O notebook `medgemma.ipynb` de imagem e o notebook de comparação continuam como originais, sem entrar na sequência de fine-tuning.

`auditoria/*.diff` e `manifesto.json` registram a transformação e hashes. Esta versão muda o protocolo de forma intencional; não é apresentada como código numericamente idêntico aos notebooks iniciais. O arquivo `REVISAO_INICIAL_HISTORICA.md` preserva a análise anterior apenas como histórico identificado.

## Referências para aprofundamento

O trecho Experimental Setup e sua tabela foram fornecidos pelo usuário. A ausência do título ou URL impede verificar detalhes além do texto copiado, como revisão do dataset, seeds e implementação exata. A explicação acima não atribui aos autores decisões não descritas.

1. [Brown e colaboradores — Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
2. [Dettmers e colaboradores — QLoRA](https://arxiv.org/abs/2305.14314).
3. [Hugging Face PEFT — Quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
4. [TRL — SFTTrainer e objetivos de treinamento](https://huggingface.co/docs/trl/sft_trainer).
5. [Transformers — Trainer e TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer).
6. [Unsloth — Notebooks oficiais](https://unsloth.ai/docs/get-started/unsloth-notebooks).
7. [Unsloth — Gemma 4](https://unsloth.ai/docs/models/gemma-4/train).
8. [Unsloth — Qwen3.5](https://unsloth.ai/docs/models/qwen3.5/fine-tune).
9. [Unsloth — Instalação Windows](https://unsloth.ai/docs/get-started/install/windows-installation).
10. [Google — MedGemma model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card).
11. [Google Health — Exemplos MedGemma](https://github.com/Google-Health/medgemma/tree/main/notebooks).
12. [MedGemma 1.5 Technical Report](https://arxiv.org/abs/2604.05081).
13. [HuatuoGPT-o1-7B model card](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B).
14. [HuatuoGPT-o1 Technical Report](https://arxiv.org/abs/2412.18925).
15. [MedQuAD original](https://github.com/abachaa/MedQuAD).
16. [MedQuAD convertido usado no código](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset).
17. [MedQA de quatro alternativas](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).
18. [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA).
19. [Lin — ROUGE](https://aclanthology.org/W04-1013/).
20. [Transformers — Perplexidade](https://huggingface.co/docs/transformers/perplexity).
21. [PyTorch — Reprodutibilidade](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
22. [NIST — Intervalos para médias](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).
