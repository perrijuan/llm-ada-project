# Histórico do protocolo anterior

Não descreve os scripts atuais. Para o protocolo vigente, consulte RELATORIO.md na raiz.

# Validação dos notebooks e execução repetida de fine tuning

Relatório técnico e guia de estudo • 13 de outubro de 2026

## Conclusão da revisão

Os quatro notebooks de fine-tuning seguem uma estrutura coerente de experimento inicial: carregar um modelo em quatro bits, adicionar adapters LoRA, preparar pares de perguntas e respostas, treinar com SFT e comparar antes e depois. Eles são uma base de aprendizado, mas as métricas e o desenho experimental ainda não permitem afirmar que um modelo é clinicamente melhor ou que uma família é superior às demais.

A entrega preserva essa estrutura em quatro scripts separados e diretos. As fórmulas das métricas e os hiperparâmetros originais foram mantidos. A alteração de configuração solicitada foi explicitar `max_steps=-1`. O restante das adições registra etapas, progresso, métricas, dados utilizados e arquivos de saída. Sugestões metodológicas deste relatório não foram aplicadas silenciosamente aos modelos.

A validação realizada é estática, documental e funcional da automação sem GPU. Não houve novo treinamento dos LLMs nesta revisão. Não foram produzidas acurácias ou tempos de inferência fictícios. O piloto no ambiente NVIDIA é a etapa necessária para confirmar compatibilidade e execução completa.

## Material recebido e comparação dos arquivos

O ZIP contém seis notebooks e quatro arquivos de texto. Quatro notebooks fazem fine-tuning; um compara resultados já salvos; outro demonstra inferência com imagem. Os textos contêm exemplos de respostas, não configurações suficientes para reconstruir os respectivos treinamentos.

| Arquivo de treino | Modelo carregado | Treino inicial | Benchmark final | Seção extra |
|---|---|---|---|---|
| gemma4_e4b_medquad_unsloth.ipynb | unsloth/gemma-4-E4B-it | 216 exemplos e 1 época | Velocidade, MedQA e PubMedQA | Zero-shot, 3x e 5x |
| huatuogpt_o1_7b_medquad_unsloth.ipynb | FreedomIntelligence/HuatuoGPT-o1-7B | 216 exemplos e 1 época | Velocidade, MedQA e PubMedQA | Nenhuma |
| medgemma15_4b_medquad_unsloth.ipynb | unsloth/medgemma-1.5-4b-it | 216 exemplos e 1 época | Velocidade, MedQA e PubMedQA | Nenhuma |
| qwen35_4b_medquad_unsloth.ipynb | unsloth/Qwen3.5-4B | 216 exemplos e 1 época | Velocidade, MedQA e PubMedQA | Nenhuma |

`medgemma.ipynb` usa `google/medgemma-1.5-4b-it`, uma radiografia e geração de até 2.000 tokens. Não faz LoRA nem treino e, por isso, não entra como uma quinta família de treinamento. `benchmark_comparison.ipynb` lê CSVs, calcula rankings e desenha gráficos; também não treina. `phi 4.txt` só contém respostas. Não foi inventado um script de Phi-4 sem o notebook correspondente.

As páginas dos quatro IDs de modelos foram consultadas e estão listadas nas fontes. A identidade do checkpoint importa mais que o rótulo informal de família. HuatuoGPT-o1-7B deriva de Qwen2.5-7B e usa um formato de raciocínio próprio; tratá-lo como mais uma variante de Qwen3.5 seria incorreto. O model card documenta as seções `## Thinking` e `## Final Response`. [HuatuoGPT-o1-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B)

MedGemma 1.5 4B é um modelo multimodal especializado, baseado em Gemma 3. Neste experimento ele recebe apenas texto e as camadas visuais não são ajustadas. Portanto, os resultados não avaliam a capacidade de interpretação de imagens. Os números publicados no model card usam outros protocolos e não são uma meta diretamente comparável à amostra de 100 questões utilizada aqui. [Model card oficial do MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)

## Como funcionam SFT LoRA QLoRA e Unsloth

SFT é treinamento supervisionado com exemplos de entrada e saída. Neste projeto, a pergunta médica é a entrada e a resposta do MedQuAD é o texto que o modelo aprende a prever. Durante o treino, a loss penaliza atribuir baixa probabilidade aos tokens corretos. Ela é diferente da acurácia de uma questão de múltipla escolha.

LoRA adapta algumas transformações do modelo usando matrizes pequenas treináveis. O modelo principal fica congelado e os adapters aprendem ajustes. `r=8` controla o rank dessas matrizes; `lora_alpha=16` participa da escala do ajuste. Não significam oito épocas nem dezesseis exemplos.

QLoRA combina adapters com um modelo base quantizado. O artigo original propõe NF4, dupla quantização e otimizadores paginados para reduzir memória. Os notebooks usam `load_in_4bit=True` e LoRA: isso segue a ideia central de adaptação de uma base quantizada. Entretanto, `optim="adamw_8bit"` não é sinônimo de otimizador paginado. O código também não declara explicitamente todas as opções de quantização do artigo. A configuração efetivamente carregada deve ser registrada, sem atribuir a ela propriedades não verificadas. [Artigo QLoRA](https://arxiv.org/abs/2305.14314)

Unsloth é a biblioteca que fornece carregamento, preparação dos adapters e otimizações neste fluxo. QLoRA é uma técnica; Unsloth é uma ferramenta que pode implementá-la. TRL fornece `SFTTrainer` e `SFTConfig`, e Transformers fornece parte da infraestrutura. A combinação exata de versões influencia as APIs disponíveis.

O guia de PEFT mostra como declarar NF4, dupla quantização e dtype de computação em uma implementação explícita. Ele ajuda a entender o que `load_in_4bit` não descreve sozinho. Não substituí o carregador Unsloth por esse caminho, pois isso mudaria seu tutorial. [PEFT e quantização](https://huggingface.co/docs/peft/developer_guides/quantization)

## Dados épocas e steps no seu código

O código filtra perguntas menores que 12 caracteres, respostas menores que 40 ou maiores que 900 caracteres e alguns prefixos de respostas pouco informativas. Em seguida, embaralha com seed 42, seleciona 240 linhas e reserva 24 para avaliação. Restam 216 para o treino inicial.

Caracteres não são tokens. Uma resposta com 900 caracteres pode ocupar quantidades diferentes de tokens em tokenizadores distintos. Por isso, o filtro de caracteres não garante que o texto formatado caiba em `MAX_SEQ_LENGTH=768`.

| Conceito | Significado neste projeto |
|---|---|
| Exemplo | Um par pergunta e resposta |
| Batch por dispositivo | 2 exemplos processados por GPU em cada microbatch |
| Acumulação de gradientes | 2 microbatches antes da atualização dos pesos |
| Batch efetivo em uma GPU | 2 × 2 = 4 exemplos por atualização |
| Época | Uma passagem pelo conjunto de treino |
| Step do otimizador | Uma atualização dos pesos dos adapters |
| Repetição ou execução | Rodar novamente o script completo desde o carregamento do modelo |
| Turno registrado | Aqui é identificado por execução, condição ou regime e pergunta; não é necessariamente um step |

Em uma GPU, com estes conjuntos divisíveis por quatro, o treino inicial tem 216 ÷ 4 × 1 = 54 atualizações. O Trainer também registra `global_step`, que é a contagem efetivamente observada. Várias GPUs ou mudanças no tratamento do último batch podem alterar essa relação; o pacote foi pensado para execução sequencial em uma GPU.

### O significado de zero shot 3x e 5x

Esses nomes aparecem apenas no notebook de Gemma. `REGIMES = [("zero-shot", 0), ("3x", 3), ("5x", 5)]` controla o tamanho do conjunto, enquanto `EXTENDED_EPOCHS=3` controla as épocas dos regimes treinados.

| Etapa | Exemplos de treino | Épocas | Exposições a exemplos | Atualizações esperadas |
|---|---:|---:|---:|---:|
| Treino inicial | 216 | 1 | 216 | 54 |
| Zero-shot da seção estendida | 0 | 0 | 0 | 0 |
| 3x | 648 | 3 | 1.944 | 486 |
| 5x | 1.080 | 3 | 3.240 | 810 |

O código acrescenta 432 linhas ao conjunto inicial para formar 648, e acrescenta 864 para formar 1.080. Ele não repete três ou cinco vezes exatamente as mesmas 216 linhas. As linhas vêm depois das primeiras 240 no mesmo embaralhamento, evitando sobreposição de índices com a avaliação. Isso não constitui uma garantia de ausência de perguntas duplicadas por conteúdo.

Zero-shot significa, nesta seção, avaliar sem seu novo treinamento em MedQuAD. O checkpoint já tem treinamento anterior dos autores. O código ainda cria adapters novos, mas não os treina nesse regime. A expectativa usual de LoRA é que a inicialização preserve a saída da base; isso depende da implementação e deve ser conferido no ambiente.

Cada regime carrega um modelo novo. O `5x` não continua o treino do `3x`. Ao final da seção estendida, o modelo é removido; o benchmark seguinte recarrega `OUTPUT_DIR`, onde ficou salvo o adapter do treino inicial de uma época. Assim, `fine-tuned` no benchmark final não significa o adapter `5x`.

Uma execução inteira do Gemma faz três treinamentos: inicial, 3x e 5x. Dez execuções do Gemma fazem trinta treinamentos, além das avaliações zero-shot. Com os outros três modelos, a campanha completa de quarenta execuções faz sessenta chamadas de treinamento. O cálculo esperado é 10 × (54 + 486 + 810) + 30 × 54 = 15.120 atualizações em uma GPU, se todas as execuções terminarem.

### O que foi feito com max steps

Todas as cinco construções de `SFTConfig` presentes nos quatro scripts agora declaram `max_steps=-1`. Esse valor deixa `num_train_epochs` controlar a duração em datasets finitos. Ele não transforma os steps realizados em números negativos, nem significa treinamento infinito. Um `max_steps` positivo teria precedência sobre as épocas. [Documentação de TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer)

Não alterei `warmup_steps=5`, `logging_steps=5` no treino inicial ou `logging_steps=10` na seção estendida. Esses parâmetros têm funções diferentes e não devem receber `-1` para controlar a duração. As épocas continuam em uma e três, conforme o original.

## Pontos de validação e suas consequências

### Evidência de execução incompleta

O notebook de Gemma guarda um `ImportError` na célula 6; o de Qwen guarda erro equivalente na célula 7. A mensagem informa falha ao carregar `_C` do PyTorch porque uma política de Controle de Aplicativo bloqueou o arquivo. Isso aponta para o ambiente, antes do treinamento, e não permite concluir que a arquitetura do notebook esteja errada.

Não há uma coleção completa de CSVs de benchmark no ZIP para recalcular rankings. Os exemplos textuais não informam todos os detalhes de versão, checkpoint e execução. Portanto, a revisão pode validar as fórmulas e identificar limitações, mas não comprovar numericamente os resultados de uma campanha anterior.

O ambiente do treinamento precisa conseguir importar Unsloth e PyTorch, reconhecer a GPU e acessar os checkpoints. O roteiro de instalação do Unsloth oferece caminhos para Windows, Docker e WSL. A solução para um bloqueio de política é um ambiente permitido ou ajuste pela administração responsável, não mudar a loss ou os steps. [Instalação oficial no Windows](https://unsloth.ai/docs/get-started/install/windows-installation)

### Seleção dos dados e risco de sobreposição

Separar 24 exemplos é correto como primeiro teste, mas é uma amostra pequena. O filtro favorece respostas curtas e remove só alguns textos pouco informativos; isso muda a população avaliada. O código não remove duplicatas por pergunta, não agrupa por doença e não separa documentos de origem. Perguntas parecidas sobre a mesma entidade podem aparecer em treino e avaliação mesmo com índices diferentes.

Como próxima versão científica, a recomendação é deduplicar, conferir sobreposição e separar grupos por origem ou entidade quando os metadados permitirem. Isso é uma proposta; não foi aplicada aos scripts preservados. O repositório original do MedQuAD descreve sua construção e metadados, e deve ser distinguido da versão convertida que os notebooks carregam. [MedQuAD original](https://github.com/abachaa/MedQuAD), [versão utilizada](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)

### O objetivo da loss precisa ser descrito corretamente

Os notebooks transformam a conversa inteira em uma coluna `text` e não configuram máscara explícita para treinar somente a resposta. No fluxo usual de language modeling do SFTTrainer, isso inclui tokens do prompt no objetivo. Não é automaticamente um erro: é uma escolha de treinamento. Porém, não se deve escrever no trabalho que a loss considera exclusivamente a resposta sem verificar as labels preparadas.

Treinar só respostas exige um formato e uma máscara compatíveis. A documentação de TRL explica `assistant_only_loss` e suas exigências de template. Não ativei essa opção porque mudaria o experimento e poderia exigir alterações específicas por família. [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

O `eval_dataset` local serve aos previews e benchmarks, mas não é passado ao Trainer inicial. Logo, o treino não produz uma curva de `eval_loss` de validação nem seleção de melhor checkpoint. `train_loss` menor não demonstra por si só melhor generalização.

### Templates e modelos com raciocínio

Os notebooks usam o template de chat de cada tokenizador e removem o BOS inicial antes de retokenizar. A intenção é evitar duplicação do token inicial. É necessário confirmar, no piloto, os tokens efetivos, encerramento de turno e compatibilidade da mensagem de sistema para cada checkpoint.

`enable_thinking=False` só funciona nos templates que reconhecem essa opção; não é um mecanismo universal para desativar raciocínio. HuatuoGPT possui extração específica de `## Final Response` nos previews, que permitem até 1.024 tokens. Já seu benchmark de velocidade usa 160 tokens e remove apenas blocos `<think>...</think>`. Isso pode medir uma resposta truncada ainda na fase de raciocínio.

O avaliador de múltipla escolha também força uma resposta imediata depois de `Answer:`. Ele mede seleção de alternativa sob esse formato, não a qualidade de raciocínio livre e prolongado. Mantenha isso explícito ao comparar modelos instrucionais e modelos de raciocínio. O artigo do HuatuoGPT descreve o objetivo distinto dessa família. [Relatório HuatuoGPT-o1](https://arxiv.org/abs/2412.18925)

### Compatibilidade dos tutoriais atuais

Os guias oficiais de Gemma 4 e Qwen3.5 mostram receitas e carregadores que variam por arquitetura e modalidade. A existência de um exemplo com outro loader não prova que `FastModel` no notebook esteja errado. A versão instalada e o checkpoint resolvido precisam ser testados juntos. Não misture argumentos de exemplos de versões diferentes do TRL durante uma mesma campanha. [Gemma 4 no Unsloth](https://unsloth.ai/docs/models/gemma-4/train), [Qwen3.5 no Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

## Dicionário e avaliação das métricas

### Benchmark comum aos quatro modelos

O teste de velocidade usa até 20 perguntas do conjunto de 24 exemplos MedQuAD. MedQA usa até 100 questões do split `test`; PubMedQA usa até 100 registros de `pqa_labeled`, split `train`. O nome `train` desse repositório não significa que o seu script o usa para fine-tuning: aqui ele só avalia. Porém, é preciso descrever esse subconjunto, e não afirmar que foi usado o protocolo oficial completo de teste do PubMedQA. [MedQA utilizado](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)

| Campo | Cálculo no benchmark comum | Interpretação |
|---|---|---|
| ttft_s | Instante do primeiro token menos início da requisição | Latência até o primeiro token, incluindo preparação do prompt |
| ttlt_s | Instante do último token menos início | Tempo até o último token observado pelo streamer |
| tokens_per_s | (N − 1) ÷ (TTLT − TTFT) | Velocidade de decodificação após o primeiro token |
| total_s | Fim do pós-processamento menos início | Tempo total observado, incluindo decodificação de texto |
| new_tokens | Quantidade de eventos de tokens no streamer | Tamanho gerado; pode incluir token especial |
| token_times | Lista de instantes relativos de emissão | Dados para a curva de tokens ao longo do tempo |
| correct | 1 se alternativa prevista coincide com a referência; senão 0 | Acerto por pergunta |
| mean | Média dos tempos, velocidades ou acertos na execução | Acurácia é a média de `correct` |
| ci_low e ci_high | Média ± 1,96 × desvio amostral ÷ raiz de n | Aproximação normal usada pelo notebook |
| n | Quantidade de perguntas daquela métrica | Não é o número de repetições do script |
| better | `higher` ou `lower` | Direção usada no ranking |

`precision` é o nome da categoria no arquivo, mas o cálculo é **accuracy ou acurácia**, isto é, acertos divididos pelo total. Não calcula precision de classificação, que exigiria uma definição de positivos e verdadeiros/falsos positivos. Preservei o nome da coluna para compatibilidade e explicitei o significado no relatório.

O classificador usa o maior logit entre a letra sem espaço e a letra com espaço. Ele não calcula a probabilidade de toda a alternativa. `encode(...)[-1]` também pressupõe que o token final representa adequadamente a letra. Se a codificação produzir vários tokens, esse atalho pode ser inadequado. Para uma revisão futura, valide as tokenizações ou pontue a sequência completa. Isso não foi trocado nesta entrega.

A acurácia em 100 questões tem resolução de um ponto percentual por acerto. Uma diferença de um ou dois acertos não sustenta uma conclusão forte. O gráfico de chance usa 25% para quatro alternativas e 33,3% para três; no PubMedQA desbalanceado, compare também com a classe majoritária. Os intervalos normais do notebook, mesmo limitados ao intervalo de zero a um, podem ser ruins perto de 0% ou 100%; Wilson ou bootstrap por questão são opções para um protocolo posterior.

Os tempos do streamer descrevem esta implementação local em batch 1. Não são um benchmark de servidor com concorrência. Embora a entrega de tokens para o host envolva sincronizações na implementação usual, o código não declara sincronização CUDA nas fronteiras de todo o cronômetro. Warm-up de oito tokens pode não estabilizar compilação e caches para todos os comprimentos. Hardware, carga concorrente, tokenizador e comprimento das respostas precisam acompanhar a comparação.

### Métricas exclusivas da seção estendida de Gemma

| Campo | Significado e unidade |
|---|---|
| train_examples e epochs | Linhas de treino e passagens pelo conjunto |
| train_runtime_min | Tempo retornado pelo Trainer dividido por 60 |
| train_loss | Loss final agregada do treinamento |
| train_samples_per_s | Throughput de amostras informado pelo Trainer |
| train_tokens_per_s | Soma estimada de tokens truncados × épocas ÷ runtime |
| train_peak_vram_gb | Pico de memória alocada pelo PyTorch no treino, em GiB apesar do nome GB |
| prompt_tokens e new_tokens | Tamanho tokenizado da entrada e da saída |
| latency_s | Duração da chamada de geração completa, com sincronização CUDA |
| ttft_s | Duração de uma outra chamada que gera apenas um token |
| tokens_per_s por pergunta | Novos tokens divididos pela latência completa |
| gen_tokens_per_s agregado | Soma dos novos tokens dividida pela soma das latências |
| decode_tokens_per_s | (N − 1) dividido pela diferença entre latência completa e chamada de um token |
| finished e finished_rate | Saída com menos tokens que o limite e média desse indicador |
| token_f1 | F1 de sobreposição de palavras entre resposta e referência |
| rouge_l | F1 baseado na maior subsequência comum de palavras |
| answer_words e reference_words | Contagem de palavras pela expressão regular do notebook |
| nll_sum e nll_tokens | NLL acumulada da referência e número de tokens pontuados |
| answer_nll | Soma das NLLs dividida pela soma de tokens pontuados |
| answer_ppl | Exponencial da NLL média |
| gen_peak_vram_gb | Pico de memória alocada durante geração e avaliação de NLL |

Token F1 usa palavras obtidas por `\w+` em letras minúsculas e considera multiplicidades via `Counter`. Se a interseção tem tamanho O, a precisão lexical é O/P, a revocação é O/R e F1 = 2pr/(p+r). Isso não é F1 dos tokens do tokenizador do LLM nem F1 de entidades clínicas.

ROUGE-L usa a maior subsequência comum e depois a mesma combinação harmônica de precisão e revocação. A implementação local não inclui todas as opções de bibliotecas externas, como stemming. Ela é consistente como métrica própria, mas deve ser descrita com essa normalização. O trabalho original apresenta a família ROUGE para comparação automática de textos; semelhança lexical não comprova verdade médica. [ROUGE de Lin](https://aclanthology.org/W04-1013/)

Os textos anexos ilustram essa limitação: há respostas com vocabulário semelhante à referência, mas que divergem dela quanto a dominante/recessivo; há também respostas repetitivas e exemplos que mudam o assunto da pergunta. Essa observação compara os textos fornecidos e suas referências, não constitui uma nova avaliação médica. Uma revisão humana com rubrica de correção, completude, contradição e repetição complementaria as métricas.

A NLL deve pontuar exatamente os tokens da resposta de referência, condicionados à pergunta. O código estima o corte pelo comprimento do prompt tokenizado separadamente. É preciso verificar que os IDs do prompt são de fato prefixo dos IDs da conversa completa; templates e tokenização na fronteira podem mudar isso. O cálculo também pode incluir encerramento de turno e truncar parte da resposta. Se não houver tokens avaliáveis, a proteção com denominador mínimo um pode produzir uma perplexidade enganosa igual a um. Registre e confira `nll_tokens`.

PPL = exp(NLL média). Quanto menor, maior a probabilidade atribuída à referência, dentro do mesmo protocolo. A tokenização altera a escala: não use PPL bruta de famílias com tokenizadores diferentes como ranking universal. Aqui a métrica existe apenas no Gemma. [Perplexidade no Transformers](https://huggingface.co/docs/transformers/perplexity)

Na seção estendida, o TTFT é aproximado por uma chamada separada que gera um token. Subtrair essa duração da geração completa combina medições distintas; a diferença pode ser negativa e virar NaN. Já no benchmark comum TTFT e TTLT vêm da mesma geração via streamer. Além disso, o `tokens_per_s` estendido inclui o primeiro token, enquanto o benchmark comum mede só decodificação posterior. **Essas colunas não representam a mesma medida e não devem ser misturadas.**

`finished = n_new < max_new_tokens` é só um indicador de não atingir o limite. Uma resposta pode terminar exatamente no último token permitido; o código a marcaria como não finalizada. A medição de VRAM usa memória alocada pelo PyTorch, não toda a memória reservada nem todo o consumo visto em `nvidia-smi`. Na avaliação estendida, esse pico inclui o forward usado para NLL, não apenas geração.

### Logs de treinamento

O callback novo exporta todos os campos que o Trainer disponibilizar em `state.log_history`. Isso pode incluir `step`, `epoch`, `loss`, `learning_rate`, `grad_norm`, `mean_token_accuracy`, `entropy`, tempo e throughput, dependendo da versão. O código não fabrica uma coluna quando a biblioteca não a produz. `TrainOutput.metrics`, `training_loss` e `global_step` são salvos separadamente.

A frequência dos logs foi preservada. Uma linha de loss a cada cinco steps não é uma medida isolada do quinto exemplo; geralmente representa a agregação do intervalo de log. Os registros em disco têm pequeno custo de tempo. Use a mesma instrumentação em todos os modelos e mantenha essa observação ao interpretar runtime.

## Como interpretar as dez repetições

O pacote mantém `SEED=42` e `do_sample=False`, conforme o pedido de preservar o tutorial. Isso é adequado para investigar repetibilidade e variação operacional na mesma configuração. A qualidade pode sair idêntica em várias execuções; isso não significa que a automação falhou.

Para medir variação causada pela aleatoriedade do treinamento, um protocolo diferente deveria variar a seed de inicialização dos adapters e do treinamento, mantendo fixa a seed de seleção dos dados. Neste código, `SEED` controla as duas coisas. Simplesmente trocar 42 por 43 também troca os dados de treino e avaliação, confundindo as causas da diferença. Não fiz essa alteração.

Fixar a seed não garante resultados idênticos entre versões, GPUs e operações não determinísticas. Registre ambiente, mantenha versões estáveis e use o mesmo conjunto de perguntas. [Reprodutibilidade no PyTorch](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

Há três níveis que não devem ser confundidos: uma pergunta, uma execução completa e o conjunto das dez execuções. Primeiro calcule a métrica por execução. Depois apresente as dez métricas, média e desvio padrão. Para avaliar o efeito do fine-tuning, calcule a diferença `fine-tuned − base` dentro da mesma execução antes de agregar essas diferenças.

`resumir.py` faz essa agregação e calcula um intervalo t para a média. Com dez observações, o valor crítico de 95% é aproximadamente 2,262, não 1,96. O intervalo depende de hipóteses sobre as observações; com seeds fixas e dependência temporal, sua interpretação é limitada à repetibilidade observada. Não trate 10 × 100 avaliações das mesmas perguntas como mil questões independentes. [Intervalos para médias no NIST](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)

Rodar dez vezes uma família e depois trocar atende ao fluxo solicitado, mas a ordem fixa pode confundir família com temperatura, caches, atualizações e carga do sistema. Mantenha a máquina estável e registre horário e GPU. Em um estudo posterior, alternar a ordem dos modelos por blocos reduziria esse efeito. Não foi aplicada aleatorização da ordem nesta entrega.

## Rankings e comparação entre famílias

O notebook `benchmark_comparison.ipynb` usa `pivot_table` para combinar métricas. Com múltiplas execuções, isso pode tirar a média de linhas sem manter a identidade de cada repetição. Seu `drop_duplicates()` ainda pode eliminar resultados realmente iguais. Por isso, a nova consolidação inclui modelo e número da execução e não remove repetições iguais.

A média dos ranks atribui pesos iguais às métricas e ignora o tamanho das diferenças e sua incerteza. Também inclui várias medidas correlacionadas de velocidade. O gráfico que usa a média de MedQA e PubMedQA dá peso igual aos dois benchmarks; ele não é uma medida clínica universal de precisão. Recomendo apresentar cada benchmark e seu delta separadamente, com as observações por execução.

Comparar `3x` com `5x` muda dados e quantidade de atualizações ao mesmo tempo. Mesmo usando três épocas nos dois, o `5x` tem mais trabalho de treinamento. Logo, não atribua uma melhoria só à quantidade de dados sem controlar o orçamento de treino em um experimento adicional. Compare os quatro modelos entre si usando o benchmark comum do treino inicial; trate a seção estendida do Gemma como um estudo separado.

## Estrutura e decisões da implementação

```text
experimento_llms/
  modelos/
    gemma4.py
    huatuogpt.py
    medgemma.py
    qwen35.py
  rodar_todos.py
  registro.py
  consolidar.py
  resumir.py
  requirements.txt
  README.md
  RELATORIO.md
  auditoria/
  originais/
  testes/
```

Cada arquivo de modelo contém seu código completo de treino e avaliação em sequência. `registro.py` só fornece funções pequenas para salvar CSVs, configurações e figuras; ele não contém um modelo genérico nem substitui as funções científicas dos notebooks. Mantenha a pasta completa ao copiar para o servidor, porque os scripts importam esse módulo auxiliar.

O fluxo de uma campanha é: escolher os modelos → escolher uma pasta nova → executar o primeiro modelo dez vezes → executar o próximo dez vezes → consolidar somente execuções concluídas → calcular os resumos. Cada repetição usa um subprocesso novo e uma pasta exclusiva. Isso evita continuar acidentalmente o adapter da repetição anterior e evita sobrescrever checkpoints ou CSVs.

O executor para no primeiro erro. Ele não considera uma falha como resultado zero e não a inclui nas médias. A retomada pula apenas execuções concluídas com o mesmo hash de código. Falhas e pastas preexistentes são preservadas. Para refazer uma campanha problemática, use uma pasta nova. A retomada não garante igualdade do ambiente externo: o hash não congela versões das bibliotecas, arquivos remotos ou drivers.

Os CSVs preservam respostas completas e tempos de tokens, mesmo quando o notebook mostrava um preview cortado. Gráficos são salvos sem abrir janelas, permitindo execução em servidor. Não foram acrescentados upload de modelos, chamadas a APIs pagas ou treino de Phi-4. `PUSH_TO_HUB` permanece falso.

`auditoria/*.diff` mostra as mudanças em relação às células originais. `manifesto.json` contém hashes dos notebooks e scripts. Um teste compara as árvores sintáticas de todas as células, removendo somente as adições de registro, progresso, `max_steps` e exibição de gráficos. Essa verificação confirma a preservação estrutural da lógica; não promete igualdade bit a bit de tempos ou resultados numéricos.

## Como executar cada modelo separadamente

No terminal, dentro de `experimento_llms`, após instalar o ambiente:

```bash
python rodar_todos.py --modelos gemma4 --repeticoes 10 --saida campanha_01
python rodar_todos.py --modelos huatuogpt --repeticoes 10 --saida campanha_01
python rodar_todos.py --modelos medgemma --repeticoes 10 --saida campanha_01
python rodar_todos.py --modelos qwen35 --repeticoes 10 --saida campanha_01
```

Execute um comando por vez. Para uma única execução direta, use `python modelos/medgemma.py`, por exemplo. Para automatizar todas as famílias, use `python rodar_todos.py --repeticoes 10 --saida campanha_01`. Não use a mesma pasta de campanha simultaneamente em dois terminais.

Antes das dez repetições, faça um piloto completo em outra pasta. Confirme que o log mostra treinamento, geração e benchmark, e que `status.json` termina com `ok`. Em seguida, congele o ambiente com `python -m pip freeze > ambiente_validado.txt`. Se faltar memória, o piloto vai revelar isso; reduções de batch, contexto ou quantidade de exemplos devem ser registradas como nova configuração experimental, pois deixariam de ser o tutorial preservado.

Depois da campanha, `python consolidar.py campanha_01` reconstrói os CSVs e `python resumir.py campanha_01` calcula os resumos. Leia primeiro `consolidado/execucoes.csv` para verificar quantas execuções realmente terminaram. Valores ausentes são ausentes; não os substitua por zero.

## Leitura orientada para o primeiro projeto

Comece pelo artigo QLoRA para entender por que quantização e adapters economizam memória. Em seguida, leia o guia PEFT para identificar as opções efetivas de quantização. Use o catálogo de notebooks do Unsloth para comparar um exemplo oficial da mesma família, sem copiar parâmetros de outra arquitetura por semelhança do nome. [QLoRA](https://arxiv.org/abs/2305.14314), [PEFT](https://huggingface.co/docs/peft/developer_guides/quantization), [notebooks oficiais Unsloth](https://unsloth.ai/docs/get-started/unsloth-notebooks)

Para compreender a preparação dos dados e a loss, consulte SFTTrainer. Para a parte específica de MedGemma, os notebooks mantidos pelo Google oferecem exemplos e o relatório técnico contextualiza as capacidades avaliadas. Eles são referências de implementação e protocolo, não evidência de que seu fine-tuning obteve os mesmos resultados. [Notebooks Google Health](https://github.com/Google-Health/medgemma/tree/main/notebooks), [MedGemma 1.5 Technical Report](https://arxiv.org/abs/2604.05081)

Finalize o estudo pelas referências de métricas e reprodutibilidade. Ao escrever seu trabalho, descreva modelo e revisão, dados e seleção, quantização efetiva, adapters, loss, épocas, batch, seeds, hardware, quantidade de repetições e a fórmula exata de cada medida. Essa descrição torna o experimento verificável mesmo quando o resultado é negativo ou o fine-tuning piora uma métrica.

## Fontes consultadas

Consulta realizada em 13 de outubro de 2026. Foram priorizados artigos, documentação oficial e páginas dos autores ou distribuidores dos modelos e datasets.

1. [Dettmers e colaboradores — QLoRA](https://arxiv.org/abs/2305.14314).
2. [Hugging Face PEFT — Quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
3. [Hugging Face TRL — SFTTrainer](https://huggingface.co/docs/trl/sft_trainer).
4. [Hugging Face Transformers — Trainer e TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer).
5. [Unsloth — Catálogo de notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
6. [Unsloth — Fine tuning de Gemma 4](https://unsloth.ai/docs/models/gemma-4/train).
7. [Unsloth — Fine tuning de Qwen3.5](https://unsloth.ai/docs/models/qwen3.5/fine-tune).
8. [Unsloth — Instalação no Windows](https://unsloth.ai/docs/get-started/install/windows-installation).
9. [Checkpoint Gemma 4 E4B](https://huggingface.co/unsloth/gemma-4-E4B-it).
10. [Checkpoint Qwen3.5 4B](https://huggingface.co/unsloth/Qwen3.5-4B).
11. [Checkpoint MedGemma 1.5 4B](https://huggingface.co/unsloth/medgemma-1.5-4b-it).
12. [FreedomIntelligence — HuatuoGPT-o1-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B).
13. [Chen e colaboradores — HuatuoGPT-o1](https://arxiv.org/abs/2412.18925).
14. [Google — MedGemma model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card).
15. [Sellergren e colaboradores — MedGemma 1.5 Technical Report](https://arxiv.org/abs/2604.05081).
16. [Google Health — Notebooks MedGemma](https://github.com/Google-Health/medgemma/tree/main/notebooks).
17. [Ben Abacha e Demner-Fushman — Repositório MedQuAD](https://github.com/abachaa/MedQuAD).
18. [MedQuAD — Conversão usada nos notebooks](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset).
19. [MedQA — Versão de quatro alternativas](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).
20. [PubMedQA — Dataset](https://huggingface.co/datasets/qiaojin/PubMedQA).
21. [Lin — ROUGE](https://aclanthology.org/W04-1013/).
22. [Transformers — Perplexity](https://huggingface.co/docs/transformers/perplexity).
23. [PyTorch — Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
24. [NIST — Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).
