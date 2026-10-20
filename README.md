# Experimentos médicos: zero-shot, 3-shot, 5-shot e fine-tuning

Versão corrigida em 15/10/2026 a partir do ZIP recebido. Veja `CORRECAO_FINETUNING.md` para as mudanças e a relação com o tutorial. Os quatro scripts e as tarefas originais foram preservados.
O executor termina todas as seeds/repetições de um modelo antes de iniciar o próximo:
**Gemma4 → HuatuoGPT → MedGemma → Qwen3.5**. Cada execução usa um processo novo.

## Executar

No ambiente com Python, PyTorch CUDA, Unsloth, Transformers e TRL compatíveis com
os checkpoints, instale as dependências auxiliares:
```bash
python -m pip install -r requirements.txt
```
O arquivo requirements.txt recebido não instala sozinho a pilha de treinamento.
Siga a instalação oficial do Unsloth: https://unsloth.ai/docs/get-started/install
Para checkpoints restritos, configure o acesso no Hugging Face antes da execução.
Não atualize as bibliotecas durante uma campanha; após validar um piloto, salve
`python -m pip freeze > ambiente_validado.txt`.

Primeiro confira o plano (não baixa modelos e não utiliza GPU):
```bash
python rodar_todos.py --repeticoes 1 --seeds 42 123 2026 --persona clinico --planejar
```

Rode os quatro modelos, um por vez, com três seeds:
```bash
python rodar_todos.py --repeticoes 1 --seeds 42 123 2026 --persona clinico --saida campanha_clinico
```
São 12 processos e 12 fine-tunings: três execuções completas por modelo.
Em cada processo: zero-shot, 3-shot, 5-shot, fine-tuning e avaliação sem demonstrações.
As condições são braços independentes; a ordem é apenas operacional.

Para um piloto completo com apenas um modelo:
```bash
python rodar_todos.py --modelos medgemma --repeticoes 1 --seeds 42 --saida piloto
```

Para dez repetições com a mesma seed, como no pacote recebido:
```bash
python rodar_todos.py --repeticoes 10 --seeds 42 --saida repetibilidade
```
O padrão ainda é dez repetições por seed e seed 42. Use explicitamente
`--repeticoes 1` para executar uma vez por seed.

## Prompts prontos e exemplos

`prompts_medicos.py` contém as personas `clinico` (padrão), `biomedico` e
`original` (instruções do ZIP sem persona adicional). Os textos completos estão
também em `prompts_prontos.json`; fontes e adaptações em `PROMPTS_E_PROTOCOLO.md`.

A persona escolhida permanece igual no zero-shot, 3-shot, 5-shot e no braço de
fine-tuning. O formato de resposta também permanece igual dentro de cada tarefa.
Para trocar a persona, use outra campanha:
```bash
python rodar_todos.py --repeticoes 1 --seeds 42 123 2026 --persona biomedico --saida campanha_biomedico
```

3-shot e 5-shot significam exatamente três e cinco pares pergunta/resposta no
prompt do checkpoint original, sem adapters treinados neste experimento.
As demonstrações MedQuAD vêm das perguntas filtradas fora dos 192 registros de
fine-tuning e dos 48 de teste. MedQA e PubMedQA usam seus bancos separados de
demonstrações; nenhum desses exemplos é enviado ao SFTTrainer. A seleção muda por pergunta e seed.
Os três exemplos do 3-shot são os três primeiros da seleção de cinco, para uma
comparação controlada. As perguntas de teste são excluídas dos bancos.

## Seeds e treino

- `--split-seed 42`: mantém o subconjunto, treino/teste e perguntas de benchmark fixos.
- `--seeds 42 123 2026`: varia a seleção de demonstrações e a aleatoriedade do treino.
- Todas as famílias usam a mesma lista de seeds e os mesmos dados.
- Fine-tuning: novo carregamento do checkpoint original, LoRA, 5 épocas,
  AdamW, lr 2e-5, microbatch 2 × acumulação 16 = batch efetivo 32 em uma GPU.
- MedQuAD: 240 perguntas únicas filtradas, 192 de treino e 48 de teste.
- Comprimento de treino 768; contexto de inferência 16.384; prompt longo gera erro
  em vez de reduzir silenciosamente o número de demonstrações.

Variação entre seeds inclui ICL e aleatoriedade do treino; não mede variação do
split. Repetições da mesma seed não equivalem a seeds independentes.
Não há garantia de determinismo numérico em todas as GPUs.

## Resultados, falhas e retomada

Cada execução fica em `campanha_clinico/modelo/run_01/`, com log e status.
`campanha.json` fixa persona, seeds, split e repetições.
`ambiente.json` registra os prompts, fontes, versões e hiperparâmetros.
`registros/prompts.csv` registra mensagens, exemplos, seed, persona e condição.
Os CSVs de respostas, métricas e os adapters continuam nos diretórios originais.

O executor espera o processo terminar antes de iniciar o próximo. Se um modelo
falhar, registra o erro, preserva resultados parciais, pula as demais repetições/seeds
desse modelo e continua no próximo. Ao final retorna código 1 se houve falhas.
Ctrl+C interrompe a campanha, sem iniciar outro modelo. Não retoma o treinamento no meio de uma época.
Para pular somente execuções completas, use o mesmo comando com `--retomar`.
Mudanças de código, personas ou seeds exigem uma nova pasta de saída.
Uma trava impede dois executores na mesma campanha.

```bash
python consolidar.py campanha_clinico
python resumir.py campanha_clinico
```
A consolidação é automática após cada processo e inclui apenas execuções completas.
O resumo mantém a análise entre execuções do pacote original. Para estudar seeds,
prefira uma repetição por seed; não interprete repetições idênticas como amostras
independentes. As cópias `registro__*` não devem ser somadas aos CSVs originais.

## Métricas de velocidade ([METRICS])

O bloco impresso por condição tem este formato:
```text
[METRICS]
  prefill_ms:        12.34
  decode_mean_ms:    12.34
  decode_p95_ms:     12.34
  ttft_ms:           12.34
  overall_tok/s:     12.34
  peak_rss:          123456
  prompt_tokens:     123.4
  generated_tokens:  123.4
```

- `prefill_ms`: callback do streamer para o prompt até o primeiro token (forward do prompt + primeiro token).
- `decode_mean_ms` / `decode_p95_ms`: intervalo entre tokens consecutivos, agrupando os tokens de todas as perguntas da condição.
- `ttft_ms`: início de `generate` até o primeiro token; inclui a preparação.
- `overall_tok/s`: soma dos tokens gerados dividida pela soma do tempo total.
- `peak_rss`: pico de RAM do processo (`ru_maxrss` no Linux; `psutil` `peak_wset` no Windows se instalado; caso contrário `None`), incluindo o modelo carregado, não por geração.
- `prompt_tokens` / `generated_tokens`: média por pergunta.

As linhas são gravadas em `benchmark_summary.csv` com `category=speed`, `metric=<nome>` e o valor em `mean`.
O pico de VRAM continua em `gen_peak_vram_gb`. Como o código mudou, use uma nova pasta de saída.


## Métricas extras (plugins)

### Phi-4 e BERTScore na configuracao portuguesa do ADA

Depois que as runs terminarem, instale as dependencias opcionais no mesmo ambiente
Python e disponibilize o modelo `phi4` em um Ollama local. As implementacoes em
`metricas/avaliacao_pt/` foram copiadas do projeto ADA para manter esta copia
executavel sem depender de caminhos externos. O judge utiliza o prompt portugues,
temperatura zero e seed 42. O BERTScore usa `bert-base-multilingual-cased`, camada
9, `language="pt"`, sem IDF e sem reescala.

```bash
python -m pip install -r requirements-metricas-pt.txt
ollama pull phi4
python rodar_metricas.py campanha_clinico --plugins metricas.plugins.bert_score_pt
python rodar_metricas.py campanha_clinico --plugins metricas.plugins.phi4_judge_pt
python consolidar.py campanha_clinico
python resumir.py campanha_clinico
```

Os arquivos `metricas_bert_score_pt.csv` e `metricas_phi4_judge_pt.csv` guardam
uma linha por pergunta e condicao. O resumo gera `metricas_pt_por_execucao.csv`,
`resumo_metricas_pt.csv`, `deltas_metricas_pt.csv` e
`resumo_deltas_metricas_pt.csv` em `consolidado/`. Os deltas usam apenas perguntas
com nota valida tanto no zero-shot quanto na condicao comparada. Respostas vazias
ou falhas do avaliador ficam sem nota e com o motivo na coluna `erro`.

As respostas MedQuAD estao em ingles. Esta configuracao portuguesa avalia esses
textos sem traduzi-los; os resultados devem ser identificados como uma serie
distinta do BERTScore em ingles do plugin `bert_score` anterior. O Phi-4 depende
do Ollama em execucao, e sua trilha fica em `registros/phi4_judge_pt_chamadas.csv`
em cada run. Os plugins podem ser executados novamente sobre runs concluidas
sem repetir treino ou inferencia. Execute os dois comandos separadamente para
liberar a memoria do BERTScore antes que o Ollama carregue o Phi-4.

### Plugins existentes

Depois das runs, `rodar_metricas.py` calcula métricas a partir dos arquivos já salvos
(não usa GPU, salvo se o plugin precisar) e grava `metricas_<nome>.csv` em cada run com `status=ok`:
```bash
python rodar_metricas.py campanha_clinico                      # só o builtin desempenho
python rodar_metricas.py campanha_clinico --plugins metricas.plugins.desempenho metricas.plugins.bert_score
python rodar_metricas.py campanha_clinico --plugins metricas.plugins.bert_score --ignorar-faltantes
python consolidar.py campanha_clinico                          # junta em consolidado/metricas_<nome>.csv
```
Sem `--ignorar-faltantes`, um plugin que não importa (ex.: `bert_score` não instalado) para tudo
antes de rodar; com a opção, segue com os demais. O CLI lista por run quais plugins rodaram ou falharam.

O `aplicar_metricas.py` conserva o pacote historico de plugins e nao inclui os
novos modulos portugueses. Para aplicar esta integracao em outra copia, use os
arquivos da branch completa.

Passo a passo para criar um plugin novo (arquitetura, contrato, testes e erros comuns): [docs/GUIA_PLUGINS_METRICAS.md](docs/GUIA_PLUGINS_METRICAS.md).

Cada plugin é um módulo Python importável (`--plugins pacote.modulo`) com o contrato de `metricas/base.py`:

- `criar()`: função do módulo que devolve o plugin;
- `nome`: vira `metricas_<nome>.csv`;
- `escopo`: `"pergunta"` ou `"run"`;
- `avaliar_pergunta(row) -> dict`: chamado em cada linha de `answers_per_question.csv`
  (`regime`, `question`, `prediction`, `reference`, tempos...); o CSV recebe `pergunta_idx`, `regime`, `question` + o dict;
- `avaliar_run(run) -> dict` (ou lista de dicts): chamado uma vez por run com um `RunContext`
  (`pasta`, `modelo_id`, `modelo`, `status`, `respostas`, `treino`). `treino` é `{tag: linha de registros/treino_<tag>_metricas.csv}`,
  com `train_runtime` e `training_loss` do Trainer (hoje só a tag `fine-tuned`).

O método do escopo que não é usado pode devolver `{}`. Um modelo/módulo externo de métricas não precisa
mudar nada no repositório: basta expor essa interface e estar no `PYTHONPATH`.
Para criar um, copie `metricas/plugins/template.py`: ele é comentado passo a passo (PASSO 0 a 7), roda sem dependências
e tem autoteste (`python -m metricas.plugins.template`). `metricas/plugins/bert_score.py` é um exemplo real com lib externa
(`pip install bert-score`, gera `bert_p`, `bert_r`, `bert_f1`).
`metricas/plugins/desempenho.py` (builtin, escopo run) grava por regime `prefill_ms`, `decode_mean_ms`,
`decode_p95_ms`, `ttft_ms`, `overall_tok/s`, `peak_rss_bytes`, `prompt_tokens`, `generated_tokens`,
`train_runtime` e `training_loss`. Colunas ausentes ou vazias ficam em branco; o pico de VRAM continua
em `registros/summary_df.csv` (`gen_peak_vram_gb`).
Como `ambiente.json` ainda não grava o ID do Hugging Face, `modelo_id` vem de `registros/model_config.json` (`_name_or_path`) quando faltar.

## Escopo e validação

Isto adapta o protocolo de comparação informado pelo usuário aos experimentos
do ZIP: **MedQuAD (respostas abertas), MedQA e PubMedQA**.
Não reproduz a classificação de especialidades MedPT Top-5/10/20, não troca
Qwen3.5-4B por Qwen3-1.7B e não promete os F1 macro do paper.
O fine-tuning usa LoRA do pacote, não atualização de todos os pesos.
F1 por sobreposição textual e F1 macro de alternativas não são F1 de especialidade.

Os 12 testes disponiveis passaram sem GPU. Eles cobrem o runner de plugins,
consolidacao, resumos e falhas simuladas; nao exercitam treino ou inferencia real.
Não foram executados downloads, inferência ou treinamento de LLMs nesta revisão.

```bash
python -m unittest discover -s testes -v
```

`originais/` está intacto. `RELATORIO.md`, `RELATORIO.html` e os arquivos de
auditoria anteriores descrevem a versão recebida, inclusive sua seed fixa e
prompts antigos; são históricos. Para esta versão, consulte este README,
`PROMPTS_E_PROTOCOLO.md` e `auditoria/REVISAO_PERSONAS_V3.md`.

diagrama do projeto 

<img width="5280" height="4276" alt="ada - finetuning" src="https://github.com/user-attachments/assets/5cf697fb-ec8d-403e-84ec-5aedf3bad214" />


