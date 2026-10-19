# Guia: como criar um plugin de métricas

Este guia explica o pacote `metricas/` e o passo a passo para adicionar uma métrica nova
sem mexer no resto do repositório. Os comandos assumem que você está na raiz do repo.

## 1. Arquitetura

```
rodar_metricas.py  ──>  metricas/loader.py  ──>  importa cada módulo de --plugins e chama criar()
        │
        └────────────>  metricas/runner.py  ──>  para cada <campanha>/<modelo>/run_NN com status "ok":
                                                   monta um RunContext e grava <run>/metricas_<nome>.csv
consolidar.py  ──────>  junta os metricas_<nome>.csv de todas as runs em consolidado/metricas_<nome>.csv
```

| Arquivo | O que faz |
|---|---|
| `metricas/base.py` | Define o contrato `MetricaPlugin`, os escopos válidos (`ESCOPOS = ('pergunta', 'run')`) e a dataclass `RunContext`. |
| `metricas/loader.py` | `carregar_plugins(modulos)` importa os módulos, chama `criar()` e confere o contrato. Devolve `(plugins, erros)`. |
| `metricas/runner.py` | `executar(campanha, plugins)` percorre `*/run_*`, pula runs sem `status=ok`, aplica cada plugin e grava o CSV. Se um plugin falha numa run, os outros continuam. |
| `metricas/plugins/desempenho.py` | Builtin (escopo `run`): média/p50/p95 de latência e vazão por regime + `train_runtime` e `training_loss`. |
| `metricas/plugins/bert_score.py` | Exemplo real com lib externa (escopo `pergunta`, `pip install bert-score`). |
| `metricas/plugins/template.py` | Ponto de partida comentado (PASSO 0 a 7) com autoteste. |

Você **não** precisa editar `base.py`, `loader.py` nem `runner.py`: o plugin só precisa seguir o contrato.

## 2. Contrato `MetricaPlugin`

Um plugin é um módulo Python importável que expõe `criar()`. `criar()` devolve um objeto com:

| Membro | Tipo | Descrição |
|---|---|---|
| `nome` | `str` | Vira o arquivo `metricas_<nome>.csv`. Use letras, números e `_`. |
| `escopo` | `str` | `"pergunta"` ou `"run"`. |
| `avaliar_pergunta(row)` | `dict` | Escopo `pergunta`: chamado 1 vez por linha de `answers_per_question.csv`. |
| `avaliar_run(run)` | `dict` ou `list[dict]` | Escopo `run`: chamado 1 vez por run. Cada dict vira uma linha do CSV. |

Os dois métodos precisam existir; o que não é usado pelo escopo escolhido pode só devolver `{}`.

**`row` (escopo `pergunta`)** é um dict com as colunas do CSV, **todas como texto** (`str`):
`regime` (`zero-shot`, `3-shot`, `5-shot`, `fine-tuned`), `question`, `prediction`, `reference`,
`latency_s`, `ttft_s`, `tokens_per_s`, `decode_tokens_per_s`, `new_tokens`, `token_f1`, `rouge_l`, ...
O runner acrescenta sozinho `pergunta_idx`, `regime` e `question` em cada linha de saída.

**`run` (escopo `run`)** é um `RunContext`:

| Campo | Conteúdo |
|---|---|
| `pasta` | `Path` da run (ex.: `campanha_clinico/gemma4/run_01`) |
| `modelo` | nome curto (`gemma4`, `huatuogpt`, `medgemma`, `qwen35`) |
| `modelo_id` | ID do Hugging Face (vem de `registros/model_config.json` quando `ambiente.json` não tem) |
| `status` | dict do `status.json` (`seed`, `repeticao`, `persona`, ...) |
| `respostas` | lista com as linhas de todos os `answers_per_question.csv` da run |
| `treino` | `{tag: última linha de registros/treino_<tag>_metricas.csv}` (hoje só `fine-tuned`) |

Exemplo mínimo:

```python
class MinhaMetrica:
    nome = 'minha_metrica'
    escopo = 'pergunta'

    def avaliar_pergunta(self, row):
        return {'n_chars': len(row.get('prediction') or '')}

    def avaliar_run(self, run):
        return {}


def criar():
    return MinhaMetrica()
```

## 3. Passo a passo para um plugin novo

1. **Copie o template**
   ```bash
   cp metricas/plugins/template.py metricas/plugins/minha_metrica.py
   ```
   (Pode ficar em qualquer pacote seu, desde que esteja no `PYTHONPATH`.)
2. **Imports (PASSO 1):** libs externas no topo do arquivo. Se faltarem, o loader avisa
   `dependencia 'x' nao instalada`.
3. **Nome (PASSO 2):** troque `nome = 'template'` por um nome único, ex. `nome = 'minha_metrica'`.
4. **Escopo (PASSO 3):** `"pergunta"` para métricas por resposta (BERTScore, juiz LLM, similaridade);
   `"run"` para agregados (médias, tempos, loss).
5. **`__init__` (PASSO 4):** carregue aqui o que for pesado (modelo, tokenizer, cliente de API).
   `criar()` roda uma vez só, então nada é recarregado a cada pergunta.
6. **Implemente o método do seu escopo (PASSO 5 ou 6)** e deixe o outro devolvendo `{}`.
   Converta números com `float()` (os valores de `row` são texto).
7. **Renomeie a classe e ajuste `criar()` (PASSO 7).** Parâmetros podem vir de variável de ambiente,
   ex.: `os.getenv('MINHA_METRICA_LANG', 'en')`.
8. **Ajuste o autoteste** no bloco `if __name__ == '__main__':` se precisar de outras colunas na `row` fake.

## 4. Comandos de teste

```bash
# 1) Autoteste do plugin com dados fake (não precisa de campanha nem GPU)
python -m metricas.plugins.minha_metrica

# 2) Rodar na campanha (pode juntar com outros plugins)
python rodar_metricas.py campanha_clinico --plugins metricas.plugins.minha_metrica
python rodar_metricas.py campanha_clinico --plugins metricas.plugins.desempenho metricas.plugins.minha_metrica

# 3) Juntar todas as runs em consolidado/metricas_minha_metrica.csv
python consolidar.py campanha_clinico

# 4) Testes do pacote continuam passando
python -m unittest discover -s testes -v
```

Saída esperada do `rodar_metricas.py`: uma linha por run, por exemplo
`gemma4/run_01: ok=['minha_metrica']`. Runs sem `status=ok` aparecem como `ignorada`.
O comando termina com código 1 se algum plugin falhou em alguma run.

Plugin fora do repositório (módulo `meu_pacote.minha_metrica`):

```bash
PYTHONPATH=/caminho/do/meu_pacote_pai python rodar_metricas.py campanha_clinico --plugins meu_pacote.minha_metrica
# Windows (PowerShell): $env:PYTHONPATH="C:\caminho\do\meu_pacote_pai"
```

## 5. Erros comuns

| Mensagem / sintoma | Causa | Como resolver |
|---|---|---|
| `modulo nao encontrado (confira o nome e o PYTHONPATH)` | Nome do módulo errado ou fora do `PYTHONPATH`. | Use o caminho com pontos (`metricas.plugins.minha_metrica`, sem `.py`) e rode da raiz do repo. |
| `dependencia 'x' nao instalada` | Lib externa importada no topo não está instalada. | `pip install x`, ou `--ignorar-faltantes` para seguir com os outros plugins. |
| `Plugins indisponiveis. Instale as dependencias ou use --ignorar-faltantes.` | Algum plugin não carregou; sem a opção o CLI para antes de rodar. | Corrija o erro listado acima dessa linha ou use `--ignorar-faltantes`. |
| `nao expoe criar() -> MetricaPlugin` | O módulo não tem a função `criar()`. | Adicione `def criar(): return MinhaMetrica()`. |
| `criar() falhou (...)` | Erro no `__init__` (ex.: modelo não baixou). | Rode o autoteste para ver o traceback completo. |
| `plugin precisa de nome, escopo em ('pergunta', 'run'), avaliar_pergunta e avaliar_run` | Falta um membro do contrato ou o escopo está escrito errado. | Confira os quatro membros; o escopo é exatamente `'pergunta'` ou `'run'`. |
| `falhou minha_metrica: ValueError: could not convert string to float: ''` | Coluna vazia na `row` (tudo chega como texto). | Trate vazio antes do `float()` (veja `numero()` em `desempenho.py`). |
| `falhou minha_metrica: TypeError ...` (ou `AttributeError`) em `avaliar_run` | Retorno que não é `dict` nem lista de dicts. | Devolva `{coluna: valor}` ou `[{...}, {...}]`. |
| `metricas_<nome>.csv` não aparece | O plugin devolveu lista vazia (nada é gravado) ou a run foi `ignorada`. | Confira se a run tem `status=ok` e se o método do escopo certo devolve linhas. |
| CSV de um plugin some/é trocado por outro | Dois plugins com o mesmo `nome` gravam o mesmo arquivo. | Use um `nome` único por plugin. |
| Colunas `regime`/`question` com valor estranho | O dict do plugin usou as chaves `pergunta_idx`, `regime` ou `question` e sobrescreveu as do runner. | Não repita essas chaves no retorno de `avaliar_pergunta`. |
| Execução muito lenta | Modelo carregado dentro de `avaliar_pergunta`. | Carregue uma vez no `__init__` (PASSO 4). |

Observações:

- Rodar de novo sobrescreve `metricas_<nome>.csv` da run.
- Valores `list`/`dict` no retorno viram JSON dentro da célula do CSV.
- Hoje o pico de VRAM da geração fica em `registros/summary_df.csv` (`gen_peak_vram_gb`), não em
  `answers_per_question.csv`; por isso `peak_vram_gb` não aparece no `desempenho`.
