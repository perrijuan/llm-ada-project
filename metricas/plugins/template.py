"""TEMPLATE de plugin de metricas: copie este arquivo e preencha os PASSOS 1 a 7.

Como funciona (resumo):
  rodar_metricas.py importa o modulo indicado em --plugins, chama criar() e recebe um objeto
  com nome, escopo, avaliar_pergunta(row) e avaliar_run(run). O runner percorre cada
  <campanha>/<modelo>/run_NN com status "ok" e grava <run>/metricas_<nome>.csv.
  Depois, consolidar.py junta esse CSV de todas as runs em consolidado/metricas_<nome>.csv,
  com exec_modelo, exec_seed, exec_repeticao, exec_persona.

PASSO 0 - copiar:
  copie para metricas/plugins/<sua_metrica>.py (ou para qualquer pacote seu no PYTHONPATH).
  Nao edite base.py, loader.py nem runner.py: o plugin so precisa seguir este contrato.

Rodar e verificar (troque "template" pelo nome do seu arquivo):
  python -m metricas.plugins.template                     # autoteste com dados fake, sem campanha
  python rodar_metricas.py campanha_clinico --plugins metricas.plugins.template
  python consolidar.py campanha_clinico                   # gera consolidado/metricas_template.csv
  python -m unittest discover -s testes -v                # testes do pacote continuam passando
"""
# PASSO 1 - imports. Bibliotecas externas vao aqui no topo: se nao estiverem instaladas,
# o loader mostra "dependencia 'x' nao instalada" e o CLI para (ou pula com --ignorar-faltantes).
# Ex.: from bert_score import BERTScorer
from metricas.base import RunContext


class Template:
    # PASSO 2 - nome: vira o arquivo metricas_<nome>.csv. Use letras, numeros e _ (sem espacos).
    # Dois plugins com o mesmo nome sobrescrevem o mesmo CSV.
    nome = 'template'

    # PASSO 3 - escopo:
    #   "pergunta": avaliar_pergunta e chamado 1 vez por linha de answers_per_question.csv
    #               (48 perguntas x 4 regimes = 192 linhas por run). Use para metricas por resposta
    #               (BERTScore, juiz LLM, similaridade...).
    #   "run":      avaliar_run e chamado 1 vez por run. Use para agregados (medias, tempos, loss).
    escopo = 'pergunta'

    def __init__(self):
        # PASSO 4 - carregue aqui o que for pesado (modelo, tokenizer, cliente de API).
        # criar() roda uma vez, entao o modelo nao e recarregado a cada pergunta.
        # Ex.: self.scorer = BERTScorer(lang='en')
        pass

    def avaliar_pergunta(self, row):
        # PASSO 5 - so para escopo "pergunta". "row" e um dict com as colunas do CSV, TODAS como
        # texto (str); converta com float() quando precisar. Colunas principais:
        #   regime ('zero-shot', '3-shot', '5-shot', 'fine-tuned'), question, prediction, reference,
        #   latency_s, ttft_s, tokens_per_s, decode_tokens_per_s, new_tokens, token_f1, rouge_l,
        #   answer_words, reference_words, nll_sum, nll_tokens.
        # Devolva um dict {coluna: valor}. O runner ja acrescenta pergunta_idx, regime e question,
        # entao nao precisa repetir esses campos.
        prediction = row.get('prediction') or ''
        reference = row.get('reference') or ''
        n_pred, n_ref = len(prediction.split()), len(reference.split())
        return {'palavras_resposta': n_pred,
                'razao_tamanho': n_pred / n_ref if n_ref else None}

    def avaliar_run(self, run: RunContext):
        # PASSO 6 - so para escopo "run". "run" e um RunContext com:
        #   run.pasta      Path da run (ex.: campanha/gemma4/run_01)
        #   run.modelo     nome curto (gemma4, huatuogpt, medgemma, qwen35)
        #   run.modelo_id  ID do Hugging Face (ex.: unsloth/gemma-4-E4B-it)
        #   run.status     dict do status.json (seed, repeticao, persona, ...)
        #   run.respostas  lista de dicts, as mesmas linhas que avaliar_pergunta recebe
        #   run.treino     {'fine-tuned': {'train_runtime': ..., 'training_loss': ..., ...}}
        # Devolva um dict (1 linha no CSV) ou uma lista de dicts (ex.: 1 linha por regime).
        # No escopo "pergunta" este metodo nao e chamado; devolver {} basta.
        return {}


# PASSO 7 - criar(): o loader exige esta funcao. Ela devolve o plugin pronto.
# Para receber parametros, leia variaveis de ambiente aqui (ex.: os.getenv('MINHA_METRICA_LANG', 'en')).
def criar():
    return Template()


if __name__ == '__main__':
    # Autoteste: confere o contrato e mostra a saida com uma linha e uma run fake.
    from pathlib import Path
    from metricas.base import MetricaPlugin
    plugin = criar()
    assert isinstance(plugin, MetricaPlugin), 'faltam nome, escopo, avaliar_pergunta ou avaliar_run'
    assert plugin.escopo in ('pergunta', 'run'), 'escopo deve ser "pergunta" ou "run"'
    row = {'regime': 'zero-shot', 'question': 'What is diabetes?',
           'prediction': 'Diabetes is a chronic disease.', 'reference': 'Diabetes is a disease of high blood sugar.',
           'latency_s': '5.2', 'ttft_s': '0.15'}
    run = RunContext(pasta=Path('.'), modelo_id='unsloth/gemma-4-E4B-it', modelo='gemma4',
                     status={'status': 'ok', 'seed': 42}, respostas=[row],
                     treino={'fine-tuned': {'train_runtime': '812.4', 'training_loss': '1.23'}})
    print('nome:', plugin.nome, '| escopo:', plugin.escopo)
    print('avaliar_pergunta:', plugin.avaliar_pergunta(row))
    print('avaliar_run:', plugin.avaliar_run(run))
