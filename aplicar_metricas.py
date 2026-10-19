# =============================================================================================
# BLOCO DE IMPLEMENTACAO - pacote metricas/ (PR #3 de perrijuan/llm-ada-project)
# =============================================================================================
# O que faz: cria o pacote de plugins de metricas, o CLI rodar_metricas.py, os testes e a secao
# do README, e roda a verificacao no final. Nao baixa modelos e nao usa GPU.
#
# Como usar:
#   1. salve como aplicar_metricas.py na RAIZ do repo (onde estao consolidar.py e registro.py) e rode
#        python aplicar_metricas.py
#      ou cole numa celula do Jupyter aberta na raiz do repo.
#   2. confira a saida: "ok <arquivo>" para cada arquivo, "Ran 8 tests ... OK" e o autoteste do template.
#   3. se algum arquivo ja existir com conteudo diferente, ele e PULADO; troque SOBRESCREVER = True.
#
# Arquivos criados (o que cada um faz):
#   metricas/base.py              contrato: Protocol MetricaPlugin + dataclass RunContext
#   metricas/loader.py            carregar_plugins(modulos): importa, exige criar(), avisa dependencia faltante
#   metricas/runner.py            executar(campanha, plugins): percorre */run_*, grava metricas_<nome>.csv
#   metricas/plugins/desempenho.py builtin (escopo run): prefill, decode mean/p95, TTFT, tok/s e peak RSS por regime + treino
#   metricas/plugins/template.py  TEMPLATE comentado (PASSO 0-7) para o colega criar a metrica dele
#   metricas/plugins/bert_score.py exemplo com lib externa (pip install bert-score)
#   rodar_metricas.py             CLI: pasta + --plugins mod1 mod2 + --ignorar-faltantes
#   testes/test_metricas.py       8 testes com plugin fake (sem GPU, sem dependencias)
#
# Para o colega adicionar uma metrica: copie metricas/plugins/template.py para
# metricas/plugins/<sua_metrica>.py, siga os PASSOS no arquivo e rode
#   python -m metricas.plugins.<sua_metrica>                 (autoteste)
#   python rodar_metricas.py <campanha> --plugins metricas.plugins.<sua_metrica>
#   python consolidar.py <campanha>                          (junta em consolidado/metricas_<nome>.csv)
# =============================================================================================
import subprocess
import sys
from pathlib import Path

REPO = Path('.').resolve()  # troque se o notebook nao estiver na raiz do repo
SOBRESCREVER = False        # True para trocar arquivos que ja existem

# Conteudo de cada arquivo, igual ao do PR.
ARQUIVOS = {}

# ---------------------------------------------------------------- metricas/__init__.py
ARQUIVOS['metricas/__init__.py'] = r'''"""Metricas extras calculadas depois das runs, a partir dos CSVs salvos."""
from metricas.base import MetricaPlugin, RunContext
from metricas.loader import carregar_plugins
from metricas.runner import executar

__all__ = ['MetricaPlugin', 'RunContext', 'carregar_plugins', 'executar']
'''

# ---------------------------------------------------------------- metricas/base.py
ARQUIVOS['metricas/base.py'] = r'''"""Contrato dos plugins e dados de uma run lidos do disco."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

ESCOPOS = ('pergunta', 'run')


@dataclass
class RunContext:
    pasta: Path
    modelo_id: str
    modelo: str
    status: dict
    respostas: list = field(default_factory=list)  # linhas de answers_per_question.csv
    treino: dict = field(default_factory=dict)  # tag -> linha de registros/treino_<tag>_metricas.csv


@runtime_checkable
class MetricaPlugin(Protocol):
    nome: str  # vira metricas_<nome>.csv
    escopo: str  # "pergunta" (uma vez por linha) ou "run" (uma vez por run)

    def avaliar_pergunta(self, row: dict) -> dict:
        ...

    def avaliar_run(self, run: RunContext) -> dict | list[dict]:
        ...
'''

# ---------------------------------------------------------------- metricas/loader.py
ARQUIVOS['metricas/loader.py'] = r'''"""Importa os modulos de plugin. Cada modulo precisa expor criar() -> MetricaPlugin."""
import importlib

from metricas.base import ESCOPOS, MetricaPlugin


def carregar_plugins(modulos):
    # Retorna (plugins, erros); erros e uma lista de mensagens, uma por modulo que falhou.
    plugins, erros = [], []
    for modulo in modulos:
        try:
            mod = importlib.import_module(modulo)
        except ModuleNotFoundError as exc:
            if exc.name and (modulo == exc.name or modulo.startswith(exc.name + '.')):
                erros.append(f'{modulo}: modulo nao encontrado (confira o nome e o PYTHONPATH).')
            else:
                erros.append(f"{modulo}: dependencia '{exc.name}' nao instalada "
                             f'(ex.: pip install {exc.name}).')
            continue
        except Exception as exc:
            erros.append(f'{modulo}: erro ao importar ({type(exc).__name__}: {exc}).')
            continue
        if not callable(getattr(mod, 'criar', None)):
            erros.append(f'{modulo}: nao expoe criar() -> MetricaPlugin.')
            continue
        try:
            plugin = mod.criar()
        except Exception as exc:
            erros.append(f'{modulo}: criar() falhou ({type(exc).__name__}: {exc}).')
            continue
        if not isinstance(plugin, MetricaPlugin) or plugin.escopo not in ESCOPOS:
            erros.append(f'{modulo}: plugin precisa de nome, escopo em {ESCOPOS}, '
                         'avaliar_pergunta e avaliar_run.')
            continue
        plugins.append(plugin)
    return plugins, erros
'''

# ---------------------------------------------------------------- metricas/runner.py
ARQUIVOS['metricas/runner.py'] = r'''"""Aplica os plugins em cada run da campanha e grava metricas_<nome>.csv na pasta da run."""
import csv
import json
from pathlib import Path

from metricas.base import RunContext
from registro import escrever_csv


def ler_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def ler_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def montar_contexto(pasta):
    pasta = Path(pasta)
    status = ler_json(pasta / 'status.json')
    ambiente = ler_json(pasta / 'ambiente.json')
    modelo = status.get('modelo') or ambiente.get('modelo') or pasta.parent.name
    # ambiente.json atual nao grava o ID do Hugging Face; model_config.json tem _name_or_path.
    modelo_id = (ambiente.get('modelo_id') or ambiente.get('model_id')
                 or ler_json(pasta / 'registros' / 'model_config.json').get('_name_or_path') or modelo)
    respostas = []
    for path in sorted(pasta.rglob('answers_per_question.csv')):
        respostas.extend(ler_csv(path))
    treino = {}
    for path in sorted((pasta / 'registros').glob('treino_*_metricas.csv')):
        tag = path.name[len('treino_'):-len('_metricas.csv')]
        linhas = ler_csv(path)
        if linhas:
            treino[tag] = linhas[-1]
    return RunContext(pasta=pasta, modelo_id=modelo_id, modelo=modelo, status=status,
                      respostas=respostas, treino=treino)


def aplicar(plugin, run):
    if plugin.escopo == 'pergunta':
        linhas = []
        for i, row in enumerate(run.respostas):
            # regime/question/indice permitem juntar com answers_per_question.csv depois.
            linhas.append({'pergunta_idx': i, 'regime': row.get('regime'),
                           'question': row.get('question'), **plugin.avaliar_pergunta(row)})
        return linhas
    resultado = plugin.avaliar_run(run)
    return [resultado] if isinstance(resultado, dict) else list(resultado)


def executar(campanha: Path, plugins):
    # Retorna {pasta_da_run: {'ok': [nomes], 'falhou': {nome: erro}, 'ignorada': motivo}}.
    relatorio = {}
    for pasta in sorted(Path(campanha).glob('*/run_*')):
        if not pasta.is_dir():
            continue
        run = montar_contexto(pasta)
        item = {'ok': [], 'falhou': {}, 'ignorada': None}
        relatorio[pasta] = item
        # consolidar.py so junta runs completas; parciais ficam de fora aqui tambem.
        if run.status.get('status') != 'ok':
            item['ignorada'] = f"status={run.status.get('status', 'sem status.json')}"
            continue
        for plugin in plugins:
            try:
                linhas = aplicar(plugin, run)
                escrever_csv(pasta / f'metricas_{plugin.nome}.csv', linhas)
                item['ok'].append(plugin.nome)
            except Exception as exc:
                item['falhou'][plugin.nome] = f'{type(exc).__name__}: {exc}'
    return relatorio
'''

# ---------------------------------------------------------------- metricas/plugins/__init__.py
ARQUIVOS['metricas/plugins/__init__.py'] = r'''"""Plugins de metricas. desempenho e builtin; template.py e o ponto de partida; bert_score e exemplo com lib externa."""
'''

# ---------------------------------------------------------------- metricas/plugins/desempenho.py
ARQUIVOS['metricas/plugins/desempenho.py'] = r'''"""Builtin, escopo run: metricas [METRICS] de velocidade por regime + tempo e loss do treino."""
import json
import math

import numpy as np


def numero(valor):
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(valor) else valor


class Desempenho:
    nome = 'desempenho'
    escopo = 'run'

    def avaliar_pergunta(self, row):
        return {}

    def avaliar_run(self, run):
        regimes = list(dict.fromkeys(row.get('regime') for row in run.respostas))
        linhas = []
        for regime in regimes:
            rows = [row for row in run.respostas if row.get('regime') == regime]
            prefill = [v for v in (numero(row.get('prefill_s')) for row in rows) if v is not None]
            ttft = [v for v in (numero(row.get('ttft_s')) for row in rows) if v is not None]
            prompt_tokens = [v for v in (numero(row.get('prompt_tokens')) for row in rows) if v is not None]
            generated_tokens = [v for v in (numero(row.get('new_tokens')) for row in rows) if v is not None]
            latency = [v for v in (numero(row.get('latency_s')) for row in rows) if v is not None]
            rss = [v for v in (numero(row.get('peak_rss_bytes')) for row in rows) if v is not None]

            intervals = []
            for row in rows:
                try:
                    times = json.loads(row.get('token_times', ''))
                    if not isinstance(times, list):
                        continue
                    times = [numero(value) for value in times]
                    if any(value is None for value in times):
                        continue
                    intervals.extend((b - a) * 1000 for a, b in zip(times, times[1:]))
                except (TypeError, ValueError):
                    continue

            total_time = sum(latency)
            linha = {
                'regime': regime,
                'n_perguntas': len(rows),
                'prefill_ms': float(np.mean(prefill)) * 1000 if prefill else None,
                'decode_mean_ms': float(np.mean(intervals)) if intervals else None,
                'decode_p95_ms': float(np.percentile(intervals, 95)) if intervals else None,
                'ttft_ms': float(np.mean(ttft)) * 1000 if ttft else None,
                'overall_tok/s': sum(generated_tokens) / total_time if generated_tokens and total_time > 0 else None,
                'peak_rss_bytes': max(rss) if rss else None,
                'prompt_tokens': float(np.mean(prompt_tokens)) if prompt_tokens else None,
                'generated_tokens': float(np.mean(generated_tokens)) if generated_tokens else None,
            }
            # So o regime fine-tuned treina; os outros ficam vazios.
            treino = run.treino.get(regime, {})
            linha['train_runtime'] = numero(treino.get('train_runtime'))
            linha['training_loss'] = numero(treino.get('training_loss'))
            linhas.append(linha)
        return linhas


def criar():
    return Desempenho()
'''

# ---------------------------------------------------------------- metricas/plugins/template.py
ARQUIVOS['metricas/plugins/template.py'] = r'''"""TEMPLATE de plugin de metricas: copie este arquivo e preencha os PASSOS 1 a 7.

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
'''

# ---------------------------------------------------------------- metricas/plugins/bert_score.py
ARQUIVOS['metricas/plugins/bert_score.py'] = r'''"""Exemplo de modulo externo, escopo pergunta: BERTScore entre prediction e reference.

Exemplo real com lib externa; para comecar um plugin novo use template.py (comentado passo a passo).
Requer: pip install bert-score
"""
from bert_score import BERTScorer


class BertScore:
    nome = 'bert_score'
    escopo = 'pergunta'

    def __init__(self, lang='en'):
        # MedQuAD esta em ingles. O scorer carrega o modelo uma vez e reaproveita em todas as linhas.
        self.scorer = BERTScorer(lang=lang)

    def avaliar_pergunta(self, row):
        prediction, reference = row.get('prediction') or '', row.get('reference') or ''
        if not prediction.strip() or not reference.strip():
            # Texto vazio quebra o tokenizer em algumas versoes; resposta vazia conta como 0.
            return {'bert_p': 0.0, 'bert_r': 0.0, 'bert_f1': 0.0}
        p, r, f1 = self.scorer.score([prediction], [reference])
        return {'bert_p': p.item(), 'bert_r': r.item(), 'bert_f1': f1.item()}

    def avaliar_run(self, run):
        return {}


def criar():
    return BertScore()
'''

# ---------------------------------------------------------------- rodar_metricas.py
ARQUIVOS['rodar_metricas.py'] = r'''"""Calcula metricas extras nas runs de uma campanha e grava metricas_<nome>.csv em cada run."""
import argparse
import sys
from pathlib import Path

from metricas import carregar_plugins, executar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pasta', type=Path, help='pasta da campanha (ex.: campanha_clinico)')
    parser.add_argument('--plugins', nargs='+', default=['metricas.plugins.desempenho'],
                        help='modulos que expoem criar() -> MetricaPlugin')
    parser.add_argument('--ignorar-faltantes', action='store_true',
                        help='segue com os plugins que importaram, em vez de parar')
    args = parser.parse_args()

    plugins, erros = carregar_plugins(args.plugins)
    for erro in erros:
        print(f'[plugin] {erro}', flush=True)
    if erros and not args.ignorar_faltantes:
        print('Plugins indisponiveis. Instale as dependencias ou use --ignorar-faltantes.')
        return 1
    if not plugins:
        print('Nenhum plugin carregado.')
        return 1

    relatorio = executar(args.pasta, plugins)
    if not relatorio:
        print(f'Nenhuma run encontrada em {args.pasta}/*/run_*.')
        return 1
    falhas = False
    for pasta, item in relatorio.items():
        nome = pasta.relative_to(args.pasta).as_posix()
        if item['ignorada']:
            print(f'{nome}: ignorada ({item["ignorada"]})')
            continue
        print(f'{nome}: ok={item["ok"] or "-"}')
        for plugin, erro in item['falhou'].items():
            falhas = True
            print(f'  falhou {plugin}: {erro}')
    print('Para juntar nas tabelas: python consolidar.py', args.pasta)
    return 1 if falhas else 0


if __name__ == '__main__':
    sys.exit(main())
'''

# ---------------------------------------------------------------- testes/test_metricas.py
ARQUIVOS['testes/test_metricas.py'] = r'''"""Pacote metricas sem GPU: plugin fake, ambos os escopos, CSV gerado e plugin faltante."""
import ast
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from consolidar import consolidar
from metricas import RunContext, carregar_plugins, executar
from metricas.runner import montar_contexto
from registro import escrever_csv

PLUGIN_FAKE = """
class Fake:
    nome = 'fake'
    escopo = 'pergunta'

    def avaliar_pergunta(self, row):
        return {'n_chars': len(row['prediction'])}

    def avaliar_run(self, run):
        return {}


class FakeRun:
    nome = 'fake_run'
    escopo = 'run'

    def avaliar_pergunta(self, row):
        return {}

    def avaliar_run(self, run):
        return {'modelo_id': run.modelo_id, 'n_respostas': len(run.respostas)}


def criar():
    return Fake()


def criar_run():
    return FakeRun()
"""


def ler(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class MetricasTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.campanha = Path(self.tmp.name) / 'campanha'
        self.run = self.campanha / 'gemma4' / 'run_01'
        self.run.mkdir(parents=True)
        (self.run / 'status.json').write_text(json.dumps(
            {'modelo': 'gemma4', 'repeticao': 1, 'seed': 42, 'status': 'ok'}), encoding='utf-8')
        (self.run / 'ambiente.json').write_text(json.dumps(
            {'modelo': 'gemma4', 'modelo_id': 'unsloth/gemma-4-E4B-it'}), encoding='utf-8')
        linhas = []
        for regime in ['zero-shot', 'fine-tuned']:
            for i in range(4):
                linha = {'regime': regime, 'question': f'q{i}', 'prediction': 'x' * (i + 1),
                         'reference': 'ref', 'latency_s': i + 1, 'ttft_s': 0.1, 'prefill_s': 0.05,
                         'token_times': [0.1, 0.2, 0.4], 'new_tokens': 3, 'prompt_tokens': 10 + i,
                         'peak_rss_bytes': 1000 * (i + 1), 'tokens_per_s': 10, 'decode_tokens_per_s': 'nan'}
                if regime == 'fine-tuned':
                    del linha['prefill_s']
                    del linha['peak_rss_bytes']
                linhas.append(linha)
        escrever_csv(self.run / 'gemma4-e4b-medquad-unsloth-extended' / 'answers_per_question.csv', linhas)
        escrever_csv(self.run / 'registros' / 'treino_fine-tuned_metricas.csv',
                     [{'training_loss': 1.5, 'global_step': 30, 'train_runtime': 120.0}])
        # Run com erro deve ser ignorada.
        erro = self.campanha / 'gemma4' / 'run_02'
        erro.mkdir()
        (erro / 'status.json').write_text(json.dumps({'modelo': 'gemma4', 'status': 'erro'}), encoding='utf-8')

        modulos = Path(self.tmp.name) / 'mods'
        modulos.mkdir()
        (modulos / 'plugin_fake.py').write_text(PLUGIN_FAKE, encoding='utf-8')
        (modulos / 'plugin_fake_run.py').write_text(
            'from plugin_fake import criar_run as criar\n', encoding='utf-8')
        (modulos / 'plugin_sem_dep.py').write_text(
            'import pacote_que_nao_existe_xyz\n\ndef criar():\n    return None\n', encoding='utf-8')
        sys.path.insert(0, str(modulos))
        self.modulos = modulos

    def tearDown(self):
        sys.path.remove(str(self.modulos))
        for nome in ['plugin_fake', 'plugin_fake_run', 'plugin_sem_dep']:
            sys.modules.pop(nome, None)
        self.tmp.cleanup()

    def test_run_context(self):
        run = montar_contexto(self.run)
        self.assertIsInstance(run, RunContext)
        self.assertEqual(run.modelo, 'gemma4')
        self.assertEqual(run.modelo_id, 'unsloth/gemma-4-E4B-it')
        self.assertEqual(run.status['status'], 'ok')
        self.assertEqual(len(run.respostas), 8)
        self.assertEqual(run.treino['fine-tuned']['train_runtime'], '120.0')

    def test_escopos_e_csv(self):
        plugins, erros = carregar_plugins(['plugin_fake', 'plugin_fake_run', 'metricas.plugins.desempenho'])
        self.assertEqual(erros, [])
        relatorio = executar(self.campanha, plugins)
        self.assertEqual(relatorio[self.run]['ok'], ['fake', 'fake_run', 'desempenho'])
        self.assertIn('status=erro', relatorio[self.campanha / 'gemma4' / 'run_02']['ignorada'])

        pergunta = ler(self.run / 'metricas_fake.csv')
        self.assertEqual(len(pergunta), 8)
        self.assertEqual(pergunta[2]['n_chars'], '3')
        self.assertEqual(pergunta[2]['regime'], 'zero-shot')

        por_run = ler(self.run / 'metricas_fake_run.csv')
        self.assertEqual(por_run, [{'modelo_id': 'unsloth/gemma-4-E4B-it', 'n_respostas': '8'}])

        desempenho = {row['regime']: row for row in ler(self.run / 'metricas_desempenho.csv')}
        self.assertAlmostEqual(float(desempenho['zero-shot']['prefill_ms']), 50)
        self.assertAlmostEqual(float(desempenho['zero-shot']['decode_mean_ms']), 150)
        self.assertAlmostEqual(float(desempenho['zero-shot']['decode_p95_ms']), 200)
        self.assertAlmostEqual(float(desempenho['zero-shot']['ttft_ms']), 100)
        self.assertAlmostEqual(float(desempenho['zero-shot']['overall_tok/s']), 1.2)
        self.assertEqual(float(desempenho['zero-shot']['peak_rss_bytes']), 4000)
        self.assertAlmostEqual(float(desempenho['zero-shot']['prompt_tokens']), 11.5)
        self.assertAlmostEqual(float(desempenho['zero-shot']['generated_tokens']), 3)
        self.assertEqual(desempenho['zero-shot']['train_runtime'], '')
        self.assertEqual(desempenho['fine-tuned']['prefill_ms'], '')
        self.assertEqual(desempenho['fine-tuned']['peak_rss_bytes'], '')
        self.assertAlmostEqual(float(desempenho['fine-tuned']['decode_mean_ms']), 150)
        self.assertEqual(float(desempenho['fine-tuned']['train_runtime']), 120.0)
        self.assertEqual(float(desempenho['fine-tuned']['training_loss']), 1.5)
        self.assertFalse((self.campanha / 'gemma4' / 'run_02' / 'metricas_fake.csv').exists())

        # consolidar.py junta o CSV novo com exec_modelo/exec_seed/exec_repeticao.
        consolidar(self.campanha)
        junto = ler(self.campanha / 'consolidado' / 'metricas_fake.csv')
        self.assertEqual(junto[0]['exec_modelo'], 'gemma4')
        self.assertEqual(junto[0]['exec_seed'], '42')

    def test_metricas_velocidade_modelos(self):
        nomes = ['gemma4', 'huatuogpt', 'medgemma', 'qwen35']
        funcoes = ['peak_rss_bytes', 'metricas_velocidade', 'formatar_metricas']
        fontes = []
        for nome in nomes:
            texto = (RAIZ / 'modelos' / f'{nome}.py').read_text(encoding='utf-8')
            arvore = ast.parse(texto)
            definicoes = {node.name: ast.get_source_segment(texto, node)
                          for node in ast.walk(arvore) if isinstance(node, ast.FunctionDef)}
            fontes.append('\n\n'.join(definicoes[funcao] for funcao in funcoes))
        self.assertTrue(all(fonte == fontes[0] for fonte in fontes[1:]))

        namespace = {'np': np}
        exec(fontes[0], namespace)
        rows = [
            {'total_s': 2, 'ttft_s': 0.1, 'prefill_s': 0.05, 'token_times': [0, 0.1, 0.3],
             'new_tokens': 3, 'prompt_tokens': 10, 'peak_rss_bytes': 1000},
            {'total_s': 4, 'ttft_s': 0.3, 'prefill_s': 0.15, 'token_times': [0, 0.2, 0.4],
             'new_tokens': 5, 'prompt_tokens': 20, 'peak_rss_bytes': 2000},
        ]
        metricas = namespace['metricas_velocidade'](rows)
        self.assertEqual(metricas['input'], {'prompt_tokens': 15.0, 'generated_tokens': 4.0})
        self.assertAlmostEqual(metricas['prefill']['latency_ms'], 100)
        self.assertAlmostEqual(metricas['decode']['mean_ms'], 175)
        self.assertAlmostEqual(metricas['decode']['p95_ms'], 200)
        self.assertAlmostEqual(metricas['e2e']['ttft_ms'], 200)
        self.assertAlmostEqual(metricas['e2e']['overall_tokens_per_sec'], 8 / 6)
        self.assertEqual(metricas['memory']['peak_rss_bytes'], 2000)
        bloco = namespace['formatar_metricas'](metricas)
        self.assertTrue(bloco.startswith('[METRICS]'))
        self.assertEqual(len(bloco.splitlines()), 9)
        self.assertIn('  decode_p95_ms:     200.00', bloco.splitlines())

    def test_plugin_faltante(self):
        plugins, erros = carregar_plugins(['plugin_sem_dep', 'plugin_inexistente_abc', 'plugin_fake'])
        self.assertEqual([p.nome for p in plugins], ['fake'])
        self.assertIn("dependencia 'pacote_que_nao_existe_xyz' nao instalada", erros[0])
        self.assertIn('modulo nao encontrado', erros[1])

    def test_template(self):
        # O template documentado precisa funcionar como esta, para servir de ponto de partida.
        plugins, erros = carregar_plugins(['metricas.plugins.template'])
        self.assertEqual(erros, [])
        relatorio = executar(self.campanha, plugins)
        self.assertEqual(relatorio[self.run]['ok'], ['template'])
        linhas = ler(self.run / 'metricas_template.csv')
        self.assertEqual(len(linhas), 8)
        self.assertEqual(linhas[0]['palavras_resposta'], '1')
        self.assertEqual(float(linhas[0]['razao_tamanho']), 1.0)

    def test_falha_no_plugin_nao_para_os_outros(self):
        class Quebra:
            nome, escopo = 'quebra', 'run'

            def avaliar_pergunta(self, row):
                return {}

            def avaliar_run(self, run):
                raise RuntimeError('boom')

        plugins, _ = carregar_plugins(['plugin_fake'])
        relatorio = executar(self.campanha, [Quebra()] + plugins)
        self.assertIn('boom', relatorio[self.run]['falhou']['quebra'])
        self.assertEqual(relatorio[self.run]['ok'], ['fake'])

    def test_bloco_aplicar_igual_ao_pacote(self):
        # aplicar_metricas.py carrega uma copia dos arquivos; ela precisa estar igual ao repositorio.
        texto = (RAIZ / 'aplicar_metricas.py').read_text(encoding='utf-8')
        definicoes = {}
        exec(texto.split('\n# 1. Confere')[0], definicoes)
        for nome, conteudo in definicoes['ARQUIVOS'].items():
            self.assertEqual(conteudo, (RAIZ / nome).read_text(encoding='utf-8'),
                             f'{nome} mudou: regenere aplicar_metricas.py')

    def test_cli(self):
        env_path = f'{self.modulos}'
        base = [sys.executable, str(RAIZ / 'rodar_metricas.py'), str(self.campanha), '--plugins']
        env = {'PYTHONPATH': env_path, 'PATH': '/usr/bin:/bin'}
        sem = subprocess.run(base + ['plugin_fake', 'plugin_sem_dep'], capture_output=True,
                             text=True, cwd=RAIZ, env=env)
        self.assertEqual(sem.returncode, 1)
        self.assertIn('--ignorar-faltantes', sem.stdout)
        com = subprocess.run(base + ['plugin_fake', 'plugin_sem_dep', '--ignorar-faltantes'],
                             capture_output=True, text=True, cwd=RAIZ, env=env)
        self.assertEqual(com.returncode, 0, com.stdout + com.stderr)
        self.assertIn("gemma4/run_01: ok=['fake']", com.stdout)
        self.assertIn('gemma4/run_02: ignorada', com.stdout)


if __name__ == '__main__':
    unittest.main()
'''

# ---------------------------------------------------------------- README.md (secao nova)
SECAO_README = r'''## Métricas extras (plugins)

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

Para aplicar o pacote em outra cópia do projeto (ou conferir antes de usar), rode `python aplicar_metricas.py` na raiz:
ele recria os arquivos de `metricas/`, `rodar_metricas.py` e os testes, e roda a verificação.

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

'''


# 1. Confere se esta na raiz do repo e grava os arquivos.
if not (REPO / 'consolidar.py').exists() or not (REPO / 'registro.py').exists():
    raise SystemExit(f'{REPO} nao parece a raiz do llm-ada-project (faltam consolidar.py/registro.py).')
for nome, conteudo in ARQUIVOS.items():
    path = REPO / nome
    if path.exists() and path.read_text(encoding='utf-8') != conteudo and not SOBRESCREVER:
        print(f'PULADO (ja existe e e diferente): {nome}  -> use SOBRESCREVER = True')
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(conteudo, encoding='utf-8', newline='\n')
    print(f'ok  {nome}')

# 2. Coloca a secao do README antes de "## Escopo e validação".
readme = REPO / 'README.md'
texto = readme.read_text(encoding='utf-8')
if '## Métricas extras (plugins)' in texto:
    print('README ja tem a secao de metricas.')
elif '## Escopo e validação' in texto:
    readme.write_text(texto.replace('## Escopo e validação', SECAO_README + '## Escopo e validação', 1),
                      encoding='utf-8', newline='\n')
    print('ok  README.md')
else:
    print('README sem "## Escopo e validação": cole SECAO_README manualmente.')

# 3. Verificacao: testes do pacote (esperado: Ran 8 tests ... OK).
teste = subprocess.run([sys.executable, '-m', 'unittest', 'testes.test_metricas', '-v'],
                       cwd=REPO, capture_output=True, text=True)
print(teste.stdout + teste.stderr)

# 4. Verificacao: autoteste do template (o que o colega vai copiar).
auto = subprocess.run([sys.executable, '-m', 'metricas.plugins.template'], cwd=REPO, capture_output=True, text=True)
print(auto.stdout + auto.stderr)

# 5. Verificacao: bert_score aparece como faltante se a lib nao estiver instalada (esperado).
sys.path.insert(0, str(REPO))
from metricas import carregar_plugins
plugins, erros = carregar_plugins(['metricas.plugins.desempenho', 'metricas.plugins.template', 'metricas.plugins.bert_score'])
print('carregados:', [p.nome for p in plugins])
print('erros:', erros or 'nenhum')
print('Numa campanha real: python rodar_metricas.py <campanha> && python consolidar.py <campanha>')
