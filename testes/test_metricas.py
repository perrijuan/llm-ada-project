"""Pacote metricas sem GPU: plugin fake, ambos os escopos, CSV gerado e plugin faltante."""
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
