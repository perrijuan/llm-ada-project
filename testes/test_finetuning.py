"""Regressoes CPU: entrada SFT, isolamento de dados e falha de subprocesso."""
import ast
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from finetuning_dados import textos_para_tokenizacao, indices_demonstracoes
import rodar_todos


class FineTuning(unittest.TestCase):
    def test_coluna_iteravel_e_validacao(self):
        class Column:
            def __iter__(self):
                return iter(['pergunta resposta', 'outro exemplo'])
        texts = textos_para_tokenizacao({'text': Column()})
        self.assertIs(type(texts), list)
        self.assertEqual(texts, ['pergunta resposta', 'outro exemplo'])
        for bad in ([], [''], ['  '], [None], [['tokens']], [123]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                textos_para_tokenizacao({'text': bad})

    def test_demos_nao_compartilham_perguntas_com_treino_ou_teste(self):
        rows = [{'Question': f'q{i}'} for i in range(9)]
        rows.append({'Question': ' Q0  '})
        indices = indices_demonstracoes(rows, rows[:2], rows[2:4])
        self.assertEqual(indices, [4, 5, 6, 7, 8])
        with self.assertRaises(ValueError):
            indices_demonstracoes(rows[:8], rows[:2], rows[2:4])

    def test_formatacao_sft_independe_de_few_shot_nos_quatro_modelos(self):
        for path in (BASE / 'modelos').glob('*.py'):
            tree = ast.parse(path.read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'format_training_example')
            def template(messages, add_generation_prompt):
                self.assertFalse(add_generation_prompt)
                self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant'])
                self.assertEqual(messages[-1]['content'], 'resposta alvo')
                return json.dumps(messages)
            env = dict(SYSTEM_PROMPT='sistema', make_user_prompt=lambda q: q,
                       clean_answer_text=lambda a: a, apply_chat_template=template)
            # Sem CURRENT_SHOTS, demo_pools ou make_messages: nao pode depender deles.
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), env)
            formatted = env['format_training_example']({'Question': 'pergunta', 'Answer': 'resposta alvo'})
            self.assertIsInstance(formatted['text'], str)

    def test_erro_ao_abrir_processo_continua_no_proximo_modelo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / 'pacote'
            (package / 'modelos').mkdir(parents=True)
            for model in ('gemma4', 'qwen35'):
                (package / 'modelos' / f'{model}.py').write_text('print("ok")')
            real_popen = rodar_todos.subprocess.Popen
            def popen(command, **kwargs):
                if Path(command[-1]).stem == 'gemma4':
                    raise OSError('falha de abertura simulada')
                return real_popen(command, **kwargs)
            progress = types.ModuleType('tqdm.auto')
            progress.tqdm = lambda items, **kw: items
            argv = ['rodar_todos.py', '--modelos', 'gemma4', 'qwen35',
                    '--repeticoes', '1', '--saida', str(root / 'saida')]
            with patch.object(rodar_todos, 'BASE', package), patch.object(sys, 'argv', argv), \
                 patch.object(rodar_todos.subprocess, 'Popen', side_effect=popen), \
                 patch.dict(sys.modules, {'tqdm': types.ModuleType('tqdm'), 'tqdm.auto': progress}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rodar_todos.main(), 1)
            for model, expected in [('gemma4', 'erro'), ('qwen35', 'ok')]:
                status = json.loads((root / 'saida' / model / 'run_01/status.json').read_text())
                self.assertEqual(status['status'], expected)
            self.assertIn('falha de abertura simulada',
                          (root / 'saida/gemma4/run_01/execucao.log').read_text())
            self.assertFalse((root / 'saida/.em_execucao').exists())
