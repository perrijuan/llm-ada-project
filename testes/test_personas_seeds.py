"""Testes das personas, seeds e execucao sequencial, sem GPU."""
import ast
import contextlib
import hashlib
import io
import json
import random
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import rodar_todos
from prompts_medicos import configuracao, obter_prompts


class PersonasESeeds(unittest.TestCase):
    def test_configuracao_e_personas(self):
        with patch.dict('os.environ', {'LLM_SEED': '123', 'LLM_SPLIT_SEED': '42',
                                       'LLM_PERSONA': 'biomedico'}):
            cfg = configuracao()
        self.assertEqual((cfg['seed'], cfg['split_seed'], cfg['persona']), (123, 42, 'biomedico'))
        self.assertIn('biomedical researcher', cfg['system_prompt'])
        self.assertIn('2-4 factual sentences', cfg['system_prompt'])
        self.assertTrue(cfg['fontes_persona'])
        self.assertEqual(obter_prompts('original')[0],
                         'Answer the medical question directly in 2-4 factual sentences.')
        with self.assertRaises(ValueError):
            obter_prompts('inexistente')

    def test_seeds_dinamicas_mesmo_split_e_barreira_ft(self):
        for path in (BASE / 'modelos').glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name in {'select_demonstrations', 'make_messages'}]
            env = dict(SEED=42, SPLIT_SEED=42, PERSONA='clinico', hashlib=hashlib,
                       random=random, prompt_records={},
                       demo_pools={'test': [{'id': i, 'user': str(i), 'assistant': str(i)}
                                            for i in range(100)]})
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), env)
            first = env['select_demonstrations']('test', 'q', 5)
            env['SEED'] = 123
            second = env['select_demonstrations']('test', 'q', 5)
            self.assertNotEqual(first, second)
            self.assertEqual(second[:3], env['select_demonstrations']('test', 'q', 3))
            with self.assertRaises(ValueError):
                env['select_demonstrations']('test', 'q', 2)
            env.update(CURRENT_CONDITION='fine-tuned', CURRENT_SHOTS=3)
            with self.assertRaises(ValueError):
                env['make_messages']('test', 'q', 'q', 'system')
            env['CURRENT_SHOTS'] = 0
            self.assertEqual(len(env['make_messages']('test', 'q', 'q', 'system')), 2)
            # Toda selecao/split de dados utiliza a seed fixa.
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)
                     and n.func.attr in {'shuffle', 'train_test_split'}]
            self.assertTrue(calls)
            for call in calls:
                seed = next(k.value for k in call.keywords if k.arg == 'seed')
                self.assertEqual(seed.id, 'SPLIT_SEED')

    def test_processos_sequenciais_multiseed_e_retomada(self):
        with tempfile.TemporaryDirectory(dir=BASE / 'testes') as temp:
            package = Path(temp) / 'pacote'
            (package / 'modelos').mkdir(parents=True)
            for name in ['registro.py', 'prompts_medicos.py', 'rodar_todos.py']:
                (package / name).write_bytes((BASE / name).read_bytes())
            fixture = """import json, os, time
from pathlib import Path
root = Path(os.environ['LLM_RUN_DIR']).parents[1]
lock = root / 'modelo_ativo'
lock.touch(exist_ok=False)
try:
    with (root / 'ordem.jsonl').open('a') as f:
        f.write(json.dumps(dict(modelo=Path(__file__).stem, seed=os.environ['LLM_SEED'],
                               split=os.environ['LLM_SPLIT_SEED'], persona=os.environ['LLM_PERSONA']))+'\\n')
    time.sleep(.02)
finally:
    lock.unlink()
"""
            for model in ['gemma4', 'qwen35']:
                (package / 'modelos' / f'{model}.py').write_text(fixture, encoding='utf-8')
            root = Path(temp) / 'saida'
            args = ['rodar_todos.py', '--modelos', 'gemma4', 'qwen35', '--repeticoes', '2',
                    '--seeds', '42', '123', '--persona', 'biomedico', '--saida', str(root)]
            progress = types.ModuleType('tqdm.auto')
            progress.tqdm = lambda iterable, **kw: iterable
            with patch.object(rodar_todos, 'BASE', package), patch.dict(sys.modules, {
                    'tqdm': types.ModuleType('tqdm'), 'tqdm.auto': progress
                    }), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', args):
                    self.assertEqual(rodar_todos.main(), 0)
                order = [json.loads(line) for line in (root / 'ordem.jsonl').read_text().splitlines()]
                self.assertEqual([(r['modelo'], r['seed']) for r in order],
                    [(m, s) for m in ['gemma4', 'qwen35'] for s in ['42', '42', '123', '123']])
                self.assertTrue(all(r['split'] == '42' and r['persona'] == 'biomedico' for r in order))
                with patch.object(sys, 'argv', args + ['--retomar']):
                    self.assertEqual(rodar_todos.main(), 0)
                self.assertEqual(len((root / 'ordem.jsonl').read_text().splitlines()), 8)
                changed = args + ['--retomar', '--persona', 'clinico']
                with patch.object(sys, 'argv', changed), self.assertRaises(ValueError):
                    rodar_todos.main()
                (package / 'prompts_medicos.py').write_text('# alterado', encoding='utf-8')
                with patch.object(sys, 'argv', args + ['--retomar']), self.assertRaises(FileExistsError):
                    rodar_todos.main()
            self.assertFalse((root / '.em_execucao').exists())


if __name__ == '__main__':
    unittest.main()

