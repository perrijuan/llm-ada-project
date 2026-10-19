"""Testes sem GPU. Usa processos reais com modelos artificiais, nunca LLMs."""
import ast
import contextlib
import csv
import io
import json
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
from consolidar import consolidar
from registro import escrever_csv


class Normalizar(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id in {
                'etapa', 'registrar_trainer', 'salvar_resultado', 'salvar_tabelas', 'salvar_figuras'
            }:
                return None
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'plt' and call.func.attr == 'show':
                return None
            if isinstance(call.func, ast.Name) and call.func.id == 'print' and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == 'Execucao concluida.':
                return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'tqdm':
            return node.args[0]
        if isinstance(node.func, ast.Name) and node.func.id == 'SFTConfig':
            node.keywords = [kw for kw in node.keywords if kw.arg != 'max_steps']
        return node


class Testes(unittest.TestCase):
    def test_configuracao_do_protocolo_nos_quatro_modelos(self):
        for item in json.loads((BASE / 'auditoria/manifesto.json').read_text()):
            script = (BASE / 'modelos' / (item['modelo'] + '.py')).read_text(encoding='utf-8')
            tree = ast.parse(script)
            assignments = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    try:
                        assignments[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
            self.assertEqual(assignments['CONDITIONS'], [('zero-shot', 0, False), ('3-shot', 3, False), ('5-shot', 5, False), ('fine-tuned', 0, True)])
            self.assertEqual(assignments['LEARNING_RATE'], 2e-5)
            self.assertEqual(assignments['FINE_TUNING_EPOCHS'], 5)
            self.assertEqual(assignments['MICRO_BATCH_SIZE'] * assignments['GRADIENT_ACCUMULATION_STEPS'], 32)
            self.assertEqual(assignments['EVAL_SIZE'] / assignments['SAMPLE_SIZE'], .2)
            configs = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'SFTConfig']
            self.assertEqual(len(configs), 1)
            for config in configs:
                kw = next(k for k in config.keywords if k.arg == 'max_steps')
                self.assertEqual(ast.literal_eval(kw.value), -1)
                self.assertEqual(ast.literal_eval(next(k for k in config.keywords if k.arg == 'optim').value), 'adamw_torch')

    def test_formulas_originais_preservadas(self):
        names = {'mean_ci', 'token_f1', 'rouge_l_f1', 'letter_ids', 'medqa_prompt', 'pubmedqa_prompt'}
        originals = {}
        notebook = json.loads((BASE / 'originais/gemma4_e4b_medquad_unsloth.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    originals[node.name] = ast.dump(node)
        self.assertEqual(set(originals), names)
        for path in (BASE / 'modelos').glob('*.py'):
            current = {n.name: ast.dump(n) for n in ast.parse(path.read_text(encoding='utf-8')).body
                       if isinstance(n, ast.FunctionDef) and n.name in names}
            self.assertEqual(current, originals)

    def test_icl_sem_resposta_do_teste_e_demos_aninhadas(self):
        import hashlib
        import random
        tree = ast.parse((BASE / 'modelos/medgemma.py').read_text(encoding='utf-8'))
        names = {'clean_text', 'make_demo_pool', 'select_demonstrations', 'make_messages', 'macro_scores'}
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        env = dict(re=re, hashlib=hashlib, random=random, SEED=42, SPLIT_SEED=42, PERSONA='clinico', prompt_records={})
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'helpers', 'exec'), env)
        rows = [{'q': f'q{i}', 'a': f'a{i}'} for i in range(10)] + [{'q': ' Q0 ', 'a': 'duplicate'}]
        pool = env['make_demo_pool'](rows, 'q', ['q9'], lambda r: r['q'], lambda r: r['a'])
        self.assertEqual(len(pool), 9)
        self.assertNotIn('q9', [r['question'] for r in pool])
        env['demo_pools'] = {'test': pool}
        three = env['select_demonstrations']('test', 'q9', 3)
        five = env['select_demonstrations']('test', 'q9', 5)
        self.assertEqual(three, five[:3])
        self.assertEqual(five, env['select_demonstrations']('test', 'q9', 5))
        for shots in (0, 3, 5):
            env.update(CURRENT_SHOTS=shots, CURRENT_CONDITION=str(shots))
            messages = env['make_messages']('test', 'q9', 'q9', 'instrucoes')
            self.assertEqual(len(messages), 2 + 2 * shots)
            self.assertEqual(messages[-1], {'role': 'user', 'content': 'q9'})
            self.assertNotIn('a9', json.dumps(messages))
        score = env['macro_scores'](['A'] * 3, list('ABC'), list('ABC'))
        self.assertAlmostEqual(score['precision_macro'], 1 / 9)
        self.assertAlmostEqual(score['recall_macro'], 1 / 3)
        self.assertAlmostEqual(score['f1_macro'], 1 / 6)

    def test_ordem_real_do_laco_e_treino_apenas_no_final(self):
        for path in (BASE / 'modelos').glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.iter, ast.Name) and n.iter.id == 'CONDITIONS')
            events = []
            class FakeModel:
                def save_pretrained(self, path):
                    events.append(('save', path))
            env = {'CONDITIONS': [('zero-shot', 0, False), ('3-shot', 3, False), ('5-shot', 5, False), ('fine-tuned', 0, True)],
                   'SEED': 42, 'FINE_TUNING_EPOCHS': 5, 'train_dataset': object(), 'comparison_examples': [],
                   'OUTPUT_DIR': 'fake', 'EXTENDED_DIR': 'fake-ext', 'all_rows': [], 'preview_rows': [], 'prompt_records': {},
                   'etapa': lambda *a: None, 'set_seed': lambda *a: None,
                   'FastModel': types.SimpleNamespace(for_inference=lambda m: None),
                   'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
                   'gc': types.SimpleNamespace(collect=lambda: None), 'escrever_csv': lambda *a: None,
                   'salvar_tabelas': lambda *a: None, 'generate_preview_rows': lambda d: [],
                   'evaluate_benchmarks': lambda c: events.append(('benchmark', c)), 'save_summaries': lambda *a: None}
            def build_model(fine_tune=False):
                events.append(('load', fine_tune))
                env['text_tokenizer'] = FakeModel()
                return FakeModel()
            def train_regime(model, data, epochs, tag):
                events.append(('train', tag, epochs))
                return {}
            def evaluate_model(model, condition):
                events.append(('eval', condition))
                return [], 0
            env.update(build_model=build_model, train_regime=train_regime, evaluate_model=evaluate_model)
            exec(compile(ast.Module(body=[loop], type_ignores=[]), 'loop', 'exec'), env)
            self.assertEqual([e for e in events if e[0] == 'train'], [('train', 'fine-tuned', 5)])
            self.assertEqual([e for e in events if e[0] == 'load'], [('load', False)] * 3 + [('load', True)])
            self.assertLess(events.index(('benchmark', '5-shot')), events.index(('train', 'fine-tuned', 5)))
            self.assertNotIn('model', env)
            self.assertNotIn('text_tokenizer', env)

    def test_csv_unicode_listas_e_colunas_variaveis(self):
        with tempfile.TemporaryDirectory(dir=BASE / "testes") as temp:
            p = Path(temp) / 'teste.csv'
            escrever_csv(p, [{'pergunta': 'É herdado?\n"sim", não', 'tempos': [0.1, 0.2]}, {'pergunta': 'x', 'loss': 1.2}])
            with p.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(json.loads(rows[0]['tempos']), [0.1, 0.2])
            self.assertEqual(rows[0]['pergunta'], 'É herdado?\n"sim", não')
            self.assertEqual(rows[1]['loss'], '1.2')

    def test_dez_execucoes_por_modelo_isolamento_e_retomada(self):
        with tempfile.TemporaryDirectory(dir=BASE / "testes") as temp:
            root = Path(temp)
            package = root / 'pacote'
            (package / 'modelos').mkdir(parents=True)
            (package / 'registro.py').write_text('# fixture', encoding='utf-8')
            (package / 'prompts_medicos.py').write_bytes((BASE / 'prompts_medicos.py').read_bytes())
            (package / 'rodar_todos.py').write_bytes((BASE / 'rodar_todos.py').read_bytes())
            fixture = '''import os, csv
from pathlib import Path
assert Path(os.environ['LLM_RUN_DIR']).resolve() == Path.cwd()
with open('benchmark_summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['condition','metric','mean'])
    w.writeheader(); w.writerow(dict(condition='base',metric='simulado',mean=1))
print('processo',os.getpid())
'''
            for model in ['gemma4', 'qwen35']:
                (package / 'modelos' / f'{model}.py').write_text(fixture, encoding='utf-8')
            progress = types.ModuleType('tqdm.auto')
            progress.tqdm = lambda iterable, **kwargs: iterable
            args = ['rodar_todos.py', '--modelos', 'gemma4', 'qwen35', '--saida', str(root / 'saida')]
            with patch.object(rodar_todos, 'BASE', package), patch.dict(sys.modules, {'tqdm': types.ModuleType('tqdm'), 'tqdm.auto': progress}), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', args):
                    self.assertEqual(rodar_todos.main(), 0)
                statuses = sorted((root / 'saida').glob('*/run_*/status.json'))
                self.assertEqual(len(statuses), 20)
                original = [s.read_bytes() for s in statuses]
                with patch.object(sys, 'argv', args + ['--retomar']):
                    self.assertEqual(rodar_todos.main(), 0)
                self.assertEqual(original, [s.read_bytes() for s in statuses])
                with patch.object(sys, 'argv', args):
                    with self.assertRaises(FileExistsError):
                        rodar_todos.main()
            with (root / 'saida/consolidado/benchmark_summary.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 20)
            self.assertEqual([r['exec_repeticao'] for r in rows[:10]], [str(i) for i in range(1, 11)])
            self.assertFalse((root / 'saida/.em_execucao').exists())

    def test_consolidacao_exclui_falhas(self):
        with tempfile.TemporaryDirectory(dir=BASE / "testes") as temp:
            root = Path(temp)
            for run, status in [(1, 'ok'), (2, 'erro')]:
                folder = root / 'gemma4' / f'run_{run:02}'
                folder.mkdir(parents=True)
                (folder / 'status.json').write_text(json.dumps(dict(modelo='gemma4', repeticao=run, seed=42, status=status)))
                escrever_csv(folder / 'benchmark_summary.csv', [{'mean': run}])
            result = consolidar(root)
            self.assertEqual(len(result['benchmark_summary.csv']), 1)

    def test_falha_pula_modelo_e_salva_log(self):
        with tempfile.TemporaryDirectory(dir=BASE / 'testes') as temp:
            root = Path(temp)
            (root / 'modelos').mkdir()
            (root / 'registro.py').write_text('# fixture')
            (root / 'prompts_medicos.py').write_bytes((BASE / 'prompts_medicos.py').read_bytes())
            (root / 'rodar_todos.py').write_bytes((BASE / 'rodar_todos.py').read_bytes())
            (root / 'modelos/gemma4.py').write_text('raise RuntimeError("falha simulada")')
            (root / 'modelos/qwen35.py').write_text('print("modelo seguinte executado")')
            progress = types.ModuleType('tqdm.auto')
            progress.tqdm = lambda iterable, **kwargs: iterable
            with patch.object(rodar_todos, 'BASE', root), patch.dict(sys.modules, {'tqdm': types.ModuleType('tqdm'), 'tqdm.auto': progress}), patch.object(sys, 'argv', ['rodar_todos.py', '--modelos', 'gemma4', 'qwen35', '--repeticoes', '2', '--saida', str(root / 'saida')]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rodar_todos.main(), 1)
            folder = root / 'saida/gemma4/run_01'
            self.assertEqual(json.loads((folder / 'status.json').read_text())['status'], 'erro')
            self.assertIn('falha simulada', (folder / 'execucao.log').read_text())
            self.assertFalse((root / 'saida/gemma4/run_02').exists())
            for run in (1, 2):
                next_status = root / 'saida/qwen35' / f'run_{run:02}' / 'status.json'
                self.assertEqual(json.loads(next_status.read_text())['status'], 'ok')
            self.assertFalse((root / 'saida/.em_execucao').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

