"""Estatistica de velocidade: media, mediana e pior cauda; sem GPU."""
import ast
from pathlib import Path
import unittest
import warnings

import numpy as np

BASE = Path(__file__).resolve().parents[1]


def carregar(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mean_median_p99']
    assert len(nodes) == 1, path
    env = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env['mean_median_p99']


class EstatisticaTests(unittest.TestCase):
    def test_valores_conhecidos_e_aviso_com_n_pequeno(self):
        for p in (BASE / 'modelos').glob('*.py'):
            f = carregar(p)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                mean, median, tail = f(list(range(1, 21)))
            self.assertAlmostEqual(mean, 10.5)
            self.assertEqual(median, 10.5)
            self.assertAlmostEqual(tail, 19.81)  # linear: 19 + 0.81 * (20 - 19)
            self.assertEqual(len(w), 1)

    def test_sem_aviso_com_100_ou_mais(self):
        for p in (BASE / 'modelos').glob('*.py'):
            f = carregar(p)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                f(np.arange(100.))
            self.assertEqual(w, [])

    def test_maior_e_melhor_usa_p1_e_equivale_ao_p99_do_tpot(self):
        # n=101 torna os indices exatos, entao 1/P99(tempo) == P1(1/tempo).
        for p in (BASE / 'modelos').glob('*.py'):
            f = carregar(p)
            tempos = np.linspace(1.0, 2.0, 101)
            _, _, p99_tempo = f(tempos, 'lower')
            _, _, p1_taxa = f(1 / tempos, 'higher')
            self.assertAlmostEqual(p1_taxa, 1 / p99_tempo)
            self.assertLess(p1_taxa, np.median(1 / tempos))

    def test_nan_ignorado_e_vazio(self):
        for p in (BASE / 'modelos').glob('*.py'):
            f = carregar(p)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mean, median, _ = f([1., float('nan'), 3.])
                self.assertEqual((mean, median), (2., 2.))
                self.assertTrue(all(np.isnan(v) for v in f([float('nan')])))

    def test_bloco_de_velocidade_usa_a_funcao_e_acuracia_mantem_ic(self):
        for p in (BASE / 'modelos').glob('*.py'):
            texto = p.read_text(encoding='utf-8')
            self.assertIn("mean_median_p99(values, better)", texto)
            self.assertIn("'p99_pior_caso': p99", texto)
            self.assertIn("mean, low, high = mean_ci([r['correct']", texto)


if __name__ == '__main__':
    unittest.main()
