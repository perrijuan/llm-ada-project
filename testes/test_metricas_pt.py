"""Integracao das metricas portuguesas sem baixar modelos nem chamar Ollama."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from consolidar import consolidar
from metricas.avaliacao_pt.bertscore import BERTScoreConfig
from metricas.avaliacao_pt.llm_judge import LLMJudgeConfig, LLMJudgeMetric
from metricas.base import RunContext
from metricas.plugins.bert_score_pt import BertScorePT
from metricas.plugins.phi4_judge_pt import Phi4JudgePT
from metricas.runner import executar
from registro import escrever_csv
from resumir import resumir_metricas_pt


def ler(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


class BERTFalso:
    def pontuar_detalhado(self, respostas, referencias):
        assert len(respostas) == len(referencias)
        return SimpleNamespace(precision=[0.7] * len(respostas),
                               recall=[0.8] * len(respostas),
                               f1=[0.75] * len(respostas))


class JudgeFalso:
    def __init__(self, config):
        assert config.language == 'pt'

    def avaliar_detalhado(self, resposta, referencia, *, contextos):
        assert contextos.row_id is not None
        if resposta == 'falha':
            return SimpleNamespace(nota=None, erro='Ollama indisponivel', tentativas=2)
        return SimpleNamespace(nota=8 if resposta == 'boa' else 6,
                               erro=None, tentativas=1)


class MetricasPTTest(unittest.TestCase):
    def test_configuracao_portuguesa_e_prompt_real(self):
        class Cliente:
            def __init__(self):
                self.chamadas = []

            def list(self):
                return {'models': []}

            def chat(self, **kwargs):
                self.chamadas.append(kwargs)
                return {'message': {'content': '{"nota": 0}'}}

        with tempfile.TemporaryDirectory() as tmp:
            cliente = Cliente()
            config = LLMJudgeConfig(language='pt', log_path=Path(tmp) / 'judge.csv')
            judge = LLMJudgeMetric(config, client=cliente)
            resultado = judge.avaliar_detalhado('resposta', 'referencia')
            self.assertEqual(resultado.nota, 0)
            self.assertEqual(resultado.erro, None)
            self.assertIn('Você avalia equivalência', cliente.chamadas[0]['messages'][0]['content'])
            self.assertIn('resposta_participante', cliente.chamadas[0]['messages'][1]['content'])
            self.assertEqual(BERTScoreConfig().language, 'pt')
            self.assertEqual(BERTScoreConfig().model_type, 'bert-base-multilingual-cased')

    def test_notas_ausentes_e_contexto(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = RunContext(pasta=Path(tmp), modelo='gemma4', modelo_id='m', status={},
                             respostas=[
                                 {'regime': 'zero-shot', 'question': 'q1',
                                  'prediction': 'boa', 'reference': 'ref'},
                                 {'regime': '3-shot', 'question': 'q1',
                                  'prediction': 'falha', 'reference': 'ref'},
                                 {'regime': '5-shot', 'question': 'q1',
                                  'prediction': '', 'reference': 'ref'},
                             ])
            bert = BertScorePT(BERTFalso()).avaliar_run(run)
            self.assertEqual([r['bert_f1'] for r in bert], [0.75, 0.75, None])
            self.assertTrue(bert[2]['erro'])
            judge = Phi4JudgePT(JudgeFalso).avaliar_run(run)
            self.assertEqual([r['phi4_judge'] for r in judge], [8, None, None])
            self.assertEqual(judge[1]['tentativas'], 2)
            self.assertEqual(judge[1]['erro'], 'Ollama indisponivel')
            self.assertEqual(judge[2]['erro'], 'resposta ou referencia vazia')

    def test_consolidacao_resumo_e_deltas_pareados(self):
        with tempfile.TemporaryDirectory() as tmp:
            campanha = Path(tmp) / 'campanha'
            pasta = campanha / 'gemma4' / 'run_01'
            pasta.mkdir(parents=True)
            (pasta / 'status.json').write_text(json.dumps({
                'modelo': 'gemma4', 'repeticao': 1, 'seed': 42,
                'split_seed': 42, 'persona': 'clinico', 'status': 'ok',
            }), encoding='utf-8')
            respostas = [
                {'regime': regime, 'question': question,
                 'prediction': prediction, 'reference': 'ref'}
                for regime, values in (
                    ('zero-shot', [('q1', 'normal'), ('q2', 'normal')]),
                    ('3-shot', [('q1', 'boa'), ('q2', 'falha')]),
                ) for question, prediction in values
            ]
            escrever_csv(pasta / 'modelo-extended' / 'answers_per_question.csv', respostas)
            plugins = [BertScorePT(BERTFalso()), Phi4JudgePT(JudgeFalso)]
            relatorio = executar(campanha, plugins)
            self.assertEqual(relatorio[pasta]['ok'], ['bert_score_pt', 'phi4_judge_pt'])
            self.assertEqual(len(ler(pasta / 'metricas_phi4_judge_pt.csv')), 4)
            consolidar(campanha)
            resumir_metricas_pt(campanha / 'consolidado')
            por_run = ler(campanha / 'consolidado' / 'metricas_pt_por_execucao.csv')
            juiz = next(r for r in por_run if r['metrica'] == 'phi4_judge'
                        and r['regime'] == '3-shot')
            self.assertEqual((juiz['n_total'], juiz['n_validos']), ('2', '1'))
            delta = ler(campanha / 'consolidado' / 'deltas_metricas_pt.csv')
            juiz_delta = next(r for r in delta if r['metrica'] == 'phi4_judge')
            self.assertEqual(juiz_delta['n_pares'], '1')
            self.assertEqual(float(juiz_delta['delta_medio']), 2.0)

    def test_resumo_preserva_condicao_sem_notas_validas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            escrever_csv(root / 'metricas_phi4_judge_pt.csv', [{
                'exec_modelo': 'gemma4', 'exec_repeticao': 1, 'exec_seed': 42,
                'exec_split_seed': 42, 'exec_persona': 'clinico',
                'regime': 'zero-shot', 'question': 'q1', 'phi4_judge': None,
            }])
            resumir_metricas_pt(root)
            linhas = ler(root / 'resumo_metricas_pt.csv')
            self.assertEqual(len(linhas), 1)
            self.assertEqual((linhas[0]['n_total'], linhas[0]['n_validos']), ('1', '0'))
            self.assertEqual(linhas[0]['media'], '')


if __name__ == '__main__':
    unittest.main()
