"""Phi-4 Judge com o prompt portugues do ADA, por resposta salva."""

from pathlib import Path

from metricas.avaliacao_pt.llm_judge import (
    JudgeTraceContext, LLMJudgeConfig, LLMJudgeMetric,
)


class Phi4JudgePT:
    nome = 'phi4_judge_pt'
    escopo = 'run'

    def __init__(self, metric_factory=None):
        self.metric_factory = metric_factory or LLMJudgeMetric

    def avaliar_pergunta(self, row):
        return {}

    def avaliar_run(self, run):
        config = LLMJudgeConfig(
            language='pt', keep_alive='5m',
            log_path=Path(run.pasta) / 'registros' / 'phi4_judge_pt_chamadas.csv',
        )
        metric = self.metric_factory(config)
        linhas = []
        for i, row in enumerate(run.respostas):
            linha = {'pergunta_idx': i, 'regime': row.get('regime'),
                     'question': row.get('question'), 'phi4_judge': None,
                     'erro': '', 'tentativas': 0}
            resposta = (row.get('prediction') or '').strip()
            referencia = (row.get('reference') or '').strip()
            if not resposta or not referencia:
                linha['erro'] = 'resposta ou referencia vazia'
            else:
                try:
                    contexto = JudgeTraceContext(
                        run_id=f'{run.modelo}/{run.pasta.name}',
                        source_file='answers_per_question.csv', row_id=str(i),
                    )
                    result = metric.avaliar_detalhado(resposta, referencia, contextos=contexto)
                    linha.update(phi4_judge=result.nota, erro=result.erro or '',
                                 tentativas=result.tentativas)
                except Exception as exc:
                    linha['erro'] = f'{type(exc).__name__}: {exc}'
            linhas.append(linha)
        return linhas


def criar():
    return Phi4JudgePT()
