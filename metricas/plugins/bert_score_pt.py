"""BERTScore multilingue na configuracao portuguesa do ADA, por pergunta."""

from metricas.avaliacao_pt.bertscore import BERTScoreConfig, BERTScoreMetric


class BertScorePT:
    nome = 'bert_score_pt'
    escopo = 'run'

    def __init__(self, metric=None):
        self.metric = metric or BERTScoreMetric(BERTScoreConfig())

    def avaliar_pergunta(self, row):
        return {}

    def avaliar_run(self, run):
        linhas = [
            {'pergunta_idx': i, 'regime': row.get('regime'), 'question': row.get('question'),
             'bert_p': None, 'bert_r': None, 'bert_f1': None, 'erro': ''}
            for i, row in enumerate(run.respostas)
        ]
        indices = []
        respostas, referencias = [], []
        for i, row in enumerate(run.respostas):
            resposta = (row.get('prediction') or '').strip()
            referencia = (row.get('reference') or '').strip()
            if not resposta or not referencia:
                linhas[i]['erro'] = 'resposta ou referencia vazia'
                continue
            indices.append(i)
            respostas.append(resposta)
            referencias.append(referencia)
        if indices:
            try:
                result = self.metric.pontuar_detalhado(respostas, referencias)
                if not all(len(values) == len(indices) for values in
                           (result.precision, result.recall, result.f1)):
                    raise ValueError('BERTScore retornou quantidade incorreta de notas')
                for pos, i in enumerate(indices):
                    linhas[i].update(bert_p=result.precision[pos], bert_r=result.recall[pos],
                                     bert_f1=result.f1[pos])
            except Exception as exc:
                erro = f'{type(exc).__name__}: {exc}'
                for i in indices:
                    linhas[i]['erro'] = erro
        return linhas


def criar():
    return BertScorePT()
