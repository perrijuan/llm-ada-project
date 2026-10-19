"""Exemplo de modulo externo, escopo pergunta: BERTScore entre prediction e reference.

Template para novos plugins: expor criar() que devolve um objeto com nome, escopo,
avaliar_pergunta(row) e avaliar_run(run). Requer: pip install bert-score
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
