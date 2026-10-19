"""Builtin, escopo run: media/p50/p95 de latencia e vazao por regime + tempo e loss do treino."""
import math

import numpy as np

COLUNAS = ['latency_s', 'ttft_s', 'tokens_per_s', 'decode_tokens_per_s', 'peak_vram_gb']


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
            linha = {'regime': regime, 'n_perguntas': len(rows)}
            for coluna in COLUNAS:
                # peak_vram_gb so entra se existir no CSV; hoje fica no summary_df (gen_peak_vram_gb).
                if not any(coluna in row for row in rows):
                    continue
                valores = [v for v in (numero(row.get(coluna)) for row in rows) if v is not None]
                if valores:
                    linha[f'{coluna}_media'] = float(np.mean(valores))
                    linha[f'{coluna}_p50'] = float(np.percentile(valores, 50))
                    linha[f'{coluna}_p95'] = float(np.percentile(valores, 95))
                else:
                    linha[f'{coluna}_media'] = linha[f'{coluna}_p50'] = linha[f'{coluna}_p95'] = None
            # So o regime fine-tuned treina; os outros ficam vazios.
            treino = run.treino.get(regime, {})
            linha['train_runtime'] = numero(treino.get('train_runtime'))
            linha['training_loss'] = numero(treino.get('training_loss'))
            linhas.append(linha)
        return linhas


def criar():
    return Desempenho()
