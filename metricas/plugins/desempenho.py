"""Builtin, escopo run: media/mediana/p99_pior_caso de latencia e vazao por regime + tempo e loss do treino.

Mesmo criterio de mean_median_p99 dos scripts em modelos/: P99 onde menor e melhor (tempos)
e P1 onde maior e melhor (tokens/s), a cauda lenta. Com n<100 o P99 e quase o maximo.
"""
import math

import numpy as np

# coluna -> o que e melhor
COLUNAS = {'latency_s': 'lower', 'ttft_s': 'lower', 'tokens_per_s': 'higher',
           'decode_tokens_per_s': 'higher', 'peak_vram_gb': 'lower'}


def numero(valor):
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(valor) else valor


def mean_median_p99(valores, better='lower'):
    tail = 99 if better == 'lower' else 1
    return float(np.mean(valores)), float(np.median(valores)), float(np.percentile(valores, tail))


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
            for coluna, better in COLUNAS.items():
                # peak_vram_gb so entra se existir no CSV; hoje fica no summary_df (gen_peak_vram_gb).
                if not any(coluna in row for row in rows):
                    continue
                valores = [v for v in (numero(row.get(coluna)) for row in rows) if v is not None]
                media, mediana, p99 = mean_median_p99(valores, better) if valores else (None, None, None)
                linha[f'{coluna}_mean'] = media
                linha[f'{coluna}_median'] = mediana
                linha[f'{coluna}_p99_pior_caso'] = p99
            # So o regime fine-tuned treina; os outros ficam vazios.
            treino = run.treino.get(regime, {})
            linha['train_runtime'] = numero(treino.get('train_runtime'))
            linha['training_loss'] = numero(treino.get('training_loss'))
            linhas.append(linha)
        return linhas


def criar():
    return Desempenho()
