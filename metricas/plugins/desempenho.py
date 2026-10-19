"""Builtin, escopo run: metricas [METRICS] de velocidade por regime + tempo e loss do treino."""
import json
import math

import numpy as np


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
            prefill = [v for v in (numero(row.get('prefill_s')) for row in rows) if v is not None]
            ttft = [v for v in (numero(row.get('ttft_s')) for row in rows) if v is not None]
            prompt_tokens = [v for v in (numero(row.get('prompt_tokens')) for row in rows) if v is not None]
            generated_tokens = [v for v in (numero(row.get('new_tokens')) for row in rows) if v is not None]
            latency = [v for v in (numero(row.get('latency_s')) for row in rows) if v is not None]
            rss = [v for v in (numero(row.get('peak_rss_bytes')) for row in rows) if v is not None]

            intervals = []
            for row in rows:
                try:
                    times = json.loads(row.get('token_times', ''))
                    if not isinstance(times, list):
                        continue
                    times = [numero(value) for value in times]
                    if any(value is None for value in times):
                        continue
                    intervals.extend((b - a) * 1000 for a, b in zip(times, times[1:]))
                except (TypeError, ValueError):
                    continue

            total_time = sum(latency)
            linha = {
                'regime': regime,
                'n_perguntas': len(rows),
                'prefill_ms': float(np.mean(prefill)) * 1000 if prefill else None,
                'decode_mean_ms': float(np.mean(intervals)) if intervals else None,
                'decode_p95_ms': float(np.percentile(intervals, 95)) if intervals else None,
                'ttft_ms': float(np.mean(ttft)) * 1000 if ttft else None,
                'overall_tok/s': sum(generated_tokens) / total_time if generated_tokens and total_time > 0 else None,
                'peak_rss_bytes': max(rss) if rss else None,
                'prompt_tokens': float(np.mean(prompt_tokens)) if prompt_tokens else None,
                'generated_tokens': float(np.mean(generated_tokens)) if generated_tokens else None,
            }
            # So o regime fine-tuned treina; os outros ficam vazios.
            treino = run.treino.get(regime, {})
            linha['train_runtime'] = numero(treino.get('train_runtime'))
            linha['training_loss'] = numero(treino.get('training_loss'))
            linhas.append(linha)
        return linhas


def criar():
    return Desempenho()
