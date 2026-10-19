"""Resumo entre repeticoes completas, mantendo cada protocolo separado."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t


def resumo(df, keys, value):
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        values = pd.to_numeric(group[value], errors='coerce')
        values = values[np.isfinite(values)]
        n = len(values)
        if not n:
            continue
        mean = values.mean()
        sd = values.std(ddof=1) if n > 1 else float('nan')
        half = t.ppf(.975, n - 1) * sd / np.sqrt(n) if n > 1 else float('nan')
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, key)), 'n_execucoes': n, 'media': mean, 'desvio_padrao': sd,
                     'minimo': values.min(), 'maximo': values.max(),
                     'ic95_media_inferior': mean - half, 'ic95_media_superior': mean + half})
    return pd.DataFrame(rows)


def main(root):
    root = Path(root) / 'consolidado'
    path = root / 'benchmark_summary.csv'
    if path.exists():
        df = pd.read_csv(path)
        keys = ['exec_modelo', 'category', 'metric', 'condition']
        resumo(df, keys, 'mean').to_csv(root / 'resumo_entre_execucoes.csv', index=False)
        paired = df.pivot(index=['exec_modelo', 'exec_repeticao', 'category', 'metric'], columns='condition', values='mean')
        comparisons = []
        for condition in ['3-shot', '5-shot', 'fine-tuned']:
            if {'zero-shot', condition} <= set(paired.columns):
                part = paired[[condition, 'zero-shot']].dropna().copy()
                part['delta'] = part[condition] - part['zero-shot']
                part['comparacao'] = condition + ' - zero-shot'
                comparisons.append(part.reset_index()[['exec_modelo', 'exec_repeticao', 'category', 'metric', 'comparacao', 'delta']])
        if comparisons:
            deltas = pd.concat(comparisons, ignore_index=True)
            deltas.to_csv(root / 'deltas_pareados.csv', index=False)
            resumo(deltas, ['exec_modelo', 'category', 'metric', 'comparacao'], 'delta').to_csv(root / 'resumo_deltas.csv', index=False)
    path = root / 'metrics_summary.csv'
    if path.exists():
        df = pd.read_csv(path)
        identifiers = ['exec_modelo', 'exec_repeticao', 'exec_seed', 'exec_split_seed', 'exec_persona', 'exec_origem', 'regime']
        values = [c for c in df if c not in identifiers]
        long = df.melt(id_vars=identifiers, value_vars=values, var_name='metric', value_name='value')
        resumo(long, ['exec_modelo', 'regime', 'metric'], 'value').to_csv(root / 'resumo_medquad_condicoes.csv', index=False)
    print(f'Resumos gravados em {root}. IC entre repeticoes nao substitui incerteza entre perguntas.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pasta', type=Path, nargs='?', default=Path('resultados_icl'))
    main(parser.parse_args().pasta)
