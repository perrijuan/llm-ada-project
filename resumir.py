"""Resumo entre repeticoes completas, mantendo cada protocolo separado."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


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
        if n > 1:
            from scipy.stats import t
            half = t.ppf(.975, n - 1) * sd / np.sqrt(n)
        else:
            half = float('nan')
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, key)), 'n_execucoes': n, 'media': mean, 'desvio_padrao': sd,
                     'minimo': values.min(), 'maximo': values.max(),
                     'ic95_media_inferior': mean - half, 'ic95_media_superior': mean + half})
    return pd.DataFrame(rows, columns=[*keys, 'n_execucoes', 'media',
                                       'desvio_padrao', 'minimo', 'maximo',
                                       'ic95_media_inferior', 'ic95_media_superior'])


def resumir_metricas_pt(root):
    """Resume os plugins portugueses por run e compara perguntas pareadas."""
    fontes = {
        'metricas_bert_score_pt.csv': ('bert_p', 'bert_r', 'bert_f1'),
        'metricas_phi4_judge_pt.csv': ('phi4_judge',),
    }
    identificadores = ['exec_modelo', 'exec_repeticao', 'exec_seed',
                      'exec_split_seed', 'exec_persona']
    por_execucao, deltas = [], []
    for arquivo, metricas in fontes.items():
        path = root / arquivo
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        for metrica in metricas:
            if metrica not in df:
                continue
            dados = df[identificadores + ['regime', 'question', metrica]].copy()
            dados[metrica] = pd.to_numeric(dados[metrica], errors='coerce')
            grupos = dados.groupby(identificadores + ['regime'], dropna=False)[metrica]
            agregado = grupos.agg(n_total='size', n_validos='count', media='mean').reset_index()
            agregado['metrica'] = metrica
            por_execucao.extend(agregado.to_dict('records'))

            # A pergunta, e nao a posicao no CSV, identifica o par entre condicoes.
            pares = dados.pivot_table(index=identificadores + ['question'],
                                      columns='regime', values=metrica, aggfunc='first')
            if 'zero-shot' not in pares:
                continue
            for regime in ('3-shot', '5-shot', 'fine-tuned'):
                if regime not in pares:
                    continue
                validos = pares[['zero-shot', regime]].dropna()
                if validos.empty:
                    continue
                diff = (validos[regime] - validos['zero-shot']).rename('delta').reset_index()
                for chave, grupo in diff.groupby(identificadores, dropna=False):
                    chave = chave if isinstance(chave, tuple) else (chave,)
                    deltas.append({**dict(zip(identificadores, chave)), 'metrica': metrica,
                                   'comparacao': regime + ' - zero-shot',
                                   'n_pares': len(grupo), 'delta_medio': grupo['delta'].mean()})

    if por_execucao:
        runs = pd.DataFrame(por_execucao)
        runs.to_csv(root / 'metricas_pt_por_execucao.csv', index=False)
        agrupamento = ['exec_modelo', 'regime', 'metrica']
        agregado = resumo(runs, agrupamento, 'media')
        cobertura = runs.groupby(agrupamento, dropna=False)[['n_total', 'n_validos']].sum().reset_index()
        cobertura.merge(agregado, on=agrupamento, how='left').to_csv(
            root / 'resumo_metricas_pt.csv', index=False)
    if deltas:
        pares = pd.DataFrame(deltas)
        pares.to_csv(root / 'deltas_metricas_pt.csv', index=False)
        resumo(pares, ['exec_modelo', 'metrica', 'comparacao'], 'delta_medio').to_csv(
            root / 'resumo_deltas_metricas_pt.csv', index=False)


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
    resumir_metricas_pt(root)
    print(f'Resumos gravados em {root}. IC entre repeticoes nao substitui incerteza entre perguntas.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pasta', type=Path, nargs='?', default=Path('resultados_icl'))
    main(parser.parse_args().pasta)
