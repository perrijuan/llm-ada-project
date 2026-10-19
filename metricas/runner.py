"""Aplica os plugins em cada run da campanha e grava metricas_<nome>.csv na pasta da run."""
import csv
import json
from pathlib import Path

from metricas.base import RunContext
from registro import escrever_csv


def ler_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def ler_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def montar_contexto(pasta):
    pasta = Path(pasta)
    status = ler_json(pasta / 'status.json')
    ambiente = ler_json(pasta / 'ambiente.json')
    modelo = status.get('modelo') or ambiente.get('modelo') or pasta.parent.name
    # ambiente.json atual nao grava o ID do Hugging Face; model_config.json tem _name_or_path.
    modelo_id = (ambiente.get('modelo_id') or ambiente.get('model_id')
                 or ler_json(pasta / 'registros' / 'model_config.json').get('_name_or_path') or modelo)
    respostas = []
    for path in sorted(pasta.rglob('answers_per_question.csv')):
        respostas.extend(ler_csv(path))
    treino = {}
    for path in sorted((pasta / 'registros').glob('treino_*_metricas.csv')):
        tag = path.name[len('treino_'):-len('_metricas.csv')]
        linhas = ler_csv(path)
        if linhas:
            treino[tag] = linhas[-1]
    return RunContext(pasta=pasta, modelo_id=modelo_id, modelo=modelo, status=status,
                      respostas=respostas, treino=treino)


def aplicar(plugin, run):
    if plugin.escopo == 'pergunta':
        linhas = []
        for i, row in enumerate(run.respostas):
            # regime/question/indice permitem juntar com answers_per_question.csv depois.
            linhas.append({'pergunta_idx': i, 'regime': row.get('regime'),
                           'question': row.get('question'), **plugin.avaliar_pergunta(row)})
        return linhas
    resultado = plugin.avaliar_run(run)
    return [resultado] if isinstance(resultado, dict) else list(resultado)


def executar(campanha: Path, plugins):
    # Retorna {pasta_da_run: {'ok': [nomes], 'falhou': {nome: erro}, 'ignorada': motivo}}.
    relatorio = {}
    for pasta in sorted(Path(campanha).glob('*/run_*')):
        if not pasta.is_dir():
            continue
        run = montar_contexto(pasta)
        item = {'ok': [], 'falhou': {}, 'ignorada': None}
        relatorio[pasta] = item
        # consolidar.py so junta runs completas; parciais ficam de fora aqui tambem.
        if run.status.get('status') != 'ok':
            item['ignorada'] = f"status={run.status.get('status', 'sem status.json')}"
            continue
        for plugin in plugins:
            try:
                linhas = aplicar(plugin, run)
                escrever_csv(pasta / f'metricas_{plugin.nome}.csv', linhas)
                item['ok'].append(plugin.nome)
            except Exception as exc:
                item['falhou'][plugin.nome] = f'{type(exc).__name__}: {exc}'
    return relatorio
