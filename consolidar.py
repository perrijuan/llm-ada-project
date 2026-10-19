"""Junta CSVs preservando modelo, repeticao, condicao e origem das metricas."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from registro import escrever_csv


def consolidar(root):
    root = Path(root)
    groups = defaultdict(list)
    runs = []
    versions = {json.loads(p.read_text(encoding='utf-8')).get('protocolo', 'legacy')
                for p in root.glob('*/run_*/status.json')}
    if len(versions) > 1:
        raise ValueError('Protocolos diferentes na mesma campanha. Use uma --saida nova para ICL.')
    for status_path in sorted(root.glob('*/run_*/status.json')):
        status = json.loads(status_path.read_text(encoding='utf-8'))
        runs.append(status)
        # Resultados parciais ficam na pasta original, fora da comparacao final.
        if status['status'] != 'ok':
            continue
        folder = status_path.parent
        for path in sorted(folder.rglob('*.csv')):
            if path.name.startswith('dados_'):
                continue
            with path.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            # O nome do arquivo diferencia benchmark, extended e registro.
            key = ('registro__' if path.parent.name == 'registros' else '') + path.name
            for row in rows:
                groups[key].append({'exec_modelo': status['modelo'], 'exec_repeticao': status['repeticao'],
                                    'exec_seed': status['seed'],
                                    'exec_split_seed': status.get('split_seed', 42),
                                    'exec_persona': status.get('persona', 'original'),
                                    'exec_origem': path.relative_to(folder).as_posix(),
                                    **row})
    out = root / 'consolidado'
    out.mkdir(parents=True, exist_ok=True)
    escrever_csv(out / 'execucoes.csv', runs)
    for name, rows in groups.items():
        escrever_csv(out / name, rows)
    return groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pasta', type=Path, nargs='?', default=Path('resultados_icl'))
    consolidar(parser.parse_args().pasta)
