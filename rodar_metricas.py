"""Calcula metricas extras nas runs de uma campanha e grava metricas_<nome>.csv em cada run."""
import argparse
import sys
from pathlib import Path

from metricas import carregar_plugins, executar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pasta', type=Path, help='pasta da campanha (ex.: campanha_clinico)')
    parser.add_argument('--plugins', nargs='+', default=['metricas.plugins.desempenho'],
                        help='modulos que expoem criar() -> MetricaPlugin')
    parser.add_argument('--ignorar-faltantes', action='store_true',
                        help='segue com os plugins que importaram, em vez de parar')
    args = parser.parse_args()

    plugins, erros = carregar_plugins(args.plugins)
    for erro in erros:
        print(f'[plugin] {erro}', flush=True)
    if erros and not args.ignorar_faltantes:
        print('Plugins indisponiveis. Instale as dependencias ou use --ignorar-faltantes.')
        return 1
    if not plugins:
        print('Nenhum plugin carregado.')
        return 1

    relatorio = executar(args.pasta, plugins)
    if not relatorio:
        print(f'Nenhuma run encontrada em {args.pasta}/*/run_*.')
        return 1
    falhas = False
    for pasta, item in relatorio.items():
        nome = pasta.relative_to(args.pasta).as_posix()
        if item['ignorada']:
            print(f'{nome}: ignorada ({item["ignorada"]})')
            continue
        print(f'{nome}: ok={item["ok"] or "-"}')
        for plugin, erro in item['falhou'].items():
            falhas = True
            print(f'  falhou {plugin}: {erro}')
    print('Para juntar nas tabelas: python consolidar.py', args.pasta)
    return 1 if falhas else 0


if __name__ == '__main__':
    sys.exit(main())
