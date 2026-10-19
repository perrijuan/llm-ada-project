"""Importa os modulos de plugin. Cada modulo precisa expor criar() -> MetricaPlugin."""
import importlib

from metricas.base import ESCOPOS, MetricaPlugin


def carregar_plugins(modulos):
    # Retorna (plugins, erros); erros e uma lista de mensagens, uma por modulo que falhou.
    plugins, erros = [], []
    for modulo in modulos:
        try:
            mod = importlib.import_module(modulo)
        except ModuleNotFoundError as exc:
            if exc.name and (modulo == exc.name or modulo.startswith(exc.name + '.')):
                erros.append(f'{modulo}: modulo nao encontrado (confira o nome e o PYTHONPATH).')
            else:
                erros.append(f"{modulo}: dependencia '{exc.name}' nao instalada "
                             f'(ex.: pip install {exc.name}).')
            continue
        except Exception as exc:
            erros.append(f'{modulo}: erro ao importar ({type(exc).__name__}: {exc}).')
            continue
        if not callable(getattr(mod, 'criar', None)):
            erros.append(f'{modulo}: nao expoe criar() -> MetricaPlugin.')
            continue
        try:
            plugin = mod.criar()
        except Exception as exc:
            erros.append(f'{modulo}: criar() falhou ({type(exc).__name__}: {exc}).')
            continue
        if not isinstance(plugin, MetricaPlugin) or plugin.escopo not in ESCOPOS:
            erros.append(f'{modulo}: plugin precisa de nome, escopo em {ESCOPOS}, '
                         'avaliar_pergunta e avaliar_run.')
            continue
        plugins.append(plugin)
    return plugins, erros
