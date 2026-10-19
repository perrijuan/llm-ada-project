"""Metricas extras calculadas depois das runs, a partir dos CSVs salvos."""
from metricas.base import MetricaPlugin, RunContext
from metricas.loader import carregar_plugins
from metricas.runner import executar

__all__ = ['MetricaPlugin', 'RunContext', 'carregar_plugins', 'executar']
