"""Contrato dos plugins e dados de uma run lidos do disco."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

ESCOPOS = ('pergunta', 'run')


@dataclass
class RunContext:
    pasta: Path
    modelo_id: str
    modelo: str
    status: dict
    respostas: list = field(default_factory=list)  # linhas de answers_per_question.csv
    treino: dict = field(default_factory=dict)  # tag -> linha de registros/treino_<tag>_metricas.csv


@runtime_checkable
class MetricaPlugin(Protocol):
    nome: str  # vira metricas_<nome>.csv
    escopo: str  # "pergunta" (uma vez por linha) ou "run" (uma vez por run)

    def avaliar_pergunta(self, row: dict) -> dict:
        ...

    def avaliar_run(self, run: RunContext) -> dict | list[dict]:
        ...
