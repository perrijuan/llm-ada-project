"""Contratos e validações compartilhados pelas métricas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, overload


EntradaMetrica = str | list[str]
SaidaMetrica = float | list[float]


class Metrica(Protocol):
    """Interface pública comum que toda métrica escalar deve oferecer."""

    @overload
    def __call__(self, resposta: str, referencia: str) -> float: ...

    @overload
    def __call__(
        self, resposta: list[str], referencia: list[str]
    ) -> list[float]: ...

    def __call__(
        self, resposta: EntradaMetrica, referencia: EntradaMetrica
    ) -> SaidaMetrica: ...


@dataclass(frozen=True, slots=True)
class EntradasNormalizadas:
    respostas: list[str]
    referencias: list[str]
    entrada_individual: bool


def normalizar_entradas(
    resposta: EntradaMetrica, referencia: EntradaMetrica
) -> EntradasNormalizadas:
    """Valida os dois modos suportados e normaliza escalares para listas."""

    if isinstance(resposta, str) and isinstance(referencia, str):
        return EntradasNormalizadas([resposta], [referencia], True)

    if isinstance(resposta, list) and isinstance(referencia, list):
        if len(resposta) != len(referencia):
            raise ValueError(
                "resposta e referencia devem ter o mesmo tamanho no modo batch; "
                f"recebidos {len(resposta)} e {len(referencia)}"
            )

        _validar_lista_textual("resposta", resposta)
        _validar_lista_textual("referencia", referencia)
        return EntradasNormalizadas(list(resposta), list(referencia), False)

    raise TypeError(
        "resposta e referencia devem ser ambas str ou ambas list[str]; "
        f"recebidos {type(resposta).__name__} e {type(referencia).__name__}"
    )


def _validar_lista_textual(nome: str, valores: list[object]) -> None:
    for indice, valor in enumerate(valores):
        if not isinstance(valor, str):
            raise TypeError(
                f"{nome}[{indice}] deve ser str; recebido {type(valor).__name__}"
            )
