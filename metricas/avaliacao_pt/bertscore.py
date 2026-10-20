"""BERTScore para português com suporte a entradas individuais e em batch."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, overload
import warnings

from .base import EntradaMetrica, SaidaMetrica, normalizar_entradas


@dataclass(frozen=True, slots=True)
class BERTScoreConfig:
    """Configuração explícita e reutilizável do BERTScore."""

    model_type: str = "bert-base-multilingual-cased"
    language: str = "pt"
    num_layers: int = 9
    idf: bool = False
    rescale_with_baseline: bool = False
    use_fast_tokenizer: bool = False
    device: str = "auto"
    batch_size: int = 64
    nthreads: int = 4

    def __post_init__(self) -> None:
        if not self.model_type.strip():
            raise ValueError("model_type não pode ser vazio")
        if not self.language.strip():
            raise ValueError("language não pode ser vazio")
        if self.num_layers < 0:
            raise ValueError("num_layers deve ser maior ou igual a zero")
        if not self.device.strip():
            raise ValueError("device não pode ser vazio")
        if self.batch_size <= 0:
            raise ValueError("batch_size deve ser maior que zero")
        if self.nthreads < 0:
            raise ValueError("nthreads deve ser maior ou igual a zero")


ValorDetalhado = float | list[float]


@dataclass(frozen=True, slots=True)
class BERTScoreResultado:
    """Precision, Recall e F1 preservando o modo da entrada."""

    precision: ValorDetalhado
    recall: ValorDetalhado
    f1: ValorDetalhado


class BERTScoreMetric:
    """Calcula BERTScore reutilizando o modelo entre chamadas equivalentes."""

    def __init__(self, config: BERTScoreConfig | None = None) -> None:
        self.config = config or BERTScoreConfig()

    @overload
    def metrica(self, resposta: str, referencia: str) -> float: ...

    @overload
    def metrica(
        self, resposta: list[str], referencia: list[str]
    ) -> list[float]: ...

    def metrica(
        self, resposta: EntradaMetrica, referencia: EntradaMetrica
    ) -> SaidaMetrica:
        return self.pontuar_detalhado(resposta, referencia).f1

    @overload
    def pontuar_detalhado(
        self, resposta: str, referencia: str
    ) -> BERTScoreResultado: ...

    @overload
    def pontuar_detalhado(
        self, resposta: list[str], referencia: list[str]
    ) -> BERTScoreResultado: ...

    def pontuar_detalhado(
        self, resposta: EntradaMetrica, referencia: EntradaMetrica
    ) -> BERTScoreResultado:
        entradas = normalizar_entradas(resposta, referencia)
        if not entradas.respostas:
            return BERTScoreResultado([], [], [])

        device = _resolver_device(self.config.device)
        scorer = _obter_scorer(self.config, device)
        precision, recall, f1 = scorer.score(
            entradas.respostas,
            entradas.referencias,
            batch_size=self.config.batch_size,
            return_hash=False,
        )

        tamanho = len(entradas.respostas)
        precision_lista = _converter_pontuacoes("precision", precision, tamanho)
        recall_lista = _converter_pontuacoes("recall", recall, tamanho)
        f1_lista = _converter_pontuacoes("f1", f1, tamanho)

        if entradas.entrada_individual:
            return BERTScoreResultado(
                precision_lista[0], recall_lista[0], f1_lista[0]
            )
        return BERTScoreResultado(precision_lista, recall_lista, f1_lista)

    @property
    def configuration_hash(self) -> str:
        """Hash de configuração produzido pela biblioteca BERTScore."""

        device = _resolver_device(self.config.device)
        return str(_obter_scorer(self.config, device).hash)


def _carregar_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch não está instalado. Instale as dependências do projeto."
        ) from exc
    return torch


def _resolver_device(device_configurado: str) -> str:
    torch = _carregar_torch()
    solicitado = device_configurado.strip().lower()

    if solicitado == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    try:
        torch.device(solicitado)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"device inválido: {device_configurado!r}") from exc

    if solicitado.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            f"Device {device_configurado!r} solicitado, mas CUDA não está "
            "disponível; usando CPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu"

    if solicitado.startswith("mps"):
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            warnings.warn(
                f"Device {device_configurado!r} solicitado, mas MPS não está "
                "disponível; usando CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "cpu"

    return solicitado


@lru_cache(maxsize=None)
def _obter_scorer(config: BERTScoreConfig, device: str) -> Any:
    try:
        from bert_score import BERTScorer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "bert-score não está instalado. Instale as dependências do projeto."
        ) from exc

    return BERTScorer(
        model_type=config.model_type,
        num_layers=config.num_layers,
        batch_size=config.batch_size,
        nthreads=config.nthreads,
        all_layers=False,
        idf=config.idf,
        device=device,
        lang=config.language,
        rescale_with_baseline=config.rescale_with_baseline,
        use_fast_tokenizer=config.use_fast_tokenizer,
    )


def _converter_pontuacoes(nome: str, valores: Any, tamanho: int) -> list[float]:
    if hasattr(valores, "detach"):
        valores = valores.detach()
    if hasattr(valores, "cpu"):
        valores = valores.cpu()
    if hasattr(valores, "tolist"):
        valores = valores.tolist()

    try:
        resultado = [float(valor) for valor in valores]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"BERTScore retornou valores incompatíveis para {nome}"
        ) from exc

    if len(resultado) != tamanho:
        raise RuntimeError(
            f"BERTScore retornou {len(resultado)} valores para {nome}; "
            f"esperados {tamanho}"
        )
    return resultado


_METRICA_PADRAO = BERTScoreMetric()


@overload
def metrica(resposta: str, referencia: str) -> float: ...


@overload
def metrica(resposta: list[str], referencia: list[str]) -> list[float]: ...


def metrica(
    resposta: EntradaMetrica, referencia: EntradaMetrica
) -> SaidaMetrica:
    """Retorna o F1 do BERTScore na interface comum do projeto."""

    return _METRICA_PADRAO.metrica(resposta, referencia)


@overload
def pontuar_detalhado(
    resposta: str, referencia: str
) -> BERTScoreResultado: ...


@overload
def pontuar_detalhado(
    resposta: list[str], referencia: list[str]
) -> BERTScoreResultado: ...


def pontuar_detalhado(
    resposta: EntradaMetrica, referencia: EntradaMetrica
) -> BERTScoreResultado:
    """Retorna Precision, Recall e F1 sem recarregar o modelo padrão."""

    return _METRICA_PADRAO.pontuar_detalhado(resposta, referencia)


__all__ = [
    "BERTScoreConfig",
    "BERTScoreMetric",
    "BERTScoreResultado",
    "metrica",
    "pontuar_detalhado",
]
