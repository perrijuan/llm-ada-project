"""Preparacao de SFT separada de ICL; nao importa modelos nem requer GPU."""
import re


def chave_pergunta(value):
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def indices_demonstracoes(dataset, treino, avaliacao):
    """Reserva para ICL somente perguntas externas ao treino E ao teste."""
    proibidas = {chave_pergunta(row['Question'])
                 for parte in (treino, avaliacao) for row in parte}
    indices = [i for i, row in enumerate(dataset)
               if chave_pergunta(row['Question']) not in proibidas]
    if len({chave_pergunta(dataset[i]['Question']) for i in indices}) < 5:
        raise ValueError('Necessarias 5 perguntas exclusivas para few-shot, fora do treino e teste.')
    return indices


def textos_para_tokenizacao(formatted):
    """Materializa datasets.Column em list[str], sem converter dados invalidos."""
    texts = list(formatted['text'])
    if not texts:
        raise ValueError('Dataset de fine-tuning vazio.')
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Texto de fine-tuning invalido no indice {index}: '
                             f'esperado str nao vazia, recebido {type(text).__name__}.')
    return texts
