# Registro da validação

Data: 13 de outubro de 2026.

Protocolo atual: `icl_0_3_5_ft5_v2`. Os oito testes em `testes/test_automacao.py` passaram:

1. Configuração das quatro condições, learning rate 2e-5, cinco épocas, AdamW, batch efetivo 32, split 80/20 e max_steps=-1 nos quatro scripts.
2. CSV com texto Unicode, aspas, vírgulas, quebras de linha, listas JSON e colunas de log variáveis.
3. Vinte subprocessos artificiais, dez para cada um de dois modelos simulados, incluindo isolamento de diretório, consolidação, retomada e proteção contra sobrescrita.
4. Exclusão de execuções com erro dos CSVs consolidados de resultados.
5. Interrupção da campanha no primeiro erro, preservando status e log.
6. Preservação estrutural das funções originais de F1 lexical, ROUGE-L, intervalo por pergunta, seleção de IDs de letras e formatação de MedQA/PubMedQA.
7. Seleção de demonstrações sem pergunta do teste, sem duplicatas exatas, pareamento 3/5-shot, mensagens sem resposta-alvo e fórmula de macro precision/recall/F1.
8. Execução do laço real de cada script com objetos simulados: quatro carregamentos, três avaliações completas sem treino, uma chamada de treino no final e liberação das referências ao modelo.

Também foi verificada a sintaxe de todos os arquivos Python e a ordem de quarenta execuções no modo `--planejar`.

O teste de subprocessos substitui apenas a barra externa `tqdm` por um iterador comum no ambiente de teste. Ele não instala ou exercita o Unsloth, CUDA, Transformers, modelos ou datasets. Não é evidência de treinamento de LLM.

O ambiente de validação não possui PyTorch, tqdm ou SciPy instalados. O script estatístico `resumir.py` recebeu inspeção e validação de sintaxe, mas sua integração com SciPy e os benchmarks reais deverá ser conferida no ambiente preparado com `requirements.txt`. Não foram inventadas versões de dependências nem métricas experimentais.

Os hashes em `manifesto.json` identificam os arquivos de entrada e de modelo entregues. Os diffs permitem revisar todas as alterações. O ZIP inclui os originais recebidos. Esta versão altera intencionalmente o protocolo: não promete identidade completa com o código inicial. NLL e construção dos prompts foram adaptadas para suportar demonstrações e impedir truncamento silencioso.

Para repetir os testes no seu ambiente, dentro da pasta do pacote:

```bash
python -m unittest discover -s testes -v
python rodar_todos.py --planejar
```
