# Revisão personas v3 — 14/10/2026

Origem: experimento_llms_icl.zip fornecido pelo usuário.
Os originais foram comparados byte a byte com o ZIP e estão preservados.

Mudanças:
- Catálogo de personas adaptadas de MedAgents e OpenBioLLM, com fontes.
- Persona idêntica nos braços 0/3/5-shot e fine-tuned.
- Seeds de execução configuráveis; seed de seleção/split fixa separada.
- Metadados de persona e seeds em ambiente, status, prompts e consolidação.
- Manifesto de campanha e hash dos scripts auxiliares para retomada.
- Proteção explícita contra shots fora de 0/3/5 e few-shot após fine-tuning.
- Executor mantém um processo por vez e conclui uma família antes da seguinte.
- Treino, métricas e checkpoints do pacote recebido preservados.

Validação: 11 testes unittest passaram (8 existentes adaptados + 3 novos).
Incluem processos artificiais reais, sequência, interrupção em erro,
retomada, seed variável, exclusão de teste dos bancos e separação dos braços.
Análise sintática de todos os arquivos Python concluída.
Plano de 12 execuções gerado sem importar bibliotecas de modelos.

Não executados: download de checkpoints/datasets, inferência, fine-tuning ou
validação de compatibilidade CUDA/Unsloth de cada família. Não há resultados
experimentais novos nem garantia de reprodução dos números do paper.

Os relatórios e diffs anteriores neste pacote são históricos.
