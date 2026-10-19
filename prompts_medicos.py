"""Personas adaptadas de fontes publicas; nao sao prompts oficiais dos modelos.

Fontes e decisoes em PROMPTS_E_PROTOCOLO.md. Sem chamadas de rede na execucao.
"""
import os

FONTES = {
    "medagents": "https://github.com/gersteinlab/MedAgents/blob/main/prompt_generator.py",
    "openbiollm": "https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B",
    "tutorial": "https://www.datacamp.com/tutorial/fine-tuning-qwen3-6",
}
PERSONAS = {
    "clinico": {
        "descricao": "Medico generalista; adaptacao da persona por dominio do MedAgents.",
        "fontes": [FONTES["medagents"]],
        "prefixo": (
            "Adopt the perspective of a general physician answering an educational "
            "medical question. Use relevant clinical knowledge and the information "
            "provided in the question. Do not invent patient details or evidence."
        ),
    },
    "biomedico": {
        "descricao": "Especialista em biomedicina; adaptacao da model card OpenBioLLM.",
        "fontes": [FONTES["openbiollm"]],
        "prefixo": (
            "Adopt the perspective of a biomedical researcher answering an educational "
            "health question. Use established biomedical knowledge and distinguish "
            "supported findings from uncertainty. Do not invent studies or citations."
        ),
    },
    "original": {
        "descricao": "Instrucoes preservadas do pacote recebido, sem persona adicional.",
        "fontes": [FONTES["tutorial"]],
        "prefixo": "",
    },
}
QA_INSTRUCTION = "Answer the medical question directly in 2-4 factual sentences."
MCQ_INSTRUCTION = (
    "Select the best supported option for the supplied question. "
    "If a passage is supplied, base your choice on that passage. "
    "Return only the option letter requested in the question."
)

def obter_prompts(persona):
    if persona not in PERSONAS:
        raise ValueError(f"Persona desconhecida: {persona}. Opcoes: {list(PERSONAS)}")
    if persona == "original":
        return QA_INSTRUCTION, "You are a medical expert. Choose the single best option."
    prefixo = PERSONAS[persona]["prefixo"]
    return prefixo + " " + QA_INSTRUCTION, prefixo + " " + MCQ_INSTRUCTION

def configuracao():
    seed = int(os.getenv("LLM_SEED", "42"))
    split_seed = int(os.getenv("LLM_SPLIT_SEED", "42"))
    persona = os.getenv("LLM_PERSONA", "clinico")
    if not 0 <= seed < 2**32 or not 0 <= split_seed < 2**32:
        raise ValueError("Seeds devem estar entre 0 e 2**32 - 1.")
    system, mcq = obter_prompts(persona)
    return {
        "seed": seed, "split_seed": split_seed, "persona": persona,
        "system_prompt": system, "mcq_system_prompt": mcq,
        "fontes_persona": PERSONAS[persona]["fontes"],
        "protocolo": "icl_0_3_5_ft5_personas_v3",
    }

