# Adaptacao de medgemma15_4b_medquad_unsloth.ipynb. Script separado e direto.
from unsloth import FastModel, is_bfloat16_supported
import gc
import hashlib
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import set_seed
from transformers.generation.streamers import BaseStreamer
from tqdm.auto import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts_medicos import configuracao
from finetuning_dados import textos_para_tokenizacao, indices_demonstracoes
from registro import iniciar, etapa, registrar_trainer, salvar_resultado, salvar_tabelas, salvar_figuras, escrever_csv
iniciar('medgemma')
MODEL_ID = "unsloth/medgemma-1.5-4b-it"
OUTPUT_DIR = "medgemma15-4b-medquad-unsloth"
MODEL_NAME = "MedGemma 1.5 4B"



# 1. Tres inferencias SEM TREINO, seguidas de fine-tuning separado.
SAMPLE_SIZE = 240
EVAL_SIZE = 48  # 20%; os outros 192 exemplos ficam no treino.
MAX_SEQ_LENGTH = 768  # comprimento de treinamento do tutorial
MODEL_CONTEXT_LENGTH = 16384  # limite para demonstracoes completas na inferencia
COMPARISON_SAMPLE_COUNT = 3
PROMPT_CONFIG = configuracao()
SEED = PROMPT_CONFIG['seed']
SPLIT_SEED = PROMPT_CONFIG['split_seed']
PERSONA = PROMPT_CONFIG['persona']
LOAD_IN_4BIT = True
FINE_TUNING_EPOCHS = 5
LEARNING_RATE = 2e-5
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16  # batch efetivo 32 em UMA GPU
CONDITIONS = [('zero-shot', 0, False), ('3-shot', 3, False),
              ('5-shot', 5, False), ('fine-tuned', 0, True)]
SPEED_N = 20
SPEED_MAX_NEW_TOKENS = 160
GEN_MAX_NEW_TOKENS = 160
MEDQA_N = 100
PUBMEDQA_N = 100
SYSTEM_PROMPT = PROMPT_CONFIG['system_prompt']
MCQ_SYSTEM_PROMPT = PROMPT_CONFIG['mcq_system_prompt']
PUBMEDQA_LETTER = {'yes': 'A', 'no': 'B', 'maybe': 'C'}
DATASET_ID = 'keivalya/MedQuad-MedicalQnADataset'
EXTENDED_DIR = f'{OUTPUT_DIR}-extended'
BENCH_DIR = f'{OUTPUT_DIR}-benchmark'
os.makedirs(EXTENDED_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)
set_seed(SEED)
random.seed(SEED)
if not torch.cuda.is_available():
    raise RuntimeError('Este experimento requer NVIDIA/CUDA.')
if int(os.getenv('WORLD_SIZE', '1')) != 1:
    raise RuntimeError('Use uma GPU por processo para manter o batch efetivo 32.')


BAD_ANSWER_PREFIXES = ("These resources address", "These resources from MedlinePlus", "For more information, go to", "For more information go to")

def clean_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text

def keep_example(example):
    question = clean_text(example["Question"])
    answer = clean_text(example["Answer"])

    if len(question) < 12 or len(answer) < 40:
        return False
    if len(answer) > 900:
        return False
    if any(answer.startswith(prefix) for prefix in BAD_ANSWER_PREFIXES):
        return False
    return True

def apply_chat_template(messages, add_generation_prompt=False):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        text = text_tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,  # only used by templates that support it (Qwen)
            **kwargs,
        )
    except TypeError:
        text = text_tokenizer.apply_chat_template(
            messages,
            chat_template_kwargs={"enable_thinking": False},
            **kwargs,
        )
    return strip_bos(text)

def strip_bos(text):
    bos = text_tokenizer.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos) :]
    return text

def clean_answer_text(answer):
    answer = clean_text(answer)
    answer = re.sub(r"^answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    return answer

def make_user_prompt(question):
    return (
        f"Question: {clean_text(question)}\n\n"
        "Respond with only the answer. Do not restate the question."
    )

def format_training_example(example):
    # SFT: um par supervisionado por registro, sem make_messages/demonstracoes ICL.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": make_user_prompt(example["Question"]),
        },
        {
            "role": "assistant",
            "content": clean_answer_text(example["Answer"]),
        },
    ]
    return {"text": apply_chat_template(messages, add_generation_prompt=False)}

def clip_text(text, max_chars=600):
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_chars else text[:max_chars] + "..."

def medqa_prompt(row):
    options = "\n".join(f"{letter}. {row['options'][letter]}" for letter in "ABCD")
    return (
        f"Question: {clean_text(row['question'])}\n\nOptions:\n{options}\n\n"
        "Answer with only the letter (A, B, C or D)."
    )

def pubmedqa_prompt(row):
    context = clean_text(" ".join(row["context"]["contexts"]))
    return (
        f"Context: {context}\n\nQuestion: {clean_text(row['question'])}\n\n"
        "Options:\nA. yes\nB. no\nC. maybe\n\nAnswer with only the letter (A, B or C)."
    )

def letter_ids(letter):
    # A letter can be tokenized as "A" or " A"; keep both ids.
    return {
        text_tokenizer.encode(letter, add_special_tokens=False)[-1],
        text_tokenizer.encode(" " + letter, add_special_tokens=False)[-1],
    }

def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    mean = values.mean()
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, mean - half, mean + half


def mean_median_p99(values, better='lower'):
    # Cauda do pior caso: P99 quando menor e melhor (tempos); P1 quando maior e melhor (tokens/s).
    # Percentil linear (padrao do NumPy). Nao altera mean_ci, que continua valendo para a acuracia.
    import warnings
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float('nan'), float('nan'), float('nan')
    if values.size < 100:
        warnings.warn(f'P99 com n={values.size} e quase o maximo; use n>=100 (ideal ~1000).')
    tail = 99 if better == 'lower' else 1
    return values.mean(), float(np.median(values)), float(np.percentile(values, tail))


# 2. Filtros originais e deduplicacao por pergunta antes do split.
etapa(2, 'Preparar treino, teste e demonstracoes')
raw_dataset = load_dataset(DATASET_ID, split='train')
filtered_dataset = raw_dataset.filter(keep_example)
seen, indices = set(), []
for i, row in enumerate(filtered_dataset):
    key = clean_text(row['Question']).casefold()
    if key not in seen:
        seen.add(key)
        indices.append(i)
filtered_dataset = filtered_dataset.select(indices)
if len(filtered_dataset) < SAMPLE_SIZE:
    raise ValueError('Menos exemplos unicos do que SAMPLE_SIZE.')
subset = filtered_dataset.shuffle(seed=SPLIT_SEED).select(range(SAMPLE_SIZE))
split_dataset = subset.train_test_split(test_size=EVAL_SIZE, seed=SPLIT_SEED)
train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']
# Demonstracoes ICL nunca entram no SFT; treino/teste mantem o split original.
demo_indices = indices_demonstracoes(filtered_dataset, train_dataset, eval_dataset)
medquad_demo_dataset = filtered_dataset.select(demo_indices)
extended_eval = eval_dataset
comparison_examples = eval_dataset.shuffle(seed=SPLIT_SEED).select(range(COMPARISON_SAMPLE_COUNT))
speed_questions = eval_dataset.select(range(min(SPEED_N, len(eval_dataset))))['Question']

# Benchmarks auxiliares: demonstracoes da propria tarefa, nunca do teste.
medqa = load_dataset('GBaker/MedQA-USMLE-4-options', split='test').shuffle(seed=SPLIT_SEED)
medqa = medqa.select(range(min(MEDQA_N, len(medqa))))
medqa_train = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
pubmed_all = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train').shuffle(seed=SPLIT_SEED)
pubmedqa = pubmed_all.select(range(min(PUBMEDQA_N, len(pubmed_all))))
pubmed_train = pubmed_all.select(range(len(pubmedqa), len(pubmed_all)))

def make_demo_pool(dataset, question_key, test_questions, formatter, answer_getter):
    forbidden = {clean_text(q).casefold() for q in test_questions}
    used, pool = set(), []
    for index, row in enumerate(dataset):
        question = clean_text(row[question_key])
        key = question.casefold()
        if key in forbidden or key in used:
            continue
        used.add(key)
        pool.append({'id': index, 'question': question,
                     'user': formatter(row), 'assistant': answer_getter(row)})
    if len(pool) < 5:
        raise ValueError('Necessarias cinco demonstracoes sem sobreposicao com o teste.')
    return pool

demo_pools = {
    'MedQuAD': make_demo_pool(medquad_demo_dataset, 'Question', eval_dataset['Question'],
        lambda row: make_user_prompt(row['Question']), lambda row: clean_answer_text(row['Answer'])),
    'MedQA': make_demo_pool(medqa_train, 'question', medqa['question'],
        medqa_prompt, lambda row: row['answer_idx']),
    'PubMedQA': make_demo_pool(pubmed_train, 'question', pubmedqa['question'],
        pubmedqa_prompt, lambda row: PUBMEDQA_LETTER[row['final_decision']]),
}
for task, pool in demo_pools.items():
    escrever_csv(f'registros/dados_demonstracoes_{task}.csv', pool)
salvar_tabelas(globals())



# 3. ICL apenas monta mensagens; nao treina pesos.
def select_demonstrations(task, question, shots):
    if shots not in (0, 3, 5):
        raise ValueError('Shots deve ser 0, 3 ou 5.')
    if shots == 0:
        return []
    digest = hashlib.sha256(f'{SEED}|{task}|{question}'.encode('utf-8')).digest()
    rng = random.Random(int.from_bytes(digest[:8], 'big'))
    # Os primeiros tres dos mesmos cinco controlam a comparacao 3-shot / 5-shot.
    return rng.sample(demo_pools[task], 5)[:shots]

def make_messages(task, question, user_text, system_prompt):
    if CURRENT_CONDITION == 'fine-tuned' and CURRENT_SHOTS != 0:
        raise ValueError('Fine-tuned deve ser avaliado sem demonstracoes.')
    demos = select_demonstrations(task, question, CURRENT_SHOTS)
    messages = [{'role': 'system', 'content': system_prompt}]
    for demo in demos:
        messages.extend([{'role': 'user', 'content': demo['user']},
                         {'role': 'assistant', 'content': demo['assistant']}])
    messages.append({'role': 'user', 'content': user_text})
    prompt_records[(CURRENT_CONDITION, task, question)] = {
        'condition': CURRENT_CONDITION, 'task': task, 'question': question,
        'seed': SEED, 'split_seed': SPLIT_SEED, 'persona': PERSONA,
        'shots': CURRENT_SHOTS, 'demo_ids': [d['id'] for d in demos], 'messages': messages}
    return messages

def encode_prompt(prompt, reserve_tokens=GEN_MAX_NEW_TOKENS):
    inputs = text_tokenizer(prompt, return_tensors='pt')
    length = inputs['input_ids'].shape[1]
    if length + reserve_tokens > MODEL_CONTEXT_LENGTH:
        raise ValueError(f'Prompt de {length} tokens excede o contexto com a resposta. '
                         'Aumente MODEL_CONTEXT_LENGTH em nova campanha; exemplos nao foram cortados.')
    return inputs.to(model.device)

def build_prompt(question):
    messages = make_messages('MedQuAD', question, make_user_prompt(question), SYSTEM_PROMPT)
    return messages, apply_chat_template(messages, add_generation_prompt=True)

# 4. Checkpoint original; adapters apenas para a condicao fine-tuned.
def build_model(fine_tune=False):
    global text_tokenizer
    fresh_model, fresh_tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=MODEL_CONTEXT_LENGTH,
        load_in_4bit=LOAD_IN_4BIT, load_in_8bit=False, full_finetuning=False, dtype=None)
    text_tokenizer = getattr(fresh_tokenizer, 'tokenizer', fresh_tokenizer)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    if fine_tune:
        fresh_model = FastModel.get_peft_model(
            fresh_model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=8, lora_alpha=16, lora_dropout=0, bias='none',
            use_gradient_checkpointing='unsloth', random_state=SEED)
    return fresh_model


def word_tokens(text):
    return re.findall(r"\w+", str(text).lower())

def token_f1(prediction, reference):
    pred, ref = word_tokens(prediction), word_tokens(reference)
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)

def rouge_l_f1(prediction, reference):
    pred, ref = word_tokens(prediction), word_tokens(reference)
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for i in range(1, len(pred) + 1):
        current = [0] * (len(ref) + 1)
        for j in range(1, len(ref) + 1):
            if pred[i - 1] == ref[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def postprocess_answer(text):
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    return clean_answer_text(text)

@torch.inference_mode()
def generate_answer(question, max_new_tokens=160):
    _, prompt = build_prompt(question)
    model_inputs = encode_prompt(prompt, max_new_tokens)

    # eos_token_id is intentionally not overridden: the model's generation_config
    # already lists the end-of-turn token(s) for each family.
    generated = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=text_tokenizer.pad_token_id,
    )

    new_tokens = generated[0][model_inputs["input_ids"].shape[1] :]
    text = text_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return clean_answer_text(re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL))

def generate_preview_rows(dataset, sample_count=COMPARISON_SAMPLE_COUNT):
    sample_count = min(sample_count, len(dataset))
    subset = dataset.select(range(sample_count))
    previews = []

    for row in tqdm(subset, desc=f"Previews {CURRENT_CONDITION}"):
        previews.append(
            {
                "question": row["Question"],
                "prediction": generate_answer(row["Question"]),
                "reference": clean_answer_text(row["Answer"]),
            }
        )

    return previews

@torch.inference_mode()
def timed_generate(model, question, max_new_tokens, fixed_output=False):
    _, prompt = build_prompt(question)
    inputs = encode_prompt(prompt, max_new_tokens)
    prompt_ids = inputs["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    timer = TokenTimer(prompt_ids)
    cuda_sync()
    start = time.perf_counter()
    # Tutorial: min_tokens = max_tokens; equivalente local em Transformers.
    length_kwargs = ({"min_new_tokens": max_new_tokens, "repetition_penalty": 1.0}
                     if fixed_output else {})
    generated = model.generate(
        **inputs, **length_kwargs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        pad_token_id=text_tokenizer.pad_token_id, streamer=timer)
    cuda_sync()
    total = time.perf_counter() - start

    new_tokens = generated[0][prompt_len:]
    output_ids = new_tokens.tolist()
    # Nao confundir quantidade de callbacks com quantidade de tokens.
    if not timer.times or timer.ids != output_ids:
        raise ValueError('Tokens do streamer diferem da saida de generate.')
    n_new = len(output_ids)
    if fixed_output and n_new != max_new_tokens:
        raise ValueError("Benchmark de velocidade requer exatamente o numero solicitado de tokens.")
    ttft = timer.times[0] - start
    ttlt = timer.times[-1] - start
    decode_time = ttlt - ttft
    raw_text = text_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = text_tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return {
        "prediction": postprocess_answer(raw_text),
        "prompt_tokens": prompt_len,
        "new_tokens": n_new,
        "latency_s": total,
        "ttft_s": ttft,
        "tokens_per_s": n_new / total if total > 0 else float("nan"),
        "decode_tokens_per_s": (n_new - 1) / decode_time if n_new > 1 and decode_time > 0 else float("nan"),
        "finished": bool(output_ids and output_ids[-1] in (eos_ids or [])),
        "ttlt_s": ttlt,
        "token_times": [t - start for t in timer.times],
    }


@torch.inference_mode()
def answer_nll(model, question, reference):
    messages, prompt = build_prompt(question)
    full = apply_chat_template(messages + [{'role': 'assistant', 'content': reference}],
                               add_generation_prompt=False)
    prompt_ids = text_tokenizer(prompt)['input_ids']
    encoded = encode_prompt(full, reserve_tokens=0)
    input_ids = encoded['input_ids']
    n_prompt = len(prompt_ids)
    if input_ids[0, :n_prompt].tolist() != prompt_ids:
        raise ValueError('Prefixo do template difere na resposta completa; revisar mascaramento da NLL.')
    if input_ids.shape[1] <= n_prompt:
        raise ValueError('Resposta de referencia sem tokens avaliaveis.')
    logits = model(**encoded).logits[0, n_prompt - 1:-1].float()
    targets = input_ids[0, n_prompt:]
    nll = F.cross_entropy(logits, targets, reduction='sum')
    return nll.item(), int(targets.numel())


def evaluate_model(model, regime):
    FastModel.for_inference(model)
    timed_generate(model, extended_eval[0]["Question"], max_new_tokens=8)  # warm-up, not measured

    torch.cuda.reset_peak_memory_stats()
    rows = []
    for row in tqdm(extended_eval, desc=f"MedQuAD {regime}"):
        reference = clean_answer_text(row["Answer"])
        result = timed_generate(model, row["Question"], GEN_MAX_NEW_TOKENS)
        nll_sum, nll_tokens = answer_nll(model, row["Question"], reference)
        rows.append(
            {
                "regime": regime,
                "question": row["Question"],
                "reference": reference,
                **result,
                "token_f1": token_f1(result["prediction"], reference),
                "rouge_l": rouge_l_f1(result["prediction"], reference),
                "answer_words": len(word_tokens(result["prediction"])),
                "reference_words": len(word_tokens(reference)),
                "nll_sum": nll_sum,
                "nll_tokens": nll_tokens,
            }
        )
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    return rows, peak_vram_gb

class TokenTimer(BaseStreamer):
    def __init__(self, prompt_ids):
        self.prompt_ids = prompt_ids
        self.times = []
        self.ids = []
        self.is_prompt = True

    def put(self, value):
        ids = value.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError('Cronometro requer batch=1.')
            ids = ids[0]
        if self.is_prompt:
            if ids != self.prompt_ids:
                raise ValueError('Primeiro callback nao corresponde ao prompt.')
            self.is_prompt = False
            return
        # O experimento atual usa geracao autoregressiva, um token por callback.
        # Falhar explicitamente se o backend mudar esse contrato.
        if len(ids) != 1:
            raise ValueError('Cronometro requer um token por callback.')
        self.ids.extend(ids)
        self.times.append(time.perf_counter())

    def end(self):
        pass

@torch.inference_mode()
def timed_answer(question, max_new_tokens=SPEED_MAX_NEW_TOKENS):
    result = timed_generate(model, question, max_new_tokens, fixed_output=True)
    return {
        "prompt_tokens": result["prompt_tokens"],
        "ttft_s": result["ttft_s"],
        "tokens_per_s": result["decode_tokens_per_s"],
        "total_s": result["latency_s"],
        "ttlt_s": result["ttlt_s"],
        "new_tokens": result["new_tokens"],
        "token_times": result["token_times"],
        "answer": result["prediction"],
    }

@torch.inference_mode()
def pick_option(user_text, letters, task, question):
    messages = make_messages(task, question, user_text, MCQ_SYSTEM_PROMPT)
    prompt = apply_chat_template(messages, add_generation_prompt=True) + "Answer:"
    model_inputs = encode_prompt(prompt, reserve_tokens=1)
    logits = model(**model_inputs).logits[0, -1].float()
    scores = [max(logits[i].item() for i in letter_ids(letter)) for letter in letters]
    return letters[int(np.argmax(scores))]

def train_regime(model, train_ds, epochs, tag):
    formatted = train_ds.map(format_training_example, remove_columns=train_ds.column_names)
    token_lists = text_tokenizer(
        textos_para_tokenizacao(formatted), truncation=True, max_length=MAX_SEQ_LENGTH
    )["input_ids"]
    tokens_per_epoch = sum(len(ids) for ids in token_lists)

    config = SFTConfig(
        max_steps=-1,
        output_dir=f"{EXTENDED_DIR}/trainer-{tag}",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=epochs,
        learning_rate=LEARNING_RATE,
        warmup_steps=5,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        packing=False,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    FastModel.for_training(model)
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted,
        args=config,
        processing_class=text_tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.model.print_trainable_parameters()
    registrar_trainer(trainer, tag)
    result = trainer.train()
    cuda_sync()
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    salvar_resultado(result, tag)
    runtime = result.metrics.get("train_runtime", float("nan"))
    info = {
        "train_examples": len(train_ds),
        "epochs": epochs,
        "train_runtime_min": runtime / 60,
        "train_loss": result.training_loss,
        "train_samples_per_s": result.metrics.get("train_samples_per_second", float("nan")),
        "train_tokens_per_s": tokens_per_epoch * epochs / runtime if runtime else float("nan"),
        "train_peak_vram_gb": peak_vram_gb,
    }
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return info


def macro_scores(predictions, answers, labels):
    # Classes = alternativas; nao especialidades medicas.
    precision, recall, f1 = [], [], []
    for label in labels:
        tp = sum(p == label and a == label for p, a in zip(predictions, answers))
        fp = sum(p == label and a != label for p, a in zip(predictions, answers))
        fn = sum(p != label and a == label for p, a in zip(predictions, answers))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {'precision_macro': sum(precision) / len(labels),
            'recall_macro': sum(recall) / len(labels), 'f1_macro': sum(f1) / len(labels)}

def evaluate_benchmarks(condition):
    timed_answer(speed_questions[0], max_new_tokens=8)
    for question in tqdm(speed_questions, desc=f'Velocidade {condition}'):
        speed_rows.append({'condition': condition, 'question': question, **timed_answer(question)})
    escrever_csv(f'{BENCH_DIR}/speed_per_question.csv', speed_rows)
    for task, dataset, formatter, letters in [('MedQA', medqa, medqa_prompt, list('ABCD')),
                                               ('PubMedQA', pubmedqa, pubmedqa_prompt, list('ABC'))]:
        for row in tqdm(dataset, desc=f'{task} {condition}'):
            prediction = pick_option(formatter(row), letters, task, row['question'])
            answer = row['answer_idx'] if task == 'MedQA' else PUBMEDQA_LETTER[row['final_decision']]
            precision_rows.append({'condition': condition, 'benchmark': task, 'question': row['question'],
                                   'prediction': prediction, 'answer': answer, 'correct': float(prediction == answer)})
        escrever_csv(f'{BENCH_DIR}/precision_per_question.csv', precision_rows)

def save_summaries(condition, train_info, rows, peak):
    df = pd.DataFrame(rows)
    nll = df['nll_sum'].sum() / df['nll_tokens'].sum()
    extended_summaries.append({'regime': condition, **train_info,
        'gen_tokens_per_s': df['new_tokens'].sum() / df['latency_s'].sum(),
        'decode_tokens_per_s': df['decode_tokens_per_s'].mean(),
        'ttft_s': df['ttft_s'].mean(), 'latency_s': df['latency_s'].mean(),
        'avg_new_tokens': df['new_tokens'].mean(), 'finished_rate': df['finished'].mean(),
        'token_f1': df['token_f1'].mean(), 'rouge_l': df['rouge_l'].mean(),
        'answer_nll': nll, 'answer_ppl': math.exp(nll),
        'answer_words': df['answer_words'].mean(), 'reference_words': df['reference_words'].mean(),
        'gen_peak_vram_gb': peak})
    escrever_csv(f'{EXTENDED_DIR}/metrics_summary.csv', extended_summaries)
    for column, label, better in [('ttft_s', 'TTFT (s)', 'lower'), ('tokens_per_s', 'Tokens per second', 'higher'),
                                   ('total_s', 'Total response time (s)', 'lower'), ('ttlt_s', 'TTLT (s)', 'lower')]:
        values = [r[column] for r in speed_rows if r['condition'] == condition]
        mean, median, p99 = mean_median_p99(values, better)
        benchmark_summaries.append({'model': MODEL_NAME, 'category': 'speed', 'metric': label,
            'condition': condition, 'mean': mean, 'median': median, 'p99_pior_caso': p99, 'better': better, 'n': len(values)})
    for task, labels in [('MedQA', list('ABCD')), ('PubMedQA', list('ABC'))]:
        selected = [r for r in precision_rows if r['condition'] == condition and r['benchmark'] == task]
        mean, low, high = mean_ci([r['correct'] for r in selected])
        benchmark_summaries.append({'model': MODEL_NAME, 'category': 'precision', 'metric': f'{task} accuracy',
            'condition': condition, 'mean': mean, 'ci_low': max(low, 0.), 'ci_high': min(high, 1.),
            'better': 'higher', 'n': len(selected)})
        for metric, value in macro_scores([r['prediction'] for r in selected], [r['answer'] for r in selected], labels).items():
            benchmark_summaries.append({'model': MODEL_NAME, 'category': 'classification_macro',
                'metric': f'{task} {metric}', 'condition': condition, 'mean': value,
                'ci_low': None, 'ci_high': None, 'better': 'higher', 'n': len(selected)})
    escrever_csv(f'{BENCH_DIR}/benchmark_summary.csv', benchmark_summaries)



# 7. Ordem do experimento, sem nenhum treino anterior ao zero-shot.
speed_rows, precision_rows, all_rows, preview_rows = [], [], [], []
benchmark_summaries, extended_summaries = [], []
prompt_records = {}
for condition, shots, fine_tune in CONDITIONS:
    etapa(condition, f'Carregar original | exemplos no prompt={shots} | treinar={fine_tune}')
    CURRENT_CONDITION, CURRENT_SHOTS = condition, shots
    set_seed(SEED)
    model = build_model(fine_tune=fine_tune)
    train_info = {'train_examples': 0, 'epochs': 0, 'train_runtime_min': 0,
                  'train_loss': None, 'train_samples_per_s': None,
                  'train_tokens_per_s': None, 'train_peak_vram_gb': None}
    if fine_tune:
        etapa(condition, 'Fine-tuning por 5 epocas; learning rate 2e-5; batch efetivo 32')
        train_info = train_regime(model, train_dataset, FINE_TUNING_EPOCHS, condition)
        model.save_pretrained(f'{OUTPUT_DIR}/adapter-fine-tuned')
        text_tokenizer.save_pretrained(f'{OUTPUT_DIR}/adapter-fine-tuned')
    FastModel.for_inference(model)
    etapa(condition, 'Inferencia MedQuAD e metricas de respostas')
    rows, peak = evaluate_model(model, condition)
    all_rows.extend(rows)
    escrever_csv(f'{EXTENDED_DIR}/answers_per_question.csv', all_rows)
    for preview in generate_preview_rows(comparison_examples):
        preview_rows.append({'condition': condition, **preview})
    escrever_csv('registros/previews.csv', preview_rows)
    etapa(condition, 'Velocidade, MedQA e PubMedQA')
    evaluate_benchmarks(condition)
    save_summaries(condition, train_info, rows, peak)
    escrever_csv('registros/prompts.csv', prompt_records.values())
    salvar_tabelas(globals())
    del model, text_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# 8. Graficos e tabelas das quatro condicoes.
bench_summary = pd.DataFrame(benchmark_summaries)
summary_df = pd.DataFrame(extended_summaries).set_index('regime')
summary_df[['token_f1', 'rouge_l']].plot.bar(rot=0, title=f'{MODEL_NAME} - sobreposicao textual')
plt.tight_layout()
salvar_figuras('texto')
for metric in bench_summary['metric'].unique():
    table = bench_summary[bench_summary['metric'] == metric]
    table.plot.bar(x='condition', y='mean', legend=False, rot=0, title=f'{MODEL_NAME} - {metric}')
    plt.tight_layout()
    salvar_figuras(re.sub(r'[^a-zA-Z0-9]+', '_', metric))
salvar_tabelas(globals())
print('Execucao completa: zero-shot, 3-shot, 5-shot e fine-tuned.', flush=True)
