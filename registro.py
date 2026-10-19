"""Somente registro e progresso. Nao altera pesos, loss ou hiperparametros."""
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from prompts_medicos import configuracao


def escrever_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, default=str)
                             if isinstance(v, (list, dict, tuple)) else v
                             for k, v in row.items()})
    temp.replace(path)


def iniciar(modelo):
    # Ao chamar o modelo diretamente, cria uma pasta nova automaticamente.
    if not os.getenv('LLM_RUN_DIR'):
        path = Path.cwd() / 'resultados' / modelo / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path.mkdir(parents=True, exist_ok=False)
        os.chdir(path)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    print(f'\nModelo: {modelo} | resultados: {Path.cwd()}', flush=True)
    versions = {}
    for name in ['torch', 'unsloth', 'unsloth_zoo', 'transformers', 'trl', 'peft',
                 'bitsandbytes', 'datasets', 'accelerate', 'tqdm', 'numpy', 'pandas']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = 'nao instalado'
    info = {'modelo': modelo, 'inicio_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'plataforma': platform.platform(), 'versoes': versions,
            **configuracao(), 'max_steps': -1,
            'condicoes': ['zero-shot', '3-shot', '5-shot', 'fine-tuned'],
            'learning_rate': 2e-5, 'fine_tuning_epochs': 5,
            'optimizer': 'adamw_torch', 'micro_batch_size': 2,
            'gradient_accumulation_steps': 16, 'effective_batch_size': 32}
    Path('ambiente.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    try:
        gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
        Path('nvidia-smi.txt').write_text(gpu.stdout + gpu.stderr, encoding='utf-8')
    except (OSError, subprocess.TimeoutExpired) as exc:
        Path('nvidia-smi.txt').write_text(str(exc), encoding='utf-8')


def etapa(numero, descricao):
    print(f'\n[ETAPA {numero}] {descricao}', flush=True)


def registrar_trainer(trainer, nome):
    # Importacao tardia: o notebook importa Unsloth antes de Transformers.
    from transformers import TrainerCallback

    class RegistroCSV(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            escrever_csv(f'registros/treino_{nome}_steps.csv', state.log_history)

        def on_train_end(self, args, state, control, **kwargs):
            escrever_csv(f'registros/treino_{nome}_steps.csv', state.log_history)

    trainer.add_callback(RegistroCSV())
    Path('registros').mkdir(exist_ok=True)
    # Configuracao resolvida, inclusive max_steps=-1 e numero de epocas.
    Path(f'registros/config_{nome}.json').write_text(trainer.args.to_json_string(), encoding='utf-8')


def salvar_resultado(result, nome):
    escrever_csv(f'registros/treino_{nome}_metricas.csv',
                 [{'training_loss': result.training_loss, 'global_step': result.global_step,
                   **result.metrics}])


def salvar_tabelas(namespace):
    # Preserva todas as colunas, inclusive tempos individuais dos tokens.
    import pandas as pd
    for name in ['baseline_previews', 'fine_tuned_previews', 'speed_rows', 'precision_rows', 'all_rows']:
        rows = namespace.get(name)
        if isinstance(rows, list) and rows:
            escrever_csv(f'registros/{name}.csv', rows)
    for name in ['speed_df', 'precision_df', 'bench_summary', 'results_df', 'summary_df']:
        df = namespace.get(name)
        if isinstance(df, pd.DataFrame):
            if df.index.name:
                df = df.reset_index()
            escrever_csv(f'registros/{name}.csv', df.to_dict('records'))
    dados = []
    for name in ['train_dataset', 'medquad_demo_dataset', 'eval_dataset', 'comparison_examples', 'extended_eval', 'medqa', 'pubmedqa']:
        ds = namespace.get(name)
        if ds is not None:
            # Permite conferir se modelos e repeticoes receberam as mesmas perguntas.
            rows = [dict(row) for row in ds]
            escrever_csv(f'registros/dados_{name}.csv', rows)
            dados.append({'conjunto': name, 'n': len(rows), 'fingerprint': getattr(ds, '_fingerprint', '')})
    escrever_csv('registros/datasets.csv', dados)
    model = namespace.get('model')
    if model is not None and hasattr(model, 'config'):
        Path('registros/model_config.json').write_text(model.config.to_json_string(), encoding='utf-8')


def salvar_figuras(celula):
    if 'matplotlib.pyplot' not in sys.modules:
        return
    plt = sys.modules['matplotlib.pyplot']
    if plt.get_fignums():
        Path('figuras').mkdir(exist_ok=True)
        for number in plt.get_fignums():
            plt.figure(number).savefig(f'figuras/celula_{celula}_figura_{number}.png', dpi=130, bbox_inches='tight')
        plt.close('all')
