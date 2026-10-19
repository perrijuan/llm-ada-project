"""Executa um modelo por vez, em processos isolados, com ICL e FT separados."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from consolidar import consolidar
from prompts_medicos import PERSONAS

BASE = Path(__file__).resolve().parent
MODELOS = ['gemma4', 'huatuogpt', 'medgemma', 'qwen35']
PROTOCOLO = 'icl_0_3_5_ft5_personas_v4_ft_separado'


def codigo_sha256(script):
    digest = hashlib.sha256()
    for path in [script, *sorted(BASE.glob('*.py'))]:
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelos', nargs='+', choices=MODELOS, default=MODELOS)
    parser.add_argument('--repeticoes', type=int, default=10,
                        help='Repeticoes de cada seed em cada modelo (padrao: 10).')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Seeds de demonstracoes e treino. Ex.: 42 123 2026.')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Seed fixa dos conjuntos de dados, independente das seeds de execucao.')
    parser.add_argument('--persona', choices=list(PERSONAS), default='clinico')
    parser.add_argument('--saida', type=Path, default=Path('resultados_icl_personas'))
    parser.add_argument('--retomar', action='store_true',
                        help='Pula somente execucoes completas com codigo e configuracao identicos.')
    parser.add_argument('--planejar', action='store_true',
                        help='Mostra a ordem sem carregar modelos, dados ou GPU.')
    args = parser.parse_args()
    if args.repeticoes < 1:
        parser.error('--repeticoes precisa ser maior que zero')
    if len(args.modelos) != len(set(args.modelos)):
        parser.error('Nao repita nomes em --modelos')
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Nao repita seeds; para repetibilidade use --repeticoes')
    if any(not 0 <= s < 2**32 for s in [*args.seeds, args.split_seed]):
        parser.error('Seeds devem estar entre 0 e 2**32 - 1')
    jobs = []
    for model in args.modelos:
        run = 0
        for seed in args.seeds:
            for repeticao_seed in range(1, args.repeticoes + 1):
                run += 1
                jobs.append((model, run, seed, repeticao_seed))
    campaign = dict(protocolo=PROTOCOLO, seeds=args.seeds, split_seed=args.split_seed,
                    persona=args.persona, repeticoes_por_seed=args.repeticoes)
    if args.planejar:
        for model, run, seed, rep in jobs:
            print(f'{model:12} run_{run:02} | seed={seed} rep={rep} | '
                  f'persona={args.persona} split_seed={args.split_seed} | '
                  'zero-shot -> 3-shot -> 5-shot -> FT 5 epocas')
        print(f'Total: {len(jobs)} processos sequenciais; {len(jobs)} fine-tunings.')
        return 0

    from tqdm.auto import tqdm
    root = args.saida.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.em_execucao'
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        parser.error(f'Campanha em uso: {lock}. Antes de remover uma trava antiga, '
                     'confirme que nao ha executor ativo.')
    failures = 0
    failed_models = set()
    try:
        manifest = root / 'campanha.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding='utf-8'))
            if previous != campaign:
                raise ValueError('Configuracao da campanha mudou (seeds, split, persona ou repeticoes). '
                                 'Use outra --saida.')
        elif list(root.glob('*/run_*/status.json')):
            raise ValueError('Campanha sem manifesto desta versao. Use outra --saida.')
        else:
            manifest.write_text(json.dumps(campaign, indent=2), encoding='utf-8')
        for old_status in root.glob('*/run_*/status.json'):
            previous = json.loads(old_status.read_text(encoding='utf-8'))
            if previous.get('protocolo') != PROTOCOLO:
                raise ValueError('Outro protocolo encontrado. Use outra --saida.')

        for model, run, seed, repeticao_seed in tqdm(jobs, desc='Execucoes planejadas', unit='execucao'):
            if model in failed_models:
                print(f'Pulando {model} run_{run:02}: modelo falhou nesta campanha.', flush=True)
                continue
            script = BASE / 'modelos' / f'{model}.py'
            digest = codigo_sha256(script)
            folder = root / model / f'run_{run:02}'
            status_file = folder / 'status.json'
            expected = dict(modelo=model, repeticao=run, seed=seed, split_seed=args.split_seed,
                            persona=args.persona, repeticao_seed=repeticao_seed,
                            protocolo=PROTOCOLO, codigo_sha256=digest)
            if folder.exists():
                status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else {}
                if (args.retomar and status.get('status') == 'ok'
                        and all(status.get(k) == v for k, v in expected.items())):
                    print(f'Ja concluido: {model} run_{run:02}', flush=True)
                    continue
                raise FileExistsError(f'{folder} ja existe. Use outra --saida; --retomar pula '
                                      'somente resultados completos do mesmo codigo e configuracao.')
            folder.mkdir(parents=True)
            info = dict(expected, max_steps=-1, inicio_utc=datetime.now(timezone.utc).isoformat(),
                        status='rodando')
            status_file.write_text(json.dumps(info, indent=2), encoding='utf-8')
            env = dict(os.environ, LLM_RUN_DIR=str(folder), LLM_SEED=str(seed),
                       LLM_SPLIT_SEED=str(args.split_seed), LLM_PERSONA=args.persona,
                       PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
            start = time.perf_counter()
            print(f'\n=== {model} | run_{run:02} | seed={seed} | persona={args.persona} ===', flush=True)
            process = None
            try:
                with (folder / 'execucao.log').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(
                        [sys.executable, '-u', str(script)], cwd=folder, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding='utf-8', errors='replace', bufsize=1)
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                info.update(status='ok' if code == 0 else 'erro', exit_code=code)
            except Exception as exc:
                # Erros de execucao sao registrados; Ctrl+C continua interrompendo.
                info.update(status='erro', erro=f'{type(exc).__name__}: {exc}')
                with (folder / 'execucao.log').open('a', encoding='utf-8') as log:
                    traceback.print_exc(file=log)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except BaseException:
                info.update(status='interrompido')
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            finally:
                if process is not None and process.stdout is not None:
                    process.stdout.close()
                info['duracao_s'] = time.perf_counter() - start
                status_file.write_text(json.dumps(info, indent=2), encoding='utf-8')
                consolidar(root)
            if info['status'] != 'ok':
                failures += 1
                failed_models.add(model)
                print(f'Falha em {model}; veja {folder / "execucao.log"}. '
                      'Pulando as demais repeticoes deste modelo e seguindo para o proximo.', flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

