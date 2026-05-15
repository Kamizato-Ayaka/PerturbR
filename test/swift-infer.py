#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def safe_name(path_or_name: str) -> str:
    s = str(path_or_name).rstrip('/').split('/')[-1]
    if "checkpoint" in s:
        s = str(path_or_name).rstrip('/').split('/')[-3]
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    return s or 'model'


def now_tag() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def read_jsonl(path: str):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def normalize_messages(x, default_system: str = ''):
    if isinstance(x, dict) and 'messages' in x:
        x = x['messages']

    if isinstance(x, list):
        messages = []
        for m in x:
            if not isinstance(m, dict):
                raise ValueError(f'Invalid message item: {m}')
            role = m.get('role')
            content = m.get('content')
            if role is None or content is None:
                raise ValueError(f'Message must contain role/content: {m}')
            messages.append({'role': str(role), 'content': str(content)})
        return messages

    if isinstance(x, str):
        messages = []
        if default_system:
            messages.append({'role': 'system', 'content': default_system})
        messages.append({'role': 'user', 'content': x})
        return messages

    raise ValueError(f'Unsupported prompt format: {type(x)}')


def get_prompt_from_record(record, prompt_field: str):
    if prompt_field == 'messages':
        if 'messages' not in record:
            raise KeyError('prompt_field=messages but record has no messages field')
        return record['messages']
    if prompt_field not in record:
        raise KeyError(f'prompt field `{prompt_field}` not found in record keys={list(record.keys())}')
    return record[prompt_field]


def build_val_dataset(records, prompt_field: str, default_system: str = ''):
    val_rows = []
    index_rows = []

    for i, record in enumerate(records):
        messages = normalize_messages(get_prompt_from_record(record, prompt_field), default_system=default_system)
        val_rows.append({'messages': messages})
        index_rows.append({
            'idx': i,
            'record_id': record.get('record_id', i),
            'split': record.get('split', ''),
            'prompt_field': prompt_field,
        })

    return val_rows, index_rows


def shell_join(cmd):
    return ' '.join(subprocess.list2cmdline([x]) for x in cmd)


def run_swift_infer(args, run_dir: Path, val_dataset_path: Path, result_path: Path):
    env = os.environ.copy()
    env['NPROC_PER_NODE'] = str(args.nproc_per_node)
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')

    val_dataset_path = val_dataset_path.resolve()
    result_path = result_path.resolve()

    cmd = [
        'swift', 'infer',
        '--model', args.model,
        '--infer_backend', args.infer_backend,
        '--model_type', 'qwen3',
        '--torch_dtype', 'float16',
        '--enable_thinking', 'False',
        '--val_dataset', str(val_dataset_path),
        '--vllm_gpu_memory_utilization', str(args.vllm_gpu_memory_utilization),
        '--vllm_max_model_len', str(args.vllm_max_model_len),
        '--max_new_tokens', str(args.max_new_tokens),
    ]

    if args.temperature is not None:
        cmd += ['--temperature', str(args.temperature)]
    if args.top_p is not None:
        cmd += ['--top_p', str(args.top_p)]
    if args.result_path_arg:
        cmd += [args.result_path_arg, str(result_path)]
    if args.extra_swift_args:
        cmd += args.extra_swift_args

    command_text = (
        f'NPROC_PER_NODE={args.nproc_per_node} \\\n'
        f'CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} \\\n'
        f'{shell_join(cmd)}\n'
    )
    (run_dir / 'command.sh').write_text(command_text, encoding='utf-8')

    print('\n' + '=' * 80)
    print('SWIFT INFER COMMAND')
    print('=' * 80)
    print(command_text)

    if args.dry_run:
        print('[dry-run] Command saved but not executed.')
        return {'returncode': None, 'seconds': 0, 'result_path': str(result_path)}

    log_path = run_dir / 'infer.log'
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as log_f:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(Path.cwd()),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    seconds = time.time() - t0

    info = {
        'returncode': proc.returncode,
        'seconds': seconds,
        'result_path': str(result_path),
        'log_path': str(log_path),
    }
    write_json(run_dir / 'infer_status.json', info)

    if proc.returncode != 0:
        raise RuntimeError(f'swift infer failed with returncode={proc.returncode}. Check log: {log_path}')

    return info


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Prepare messages-format jsonl and run ms-swift infer with DP.')

    parser.add_argument('--test-data', type=str, default="/root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_id.jsonl")
    parser.add_argument('--model', type=str, required=True, help='Model path or model name for swift infer.')
    parser.add_argument('--prompt-field', type=str, default='planner_prompt', help='Field used as prompt/messages, e.g. planner_prompt, reranker_isolated_prompt, messages.')
    parser.add_argument('--default-system', type=str, default='', help='System prompt added only when prompt field is a plain string.')
    parser.add_argument('--outdir', type=str, default='./id_runs')
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--nproc-per-node', type=int, default=4)
    parser.add_argument('--cuda-visible-devices', type=str, default='0,1,2,3')
    parser.add_argument('--infer-backend', type=str, default='vllm')
    parser.add_argument('--vllm-gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--vllm-max-model-len', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--result-path-arg', type=str, default='--result_path', help='Argument name for saving swift result. Use empty string to disable.')
    parser.add_argument('--extra-swift-args', nargs=argparse.REMAINDER, default=[], help='Extra args appended to swift infer. Put this argument last.')
    return parser


def main():
    args = build_arg_parser().parse_args()

    records = read_jsonl(args.test_data)
    run_name = args.run_name or f'{safe_name(args.model)}__{args.prompt_field}__{now_tag()}'
    run_dir = Path(args.outdir) / run_name

    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    val_rows, index_rows = build_val_dataset(
        records,
        prompt_field=args.prompt_field,
        default_system=args.default_system,
    )

    run_dir_abs = run_dir.resolve()
    val_dataset_path = run_dir_abs / 'val_dataset.messages.jsonl'
    index_path = run_dir_abs / 'index_map.jsonl'
    result_path = run_dir_abs / 'swift_result.jsonl'

    write_jsonl(val_dataset_path, val_rows)
    write_jsonl(index_path, index_rows)

    config = vars(args).copy()
    config.update({
        'num_records': len(records),
        'run_dir': str(run_dir.resolve()),
        'val_dataset_path': str(val_dataset_path),
        'index_map_path': str(index_path),
        'result_path': str(result_path),
    })
    write_json(run_dir / 'run_config.json', config)

    print(f'Loaded samples : {len(records)}')
    print(f'Prompt field   : {args.prompt_field}')
    print(f'Run dir        : {run_dir}')
    print(f'Val dataset    : {val_dataset_path}')
    print(f'Index map      : {index_path}')

    info = run_swift_infer(args, run_dir, val_dataset_path, result_path)

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)
    print(f'Run dir    : {run_dir}')
    print(f'Result path: {info.get("result_path")}')
    print(f'Log path   : {info.get("log_path", "") or str(run_dir / "infer.log")}')


if __name__ == '__main__':
    main()
