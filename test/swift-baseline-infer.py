#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shlex
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter


def now_tag():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def safe_name(path_or_name: str) -> str:
    s = str(path_or_name).rstrip('/').split('/')[-1]
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    return s or 'model'


def read_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def clean_sentence(sentence):
    if sentence is None:
        return []
    if isinstance(sentence, list):
        raw_tokens = sentence
    else:
        raw_tokens = str(sentence).replace('\n', ' ').split()

    bad_tokens = {'...', '....', '.....', '.', ',', ';', ':', '[', ']', '(', ')'}
    genes, seen = [], set()
    for token in raw_tokens:
        token = str(token).strip()
        if not token:
            continue
        if token in bad_tokens:
            continue
        if set(token) == {'.'}:
            continue
        if token not in seen:
            genes.append(token)
            seen.add(token)
    return genes


def safe_text(value):
    if value is None:
        return 'N/A'
    value = str(value).strip()
    if value == '' or value.lower() in {'nan', 'none', 'null'} or value in {'-666', '-666.0'}:
        return 'N/A'
    return value


def format_gene_list(genes):
    return ' '.join(map(str, genes))


def flatten_blocks(blocks_dict):
    if not isinstance(blocks_dict, dict):
        return []
    out = []
    def key_fn(x):
        try:
            return int(x)
        except Exception:
            return str(x)
    for k in sorted(blocks_dict.keys(), key=key_fn):
        v = blocks_dict[k]
        if isinstance(v, list):
            out.extend(v)
    return clean_sentence(out)


def messages_to_text(messages):
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if isinstance(m, dict):
                parts.append(str(m.get('content', '')))
            else:
                parts.append(str(m))
        return '\n'.join(parts)
    return str(messages or '')


def get_prompt_text(value):
    if isinstance(value, list):
        return messages_to_text(value)
    if isinstance(value, dict):
        if 'messages' in value:
            return messages_to_text(value['messages'])
        return json.dumps(value, ensure_ascii=False)
    return str(value or '')


def extract_section_text(text, section_names):
    """Extract content after a markdown-like [Section] header until next [Header]."""
    if not text:
        return ''
    names = '|'.join(re.escape(x) for x in section_names)
    pattern = re.compile(rf'\[\s*(?:{names})\s*\]\s*(.*?)(?=\n\s*\[[^\]]+\]|\Z)', re.I | re.S)
    m = pattern.search(text)
    if not m:
        return ''
    return m.group(1).strip()


def extract_control_from_prompt(record, prompt_fields):
    for field in prompt_fields:
        if field not in record:
            continue
        text = get_prompt_text(record.get(field))
        section = extract_section_text(
            text,
            [
                'Control Expression',
                'Control Expression Sentence',
                'Control Sentence',
                'Input Expression',
                'Input Sentence',
                'Original Expression',
                'Original Sentence',
            ],
        )
        genes = clean_sentence(section)
        if genes:
            return genes, f'prompt_section:{field}'
    return [], ''


def first_valid_field(record, field_names):
    for field in field_names:
        if field in record and record.get(field) is not None:
            genes = clean_sentence(record.get(field))
            if genes:
                return genes, field
    return [], ''


def resolve_control_genes(record, args):
    fields = [x.strip() for x in args.control_fields.split(',') if x.strip()]
    genes, source = first_valid_field(record, fields)
    if genes:
        return genes, source

    # Common two-stage test data fallback: reconstruct the initial/control sequence from blocks.
    for field in ['control_blocks', 'input_blocks', 'source_blocks', 'gt_control_blocks']:
        if field in record:
            genes = flatten_blocks(record.get(field))
            if genes:
                return genes, field

    prompt_fields = [x.strip() for x in args.prompt_fallback_fields.split(',') if x.strip()]
    return extract_control_from_prompt(record, prompt_fields)


def resolve_perturb_genes(record, args):
    fields = [x.strip() for x in args.perturb_fields.split(',') if x.strip()]
    genes, source = first_valid_field(record, fields)
    if genes:
        return genes, source

    # Common two-stage test data fallback: final sequence label.
    for field in ['gt_sequence', 'target_sequence', 'perturbed_sequence']:
        if field in record:
            genes = clean_sentence(record.get(field))
            if genes:
                return genes, field

    for field in ['gt_reranker_blocks', 'target_blocks', 'perturb_blocks']:
        if field in record:
            genes = flatten_blocks(record.get(field))
            if genes:
                return genes, field

    return [], ''


def get_condition_value(record, key):
    if key in record:
        return record.get(key)
    meta = record.get('condition_metadata')
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def build_condition_text(record):
    return (
        f"Cell line: {safe_text(get_condition_value(record, 'cell_id'))}\n"
        f"Perturbation type: {safe_text(get_condition_value(record, 'perturbation_type'))}\n"
        f"Perturbation name: {safe_text(get_condition_value(record, 'pert_name'))}\n"
        f"SMILES: {safe_text(get_condition_value(record, 'canonical_smiles'))}\n"
        f"Treatment time: {safe_text(get_condition_value(record, 'pert_itime'))}"
    )


def build_baseline_system():
    return (
        'You are an AI expert in transcriptomic perturbation analysis. '
        'Your task is to directly predict the complete perturbed gene expression sentence '
        'after a given perturbation. '
        'The output should be a single whitespace-separated sequence of gene symbols.'
    )


def build_baseline_user(record, control_genes):
    return (
        '[Condition]\n'
        f'{build_condition_text(record)}\n\n'
        '[Control Expression]\n'
        f'{format_gene_list(control_genes)}\n\n'
        '[Task]\n'
        'Predict the complete perturbed expression sentence.'
    )


def build_messages(system_content, user_content):
    return [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content},
    ]


def build_infer_sample(record, raw_idx, args):
    if args.split != 'all':
        split = record.get('split', 'unknown')
        if split != args.split:
            return {'status': 'skip_split', 'split': split}

    control_genes, control_source = resolve_control_genes(record, args)
    perturb_genes, perturb_source = resolve_perturb_genes(record, args)

    if not control_genes and not perturb_genes:
        return {'status': 'skip_empty', 'split': record.get('split', 'unknown')}
    if not control_genes:
        return {'status': 'skip_empty_control', 'split': record.get('split', 'unknown'), 'perturb_source': perturb_source}
    if not perturb_genes:
        return {'status': 'skip_empty_perturb', 'split': record.get('split', 'unknown'), 'control_source': control_source}

    if len(control_genes) < args.min_control_genes:
        return {
            'status': 'skip_short_control',
            'split': record.get('split', 'unknown'),
            'control_gene_count': len(control_genes),
            'perturb_gene_count': len(perturb_genes),
            'control_source': control_source,
            'perturb_source': perturb_source,
        }
    if len(perturb_genes) < args.min_perturb_genes:
        return {
            'status': 'skip_short_perturb',
            'split': record.get('split', 'unknown'),
            'control_gene_count': len(control_genes),
            'perturb_gene_count': len(perturb_genes),
            'control_source': control_source,
            'perturb_source': perturb_source,
        }

    messages = build_messages(build_baseline_system(), build_baseline_user(record, control_genes))
    label = format_gene_list(perturb_genes)
    return {
        'status': 'ok',
        'raw_idx': raw_idx,
        'record_id': record.get('record_id', record.get('id', raw_idx)),
        'split': record.get('split', 'unknown'),
        'messages': messages,
        'label': label,
        'control_gene_count': len(control_genes),
        'perturb_gene_count': len(perturb_genes),
        'control_source': control_source,
        'perturb_source': perturb_source,
    }


def build_val_dataset(records, args):
    val_rows, index_rows, label_rows = [], [], []
    status_counter = Counter()
    source_counter = Counter()

    max_samples = args.max_samples
    for raw_idx, record in enumerate(records):
        item = build_infer_sample(record, raw_idx, args)
        status = item.get('status', 'unknown')
        status_counter[status] += 1
        if status != 'ok':
            continue

        idx = len(val_rows)
        val_rows.append({'messages': item['messages']})
        index_rows.append({
            'idx': idx,
            'raw_idx': item['raw_idx'],
            'record_id': item['record_id'],
            'split': item['split'],
            'control_gene_count': item['control_gene_count'],
            'perturb_gene_count': item['perturb_gene_count'],
            'control_source': item['control_source'],
            'perturb_source': item['perturb_source'],
        })
        label_rows.append({
            'idx': idx,
            'raw_idx': item['raw_idx'],
            'record_id': item['record_id'],
            'split': item['split'],
            'label': item['label'],
            'label_genes': item['label'].split(),
        })
        source_counter[f"control={item['control_source']}|perturb={item['perturb_source']}"] += 1

        if max_samples > 0 and len(val_rows) >= max_samples:
            break

    return val_rows, index_rows, label_rows, status_counter, source_counter


def shell_join(cmd):
    return ' '.join(shlex.quote(str(x)) for x in cmd)


def build_swift_command(args, val_dataset_path, result_path):
    cmd = [
        'swift', 'infer',
        '--model', str(Path(args.model).resolve()),
        '--torch_dtype', 'float16',
        '--infer_backend', args.infer_backend,
        '--val_dataset', str(Path(val_dataset_path).resolve()),
        '--vllm_gpu_memory_utilization', str(args.vllm_gpu_memory_utilization),
        '--vllm_max_model_len', str(args.vllm_max_model_len),
        '--max_new_tokens', str(args.max_new_tokens),
        '--result_path', str(Path(result_path).resolve()),
    ]

    if args.temperature is not None:
        cmd += ['--temperature', str(args.temperature)]
    if args.top_p is not None:
        cmd += ['--top_p', str(args.top_p)]
    if args.extra_args.strip():
        cmd += shlex.split(args.extra_args.strip())
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description='Build baseline one-step messages-format test jsonl and run ms-swift infer.')
    parser.add_argument('--test-data', type=str, default="/root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_id.jsonl")
    parser.add_argument('--model', type=str, default="vandijklab/C2S-Scale-Gemma-2-2B")
    parser.add_argument('--outdir', type=str, default='./id_runs')
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--split', type=str, default='all', help='Use all records or a specific split name.')
    parser.add_argument('--max-samples', type=int, default=-1)
    parser.add_argument('--min-control-genes', type=int, default=100)
    parser.add_argument('--min-perturb-genes', type=int, default=100)

    parser.add_argument('--control-fields', type=str, default='control_sentence,control_expression,control_sequence,input_sentence,source_sentence')
    parser.add_argument('--perturb-fields', type=str, default='perturb_sentence,perturbed_sentence,target_sentence,response_sentence,gt_sequence')
    parser.add_argument('--prompt-fallback-fields', type=str, default='baseline_prompt,planner_prompt,prompt,messages')

    parser.add_argument('--nproc-per-node', type=int, default=4)
    parser.add_argument('--cuda-visible-devices', type=str, default='0,1,2,3')
    parser.add_argument('--infer-backend', type=str, default='vllm')
    parser.add_argument('--vllm-gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--vllm-max-model-len', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--extra-args', type=str, default='')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    test_data = Path(args.test_data).resolve()
    model_path = Path(args.model).resolve()
    records = read_jsonl(test_data)

    run_name = args.run_name or f'{safe_name(model_path)}__baseline__{args.split}__{now_tag()}'
    run_dir = (Path(args.outdir) / run_name).resolve()
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    val_rows, index_rows, label_rows, status_counter, source_counter = build_val_dataset(records, args)

    val_dataset_path = (run_dir / 'val_dataset.messages.jsonl').resolve()
    index_map_path = (run_dir / 'index_map.jsonl').resolve()
    labels_path = (run_dir / 'labels.jsonl').resolve()
    result_path = (run_dir / 'swift_result.jsonl').resolve()
    log_path = (run_dir / 'infer.log').resolve()
    command_path = (run_dir / 'command.sh').resolve()
    meta_path = (run_dir / 'cook_baseline_infer_meta.json').resolve()
    config_path = (run_dir / 'run_config.json').resolve()

    write_jsonl(val_dataset_path, val_rows)
    write_jsonl(index_map_path, index_rows)
    write_jsonl(labels_path, label_rows)

    config = vars(args).copy()
    config.update({
        'test_data': str(test_data),
        'model': str(model_path),
        'run_name': run_name,
        'run_dir': str(run_dir),
        'num_input_records': len(records),
        'num_infer_samples': len(val_rows),
    })
    write_json(config_path, config)

    meta = {
        'num_input_records': len(records),
        'num_infer_samples': len(val_rows),
        'status_counter': dict(status_counter),
        'source_counter': dict(source_counter),
        'val_dataset': str(val_dataset_path),
        'index_map': str(index_map_path),
        'labels': str(labels_path),
    }
    write_json(meta_path, meta)

    print(f'Loaded input records : {len(records)}')
    print(f'Used infer samples   : {len(val_rows)}')
    print(f'Target split         : {args.split}')
    print(f'Run dir              : {run_dir}')
    print(f'Val dataset          : {val_dataset_path}')
    print(f'Index map            : {index_map_path}')
    print(f'Labels               : {labels_path}')
    print(f'Status counter       : {dict(status_counter)}')
    print(f'Source counter       : {dict(source_counter)}')

    if len(val_rows) == 0:
        print('\nNo valid samples were built. Debug tips:')
        print('  1) Check available keys:')
        print(f'     python - <<\'PY2\'\nimport json\nr=json.loads(open("{test_data}").readline())\nprint(r.keys())\nprint(json.dumps(r, ensure_ascii=False)[:2000])\nPY2')
        print('  2) If control is inside another field, pass --control-fields or --prompt-fallback-fields.')
        print('  3) If label is inside another field, pass --perturb-fields.')
        raise RuntimeError('No valid samples were built. Check split/min genes/field names.')

    cmd = build_swift_command(args, val_dataset_path, result_path)
    env_prefix = f'NPROC_PER_NODE={args.nproc_per_node} CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)}'
    command_text = env_prefix + ' ' + shell_join(cmd)
    command_path.write_text(command_text + '\n', encoding='utf-8')

    print(f'Command              : {command_path}')
    print(f'Result path          : {result_path}')
    print(f'Log path             : {log_path}')

    if args.dry_run:
        print('\nDry run enabled. Dataset and command were generated, but swift infer was not executed.')
        return

    env = os.environ.copy()
    env['NPROC_PER_NODE'] = str(args.nproc_per_node)
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write(command_text + '\n\n')
        log_f.flush()
        proc = subprocess.run(cmd, env=env, cwd=str(Path.cwd()), stdout=log_f, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f'swift infer failed with return code {proc.returncode}. Check log: {log_path}')

    print('\nDone.')
    print(f'Swift result: {result_path}')


if __name__ == '__main__':
    main()
