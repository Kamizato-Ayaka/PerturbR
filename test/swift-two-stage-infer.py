#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage planner -> reranker inference for ms-swift.

Design goal:
1. Keep planner inference exactly consistent with the existing swift-infer.py wrapper.
2. Support two modes:
   - no --planner-result: run planner first, then build reranker input from planner output.
   - with --planner-result: skip planner inference and directly build reranker input from the given planner result.
3. Avoid GT leakage by default. Reranker prompts must be built from a template/placeholder/regex replacement.

Typical usage:

python swift-two-stage-infer.py \
  --test-data /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_id.jsonl \
  --planner-model /path/to/planner/checkpoint \
  --reranker-model /path/to/reranker/checkpoint \
  --outdir ./id_runs \
  --run-name Qwen3-4B__e2e__id \
  --nproc-per-node 8 \
  --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --vllm-gpu-memory-utilization 0.9 \
  --vllm-max-model-len 12000 \
  --planner-max-new-tokens 8192 \
  --reranker-max-new-tokens 8192

If planner result already exists:

python swift-two-stage-infer.py \
  --test-data /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_id.jsonl \
  --planner-result ./id_runs/xxx/planner/swift_result.jsonl \
  --reranker-model /path/to/reranker/checkpoint \
  --outdir ./id_runs \
  --run-name Qwen3-4B__e2e_from_existing_planner__id \
  --nproc-per-node 8 \
  --cuda-visible-devices 0,1,2,3,4,5,6,7
"""

import os
import re
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional


# -----------------------------
# Basic IO utilities
# -----------------------------

def now_tag() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def safe_name(path_or_name: str) -> str:
    s = str(path_or_name).rstrip('/').split('/')[-1]
    if 'checkpoint' in s:
        parts = str(path_or_name).rstrip('/').split('/')
        if len(parts) >= 3:
            s = parts[-3]
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    return s or 'model'


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def shell_join(cmd: List[str]) -> str:
    return ' '.join(subprocess.list2cmdline([str(x)]) for x in cmd)


# -----------------------------
# Prompt/message normalization
# Keep this consistent with old swift-infer.py
# -----------------------------

def normalize_messages(x: Any, default_system: str = '') -> List[Dict[str, str]]:
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


def get_prompt_from_record(record: Dict[str, Any], prompt_field: str):
    if prompt_field == 'messages':
        if 'messages' not in record:
            raise KeyError('prompt_field=messages but record has no messages field')
        return record['messages']
    if prompt_field not in record:
        raise KeyError(f'prompt field `{prompt_field}` not found in record keys={list(record.keys())}')
    return record[prompt_field]


def build_val_dataset(records: List[Dict[str, Any]], prompt_field: str, default_system: str = ''):
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


# -----------------------------
# Swift execution
# -----------------------------

def run_cmd(cmd: List[str], env: Dict[str, str], log_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = shell_join(cmd)
    (log_path.parent / 'command.raw.sh').write_text(command_text + '\n', encoding='utf-8')

    if dry_run:
        print('[dry-run] Command saved but not executed:')
        print(command_text)
        return {'returncode': None, 'seconds': 0, 'log_path': str(log_path)}

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
    info = {'returncode': proc.returncode, 'seconds': seconds, 'log_path': str(log_path)}
    write_json(log_path.parent / 'infer_status.json', info)
    if proc.returncode != 0:
        raise RuntimeError(f'Command failed with returncode={proc.returncode}. Check log: {log_path}')
    return info


def run_existing_swift_infer(
    *,
    script_path: str,
    test_data: Path,
    model: str,
    prompt_field: str,
    outdir: Path,
    run_name: str,
    args,
    max_new_tokens: int,
    stage_name: str,
) -> Path:
    """Run the old swift-infer.py wrapper to keep input construction identical."""
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f'Cannot find swift infer script: {script_path}')

    cmd = [
        sys.executable, str(script),
        '--test-data', str(test_data),
        '--model', model,
        '--prompt-field', prompt_field,
        '--outdir', str(outdir),
        '--run-name', run_name,
        '--nproc-per-node', str(args.nproc_per_node),
        '--cuda-visible-devices', args.cuda_visible_devices,
        '--infer-backend', args.infer_backend,
        '--vllm-gpu-memory-utilization', str(args.vllm_gpu_memory_utilization),
        '--vllm-max-model-len', str(args.vllm_max_model_len),
        '--max-new-tokens', str(max_new_tokens),
        f'--result-path-arg={args.result_path_arg}',
    ]

    if args.default_system:
        cmd += ['--default-system', args.default_system]
    if args.temperature is not None:
        cmd += ['--temperature', str(args.temperature)]
    if args.top_p is not None:
        cmd += ['--top-p', str(args.top_p)]
    if args.overwrite:
        cmd += ['--overwrite']
    if args.dry_run:
        cmd += ['--dry-run']
    if args.extra_swift_args:
        # old script expects --extra-swift-args as the last argument
        cmd += ['--extra-swift-args'] + args.extra_swift_args

    env = os.environ.copy()
    env['NPROC_PER_NODE'] = str(args.nproc_per_node)
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')

    stage_dir = outdir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    command_text = (
        f'NPROC_PER_NODE={args.nproc_per_node} \\\n'
        f'CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} \\\n'
        f'{shell_join(cmd)}\n'
    )
    (stage_dir / f'{stage_name}.launch.sh').write_text(command_text, encoding='utf-8')

    print('\n' + '=' * 80)
    print(f'RUN {stage_name.upper()} WITH EXISTING swift-infer.py')
    print('=' * 80)
    print(command_text)

    if args.dry_run:
        return stage_dir / 'swift_result.jsonl'

    log_path = stage_dir / f'{stage_name}.wrapper.log'
    info = run_cmd(cmd, env, log_path, dry_run=False)
    result_path = stage_dir / 'swift_result.jsonl'
    if not result_path.exists():
        # Some swift versions may ignore --result_path; try to locate a jsonl result.
        candidates = sorted(stage_dir.rglob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = [p for p in candidates if p.name not in {'val_dataset.messages.jsonl', 'index_map.jsonl'}]
        if candidates:
            result_path = candidates[0]
        else:
            raise FileNotFoundError(f'Cannot find stage result jsonl under {stage_dir}. Check log: {log_path}')
    info['result_path'] = str(result_path)
    write_json(stage_dir / f'{stage_name}.status.json', info)
    return result_path


def run_direct_swift_infer(
    *,
    records: List[Dict[str, Any]],
    model: str,
    prompt_field: str,
    run_dir: Path,
    args,
    max_new_tokens: int,
    stage_name: str,
) -> Path:
    """Run swift infer directly for the generated reranker records."""
    val_rows, index_rows = build_val_dataset(records, prompt_field=prompt_field, default_system=args.default_system)
    val_dataset_path = run_dir / 'val_dataset.messages.jsonl'
    index_path = run_dir / 'index_map.jsonl'
    result_path = run_dir / 'swift_result.jsonl'
    write_jsonl(val_dataset_path, val_rows)
    write_jsonl(index_path, index_rows)

    env = os.environ.copy()
    env['NPROC_PER_NODE'] = str(args.nproc_per_node)
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')

    cmd = [
        'swift', 'infer',
        '--model', model,
        '--infer_backend', args.infer_backend,
        '--model_type', args.model_type,
        '--torch_dtype', args.torch_dtype,
        '--enable_thinking', str(args.enable_thinking),
        '--val_dataset', str(val_dataset_path.resolve()),
        '--vllm_gpu_memory_utilization', str(args.vllm_gpu_memory_utilization),
        '--vllm_max_model_len', str(args.vllm_max_model_len),
        '--max_new_tokens', str(max_new_tokens),
    ]
    if args.temperature is not None:
        cmd += ['--temperature', str(args.temperature)]
    if args.top_p is not None:
        cmd += ['--top_p', str(args.top_p)]
    if args.result_path_arg:
        cmd += [args.result_path_arg, str(result_path.resolve())]
    if args.extra_swift_args:
        cmd += args.extra_swift_args

    command_text = (
        f'NPROC_PER_NODE={args.nproc_per_node} \\\n'
        f'CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} \\\n'
        f'{shell_join(cmd)}\n'
    )
    (run_dir / 'command.sh').write_text(command_text, encoding='utf-8')

    print('\n' + '=' * 80)
    print(f'RUN {stage_name.upper()} DIRECTLY')
    print('=' * 80)
    print(command_text)

    if args.dry_run:
        print('[dry-run] Reranker command saved but not executed.')
        return result_path

    log_path = run_dir / 'infer.log'
    info = run_cmd(cmd, env, log_path, dry_run=False)
    info['result_path'] = str(result_path)
    write_json(run_dir / f'{stage_name}.status.json', info)
    return result_path


# -----------------------------
# Result parsing
# -----------------------------

def strip_think(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    return text


def extract_response(row: Dict[str, Any], response_key: str = 'auto') -> str:
    if response_key != 'auto':
        if response_key not in row:
            raise KeyError(f'Cannot find response key `{response_key}` in result row keys={list(row.keys())}')
        return strip_think(row[response_key])

    # Common ms-swift/output variants.
    keys = [
        'response', 'predict', 'prediction', 'generated_text', 'output', 'answer', 'text',
        'completion', 'content', 'model_output', 'gen', 'infer_output'
    ]
    for k in keys:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, str):
                return strip_think(v)
            if isinstance(v, dict):
                # Sometimes response is nested.
                for kk in keys:
                    if kk in v and v[kk] is not None:
                        return strip_think(v[kk])
            return strip_think(json.dumps(v, ensure_ascii=False))

    # Some result rows keep the generated assistant message in messages.
    if 'messages' in row and isinstance(row['messages'], list):
        for m in reversed(row['messages']):
            if isinstance(m, dict) and m.get('role') == 'assistant' and m.get('content') is not None:
                return strip_think(m['content'])

    # Last resort: stringify whole row for debugging, but do not silently pass.
    raise KeyError(f'Cannot auto-detect response in result row keys={list(row.keys())}')


def load_planner_outputs(planner_result_path: Path, response_key: str = 'auto') -> List[str]:
    rows = read_jsonl(str(planner_result_path))
    outputs = []
    for i, row in enumerate(rows):
        try:
            outputs.append(extract_response(row, response_key=response_key))
        except Exception as e:
            raise RuntimeError(f'Failed to extract planner response at row={i}: {e}') from e
    return outputs


# -----------------------------
# Reranker prompt construction
# -----------------------------

def render_template_obj(obj: Any, planner_output: str, record: Dict[str, Any], args) -> Any:
    """Render placeholders inside a string/list/dict prompt object."""
    mapping = dict(record)
    mapping.update({
        args.planner_output_field: planner_output,
        'planner_output': planner_output,
        'planner_pred': planner_output,
        'planner_result': planner_output,
    })

    def render_str(s: str) -> str:
        # First support explicit replacement token, which is safer than Python format.
        for ph in [args.planner_placeholder, '{planner_output}', '{planner_pred}', '{planner_result}']:
            if ph and ph in s:
                s = s.replace(ph, planner_output)
        # Then support normal str.format_map for fields like {control_sentence}.
        try:
            s = s.format_map(DefaultFormatDict(mapping))
        except Exception:
            # Keep original if it contains unrelated braces, e.g. JSON examples.
            pass
        return s

    if isinstance(obj, str):
        return render_str(obj)
    if isinstance(obj, list):
        out = []
        for x in obj:
            if isinstance(x, dict):
                y = dict(x)
                if 'content' in y and isinstance(y['content'], str):
                    y['content'] = render_str(y['content'])
                out.append(y)
            else:
                out.append(render_template_obj(x, planner_output, record, args))
        return out
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = render_template_obj(v, planner_output, record, args)
        return out
    return obj


class DefaultFormatDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def has_planner_placeholder(obj: Any, args) -> bool:
    text = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    placeholders = [args.planner_placeholder, '{planner_output}', '{planner_pred}', '{planner_result}', '{' + args.planner_output_field + '}']
    return any(ph and ph in text for ph in placeholders)


def replace_by_regex(prompt_obj: Any, planner_output: str, args) -> Any:
    if not args.replace_regex:
        return None
    if not isinstance(prompt_obj, str):
        prompt_text = json.dumps(prompt_obj, ensure_ascii=False)
    else:
        prompt_text = prompt_obj
    replaced, n = re.subn(args.replace_regex, planner_output, prompt_text, count=1, flags=re.S)
    if n <= 0:
        raise ValueError(f'--replace-regex did not match the reranker prompt. regex={args.replace_regex}')
    return replaced


def default_build_reranker_prompt(record: Dict[str, Any], planner_output: str, args) -> str:
    """
    Fallback builder. Use only when explicitly allowed.
    This avoids using GT reranker_prompt, but may not be identical to your SFT prompt.
    Prefer providing a reranker_prompt_template with {planner_output}.
    """
    cell_id = record.get('cell_id', '')
    pert_id = record.get('pert_id', '')
    pert_name = record.get('pert_name', '')
    smiles = record.get('canonical_smiles', record.get('smiles', ''))
    dose = record.get('pert_idose', record.get('dose', ''))
    time_ = record.get('pert_itime', record.get('time', ''))
    control_sentence = record.get('control_sentence', record.get('ctrl_sentence', ''))

    return f"""You are given a control cell gene-expression sentence and a candidate gene block predicted by a planner.
Your task is to rerank the candidate genes according to their expected expression order after perturbation.

Cell: {cell_id}
Perturbation ID: {pert_id}
Perturbation name: {pert_name}
SMILES: {smiles}
Dose: {dose}
Time: {time_}

Control sentence:
{control_sentence}

Planner candidate genes:
{planner_output}

Output only the final reranked gene sequence. Do not explain.""".strip()


def build_one_e2e_reranker_prompt(record: Dict[str, Any], planner_output: str, args) -> Any:
    # 1) Best case: explicit template field in the data.
    template_fields = []
    if args.reranker_template_field:
        template_fields.append(args.reranker_template_field)
    template_fields += [
        'e2e_reranker_prompt_template',
        'reranker_prompt_template',
        'reranker_template',
        'reranker_e2e_prompt_template',
    ]
    for field in template_fields:
        if field and field in record and record[field]:
            return render_template_obj(record[field], planner_output, record, args)

    # 2) If reranker prompt itself has placeholder, render it.
    if args.reranker_prompt_field in record and has_planner_placeholder(record[args.reranker_prompt_field], args):
        return render_template_obj(record[args.reranker_prompt_field], planner_output, record, args)

    # 3) Regex replacement against existing reranker prompt.
    if args.replace_regex:
        if args.reranker_prompt_field not in record:
            raise KeyError(f'Cannot find --reranker-prompt-field `{args.reranker_prompt_field}` for regex replacement.')
        return replace_by_regex(record[args.reranker_prompt_field], planner_output, args)

    # 4) Explicitly allowed fallback. This does NOT use GT reranker candidates.
    if args.allow_fallback_builder:
        return default_build_reranker_prompt(record, planner_output, args)

    raise RuntimeError(
        'Cannot build e2e reranker prompt without risking GT leakage.\n'
        'Please provide one of the following:\n'
        '  1. a data field `reranker_prompt_template` or `e2e_reranker_prompt_template` containing {planner_output};\n'
        '  2. --reranker-template-field FIELD where FIELD contains {planner_output};\n'
        '  3. --replace-regex REGEX to replace the oracle candidate section in reranker_prompt;\n'
        '  4. --allow-fallback-builder to use a generic prompt built from raw fields.\n'
        'This stop is intentional: using raw reranker_prompt directly may leak GT blocks.'
    )


def build_e2e_reranker_records(records: List[Dict[str, Any]], planner_outputs: List[str], args) -> List[Dict[str, Any]]:
    if len(records) != len(planner_outputs):
        raise ValueError(f'Number mismatch: records={len(records)}, planner_outputs={len(planner_outputs)}')

    out_rows = []
    preview_rows = []
    for i, (record, planner_output) in enumerate(zip(records, planner_outputs)):
        new_record = dict(record)
        new_record[args.planner_output_field] = planner_output
        new_prompt = build_one_e2e_reranker_prompt(record, planner_output, args)
        new_record[args.generated_reranker_prompt_field] = new_prompt
        out_rows.append(new_record)

        if i < args.preview_num:
            preview_rows.append({
                'idx': i,
                'record_id': record.get('record_id', i),
                'planner_output': planner_output,
                'generated_reranker_prompt': new_prompt,
            })
    return out_rows, preview_rows


# -----------------------------
# Merge final outputs
# -----------------------------

def build_final_e2e_results(
    records: List[Dict[str, Any]],
    planner_outputs: List[str],
    reranker_result_path: Path,
    args,
) -> List[Dict[str, Any]]:
    reranker_rows = read_jsonl(str(reranker_result_path)) if reranker_result_path.exists() else []
    if reranker_rows and len(reranker_rows) != len(records):
        raise ValueError(f'Reranker result length mismatch: reranker={len(reranker_rows)}, records={len(records)}')

    final_rows = []
    for i, record in enumerate(records):
        reranker_output = ''
        raw_reranker_row = None
        if reranker_rows:
            raw_reranker_row = reranker_rows[i]
            reranker_output = extract_response(raw_reranker_row, response_key=args.reranker_response_key)

        final_rows.append({
            'idx': i,
            'record_id': record.get('record_id', i),
            'split': record.get('split', ''),
            'cell_id': record.get('cell_id', ''),
            'pert_id': record.get('pert_id', ''),
            'pert_name': record.get('pert_name', ''),
            args.planner_output_field: planner_outputs[i],
            args.final_output_field: reranker_output,
            'raw_reranker_result': raw_reranker_row,
        })
    return final_rows


# -----------------------------
# Args
# -----------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Two-stage planner -> reranker inference with ms-swift.')

    # Data/model paths.
    parser.add_argument('--test-data', type=str, default='/root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_id.jsonl')
    parser.add_argument('--swift-infer-script', type=str, default='./swift-infer.py', help='Existing single-stage wrapper in the same directory.')
    parser.add_argument('--planner-model', type=str, default='', help='Planner model path. Required if --planner-result is not provided.')
    parser.add_argument('--reranker-model', type=str, required=True, help='Reranker model path.')
    parser.add_argument('--planner-result', type=str, default='', help='Existing planner swift_result.jsonl. If set, skip planner inference.')

    # Prompt fields.
    parser.add_argument('--planner-prompt-field', type=str, default='planner_prompt')
    parser.add_argument('--reranker-prompt-field', type=str, default='reranker_prompt')
    parser.add_argument('--reranker-template-field', type=str, default='', help='Preferred reranker template field with {planner_output}.')
    parser.add_argument('--generated-reranker-prompt-field', type=str, default='e2e_reranker_prompt')
    parser.add_argument('--planner-output-field', type=str, default='planner_output')
    parser.add_argument('--final-output-field', type=str, default='reranker_output')
    parser.add_argument('--planner-placeholder', type=str, default='{planner_output}')
    parser.add_argument('--replace-regex', type=str, default='', help='Regex used to replace oracle candidate section in reranker_prompt.')
    parser.add_argument('--allow-fallback-builder', action='store_true', help='Use generic reranker prompt if no template/placeholder/regex exists.')
    parser.add_argument('--default-system', type=str, default='')

    # Result parse.
    parser.add_argument('--planner-response-key', type=str, default='auto')
    parser.add_argument('--reranker-response-key', type=str, default='auto')

    # Run control.
    parser.add_argument('--outdir', type=str, default='./id_runs')
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--split', type=str, default='all', help='all or a split value in record["split"].')
    parser.add_argument('--max-samples', type=int, default=-1)
    parser.add_argument('--preview-num', type=int, default=3)

    # Swift/inference args, aligned with old script.
    parser.add_argument('--nproc-per-node', type=int, default=4)
    parser.add_argument('--cuda-visible-devices', type=str, default='0,1,2,3')
    parser.add_argument('--infer-backend', type=str, default='vllm')
    parser.add_argument('--model-type', type=str, default='qwen3')
    parser.add_argument('--torch-dtype', type=str, default='float16')
    parser.add_argument('--enable-thinking', type=str, default='False')
    parser.add_argument('--vllm-gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--vllm-max-model-len', type=int, default=8192)
    parser.add_argument('--planner-max-new-tokens', type=int, default=8192)
    parser.add_argument('--reranker-max-new-tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--result-path-arg', type=str, default='--result_path')
    parser.add_argument('--extra-swift-args', nargs=argparse.REMAINDER, default=[], help='Extra args appended to swift infer. Put this argument last.')
    return parser


def main():
    args = build_arg_parser().parse_args()

    all_records = read_jsonl(args.test_data)
    records = all_records
    if args.split != 'all':
        records = [r for r in all_records if str(r.get('split', '')) == args.split]
    if args.max_samples is not None and args.max_samples > 0:
        records = records[:args.max_samples]

    if not records:
        raise RuntimeError(f'No records selected. test_data={args.test_data}, split={args.split}')

    if not args.planner_result and not args.planner_model:
        raise ValueError('Either --planner-result or --planner-model must be provided.')

    if args.run_name:
        run_name = args.run_name
    else:
        if args.planner_result:
            planner_name = 'existing_planner'
        else:
            planner_name = safe_name(args.planner_model)
        run_name = f'{planner_name}__to__{safe_name(args.reranker_model)}__e2e__{args.split}__{now_tag()}'

    root_run_dir = Path(args.outdir) / run_name
    if root_run_dir.exists() and args.overwrite:
        shutil.rmtree(root_run_dir)
    root_run_dir.mkdir(parents=True, exist_ok=True)

    selected_test_path = root_run_dir / 'selected_test_data.jsonl'
    write_jsonl(selected_test_path, records)
    write_json(root_run_dir / 'two_stage_config.json', vars(args))

    print(f'Loaded input records : {len(all_records)}')
    print(f'Used records         : {len(records)}')
    print(f'Split                : {args.split}')
    print(f'Run dir              : {root_run_dir.resolve()}')
    print(f'Selected test data   : {selected_test_path.resolve()}')

    # Stage 1: planner.
    if args.planner_result:
        planner_result_path = Path(args.planner_result)
        if not planner_result_path.exists():
            raise FileNotFoundError(f'--planner-result does not exist: {planner_result_path}')
        print(f'Use existing planner result: {planner_result_path}')
    else:
        planner_result_path = run_existing_swift_infer(
            script_path=args.swift_infer_script,
            test_data=selected_test_path,
            model=args.planner_model,
            prompt_field=args.planner_prompt_field,
            outdir=root_run_dir,
            run_name='planner',
            args=args,
            max_new_tokens=args.planner_max_new_tokens,
            stage_name='planner',
        )

    if args.dry_run and not planner_result_path.exists():
        print('[dry-run] Planner result not available; skip building reranker records.')
        return

    planner_outputs = load_planner_outputs(Path(planner_result_path), response_key=args.planner_response_key)
    if len(planner_outputs) != len(records):
        raise ValueError(
            f'Planner output length mismatch: planner_outputs={len(planner_outputs)}, records={len(records)}. '
            'Make sure --planner-result was generated from the same selected test data and order.'
        )

    planner_pred_path = root_run_dir / 'planner_outputs.jsonl'
    write_jsonl(planner_pred_path, [
        {
            'idx': i,
            'record_id': records[i].get('record_id', i),
            args.planner_output_field: out,
        }
        for i, out in enumerate(planner_outputs)
    ])

    # Stage 2 input construction.
    reranker_records, preview_rows = build_e2e_reranker_records(records, planner_outputs, args)
    reranker_input_path = root_run_dir / 'reranker_input_from_planner.jsonl'
    prompt_preview_path = root_run_dir / 'reranker_prompt_preview.json'
    write_jsonl(reranker_input_path, reranker_records)
    write_json(prompt_preview_path, preview_rows)

    print(f'Planner output path  : {planner_pred_path.resolve()}')
    print(f'Reranker input path  : {reranker_input_path.resolve()}')
    print(f'Prompt preview path  : {prompt_preview_path.resolve()}')

    # Stage 2: reranker.
    reranker_dir = root_run_dir / 'reranker'
    reranker_dir.mkdir(parents=True, exist_ok=True)
    reranker_result_path = run_direct_swift_infer(
        records=reranker_records,
        model=args.reranker_model,
        prompt_field=args.generated_reranker_prompt_field,
        run_dir=reranker_dir,
        args=args,
        max_new_tokens=args.reranker_max_new_tokens,
        stage_name='reranker',
    )

    if args.dry_run:
        print('[dry-run] Done.')
        return

    final_rows = build_final_e2e_results(records, planner_outputs, Path(reranker_result_path), args)
    final_path = root_run_dir / 'e2e_result.jsonl'
    write_jsonl(final_path, final_rows)

    print('\n' + '=' * 80)
    print('TWO-STAGE DONE')
    print('=' * 80)
    print(f'Run dir        : {root_run_dir.resolve()}')
    print(f'Planner result : {Path(planner_result_path).resolve()}')
    print(f'Reranker result: {Path(reranker_result_path).resolve()}')
    print(f'E2E result     : {final_path.resolve()}')


if __name__ == '__main__':
    main()
