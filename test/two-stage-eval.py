#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate C2S-style perturbation prediction results.

Task types
----------
1) baseline
   Direct full-sequence prediction:
   control full sentence + perturbation -> full perturbed sentence.

2) recall
   Stage-1 planner / block recall:
   control full sentence + perturbation -> predicted perturbed blocks as gene sets.
   This evaluates set/block recovery only, not within-block order.

3) rerank
   Stage-2 block-level final prediction:
   input/control block + perturbation -> perturbed block sentence.
   This is evaluated as a final block-level sequence prediction. It does NOT require
   before/after delta against the input order.

Expected run directory layout
-----------------------------
The script can evaluate an ms-swift/vLLM run directory containing:
  - result jsonl, e.g. swift_result.jsonl / result.jsonl / infer_result.jsonl / predictions.jsonl
  - optional index_map.jsonl mapping result rows back to test-data rows
  - optional labels.jsonl

The script is intentionally robust to multiple field names in both test data and
result data, because different inference scripts often write different keys.

Main output files
-----------------
  summary.json
  all_details.jsonl / all_details.csv
  baseline_details.jsonl / baseline_details.csv / baseline_failures.jsonl
  recall_details.jsonl / recall_details.csv / recall_failures.jsonl
  rerank_details.jsonl / rerank_details.csv / rerank_failures.jsonl
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BAD_TOKENS = {
    "...", "....", ".....", ".", ",", ";", ":", "[", "]", "(", ")", "{", "}",
    "<|endoftext|>", "<|im_end|>", "<|im_start|>", "</s>", "<s>", "<pad>",
}

GENE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
BLOCK_LINE_RE = re.compile(r"^\s*(?:Block|block)\s*([0-9]+)\s*[:：]\s*(.+?)\s*$")
CTRL_TOKEN_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)
SECTION_STOP_RE = re.compile(
    r"\n\s*\[(?:Task|Instruction|Output|Response|Answer|Condition|Input Blocks|Control Expression|Perturbed Expression|Control|Input|Target)\]",
    re.IGNORECASE,
)

# -----------------------------
# IO utilities
# -----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path is None or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    rows.append({"__value__": obj})
            except json.JSONDecodeError:
                rows.append({"__raw_line__": line, "__line_no__": line_no})
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# -----------------------------
# Parsing utilities
# -----------------------------

def unique_keep_first(tokens: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def clean_sentence(sentence: Any, dedup: bool = False) -> List[str]:
    """Parse a sentence/list into gene-like tokens.

    This function is intentionally conservative: it removes obvious special tokens,
    punctuation, and malformed tokens, but keeps ordinary gene symbols such as
    HLA-DRA, MT-CO1, ENSG-like IDs, etc.
    """
    if sentence is None:
        return []
    if isinstance(sentence, list):
        raw_tokens: List[str] = []
        for x in sentence:
            if isinstance(x, str):
                raw_tokens.extend(CTRL_TOKEN_RE.sub(" ", x).replace("\n", " ").split())
            else:
                raw_tokens.append(str(x))
    else:
        raw_tokens = CTRL_TOKEN_RE.sub(" ", str(sentence)).replace("\n", " ").split()

    tokens: List[str] = []
    seen = set()
    for token in raw_tokens:
        token = token.strip().strip('"\'`,')
        token = token.strip()
        if not token or token in BAD_TOKENS:
            continue
        if set(token) == {"."}:
            continue
        if not GENE_TOKEN_RE.match(token):
            continue
        if dedup:
            if token in seen:
                continue
            seen.add(token)
        tokens.append(token)
    return tokens


def parse_blocks(text: Any) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {}
    if text is None:
        return blocks
    if isinstance(text, dict):
        return {str(k): clean_sentence(v, dedup=False) for k, v in text.items()}
    for line in str(text).splitlines():
        m = BLOCK_LINE_RE.match(line)
        if not m:
            continue
        block_id = str(int(m.group(1)))
        genes = clean_sentence(m.group(2), dedup=False)
        if genes:
            blocks[block_id] = genes
    return blocks


def flatten_blocks(blocks: Any) -> List[str]:
    if not isinstance(blocks, dict):
        return []
    try:
        keys = sorted(blocks.keys(), key=lambda x: int(x))
    except Exception:
        keys = sorted(blocks.keys(), key=str)
    out: List[str] = []
    for k in keys:
        out.extend(clean_sentence(blocks.get(k), dedup=False))
    return out


def extract_section_from_text(text: str, section_names: List[str]) -> str:
    if not text:
        return ""
    for name in section_names:
        pattern = re.compile(rf"\[{re.escape(name)}\]\s*\n", re.IGNORECASE)
        m = pattern.search(text)
        if not m:
            continue
        start = m.end()
        rest = text[start:]
        stop = SECTION_STOP_RE.search(rest)
        return rest[: stop.start()].strip() if stop else rest.strip()
    return ""


def messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            parts.append(f"[{msg.get('role', '')}]\n{msg.get('content', '')}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def get_nested(obj: Any, path: List[Any]) -> Any:
    cur = obj
    for key in path:
        try:
            if isinstance(key, int) and isinstance(cur, list):
                cur = cur[key]
            elif isinstance(cur, dict):
                cur = cur[key]
            else:
                return None
        except Exception:
            return None
    return cur


def extract_generation(row: Dict[str, Any]) -> str:
    if row is None:
        return ""
    direct_keys = [
        "response", "responses", "predict", "prediction", "generated_text", "text", "output",
        "answer", "content", "completion", "generated", "infer_result", "assistant",
    ]
    for key in direct_keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list) and val:
            if isinstance(val[0], str):
                return val[0]
            if isinstance(val[0], dict):
                s = extract_generation(val[0])
                if s:
                    return s
        if isinstance(val, dict):
            s = extract_generation(val)
            if s:
                return s

    nested_paths = [
        ["choices", 0, "message", "content"],
        ["choices", 0, "text"],
        ["outputs", 0, "text"],
        ["output", "text"],
        ["output", "choices", 0, "message", "content"],
        ["message", "content"],
    ]
    for path in nested_paths:
        val = get_nested(row, path)
        if isinstance(val, str) and val.strip():
            return val

    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                return str(msg.get("content"))

    raw = row.get("__raw_line__")
    if isinstance(raw, str):
        return raw
    return ""


def find_result_file(run_dir: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = run_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Explicit result file does not exist: {p}")
        return p

    candidates = [
        run_dir / "swift_result.jsonl",
        run_dir / "result.jsonl",
        run_dir / "infer_result.jsonl",
        run_dir / "predictions.jsonl",
        run_dir / "outputs.jsonl",
        run_dir / "raw" / "planner.jsonl",
        run_dir / "raw" / "baseline.jsonl",
        run_dir / "raw" / "reranker.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p

    ignore_names = {
        "val_dataset.messages.jsonl", "index_map.jsonl", "labels.jsonl",
        "details.jsonl", "all_details.jsonl",
    }
    all_jsonl = sorted(run_dir.rglob("*.jsonl"), key=lambda x: (len(str(x)), str(x)))
    for p in all_jsonl:
        if p.name not in ignore_names:
            return p
    raise FileNotFoundError(f"Cannot find result jsonl under run dir: {run_dir}")


def align_predictions(result_rows: List[Dict[str, Any]], index_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    has_idx = False
    for row in result_rows:
        idx = row.get("idx", row.get("index", row.get("sample_id", row.get("raw_idx", None))))
        if idx is not None:
            try:
                by_idx[int(idx)] = row
                has_idx = True
            except Exception:
                pass
    if has_idx:
        return [by_idx.get(i, {}) for i in range(len(index_rows))]
    return [result_rows[i] if i < len(result_rows) else {} for i in range(len(index_rows))]


def infer_sample_identity(index_row: Dict[str, Any], fallback_idx: int) -> Tuple[int, Any, str]:
    raw_idx = index_row.get("raw_idx", index_row.get("source_idx", index_row.get("idx", fallback_idx)))
    try:
        raw_idx_int = int(raw_idx)
    except Exception:
        raw_idx_int = fallback_idx
    record_id = index_row.get("record_id", raw_idx_int)
    split = index_row.get("split", "unknown")
    return raw_idx_int, record_id, split


# -----------------------------
# Field extraction
# -----------------------------

def get_sequence(record: Dict[str, Any], keys: List[str]) -> List[str]:
    for k in keys:
        v = record.get(k)
        if v:
            if isinstance(v, dict) and "genes" in v:
                return clean_sentence(v["genes"], dedup=False)
            return clean_sentence(v, dedup=False)
    return []


def get_gt_sequence(record: Dict[str, Any], label_row: Optional[Dict[str, Any]] = None) -> List[str]:
    if label_row:
        seq = get_sequence(label_row, [
            "label", "gt_sequence", "perturb_sentence", "target_sentence", "assistant",
            "perturbed_sentence", "target_genes", "target_block", "gt_genes", "gt_block",
        ])
        if seq:
            return seq
    seq = get_sequence(record, [
        "gt_sequence", "perturb_sentence", "perturbed_sentence", "target_sentence", "response_sentence",
        "perturb_expression", "target_sequence", "target_genes", "target_block", "gt_genes", "gt_block",
        "output_genes", "answer_genes",
    ])
    if seq:
        return seq
    for k in ["gt_reranker_blocks", "target_blocks", "perturb_blocks", "perturbed_blocks", "gt_blocks"]:
        if isinstance(record.get(k), dict):
            return flatten_blocks(record.get(k))
    return []


def get_control_sequence(record: Dict[str, Any]) -> List[str]:
    seq = get_sequence(record, [
        "control_sentence", "control_expression", "control_sequence", "input_sentence", "source_sentence",
        "control_genes", "source_sequence", "input_genes", "candidate_genes", "input_block",
        "candidate_block", "source_block",
    ])
    if seq:
        return seq
    for k in ["control_blocks", "input_blocks", "source_blocks", "gt_control_blocks"]:
        if isinstance(record.get(k), dict):
            return flatten_blocks(record.get(k))
    for k in ["planner_prompt", "reranker_prompt", "prompt"]:
        if record.get(k):
            text = record[k] if isinstance(record[k], str) else messages_to_text(record[k])
            sec = extract_section_from_text(text, ["Control Expression", "Control", "Input Sequence", "Input Block", "Candidate Genes"])
            if sec:
                return clean_sentence(sec, dedup=False)
    if record.get("messages"):
        sec = extract_section_from_text(messages_to_text(record.get("messages")), [
            "Control Expression", "Control", "Input Sequence", "Input Block", "Candidate Genes",
        ])
        if sec:
            return clean_sentence(sec, dedup=False)
    return []


def get_gt_blocks(record: Dict[str, Any]) -> Dict[str, List[str]]:
    for k in ["gt_planner_blocks", "planner_blocks", "target_planner_blocks", "target_blocks", "perturbed_blocks", "gt_blocks"]:
        v = record.get(k)
        if isinstance(v, dict):
            return {str(kk): clean_sentence(vv, dedup=False) for kk, vv in v.items()}
    return {}


# -----------------------------
# Metric utilities
# -----------------------------

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def rank_values(values: List[float]) -> List[float]:
    # Average rank for ties, 0-based ranks.
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[pairs[k][0]] = avg
        i = j
    return ranks


def spearman_from_lists(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson(rank_values(xs), rank_values(ys))


def build_complete_rank(seq: List[str], universe: List[str]) -> Dict[str, int]:
    """Build 1-based rank. Missing genes are assigned a shared bottom rank N+1.

    Duplicates are handled by keeping the first occurrence.
    """
    unique = unique_keep_first(seq)
    N = len(universe)
    pos = {g: i + 1 for i, g in enumerate(unique) if g in set(universe)}
    bottom = N + 1
    return {g: pos.get(g, bottom) for g in universe}


def kendall_tau_from_ranks(rank_a: Dict[str, int], rank_b: Dict[str, int], genes: List[str]) -> float:
    n = len(genes)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    # Ties caused by missing genes are ignored for stability.
    for i in range(n):
        gi = genes[i]
        for j in range(i + 1, n):
            gj = genes[j]
            da = rank_a[gi] - rank_a[gj]
            db = rank_b[gi] - rank_b[gj]
            if da == 0 or db == 0:
                continue
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else 0.0


def spearman_rank_corr(rank_a: Dict[str, int], rank_b: Dict[str, int], genes: List[str]) -> float:
    if len(genes) < 2:
        return 0.0
    return pearson([float(rank_a[g]) for g in genes], [float(rank_b[g]) for g in genes])


def mean_abs_rank_error(rank_pred: Dict[str, int], rank_gt: Dict[str, int], genes: List[str]) -> float:
    if not genes:
        return 0.0
    return sum(abs(rank_pred[g] - rank_gt[g]) for g in genes) / len(genes)


def normalized_spearman_footrule(rank_pred: Dict[str, int], rank_gt: Dict[str, int], genes: List[str]) -> float:
    """Normalized Spearman footrule distance. Lower is better.

    We normalize by N^2/2 for an N-length permutation scale. With missing ranks, the
    value may be approximate but remains useful as a rank-distance summary.
    """
    n = len(genes)
    if n <= 1:
        return 0.0
    foot = sum(abs(rank_pred[g] - rank_gt[g]) for g in genes)
    denom = (n * n) / 2.0
    return foot / denom if denom else 0.0


def topk_overlap(pred: List[str], gt: List[str], k: int) -> float:
    if k <= 0 or not pred or not gt:
        return 0.0
    p = set(unique_keep_first(pred)[:k])
    g = set(unique_keep_first(gt)[:k])
    return safe_div(len(p & g), min(k, len(g)))


def jaccard_similarity(pred: List[str], gt: List[str]) -> float:
    ps = set(pred)
    gs = set(gt)
    if not ps and not gs:
        return 1.0
    return safe_div(len(ps & gs), len(ps | gs))


def rank_weighted_jaccard(pred: List[str], gt: List[str]) -> float:
    pred_u = unique_keep_first(pred)
    gt_u = unique_keep_first(gt)
    pred_rank = {g: i + 1 for i, g in enumerate(pred_u)}
    gt_rank = {g: i + 1 for i, g in enumerate(gt_u)}
    genes = set(pred_rank) | set(gt_rank)
    if not genes:
        return 0.0
    num = 0.0
    den = 0.0
    for g in genes:
        wp = 1.0 / pred_rank[g] if g in pred_rank else 0.0
        wg = 1.0 / gt_rank[g] if g in gt_rank else 0.0
        num += min(wp, wg)
        den += max(wp, wg)
    return safe_div(num, den)


def sequence_lcs_ratio(pred: List[str], gt: List[str]) -> float:
    """LCS-based normalized sequence similarity. Higher is better."""
    if not pred or not gt:
        return 0.0
    # O(nm), but block sizes and 978-gene sequences are okay for eval.
    n, m = len(pred), len(gt)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        pi = pred[i - 1]
        for j in range(1, m + 1):
            if pi == gt[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[m]
    return 2.0 * lcs / (n + m) if n + m else 0.0


def duplicate_stats(tokens: List[str]) -> Tuple[int, float]:
    if not tokens:
        return 0, 0.0
    c = Counter(tokens)
    dup_count = sum(v - 1 for v in c.values() if v > 1)
    return dup_count, dup_count / len(tokens)


def validity_stats(pred: List[str], gt: List[str], allowed: Optional[set] = None) -> Dict[str, float]:
    if allowed is None:
        allowed = set(gt)
    pred_u = unique_keep_first(pred)
    valid = [g for g in pred if g in allowed]
    valid_u = unique_keep_first(valid)
    invalid_count = len(pred) - len(valid)
    dup_count, dup_rate = duplicate_stats(valid)
    target_n = len(set(gt)) if gt else len(allowed)
    missing = len(set(gt) - set(valid_u)) if gt else 0
    return {
        "pred_len": len(pred),
        "pred_unique_len": len(pred_u),
        "gt_len": len(gt),
        "invalid_gene_count": invalid_count,
        "invalid_gene_rate": safe_div(invalid_count, len(pred)),
        "valid_gene_rate": safe_div(len(valid), len(pred)),
        "duplicate_count": dup_count,
        "duplicate_rate": dup_rate,
        "missing_count": missing,
        "missing_rate": safe_div(missing, target_n),
        "length_error": safe_div(abs(len(valid_u) - target_n), target_n),
    }


def full_sequence_metrics(
    pred: List[str],
    gt: List[str],
    control: Optional[List[str]] = None,
    topks: Optional[List[int]] = None,
    allowed: Optional[set] = None,
) -> Dict[str, float]:
    """Evaluate a predicted final sentence against a target final sentence."""
    topks = topks or [100, 200]
    if not gt:
        return {}

    # Use GT order as canonical universe; append valid extra genes if needed.
    universe = unique_keep_first(gt)
    if allowed:
        for g in allowed:
            if g not in set(universe):
                universe.append(g)
    pred_u = unique_keep_first([g for g in pred if g in set(universe)])
    gt_u = unique_keep_first(gt)

    rank_pred = build_complete_rank(pred_u, universe)
    rank_gt = build_complete_rank(gt_u, universe)

    out: Dict[str, float] = {}
    out.update(validity_stats(pred, gt_u, allowed=set(universe)))
    out.update({
        "kendall_tau": kendall_tau_from_ranks(rank_pred, rank_gt, universe),
        "kendall_distance": (1.0 - kendall_tau_from_ranks(rank_pred, rank_gt, universe)) / 2.0,
        "spearman": spearman_rank_corr(rank_pred, rank_gt, universe),
        "mare": mean_abs_rank_error(rank_pred, rank_gt, universe),
        "spearman_footrule_dist": normalized_spearman_footrule(rank_pred, rank_gt, universe),
        "jaccard": jaccard_similarity(pred_u, gt_u),
        "jaccard_distance": 1.0 - jaccard_similarity(pred_u, gt_u),
        "rwjs": rank_weighted_jaccard(pred_u, gt_u),
        "lcs_ratio": sequence_lcs_ratio(pred_u, gt_u),
    })
    for k in topks:
        out[f"top{k}_overlap"] = topk_overlap(pred_u, gt_u, k)

    # Rank movement metrics: compare movement from control -> perturbed.
    if control:
        control_u = unique_keep_first([g for g in control if g in set(universe)])
        rank_ctrl = build_complete_rank(control_u, universe)
        delta_true = [float(rank_ctrl[g] - rank_gt[g]) for g in universe]
        delta_pred = [float(rank_ctrl[g] - rank_pred[g]) for g in universe]
        out["rank_delta_spearman"] = spearman_from_lists(delta_pred, delta_true)
        out["rank_delta_pearson"] = pearson(delta_pred, delta_true)
        # Up/down moved genes according to rank movement.
        for k in topks:
            k_eff = min(k, len(universe))
            true_up = set(sorted(universe, key=lambda g: rank_ctrl[g] - rank_gt[g], reverse=True)[:k_eff])
            pred_up = set(sorted(universe, key=lambda g: rank_ctrl[g] - rank_pred[g], reverse=True)[:k_eff])
            true_down = set(sorted(universe, key=lambda g: rank_ctrl[g] - rank_gt[g])[:k_eff])
            pred_down = set(sorted(universe, key=lambda g: rank_ctrl[g] - rank_pred[g])[:k_eff])
            out[f"up_recall@{k}"] = safe_div(len(true_up & pred_up), k_eff)
            out[f"down_recall@{k}"] = safe_div(len(true_down & pred_down), k_eff)
    else:
        out["rank_delta_spearman"] = 0.0
        out["rank_delta_pearson"] = 0.0
    return out


# -----------------------------
# Recall/block metrics
# -----------------------------

def block_recall_metrics(
    pred_blocks: Dict[str, List[str]],
    gt_blocks: Dict[str, List[str]],
    block_size: int,
    topks: List[int],
) -> Dict[str, float]:
    if not pred_blocks or not gt_blocks:
        return {
            "block_recall": 0.0,
            "weighted_block_recall": 0.0,
            "block_assignment_error": 0.0,
            "block_assignment_coverage": 0.0,
        }

    try:
        keys = sorted(set(pred_blocks) | set(gt_blocks), key=lambda x: int(x))
    except Exception:
        keys = sorted(set(pred_blocks) | set(gt_blocks), key=str)

    recalls = []
    weighted_num = 0.0
    weighted_den = 0.0
    for kk in keys:
        p = set(unique_keep_first(pred_blocks.get(kk, [])))
        g = set(unique_keep_first(gt_blocks.get(kk, [])))
        rec = safe_div(len(p & g), len(g)) if g else 0.0
        recalls.append(rec)
        try:
            t = int(kk)
        except Exception:
            t = len(recalls)
        w = 1.0 / max(t, 1)
        weighted_num += w * rec
        weighted_den += w

    pred_flat = flatten_blocks(pred_blocks)
    gt_flat = flatten_blocks(gt_blocks)
    pred_flat_u = unique_keep_first(pred_flat)
    gt_flat_u = unique_keep_first(gt_flat)

    out = {
        "pred_blocks": len(pred_blocks),
        "gt_blocks": len(gt_blocks),
        "pred_len": len(pred_flat),
        "pred_unique_len": len(pred_flat_u),
        "gt_len": len(gt_flat_u),
        "duplicate_rate": duplicate_stats(pred_flat)[1],
        "block_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "weighted_block_recall": safe_div(weighted_num, weighted_den),
        "set_recall": safe_div(len(set(pred_flat_u) & set(gt_flat_u)), len(set(gt_flat_u))),
        "set_precision": safe_div(len(set(pred_flat_u) & set(gt_flat_u)), len(set(pred_flat_u))),
    }

    # Cumulative top-k recall. This treats block outputs as a concatenated coarse sentence.
    for k in topks:
        out[f"cum_recall_top{k}"] = topk_overlap(pred_flat_u, gt_flat_u, k)

    # Assignment error: for genes that appear in both pred and gt, compare block ids.
    gt_bid: Dict[str, int] = {}
    pred_bid: Dict[str, int] = {}
    for kk, genes in gt_blocks.items():
        try:
            bid = int(kk)
        except Exception:
            continue
        for g in unique_keep_first(genes):
            gt_bid[g] = bid
    for kk, genes in pred_blocks.items():
        try:
            bid = int(kk)
        except Exception:
            continue
        for g in unique_keep_first(genes):
            if g not in pred_bid:
                pred_bid[g] = bid
    common = sorted(set(gt_bid) & set(pred_bid))
    out["block_assignment_coverage"] = safe_div(len(common), len(gt_bid))
    out["block_assignment_error"] = sum(abs(pred_bid[g] - gt_bid[g]) for g in common) / len(common) if common else 0.0
    out["block_accuracy"] = safe_div(sum(1 for g in common if pred_bid[g] == gt_bid[g]), len(common))
    return out


# -----------------------------
# Evaluation tasks
# -----------------------------

def load_run(run_dir: Path, explicit_file: Optional[str], test_records: List[Dict[str, Any]]) -> Tuple[Path, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    result_path = find_result_file(run_dir, explicit_file)
    result_rows = read_jsonl(result_path)
    index_rows = read_jsonl(run_dir / "index_map.jsonl")
    label_rows = read_jsonl(run_dir / "labels.jsonl")
    if not index_rows:
        index_rows = [
            {"idx": i, "raw_idx": i, "record_id": i, "split": test_records[i].get("split", "unknown") if i < len(test_records) else "unknown"}
            for i in range(len(result_rows))
        ]
    aligned = align_predictions(result_rows, index_rows)
    return result_path, result_rows, index_rows, label_rows, aligned


def eval_baseline(
    test_records: List[Dict[str, Any]],
    run_dir: Path,
    result_file: Optional[str],
    topks: List[int],
    min_pred_genes: int,
    strict_gene_vocab: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    result_path, result_rows, index_rows, label_rows, aligned = load_run(run_dir, result_file, test_records)
    details: List[Dict[str, Any]] = []
    status_counter = Counter()

    for i, index_row in enumerate(index_rows):
        raw_idx, record_id, split = infer_sample_identity(index_row, i)
        record = test_records[raw_idx] if 0 <= raw_idx < len(test_records) else {}
        label_row = label_rows[i] if i < len(label_rows) else None
        gt = get_gt_sequence(record, label_row)
        control = get_control_sequence(record)
        text = extract_generation(aligned[i])
        pred = clean_sentence(text, dedup=False)

        allowed = set(gt) | set(control) if strict_gene_vocab else set(gt) if gt else None
        metrics = full_sequence_metrics(pred, gt, control=control, topks=topks, allowed=allowed)
        success = int(bool(pred) and len(unique_keep_first(pred)) >= min_pred_genes and bool(gt))
        row = {
            "task": "baseline",
            "idx": i,
            "raw_idx": raw_idx,
            "record_id": record_id,
            "split": split,
            "success": success,
            "non_empty": int(bool(pred)),
            "prediction_preview": " ".join(pred[:60]),
        }
        row.update(metrics)
        details.append(row)
        status_counter["success" if success else "fail"] += 1
        if not pred:
            status_counter["empty"] += 1
        elif len(unique_keep_first(pred)) < min_pred_genes:
            status_counter["too_short"] += 1

    summary = make_summary("baseline", run_dir, result_path, result_rows, index_rows, details, status_counter)
    return details, summary


def eval_rerank(
    test_records: List[Dict[str, Any]],
    run_dir: Path,
    result_file: Optional[str],
    topks: List[int],
    min_pred_genes: int,
    strict_input_vocab: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate rerank as block-level final sequence prediction.

    Unlike before/after rerank evaluation, this task treats the model output as the
    final block answer. Input order is not required. If an input/candidate block is
    available, it is used only as the control/source sequence for movement metrics and
    as the allowed vocabulary when strict_input_vocab is enabled.
    """
    result_path, result_rows, index_rows, label_rows, aligned = load_run(run_dir, result_file, test_records)
    details: List[Dict[str, Any]] = []
    status_counter = Counter()

    for i, index_row in enumerate(index_rows):
        raw_idx, record_id, split = infer_sample_identity(index_row, i)
        record = test_records[raw_idx] if 0 <= raw_idx < len(test_records) else {}
        label_row = label_rows[i] if i < len(label_rows) else None
        gt = get_gt_sequence(record, label_row)
        control = get_control_sequence(record)
        text = extract_generation(aligned[i])
        pred = clean_sentence(text, dedup=False)

        if strict_input_vocab and control:
            allowed = set(control)
        else:
            allowed = set(gt) | set(control) if control else set(gt)

        metrics = full_sequence_metrics(pred, gt, control=control, topks=topks, allowed=allowed)
        # Rename full-seq ranking keys to block-level keys for clarity.
        renamed = {}
        for k, v in metrics.items():
            if k in {"kendall_tau", "kendall_distance", "spearman", "mare", "spearman_footrule_dist"}:
                renamed[f"block_{k}"] = v
            elif k.startswith("top") and k.endswith("_overlap"):
                renamed[f"block_{k}"] = v
            elif k in {"rank_delta_spearman", "rank_delta_pearson"}:
                renamed[f"block_{k}"] = v
            else:
                renamed[k] = v

        success = int(bool(pred) and len(unique_keep_first(pred)) >= min_pred_genes and bool(gt))
        row = {
            "task": "rerank",
            "idx": i,
            "raw_idx": raw_idx,
            "record_id": record_id,
            "split": split,
            "success": success,
            "non_empty": int(bool(pred)),
            "input_found": int(bool(control)),
            "prediction_preview": " ".join(pred[:60]),
        }
        row.update(renamed)
        details.append(row)
        status_counter["success" if success else "fail"] += 1
        if not pred:
            status_counter["empty"] += 1
        elif len(unique_keep_first(pred)) < min_pred_genes:
            status_counter["too_short"] += 1

    summary = make_summary("rerank", run_dir, result_path, result_rows, index_rows, details, status_counter)
    return details, summary


def eval_recall(
    test_records: List[Dict[str, Any]],
    run_dir: Path,
    result_file: Optional[str],
    topks: List[int],
    block_size: int,
    min_blocks: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    result_path, result_rows, index_rows, label_rows, aligned = load_run(run_dir, result_file, test_records)
    del label_rows
    details: List[Dict[str, Any]] = []
    status_counter = Counter()

    for i, index_row in enumerate(index_rows):
        raw_idx, record_id, split = infer_sample_identity(index_row, i)
        record = test_records[raw_idx] if 0 <= raw_idx < len(test_records) else {}
        gt_blocks = get_gt_blocks(record)
        # If gt blocks are not stored, derive them from full perturbed sequence.
        if not gt_blocks:
            gt_seq = get_gt_sequence(record, None)
            gt_blocks = {str(j + 1): gt_seq[j * block_size : (j + 1) * block_size] for j in range(math.ceil(len(gt_seq) / block_size))}
        text = extract_generation(aligned[i])
        pred_blocks = parse_blocks(text)
        metrics = block_recall_metrics(pred_blocks, gt_blocks, block_size=block_size, topks=topks)
        success = int(bool(pred_blocks) and len(pred_blocks) >= min_blocks and bool(gt_blocks))
        pred_flat = flatten_blocks(pred_blocks)
        row = {
            "task": "recall",
            "idx": i,
            "raw_idx": raw_idx,
            "record_id": record_id,
            "split": split,
            "success": success,
            "non_empty": int(bool(pred_flat)),
            "prediction_preview": " ".join(pred_flat[:60]),
        }
        row.update(metrics)
        details.append(row)
        status_counter["success" if success else "fail"] += 1
        if not pred_blocks:
            status_counter["empty_or_no_blocks"] += 1
        elif len(pred_blocks) < min_blocks:
            status_counter["too_few_blocks"] += 1

    summary = make_summary("recall", run_dir, result_path, result_rows, index_rows, details, status_counter)
    return details, summary


# -----------------------------
# Summary and CLI
# -----------------------------

def mean_metric(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"count": len(rows)}
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    for k in keys:
        out[k] = mean_metric(rows, k)
    return out


def make_summary(
    task: str,
    run_dir: Path,
    result_path: Path,
    result_rows: List[Dict[str, Any]],
    index_rows: List[Dict[str, Any]],
    details: List[Dict[str, Any]],
    status_counter: Counter,
) -> Dict[str, Any]:
    by_split: Dict[str, Any] = {}
    for sp in sorted({r.get("split", "unknown") for r in details}):
        by_split[sp] = summarize_rows([r for r in details if r.get("split", "unknown") == sp])
    return {
        "task": task,
        "run_dir": str(run_dir),
        "result_file": str(result_path),
        "num_result_rows": len(result_rows),
        "num_index_rows": len(index_rows),
        "status_counter": dict(status_counter),
        "global": summarize_rows(details),
        "by_split": by_split,
    }


def parse_topks(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return sorted(set(out))


def print_task_summary(task: str, summary: Dict[str, Any], topks: List[int]) -> None:
    g = summary.get("global", {})
    print(f"\n[{task}]")
    print(f"Result file : {summary.get('result_file')}")
    print(f"Rows        : result={summary.get('num_result_rows')} index={summary.get('num_index_rows')}")
    print(f"Status      : {summary.get('status_counter')}")

    # Print task-specific compact metrics.
    if task == "baseline":
        keys = [
            "success", "valid_gene_rate", "duplicate_rate", "missing_rate",
            "kendall_tau", "spearman", "mare", "kendall_distance", "spearman_footrule_dist",
            "rwjs", "jaccard_distance", "rank_delta_spearman",
        ]
        keys += [f"top{k}_overlap" for k in topks]
    elif task == "rerank":
        keys = [
            "success", "valid_gene_rate", "duplicate_rate", "missing_rate",
            "block_kendall_tau", "block_spearman", "block_mare", "block_kendall_distance",
            "block_spearman_footrule_dist", "rwjs", "jaccard_distance", "block_rank_delta_spearman",
        ]
        keys += [f"block_top{k}_overlap" for k in topks]
    else:
        keys = [
            "success", "block_recall", "weighted_block_recall", "set_recall",
            "block_assignment_error", "block_assignment_coverage", "block_accuracy",
        ]
        keys += [f"cum_recall_top{k}" for k in topks]

    for k in keys:
        if k in g:
            print(f"{k:<30s}: {g.get(k, 0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C2S-style baseline, recall, and rerank sentence predictions.")
    parser.add_argument("--test-data", type=str, required=True, help="Original test jsonl used to build inference data.")
    parser.add_argument("--baseline-run-dir", type=str, default="", help="Run dir for direct full-sequence prediction baseline.")
    parser.add_argument("--recall-run-dir", type=str, default="", help="Run dir for stage-1 block recall/planner.")
    parser.add_argument("--rerank-run-dir", type=str, default="", help="Run dir for stage-2 block-level final prediction.")
    parser.add_argument("--baseline-result-file", type=str, default="")
    parser.add_argument("--recall-result-file", type=str, default="")
    parser.add_argument("--rerank-result-file", type=str, default="")
    parser.add_argument("--outdir", type=str, default="./c2s_sentence_eval_v2")
    parser.add_argument("--topks", type=str, default="20,50,100,200", help="Top-k values for overlap/movement metrics.")
    parser.add_argument("--block-size", type=int, default=125, help="Used to derive GT blocks when gt block fields are absent.")
    parser.add_argument("--min-baseline-pred-genes", type=int, default=100)
    parser.add_argument("--min-rerank-pred-genes", type=int, default=10)
    parser.add_argument("--min-recall-blocks", type=int, default=1)
    parser.add_argument("--strict-gene-vocab", action="store_true", help="For baseline, restrict valid genes to GT/control vocabulary.")
    parser.add_argument("--strict-rerank-input-vocab", action="store_true", help="For rerank, restrict valid genes to input/candidate block if available.")
    parser.add_argument("--save-failures", type=int, default=100)
    args = parser.parse_args()

    test_path = Path(args.test_data).resolve()
    test_records = read_jsonl(test_path)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)

    summaries: Dict[str, Any] = {
        "test_data": str(test_path),
        "num_test_records": len(test_records),
        "topks": topks,
        "block_size": args.block_size,
        "tasks": {},
    }
    all_details: List[Dict[str, Any]] = []

    print(f"Loaded test records: {len(test_records)}")
    print(f"Output dir: {outdir}")

    if args.baseline_run_dir:
        print(f"\nEvaluating baseline/full-sequence prediction: {args.baseline_run_dir}")
        details, summary = eval_baseline(
            test_records=test_records,
            run_dir=Path(args.baseline_run_dir).resolve(),
            result_file=args.baseline_result_file or None,
            topks=topks,
            min_pred_genes=args.min_baseline_pred_genes,
            strict_gene_vocab=args.strict_gene_vocab,
        )
        summaries["tasks"]["baseline"] = summary
        all_details.extend(details)
        write_jsonl(outdir / "baseline_details.jsonl", details)
        write_csv(outdir / "baseline_details.csv", details)
        write_jsonl(outdir / "baseline_failures.jsonl", [r for r in details if not r.get("success")][: args.save_failures])

    if args.recall_run_dir:
        print(f"\nEvaluating stage-1 recall/planner: {args.recall_run_dir}")
        details, summary = eval_recall(
            test_records=test_records,
            run_dir=Path(args.recall_run_dir).resolve(),
            result_file=args.recall_result_file or None,
            topks=topks,
            block_size=args.block_size,
            min_blocks=args.min_recall_blocks,
        )
        summaries["tasks"]["recall"] = summary
        all_details.extend(details)
        write_jsonl(outdir / "recall_details.jsonl", details)
        write_csv(outdir / "recall_details.csv", details)
        write_jsonl(outdir / "recall_failures.jsonl", [r for r in details if not r.get("success")][: args.save_failures])

    if args.rerank_run_dir:
        print(f"\nEvaluating stage-2 rerank/block-level prediction: {args.rerank_run_dir}")
        details, summary = eval_rerank(
            test_records=test_records,
            run_dir=Path(args.rerank_run_dir).resolve(),
            result_file=args.rerank_result_file or None,
            topks=topks,
            min_pred_genes=args.min_rerank_pred_genes,
            strict_input_vocab=args.strict_rerank_input_vocab,
        )
        summaries["tasks"]["rerank"] = summary
        all_details.extend(details)
        write_jsonl(outdir / "rerank_details.jsonl", details)
        write_csv(outdir / "rerank_details.csv", details)
        write_jsonl(outdir / "rerank_failures.jsonl", [r for r in details if not r.get("success")][: args.save_failures])

    if all_details:
        write_jsonl(outdir / "all_details.jsonl", all_details)
        write_csv(outdir / "all_details.csv", all_details)
    write_json(outdir / "summary.json", summaries)

    print("\n" + "=" * 88)
    print("C2S SENTENCE EVAL SUMMARY")
    print("=" * 88)
    for task, summary in summaries["tasks"].items():
        print_task_summary(task, summary, topks)
    print(f"\nSaved summary: {outdir / 'summary.json'}")
    print(f"Saved details: {outdir}")


if __name__ == "__main__":
    main()
