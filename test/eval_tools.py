#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2S-style embedding evaluation for generated gene sentences.

What this script does:
1. Read GT sequences from test jsonl, default field = gt_sequence.
2. Read generated sequences from result jsonl, default auto-detects reranker_output/response/etc.
3. Convert each sequence to a C2S-style cell sentence: space-separated gene names.
4. Load an official/public C2S HuggingFace model as an independent embedding model.
5. Extract cell embeddings from hidden states.
6. Compute scFID-style FID, RBF-MMD, and sliced Wasserstein on embeddings.

Important:
- This is much closer to C2S-style scFID than rank-vector FID.
- It uses an independent C2S model for evaluation, not your trained perturbation model.
- Default model is vandijklab/C2S-Pythia-410m-cell-type-prediction, because the official tutorial
  recommends cell-type/tissue-prediction C2S models for strong cell embeddings.
- If you want to use C2S-Scale 1B, pass:
    --embed-model vandijklab/C2S-Scale-Pythia-1b-pt

Example:
python c2s_embedding_eval.py \
  --test-data /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_id.jsonl \
  --result-jsonl /root/dengjie/AI4SCI/PerturbR/test/id_runs/qwen3_two_stage_e2e_rerank_only/e2e_result.jsonl \
  --out /root/dengjie/AI4SCI/PerturbR/test/id_runs/qwen3_two_stage_e2e_rerank_only/c2s_embed_metrics.json \
  --true-field gt_sequence \
  --pred-field reranker_output \
  --embed-model vandijklab/C2S-Pythia-410m-cell-type-prediction \
  --n-genes 200 \
  --batch-size 8 \
  --device cuda \
  --dtype float16
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


DEFAULT_TRUE_FIELDS = [
    "gt_sequence",
    "target_sequence",
    "perturb_sequence",
    "perturbed_sequence",
    "treated_sequence",
    "response_sequence",
    "perturb_sentence",
    "treated_sentence",
]

DEFAULT_PRED_FIELDS = [
    "reranker_output",
    "final_output",
    "pred_sequence",
    "prediction",
    "predict",
    "response",
    "generated_text",
    "output",
    "text",
    "answer",
]

GENE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]*$")


def read_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_field(row: dict, candidates: Sequence[str], explicit: str = "") -> Tuple[str, Any]:
    if explicit:
        if explicit not in row:
            raise KeyError(f"Explicit field `{explicit}` not found. Available keys={list(row.keys())}")
        return explicit, row[explicit]
    for k in candidates:
        if k in row and row[k] not in (None, ""):
            return k, row[k]
    raise KeyError(f"None of fields {candidates} found. Available keys={list(row.keys())}")


def remove_think_blocks(s: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL | re.IGNORECASE)


def parse_sequence(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(g).strip() for g in x if str(g).strip()]
    if isinstance(x, dict):
        for k in ["sequence", "genes", "gene_sequence", "pred_sequence", "answer", "output"]:
            if k in x:
                return parse_sequence(x[k])
        return []

    s = remove_think_blocks(str(x)).replace("\\n", "\n").strip()

    # JSON/Python list string.
    if s.startswith("[") and s.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                return parse_sequence(loader(s))
            except Exception:
                pass

    # Cleanup common generated decorations.
    s = re.sub(r"```(?:json|python|text)?", " ", s, flags=re.IGNORECASE)
    s = s.replace("```", " ")
    s = re.sub(r"\bBlock\s*\d+\s*:", " ", s, flags=re.IGNORECASE)
    s = re.sub(
        r"\b(Perturbed|Predicted|Final|Gene|Genes|Sequence|Answer|Output)\s*(cell sentence|sequence)?\s*:",
        " ",
        s,
        flags=re.IGNORECASE,
    )

    toks = re.split(r"[\s,\[\]\(\)\{\}\"'`;|]+", s)
    out = []
    for tok in toks:
        tok = tok.strip().strip(".")
        if not tok:
            continue
        low = tok.lower()
        if low in {"block", "gene", "genes", "sequence", "answer", "output", "predicted", "perturbed"}:
            continue
        if GENE_RE.match(tok):
            out.append(tok)
    return out


def dedup_keep_first(seq: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for g in seq:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def align_rows(test_rows: List[dict], result_rows: List[dict], align_by_id: bool, id_field: str) -> List[Tuple[dict, dict]]:
    if not align_by_id:
        n = min(len(test_rows), len(result_rows))
        return list(zip(test_rows[:n], result_rows[:n]))

    mp = {}
    for r in result_rows:
        rid = r.get(id_field, r.get("idx", None))
        if rid is not None:
            mp[str(rid)] = r

    pairs = []
    missing = 0
    for t in test_rows:
        rid = t.get(id_field, t.get("idx", None))
        r = mp.get(str(rid))
        if r is None:
            missing += 1
            continue
        pairs.append((t, r))
    if missing:
        print(f"[warn] missing aligned result rows: {missing}")
    return pairs


def seq_to_sentence(seq: Sequence[str], n_genes: int) -> str:
    seq = dedup_keep_first(seq)
    if n_genes > 0:
        seq = seq[:n_genes]
    return " ".join(seq)


def load_sequences(args) -> Tuple[List[str], List[str], Dict[str, Any]]:
    test_rows = read_jsonl(args.test_data)
    result_rows = read_jsonl(args.result_jsonl)
    pairs = align_rows(test_rows, result_rows, args.align_by_id, args.id_field)
    if not pairs:
        raise RuntimeError("No aligned rows")

    true_field = args.true_field
    pred_field = args.pred_field
    true_sentences, pred_sentences = [], []
    examples = []
    empty_pred = 0

    for i, (t, r) in enumerate(pairs):
        tf, tv = pick_field(t, DEFAULT_TRUE_FIELDS, true_field)
        true_field = true_field or tf
        pf, pv = pick_field(r, DEFAULT_PRED_FIELDS, pred_field)
        pred_field = pred_field or pf

        true_seq = dedup_keep_first(parse_sequence(tv))
        pred_seq = dedup_keep_first(parse_sequence(pv))
        if not pred_seq:
            empty_pred += 1

        true_sentences.append(seq_to_sentence(true_seq, args.n_genes))
        pred_sentences.append(seq_to_sentence(pred_seq, args.n_genes))

        if i < 5:
            examples.append({
                "idx": i,
                "record_id": t.get(args.id_field, r.get(args.id_field, i)),
                "true_len": len(true_seq),
                "pred_len_unique": len(pred_seq),
                "true_sentence_head": true_seq[:20],
                "pred_sentence_head": pred_seq[:20],
                "raw_pred_preview": str(pv)[:500],
            })

    meta = {
        "num_test_rows": len(test_rows),
        "num_result_rows": len(result_rows),
        "num_aligned": len(pairs),
        "true_field": true_field,
        "pred_field": pred_field,
        "empty_pred_parse": empty_pred,
        "n_genes_used": args.n_genes,
        "examples": examples,
    }
    return true_sentences, pred_sentences, meta


def get_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def mean_pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def last_token_pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
    idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[idx, lengths]


@torch.no_grad()
def embed_sentences(sentences: List[str], args, tag: str) -> np.ndarray:
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    dtype = get_dtype(args.dtype)

    print(f"[load] tokenizer/model: {args.embed_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.embed_model,
        torch_dtype=dtype if device != "cpu" else torch.float32,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)

    embs = []
    for start in tqdm(range(0, len(sentences), args.batch_size), desc=f"embed {tag}"):
        batch = sentences[start : start + args.batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer]
        if args.pooling == "mean":
            emb = mean_pool_hidden(h, enc["attention_mask"])
        elif args.pooling == "last":
            emb = last_token_pool_hidden(h, enc["attention_mask"])
        else:
            raise ValueError(f"Unknown pooling={args.pooling}")
        if args.l2_normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        embs.append(emb.float().cpu().numpy())

    return np.concatenate(embs, axis=0)


def fid(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    cx = np.cov(x, rowvar=False) + np.eye(x.shape[1], dtype=np.float64) * eps
    cy = np.cov(y, rowvar=False) + np.eye(y.shape[1], dtype=np.float64) * eps
    cs = sqrtm(cx @ cy)
    if np.iscomplexobj(cs):
        cs = cs.real
    val = float(np.sum((mu_x - mu_y) ** 2) + np.trace(cx + cy - 2 * cs))
    return max(val, 0.0)


def median_sigma(x: np.ndarray, y: np.ndarray, max_points: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    z = np.vstack([x, y])
    if z.shape[0] > max_points:
        z = z[rng.choice(z.shape[0], size=max_points, replace=False)]
    d = cdist(z, z, metric="euclidean")
    tri = d[np.triu_indices_from(d, k=1)]
    tri = tri[tri > 0]
    return float(np.median(tri)) if tri.size else 1.0


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float, chunk: int = 1024, unbiased: bool = False) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    nx, ny = x.shape[0], y.shape[0]
    denom = 2.0 * sigma * sigma
    sx = sy = sxy = 0.0
    for i in range(0, nx, chunk):
        sx += float(np.exp(-cdist(x[i:i+chunk], x, "sqeuclidean") / denom).sum())
    for i in range(0, ny, chunk):
        sy += float(np.exp(-cdist(y[i:i+chunk], y, "sqeuclidean") / denom).sum())
    for i in range(0, nx, chunk):
        sxy += float(np.exp(-cdist(x[i:i+chunk], y, "sqeuclidean") / denom).sum())
    if unbiased:
        if nx < 2 or ny < 2:
            return float("nan")
        return (sx - nx) / (nx * (nx - 1)) + (sy - ny) / (ny * (ny - 1)) - 2 * sxy / (nx * ny)
    return sx / (nx * nx) + sy / (ny * ny) - 2 * sxy / (nx * ny)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 512, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = x.shape[1]
    dirs = rng.normal(size=(d, n_proj))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=0, keepdims=True), 1e-12)
    xp = np.sort(x @ dirs, axis=0)
    yp = np.sort(y @ dirs, axis=0)
    if xp.shape[0] != yp.shape[0]:
        m = max(xp.shape[0], yp.shape[0])
        q = np.linspace(0.0, 1.0, m)
        qx = np.linspace(0.0, 1.0, xp.shape[0])
        qy = np.linspace(0.0, 1.0, yp.shape[0])
        xp = np.stack([np.interp(q, qx, xp[:, j]) for j in range(n_proj)], axis=1)
        yp = np.stack([np.interp(q, qy, yp[:, j]) for j in range(n_proj)], axis=1)
    return float(np.mean(np.abs(xp - yp)))


def compute_metrics(true_emb: np.ndarray, pred_emb: np.ndarray, args) -> Dict[str, Any]:
    sigma = args.mmd_sigma if args.mmd_sigma > 0 else median_sigma(true_emb, pred_emb, seed=args.seed)
    mmd2 = mmd_rbf(true_emb, pred_emb, sigma=sigma, chunk=args.mmd_chunk_size, unbiased=args.mmd_unbiased)
    return {
        "n_true": int(true_emb.shape[0]),
        "n_pred": int(pred_emb.shape[0]),
        "dim": int(true_emb.shape[1]),
        "scfid": float(fid(true_emb, pred_emb, eps=args.fid_eps)),
        "mmd2_rbf": float(mmd2),
        "mmd_rbf": float(math.sqrt(max(mmd2, 0.0))),
        "mmd_sigma": float(sigma),
        "wasserstein_sliced": float(sliced_wasserstein(true_emb, pred_emb, n_proj=args.n_projections, seed=args.seed)),
        "lower_is_better": True,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compute C2S-embedding scFID/MMD/Wasserstein for generated gene sentences.")
    p.add_argument("--test-data", required=True)
    p.add_argument("--result-jsonl", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--true-field", default="gt_sequence")
    p.add_argument("--pred-field", default="")
    p.add_argument("--align-by-id", action="store_true")
    p.add_argument("--id-field", default="record_id")

    p.add_argument("--embed-model", default="vandijklab/C2S-Pythia-410m-cell-type-prediction")
    p.add_argument("--n-genes", type=int, default=200, help="Use top N genes from each sentence. Official 410M cell-type model was trained with top 200.")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--layer", type=int, default=-1, help="Hidden layer index used as embedding source.")
    p.add_argument("--pooling", choices=["mean", "last"], default="mean")
    p.add_argument("--l2-normalize", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--save-emb-prefix", default="", help="Optional prefix; saves .true.npy and .pred.npy")
    p.add_argument("--load-true-emb", default="", help="Skip true embedding if provided.")
    p.add_argument("--load-pred-emb", default="", help="Skip pred embedding if provided.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fid-eps", type=float, default=1e-6)
    p.add_argument("--mmd-sigma", type=float, default=0.0)
    p.add_argument("--mmd-chunk-size", type=int, default=1024)
    p.add_argument("--mmd-unbiased", action="store_true")
    p.add_argument("--n-projections", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()

    true_sentences, pred_sentences, input_meta = load_sequences(args)

    if args.load_true_emb and args.load_pred_emb:
        true_emb = np.load(args.load_true_emb)
        pred_emb = np.load(args.load_pred_emb)
    else:
        true_emb = embed_sentences(true_sentences, args, tag="true")
        # Free CUDA cache between passes.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pred_emb = embed_sentences(pred_sentences, args, tag="pred")

    if true_emb.shape != pred_emb.shape:
        raise ValueError(f"Embedding shape mismatch: true={true_emb.shape}, pred={pred_emb.shape}")

    metrics = compute_metrics(true_emb, pred_emb, args)
    out = {
        "input": input_meta,
        "embedding_model": {
            "model": args.embed_model,
            "n_genes": args.n_genes,
            "max_length": args.max_length,
            "layer": args.layer,
            "pooling": args.pooling,
            "l2_normalize": args.l2_normalize,
            "dtype": args.dtype,
        },
        "distribution_metrics": metrics,
        "timings": {"total_sec": time.perf_counter() - t0},
        "note": (
            "This computes scFID-style FID/MMD/Wasserstein on embeddings extracted by an independent C2S HuggingFace model. "
            "For exact paper reproduction, use the same C2S/scFM checkpoint, prompt formatting, n_genes, pooling, and preprocessing as the paper."
        ),
    }

    write_json(args.out, out)

    if args.save_emb_prefix:
        pref = Path(args.save_emb_prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(pref) + ".true.npy", true_emb)
        np.save(str(pref) + ".pred.npy", pred_emb)

    print(json.dumps(out["distribution_metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
