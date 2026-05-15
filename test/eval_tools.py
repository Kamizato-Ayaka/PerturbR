#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple scGPT embedding evaluation for generated perturbation gene sequences.

Default usage matches our current files:
  - test jsonl: field `gt_sequence`
  - result jsonl: field `reranker_output`
  - scGPT checkpoint: ModelScope `ZhejiangLab-LifeScience/scGPT`

Pipeline:
  gt_sequence / reranker_output
    -> clean gene sequences
    -> rank-derived pseudo-expression AnnData
    -> frozen pretrained scGPT embedding, X_scGPT
    -> scFID / RBF-MMD / sliced Wasserstein

Install:
  pip install modelscope scgpt anndata pandas scipy tqdm

If scgpt pip fails:
  git clone https://github.com/bowang-lab/scGPT.git
  cd scGPT && pip install -e .
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
import pandas as pd
import anndata as ad
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist


TRUE_FIELD = "gt_sequence"
PRED_FIELD_CANDIDATES = [
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
MODEL_SCOPE_ID = "ZhejiangLab-LifeScience/scGPT"

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

    if s.startswith("[") and s.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                return parse_sequence(loader(s))
            except Exception:
                pass

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
    genes = []
    for tok in toks:
        tok = tok.strip().strip(".")
        if not tok:
            continue
        if tok.lower() in {
            "block", "gene", "genes", "sequence", "answer", "output",
            "predicted", "perturbed", "final", "cell", "sentence",
        }:
            continue
        if GENE_RE.match(tok):
            genes.append(tok)
    return genes


def dedup_keep_first(seq: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for g in seq:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def pick_pred_field(row: dict, explicit: str = "") -> Tuple[str, Any]:
    if explicit:
        if explicit not in row:
            raise KeyError(f"--pred-field `{explicit}` not found. Available keys={list(row.keys())}")
        return explicit, row[explicit]
    for k in PRED_FIELD_CANDIDATES:
        if k in row and row[k] not in (None, ""):
            return k, row[k]
    raise KeyError(f"No prediction field found. Available keys={list(row.keys())}")


def build_vocab(true_seqs: List[List[str]]) -> List[str]:
    seen = set()
    vocab = []
    for seq in true_seqs:
        for g in seq:
            if g not in seen:
                seen.add(g)
                vocab.append(g)
    if not vocab:
        raise ValueError("Empty vocab from gt_sequence. Check test jsonl.")
    return vocab


def rank_score(pos: int, n: int) -> float:
    # Positive pseudo-expression. Top-ranked gene gets highest value.
    # This is rank-derived because our outputs are sequences, not measured expression values.
    return float(np.log1p(max(n - pos, 1)))


def sequences_to_matrix(seqs: List[List[str]], genes: List[str], max_genes: int) -> np.ndarray:
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    x = np.zeros((len(seqs), len(genes)), dtype=np.float32)

    for i, seq in enumerate(seqs):
        seq = dedup_keep_first(seq)
        if max_genes > 0:
            seq = seq[:max_genes]
        n = len(seq)
        for pos, g in enumerate(seq):
            j = gene_to_idx.get(g)
            if j is not None:
                x[i, j] = rank_score(pos, n)
    return x


def load_pairs(test_data: str, result_jsonl: str, pred_field: str = "") -> Tuple[ad.AnnData, Dict[str, Any]]:
    test_rows = read_jsonl(test_data)
    result_rows = read_jsonl(result_jsonl)
    n = min(len(test_rows), len(result_rows))
    test_rows = test_rows[:n]
    result_rows = result_rows[:n]

    true_seqs, pred_seqs = [], []
    examples = []
    empty_pred = 0
    detected_pred_field = pred_field

    for i, (t, r) in enumerate(zip(test_rows, result_rows)):
        if TRUE_FIELD not in t:
            raise KeyError(f"test row has no `{TRUE_FIELD}`. Available keys={list(t.keys())}")

        pf, pv = pick_pred_field(r, pred_field)
        detected_pred_field = detected_pred_field or pf

        tseq = dedup_keep_first(parse_sequence(t[TRUE_FIELD]))
        pseq = dedup_keep_first(parse_sequence(pv))
        if not pseq:
            empty_pred += 1

        true_seqs.append(tseq)
        pred_seqs.append(pseq)

        if i < 5:
            examples.append({
                "idx": i,
                "record_id": t.get("record_id", r.get("record_id", i)),
                "true_len": len(tseq),
                "pred_unique_len": len(pseq),
                "true_head": tseq[:20],
                "pred_head": pseq[:20],
                "raw_pred_preview": str(pv)[:500],
            })

    genes = build_vocab(true_seqs)

    true_x = sequences_to_matrix(true_seqs, genes, max_genes=len(genes))
    pred_x = sequences_to_matrix(pred_seqs, genes, max_genes=len(genes))

    x = np.vstack([true_x, pred_x]).astype(np.float32)

    obs = pd.DataFrame({
        "source": ["true"] * n + ["pred"] * n,
        "pair_idx": list(range(n)) + list(range(n)),
    })

    # scGPT embed_data uses this gene column by default in this script.
    var = pd.DataFrame({"feature_name": genes}, index=genes)

    adata = ad.AnnData(X=x, obs=obs, var=var, dtype="float32")

    gt_set = set(genes)
    hit_ratios = []
    for seq in pred_seqs:
        seq = dedup_keep_first(seq)
        hit_ratios.append(len([g for g in seq if g in gt_set]) / max(len(seq), 1))

    meta = {
        "num_test_rows": len(test_rows),
        "num_result_rows": len(result_rows),
        "num_aligned_by_order": n,
        "true_field": TRUE_FIELD,
        "pred_field": detected_pred_field,
        "empty_pred_parse": empty_pred,
        "gt_vocab_size": len(genes),
        "mean_pred_vocab_hit_ratio": float(np.mean(hit_ratios)),
        "examples": examples,
        "pseudo_expression": "score = log1p(num_genes_in_sequence - rank_position)",
    }
    return adata, meta


def find_scgpt_model_dir(model_dir: str = "") -> Path:
    if model_dir:
        p = Path(model_dir).expanduser().resolve()
    else:
        try:
            from modelscope import snapshot_download
        except Exception as e:
            raise ImportError("Please install ModelScope: pip install modelscope") from e
        p = Path(snapshot_download(MODEL_SCOPE_ID)).resolve()

    # Direct folder case.
    required = ["best_model.pt", "vocab.json", "args.json"]
    if all((p / f).exists() for f in required):
        return p

    # Some ModelScope repos may have one nested checkpoint folder.
    candidates = []
    for sub in [p] + [x for x in p.rglob("*") if x.is_dir()]:
        if all((sub / f).exists() for f in required):
            candidates.append(sub)
    if candidates:
        candidates = sorted(candidates, key=lambda x: len(str(x)))
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot find scGPT checkpoint files {required} under {p}. "
        f"Please pass --model-dir to a folder containing these files."
    )


def embed_with_scgpt(adata: ad.AnnData, model_dir: Path, batch_size: int, max_length: int, device: str) -> np.ndarray:
    try:
        from scgpt.tasks.cell_emb import embed_data
    except Exception as e:
        raise ImportError(
            "Cannot import scGPT embed_data. Install scGPT first:\n"
            "  pip install scgpt\n"
            "or:\n"
            "  git clone https://github.com/bowang-lab/scGPT.git && cd scGPT && pip install -e ."
        ) from e

    print(f"[scGPT] model_dir = {model_dir}")
    print(f"[scGPT] cells = {adata.n_obs}, genes = {adata.n_vars}")

    embedded = embed_data(
        adata,
        model_dir=model_dir,
        gene_col="feature_name",
        max_length=max_length,
        batch_size=batch_size,
        obs_to_save=None,
        device=device,
        use_fast_transformer=False,
        return_new_adata=False,
    )
    return np.asarray(embedded.obsm["X_scGPT"], dtype=np.float32)


def fid(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx, my = x.mean(axis=0), y.mean(axis=0)
    cx = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * eps
    cy = np.cov(y, rowvar=False) + np.eye(y.shape[1]) * eps
    covmean = sqrtm(cx @ cy)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    val = float(np.sum((mx - my) ** 2) + np.trace(cx + cy - 2.0 * covmean))
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


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float, chunk: int = 1024) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    nx, ny = x.shape[0], y.shape[0]
    denom = 2.0 * sigma * sigma
    sx = sy = sxy = 0.0

    for i in range(0, nx, chunk):
        sx += float(np.exp(-cdist(x[i:i + chunk], x, "sqeuclidean") / denom).sum())
    for i in range(0, ny, chunk):
        sy += float(np.exp(-cdist(y[i:i + chunk], y, "sqeuclidean") / denom).sum())
    for i in range(0, nx, chunk):
        sxy += float(np.exp(-cdist(x[i:i + chunk], y, "sqeuclidean") / denom).sum())

    return sx / (nx * nx) + sy / (ny * ny) - 2.0 * sxy / (nx * ny)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 512, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dim = x.shape[1]
    dirs = rng.normal(size=(dim, n_proj))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=0, keepdims=True), 1e-12)

    xp = np.sort(x @ dirs, axis=0)
    yp = np.sort(y @ dirs, axis=0)
    return float(np.mean(np.abs(xp - yp)))


def compute_metrics(true_emb: np.ndarray, pred_emb: np.ndarray) -> Dict[str, Any]:
    sigma = median_sigma(true_emb, pred_emb)
    mmd2 = mmd_rbf(true_emb, pred_emb, sigma=sigma)
    return {
        "n_true": int(true_emb.shape[0]),
        "n_pred": int(pred_emb.shape[0]),
        "dim": int(true_emb.shape[1]),
        "scfid": float(fid(true_emb, pred_emb)),
        "mmd2_rbf": float(mmd2),
        "mmd_rbf": float(math.sqrt(max(mmd2, 0.0))),
        "mmd_sigma": float(sigma),
        "wasserstein_sliced": float(sliced_wasserstein(true_emb, pred_emb)),
        "metric_direction": "lower_is_better",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple scGPT embedding eval for generated gene sequences.")
    p.add_argument("--test-data", required=True)
    p.add_argument("--result-jsonl", required=True)
    p.add_argument("--out", required=True)

    # Minimal optional args.
    p.add_argument("--model-dir", default="", help="Optional local scGPT folder. If empty, download ModelScope ZhejiangLab-LifeScience/scGPT.")
    p.add_argument("--pred-field", default="", help="Optional. Default auto-detects reranker_output/response/etc.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=1200)
    p.add_argument("--save-emb-prefix", default="")
    p.add_argument("--load-emb-prefix", default="", help="If provided, load PREFIX.true.npy and PREFIX.pred.npy and only recompute metrics.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    if args.load_emb_prefix:
        true_emb = np.load(args.load_emb_prefix + ".true.npy")
        pred_emb = np.load(args.load_emb_prefix + ".pred.npy")
        input_meta = {"loaded_embeddings_from": args.load_emb_prefix}
        model_dir = args.model_dir or f"modelscope:{MODEL_SCOPE_ID}"
    else:
        adata, input_meta = load_pairs(args.test_data, args.result_jsonl, pred_field=args.pred_field)
        model_dir = find_scgpt_model_dir(args.model_dir)
        emb = embed_with_scgpt(
            adata,
            model_dir=model_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )

        n = adata.n_obs // 2
        true_emb = emb[:n]
        pred_emb = emb[n:]

        if args.save_emb_prefix:
            prefix = Path(args.save_emb_prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(prefix) + ".true.npy", true_emb)
            np.save(str(prefix) + ".pred.npy", pred_emb)

    metrics = compute_metrics(true_emb, pred_emb)

    result = {
        "input": input_meta,
        "scgpt": {
            "model": str(model_dir),
            "source": MODEL_SCOPE_ID if not args.model_dir else "local",
            "embedding": "X_scGPT / CLS cell embedding from frozen pretrained scGPT",
            "note": (
                "The input sequences are rank-only. This script converts ranks into pseudo-expression "
                "scores before feeding scGPT. If real expression values are available, use them instead."
            ),
        },
        "distribution_metrics": metrics,
        "timings": {"total_sec": time.perf_counter() - t0},
    }

    write_json(args.out, result)
    write_json(Path(args.out).with_suffix(".debug.json"), {
        "examples": input_meta.get("examples", []),
        "metrics": metrics,
    })

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")
    print(f"debug: {Path(args.out).with_suffix('.debug.json')}")


if __name__ == "__main__":
    main()
