#!/usr/bin/env python3
"""
Build minimal JSONL outputs from LINCS L1000 Level3 GCTX.

Optimized for downstream SFT / RL tasks:
1) Treat Level3 matrix as samples x genes.
2) Export minimal JSONL instead of parquet.
3) Optionally keep only a fixed time point, e.g. 24 h.
4) Aggregate repeated samples under the same condition by mean expression.
5) Build minimal pairs so each record contains only one control sentence and one perturb sentence.

Outputs:
1) sample_sentences.jsonl
   One row per aggregated condition-level sentence.
2) paired_sentences.jsonl
   One row per minimal pair: control + perturb.

Recommended use:
- paired mode with --target-time "24 h"
- mean aggregation over repeated samples
- one control per cell/time, then pair perturbations in that cell/time to that control

Important:
- h5py cannot directly open .gctx.gz. Please gunzip first.
- If you use --mode paired, your metadata must belong to the same release as the GCTX file.
  Do NOT mix a GSE92742 matrix with GSE70138 sig/inst metadata.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


# -----------------------------
# Globals for worker processes
# -----------------------------
_GCTX_PATH: Optional[str] = None
_GENE_SYMBOLS: Optional[np.ndarray] = None
_TOPK: Optional[int] = None
_DEDUP_SYMBOL: bool = True
_KEEP_GENE_IDX: Optional[np.ndarray] = None


@dataclass
class ChunkResult:
    rows: List[dict]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gctx", required=True)
    p.add_argument("--gene-info", required=True)
    p.add_argument("--sig-info", default=None, help="Required only for paired mode.")
    p.add_argument("--outdir", required=True)

    p.add_argument("--mode", choices=["sentence_only", "paired"], default="sentence_only")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--topk", type=int, default=978)
    p.add_argument("--lm-only", action="store_true", help="Only keep landmark genes (pr_is_lm == 1).")
    p.add_argument("--dedup-symbol", action="store_true", default=True, help="Deduplicate repeated gene symbols after ranking.")
    p.add_argument("--no-dedup-symbol", dest="dedup_symbol", action="store_false")

    p.add_argument("--pair-mode", choices=["none", "by_cell_time", "by_cell_time_dose"], default="by_cell_time")
    p.add_argument("--target-time", default=None, help='Keep only one time point, e.g. "24 h".')
    p.add_argument("--aggregate-replicates", action="store_true", default=True, help="Aggregate repeated samples under the same condition by mean expression.")
    p.add_argument("--no-aggregate-replicates", dest="aggregate_replicates", action="store_false")
    p.add_argument("--control-types", nargs="*", default=["ctl_vehicle", "ctl_untrt", "ctl_vector"])
    p.add_argument("--treat-types", nargs="*", default=["trt_cp", "trt_xpr"])
    return p.parse_args()


def read_table(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", usecols=usecols, low_memory=False)


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def open_matrix_and_ids(gctx_path: str):
    if gctx_path.endswith(".gz"):
        raise ValueError(
            f"{gctx_path} is gzip-compressed. Please gunzip it first and pass the .gctx file."
        )

    f = h5py.File(gctx_path, "r")
    matrix = f["/0/DATA/0/matrix"]

    # In this file: rows are samples, cols are genes.
    sample_ids = f["/0/META/COL/id"][:].astype(str)
    gene_ids = f["/0/META/ROW/id"][:].astype(str)

    return f, matrix, sample_ids, gene_ids


def build_gene_axis(gene_info: pd.DataFrame, gene_ids: np.ndarray, lm_only: bool) -> Tuple[np.ndarray, np.ndarray]:
    gi = gene_info[["pr_gene_id", "pr_gene_symbol", "pr_is_lm"]].copy()
    gi["pr_gene_id"] = gi["pr_gene_id"].astype(str)
    gi["pr_gene_symbol"] = gi["pr_gene_symbol"].fillna("").astype(str)

    gene_map = gi.set_index("pr_gene_id")
    gene_symbols = pd.Series(gene_ids).map(gene_map["pr_gene_symbol"]).fillna("").to_numpy(dtype=object)
    lm_mask = pd.Series(gene_ids).map(gene_map["pr_is_lm"]).fillna(0).astype(int).to_numpy(dtype=np.int8) == 1

    keep_mask = np.array([s != "" for s in gene_symbols], dtype=bool)
    if lm_only:
        keep_mask &= lm_mask
    return gene_symbols, keep_mask


def explode_sig_to_sample(sig_info: pd.DataFrame) -> pd.DataFrame:
    si = sig_info[["sig_id", "pert_id", "pert_iname", "pert_type", "cell_id", "pert_idose", "pert_itime", "distil_id"]].copy()
    si["distil_id"] = si["distil_id"].fillna("")
    si = si[si["distil_id"] != ""].copy()
    si["sample_id"] = si["distil_id"].str.split("|", regex=False)
    si = si.explode("sample_id", ignore_index=True)
    si = si.drop(columns=["distil_id"])
    si["sample_id"] = si["sample_id"].astype(str)
    return si


def init_worker(gctx_path: str, gene_symbols: np.ndarray, keep_gene_idx: np.ndarray, topk: int, dedup_symbol: bool):
    global _GCTX_PATH, _GENE_SYMBOLS, _TOPK, _DEDUP_SYMBOL, _KEEP_GENE_IDX
    _GCTX_PATH = gctx_path
    _GENE_SYMBOLS = gene_symbols
    _KEEP_GENE_IDX = keep_gene_idx
    _TOPK = topk
    _DEDUP_SYMBOL = dedup_symbol


def make_ranked_sentence(values: np.ndarray, symbols: np.ndarray, topk: int, dedup_symbol: bool) -> str:
    order = np.argsort(-values, kind="stable")
    ranked_symbols = symbols[order]

    if dedup_symbol:
        seen = set()
        kept = []
        for s in ranked_symbols:
            if not s or s in seen:
                continue
            seen.add(s)
            kept.append(s)
            if len(kept) >= topk:
                break
        return " ".join(kept)
    else:
        ranked_symbols = ranked_symbols[ranked_symbols != ""]
        return " ".join(ranked_symbols[:topk].tolist())


def process_chunk(args: Tuple[int, int, np.ndarray]) -> ChunkResult:
    start, end, sample_ids_chunk = args
    global _GCTX_PATH, _GENE_SYMBOLS, _TOPK, _DEDUP_SYMBOL, _KEEP_GENE_IDX
    assert _GCTX_PATH is not None
    assert _GENE_SYMBOLS is not None
    assert _TOPK is not None
    assert _KEEP_GENE_IDX is not None

    rows: List[dict] = []
    with h5py.File(_GCTX_PATH, "r") as f:
        matrix = f["/0/DATA/0/matrix"]
        block = matrix[start:end, :][:, _KEEP_GENE_IDX]

    symbols = _GENE_SYMBOLS[_KEEP_GENE_IDX]
    for sample_id, values in zip(sample_ids_chunk, block):
        sentence = make_ranked_sentence(values, symbols, _TOPK, _DEDUP_SYMBOL)
        rows.append({
            "sample_id": str(sample_id),
            "cell_sentence": sentence,
        })

    del block
    gc.collect()
    return ChunkResult(rows=rows)


def build_sentences_parallel(
    gctx_path: str,
    matrix_shape: Tuple[int, int],
    all_sample_ids: np.ndarray,
    gene_symbols: np.ndarray,
    keep_gene_idx: np.ndarray,
    workers: int,
    chunk_size: int,
    topk: int,
    dedup_symbol: bool,
) -> pd.DataFrame:
    n_samples = matrix_shape[0]
    if len(all_sample_ids) != n_samples:
        raise RuntimeError(
            f"sample axis mismatch: len(sample_ids)={len(all_sample_ids)} but matrix rows={n_samples}"
        )

    chunks: List[Tuple[int, int, np.ndarray]] = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunks.append((start, end, all_sample_ids[start:end]))

    rows: List[dict] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(gctx_path, gene_symbols, keep_gene_idx, topk, dedup_symbol),
    ) as ex:
        futures = [ex.submit(process_chunk, ch) for ch in chunks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            rows.extend(res.rows)
            if idx % 50 == 0 or idx == len(futures):
                print(f"  finished {idx}/{len(futures)} chunks")

    return pd.DataFrame(rows)


def aggregate_condition_mean(sample_meta: pd.DataFrame, gctx_path: str, keep_gene_idx: np.ndarray) -> pd.DataFrame:
    key_cols = ["cell_id", "pert_id", "pert_iname", "pert_type", "pert_idose", "pert_itime"]
    grouped = sample_meta.groupby(key_cols, dropna=False)["sample_id"].apply(list).reset_index()

    with h5py.File(gctx_path, "r") as f:
        matrix = f["/0/DATA/0/matrix"]
        sample_ids = f["/0/META/COL/id"][:].astype(str)
        sample_to_pos = {sid: i for i, sid in enumerate(sample_ids)}

        agg_rows = []
        for idx, row in grouped.iterrows():
            sids = [s for s in row["sample_id"] if s in sample_to_pos]
            if not sids:
                continue
            pos = [sample_to_pos[s] for s in sids]
            block = matrix[pos, :][:, keep_gene_idx]
            mean_values = np.asarray(block).mean(axis=0)
            agg_rows.append({
                "cell_id": row["cell_id"],
                "pert_id": row["pert_id"],
                "pert_iname": row["pert_iname"],
                "pert_type": row["pert_type"],
                "pert_idose": row["pert_idose"],
                "pert_itime": row["pert_itime"],
                "sample_ids": sids,
                "mean_values": mean_values,
            })
            if (idx + 1) % 1000 == 0:
                print(f"  aggregated {idx + 1}/{len(grouped)} conditions")

    return pd.DataFrame(agg_rows)


def build_sentence_from_mean_df(mean_df: pd.DataFrame, gene_symbols: np.ndarray, keep_gene_idx: np.ndarray, topk: int, dedup_symbol: bool) -> pd.DataFrame:
    symbols = gene_symbols[keep_gene_idx]
    rows = []
    for _, row in mean_df.iterrows():
        sentence = make_ranked_sentence(row["mean_values"], symbols, topk, dedup_symbol)
        rows.append({
            "cell_id": row["cell_id"],
            "pert_id": row["pert_id"],
            "pert_iname": row["pert_iname"],
            "pert_type": row["pert_type"],
            "pert_idose": row["pert_idose"],
            "pert_itime": row["pert_itime"],
            "n_replicates": len(row["sample_ids"]),
            "sample_ids": row["sample_ids"],
            "cell_sentence": sentence,
        })
    return pd.DataFrame(rows)


def build_minimal_pairs(sample_df: pd.DataFrame, pair_mode: str, control_types: Sequence[str], treat_types: Sequence[str]) -> pd.DataFrame:
    if pair_mode == "none":
        return pd.DataFrame()

    df = sample_df.copy()
    df["is_control"] = df["pert_type"].isin(control_types)
    df["is_treat"] = df["pert_type"].isin(treat_types)

    key_cols = ["cell_id", "pert_itime"]
    if pair_mode == "by_cell_time_dose":
        key_cols.append("pert_idose")

    controls = df[df["is_control"]].copy()
    treats = df[df["is_treat"]].copy()
    if controls.empty or treats.empty:
        return pd.DataFrame()

    # Keep only one control per grouping key.
    controls = controls.sort_values(
        key_cols + ["n_replicates", "pert_iname"],
        ascending=[True] * len(key_cols) + [False, True],
    )
    controls = controls.groupby(key_cols, as_index=False).head(1).reset_index(drop=True)

    paired = treats.merge(
        controls[key_cols + ["cell_sentence", "n_replicates", "sample_ids"]].rename(columns={
            "cell_sentence": "control_sentence",
            "n_replicates": "control_n_replicates",
            "sample_ids": "control_sample_ids",
        }),
        on=key_cols,
        how="inner",
    )

    paired = paired.rename(columns={
        "cell_sentence": "perturb_sentence",
        "n_replicates": "perturb_n_replicates",
        "sample_ids": "perturb_sample_ids",
    })

    out_cols = [
        "cell_id", "pert_itime", "pert_idose", "pert_id", "pert_iname",
        "control_sentence", "perturb_sentence",
        "control_n_replicates", "perturb_n_replicates",
        "control_sample_ids", "perturb_sample_ids",
    ]
    out_cols = [c for c in out_cols if c in paired.columns]
    return paired[out_cols].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading metadata...")
    gene_info = read_table(args.gene_info, usecols=["pr_gene_id", "pr_gene_symbol", "pr_is_lm"])

    sig_info = None
    if args.mode == "paired":
        if not args.sig_info:
            raise ValueError("--sig-info is required when --mode paired")
        sig_info = read_table(
            args.sig_info,
            usecols=["sig_id", "pert_id", "pert_iname", "pert_type", "cell_id", "pert_idose", "pert_itime", "distil_id"],
        )

    print("[2/6] opening gctx...")
    f, matrix, sample_ids, gene_ids = open_matrix_and_ids(args.gctx)
    try:
        print(f"matrix shape = {matrix.shape} (samples x genes)")

        print("[3/6] building gene axis...")
        gene_symbols, keep_mask = build_gene_axis(gene_info, gene_ids, lm_only=args.lm_only)
        keep_gene_idx = np.flatnonzero(keep_mask)
        print(f"kept genes = {len(keep_gene_idx)} / {len(gene_ids)}")

        if args.mode == "sentence_only":
            print("[4/6] building ranked sentences in parallel...")
            sentence_df = build_sentences_parallel(
                gctx_path=args.gctx,
                matrix_shape=matrix.shape,
                all_sample_ids=sample_ids,
                gene_symbols=gene_symbols,
                keep_gene_idx=keep_gene_idx,
                workers=args.workers,
                chunk_size=args.chunk_size,
                topk=args.topk,
                dedup_symbol=args.dedup_symbol,
            )
            sentence_df = sentence_df.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
            sample_out = outdir / "sample_sentences.jsonl"
            write_jsonl(sample_out, sentence_df.to_dict(orient="records"))
            print(f"saved: {sample_out}")
            print("done")
            return

        print("[4/6] aligning paired metadata...")
        assert sig_info is not None
        meta_df = explode_sig_to_sample(sig_info)
        sample_id_set = set(sample_ids.tolist())
        meta_df = meta_df[meta_df["sample_id"].isin(sample_id_set)].copy()
        meta_df = meta_df.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
        print(f"aligned paired metadata rows = {len(meta_df)}")

        if len(meta_df) == 0:
            raise RuntimeError(
                "No sample metadata aligned to the Level3 matrix. "
                "Most likely you are mixing metadata from a different GEO release with this GCTX file."
            )

        if args.target_time is not None:
            before = len(meta_df)
            meta_df = meta_df[meta_df["pert_itime"].astype(str) == str(args.target_time)].copy()
            print(f"filtered by target_time={args.target_time}: {before} -> {len(meta_df)}")

        if len(meta_df) == 0:
            raise RuntimeError("No metadata rows remain after time filtering.")

        print("[5/6] aggregating repeated samples by mean...")
        if args.aggregate_replicates:
            mean_df = aggregate_condition_mean(meta_df, args.gctx, keep_gene_idx)
            sentence_df = build_sentence_from_mean_df(mean_df, gene_symbols, keep_gene_idx, args.topk, args.dedup_symbol)
        else:
            raw_sentence_df = build_sentences_parallel(
                gctx_path=args.gctx,
                matrix_shape=matrix.shape,
                all_sample_ids=sample_ids,
                gene_symbols=gene_symbols,
                keep_gene_idx=keep_gene_idx,
                workers=args.workers,
                chunk_size=args.chunk_size,
                topk=args.topk,
                dedup_symbol=args.dedup_symbol,
            )
            sentence_df = meta_df.merge(raw_sentence_df, on="sample_id", how="inner")
            sentence_df["n_replicates"] = 1
            sentence_df["sample_ids"] = sentence_df["sample_id"].apply(lambda x: [x])

        sample_out = outdir / "sample_sentences.jsonl"
        write_jsonl(sample_out, sentence_df.to_dict(orient="records"))
        print(f"saved: {sample_out}")

        print("[6/6] building minimal pairs...")
        paired_df = build_minimal_pairs(
            sentence_df,
            pair_mode=args.pair_mode,
            control_types=args.control_types,
            treat_types=args.treat_types,
        )
        pair_out = outdir / "paired_sentences.jsonl"
        write_jsonl(pair_out, paired_df.to_dict(orient="records"))
        print(f"saved: {pair_out}")
        print("done")
    finally:
        f.close()


if __name__ == "__main__":
    main()
