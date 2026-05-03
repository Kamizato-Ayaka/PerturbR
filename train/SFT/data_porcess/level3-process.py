#!/usr/bin/env python3

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
    p.add_argument("--sig-info", required=True)
    p.add_argument("--pert-info", default=None)
    p.add_argument("--outdir", default="../GSE92742")

    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--topk", type=int, default=978)
    p.add_argument("--lm-only", action="store_true")
    p.add_argument("--dedup-symbol", action="store_true", default=True)

    p.add_argument("--pair-mode", choices=["none", "by_cell_time", "by_cell_time_dose"], default="by_cell_time")
    p.add_argument("--target-time", default=None)

    p.add_argument("--control-types", nargs="*", default=["ctl_vehicle", "ctl_untrt", "ctl_vector"])
    p.add_argument("--treat-types", nargs="*", default=["trt_cp", "trt_xpr"])

    p.add_argument("--sampling-pairs-per-treat", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-control-pool-size", type=int, default=1)

    return p.parse_args()


def read_table(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", usecols=usecols, low_memory=False)


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def open_matrix_and_ids(gctx_path: str):
    if gctx_path.endswith(".gz"):
        raise ValueError(f"{gctx_path} is gzip-compressed. Please gunzip it first.")
    f = h5py.File(gctx_path, "r")
    matrix = f["/0/DATA/0/matrix"]
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
    cols = [
        "sig_id", "pert_id", "pert_iname", "pert_type",
        "cell_id", "pert_idose", "pert_itime",
        "canonical_smiles", "distil_id",
    ]
    si = sig_info[[c for c in cols if c in sig_info.columns]].copy()
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

    ranked_symbols = ranked_symbols[ranked_symbols != ""]
    return " ".join(ranked_symbols[:topk].tolist())


def process_chunk(args: Tuple[int, int, np.ndarray]) -> ChunkResult:
    start, end, sample_ids_chunk = args
    global _GCTX_PATH, _GENE_SYMBOLS, _TOPK, _DEDUP_SYMBOL, _KEEP_GENE_IDX

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
    gctx_path: str, matrix_shape, all_sample_ids: np.ndarray,
    gene_symbols: np.ndarray, keep_gene_idx: np.ndarray,
    workers: int, chunk_size: int, topk: int, dedup_symbol: bool,
) -> pd.DataFrame:
    n_samples = matrix_shape[0]
    chunks = []

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunks.append((start, end, all_sample_ids[start:end]))

    rows = []

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
                print(f"finished {idx}/{len(futures)} chunks")

    return pd.DataFrame(rows)


def build_sample_level_df(meta_df: pd.DataFrame, sentence_df: pd.DataFrame) -> pd.DataFrame:
    df = meta_df.merge(sentence_df, on="sample_id", how="inner")
    df = df.dropna(subset=["cell_sentence"]).reset_index(drop=True)
    return df


def normalize_key(key):
    if isinstance(key, tuple):
        return key
    return (key,)


def build_random_sample_pairs(
    sample_df: pd.DataFrame, pair_mode: str,
    control_types: Sequence[str], treat_types: Sequence[str],
    pairs_per_treat: int, seed: int, min_control_pool_size: int,
) -> pd.DataFrame:
    if pair_mode == "none":
        return pd.DataFrame()

    rng = np.random.default_rng(seed)

    df = sample_df.copy()
    df["is_control"] = df["pert_type"].isin(control_types)
    df["is_treat"] = df["pert_type"].isin(treat_types)

    key_cols = ["cell_id", "pert_itime"]
    if pair_mode == "by_cell_time_dose":
        key_cols.append("pert_idose")

    controls = df[df["is_control"]].copy()
    treats = df[df["is_treat"]].copy()

    # ========================== 新增逻辑：去重 ==========================
    # 每个细胞（cell_id）的每种扰动（pert_id）只保留一条数据
    if not treats.empty:
        treats = treats.drop_duplicates(subset=["cell_id", "pert_id"], keep="first").reset_index(drop=True)
        print(f"Deduplicated treats to {len(treats)} unique cell-perturbation combinations.")
    # ==================================================================

    if controls.empty or treats.empty:
        return pd.DataFrame()

    control_pool = {}
    for key, group in controls.groupby(key_cols, dropna=False):
        key = normalize_key(key)
        if len(group) >= min_control_pool_size:
            control_pool[key] = group.reset_index(drop=True)

    rows = []
    for _, treat_row in treats.iterrows():
        key = tuple(treat_row[c] for c in key_cols)

        if key not in control_pool:
            continue

        pool = control_pool[key]

        for _ in range(pairs_per_treat):
            ctrl_idx = int(rng.integers(0, len(pool)))
            ctrl_row = pool.iloc[ctrl_idx]

            if treat_row["pert_type"] == "trt_cp":
                perturbation_type = "drug"
            elif treat_row["pert_type"] == "trt_xpr":
                perturbation_type = "crispr"
            else:
                perturbation_type = str(treat_row["pert_type"])

            pert_name = treat_row.get("pert_iname", "")
            if perturbation_type == "drug" and str(pert_name).startswith("BRD-"):
                pert_name = ""

            rows.append({
                "cell_id": treat_row.get("cell_id", ""),
                "perturbation_type": perturbation_type,
                "pert_id": treat_row.get("pert_id", ""),
                "pert_name": pert_name,
                "canonical_smiles": treat_row.get("canonical_smiles", ""),
                "pert_itime": treat_row.get("pert_itime", ""),
                "pert_idose": treat_row.get("pert_idose", ""),
                "control_sample_id": ctrl_row.get("sample_id", ""),
                "perturb_sample_id": treat_row.get("sample_id", ""),
                "control_pert_type": ctrl_row.get("pert_type", ""),
                "control_pert_id": ctrl_row.get("pert_id", ""),
                "control_pert_name": ctrl_row.get("pert_iname", ""),
                "control_sentence": ctrl_row.get("cell_sentence", ""),
                "perturb_sentence": treat_row.get("cell_sentence", ""),
                "control_pool_size": int(len(pool)),
            })

    return pd.DataFrame(rows)


def load_sig_info(sig_info_path: str, pert_info_path: Optional[str]) -> pd.DataFrame:
    sig_info = read_table(
        sig_info_path,
        usecols=[
            "sig_id", "pert_id", "pert_iname", "pert_type",
            "cell_id", "pert_idose", "pert_itime", "distil_id",
        ],
    )

    if pert_info_path:
        pert_info = read_table(pert_info_path, usecols=["pert_id", "canonical_smiles"])
        sig_info = sig_info.merge(pert_info, on="pert_id", how="left")
        sig_info["canonical_smiles"] = sig_info["canonical_smiles"].fillna("")
    else:
        sig_info["canonical_smiles"] = ""

    # ========================== 新增逻辑：过滤 ==========================
    # 过滤掉 canonical_smiles 是 restricted 或者 -666 的数据
    invalid_smiles = {"restricted", "-666"}
    original_len = len(sig_info)
    sig_info = sig_info[~sig_info["canonical_smiles"].astype(str).str.strip().isin(invalid_smiles)].copy()
    filtered_len = len(sig_info)
    if original_len != filtered_len:
        print(f"Filtered out {original_len - filtered_len} records with restricted or -666 canonical_smiles.")
    # ==================================================================

    return sig_info


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("loading metadata")
    gene_info = read_table(args.gene_info, usecols=["pr_gene_id", "pr_gene_symbol", "pr_is_lm"])
    sig_info = load_sig_info(args.sig_info, args.pert_info)

    print("opening gctx")
    f, matrix, sample_ids, gene_ids = open_matrix_and_ids(args.gctx)

    try:
        print(f"matrix shape = {matrix.shape}")

        print("building gene axis")
        gene_symbols, keep_mask = build_gene_axis(gene_info, gene_ids, lm_only=args.lm_only)
        keep_gene_idx = np.flatnonzero(keep_mask)

        print("aligning metadata")
        meta_df = explode_sig_to_sample(sig_info)
        sample_id_set = set(sample_ids.tolist())
        meta_df = meta_df[meta_df["sample_id"].isin(sample_id_set)].copy()
        meta_df = meta_df.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)

        if args.target_time is not None:
            meta_df = meta_df[meta_df["pert_itime"].astype(str) == str(args.target_time)].copy()

        print(f"matched metadata samples = {len(meta_df)}")

        print("building sample-level sentences")
        sentence_df = build_sentences_parallel(
            args.gctx, matrix.shape, sample_ids,
            gene_symbols, keep_gene_idx,
            args.workers, args.chunk_size, args.topk, args.dedup_symbol,
        )

        sample_sentence_out = outdir / "sample_sentences.jsonl"
        write_jsonl(sample_sentence_out, sentence_df.to_dict(orient="records"))
        print(f"saved: {sample_sentence_out}")

        print("building sample-level dataframe")
        sample_level_df = build_sample_level_df(meta_df, sentence_df)

        sample_meta_out = outdir / "sample_level_with_meta.jsonl"
        write_jsonl(sample_meta_out, sample_level_df.to_dict(orient="records"))
        print(f"saved: {sample_meta_out}")

        print("building random pairs")
        paired_df = build_random_sample_pairs(
            sample_level_df,
            pair_mode=args.pair_mode,
            control_types=args.control_types,
            treat_types=args.treat_types,
            pairs_per_treat=args.sampling_pairs_per_treat,
            seed=args.seed,
            min_control_pool_size=args.min_control_pool_size,
        )

        pair_out = outdir / f"paired_sentences_sample_random_seed{args.seed}.jsonl"
        write_jsonl(pair_out, paired_df.to_dict(orient="records"))
        print(f"saved: {pair_out}")
        print(f"num pairs = {len(paired_df)}")
        print("done")

    finally:
        f.close()


if __name__ == "__main__":
    main()


