import os
import io
import gzip
import json
import math
import argparse
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
import pandas as pd


GLOBAL_MAT = None
GLOBAL_ROW_SYMBOLS = None
GLOBAL_LM_INDICES = None
GLOBAL_SELECTED_RECORDS = None
GLOBAL_TOPK = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--cells", type=str, default="A375,MCF7,PC3")
    parser.add_argument("--pert_type", type=str, default="trt_cp")
    parser.add_argument("--time_text", type=str, default="24 h")
    parser.add_argument("--topk", type=int, default=20)

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--max_per_cell", type=int, default=0)

    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_tsv_gz(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_gctx_from_gzip(gctx_gz_path):
    with gzip.open(gctx_gz_path, "rb") as f:
        gctx_bytes = f.read()
    bio = io.BytesIO(gctx_bytes)
    return h5py.File(bio, "r")


def round4(x):
    return round(float(x), 4)


def init_worker(mat, row_symbols, lm_indices, selected_records, topk):
    global GLOBAL_MAT
    global GLOBAL_ROW_SYMBOLS
    global GLOBAL_LM_INDICES
    global GLOBAL_SELECTED_RECORDS
    global GLOBAL_TOPK

    GLOBAL_MAT = mat
    GLOBAL_ROW_SYMBOLS = row_symbols
    GLOBAL_LM_INDICES = lm_indices
    GLOBAL_SELECTED_RECORDS = selected_records
    GLOBAL_TOPK = topk


def process_one(idx):
    row = GLOBAL_SELECTED_RECORDS[idx]
    sig_idx = row["matrix_idx"]

    vec = np.asarray(GLOBAL_MAT[sig_idx, :], dtype=np.float32)
    lm_vec = vec[GLOBAL_LM_INDICES]

    topk = GLOBAL_TOPK
    if topk > len(lm_vec):
        topk = len(lm_vec)

    up_idx = np.argpartition(lm_vec, -topk)[-topk:]
    up_idx = up_idx[np.argsort(lm_vec[up_idx])[::-1]]

    down_idx = np.argpartition(lm_vec, topk)[:topk]
    down_idx = down_idx[np.argsort(lm_vec[down_idx])]

    top_up = []
    top_down = []

    for i in up_idx:
        top_up.append([
            GLOBAL_ROW_SYMBOLS[i],
            round4(lm_vec[i])
        ])

    for i in down_idx:
        top_down.append([
            GLOBAL_ROW_SYMBOLS[i],
            round4(lm_vec[i])
        ])

    item = {
        "cell": row["cell_id"],
        "perturbation": row["pert_iname"],
        "dose": row["pert_idose"],
        "time": row["pert_itime"],
        "top_up": top_up,
        "top_down": top_down
    }
    return json.dumps(item, ensure_ascii=False)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    cell_list = [x.strip() for x in args.cells.split(",") if x.strip()]

    sig_path = os.path.join(raw_dir, "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz")
    gene_path = os.path.join(raw_dir, "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz")
    gctx_path = os.path.join(raw_dir, "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz")

    print("Loading metadata...")
    sig = read_tsv_gz(sig_path)
    gene = read_tsv_gz(gene_path)

    # filter signatures
    selected = sig.copy()
    if args.pert_type:
        selected = selected[selected["pert_type"] == args.pert_type]
    if args.time_text:
        selected = selected[selected["pert_itime"] == args.time_text]
    if cell_list:
        selected = selected[selected["cell_id"].isin(cell_list)]

    if args.max_per_cell and args.max_per_cell > 0:
        chunks = []
        for c in cell_list:
            sub = selected[selected["cell_id"] == c].copy()
            if len(sub) > args.max_per_cell:
                sub = sub.sample(args.max_per_cell, random_state=42)
            chunks.append(sub)
        selected = pd.concat(chunks, axis=0).reset_index(drop=True)

    selected = selected.reset_index(drop=True)
    print("Selected signatures:", len(selected))
    if len(selected) == 0:
        raise ValueError("No signatures left after filtering.")

    print("Loading GCTX...")
    f = load_gctx_from_gzip(gctx_path)

    mat = f["0"]["DATA"]["0"]["matrix"]  # (signatures, genes)
    col_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
               for x in f["0"]["META"]["COL"]["id"][:]]
    row_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
               for x in f["0"]["META"]["ROW"]["id"][:]]

    print("Matrix shape:", mat.shape)

    sig_to_idx = {sig_id: i for i, sig_id in enumerate(col_ids)}

    # align selected signatures with matrix
    selected = selected[selected["sig_id"].isin(sig_to_idx)].copy().reset_index(drop=True)
    selected["matrix_idx"] = selected["sig_id"].map(sig_to_idx)

    print("Matched signatures in matrix:", len(selected))
    if len(selected) == 0:
        raise ValueError("No selected signatures matched in matrix.")

    # landmark genes only
    gene = gene.copy()
    gene["pr_gene_id"] = gene["pr_gene_id"].astype(str)
    lm_gene = gene[gene["pr_is_lm"] == 1].copy()

    gid_to_symbol = dict(zip(gene["pr_gene_id"], gene["pr_gene_symbol"]))
    lm_gene_set = set(lm_gene["pr_gene_id"].tolist())

    lm_indices = []
    lm_symbols = []

    for j, gid in enumerate(row_ids):
        if gid in lm_gene_set:
            lm_indices.append(j)
            lm_symbols.append(gid_to_symbol.get(gid, gid))

    lm_indices = np.array(lm_indices, dtype=np.int64)

    print("Landmark genes kept:", len(lm_indices))
    print("Workers:", args.workers)

    selected_records = selected[[
        "cell_id", "pert_iname", "pert_idose", "pert_itime", "matrix_idx"
    ]].to_dict("records")

    out_jsonl = os.path.join(out_dir, "l1000_topk_minimal.jsonl")
    meta_csv = os.path.join(out_dir, "selected_signatures_minimal.csv")

    selected[["cell_id", "pert_iname", "pert_idose", "pert_itime"]].to_csv(meta_csv, index=False)

    total = len(selected_records)
    workers = min(args.workers, cpu_count())

    print("Writing:", out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as wf:
        with Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(mat, lm_symbols, lm_indices, selected_records, args.topk)
        ) as pool:
            for n, line in enumerate(pool.imap(process_one, range(total), chunksize=args.chunk_size), start=1):
                wf.write(line + "\n")
                if n % 1000 == 0 or n == total:
                    print(f"Wrote {n}/{total}")

    f.close()
    print("Done.")
    print("Saved JSONL:", out_jsonl)
    print("Saved meta  :", meta_csv)


if __name__ == "__main__":
    main()