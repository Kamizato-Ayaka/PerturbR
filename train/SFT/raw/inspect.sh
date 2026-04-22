#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-$HOME/dengjie/AI4SCI/PerturbR/train/SFT/raw}"
cd "$RAW_DIR"

echo "=================================================="
echo " L1000 RAW DIRECTORY INSPECTION"
echo "=================================================="
echo "PWD: $(pwd)"
echo

FILES=(
  "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz"
  "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
  "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz"
  "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz"
  "GSE70138_Broad_LINCS_pert_info.txt.gz"
  "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
)

echo "===== FILE EXISTENCE CHECK ====="
for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    ls -lh "$f"
  else
    echo "[MISSING] $f"
  fi
done
echo

echo "===== PREVIEW TEXT TABLES ====="
for f in \
  GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz \
  GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz \
  GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz \
  GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz \
  GSE70138_Broad_LINCS_pert_info.txt.gz
do
  echo
  echo "---------- $f ----------"
  zcat "$f" | head -n 5 || true
done
echo

python - <<'PY'
import gzip
import io
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
import h5py

RAW = os.getcwd()

sig_path  = os.path.join(RAW, "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz")
inst_path = os.path.join(RAW, "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz")
gene_path = os.path.join(RAW, "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz")
cell_path = os.path.join(RAW, "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz")
pert_path = os.path.join(RAW, "GSE70138_Broad_LINCS_pert_info.txt.gz")
gctx_path = os.path.join(RAW, "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz")

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 180)
random.seed(0)
np.random.seed(0)

def read_tsv_gz(path):
    return pd.read_csv(path, sep="\t", low_memory=False)

def decode_arr(arr, n=5):
    out = []
    for x in arr[:n]:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out

def maybe_decode_series(s):
    if len(s) == 0:
        return s
    if isinstance(s.iloc[0], bytes):
        return s.map(lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x)
    return s

print("=" * 60)
print("LOAD METADATA TABLES")
print("=" * 60)

sig = read_tsv_gz(sig_path)
inst = read_tsv_gz(inst_path)
gene = read_tsv_gz(gene_path)
cell = read_tsv_gz(cell_path)
pert = read_tsv_gz(pert_path)

tables = {
    "sig_info": sig,
    "inst_info": inst,
    "gene_info": gene,
    "cell_info": cell,
    "pert_info": pert,
}

for name, df in tables.items():
    print(f"\n----- {name} -----")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(3))

print("\n" + "=" * 60)
print("QUICK METADATA STATS")
print("=" * 60)

if "cell_id" in sig.columns:
    print("\n[sig_info] top cell_id:")
    print(sig["cell_id"].value_counts().head(15))

if "pert_iname" in sig.columns:
    print("\n[sig_info] top pert_iname:")
    print(sig["pert_iname"].value_counts().head(15))

if "pert_type" in sig.columns:
    print("\n[sig_info] pert_type distribution:")
    print(sig["pert_type"].value_counts(dropna=False).head(20))

for c in ["pert_time", "pert_time_unit", "pert_idose", "pert_itime"]:
    if c in sig.columns:
        print(f"\n[sig_info] sample values for {c}:")
        print(sig[c].astype(str).value_counts(dropna=False).head(20))

if "pr_gene_id" in gene.columns:
    print("\n[gene_info] pr_is_lm distribution if exists:")
    if "pr_is_lm" in gene.columns:
        print(gene["pr_is_lm"].value_counts(dropna=False))

if "pert_type" in pert.columns:
    print("\n[pert_info] pert_type distribution:")
    print(pert["pert_type"].value_counts(dropna=False).head(20))

print("\n" + "=" * 60)
print("OPEN LEVEL3 GCTX.GZ AS HDF5")
print("=" * 60)

# h5py generally cannot open gzip-compressed file directly by path if still compressed
# so we load bytes to memory first
with gzip.open(gctx_path, "rb") as f:
    gctx_bytes = f.read()

bio = io.BytesIO(gctx_bytes)

with h5py.File(bio, "r") as f:
    print("\nTop-level keys:")
    print(list(f.keys()))

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name} shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"GROUP:   {name}")

    print("\n===== HDF5 TREE (depth-limited manual view) =====")
    # Manual compact exploration
    for k1 in f.keys():
        obj1 = f[k1]
        print(f"/{k1} ->", type(obj1).__name__)
        if isinstance(obj1, h5py.Group):
            for k2 in obj1.keys():
                obj2 = obj1[k2]
                print(f"  /{k1}/{k2} ->", type(obj2).__name__)
                if isinstance(obj2, h5py.Group):
                    for k3 in obj2.keys():
                        obj3 = obj2[k3]
                        if isinstance(obj3, h5py.Dataset):
                            print(f"    /{k1}/{k2}/{k3} -> DATASET shape={obj3.shape} dtype={obj3.dtype}")
                        else:
                            print(f"    /{k1}/{k2}/{k3} -> GROUP")
                            if isinstance(obj3, h5py.Group):
                                for k4 in obj3.keys():
                                    obj4 = obj3[k4]
                                    if isinstance(obj4, h5py.Dataset):
                                        print(f"      /{k1}/{k2}/{k3}/{k4} -> DATASET shape={obj4.shape} dtype={obj4.dtype}")
                                    else:
                                        print(f"      /{k1}/{k2}/{k3}/{k4} -> GROUP")

    # Standard GCTX layout
    mat = f["0"]["DATA"]["0"]["matrix"]
    row_meta = f["0"]["META"]["ROW"]
    col_meta = f["0"]["META"]["COL"]

    print("\n===== MATRIX CORE INFO =====")
    print("matrix shape:", mat.shape)
    print("matrix dtype:", mat.dtype)

    row_keys = list(row_meta.keys())
    col_keys = list(col_meta.keys())
    print("ROW meta keys:", row_keys)
    print("COL meta keys:", col_keys)

    # Commonly ROW=id for genes and COL=id for signatures in standard GCTX
    row_ids = row_meta["id"][:]
    col_ids = col_meta["id"][:]

    row_ids_dec = decode_arr(row_ids, 10)
    col_ids_dec = decode_arr(col_ids, 10)

    print("\nFirst 10 row ids:")
    print(row_ids_dec)
    print("\nFirst 10 col ids:")
    print(col_ids_dec)

    # Infer orientation
    sig_id_set = set(sig["sig_id"].astype(str)) if "sig_id" in sig.columns else set()
    gene_id_candidates = []
    for gcol in ["pr_gene_id", "gene_id", "id", "pr_id"]:
        if gcol in gene.columns:
            gene_id_candidates = set(gene[gcol].astype(str))
            gene_id_col = gcol
            break
    else:
        gene_id_set = set()
        gene_id_col = None

    gene_id_set = set(gene[gene_id_col].astype(str)) if gene_id_col else set()

    row_sig_match = sum(1 for x in row_ids_dec[:200] if x in sig_id_set)
    col_sig_match = sum(1 for x in col_ids_dec[:200] if x in sig_id_set)
    row_gene_match = sum(1 for x in row_ids_dec[:200] if x in gene_id_set)
    col_gene_match = sum(1 for x in col_ids_dec[:200] if x in gene_id_set)

    print("\n===== ORIENTATION GUESS =====")
    print("row_sig_match (first 200):", row_sig_match)
    print("col_sig_match (first 200):", col_sig_match)
    print("row_gene_match (first 200):", row_gene_match)
    print("col_gene_match (first 200):", col_gene_match)

    if col_sig_match > row_sig_match and row_gene_match > col_gene_match:
        orientation = "rows=genes, cols=signatures"
    elif row_sig_match > col_sig_match and col_gene_match > row_gene_match:
        orientation = "rows=signatures, cols=genes"
    else:
        orientation = "unclear from first-pass match"
    print("guessed orientation:", orientation)

    print("\n===== MATRIX VALUE SANITY CHECK =====")
    # small block
    r_take = min(3, mat.shape[0])
    c_take = min(5, mat.shape[1])
    block = mat[:r_take, :c_take]
    print("top-left block:")
    print(block)

    # sample values from a few rows / cols
    all_vals = []
    sample_rows = np.linspace(0, mat.shape[0]-1, num=min(10, mat.shape[0]), dtype=int)
    for r in sample_rows:
        vals = mat[r, :min(100, mat.shape[1])]
        all_vals.append(np.asarray(vals).reshape(-1))
    all_vals = np.concatenate(all_vals)
    print("sampled value stats:")
    print("  min:", float(np.nanmin(all_vals)))
    print("  max:", float(np.nanmax(all_vals)))
    print("  mean:", float(np.nanmean(all_vals)))
    print("  std:", float(np.nanstd(all_vals)))
    print("  nan_count:", int(np.isnan(all_vals).sum()))

    print("\n===== ALIGNMENT WITH sig_info / gene_info =====")
    # full ID decode may be large but okay
    row_ids_full = [x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x) for x in row_ids]
    col_ids_full = [x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x) for x in col_ids]

    row_sig_overlap = sum(1 for x in row_ids_full if x in sig_id_set)
    col_sig_overlap = sum(1 for x in col_ids_full if x in sig_id_set)
    row_gene_overlap = sum(1 for x in row_ids_full if x in gene_id_set)
    col_gene_overlap = sum(1 for x in col_ids_full if x in gene_id_set)

    print("row_sig_overlap:", row_sig_overlap)
    print("col_sig_overlap:", col_sig_overlap)
    print("row_gene_overlap:", row_gene_overlap)
    print("col_gene_overlap:", col_gene_overlap)

    # Build sample inspection table
    print("\n===== RANDOM SAMPLE METADATA INSPECTION =====")
    if col_sig_overlap > row_sig_overlap:
        sig_axis = "col"
        sig_ids_for_sampling = col_ids_full
        gene_axis_ids = row_ids_full
    else:
        sig_axis = "row"
        sig_ids_for_sampling = row_ids_full
        gene_axis_ids = col_ids_full

    sample_sig_ids = random.sample(sig_ids_for_sampling, k=min(5, len(sig_ids_for_sampling)))
    print("sample_sig_ids:", sample_sig_ids)

    if "sig_id" in sig.columns:
        sample_meta = sig[sig["sig_id"].astype(str).isin(sample_sig_ids)]
        keep_cols = [c for c in [
            "sig_id", "pert_id", "pert_iname", "pert_type",
            "cell_id", "pert_time", "pert_time_unit",
            "pert_idose", "pert_itime", "tas", "is_gold"
        ] if c in sample_meta.columns]
        print(sample_meta[keep_cols].head(10))

    print("\n===== GENE AXIS INSPECTION =====")
    print("first 10 gene-axis ids:", gene_axis_ids[:10])

    if gene_id_col:
        gene_subset = gene[gene[gene_id_col].astype(str).isin(gene_axis_ids[:10])]
        print("matched gene_info rows for first few gene ids:")
        print(gene_subset.head(10))

print("\n" + "=" * 60)
print("JOIN SANITY CHECK OUTSIDE GCTX")
print("=" * 60)

# Common useful overlap checks
if "sig_id" in sig.columns and "distil_id" in sig.columns and "distil_id" in inst.columns:
    print("\n[sanity] sig_info.distil_id / inst_info.distil_id both exist")
    sig_nonnull = sig["distil_id"].notna().sum()
    inst_nonnull = inst["distil_id"].notna().sum()
    print("sig_info non-null distil_id:", sig_nonnull)
    print("inst_info non-null distil_id:", inst_nonnull)

for key in ["cell_id", "pert_id"]:
    if key in sig.columns:
        print(f"\n[sig_info] unique {key}: {sig[key].nunique()}")

if "cell_id" in cell.columns:
    print("\n[cell_info] unique cell_id:", cell["cell_id"].nunique())

if "pert_id" in pert.columns:
    print("\n[pert_info] unique pert_id:", pert["pert_id"].nunique())

print("\nDONE.")
PY