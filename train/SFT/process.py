import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def read_tsv_auto(path: Path) -> pd.DataFrame:
    if str(path).endswith(".gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def jsonl_dump(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_missing(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x.lower() in {"-666", "nan", "none", "null"}:
            return None
    if isinstance(x, (int, float)) and x == -666:
        return None
    return x


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_time_to_float(x) -> Optional[float]:
    x = normalize_missing(x)
    if x is None:
        return None
    s = str(x).lower().strip()
    for token in ["hours", "hour", "hrs", "hr", "h"]:
        s = s.replace(token, "")
    s = s.strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_dose_to_float(x) -> Optional[float]:
    x = normalize_missing(x)
    if x is None:
        return None
    s = str(x).lower().strip()
    for token in ["µm", "um", "nm", "mm", "m"]:
        s = s.replace(token, "")
    s = s.strip()
    try:
        return float(s)
    except Exception:
        return None


def build_lookups(
    pert_info: pd.DataFrame,
    cell_info: pd.DataFrame,
    gene_info: pd.DataFrame,
) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    pert_id_col = first_existing_col(pert_info, ["pert_id", "id"])
    pert_name_col = first_existing_col(pert_info, ["pert_iname", "pert_name"])
    pert_type_col = first_existing_col(pert_info, ["pert_type"])

    pert_lookup = {}
    if pert_id_col is not None:
        for _, row in pert_info.iterrows():
            pert_id = str(row[pert_id_col])
            pert_lookup[pert_id] = {
                "perturbation": normalize_missing(row.get(pert_name_col)) if pert_name_col else None,
                "pert_type_meta": normalize_missing(row.get(pert_type_col)) if pert_type_col else None,
            }

    cell_id_col = first_existing_col(cell_info, ["cell_id", "id"])
    cell_name_col = first_existing_col(cell_info, ["cell_iname", "cell_name"])

    cell_lookup = {}
    if cell_id_col is not None:
        for _, row in cell_info.iterrows():
            cell_id = str(row[cell_id_col])
            cell_lookup[cell_id] = {
                "cell_name": normalize_missing(row.get(cell_name_col)) if cell_name_col else None
            }

    gene_id_col = first_existing_col(gene_info, ["pr_gene_id", "gene_id", "id"])
    gene_symbol_col = first_existing_col(gene_info, ["pr_gene_symbol", "gene_symbol"])
    gene_prid_col = first_existing_col(gene_info, ["pr_id"])

    gene_lookup = {}
    if gene_id_col is not None:
        for _, row in gene_info.iterrows():
            gid = str(row[gene_id_col])
            gene_lookup[gid] = {
                "gene_symbol": normalize_missing(row.get(gene_symbol_col)) if gene_symbol_col else None,
                "pr_id": normalize_missing(row.get(gene_prid_col)) if gene_prid_col else None,
            }

    return pert_lookup, cell_lookup, gene_lookup


def filter_inst_info(
    inst_info: pd.DataFrame,
    target_cells: Optional[Set[str]],
    target_pert_type: Optional[str],
    target_time: Optional[float],
    max_per_cell: Optional[int],
    random_seed: int,
) -> pd.DataFrame:
    df = inst_info.copy()

    inst_id_col = first_existing_col(df, ["inst_id", "id"])
    if inst_id_col != "inst_id":
        df = df.rename(columns={inst_id_col: "inst_id"})

    cell_col = first_existing_col(df, ["cell_id", "cell_iname"])
    pert_type_col = first_existing_col(df, ["pert_type"])
    pert_id_col = first_existing_col(df, ["pert_id"])
    time_col = first_existing_col(df, ["pert_itime", "pert_time"])
    dose_col = first_existing_col(df, ["pert_idose", "pert_dose"])

    if cell_col and cell_col != "cell_id":
        df = df.rename(columns={cell_col: "cell_id"})
    if pert_type_col and pert_type_col != "pert_type":
        df = df.rename(columns={pert_type_col: "pert_type"})
    if pert_id_col and pert_id_col != "pert_id":
        df = df.rename(columns={pert_id_col: "pert_id"})
    if time_col and time_col != "pert_itime":
        df = df.rename(columns={time_col: "pert_itime"})
    if dose_col and dose_col != "pert_idose":
        df = df.rename(columns={dose_col: "pert_idose"})

    if target_pert_type is not None and "pert_type" in df.columns:
        df = df[df["pert_type"] == target_pert_type]

    if target_cells is not None and "cell_id" in df.columns:
        df = df[df["cell_id"].astype(str).isin(target_cells)]

    if target_time is not None and "pert_itime" in df.columns:
        parsed_time = df["pert_itime"].map(parse_time_to_float)
        df = df[parsed_time == float(target_time)]

    df = df.drop_duplicates(subset=["inst_id"]).reset_index(drop=True)

    if max_per_cell is not None and "cell_id" in df.columns:
        random.seed(random_seed)
        parts = []
        for _, group in df.groupby("cell_id", dropna=False):
            if len(group) > max_per_cell:
                idx = list(group.index)
                random.shuffle(idx)
                idx = idx[:max_per_cell]
                parts.append(group.loc[idx])
            else:
                parts.append(group)
        df = pd.concat(parts, axis=0).reset_index(drop=True)

    return df


def build_gct_col_to_inst_id_map(gct_path: Path) -> Tuple[Dict[str, str], int, int, str]:
    """
    智能解析 GCT 的前几行，提取 det_plate 和 det_well，组装成真实的 inst_id
    返回: (列名到inst_id的映射字典, column_meta的行数, row_meta的列数, 首列的列名)
    """
    opener = gzip.open if str(gct_path).endswith(".gz") else open
    with opener(gct_path, "rt", encoding="utf-8", errors="ignore") as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()
        header = f.readline().rstrip("\n").split("\t")

        parts = line2.split()
        num_row_meta = int(parts[2]) if len(parts) >= 3 else 11
        num_col_meta = int(parts[3]) if len(parts) >= 4 else 12

        plates = None
        wells = None

        # 逐行读取 Column Metadata，寻找 det_plate 和 det_well
        for _ in range(num_col_meta):
            line = f.readline().rstrip("\n").split("\t")
            row_name = line[0].strip('"').strip()
            if row_name == "det_plate":
                plates = line
            elif row_name == "det_well":
                wells = line

    if plates is None or wells is None:
        raise ValueError("在 GCT 文件的注释中找不到 det_plate 或 det_well，无法进行 ID 映射！")

    col_map = {}
    # 数据起始索引（跳过行注释）
    start_idx = 1 + num_row_meta

    for i in range(start_idx, len(header)):
        col_name = header[i].strip('"').strip()
        
        # 防止部分文件尾部缺失导致越界
        p = plates[i].strip('"').strip() if i < len(plates) else ""
        w = wells[i].strip('"').strip() if i < len(wells) else ""
        
        if p and w:
            inst_id = f"{p}:{w}"  # 拼凑成真正的 inst_id！
            col_map[col_name] = inst_id

    first_col_name = header[0].strip('"').strip()
    return col_map, num_col_meta, num_row_meta, first_col_name


def load_gene_row_mapping_from_standard_gct(
    gct_path: Path,
    gene_lookup: Dict[str, dict],
    target_gct_cols: List[str],
    gct_col_to_inst_id: Dict[str, str],
    num_col_meta: int,
    first_col_name: str
) -> Tuple[List[str], pd.DataFrame]:
    """
    通过跳过无关的 Metadata 行，直接抽取我们需要的样本列数据
    """
    # 我们只保留第一列(基因ID)和匹配上的样本列
    keep_cols = [first_col_name] + target_gct_cols
    
    # skiprows 逻辑: 跳过 第0, 1行，保留第2行做表头，再跳过所有的列注释行
    skiprows = [0, 1] + list(range(3, 3 + num_col_meta))
    
    df = pd.read_csv(
        gct_path,
        sep="\t",
        compression="gzip" if str(gct_path).endswith(".gz") else None,
        skiprows=skiprows,
        usecols=keep_cols,
        low_memory=False,
    )
    
    # 清理列名以防万一
    df.columns = [str(c).strip('"').strip() for c in df.columns]
    
    # 将 GCT 的内部 ID 重命名为标准的 inst_id，方便下游处理
    rename_map = {first_col_name: "id"}
    for gct_col in target_gct_cols:
        if gct_col in df.columns:
            rename_map[gct_col] = gct_col_to_inst_id[gct_col]
            
    df = df.rename(columns=rename_map)
    
    # 添加基因 Symbol 注释
    gene_names = []
    for _, row in df.iterrows():
        gid = str(row["id"])
        symbol = gene_lookup.get(gid, {}).get("gene_symbol")
        gene_names.append(symbol if symbol is not None else gid)

    df.insert(1, "__gene_symbol__", gene_names)
    
    inst_cols_in_file = [gct_col_to_inst_id[c] for c in target_gct_cols]
    return inst_cols_in_file, df


def build_instance_records_from_standard_gct(
    matrix_df: pd.DataFrame,
    selected_meta_df: pd.DataFrame,
    pert_lookup: Dict[str, dict],
    cell_lookup: Dict[str, dict],
) -> List[dict]:
    meta_by_inst = {}
    for _, row in selected_meta_df.iterrows():
        inst_id = str(row["inst_id"])
        pert_id = str(row["pert_id"]) if normalize_missing(row.get("pert_id")) is not None else None
        cell_id = str(row["cell_id"]) if normalize_missing(row.get("cell_id")) is not None else None

        meta_by_inst[inst_id] = {
            "inst_id": inst_id,
            "cell": cell_id,
            "cell_name": cell_lookup.get(cell_id, {}).get("cell_name"),
            "pert_type": normalize_missing(row.get("pert_type")),
            "pert_id": pert_id,
            "perturbation": pert_lookup.get(pert_id, {}).get("perturbation"),
            "dose": parse_dose_to_float(row.get("pert_idose")),
            "time": parse_time_to_float(row.get("pert_itime")),
        }

    inst_cols = [c for c in matrix_df.columns if c not in {"id", "__gene_symbol__"}]
    gene_symbols = matrix_df["__gene_symbol__"].tolist()

    records = []
    for inst_id in inst_cols:
        if inst_id not in meta_by_inst:
            continue
        expr_values = matrix_df[inst_id].tolist()

        expression = {}
        for gene, val in zip(gene_symbols, expr_values):
            if gene is None:
                continue
            if pd.isna(val):
                continue
            expression[str(gene)] = float(val)

        rec = {
            **meta_by_inst[inst_id],
            "expression": expression,
        }
        records.append(rec)
    return records


def build_selected_meta_records(
    selected_meta_df: pd.DataFrame,
    pert_lookup: Dict[str, dict],
    cell_lookup: Dict[str, dict],
) -> List[dict]:
    out = []
    for _, row in selected_meta_df.iterrows():
        pert_id = str(row["pert_id"]) if normalize_missing(row.get("pert_id")) is not None else None
        cell_id = str(row["cell_id"]) if normalize_missing(row.get("cell_id")) is not None else None
        out.append({
            "inst_id": str(row["inst_id"]),
            "cell": cell_id,
            "cell_name": cell_lookup.get(cell_id, {}).get("cell_name"),
            "pert_type": normalize_missing(row.get("pert_type")),
            "pert_id": pert_id,
            "perturbation": pert_lookup.get(pert_id, {}).get("perturbation"),
            "dose": parse_dose_to_float(row.get("pert_idose")),
            "time": parse_time_to_float(row.get("pert_itime")),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cells", type=str, default="A375,MCF7,PC3")
    parser.add_argument("--pert_type", type=str, default="trt_cp")
    parser.add_argument("--time_h", type=float, default=24.0)
    parser.add_argument("--max_per_cell", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=41)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gct_path = raw_dir / "GSE70138_Broad_LINCS_Level2_GEX_n113012x978_2015-12-31.gct.gz"
    inst_info_path = raw_dir / "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz"
    gene_info_path = raw_dir / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
    cell_info_path = raw_dir / "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz"
    pert_info_path = raw_dir / "GSE70138_Broad_LINCS_pert_info.txt.gz"

    print("Loading metadata...")
    inst_info = read_tsv_auto(inst_info_path)
    gene_info = read_tsv_auto(gene_info_path)
    cell_info = read_tsv_auto(cell_info_path)
    pert_info = read_tsv_auto(pert_info_path)

    pert_lookup, cell_lookup, gene_lookup = build_lookups(
        pert_info=pert_info,
        cell_info=cell_info,
        gene_info=gene_info,
    )

    target_cells = {x.strip() for x in args.cells.split(",") if x.strip()}
    selected_meta_df = filter_inst_info(
        inst_info=inst_info,
        target_cells=target_cells if len(target_cells) > 0 else None,
        target_pert_type=args.pert_type if args.pert_type else None,
        target_time=args.time_h,
        max_per_cell=args.max_per_cell if args.max_per_cell > 0 else None,
        random_seed=args.random_seed,
    )

    print(f"Selected instances: {len(selected_meta_df)}")
    if len(selected_meta_df) == 0:
        print("No instances matched the current filters.")
        return

    selected_meta_records = build_selected_meta_records(
        selected_meta_df=selected_meta_df,
        pert_lookup=pert_lookup,
        cell_lookup=cell_lookup,
    )
    jsonl_dump(selected_meta_records, out_dir / "selected_inst_meta.jsonl")

    selected_inst_ids = set(selected_meta_df["inst_id"].astype(str).tolist())

    print("Mapping GCT column names to standard inst_ids...")
    col_map, num_col_meta, num_row_meta, first_col_name = build_gct_col_to_inst_id_map(gct_path)

    # 寻找匹配的 GCT 列名
    target_gct_cols = []
    gct_col_to_inst_id = {}
    
    for gct_col, mapped_inst_id in col_map.items():
        if mapped_inst_id in selected_inst_ids:
            target_gct_cols.append(gct_col)
            gct_col_to_inst_id[gct_col] = mapped_inst_id
            
    print(f"Found {len(target_gct_cols)} matching sample columns in GCT.")
    
    if not target_gct_cols:
        raise ValueError("依然找不到重合的数据。请检查 metadata 筛选逻辑。")

    print("Loading selected expression matrix from GCT...")
    _, matrix_df = load_gene_row_mapping_from_standard_gct(
        gct_path=gct_path,
        gene_lookup=gene_lookup,
        target_gct_cols=target_gct_cols,
        gct_col_to_inst_id=gct_col_to_inst_id,
        num_col_meta=num_col_meta,
        first_col_name=first_col_name
    )

    print("Building instance jsonl...")
    instance_records = build_instance_records_from_standard_gct(
        matrix_df=matrix_df,
        selected_meta_df=selected_meta_df,
        pert_lookup=pert_lookup,
        cell_lookup=cell_lookup,
    )
    jsonl_dump(instance_records, out_dir / "instances_level2.jsonl")

    print(f"Wrote: {out_dir / 'selected_inst_meta.jsonl'}")
    print(f"Wrote: {out_dir / 'instances_level2.jsonl'}")
    print(f"Final instance records: {len(instance_records)}")


if __name__ == "__main__":
    main()