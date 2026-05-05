#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/root/dengjie/AI4SCI/PP-data/GSE92742-SFT/data.jsonl", help="Path to raw data.jsonl")
    parser.add_argument("--outdir", type=str, default="/root/dengjie/AI4SCI/PP-data/GSE92742-TEST", help="Output directory for test data")
    parser.add_argument("--block-size", type=int, default=125)
    parser.add_argument("--min-common-genes", type=int, default=100)
    # 只要 split 字段里包含 test 这个词，我们就认为它是测试集（例如 test_seen, test_ood 等）
    parser.add_argument("--test-split-keyword", type=str, default="ood")
    return parser.parse_args()

# ==========================================
# 1. 核心清洗与分块逻辑 (严格对齐 SFT)
# ==========================================
def clean_sentence(sentence):
    if sentence is None: return []
    bad_tokens = {"...", "....", ".....", ".", ",", ";", ":", "[", "]", "(", ")"}
    genes, seen = [], set()
    for token in str(sentence).replace("\n", " ").split():
        token = token.strip()
        if not token or token in bad_tokens or set(token) == {"."}: continue
        if token not in seen:
            genes.append(token)
            seen.add(token)
    return genes

def safe_text(value):
    if value is None: return "N/A"
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "none", "null"} or value in {"-666", "-666.0"}:
        return "N/A"
    return value

def format_gene_list(genes):
    return " ".join(genes)

def format_blocks(block_dict):
    lines = []
    for block_id in sorted(block_dict.keys()):
        lines.append(f"Block {block_id}: {format_gene_list(block_dict[block_id])}")
    return "\n".join(lines)

def build_condition_text(record):
    return (
        f"Cell line: {safe_text(record.get('cell_id'))}\n"
        f"Perturbation type: {safe_text(record.get('perturbation_type'))}\n"
        f"Perturbation name: {safe_text(record.get('pert_name'))}\n"
        f"SMILES: {safe_text(record.get('canonical_smiles'))}\n"
        f"Treatment time: {safe_text(record.get('pert_itime'))}"
    )

def build_target_blocks(control_genes, perturb_genes, block_size):
    perturb_rank = {gene: idx for idx, gene in enumerate(perturb_genes)}
    common_genes = [gene for gene in control_genes if gene in perturb_rank]
    common_gene_set = set(common_genes)

    target_blocks_control_order = defaultdict(list)
    target_blocks_perturb_order = defaultdict(list)

    # 生成 Planner 标准答案 (按 control 顺序的块)
    for gene in common_genes:
        target_rank = perturb_rank[gene]
        target_block = target_rank // block_size + 1
        target_blocks_control_order[target_block].append(gene)

    # 生成 Reranker 标准答案 (按 perturb 顺序的块)
    for gene in perturb_genes:
        if gene not in common_gene_set: continue
        target_rank = perturb_rank[gene]
        target_block = target_rank // block_size + 1
        target_blocks_perturb_order[target_block].append(gene)

    return dict(target_blocks_control_order), dict(target_blocks_perturb_order), common_genes


# ==========================================
# 2. 纯测试 Prompt 构建 (只有 System + User，没有 Assistant)
# ==========================================
def build_planner_prompt(record, control_genes, block_size):
    system = (
        "You are an AI expert in transcriptomic perturbation analysis. "
        "Your task is to predict the complete target gene sets for consecutive expression blocks after a given perturbation. "
        f"Each block contains {block_size} genes. "
        "Keep the genes within each output block in the same relative order as they appear in the baseline control expression."
    )
    user = (
        "[Condition]\n"
        f"{build_condition_text(record)}\n\n"
        "[Control Expression]\n"
        f"{format_gene_list(control_genes)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def build_reranker_isolated_prompt(record, gt_planner_blocks, block_size):
    system = (
        "You are an AI expert in transcriptomic perturbation analysis. "
        "Your task is to rerank the genes within each provided expression block to reflect their final perturbed state. "
        f"Each block contains {block_size} genes. Reorder genes strictly within their own blocks without moving genes across blocks."
    )
    user = (
        "[Condition]\n"
        f"{build_condition_text(record)}\n\n"
        "[Input Blocks]\n"
        f"{format_blocks(gt_planner_blocks)}"  # 传入完美的 GT block
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ==========================================
# 3. 主流程
# ==========================================
def main():
    args = parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    output_path = outdir / "test_cases.jsonl"
    meta_path = outdir / "cook_test_meta.json"
    
    print("Cooking unified Test & Eval data...")
    print(f"Target split keyword: '{args.test_split_keyword}'")
    
    status_counter = Counter()
    split_counter = Counter()
    processed_count = 0
    
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip(): continue
            record = json.loads(line)
            
            split = record.get("split", "unknown")
            if args.test_split_keyword not in split:
                continue # 跳过非 test 的数据
                
            split_counter[split] += 1
            
            control_genes = clean_sentence(record.get("control_sentence", ""))
            perturb_genes = clean_sentence(record.get("perturb_sentence", ""))
            
            if not control_genes or not perturb_genes:
                status_counter["skip_empty"] += 1
                continue
                
            # 拿到三个维度的 GT 
            # 1. gt_planner: Planner 该输出的归类
            # 2. gt_reranker: Reranker 该输出的排序
            # 3. common: 全局存在的基因序列
            gt_planner_blocks, gt_reranker_blocks, common_genes = build_target_blocks(
                control_genes, perturb_genes, args.block_size
            )
            
            if len(common_genes) < args.min_common_genes:
                status_counter["skip_short"] += 1
                continue
            
            status_counter["ok"] += 1
            processed_count += 1
            
            # 组装丰富的 Test Record
            test_record = {
                # 基础信息，保留所有的细分 split
                "record_id": record.get("id", processed_count),
                "split": split, 
                "cell_id": record.get("cell_id"),
                "perturbation_type": record.get("perturbation_type"),
                
                # Ground Truth 数据字典，用于计算 Metrics
                "gt_sequence": perturb_genes, # 端到端(E2E)评估的核心 GT
                "gt_planner_blocks": gt_planner_blocks, # Planner 独立评估的核心 GT
                "gt_reranker_blocks": gt_reranker_blocks, # Reranker 独立评估的核心 GT
                
                # Ready-to-use 的 Prompt 列表 (交由 vLLM 推理)
                "planner_prompt": build_planner_prompt(record, control_genes, args.block_size),
                # 这是用于【隔离测试】的 Reranker Prompt（传入的是完美的 GT_Planner_blocks）
                "reranker_isolated_prompt": build_reranker_isolated_prompt(record, gt_planner_blocks, args.block_size),
                
                # 保留原始的 condition 参数，如果需要端到端动态组装 reranker_prompt 会用到
                "condition_metadata": {
                    "cell_id": safe_text(record.get("cell_id")),
                    "perturbation_type": safe_text(record.get("perturbation_type")),
                    "pert_name": safe_text(record.get("pert_name")),
                    "canonical_smiles": safe_text(record.get("canonical_smiles")),
                    "pert_itime": safe_text(record.get("pert_itime"))
                }
            }
            
            fout.write(json.dumps(test_record, ensure_ascii=False) + "\n")
            
    # 输出统计信息
    meta = {
        "input_file": str(args.input),
        "output_file": str(output_path),
        "block_size": args.block_size,
        "valid_test_samples": processed_count,
        "status_counter": dict(status_counter),
        "split_distribution": dict(split_counter)
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        
    print(f"\nDone! Cooked {processed_count} valid test samples.")
    print(f"Data saved to: {output_path}")
    print(f"Split details: {dict(split_counter)}")

if __name__ == "__main__":
    main()