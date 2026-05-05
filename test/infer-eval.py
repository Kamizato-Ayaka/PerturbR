#!/usr/bin/env python3
import os
import json
import argparse
import difflib
import gc
import torch
from pathlib import Path

# vLLM imports
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==========================================
# 1. 动态 Prompt 构建工具 (用于 E2E 阶段)
# ==========================================
def format_gene_list(genes):
    return " ".join(genes)

def format_blocks(block_dict):
    lines = []
    # 强制将 block_id 转为整型排序，确保 Prompt 里的顺序严格递增
    sorted_keys = sorted(block_dict.keys(), key=lambda x: int(x))
    for block_id in sorted_keys:
        lines.append(f"Block {block_id}: {format_gene_list(block_dict[block_id])}")
    return "\n".join(lines)

def build_e2e_reranker_prompt(condition_meta, pred_planner_blocks, block_size):
    """基于 Planner 的预测结果，动态组装 Reranker 的输入"""
    condition_text = (
        f"Cell line: {condition_meta.get('cell_id')}\n"
        f"Perturbation type: {condition_meta.get('perturbation_type')}\n"
        f"Perturbation name: {condition_meta.get('pert_name')}\n"
        f"SMILES: {condition_meta.get('canonical_smiles')}\n"
        f"Treatment time: {condition_meta.get('pert_itime')}"
    )
    
    system = (
        "You are an AI expert in transcriptomic perturbation analysis. "
        "Your task is to rerank the genes within each provided expression block to reflect their final perturbed state. "
        f"Each block contains {block_size} genes. Reorder genes strictly within their own blocks without moving genes across blocks."
    )
    user = (
        "[Condition]\n"
        f"{condition_text}\n\n"
        "[Input Blocks]\n"
        f"{format_blocks(pred_planner_blocks)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ==========================================
# 2. 结果解析工具 (Robust Parsing)
# ==========================================
def parse_generated_blocks(text):
    """
    鲁棒解析: 从 LLM 混杂的输出中精准提取 Block X: geneA geneB
    不管 LLM 开头结尾说了什么废话，只要包含 Block 格式就能抽出来。
    """
    blocks = {}
    for line in text.split('\n'):
        line = line.strip()
        # 兼容性匹配 "Block 1: " 或者 "Block 1 " 等等
        if line.lower().startswith("block"):
            try:
                # 切分出前缀和基因内容
                parts = line.split(":", 1)
                if len(parts) < 2:
                    continue # 格式彻底崩坏则跳过
                
                block_id_str = parts[0].replace("Block", "").replace("block", "").strip()
                block_id = int(block_id_str)
                genes = parts[1].strip().split()
                if genes:
                    blocks[str(block_id)] = genes # 统一用 string 作为字典的 key 方便对比
            except Exception:
                continue
    return blocks

def flatten_blocks(blocks_dict):
    """将字典按照 Block ID 递增还原为一维序列"""
    flat_list = []
    sorted_keys = sorted(blocks_dict.keys(), key=lambda x: int(x))
    for block_id in sorted_keys:
        flat_list.extend(blocks_dict[block_id])
    return flat_list

# ==========================================
# 3. 核心评估指标
# ==========================================
def calc_jaccard(pred_blocks, gt_blocks):
    """Planner 指标: 集合交并比"""
    if not pred_blocks or not gt_blocks: return 0.0
    
    scores = []
    all_keys = set(pred_blocks.keys()).union(set(gt_blocks.keys()))
    
    for k in all_keys:
        p_set = set(pred_blocks.get(str(k), []))
        g_set = set(gt_blocks.get(str(k), []))
        if not p_set and not g_set: continue
        
        intersection = len(p_set.intersection(g_set))
        union = len(p_set.union(g_set))
        scores.append(intersection / union if union > 0 else 0)
        
    return sum(scores) / len(scores) if scores else 0.0

def calc_sequence_ratio(pred_list, gt_list):
    """Reranker 指标: 最长公共子序列匹配率 (LCS-based)"""
    if not pred_list or not gt_list: return 0.0
    matcher = difflib.SequenceMatcher(None, pred_list, gt_list)
    return matcher.ratio()

def calc_reranker_isolated_score(pred_blocks, gt_blocks):
    """Reranker 独立指标: 计算每一个 Block 内部的平均匹配率"""
    if not pred_blocks or not gt_blocks: return 0.0
    scores = []
    # 隔离测试中，因为输入是 GT Planner blocks，所以 GT 和 Pred 理论上 Block 数量完全一致
    for k in gt_blocks.keys():
        p_list = pred_blocks.get(str(k), [])
        g_list = gt_blocks.get(str(k), [])
        scores.append(calc_sequence_ratio(p_list, g_list))
    return sum(scores) / len(scores) if scores else 0.0

# ==========================================
# 4. vLLM 推理封装
# ==========================================
def run_vllm_inference(model_path, prompts, tokenizer, temperature=0.0):
    """封装 vLLM 推理逻辑，方便内存释放"""
    print(f"Loading Model: {model_path} ...")
    # tensor_parallel_size 根据你的 GPU 数量调整
    llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=1) 
    sampling_params = SamplingParams(temperature=temperature, max_tokens=2048)
    
    # 格式化 Prompt
    formatted_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) 
        for p in prompts
    ]
    
    outputs = llm.generate(formatted_prompts, sampling_params)
    results = [out.outputs[0].text for out in outputs]
    
    # 彻底释放显存，关键步骤！
    print("Freeing vLLM memory...")
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

# ==========================================
# 5. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, required=True, help="Path to test_cases.jsonl")
    parser.add_argument("--planner-model", type=str, required=True, help="Path to Planner model")
    parser.add_argument("--reranker-model", type=str, default="", help="Path to Reranker model (Leave empty if same as planner)")
    parser.add_argument("--block-size", type=int, default=125)
    parser.add_argument("--output", type=str, default="detailed_eval_results.json")
    args = parser.parse_args()

    # 1. 如果没有指定特定的 reranker 模型，默认使用 planner 模型
    reranker_model_path = args.reranker_model if args.reranker_model else args.planner_model
    is_shared_model = (args.planner_model == reranker_model_path)

    print(f"Loading Test Data from {args.test_data}...")
    records = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"Loaded {len(records)} test samples.")

    # 提取 Planner 和 Reranker(Isolated) 的 Prompts
    planner_prompts = [r["planner_prompt"] for r in records]
    reranker_iso_prompts = [r["reranker_isolated_prompt"] for r in records]
    
    tokenizer = AutoTokenizer.from_pretrained(args.planner_model)

    # ---------------------------------------------------------
    # 推理阶段 (按需加载模型)
    # ---------------------------------------------------------
    if is_shared_model:
        print(">>> Using ONE shared model for all tasks. Optimizing load...")
        # 为了不反复清理显存，这里手写一个特殊的联合推理，把三种 prompt 拼在一起跑
        llm = LLM(model=args.planner_model, trust_remote_code=True, tensor_parallel_size=1)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
        
        # 跑 Planner
        print("[Shared] Running Planner Inference...")
        p_texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in planner_prompts]
        planner_raw_outs = llm.generate(p_texts, sampling_params)
        planner_texts = [out.outputs[0].text for out in planner_raw_outs]
        
        # 跑 Reranker Isolated
        print("[Shared] Running Reranker Isolated Inference...")
        ri_texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in reranker_iso_prompts]
        reranker_iso_raw_outs = llm.generate(ri_texts, sampling_params)
        reranker_iso_texts = [out.outputs[0].text for out in reranker_iso_raw_outs]
        
        # 动态组装 E2E Prompt 并跑
        print("[Shared] Running End-to-End Inference...")
        reranker_e2e_prompts = []
        for i, text in enumerate(planner_texts):
            pred_blocks = parse_generated_blocks(text)
            e2e_prompt = build_e2e_reranker_prompt(records[i]["condition_metadata"], pred_blocks, args.block_size)
            reranker_e2e_prompts.append(e2e_prompt)
            
        re_texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in reranker_e2e_prompts]
        reranker_e2e_raw_outs = llm.generate(re_texts, sampling_params)
        reranker_e2e_texts = [out.outputs[0].text for out in reranker_e2e_raw_outs]
        
        # 清理
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm; gc.collect(); torch.cuda.empty_cache()

    else:
        print(">>> Using TWO separate models. Running in sequential memory-safe mode...")
        # 跑 Planner
        print("=== STAGE 1: Planner ===")
        planner_texts = run_vllm_inference(args.planner_model, planner_prompts, tokenizer)
        
        # 跑 Reranker Isolated (用 Reranker 模型)
        print("=== STAGE 2: Reranker Isolated ===")
        reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
        reranker_iso_texts = run_vllm_inference(reranker_model_path, reranker_iso_prompts, reranker_tokenizer)
        
        # 组装 E2E 并跑 (用 Reranker 模型)
        print("=== STAGE 3: End-to-End ===")
        reranker_e2e_prompts = []
        for i, text in enumerate(planner_texts):
            pred_blocks = parse_generated_blocks(text)
            e2e_prompt = build_e2e_reranker_prompt(records[i]["condition_metadata"], pred_blocks, args.block_size)
            reranker_e2e_prompts.append(e2e_prompt)
        reranker_e2e_texts = run_vllm_inference(reranker_model_path, reranker_e2e_prompts, reranker_tokenizer)

    # ---------------------------------------------------------
    # 评估与记录阶段 (Metrics & Logging)
    # ---------------------------------------------------------
    print("Calculating Metrics...")
    
    detailed_results = []
    
    # 宏观统计累加器
    metrics_sum = {"planner_jaccard": 0.0, "reranker_iso_score": 0.0, "e2e_ratio": 0.0}
    split_stats = {} # 按照 split (如 test_seen, test_ood) 进行分组统计
    
    for i in range(len(records)):
        record = records[i]
        split_name = record["split"]
        if split_name not in split_stats:
            split_stats[split_name] = {"count": 0, "planner_jaccard": 0.0, "reranker_iso_score": 0.0, "e2e_ratio": 0.0}
            
        # --- 解析预测结果 ---
        pred_planner = parse_generated_blocks(planner_texts[i])
        pred_reranker_iso = parse_generated_blocks(reranker_iso_texts[i])
        pred_reranker_e2e = parse_generated_blocks(reranker_e2e_texts[i])
        
        # --- 读取 GT ---
        gt_planner = record["gt_planner_blocks"]
        gt_reranker = record["gt_reranker_blocks"]
        gt_flat_seq = record["gt_sequence"]
        
        # --- 计算单项指标 ---
        jaccard = calc_jaccard(pred_planner, gt_planner)
        iso_score = calc_reranker_isolated_score(pred_reranker_iso, gt_reranker)
        
        pred_flat_seq = flatten_blocks(pred_reranker_e2e)
        e2e_ratio = calc_sequence_ratio(pred_flat_seq, gt_flat_seq)
        
        # 累加与记录
        metrics_sum["planner_jaccard"] += jaccard
        metrics_sum["reranker_iso_score"] += iso_score
        metrics_sum["e2e_ratio"] += e2e_ratio
        
        split_stats[split_name]["count"] += 1
        split_stats[split_name]["planner_jaccard"] += jaccard
        split_stats[split_name]["reranker_iso_score"] += iso_score
        split_stats[split_name]["e2e_ratio"] += e2e_ratio
        
        detailed_results.append({
            "record_id": record["record_id"],
            "split": split_name,
            "metrics": {
                "planner_jaccard": round(jaccard, 4),
                "reranker_isolated_score": round(iso_score, 4),
                "e2e_sequence_ratio": round(e2e_ratio, 4)
            },
            # 如果想做极度细致的 Bad Case 分析，可以把下面这几行解开注释
            # "outputs": {
            #     "pred_planner": pred_planner,
            #     "pred_reranker_isolated": pred_reranker_iso,
            #     "pred_e2e_flat_sequence": pred_flat_seq
            # }
        })

    # 计算均值
    n = len(records)
    avg_jaccard = metrics_sum["planner_jaccard"] / n if n else 0
    avg_iso = metrics_sum["reranker_iso_score"] / n if n else 0
    avg_e2e = metrics_sum["e2e_ratio"] / n if n else 0
    
    # 打印优美的终端报告
    print("\n" + "="*50)
    print("🚀 EVALUATION RESULTS (GLOBAL)")
    print("="*50)
    print(f"Total Samples Tested     : {n}")
    print(f"1. Planner Jaccard       : {avg_jaccard:.4f}  (分块归类能力)")
    print(f"2. Reranker Isolated     : {avg_iso:.4f}  (完美分块下的块内排序能力)")
    print(f"3. End-to-End Match Ratio: {avg_e2e:.4f}  (真实全量场景还原能力)")
    
    print("\n" + "-"*50)
    print("📊 SUB-SPLIT ANALYSIS")
    print("-"*50)
    for split_name, stats in split_stats.items():
        c = stats["count"]
        print(f"[{split_name}] (n={c})")
        print(f"   Planner: {stats['planner_jaccard']/c:.4f} | IsoRerank: {stats['reranker_iso_score']/c:.4f} | E2E: {stats['e2e_ratio']/c:.4f}")

    # 保存最终结果
    final_report = {
        "global_summary": {
            "total_samples": n,
            "planner_jaccard": avg_jaccard,
            "reranker_isolated_score": avg_iso,
            "e2e_ratio": avg_e2e
        },
        "split_summary": {
            k: {
                "count": v["count"],
                "planner_jaccard": v["planner_jaccard"] / v["count"],
                "reranker_isolated_score": v["reranker_iso_score"] / v["count"],
                "e2e_ratio": v["e2e_ratio"] / v["count"]
            } for k, v in split_stats.items()
        },
        "details": detailed_results
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
        
    print(f"\nDetailed report saved to: {args.output}")

if __name__ == "__main__":
    main()