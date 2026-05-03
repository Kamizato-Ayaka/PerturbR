#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default="../data.jsonl")
    parser.add_argument("--outdir", type=str, default="../SFT")
    parser.add_argument("--split", type=str, default="sft")
    parser.add_argument("--block-size", type=int, default=125)
    parser.add_argument("--min-common-genes", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 1))
    parser.add_argument("--chunk-size", type=int, default=2000)

    return parser.parse_args()


def clean_sentence(sentence):
    if sentence is None:
        return []

    bad_tokens = {
        "...", "....", ".....",
        ".", ",", ";", ":",
        "[", "]", "(", ")"
    }

    genes = []
    seen = set()

    for token in str(sentence).replace("\n", " ").split():
        token = token.strip()

        if not token:
            continue
        if token in bad_tokens:
            continue
        if set(token) == {"."}:
            continue

        if token not in seen:
            genes.append(token)
            seen.add(token)

    return genes


def safe_text(value):
    if value is None:
        return "N/A"

    value = str(value).strip()

    if value == "":
        return "N/A"
    if value.lower() in {"nan", "none", "null"}:
        return "N/A"
    if value in {"-666", "-666.0"}:
        return "N/A"

    return value


def format_gene_list(genes):
    return " ".join(genes)


def format_blocks(block_dict):
    lines = []
    for block_id in sorted(block_dict.keys()):
        genes = block_dict[block_id]
        lines.append(f"Block {block_id}: {format_gene_list(genes)}")
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
    perturb_rank = {}

    for idx, gene in enumerate(perturb_genes):
        if gene not in perturb_rank:
            perturb_rank[gene] = idx

    common_genes = [gene for gene in control_genes if gene in perturb_rank]
    common_gene_set = set(common_genes)

    target_blocks_control_order = defaultdict(list)
    target_blocks_perturb_order = defaultdict(list)

    for gene in common_genes:
        target_rank = perturb_rank[gene]
        target_block = target_rank // block_size + 1
        target_blocks_control_order[target_block].append(gene)

    for gene in perturb_genes:
        if gene not in common_gene_set:
            continue

        target_rank = perturb_rank[gene]
        target_block = target_rank // block_size + 1
        target_blocks_perturb_order[target_block].append(gene)

    return (
        dict(target_blocks_control_order),
        dict(target_blocks_perturb_order),
        common_genes
    )


# ==========================================
# 优化后的 Prompt 构建逻辑 (System + User)
# ==========================================

def build_planner_system(block_size):
    return (
        "You are an AI expert in transcriptomic perturbation analysis. "
        "Your task is to predict the complete target gene sets for consecutive expression blocks after a given perturbation. "
        f"Each block contains {block_size} genes. "
        "Keep the genes within each output block in the same relative order as they appear in the baseline control expression."
    )

def build_planner_user(record, control_genes):
    return (
        "[Condition]\n"
        f"{build_condition_text(record)}\n\n"
        "[Control Expression]\n"
        f"{format_gene_list(control_genes)}"
    )

def build_planner_assistant(target_blocks_control_order):
    return format_blocks(target_blocks_control_order)

def build_reranker_system(block_size):
    return (
        "You are an AI expert in transcriptomic perturbation analysis. "
        "Your task is to rerank the genes within each provided expression block to reflect their final perturbed state. "
        f"Each block contains {block_size} genes. Reorder genes strictly within their own blocks without moving genes across blocks."
    )

def build_reranker_user(record, target_blocks_control_order):
    return (
        "[Condition]\n"
        f"{build_condition_text(record)}\n\n"
        "[Input Blocks]\n"
        f"{format_blocks(target_blocks_control_order)}"
    )

def build_reranker_assistant(target_blocks_perturb_order):
    return format_blocks(target_blocks_perturb_order)

def to_messages(system_content, user_content, assistant_content):
    return {
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }

# ==========================================


def process_record(record, args_dict):
    target_split = args_dict["split"]
    block_size = args_dict["block_size"]
    min_common_genes = args_dict["min_common_genes"]

    split = record.get("split", "unknown")

    if split != target_split:
        return {"status": "skip_split", "split": split}

    control_genes = clean_sentence(record.get("control_sentence", ""))
    perturb_genes = clean_sentence(record.get("perturb_sentence", ""))

    if not control_genes or not perturb_genes:
        return {"status": "skip_empty", "split": split}

    (
        target_blocks_control_order,
        target_blocks_perturb_order,
        common_genes
    ) = build_target_blocks(
        control_genes=control_genes,
        perturb_genes=perturb_genes,
        block_size=block_size
    )

    if len(common_genes) < min_common_genes:
        return {
            "status": "skip_short",
            "split": split,
            "common_gene_count": len(common_genes)
        }

    # 组装 Planner 的消息对话
    planner_system = build_planner_system(block_size)
    planner_user = build_planner_user(record, control_genes)
    planner_assistant = build_planner_assistant(target_blocks_control_order)

    # 组装 Reranker 的消息对话
    reranker_system = build_reranker_system(block_size)
    reranker_user = build_reranker_user(record, target_blocks_control_order)
    reranker_assistant = build_reranker_assistant(target_blocks_perturb_order)

    block_gene_counts = {
        str(block_id): len(genes)
        for block_id, genes in target_blocks_control_order.items()
    }

    return {
        "status": "ok",
        "split": split,
        "common_gene_count": len(common_genes),
        "num_blocks": len(target_blocks_control_order),
        "block_gene_counts": block_gene_counts,
        "planner": to_messages(planner_system, planner_user, planner_assistant),
        "reranker": to_messages(reranker_system, reranker_user, reranker_assistant)
    }


def process_chunk(chunk, args_dict):
    results = []
    for record in chunk:
        results.append(process_record(record, args_dict))
    return results


def read_jsonl_in_chunks(input_path, chunk_size, max_sft_samples, target_split):
    chunk = []
    total_seen = 0
    sft_seen = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_seen += 1
            record = json.loads(line)

            if record.get("split", "unknown") == target_split:
                sft_seen += 1

            if max_sft_samples > 0 and sft_seen > max_sft_samples:
                break

            chunk.append(record)

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

    if chunk:
        yield chunk


def write_jsonl_line(f, record):
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    planner_path = outdir / "Planner-SFT.jsonl"
    reranker_path = outdir / "Reranker-SFT.jsonl"
    meta_path = outdir / "cook_sft_meta.json"

    args_dict = {
        "split": args.split,
        "block_size": args.block_size,
        "min_common_genes": args.min_common_genes
    }

    print("Cooking two-stage SFT data (Optimized Prompt Version)...")
    print(f"Input:        {input_path}")
    print(f"Output dir:   {outdir}")
    print(f"Workers:      {args.workers}")
    print(f"Chunk size:   {args.chunk_size}")
    print(f"Target split: {args.split}")
    print(f"Block size:   {args.block_size}")

    total_chunks = 0
    futures = []

    status_counter = Counter()
    split_counter = Counter()
    common_gene_counter = Counter()
    block_counter = Counter()

    used_samples = 0
    total_results = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for chunk in read_jsonl_in_chunks(
            input_path=input_path,
            chunk_size=args.chunk_size,
            max_sft_samples=args.max_samples,
            target_split=args.split
        ):
            total_chunks += 1
            futures.append(
                executor.submit(
                    process_chunk,
                    chunk,
                    args_dict
                )
            )

        with open(planner_path, "w", encoding="utf-8") as planner_f, \
             open(reranker_path, "w", encoding="utf-8") as reranker_f:

            finished_chunks = 0

            for future in as_completed(futures):
                finished_chunks += 1
                results = future.result()

                for result in results:
                    total_results += 1

                    status = result.get("status", "unknown")
                    status_counter[status] += 1

                    split = result.get("split", "unknown")
                    split_counter[split] += 1

                    if status != "ok":
                        continue

                    write_jsonl_line(planner_f, result["planner"])
                    write_jsonl_line(reranker_f, result["reranker"])

                    used_samples += 1
                    common_gene_counter[result["common_gene_count"]] += 1

                    for block_id, count in result["block_gene_counts"].items():
                        block_counter[f"Block {block_id}"] += count

                if finished_chunks % 10 == 0 or finished_chunks == total_chunks:
                    print(
                        f"Finished chunks: {finished_chunks}/{total_chunks} | "
                        f"used samples: {used_samples}"
                    )

    meta = {
        "input": str(input_path),
        "outdir": str(outdir),
        "planner_output": str(planner_path),
        "reranker_output": str(reranker_path),
        "target_split": args.split,
        "block_size": args.block_size,
        "min_common_genes": args.min_common_genes,
        "max_samples": args.max_samples,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "total_chunks": total_chunks,
        "total_processed_records": total_results,
        "used_samples": used_samples,
        "planner_records": used_samples,
        "reranker_records": used_samples,
        "status_counter": dict(status_counter),
        "split_counter": dict(split_counter),
        "common_gene_count_distribution": {
            str(k): v for k, v in sorted(common_gene_counter.items())
        },
        "block_gene_counter": dict(block_counter)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Planner SFT:      {planner_path}")
    print(f"Reranker SFT:     {reranker_path}")
    print(f"Meta:             {meta_path}")
    print(f"Used samples:     {used_samples}")
    print(f"Status counter:   {dict(status_counter)}")


if __name__ == "__main__":
    main()



    