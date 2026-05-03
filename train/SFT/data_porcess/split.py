#!/usr/bin/env python3
import argparse
import json
import os
import random
import tempfile
from collections import Counter, defaultdict


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON at line {line_no}: {e}")
    return data


def write_jsonl_atomic(data, path):
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".split_tmp_", suffix=".jsonl", dir=out_dir)

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    os.replace(tmp_path, path)


def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def get_cell_key(x):
    for k in ["cell_id", "cell", "cell_line", "cell_name"]:
        v = x.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return "UNKNOWN_CELL"


def get_drug_key(x):
    pert_id = str(x.get("pert_id", "")).strip()
    pert_name = str(
        x.get("pert_iname", x.get("pert_name", x.get("perturbation", "")))
    ).strip()
    smiles = str(x.get("canonical_smiles", x.get("smiles", ""))).strip()

    if pert_id or pert_name or smiles:
        return f"{pert_id}||{pert_name}||{smiles}"

    return "UNKNOWN_DRUG"


def get_pair_key(x):
    return f"{get_cell_key(x)}@@{get_drug_key(x)}"


def choose_n(items, n, rng):
    items = list(items)
    rng.shuffle(items)
    n = max(0, min(n, len(items)))
    return set(items[:n])


def choose_by_frac_or_count(items, frac, count, rng, min_count=0):
    items = list(items)
    rng.shuffle(items)

    if count is not None:
        n = count
    else:
        n = int(round(len(items) * frac))

    if frac > 0 and len(items) > 0:
        n = max(n, min_count)

    n = max(0, min(n, len(items)))
    return set(items[:n])


def assign_ood_cell(data, rng, num_groups):
    cells = sorted({get_cell_key(x) for x in data})
    ood_cells = choose_n(cells, num_groups, rng)

    for x in data:
        if get_cell_key(x) in ood_cells:
            x["split"] = "oodcell"

    return ood_cells


def assign_ood_drug(data, rng, num_groups):
    candidate_drugs = sorted({
        get_drug_key(x)
        for x in data
        if x.get("split") is None
    })

    ood_drugs = choose_n(candidate_drugs, num_groups, rng)

    for x in data:
        if x.get("split") is None and get_drug_key(x) in ood_drugs:
            x["split"] = "ooddrug"

    return ood_drugs


def assign_group_split(data, rng, split_name, frac, count):
    groups = defaultdict(list)

    for i, x in enumerate(data):
        if x.get("split") is None:
            groups[get_pair_key(x)].append(i)

    chosen_groups = choose_by_frac_or_count(
        sorted(groups.keys()),
        frac=frac,
        count=count,
        rng=rng,
        min_count=1,
    )

    for g in chosen_groups:
        for i in groups[g]:
            data[i]["split"] = split_name

    return chosen_groups


def assign_sft_grpo(data, rng, grpo_frac, grpo_count):
    groups = defaultdict(list)

    for i, x in enumerate(data):
        if x.get("split") is None:
            groups[get_pair_key(x)].append(i)

    all_groups = sorted(groups.keys())
    grpo_groups = choose_by_frac_or_count(
        all_groups,
        frac=grpo_frac,
        count=grpo_count,
        rng=rng,
        min_count=1,
    )

    for g, indices in groups.items():
        split_name = "grpo" if g in grpo_groups else "sft"
        for i in indices:
            data[i]["split"] = split_name

    return grpo_groups


def check_leakage(data):
    split_to_cells = defaultdict(set)
    split_to_drugs = defaultdict(set)
    split_to_pairs = defaultdict(set)

    for x in data:
        sp = x["split"]
        split_to_cells[sp].add(get_cell_key(x))
        split_to_drugs[sp].add(get_drug_key(x))
        split_to_pairs[sp].add(get_pair_key(x))

    train_splits = {"sft", "grpo"}

    train_cells = set()
    train_drugs = set()
    train_pairs = set()

    for sp in train_splits:
        train_cells |= split_to_cells.get(sp, set())
        train_drugs |= split_to_drugs.get(sp, set())
        train_pairs |= split_to_pairs.get(sp, set())

    leakage = {
        "oodcell_cell_in_train": sorted(split_to_cells.get("oodcell", set()) & train_cells),
        "ooddrug_drug_in_train": sorted(split_to_drugs.get("ooddrug", set()) & train_drugs),
        "test_id_pair_in_train": sorted(split_to_pairs.get("test_id", set()) & train_pairs),
    }

    return leakage


def print_stats(data, ood_cells, ood_drugs, test_groups, grpo_groups):
    split_counter = Counter(x["split"] for x in data)

    print("\n=== Split Sample Counts ===")
    for k in ["sft", "grpo", "test_id", "oodcell", "ooddrug"]:
        print(f"{k:10s}: {split_counter.get(k, 0)}")

    print("\n=== Split Group Counts ===")
    print(f"oodcell cells : {len(ood_cells)}")
    print(f"ooddrug drugs : {len(ood_drugs)}")
    print(f"test_id pairs : {len(test_groups)}")
    print(f"grpo pairs    : {len(grpo_groups)}")

    print("\n=== OOD Cell Groups ===")
    for c in sorted(ood_cells):
        print(c)

    print("\n=== OOD Drug Groups ===")
    for d in sorted(ood_drugs):
        print(d)

    leakage = check_leakage(data)

    print("\n=== Leakage Check ===")
    for k, v in leakage.items():
        print(f"{k:24s}: {len(v)}")

    if any(len(v) > 0 for v in leakage.values()):
        print("\nWARNING: leakage detected.")
    else:
        print("\nNo train leakage detected for oodcell / ooddrug / test_id.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ood-cell-groups", type=int, default=4)
    parser.add_argument("--ood-drug-groups", type=int, default=20)

    parser.add_argument("--test-id-frac", type=float, default=0.05)
    parser.add_argument("--grpo-frac", type=float, default=0.10)

    parser.add_argument("--test-id-count", type=int, default=None)
    parser.add_argument("--grpo-count", type=int, default=None)

    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    out_path = args.input if args.output is None else args.output

    if args.output is None and not args.force:
        print(
            "You are going to overwrite the input file in-place. "
            "Add --force to confirm, or use --output to write a new file."
        )
        return

    rng = random.Random(args.seed)

    data = load_jsonl(args.input)

    for x in data:
        x.pop("split", None)

    ood_cells = assign_ood_cell(
        data,
        rng=rng,
        num_groups=args.ood_cell_groups,
    )

    ood_drugs = assign_ood_drug(
        data,
        rng=rng,
        num_groups=args.ood_drug_groups,
    )

    test_groups = assign_group_split(
        data,
        rng=rng,
        split_name="test_id",
        frac=args.test_id_frac,
        count=args.test_id_count,
    )

    grpo_groups = assign_sft_grpo(
        data,
        rng=rng,
        grpo_frac=args.grpo_frac,
        grpo_count=args.grpo_count,
    )

    if args.output is None:
        write_jsonl_atomic(data, out_path)
    else:
        write_jsonl(data, out_path)

    print_stats(data, ood_cells, ood_drugs, test_groups, grpo_groups)

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()