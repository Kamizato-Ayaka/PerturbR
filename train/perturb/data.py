import json
import os

from perturbqa import load_de, load_dir

OUT_PATH = "./perturbqa-train-4tuple.jsonl"

# 官方 subset
SUBSETS = ["k562", "rpe1", "hepg2", "jurkat", "k562_set"]

# DIR 映射
DIR_MAP = {
    0: "down",
    1: "up",
}


def try_load(loader, subset):
    try:
        return loader(subset)
    except:
        return None


def build_dir_lookup(dir_train):
    """
    构建 (pert, gene) → up/down 的字典
    """
    lookup = {}
    for x in dir_train:
        key = (x["pert"], x["gene"])
        lookup[key] = DIR_MAP[x["label"]]
    return lookup


# 清空旧文件
if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)


total = 0

for subset in SUBSETS:
    print(f"\nProcessing {subset}")

    de_data = try_load(load_de, subset)
    dir_data = try_load(load_dir, subset)

    if de_data is None:
        print(f"  skip {subset} (no DE)")
        continue

    de_train = de_data["train"]

    # DIR lookup（可能不存在）
    dir_lookup = {}
    if dir_data is not None:
        dir_lookup = build_dir_lookup(dir_data["train"])

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for x in de_train:
            pert = x["pert"]
            gene = x["gene"]

            if x["label"] == 0:
                label = "NA"
            else:
                label = dir_lookup.get((pert, gene), "NA")  # fallback（理论上不会缺）

            record = {
                "cell": subset,
                "pert": pert,
                "gene": gene,
                "label": label,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    print(f"  done {subset}, samples={len(de_train)}")


print("\n==============================")
print(f"Saved to: {OUT_PATH}")
print(f"Total samples: {total}")
print("==============================")