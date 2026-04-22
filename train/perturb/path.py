import json
import os
import urllib.parse
from collections import defaultdict

import networkx as nx
import pandas as pd
import requests


INPUT_JSONL = "perturbqa-train-raw.jsonl"
OUTPUT_JSONL = "perturbqa-train-with-paths.jsonl"

MAX_HOPS = 5
MAX_PATHS = 20


def fetch_omnipath_basic():
    """
    用最基础的参数先拉一份 interactions 表，避免 fields 参数导致接口报错。
    """
    base = "https://omnipathdb.org/interactions"
    params = {
        "format": "tsv",
        "genesymbols": "1",
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"

    print("Requesting:", url)
    r = requests.get(url, timeout=120)
    print("HTTP status:", r.status_code)

    # 先看前几行原始返回
    preview = r.text[:500]
    print("\n=== Response preview ===")
    print(preview)
    print("========================\n")

    r.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(r.text), sep="\t")
    print("Downloaded interactions shape:", df.shape)
    print("Columns:", list(df.columns))
    return df


def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_columns(df):
    # OmniPath 常见列名
    src_col = pick_first_existing(df, [
        "source_genesymbol", "genesymbol_source", "source", "source_symbol",
        "genesymbol_a", "source_genesymbols"
    ])
    dst_col = pick_first_existing(df, [
        "target_genesymbol", "genesymbol_target", "target", "target_symbol",
        "genesymbol_b", "target_genesymbols"
    ])

    stim_col = pick_first_existing(df, [
        "is_stimulation", "consensus_stimulation", "stimulation"
    ])
    inhib_col = pick_first_existing(df, [
        "is_inhibition", "consensus_inhibition", "inhibition"
    ])
    directed_col = pick_first_existing(df, [
        "is_directed", "consensus_direction", "directed"
    ])
    sources_col = pick_first_existing(df, [
        "sources", "resource", "resources"
    ])

    if src_col is None or dst_col is None:
        raise ValueError(
            f"Could not detect source/target columns.\nColumns found: {list(df.columns)}"
        )

    print("Detected columns:")
    print("  src      =", src_col)
    print("  dst      =", dst_col)
    print("  stim     =", stim_col)
    print("  inhib    =", inhib_col)
    print("  directed =", directed_col)
    print("  sources  =", sources_col)

    return {
        "src": src_col,
        "dst": dst_col,
        "stim": stim_col,
        "inhib": inhib_col,
        "directed": directed_col,
        "sources": sources_col,
    }


def truthy(x):
    if pd.isna(x):
        return False
    if isinstance(x, (bool, int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def normalize_sources(x):
    if pd.isna(x):
        return []
    s = str(x)
    parts = [p.strip() for p in s.replace(",", ";").split(";") if p.strip()]
    return sorted(set(parts))


def build_signed_digraph(df):
    cols = detect_columns(df)
    G = nx.DiGraph()

    skipped_unsigned = 0
    skipped_undirected = 0
    kept = 0

    for _, row in df.iterrows():
        u = row[cols["src"]]
        v = row[cols["dst"]]

        if pd.isna(u) or pd.isna(v):
            continue
        u = str(u).strip()
        v = str(v).strip()
        if not u or not v:
            continue

        if cols["directed"] is not None and not truthy(row[cols["directed"]]):
            skipped_undirected += 1
            continue

        sign = None
        if cols["stim"] is not None or cols["inhib"] is not None:
            stim = truthy(row[cols["stim"]]) if cols["stim"] else False
            inhib = truthy(row[cols["inhib"]]) if cols["inhib"] else False

            if stim and not inhib:
                sign = +1
            elif inhib and not stim:
                sign = -1

        if sign is None:
            skipped_unsigned += 1
            continue

        G.add_edge(
            u, v,
            sign=sign,
            sign_name="positive" if sign == +1 else "negative",
            datasets=normalize_sources(row[cols["sources"]]) if cols["sources"] else [],
        )
        kept += 1

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Kept signed edges: {kept}")
    print(f"Skipped undirected: {skipped_undirected}")
    print(f"Skipped unsigned: {skipped_unsigned}")
    return G


def path_sign(G, nodes):
    sign = +1
    edges = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        data = G[a][b]
        sign *= data["sign"]
        edges.append({
            "src": a,
            "dst": b,
            "sign": data["sign_name"],
            "datasets": data.get("datasets", []),
        })
    return sign, edges


def implied_label_under_knockdown(net_sign):
    # 假设 pert 是 knockdown / loss-of-function
    # 正路径: knockdown causes target down
    # 负路径: knockdown causes target up
    return "down" if net_sign == +1 else "up"


def enumerate_paths(G, src, dst, max_hops=3, max_paths=20):
    if src not in G or dst not in G:
        return []

    results = []
    try:
        for p in nx.all_simple_paths(G, src, dst, cutoff=max_hops):
            net_sign, edges = path_sign(G, p)
            results.append({
                "nodes": p,
                "edges": edges,
                "path_sign": "positive" if net_sign == +1 else "negative",
                "implied_label_under_knockdown": implied_label_under_knockdown(net_sign),
                "num_hops": len(p) - 1,
            })
            if len(results) >= max_paths:
                break
    except nx.NetworkXNoPath:
        return []

    results.sort(key=lambda x: x["num_hops"])
    return results


def main():
    df = fetch_omnipath_basic()
    G = build_signed_digraph(df)

    total = 0
    routed = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as f, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as out:

        for line in f:
            r = json.loads(line)
            total += 1

            label = r["label"]
            pert = r["pert"]
            gene = r["gene"]

            out_record = dict(r)

            if label == "NA":
                out_record["paths"] = []
                out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                continue

            paths = enumerate_paths(G, pert, gene, max_hops=MAX_HOPS, max_paths=MAX_PATHS)

            for p in paths:
                p["matches_observed_label"] = (p["implied_label_under_knockdown"] == label)

            out_record["paths"] = paths
            if paths:
                routed += 1

            out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Done: {OUTPUT_JSONL}")
    print(f"Total records: {total}")
    print(f"Significant records with >=1 path: {routed}")


if __name__ == "__main__":
    main()