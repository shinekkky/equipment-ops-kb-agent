"""检索效果评测：对三种检索模式计算 Hit@k 与 MRR。

把"我做了 RAG"变成"我做了 RAG 并量化了效果"。评测集里每条查询都有唯一
正确的 (手册, 章节号)，命中即算对。

指标说明：
  Hit@k  前 k 条结果中包含正确答案的查询占比。衡量"能不能找到"。
  MRR    正确答案排名的倒数均值。衡量"排得够不够靠前"，比 Hit@k 更敏感。

用法：
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --mode hybrid --show-misses
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import EVAL_SET_JSONL, MANUALS_DIR  # noqa: E402
from core.kb.index import build_index, get_embedder  # noqa: E402

MODES = ["bm25", "vector", "hybrid"]
KS = (1, 3, 5)


def load_eval_set(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(index, cases: list[dict], mode: str, k_max: int = 5) -> dict:
    """跑一遍评测集，返回 Hit@k、MRR 与未命中清单。"""
    hit_counts = {k: 0 for k in KS}
    reciprocal_ranks: list[float] = []
    misses: list[dict] = []
    per_case: list[dict] = []

    for case in cases:
        hits = index.search(case["query"], top_k=k_max, mode=mode)

        # 找到第一条与标注 (手册, 章节号) 一致的结果
        rank = 0
        got = ""
        for i, h in enumerate(hits, start=1):
            if h.chunk.doc_title == case["doc"] and h.chunk.section_no == case["section_no"]:
                rank = i
                got = h.citation
                break

        for k in KS:
            if 0 < rank <= k:
                hit_counts[k] += 1

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        top = hits[0].citation if hits else "（无结果）"
        per_case.append({
            "query": case["query"],
            "rank": rank,
            "expected": f"《{case['doc']}》第 {case['section_no']} 节",
            "top1": top,
        })
        if not rank:
            misses.append({
                "query": case["query"],
                "expected": f"《{case['doc']}》第 {case['section_no']} 节",
                "top1": top,
            })

    n = len(cases) or 1
    return {
        "mode": mode,
        "n": len(cases),
        "hit": {k: hit_counts[k] / n for k in KS},
        "mrr": sum(reciprocal_ranks) / n,
        "misses": misses,
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检索效果评测")
    parser.add_argument("--mode", default=None, choices=MODES, help="只评测指定模式")
    parser.add_argument("--show-misses", action="store_true", help="打印未命中的查询")
    parser.add_argument("--show-detail", action="store_true", help="打印每条查询的排名")
    args = parser.parse_args()

    cases = load_eval_set(EVAL_SET_JSONL)
    if not cases:
        print(f"评测集为空：{EVAL_SET_JSONL}")
        return

    embedder = get_embedder()
    if embedder is None:
        print("[警告] 嵌入模型不可用，仅评测 BM25\n")

    index = build_index(sorted(MANUALS_DIR.glob("*.md")), embedder=embedder)
    stats = index.stats()
    print(f"索引：{stats['documents']} 份手册 / {stats['children']} 子块 / "
          f"{stats['parents']} 节；向量检索 {'已启用' if stats['vector_enabled'] else '未启用'}")
    print(f"评测集：{len(cases)} 条查询\n")

    modes = [args.mode] if args.mode else [m for m in MODES if m in index.modes]
    results = [evaluate(index, cases, m) for m in modes]

    header = f"{'检索模式':<10}" + "".join(f"{'Hit@' + str(k):>10}" for k in KS) + f"{'MRR':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r['mode']:<10}"
        line += "".join(f"{r['hit'][k] * 100:>9.1f}%" for k in KS)
        line += f"{r['mrr']:>10.3f}"
        print(line)

    if args.show_detail:
        best = max(results, key=lambda r: r["mrr"])
        print(f"\n各查询排名（{best['mode']}）：")
        for row in best["per_case"]:
            mark = "OK " if row["rank"] else "MISS"
            rank = f"#{row['rank']}" if row["rank"] else " - "
            print(f"  [{mark}] {rank:>3}  {row['query']}")
            if not row["rank"]:
                print(f"          期望 {row['expected']}")
                print(f"          实际 {row['top1']}")

    if args.show_misses:
        print("\n未命中明细：")
        for r in results:
            if not r["misses"]:
                continue
            print(f"\n  [{r['mode']}] {len(r['misses'])} 条未命中")
            for m in r["misses"]:
                print(f"    - {m['query']}")
                print(f"      期望：{m['expected']}")
                print(f"      实际：{m['top1']}")


if __name__ == "__main__":
    main()
