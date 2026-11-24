"""检索效果评测：对三种检索模式计算 Hit@k 与 MRR。

把"我做了 RAG"变成"我做了 RAG 并量化了效果"。评测集里每条查询都有唯一
正确的 (手册, 章节号)，命中即算对。

指标说明：
  Hit@k  前 k 条结果中包含正确答案的查询占比。衡量"能不能找到"。
  MRR    正确答案排名的倒数均值。衡量"排得够不够靠前"，比 Hit@k 更敏感。

可信性约定：
  * 结果除打印外，还会写入 experiments/results/retrieval_eval.json，
    其中记录模型名、分块参数、检索参数、评测集哈希与运行时间。
  * 向量模型不可用时，hybrid 不会被当作一种独立方法评测（它会退化成 BM25）。
    显式指定 --mode hybrid 会直接报错，而不是给出一张误导性的表。

用法：
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --mode hybrid --show-detail
    python scripts/eval_retrieval.py --output experiments/results/my_run.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (  # noqa: E402
    EMBEDDING_MODEL,
    EVAL_SET_JSONL,
    MANUALS_DIR,
    RESULTS_DIR,
    TOP_K_MAX,
)
from core.kb.chunker import CHILD_OVERLAP, CHILD_SIZE  # noqa: E402
from core.kb.index import RRF_K, build_index, get_embedder  # noqa: E402

MODES = ["bm25", "vector", "hybrid"]
KS = (1, 3, 5)
DEFAULT_OUTPUT = RESULTS_DIR / "retrieval_eval.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def package_version(name: str) -> str | None:
    import importlib

    try:
        module = importlib.import_module(name.replace("-", "_"))
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


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
        row = {
            "id": case.get("id", ""),
            "query": case["query"],
            "type": case.get("type", ""),
            "rank": rank,
            "expected": f"《{case['doc']}》第 {case['section_no']} 节",
            "top1": top,
        }
        per_case.append(row)
        if not rank:
            misses.append({
                "query": case["query"],
                "expected": row["expected"],
                "top1": top,
            })

    n = len(cases) or 1
    return {
        "mode": mode,
        "n": len(cases),
        "hit": {str(k): hit_counts[k] / n for k in KS},
        "mrr": sum(reciprocal_ranks) / n,
        "hit_counts": {str(k): hit_counts[k] for k in KS},
        "misses": misses,
        "per_case": per_case,
    }


def build_config(cases: list[dict], index, embedder) -> dict:
    stats = index.stats()
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "eval_set": {
            "path": str(EVAL_SET_JSONL.relative_to(Path(__file__).resolve().parent.parent)),
            "sha256": sha256_file(EVAL_SET_JSONL),
            "n_queries": len(cases),
            "types": sorted({c.get("type", "") for c in cases if c.get("type")}),
        },
        "index": {
            "documents": stats["documents"],
            "parents": stats["parents"],
            "children": stats["children"],
            "vector_enabled": stats["vector_enabled"],
            "embedding_model": EMBEDDING_MODEL if embedder is not None else None,
            "chunk_size": CHILD_SIZE,
            "chunk_overlap": CHILD_OVERLAP,
            "rrf_k": RRF_K,
            "top_k_max": TOP_K_MAX,
        },
        "packages": {
            name: package_version(name)
            for name in ("numpy", "jieba", "sentence-transformers", "langchain-text-splitters")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检索效果评测")
    parser.add_argument("--mode", default=None, choices=MODES, help="只评测指定模式")
    parser.add_argument("--show-misses", action="store_true", help="打印未命中的查询")
    parser.add_argument("--show-detail", action="store_true", help="打印每条查询的排名")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"JSON 结果路径（默认 {DEFAULT_OUTPUT}）")
    parser.add_argument("--no-save", action="store_true", help="只打印，不写 JSON")
    args = parser.parse_args()

    cases = load_eval_set(EVAL_SET_JSONL)
    if not cases:
        print(f"评测集为空：{EVAL_SET_JSONL}")
        return

    embedder = get_embedder()
    if embedder is None:
        print("[警告] 嵌入模型不可用，本次只评测 BM25；hybrid 会退化成 BM25，故不列入结果。\n")

    index = build_index(sorted(MANUALS_DIR.glob("*.md")), embedder=embedder)
    available = index.modes

    if args.mode:
        if args.mode not in available:
            hint = "（向量模型不可用，hybrid 已退化为 BM25，不能作为独立方法评测）" \
                if args.mode == "hybrid" else ""
            print(f"[错误] 模式 {args.mode} 当前不可评测{hint}")
            print(f"       可用模式：{', '.join(available)}")
            raise SystemExit(2)
        modes = [args.mode]
    else:
        modes = [m for m in MODES if m in available]

    stats = index.stats()
    print(f"索引：{stats['documents']} 份手册 / {stats['children']} 子块 / "
          f"{stats['parents']} 节；向量检索 {'已启用' if stats['vector_enabled'] else '未启用'}")
    print(f"评测集：{len(cases)} 条查询（sha256 {sha256_file(EVAL_SET_JSONL)[:12]}…）")
    print(f"分块：size={CHILD_SIZE} / overlap={CHILD_OVERLAP}；RRF k={RRF_K}\n")

    results = [evaluate(index, cases, m) for m in modes]

    header = f"{'检索模式':<10}" + "".join(f"{'Hit@' + str(k):>10}" for k in KS) + f"{'MRR':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r['mode']:<10}"
        line += "".join(f"{r['hit'][str(k)] * 100:>9.1f}%" for k in KS)
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

    payload = {
        "config": build_config(cases, index, embedder),
        "summary": [
            {k: r[k] for k in ("mode", "n", "hit", "mrr", "hit_counts")}
            for r in results
        ],
        "runs": results,
    }

    if args.no_save:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n结果已保存：{args.output}")


if __name__ == "__main__":
    main()
