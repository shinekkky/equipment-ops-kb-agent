"""Agent 执行质量评测。

指标与判定方式（全部程序化，详见 core/evaluation/agent_scoring.py）：
  * Tool Selection Accuracy：期望工具是否被调用（required_any 允许替代路径）。
  * Tool Argument Accuracy：调用参数是否包含期望的 device_id / metric 等。
  * Task Success Rate：答案是否包含全部必需结论，且没有出现禁止表述。
  * Numeric Accuracy：期望数值是否出现在答案中（容差 2%）。
  * Evidence Accuracy：手册类问题是否同时给出手册名与正确章节号。
  * Unsupported Claim Rate：答案中的数字能否追溯到工具结果/问题/允许的派生值。

用法：
    # 先校验测试集，不调用模型
    python scripts/eval_agent.py --dry-run

    # 真实运行（需要 API Key）
    set AGENT_EVAL_API_KEY=sk-...
    python scripts/eval_agent.py --provider "DeepSeek（深度求索）" --model deepseek-chat

    # 重放已保存的执行记录（不调用模型）
    python scripts/eval_agent.py --replay experiments/traces/agent_eval_traces.jsonl

没有真实模型运行结果时，不要把 dry-run 或重放自测当成 Agent 质量结论。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent import AgentContext, answer_question, build_llm, build_tools  # noqa: E402
from core.config import (  # noqa: E402
    AGENT_EVAL_SET_JSONL,
    MANUALS_DIR,
    RESULTS_DIR,
    TRACE_LOG_PATH,
)
from core.diagnostics import load_registry, load_sensor_data, scan  # noqa: E402
from core.evaluation.agent_scoring import (  # noqa: E402
    aggregate,
    load_dataset,
    normalize,
    score_case,
    validate_dataset,
)
from core.kb.index import build_index, get_embedder  # noqa: E402

DEFAULT_OUTPUT = RESULTS_DIR / "agent_eval.json"
DEFAULT_TRACE = TRACE_LOG_PATH.parent / "agent_eval_traces.jsonl"


def _provider_default() -> str:
    from core.config import PROVIDER_PRESETS

    return "DeepSeek（深度求索）" if "DeepSeek（深度求索）" in PROVIDER_PRESETS else next(iter(PROVIDER_PRESETS))


def _api_key(args) -> str:
    return (
        args.api_key
        or os.environ.get("AGENT_EVAL_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or ""
    )


def _build_context():
    data = load_sensor_data()
    registry = load_registry()
    result = scan(data=data, registry=registry)
    embedder = get_embedder()
    index = build_index(sorted(MANUALS_DIR.glob("*.md")), embedder=embedder)
    ctx = AgentContext(scan=result, data=data, index=index,
                       retrieval_mode="hybrid" if embedder is not None else "bm25")
    return ctx, result, index, embedder is not None


def _replay_records(path: Path) -> dict[str, dict]:
    """按问题文本索引执行记录；同一问题多次运行时取最后一条。"""
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = normalize(record.get("question", ""))
        if key:
            out[key] = record
    return out


def _print_summary(summary: dict) -> None:
    def pct(value):
        return "—" if value is None else f"{value * 100:.1f}%"

    print("\nAgent 评测结果")
    print("-" * 46)
    print(f"用例数              ：{summary['n']}")
    print(f"Tool Selection Acc  ：{pct(summary['tool_selection_accuracy'])}")
    print(f"Tool Argument Acc   ：{pct(summary['tool_argument_accuracy'])}")
    print(f"Task Success Rate   ：{pct(summary['task_success_rate'])}")
    print(f"Numeric Accuracy    ：{pct(summary['numeric_accuracy'])}")
    print(f"Evidence Accuracy   ：{pct(summary['evidence_accuracy'])}")
    print(f"Unsupported Claim   ：{pct(summary['unsupported_claim_rate'])}"
          f"（{summary['numeric_claims_unsupported']}/{summary['numeric_claims_total']} 个数字）")
    print(f"执行错误率          ：{pct(summary['error_rate'])}")
    print(f"全项通过率          ：{pct(summary['success_rate'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 执行质量评测")
    parser.add_argument("--dataset", type=Path, default=AGENT_EVAL_SET_JSONL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", default=_provider_default())
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于冒烟")
    parser.add_argument("--dry-run", action="store_true", help="只校验测试集，不调用模型")
    parser.add_argument("--replay", type=Path, default=None,
                        help="重放已保存的 JSONL 执行记录，不调用模型")
    parser.add_argument("--trace-log", type=Path, default=DEFAULT_TRACE,
                        help="真实运行时执行记录写入位置")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    ctx, result, index, vector_enabled = _build_context()
    tool_names = {t.name for t in build_tools(ctx)}
    problems = validate_dataset(cases, tool_names)

    print(f"测试集：{args.dataset}（{len(cases)} 条）")
    print(f"知识库：{index.stats()['documents']} 份手册 / {index.stats()['children']} 块；"
          f"向量检索 {'启用' if vector_enabled else '未启用（仅 BM25）'}")

    if problems:
        print("\n测试集校验发现问题：")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    if args.dry_run:
        from collections import Counter

        counts = Counter(c["category"] for c in cases)
        print("\n测试集校验通过。分类覆盖：")
        for category, count in counts.items():
            print(f"  - {category}: {count} 条")
        print("\n[dry-run] 未调用模型，不产生质量指标。")
        return

    if args.replay:
        if not args.replay.exists():
            print(f"[错误] 找不到执行记录：{args.replay}")
            raise SystemExit(2)
        records = _replay_records(args.replay)
        scores: list[dict] = []
        for case in cases:
            record = records.get(normalize(case["question"]))
            if record is None:
                record = {
                    "final_answer": "", "tool_calls": [], "error": "未找到该问题的执行记录",
                    "trace_id": None, "question": case["question"],
                }
            scores.append(score_case(case, record))
        mode = f"replay:{args.replay}"
    else:
        api_key = _api_key(args)
        if not api_key:
            print("\n[提示] 未提供 API Key，无法运行真实模型评测。")
            print("       可设置 AGENT_EVAL_API_KEY，或用 --dry-run 校验测试集。")
            print("       本次不会生成任何 Agent 质量指标。")
            raise SystemExit(2)

        os.environ.setdefault("AGENT_TRACE_LOG", str(args.trace_log))
        llm = build_llm(
            args.provider, api_key,
            base_url=args.base_url or None,
            model=args.model or None,
            temperature=args.temperature,
        )
        scores = []
        for i, case in enumerate(cases, start=1):
            print(f"[{i}/{len(cases)}] {case['id']} {case['question']}")
            try:
                run = answer_question(ctx, llm, case["question"])
                record = run.to_dict()
            except Exception as exc:
                record = {
                    "final_answer": "", "tool_calls": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace_id": None, "question": case["question"],
                }
            score = score_case(case, record)
            scores.append(score)
            mark = "OK" if score["success"] else "FAIL"
            print(f"    [{mark}] {'；'.join(score['failure_reasons']) or '全部通过'}")
        mode = f"live:{args.provider}/{args.model or 'default'}"

    summary = aggregate(scores)
    _print_summary(summary)

    failures = [s for s in scores if not s["success"]]
    payload = {
        "config": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "mode": mode,
            "dataset": str(args.dataset),
            "n_cases": len(cases),
            "provider": None if args.replay else args.provider,
            "model": None if args.replay else (args.model or "default"),
            "temperature": None if args.replay else args.temperature,
            "retrieval_mode": "hybrid" if vector_enabled else "bm25",
        },
        "summary": summary,
        "cases": scores,
        "failures": [
            {
                "id": s["id"],
                "category": s["category"],
                "question": s["question"],
                "reasons": s["failure_reasons"],
                "answer": s["answer"][:500],
                "trace_id": s["trace_id"],
            }
            for s in failures
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\n失败用例：{len(failures)} 条")
    print(f"结果已保存：{args.output}")
    if not args.replay:
        print(f"执行记录：{args.trace_log}")


if __name__ == "__main__":
    main()
