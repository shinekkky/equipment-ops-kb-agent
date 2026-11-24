"""Agent 执行记录查询工具。

用法：
    python scripts/agent_trace.py list --limit 20
    python scripts/agent_trace.py show <trace_id>
    python scripts/agent_trace.py export <trace_id> --out experiments/traces/one.json
    python scripts/agent_trace.py stats

记录默认位于 experiments/traces/agent_traces.jsonl，可用环境变量
AGENT_TRACE_LOG 覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent.trace_store import (  # noqa: E402
    export_trace,
    iter_traces,
    list_traces,
    load_trace,
    trace_path,
    trace_stats,
)


def _short(text: str, width: int = 46) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def cmd_list(args) -> None:
    rows = list_traces(args.limit, args.path)
    if not rows:
        print(f"暂无执行记录：{trace_path(args.path)}")
        return
    print(f"记录文件：{trace_path(args.path)}\n")
    header = f"{'trace_id':<12}{'时间':<26}{'类型':<10}{'轮':>3}{'工具':>5}{'耗时(ms)':>10}{'tokens':>8}  问题"
    print(header)
    print("-" * (len(header) + 20))
    for row in rows:
        print(
            f"{row.get('trace_id', '')[:10]:<12}"
            f"{(row.get('started_at') or '')[:25]:<26}"
            f"{row.get('kind', ''):<10}"
            f"{row.get('rounds', 0):>3}"
            f"{row.get('tool_call_count', 0):>5}"
            f"{row.get('duration_ms', 0):>10}"
            f"{row.get('total_tokens') or '—':>8}  "
            f"{_short(row.get('question', ''))}"
            f"{'  [ERROR]' if row.get('error') else ''}"
        )


def cmd_show(args) -> None:
    record = load_trace(args.trace_id, args.path)
    if record is None:
        print(f"未找到 trace_id={args.trace_id}（{trace_path(args.path)}）")
        raise SystemExit(1)
    print(json.dumps(record, ensure_ascii=False, indent=2, default=str))


def cmd_export(args) -> None:
    try:
        target = export_trace(args.trace_id, args.out, args.path)
    except KeyError as exc:
        print(str(exc))
        raise SystemExit(1)
    print(f"已导出：{target}")


def cmd_stats(args) -> None:
    rows = list(iter_traces(args.path))
    stats = trace_stats(rows)
    if not stats.get("n"):
        print(f"暂无执行记录：{trace_path(args.path)}")
        return
    print(f"记录文件：{trace_path(args.path)}")
    print(f"运行次数        ：{stats['n']}")
    print(f"失败次数        ：{stats['error_runs']}（{stats['error_rate'] * 100:.1f}%）")
    print(f"平均耗时        ：{stats['avg_duration_ms']:.0f} ms")
    print(f"最大耗时        ：{stats['max_duration_ms']} ms")
    print(f"平均工具调用次数：{stats['avg_tool_calls']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 执行记录查询")
    parser.add_argument("--path", default=None, help="覆盖 JSONL 路径")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出最近的执行记录")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="按 trace_id 查看完整记录")
    p_show.add_argument("trace_id")
    p_show.set_defaults(func=cmd_show)

    p_export = sub.add_parser("export", help="导出单次执行记录")
    p_export.add_argument("trace_id")
    p_export.add_argument("--out", type=Path, required=True)
    p_export.set_defaults(func=cmd_export)

    p_stats = sub.add_parser("stats", help="汇总统计")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
