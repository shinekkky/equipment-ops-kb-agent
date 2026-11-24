"""Agent 执行记录的持久化：JSONL 追加写、按 trace_id 查询、单次导出。

设计取舍
--------
* 用 JSONL 而不是数据库：一次运行一行，追加写不会因为进程崩溃丢历史，
  也方便直接用 ``grep`` / ``jq`` 看。评测与演示都不需要引入 SQLite。
* 记录里不写 API Key、不写完整提示词：只保留问题、工具调用、最终回答与
  耗时/Token 等执行元数据。提示词里可能含企业内部手册片段，不适合落盘。
* 写入失败不影响主流程：可观测性不能成为 Agent 回答失败的借口。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from ..config import TRACE_LOG_PATH

TRACE_ENV = "AGENT_TRACE_LOG"


def trace_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get(TRACE_ENV, TRACE_LOG_PATH))


def append_trace(record: dict, path: Path | str | None = None) -> Path | None:
    """追加一条执行记录，返回实际写入路径；失败返回 None。"""
    target = trace_path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return target
    except Exception:
        return None


def iter_traces(path: Path | str | None = None) -> Iterator[dict]:
    target = trace_path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # 半行写入或手工编辑损坏时跳过，不让整份日志不可读


def list_traces(limit: int = 20, path: Path | str | None = None) -> list[dict]:
    rows = list(iter_traces(path))
    return rows[-limit:][::-1]


def load_trace(trace_id: str, path: Path | str | None = None) -> dict | None:
    for row in iter_traces(path):
        if row.get("trace_id") == trace_id:
            return row
    return None


def export_trace(trace_id: str, output: Path | str, path: Path | str | None = None) -> Path:
    record = load_trace(trace_id, path)
    if record is None:
        raise KeyError(f"未找到 trace_id={trace_id}")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def trace_stats(records: Iterable[dict]) -> dict:
    rows = list(records)
    if not rows:
        return {"n": 0}
    durations = [r.get("duration_ms", 0) for r in rows]
    tool_counts = [r.get("tool_call_count", 0) for r in rows]
    errors = sum(1 for r in rows if r.get("error"))
    return {
        "n": len(rows),
        "error_runs": errors,
        "error_rate": errors / len(rows),
        "avg_duration_ms": sum(durations) / len(durations),
        "max_duration_ms": max(durations),
        "avg_tool_calls": sum(tool_counts) / len(tool_counts),
    }
