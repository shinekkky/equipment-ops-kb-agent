"""一条命令运行全部可复现实验。

默认运行：
  1. 检索评测（实验 A）           -> experiments/results/retrieval_eval.json
  2. 异常检测评测（实验 B）       -> experiments/results/detection_eval.json
  3. 指标核对表                   -> experiments/results/metrics_audit.json
  4. Agent 测试集校验（dry-run）  -> 不产生 Agent 质量指标

Agent 实验 C 需要真实模型，只有显式加 --with-agent（或 --agent-replay）才运行：
    set AGENT_EVAL_API_KEY=sk-...
    python scripts/eval_all.py --with-agent

用法：
    python scripts/eval_all.py
    python scripts/eval_all.py --with-agent --provider "DeepSeek（深度求索）" --model deepseek-chat
    python scripts/eval_all.py --agent-replay experiments/traces/agent_eval_traces.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(title: str, args: list[str]) -> bool:
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        print(f"[失败] {title}（退出码 {result.returncode}）")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="运行全部实验")
    parser.add_argument("--with-agent", action="store_true",
                        help="运行真实模型 Agent 评测（需要 AGENT_EVAL_API_KEY）")
    parser.add_argument("--agent-replay", type=Path, default=None,
                        help="重放已保存的 Agent 执行记录")
    parser.add_argument("--provider", default="DeepSeek（深度求索）")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    steps: list[tuple[str, list[str]]] = [
        ("实验 A：检索方法比较", ["scripts/eval_retrieval.py"]),
        ("实验 B：异常检测方法消融", ["scripts/eval_detection.py"]),
        ("指标核对：README 数字重算", ["scripts/audit_metrics.py"]),
    ]

    if args.agent_replay:
        steps.append((
            f"实验 C：Agent 执行质量（重放 {args.agent_replay}）",
            ["scripts/eval_agent.py", "--replay", str(args.agent_replay)],
        ))
    elif args.with_agent:
        cmd = ["scripts/eval_agent.py", "--provider", args.provider]
        if args.model:
            cmd += ["--model", args.model]
        if args.api_key:
            cmd += ["--api-key", args.api_key]
        steps.append(("实验 C：Agent 执行质量（真实模型）", cmd))
    else:
        steps.append((
            "Agent 测试集校验（dry-run，不产生质量指标）",
            ["scripts/eval_agent.py", "--dry-run"],
        ))

    ok = True
    for title, cmd in steps:
        ok = run(title, cmd) and ok

    print("\n" + "=" * 72, flush=True)
    print("结果目录：experiments/results/", flush=True)
    print("指标核对表：docs/指标核对表.md", flush=True)
    print("实验报告：docs/实验报告.md", flush=True)
    if not (args.with_agent or args.agent_replay):
        print("提示：实验 C 尚未运行。使用 --with-agent 或 --agent-replay 生成真实指标。",
              flush=True)
    print("=" * 72, flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
