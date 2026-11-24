"""异常检测评测：四层检测 vs 单层/仅阈值，基于注入故障真值。

评测口径（详见 core/evaluation/matching.py）：
  * 故障事件级：一次注入故障是否被至少一个主指标发现；主指标是否全部发现。
  * 指标事件级：以 (故障, 指标) 为召回单位，以 Finding 为精确率单位。
  * 采样点级：每个 (设备, 指标, 时刻) 的二分类 P/R/F1。

另外输出：
  * 四层相对仅阈值层的新增发现，以及其中被真值证实的部分；
  * 逐主指标的提前量（均值/中位数/最大值），阈值层漏报的项单独列出；
  * 按故障类型拆分的检出情况。

用法：
    python scripts/eval_detection.py
    python scripts/eval_detection.py --output experiments/results/detection_eval.json
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
    DEVICE_REGISTRY_CSV,
    FAULT_TRUTH_JSON,
    RESULTS_DIR,
    SENSOR_CSV,
)
from core.diagnostics import load_registry, load_sensor_data, scan  # noqa: E402
from core.diagnostics.detectors import (  # noqa: E402
    BASELINE_DAYS,
    EVENT_MAX_GAP_HOURS,
    MV_MIN_JOINT_Z,
    TREND_MIN_NET_SIGMA,
    TREND_MIN_SIGMA_PER_DAY,
    TREND_WINDOW_HOURS,
    Z_THRESHOLD,
    Z_WINDOW_HOURS,
)
from core.evaluation.ground_truth import (  # noqa: E402
    FAULT_EVENTS,
    write_ground_truth,
)
from core.evaluation.matching import (  # noqa: E402
    added_findings,
    event_metrics,
    finding_metrics,
    lead_times,
    per_fault_type,
    point_metrics,
)

DEFAULT_OUTPUT = RESULTS_DIR / "detection_eval.json"

METHODS: dict[str, set[str] | None] = {
    "阈值检测": {"threshold"},
    "滚动 Z-score": {"zscore"},
    "趋势检测": {"trend"},
    "多变量检测": {"multivariate"},
    "四层组合": None,
}


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


def build_config() -> dict:
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "data": {
            "sensor_csv": SENSOR_CSV.name,
            "sensor_sha256": sha256_file(SENSOR_CSV),
            "registry_csv": DEVICE_REGISTRY_CSV.name,
            "registry_sha256": sha256_file(DEVICE_REGISTRY_CSV),
            "ground_truth_sha256": sha256_file(FAULT_TRUTH_JSON),
            "n_events": len(FAULT_EVENTS),
        },
        "detector_params": {
            "baseline_days": BASELINE_DAYS,
            "z_window_hours": Z_WINDOW_HOURS,
            "z_threshold": Z_THRESHOLD,
            "trend_window_hours": TREND_WINDOW_HOURS,
            "trend_min_sigma_per_day": TREND_MIN_SIGMA_PER_DAY,
            "trend_min_net_sigma": TREND_MIN_NET_SIGMA,
            "mv_min_joint_z": MV_MIN_JOINT_Z,
            "event_max_gap_hours": EVENT_MAX_GAP_HOURS,
            "match_tolerance": "min(24h, max(2h, event_duration))",
        },
        "methods": list(METHODS),
    }


def run_methods(data, registry) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, layers in METHODS.items():
        result = scan(data=data, registry=registry, layers=layers)
        out[name] = {
            "scan": result,
            "findings": finding_metrics(result.findings, FAULT_EVENTS),
            "points": point_metrics(data, result.findings, FAULT_EVENTS),
            "events": event_metrics(result.findings, FAULT_EVENTS),
            "by_fault_type": per_fault_type(
                event_metrics(result.findings, FAULT_EVENTS)
            ),
        }
    return out


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# 异常检测实验报告",
        "",
        f"> 自动生成于 {payload['config']['generated_at']}；"
        f"数据 `{payload['config']['data']['sensor_csv']}`；"
        f"注入故障 {payload['config']['data']['n_events']} 起。",
        "",
        "## 1. 方法对比",
        "",
        "| 方法 | Findings | 指标事件级 P | R | F1 | 采样点级 P | R | F1 | 事件召回 | 主指标全覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.items():
        lines.append(
            f"| {name} | {row['n_findings']} | {pct(row['finding_precision'])} | "
            f"{pct(row['finding_recall'])} | {pct(row['finding_f1'])} | "
            f"{pct(row['point_precision'])} | {pct(row['point_recall'])} | "
            f"{pct(row['point_f1'])} | {pct(row['event_recall'])} | "
            f"{pct(row['full_event_recall'])} |"
        )

    lines += [
        "",
        "口径说明：",
        "",
        "- **指标事件级**：召回单位是 (故障, 主/次指标)，精确率单位是 Finding。"
        "同一个问题被多层重复报时，会重复计入精确率分母。",
        "- **采样点级**：每个 (设备, 指标, 时刻) 的二分类结果；真值只覆盖注入故障的"
        "主/次指标与时间窗，其他任何命中都算 FP。",
        "- **事件召回**：6 起注入故障中，至少一个主指标被发现的占比；"
        "**主指标全覆盖**：所有主指标都被发现的占比。",
        "",
        "## 2. 四层相对阈值层的新增发现",
        "",
        f"- 新增 Finding：**{payload['added']['added_findings']}** 条；"
        f"其中被真值证实：**{payload['added']['validated_by_ground_truth']}** 条；"
        f"未被证实：**{payload['added']['unvalidated']}** 条。",
        "",
        "| 设备 | 指标 | 时段 | 检出层 | 真值证实 |",
        "|---|---|---|---|---|",
    ]
    for item in payload["added"]["items"]:
        lines.append(
            f"| {item['device_id']} | {item['metric']} | {item['start']} ~ {item['end']} | "
            f"{'、'.join(item['layers'])} | {'是' if item['validated'] else '否'} |"
        )

    lead = payload["lead_times"]
    lines += [
        "",
        "## 3. 提前量（四层 vs 仅阈值）",
        "",
        f"- 有效提前量：**{lead['positive_leads']}** 项；"
        f"均值 **{num(lead['mean_hours'], 1)} h**，"
        f"中位数 **{num(lead['median_hours'], 1)} h**，"
        f"最大 **{num(lead['max_hours'], 1)} h**。",
        f"- 阈值层漏报、提前量无定义：**{len(lead['threshold_missed_items'])}** 项。",
        "",
        "| 故障 | 设备 | 指标 | 四层首次 | 阈值层首次 | 提前量(h) | 备注 |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in lead["rows"]:
        lines.append(
            f"| {row['event_id']} | {row['device_id']} | {row['metric']} | "
            f"{row['four_layer_first'] or '—'} | {row['threshold_first'] or '—'} | "
            f"{num(row['lead_hours'], 1)} | {row['note']} |"
        )

    lines += [
        "",
        "## 4. 分故障类型检出",
        "",
        "| 方法 | 故障类型 | 事件数 | 检出 | 主指标全覆盖 | 事件召回 | 平均检出滞后(h) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["methods"].items():
        for fault_type, stats in row["by_fault_type"].items():
            lines.append(
                f"| {name} | {fault_type} | {stats['n_events']} | {stats['detected']} | "
                f"{stats['fully_detected']} | {pct(stats['event_recall'])} | "
                f"{num(stats['mean_detection_lag_hours'], 1)} |"
            )

    lines += [
        "",
        "## 5. 复现",
        "",
        "```bash",
        "python scripts/gen_sensor_data.py   # 可选：重建仿真数据",
        "python scripts/eval_detection.py    # 生成本报告与 JSON",
        "```",
        "",
        "> 评测结果可能不理想。误报、漏报和重复报都保留在明细里，不做剔除。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="异常检测评测")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=RESULTS_DIR / "detection_eval.md")
    args = parser.parse_args()

    write_ground_truth(FAULT_TRUTH_JSON)
    data = load_sensor_data()
    registry = load_registry()

    results = run_methods(data, registry)

    summary: dict[str, dict] = {}
    methods_payload: dict[str, dict] = {}
    for name, row in results.items():
        findings = row["findings"]
        events = row["events"]
        points = row["points"]
        summary[name] = {
            "n_findings": findings["n_findings"],
            "finding_precision": findings["precision"],
            "finding_recall": findings["recall"],
            "finding_f1": findings["f1"],
            "point_precision": points["precision"],
            "point_recall": points["recall"],
            "point_f1": points["f1"],
            "event_recall": events["event_recall"],
            "full_event_recall": events["full_event_recall"],
            "primary_metric_recall": events["primary_metric_recall"],
            "detected_events": events["detected_events"],
            "fully_detected_events": events["fully_detected_events"],
        }
        methods_payload[name] = {
            "n_findings": findings["n_findings"],
            "finding": findings,
            "point": points,
            "events": events,
            "by_fault_type": row["by_fault_type"],
        }

    four = results["四层组合"]["scan"].findings
    threshold = results["阈值检测"]["scan"].findings
    added = added_findings(four, threshold, FAULT_EVENTS)
    leads = lead_times(four, threshold, FAULT_EVENTS)

    payload = {
        "config": build_config(),
        "summary": summary,
        "methods": methods_payload,
        "added": added,
        "lead_times": leads,
    }

    header = (
        f"{'方法':<12}{'Findings':>9}{'指标P':>8}{'指标R':>8}{'指标F1':>9}"
        f"{'点P':>8}{'点R':>8}{'点F1':>8}{'事件召回':>9}{'全覆盖':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, row in summary.items():
        print(
            f"{name:<12}{row['n_findings']:>9}"
            f"{row['finding_precision'] * 100:>7.1f}%{row['finding_recall'] * 100:>7.1f}%"
            f"{row['finding_f1'] * 100:>8.1f}%"
            f"{row['point_precision'] * 100:>7.1f}%{row['point_recall'] * 100:>7.1f}%"
            f"{row['point_f1'] * 100:>7.1f}%"
            f"{row['event_recall'] * 100:>8.1f}%{row['full_event_recall'] * 100:>7.1f}%"
        )

    print(f"\n四层新增 Finding：{added['added_findings']} 条，"
          f"真值证实 {added['validated_by_ground_truth']} 条，"
          f"未证实 {added['unvalidated']} 条。")
    print(
        f"提前量：有效 {leads['positive_leads']} 项，"
        f"均值 {num(leads['mean_hours'], 1)} h，"
        f"中位数 {num(leads['median_hours'], 1)} h，"
        f"最大 {num(leads['max_hours'], 1)} h；"
        f"阈值层漏报 {len(leads['threshold_missed_items'])} 项。"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(payload), encoding="utf-8")
    print(f"\nJSON  ：{args.output}")
    print(f"报告  ：{args.report}")


if __name__ == "__main__":
    main()
