"""项目指标核对：把 README / 产品方案里的人工统计数字全部重算一遍。

核对原则
--------
1. **算术可复现**：每个数字都能由脚本从当前数据与代码重新算出。
2. **口径可解释**：同一个词在不同段落含义不同时（"已越限" vs "阈值盲区"），
   拆成不同指标分别核对，不用一个数字硬套两种含义。
3. **真值可校验**：凡是声称"早期隐患/提前量"的数字，必须能被注入故障真值
   证实；误报不能被包装成"阈值盲区里的隐患"。

输出：
  * experiments/results/metrics_audit.json
  * docs/指标核对表.md

用法：
    python scripts/audit_metrics.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (  # noqa: E402
    DEVICE_REGISTRY_CSV,
    MANUALS_DIR,
    RESULTS_DIR,
    SENSOR_CSV,
    TOP_K_DEFAULT,
)
from core.diagnostics import load_registry, load_sensor_data, scan  # noqa: E402
from core.evaluation.ground_truth import FAULT_EVENTS  # noqa: E402
from core.evaluation.matching import (  # noqa: E402
    event_metrics,
    finding_metrics,
    match_finding_to_event,
    point_metrics,
)
from core.kb.index import build_index, get_embedder  # noqa: E402
from core.report.builder import build_advices  # noqa: E402

DEFAULT_JSON = RESULTS_DIR / "metrics_audit.json"
DEFAULT_MD = Path(__file__).resolve().parent.parent / "docs" / "指标核对表.md"


def _stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "max": None, "min": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
        "min": min(values),
    }


def _rounded(stats: dict) -> dict:
    return {
        key: (round(value, 1) if isinstance(value, float) else value)
        for key, value in stats.items()
    }


def _validated(finding) -> bool:
    return any(
        match_finding_to_event(finding, event, metric)
        for event in FAULT_EVENTS
        for metric in event.primary_metrics + event.secondary_metrics
    )


def compute() -> dict:
    data = load_sensor_data()
    registry = load_registry()

    four = scan(data=data, registry=registry)
    threshold = scan(data=data, registry=registry, layers={"threshold"})

    findings = four.findings
    total = len(findings)
    alarmed = [f for f in findings if f.is_alarm]
    hidden = [f for f in findings if not f.is_alarm]
    # Finding.layers 返回的是中文标签（"阈值规则"），不能拿英文 key 去比。
    blind = [
        f for f in findings
        if "threshold" not in {a.layer for a in f.anomalies}
    ]

    internal_leads = [f.lead_time_hours for f in findings if f.lead_time_hours]
    internal_valid = [f for f in findings if f.lead_time_hours and _validated(f)]
    internal_invalid = [f for f in findings if f.lead_time_hours and not _validated(f)]

    # 真值校验后的提前量：逐主指标比较四层与仅阈值层的首次发现时间。
    from core.evaluation.matching import lead_times

    gt_leads = lead_times(findings, threshold.findings, FAULT_EVENTS)
    gt_lead_values = [
        row["lead_hours"] for row in gt_leads["rows"]
        if row["lead_hours"] is not None and row["lead_hours"] > 0
    ]

    four_events = event_metrics(findings, FAULT_EVENTS)
    thr_events = event_metrics(threshold.findings, FAULT_EVENTS)
    threshold_missed_events = [
        frow["event_id"]
        for frow, trow in zip(four_events["rows"], thr_events["rows"])
        if frow["detected"] and not trow["detected"]
    ]

    # 手册依据覆盖：使用与界面相同的混合检索配置。
    embedder = get_embedder()
    index = build_index(sorted(MANUALS_DIR.glob("*.md")), embedder=embedder)
    mode = "hybrid" if embedder is not None else "bm25"
    advices = build_advices(
        four, index, data, top_k=TOP_K_DEFAULT, with_charts=False, mode=mode
    )
    cited = sum(1 for a in advices if a.primary)

    blind_validated = [f for f in blind if _validated(f)]
    blind_invalid = [f for f in blind if not _validated(f)]
    hidden_validated = [f for f in hidden if _validated(f)]
    all_validated = [f for f in findings if _validated(f)]

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "sensor_csv": SENSOR_CSV.name,
            "registry_csv": DEVICE_REGISTRY_CSV.name,
            "retrieval_mode": mode,
            "embedding_model": "BAAI/bge-small-zh-v1.5" if embedder is not None else None,
        },
        "counts": {
            "total_findings": total,
            "alarmed": len(alarmed),
            "hidden": len(hidden),
            "devices": int(four.devices["设备ID"].nunique()),
            "device_types": sorted(four.devices["设备类型"].unique().tolist()),
            "threshold_blind_findings": len(blind),
            "threshold_blind_ratio": len(blind) / total if total else 0.0,
            "threshold_blind_validated": len(blind_validated),
            "threshold_blind_invalid": len(blind_invalid),
            "hidden_validated": len(hidden_validated),
            "all_validated": len(all_validated),
            "manual_cited": cited,
            "manual_coverage": cited / total if total else 0.0,
        },
        "internal_leads": {
            **_rounded(_stats(internal_leads)),
            "items": [
                {
                    "device_id": f.device_id,
                    "metric": f.metric,
                    "hours": round(f.lead_time_hours, 1),
                    "validated_by_ground_truth": _validated(f),
                }
                for f in findings if f.lead_time_hours
            ],
            "validated_n": len(internal_valid),
            "invalid_n": len(internal_invalid),
            "validated_hours": _rounded(
                _stats([f.lead_time_hours for f in internal_valid])
            ),
            "invalid_items": [
                {
                    "device_id": f.device_id,
                    "metric": f.metric,
                    "hours": round(f.lead_time_hours, 1),
                }
                for f in internal_invalid
            ],
        },
        "ground_truth_leads": {
            **_rounded(_stats(gt_lead_values)),
            "items": gt_leads["rows"],
            "threshold_missed_items": gt_leads["threshold_missed_items"],
        },
        "event_level": {
            "four_layer": {
                "detected": four_events["detected_events"],
                "fully_detected": four_events["fully_detected_events"],
                "n": four_events["n_events"],
                "recall": four_events["event_recall"],
            },
            "threshold_only": {
                "detected": thr_events["detected_events"],
                "fully_detected": thr_events["fully_detected_events"],
                "n": thr_events["n_events"],
                "recall": thr_events["event_recall"],
            },
            "four_layer_extra_events": threshold_missed_events,
            "rows": four_events["rows"],
        },
        "finding_level": {
            "four_layer": finding_metrics(findings, FAULT_EVENTS),
            "threshold_only": finding_metrics(threshold.findings, FAULT_EVENTS),
        },
        "point_level": {
            "four_layer": point_metrics(data, findings, FAULT_EVENTS),
            "threshold_only": point_metrics(data, threshold.findings, FAULT_EVENTS),
        },
        "threshold_blind_items": [
            {
                "device_id": f.device_id,
                "metric": f.metric,
                "severity": f.severity,
                "start": f.start.strftime("%Y-%m-%d %H:%M"),
                "end": f.end.strftime("%Y-%m-%d %H:%M"),
                "layers": f.layers,
                "validated": _validated(f),
            }
            for f in blind
        ],
    }


def build_claims(result: dict) -> list[dict]:
    c = result["counts"]
    internal = result["internal_leads"]
    gt = result["ground_truth_leads"]
    ev = result["event_level"]

    # ground_truth_leads 直接是真值校验后的统计
    return [
        {
            "claim": "检出结论 15 条",
            "measured": f"{c['total_findings']} 条",
            "verdict": "confirmed" if c["total_findings"] == 15 else "corrected",
            "note": "脚本重算结果。",
        },
        {
            "claim": "已越过手册报警线 4 条",
            "measured": f"{c['alarmed']} 条",
            "verdict": "confirmed" if c["alarmed"] == 4 else "corrected",
            "note": "口径：Finding.severity 达到『报警』或『停机』。",
        },
        {
            "claim": "未越限、属早期劣化隐患 11 条",
            "measured": f"{c['hidden']} 条（其中真值证实 {c['hidden_validated']} 条）",
            "verdict": "partially_supported",
            "note": "11 是算术结果（15-4），但『未越限』不等于『真实隐患』；"
                    f"其中包含检测噪声，真值证实的是 {c['hidden_validated']} 条。",
        },
        {
            "claim": "单点阈值盲区 6 条（40%）",
            "measured": f"{c['threshold_blind_findings']} 条"
                        f"（{c['threshold_blind_ratio'] * 100:.1f}%），"
                        f"其中真值证实 {c['threshold_blind_validated']} 条",
            "verdict": "confirmed_with_caveat",
            "note": "『不信赖阈值层』的 Finding 数确实是 6；但其中 2 条是误报，"
                    "不能把 6 条都说成早期劣化隐患。",
        },
        {
            "claim": "这 6 条全部是尚未越限的早期劣化",
            "measured": f"真值证实 {c['threshold_blind_validated']} 条；"
                        f"误报 {c['threshold_blind_invalid']} 条；"
                        f"仅阈值层完全漏掉的故障事件 {len(ev['four_layer_extra_events'])} 起",
            "verdict": "not_supported",
            "note": "6 条里含 P-101 出口压力、P-102 电机电流的噪声告警；"
                    "真正完全不会被阈值层发现的故障事件是 P-103 短时尖峰与 M-202 多变量漂移。",
        },
        {
            "claim": "有明确提前量 5 条，平均 109 小时，最长 336 小时",
            "measured": f"内部统计 {internal['n']} 条 / 均值 {internal['mean']} h / "
                        f"最长 {internal['max']} h；真值校验后 {gt['n']} 条 / "
                        f"均值 {gt['mean']} h / 中位数 {gt['median']} h / 最长 {gt['max']} h",
            "verdict": "corrected",
            "note": "336 h 来自 SW-01 内存利用率的早期误报（阈值层 03-24、Z-score 03-10，"
                    "都早于第 25 天的注入故障），不能算作提前发现。"
                    f"应改用真值校验后的 {gt['n']} 条 / 均值 {gt['mean']} h / "
                    f"中位数 {gt['median']} h / 最长 {gt['max']} h。",
        },
        {
            "claim": "如果只有单点阈值报警，15 条结论里有 6 条根本不会出现",
            "measured": f"四层 {ev['four_layer']['detected']}/{ev['four_layer']['n']} 起故障事件；"
                        f"仅阈值 {ev['threshold_only']['detected']}/{ev['threshold_only']['n']} 起；"
                        f"多发现 {len(ev['four_layer_extra_events'])} 起",
            "verdict": "corrected",
            "note": "『6 条 Finding 不含阈值层』成立；但更可辩护的表述是："
                    "故障事件级召回从 4/6 提升到 6/6，多发现 2 起故障事件。",
        },
        {
            "claim": "手册依据覆盖 15/15",
            "measured": f"{c['manual_cited']}/{c['total_findings']}"
                        f"（{c['manual_coverage'] * 100:.1f}%）",
            "verdict": "confirmed" if c["manual_cited"] == c["total_findings"] else "corrected",
            "note": f"检索模式：{result['config']['retrieval_mode']}。",
        },
        {
            "claim": "检出结论覆盖离心泵、旋转设备、网络交换机三类设备",
            "measured": "、".join(c["device_types"]),
            "verdict": "confirmed" if len(c["device_types"]) == 3 else "corrected",
            "note": "",
        },
    ]


def render_markdown(result: dict, claims: list[dict]) -> str:
    c = result["counts"]
    ev = result["event_level"]
    gt = result["ground_truth_leads"]
    fl = result["finding_level"]
    pl = result["point_level"]

    lines = [
        "# 项目指标核对表",
        "",
        f"> 生成于 {result['generated_at']}，数据 `{result['config']['sensor_csv']}`，"
        f"检索模式 `{result['config']['retrieval_mode']}`。",
        "> 本表由 `python scripts/audit_metrics.py` 自动生成，用于核对 README 与产品方案中的"
        "人工统计数字。",
        "> 表中『原表述』指**修正前**的版本；README 与产品方案已按第 5 节的建议口径更新，"
        "保留原始表述是为了留下修正依据。",
        "",
        "## 1. 原始表述核对",
        "",
        "| 原表述 | 实测 | 结论 | 说明 |",
        "|---|---|---|---|",
    ]
    label = {
        "confirmed": "通过",
        "confirmed_with_caveat": "通过（需注明口径）",
        "partially_supported": "部分成立",
        "corrected": "需修正",
        "not_supported": "不成立",
    }
    for row in claims:
        lines.append(
            f"| {row['claim']} | {row['measured']} | "
            f"{label.get(row['verdict'], row['verdict'])} | {row['note']} |"
        )

    lines += [
        "",
        "## 2. 检测口径明细",
        "",
        "| 口径 | 四层组合 | 仅阈值层 |",
        "|---|---:|---:|",
        f"| 故障事件检出 | {ev['four_layer']['detected']}/{ev['four_layer']['n']} | "
        f"{ev['threshold_only']['detected']}/{ev['threshold_only']['n']} |",
        f"| 主指标全覆盖 | {ev['four_layer']['fully_detected']}/{ev['four_layer']['n']} | "
        f"{ev['threshold_only']['fully_detected']}/{ev['threshold_only']['n']} |",
        f"| 指标事件级 F1 | {fl['four_layer']['f1'] * 100:.1f}% | "
        f"{fl['threshold_only']['f1'] * 100:.1f}% |",
        f"| 采样点级 F1 | {pl['four_layer']['f1'] * 100:.1f}% | "
        f"{pl['threshold_only']['f1'] * 100:.1f}% |",
        "",
        "## 3. 提前量明细（真值校验后）",
        "",
        "| 故障 | 设备 | 指标 | 四层首次 | 阈值层首次 | 提前量(h) | 备注 |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in gt["items"]:
        lines.append(
            f"| {row['event_id']} | {row['device_id']} | {row['metric']} | "
            f"{row['four_layer_first'] or '—'} | {row['threshold_first'] or '—'} | "
            f"{row['lead_hours'] if row['lead_hours'] is not None else '—'} | "
            f"{row['note']} |"
        )

    lines += [
        "",
        "## 4. 阈值盲区明细",
        "",
        "| 设备 | 指标 | 严重度 | 时段 | 检出层 | 真值证实 |",
        "|---|---|---|---|---|---|",
    ]
    for item in result["threshold_blind_items"]:
        lines.append(
            f"| {item['device_id']} | {item['metric']} | {item['severity']} | "
            f"{item['start']} ~ {item['end']} | {'、'.join(item['layers'])} | "
            f"{'是' if item['validated'] else '否'} |"
        )

    lines += [
        "",
        "## 5. 建议采用的口径",
        "",
        f"- 检出结论 **{c['total_findings']}** 条；按严重度统计：已越限 **{c['alarmed']}** 条，"
        f"未越限 **{c['hidden']}** 条。",
        f"- 故障事件级召回：四层 **{ev['four_layer']['detected']}/{ev['four_layer']['n']}**，"
        f"仅阈值 **{ev['threshold_only']['detected']}/{ev['threshold_only']['n']}**；"
        f"四层多发现 **{len(ev['four_layer_extra_events'])}** 起"
        f"（{'、'.join(ev['four_layer_extra_events']) or '无'}）。",
        f"- 真值校验后的提前量：**{gt['n']}** 项，均值 **{gt['mean']} h**，"
        f"中位数 **{gt['median']} h**，最大 **{gt['max']} h**。",
        f"- 阈值盲区 Finding **{c['threshold_blind_findings']}** 条，其中真值证实 "
        f"**{c['threshold_blind_validated']}** 条；不应把误报计入『早期隐患』。",
        f"- 手册依据覆盖 **{c['manual_cited']}/{c['total_findings']}**。",
        "",
        "> 结论：原 README 的『5 条提前量 / 平均 109 h / 最长 336 h』与"
        "『6 条全部是早期劣化』需要修正；其余计数类指标（15/4/11/6/40%/15-15）"
        "在注明口径后成立。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="项目指标核对")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    result = compute()
    claims = build_claims(result)
    result["claims"] = claims

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(result, claims), encoding="utf-8")

    c = result["counts"]
    print(f"检出 {c['total_findings']} 条；已越限 {c['alarmed']}；未越限 {c['hidden']}；"
          f"阈值盲区 {c['threshold_blind_findings']}（真值证实 {c['threshold_blind_validated']}）")
    print(f"内部提前量：{result['internal_leads']['n']} 条，均值 "
          f"{result['internal_leads']['mean']} h，最长 {result['internal_leads']['max']} h")
    print(f"真值校验提前量：{result['ground_truth_leads']['n']} 条，均值 "
          f"{result['ground_truth_leads']['mean']} h，"
          f"中位数 {result['ground_truth_leads']['median']} h，"
          f"最长 {result['ground_truth_leads']['max']} h")
    print(f"事件召回：四层 {result['event_level']['four_layer']['detected']}/"
          f"{result['event_level']['four_layer']['n']}，仅阈值 "
          f"{result['event_level']['threshold_only']['detected']}/"
          f"{result['event_level']['threshold_only']['n']}")
    print(f"\nJSON：{args.json}")
    print(f"核对表：{args.markdown}")


if __name__ == "__main__":
    main()
