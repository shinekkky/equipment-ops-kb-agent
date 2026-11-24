"""检测结果与故障真值的匹配规则。

评测最容易出问题的地方，是把 "Finding 条数 ÷ 注入故障条数" 当成召回率。
本项目严格区分三种口径，并全部写进结果文件：

1. **故障事件级**：一次注入故障是否被至少一个相关指标发现（primary 指标）。
2. **指标事件级**：一个 (故障, 主/次指标) 是否被至少一个 Finding 覆盖并匹配。
3. **采样点级**：每个 (设备, 指标, 时刻) 的异常/正常二分类。

匹配不是"数量相除"，而是按 **设备 + 指标 + 时间窗** 逐项判断：

* 指标必须相同；多变量 Finding 的 ``metric`` 形如
  ``温度_℃+振动_mm_s+电流_A``，拆开后覆盖其中每个指标。
* Finding 的起始时间必须落在故障时间窗的容差范围内（容差取
  ``min(24h, max(2h, 故障时长))``），且与故障窗有交叠。
  这条规则排除了"故障发生前 20 多天的误报恰好一直延续到故障窗"被算作
  成功发现的情况。
* 趋势类故障的终点是数据结束时刻，检测只要在容差内开始且持续到窗口内即可。

注意：这些规则是为了让评测**可审计**，不是为了把数字做好看。误报就是误报。
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable

import pandas as pd

from ..diagnostics.engine import Finding
from .ground_truth import FaultEvent


@dataclass(frozen=True)
class Match:
    event_id: str
    device_id: str
    metric: str
    finding: Finding
    start: pd.Timestamp
    end: pd.Timestamp
    is_primary: bool


def metric_set(metric: str) -> set[str]:
    return {part for part in metric.split("+") if part}


def metric_interval(finding: Finding, metric: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Finding 在某个具体指标上的时间范围。多变量 Finding 覆盖所有组成指标。"""
    spans = [
        (a.start, a.end)
        for a in finding.anomalies
        if metric in metric_set(a.metric)
    ]
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def event_tolerance_hours(event: FaultEvent, metric: str) -> float:
    start, end = event.window_for(metric)
    duration = (end - start).total_seconds() / 3600.0
    return min(24.0, max(2.0, duration))


def match_finding_to_event(
    finding: Finding,
    event: FaultEvent,
    metric: str,
    tol_hours: float | None = None,
) -> Match | None:
    """逐条 Anomaly 判断，而不是拿 Finding 的整体起止时间。

    同一个 Finding 可能同时包含"故障前的噪声"和"故障期间的真正异常"
    （例如 P-101 振动：Z-score 从 03-08 就有噪声，趋势层 03-24 才捕捉到
    真实劣化）。用整体起点会把后面的有效检出一起判死；逐条判断既保留
    误报（它会作为另一条 Anomaly 计入采样点级 FP），也不会漏掉有效检出。
    """
    if finding.device_id != event.device_id:
        return None

    win_start, win_end = event.window_for(metric)
    tol = pd.Timedelta(
        hours=event_tolerance_hours(event, metric) if tol_hours is None else tol_hours
    )

    candidates: list[Match] = []
    for anomaly in finding.anomalies:
        if metric not in metric_set(anomaly.metric):
            continue
        start, end = anomaly.start, anomaly.end
        # 起始点必须在容差内；且不能早早结束在故障发生之前。
        if start < win_start - tol:
            continue
        if start > win_end + tol:
            continue
        if end < win_start:
            continue
        candidates.append(Match(
            event_id=event.event_id,
            device_id=event.device_id,
            metric=metric,
            finding=finding,
            start=start,
            end=end,
            is_primary=metric in event.primary_metrics,
        ))

    return min(candidates, key=lambda m: m.start) if candidates else None


def match_event(
    findings: Iterable[Finding],
    event: FaultEvent,
    metrics: Iterable[str] | None = None,
    tol_hours: float | None = None,
) -> dict[str, list[Match]]:
    """返回 {指标: [匹配到的 Finding]}。每个指标至少一条才算该指标被发现。"""
    wanted = list(metrics or (event.primary_metrics + event.secondary_metrics))
    out: dict[str, list[Match]] = {m: [] for m in wanted}
    for finding in findings:
        for metric in wanted:
            match = match_finding_to_event(finding, event, metric, tol_hours)
            if match is not None:
                out[metric].append(match)
    return out


def first_detection(matches: list[Match]) -> pd.Timestamp | None:
    return min((m.start for m in matches), default=None)


def event_metrics(
    findings: list[Finding],
    events: Iterable[FaultEvent],
    tol_hours: float | None = None,
) -> dict:
    """故障事件级：一次注入故障是否被发现，主指标是否全部覆盖。"""
    rows: list[dict] = []
    for event in events:
        matched = match_event(findings, event, event.primary_metrics, tol_hours)
        found = {m: bool(v) for m, v in matched.items()}
        firsts = [first_detection(v) for v in matched.values() if v]
        event_window_start = min(event.window_for(m)[0] for m in event.primary_metrics)
        rows.append({
            "event_id": event.event_id,
            "device_id": event.device_id,
            "fault_type": event.fault_type,
            "start": event.start.strftime("%Y-%m-%d %H:%M"),
            "end": event.end.strftime("%Y-%m-%d %H:%M"),
            "primary_metrics": list(event.primary_metrics),
            "metric_found": found,
            "detected": any(found.values()),
            "fully_detected": all(found.values()),
            "primary_metric_recall": (
                sum(found.values()) / len(found) if found else 0.0
            ),
            "first_detection": (
                min(firsts).strftime("%Y-%m-%d %H:%M") if firsts else None
            ),
            "detection_lag_hours": (
                round((min(firsts) - event_window_start).total_seconds() / 3600.0, 1)
                if firsts else None
            ),
        })

    n = len(rows)
    return {
        "n_events": n,
        "detected_events": sum(r["detected"] for r in rows),
        "fully_detected_events": sum(r["fully_detected"] for r in rows),
        "event_recall": (sum(r["detected"] for r in rows) / n) if n else 0.0,
        "full_event_recall": (
            sum(r["fully_detected"] for r in rows) / n if n else 0.0
        ),
        "primary_metric_recall": (
            mean(r["primary_metric_recall"] for r in rows) if n else 0.0
        ),
        "rows": rows,
    }


def finding_metrics(
    findings: list[Finding],
    events: Iterable[FaultEvent],
    tol_hours: float | None = None,
) -> dict:
    """指标事件级：以 (故障, 指标) 为召回单位，以 Finding 为精确率单位。

    一条 Finding 只要匹配任一标注指标即算 TP；同一个标注指标被多条 Finding
    覆盖只算一次召回，但会重复计入精确率的分子分母，因此能暴露"同一问题被
    多层重复报"的情况。
    """
    events = list(events)
    gt_items: set[tuple[str, str]] = set()
    for event in events:
        for metric in event.primary_metrics + event.secondary_metrics:
            gt_items.add((event.event_id, metric))

    matched_items: set[tuple[str, str]] = set()
    tp_findings = 0
    duplicates: dict[tuple[str, str], int] = {}

    for finding in findings:
        hit_items: set[tuple[str, str]] = set()
        for event in events:
            all_metrics = event.primary_metrics + event.secondary_metrics
            for metric in all_metrics:
                if match_finding_to_event(finding, event, metric, tol_hours):
                    hit_items.add((event.event_id, metric))
        if hit_items:
            tp_findings += 1
            for item in hit_items:
                duplicates[item] = duplicates.get(item, 0) + 1
        matched_items |= hit_items

    tp = tp_findings
    fp = len(findings) - tp
    fn = len(gt_items - matched_items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = len(matched_items) / len(gt_items) if gt_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_findings": len(findings),
        "tp_findings": tp,
        "fp_findings": fp,
        "gt_items": len(gt_items),
        "matched_items": len(matched_items),
        "missed_items": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "duplicate_items": {
            f"{eid}:{metric}": count
            for (eid, metric), count in duplicates.items() if count > 1
        },
    }


def _metric_columns(data: pd.DataFrame) -> list[str]:
    excluded = {
        "时间", "设备ID", "设备名称", "设备型号", "设备类型", "故障代码", "环境温度_℃",
    }
    return [
        c for c in data.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(data[c])
    ]


def point_metrics(
    data: pd.DataFrame,
    findings: list[Finding],
    events: Iterable[FaultEvent],
) -> dict:
    """采样点级：每个 (设备, 指标, 时刻) 是否被判为异常。

    真值只覆盖注入故障的主/次指标与时间窗；检测器在其他指标、其他时段的
    任何命中都计为 FP。这是最严格、也最容易暴露误报的口径。
    """
    events = list(events)
    times = sorted(pd.to_datetime(data["时间"].unique()))
    metrics = _metric_columns(data)

    truth_intervals: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for event in events:
        for metric in event.primary_metrics + event.secondary_metrics:
            truth_intervals.setdefault((event.device_id, metric), []).append(
                event.window_for(metric)
            )

    pred_intervals: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for finding in findings:
        for anomaly in finding.anomalies:
            for metric in metric_set(anomaly.metric):
                pred_intervals.setdefault((finding.device_id, metric), []).append(
                    (anomaly.start, anomaly.end)
                )

    tp = fp = fn = tn = 0
    per_metric: dict[str, dict[str, int]] = {}
    devices = sorted(data["设备ID"].unique())

    for device in devices:
        for metric in metrics:
            key = (device, metric)
            truths = truth_intervals.get(key, [])
            preds = pred_intervals.get(key, [])
            # 该设备没有这个指标的数据时跳过（避免把 NaN 列算成海量 TN）
            series = data.loc[data["设备ID"] == device, metric]
            if series.notna().sum() == 0:
                continue

            m = per_metric.setdefault(metric, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            for ts in times:
                is_true = any(s <= ts <= e for s, e in truths)
                is_pred = any(s <= ts <= e for s, e in preds)
                if is_true and is_pred:
                    tp += 1; m["tp"] += 1
                elif is_pred and not is_true:
                    fp += 1; m["fp"] += 1
                elif is_true and not is_pred:
                    fn += 1; m["fn"] += 1
                else:
                    tn += 1; m["tn"] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "positive_rate": ((tp + fn) / (tp + fp + fn + tn)) if (tp + fp + fn + tn) else 0.0,
        "per_metric": per_metric,
    }


def lead_times(
    four_layer: list[Finding],
    threshold_only: list[Finding],
    events: Iterable[FaultEvent],
) -> dict:
    """四层相对仅阈值层的提前量（逐主指标比较）。

    只有两边都匹配到同一 (故障, 主指标) 时才计算提前量；仅阈值层漏报的
    故障单独列出，不能把"阈值层根本没报"算成提前量为 0。
    """
    rows: list[dict] = []
    threshold_missed: list[dict] = []
    for event in events:
        four_matches = match_event(four_layer, event, event.primary_metrics)
        thr_matches = match_event(threshold_only, event, event.primary_metrics)
        for metric in event.primary_metrics:
            t_four = first_detection(four_matches.get(metric, []))
            t_thr = first_detection(thr_matches.get(metric, []))
            row = {
                "event_id": event.event_id,
                "device_id": event.device_id,
                "metric": metric,
                "four_layer_first": t_four.strftime("%Y-%m-%d %H:%M") if t_four else None,
                "threshold_first": t_thr.strftime("%Y-%m-%d %H:%M") if t_thr else None,
                "lead_hours": None,
                "note": "",
            }
            if t_four and t_thr:
                lead = (t_thr - t_four).total_seconds() / 3600.0
                row["lead_hours"] = round(lead, 1)
                if lead < 0:
                    row["note"] = "四层晚于阈值层"
            elif t_four and not t_thr:
                row["note"] = "阈值层漏报，提前量无定义"
                threshold_missed.append(row)
            elif t_thr and not t_four:
                row["note"] = "四层未匹配到该指标"
            rows.append(row)

    values = [r["lead_hours"] for r in rows if r["lead_hours"] is not None and r["lead_hours"] > 0]
    return {
        "rows": rows,
        "threshold_missed_items": threshold_missed,
        "positive_leads": len(values),
        "mean_hours": mean(values) if values else None,
        "median_hours": median(values) if values else None,
        "max_hours": max(values) if values else None,
        "min_hours": min(values) if values else None,
    }


def per_fault_type(event_result: dict) -> dict:
    """把故障事件级结果按故障类型汇总。"""
    grouped: dict[str, list[dict]] = {}
    for row in event_result["rows"]:
        grouped.setdefault(row["fault_type"], []).append(row)
    out: dict[str, dict] = {}
    for fault_type, rows in grouped.items():
        n = len(rows)
        out[fault_type] = {
            "n_events": n,
            "detected": sum(r["detected"] for r in rows),
            "fully_detected": sum(r["fully_detected"] for r in rows),
            "event_recall": sum(r["detected"] for r in rows) / n,
            "full_event_recall": sum(r["fully_detected"] for r in rows) / n,
            "mean_detection_lag_hours": (
                mean(r["detection_lag_hours"] for r in rows if r["detection_lag_hours"] is not None)
                if any(r["detection_lag_hours"] is not None for r in rows) else None
            ),
        }
    return out


def added_findings(
    four_layer: list[Finding],
    threshold_only: list[Finding],
    events: Iterable[FaultEvent],
) -> dict:
    """四层相对阈值层新增的 Finding 与其中被真值证实的部分。

    "新增"定义为：同一设备上，没有任何时间重叠、且覆盖同一指标的阈值 Finding。
    这比"条数相减"严格，因为多层会把同一个问题合并成一条，不能直接做差。
    """
    threshold_spans: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for finding in threshold_only:
        for anomaly in finding.anomalies:
            for metric in metric_set(anomaly.metric):
                threshold_spans.setdefault((finding.device_id, metric), []).append(
                    (anomaly.start, anomaly.end)
                )

    added: list[Finding] = []
    for finding in four_layer:
        is_new = True
        for anomaly in finding.anomalies:
            for metric in metric_set(anomaly.metric):
                for s, e in threshold_spans.get((finding.device_id, metric), []):
                    if anomaly.start <= e and anomaly.end >= s:
                        is_new = False
                        break
                if not is_new:
                    break
            if not is_new:
                break
        if is_new:
            added.append(finding)

    events = list(events)
    validated = 0
    for finding in added:
        if any(
            match_finding_to_event(finding, event, metric)
            for event in events
            for metric in event.primary_metrics + event.secondary_metrics
        ):
            validated += 1

    return {
        "added_findings": len(added),
        "validated_by_ground_truth": validated,
        "unvalidated": len(added) - validated,
        "items": [
            {
                "device_id": f.device_id,
                "metric": f.metric,
                "start": f.start.strftime("%Y-%m-%d %H:%M"),
                "end": f.end.strftime("%Y-%m-%d %H:%M"),
                "layers": f.layers,
                "validated": any(
                    match_finding_to_event(f, event, metric)
                    for event in events
                    for metric in event.primary_metrics + event.secondary_metrics
                ),
            }
            for f in added
        ],
    }
