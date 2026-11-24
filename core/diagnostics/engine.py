"""诊断引擎：载入数据 → 跑四层检测 → 合并成"一条问题一条结论"。

合并为什么必要
------------
四层各自独立，同一起故障往往被多层同时命中。例如 P-102 第 15 天的温度尖峰，
阈值层报"越 85 ℃ 报警"，Z-score 层报"偏离基线 11σ"——这是**同一件事**。
分开列会让运维以为设备有两处毛病，报告的可信度当场就没了。

因此按 (设备, 指标) 归并：严重度取最高，起始时间取最早，并把每层的判断
作为证据链保留下来。合并反而产出了报告里最有价值的一个数字——**提前量**：
趋势层在第 21 天就发现 P-101 在劣化，阈值层直到第 28 天才越线报警，
中间 7 天就是可以排产处置的窗口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import DEVICE_REGISTRY_CSV, SENSOR_CSV
from .detectors import Anomaly, detect_device, retrieval_hint
from .thresholds import ALARM, ATTENTION, NORMAL, SEVERITY_ORDER, TRIP, WARN

NUMERIC_METRICS = [
    "温度_℃", "压力_MPa", "振动_mm_s", "转速_rpm", "电流_A",
    "CPU利用率_%", "内存利用率_%", "端口丢包率_%",
]


# ---------------------------------------------------------------- 载入


def load_sensor_data(path: Path | str | None = None) -> pd.DataFrame:
    """读取传感器时序数据。时间列解析为 datetime，按设备与时间排序。"""
    path = Path(path) if path else SENSOR_CSV
    if not path.exists():
        raise FileNotFoundError(f"未找到时序数据：{path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "时间" not in df.columns:
        raise ValueError(f"数据缺少「时间」列：{path}")

    df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
    df = df.dropna(subset=["时间"])

    # 故障代码列应为字符串；空值统一成空串，避免后续把 NaN 当成一个"代码"
    if "故障代码" in df.columns:
        df["故障代码"] = df["故障代码"].fillna("").astype(str)

    return df.sort_values(["设备ID", "时间"]).reset_index(drop=True)


def load_registry(path: Path | str | None = None) -> pd.DataFrame:
    """读取设备台账。台账提供安装位置与责任人，报告里要写清楚"谁去处理"。"""
    path = Path(path) if path else DEVICE_REGISTRY_CSV
    if not path.exists():
        raise FileNotFoundError(f"未找到设备台账：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


# ---------------------------------------------------------------- 合并结果


@dataclass
class Finding:
    """合并后的单条诊断结论：一台设备上的一个指标问题。"""

    device_id: str
    device_name: str
    device_type: str
    model: str
    location: str
    owner: str
    metric: str
    metric_label: str
    unit: str
    severity: str
    anomalies: list[Anomaly]
    fault_codes: list[str] = field(default_factory=list)

    # ---- 由 anomalies 推导的派生字段，在 __post_init__ 里算好

    @property
    def layers(self) -> list[str]:
        seen: list[str] = []
        for a in sorted(self.anomalies, key=lambda x: SEVERITY_ORDER[x.severity], reverse=True):
            if a.layer_label not in seen:
                seen.append(a.layer_label)
        return seen

    @property
    def start(self) -> pd.Timestamp:
        return min(a.start for a in self.anomalies)

    @property
    def end(self) -> pd.Timestamp:
        return max(a.end for a in self.anomalies)

    @property
    def lead_time_hours(self) -> float | None:
        """相对"越限"提前了多少小时被发现。未越限或只有阈值层时返回 None。

        这是整份报告最有说服力的数字：隐患不是没出现，是以前没人提前看见。
        """
        alarmed = [a for a in self.anomalies if a.layer == "threshold"]
        early = [a for a in self.anomalies if a.layer != "threshold"]
        if not alarmed or not early:
            return None
        lead = (min(a.start for a in alarmed) - min(a.start for a in early)).total_seconds() / 3600.0
        return lead if lead > 0 else None

    @property
    def is_alarm(self) -> bool:
        """是否已越过手册的报警/停机线（区别于"注意"级隐患）。"""
        return SEVERITY_ORDER[self.severity] >= SEVERITY_ORDER[ALARM]

    @property
    def peak_text(self) -> str:
        cand = [a for a in self.anomalies if a.layer == "threshold"] or self.anomalies
        a = max(cand, key=lambda x: SEVERITY_ORDER[x.severity])
        if not self.unit:
            return f"联合偏离 {a.peak:.1f}σ"
        return f"{a.peak:.2f}{self.unit}"

    def evidence_lines(self) -> list[str]:
        """各层的判断按严重度排列，供报告展开成证据链。"""
        return [
            f"**{a.layer_label}**：{a.detail}"
            for a in sorted(self.anomalies, key=lambda x: SEVERITY_ORDER[x.severity], reverse=True)
        ]

    def to_dict(self) -> dict:
        return {
            "设备ID": self.device_id,
            "设备名称": self.device_name,
            "设备类型": self.device_type,
            "设备型号": self.model,
            "安装位置": self.location,
            "责任人": self.owner,
            "监测指标": self.metric_label,
            "严重度": self.severity,
            "起始时间": self.start.strftime("%Y-%m-%d %H:%M"),
            "结束时间": self.end.strftime("%Y-%m-%d %H:%M"),
            "峰值": self.peak_text,
            "检出层": "、".join(self.layers),
            # 无提前量时给 None 而不是空字符串：混用浮点与字符串会让
            # Streamlit 底层的 pyarrow 推断不出列类型，整列转换失败。
            "提前量(小时)": round(self.lead_time_hours, 1) if self.lead_time_hours else None,
            "故障代码": "/".join(self.fault_codes),
        }


@dataclass
class ScanResult:
    """一次全量扫描的结果。"""

    findings: list[Finding]
    devices: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    rows: int
    data_path: Path

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    def by_severity(self) -> dict[str, int]:
        counts = {TRIP: 0, ALARM: 0, WARN: 0, ATTENTION: 0}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    @property
    def alarmed_devices(self) -> list[str]:
        return sorted({f.device_id for f in self.findings if f.is_alarm})

    @property
    def hidden_risks(self) -> list[Finding]:
        """全部由统计层发现、尚未越限的隐患——本项目最想展示的那一类。"""
        return [f for f in self.findings if not f.is_alarm]

    def device_summary(self) -> list[dict]:
        rows: list[dict] = []
        for _, dev in self.devices.iterrows():
            own = [f for f in self.findings if f.device_id == dev["设备ID"]]
            worst = max((f.severity for f in own), key=lambda s: SEVERITY_ORDER[s],
                        default=NORMAL)
            rows.append({
                "设备ID": dev["设备ID"],
                "设备名称": dev["设备名称"],
                "设备类型": dev["设备类型"],
                "安装位置": dev["安装位置"],
                "责任人": dev["责任人"],
                "最高严重度": worst,
                "问题数": len(own),
                "已越限": sum(1 for f in own if f.is_alarm),
            })
        return sorted(rows, key=lambda r: (-SEVERITY_ORDER[r["最高严重度"]], r["设备ID"]))


def _consolidate(anomalies: list[Anomaly], device: dict) -> list[Finding]:
    """按 (设备, 指标) 归并同一物理问题的多层发现。"""
    grouped: dict[str, list[Anomaly]] = {}
    for a in anomalies:
        grouped.setdefault(a.metric, []).append(a)

    findings: list[Finding] = []
    for metric, group in grouped.items():
        worst = max(group, key=lambda a: SEVERITY_ORDER[a.severity])
        findings.append(Finding(
            device_id=worst.device_id,
            device_name=worst.device_name,
            device_type=worst.device_type,
            model=worst.model,
            location=worst.location,
            owner=device.get("责任人", ""),
            metric=metric,
            metric_label=worst.metric_label,
            unit=worst.unit,
            severity=worst.severity,
            anomalies=sorted(group, key=lambda a: SEVERITY_ORDER[a.severity], reverse=True),
        ))
    return findings


def scan(
    data: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    *,
    data_path: Path | str | None = None,
) -> ScanResult:
    """跑一次全量诊断。不传参数则读取 data/ 下的内置数据。"""
    resolved = Path(data_path) if data_path else SENSOR_CSV
    df = load_sensor_data(resolved) if data is None else data
    reg = load_registry() if registry is None else registry

    findings: list[Finding] = []

    for _, device in reg.iterrows():
        sub = df[df["设备ID"] == device["设备ID"]]
        if sub.empty:
            continue
        sub = sub.sort_values("时间").reset_index(drop=True)
        raw = detect_device(sub, device.to_dict())
        if not raw:
            continue

        for finding in _consolidate(raw, device.to_dict()):
            # 把该时段内控制系统实际置位的故障码附上，与检测结论互为佐证
            window = sub[(sub["时间"] >= finding.start) & (sub["时间"] <= finding.end)]
            codes: list[str] = []
            for cell in window.get("故障代码", pd.Series(dtype=str)).tolist():
                for code in str(cell).split("/"):
                    code = code.strip()
                    if code and code not in codes:
                        codes.append(code)
            finding.fault_codes = sorted(codes)
            findings.append(finding)

    findings.sort(key=lambda f: (-SEVERITY_ORDER[f.severity], f.device_id, f.metric))

    return ScanResult(
        findings=findings,
        devices=reg,
        start=df["时间"].min(),
        end=df["时间"].max(),
        rows=len(df),
        data_path=resolved,
    )


def attach_retrieval_hints(finding: Finding) -> tuple[str, str]:
    """给合并后的结论生成检索查询，取该组里层级最严重那条的提示。"""
    return retrieval_hint(finding.anomalies[0])
