"""Agent 工具集：把知识库检索与诊断引擎包装成模型可调用的工具。

工具存在的意义是**让模型不必自己算**。模型直接读原始时序去估计趋势、算均值，
结果会错得很像真的；而调用这里的工具，拿到的是确定性代码算出来的数字。
模型只负责判断"该查什么、结果意味着什么"。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import TOP_K_DEFAULT
from ..diagnostics.engine import ScanResult
from ..diagnostics.thresholds import (
    MANUAL_FOR_TYPE,
    bands_for,
    metric_for_label,
)

MAX_TOOL_CHARS = 2400  # 单个工具返回给模型的字符上限，防止一次检索吃满上下文


@dataclass
class AgentContext:
    """一次会话共享的状态：诊断结果、原始数据、知识库索引。"""

    scan: ScanResult
    data: pd.DataFrame
    index: object | None = None
    retrieval_mode: str = "hybrid"
    top_k: int = TOP_K_DEFAULT

    def device_row(self, device_id: str) -> pd.Series | None:
        hit = self.scan.devices[self.scan.devices["设备ID"] == device_id]
        return hit.iloc[0] if not hit.empty else None


def build_tools(ctx: AgentContext) -> list:
    """构造工具集合。工具通过闭包捕获 ctx，无需模型传入。"""
    from langchain_core.tools import tool

    @tool
    def query_manual(question: str, device_type: str = "") -> str:
        """检索运维手册，回答"该怎么排查/怎么处置/阈值是多少"这类问题。

        question: 自然语言问题，例如"离心泵轴承温度过高怎么处理"。
        device_type: 可选，限定手册范围，取值 离心泵 / 旋转设备 / 网络交换机。
                     问某类设备的问题时建议填写，能显著提高命中准确率。
        """
        if ctx.index is None:
            return "知识库未加载，无法检索手册。"

        doc_titles = None
        if device_type:
            doc = MANUAL_FOR_TYPE.get(device_type.strip())
            # 没给出有效的设备类型就全库检索，而不是报错——模型可能只是填了个
            # 不在枚举里的说法，硬失败会让整轮对话断掉。
            doc_titles = [doc] if doc else None

        hits = ctx.index.search(
            question, top_k=ctx.top_k, mode=ctx.retrieval_mode, doc_titles=doc_titles
        )
        if not hits:
            return f"未检索到与「{question}」相关的章节，请换一种问法或确认手册是否覆盖该主题。"

        blocks: list[str] = []
        used = 0
        for hit in hits:
            text = hit.as_tool_text()
            if used + len(text) > MAX_TOOL_CHARS and blocks:
                break
            blocks.append(text)
            used += len(text)
        return "\n\n".join(blocks)

    @tool
    def list_devices() -> str:
        """列出纳入监测的全部设备及其编号、型号、安装位置与责任人。"""
        lines = ["设备台账："]
        for _, d in ctx.scan.devices.iterrows():
            n = sum(1 for f in ctx.scan.findings if f.device_id == d["设备ID"])
            lines.append(
                f"- {d['设备ID']} {d['设备名称']}｜{d['设备型号']}｜{d['设备类型']}｜"
                f"{d['安装位置']}｜责任人 {d['责任人']}｜检出问题 {n} 条"
            )
        return "\n".join(lines)

    @tool
    def get_device_findings(device_id: str) -> str:
        """查询指定设备的全部异常结论（含检出层、严重度、峰值与判据）。

        device_id: 设备编号，如 P-101、M-201、SW-01。
        """
        device = ctx.device_row(device_id.strip())
        if device is None:
            return f"未找到设备 {device_id}。可用编号见 list_devices 的结果。"

        own = [f for f in ctx.scan.findings if f.device_id == device_id.strip()]
        if not own:
            return (
                f"{device_id} {device['设备名称']} 在本次监测区间内未检出异常。"
                f"（型号 {device['设备型号']}，安装于 {device['安装位置']}）"
            )

        lines = [
            f"{device_id} {device['设备名称']}（{device['设备型号']}，"
            f"{device['设备类型']}，{device['安装位置']}，责任人 {device['责任人']}）："
        ]
        for f in own:
            lines.append(f"\n【{f.severity}】{f.metric_label}")
            lines.append(f"  时段 {f.start:%Y-%m-%d %H:%M} ~ {f.end:%Y-%m-%d %H:%M}，峰值 {f.peak_text}")
            if f.lead_time_hours:
                lines.append(f"  早于越限 {f.lead_time_hours:.0f} 小时被发现")
            if f.fault_codes:
                lines.append(f"  控制系统故障代码：{'/'.join(f.fault_codes)}")
            for ev in f.evidence_lines():
                lines.append(f"  - {ev}")
        return "\n".join(lines)

    @tool
    def get_metric_stats(device_id: str, metric: str, start: str = "", end: str = "") -> str:
        """统计某台设备某个指标在指定时段内的真实数值（均值、极值、变化量）。

        device_id: 设备编号，如 P-101。
        metric: 指标名，如 轴承温度 / 振动烈度 / 出口压力 / 电机电流 / CPU利用率。
        start, end: 可选，格式 YYYY-MM-DD 或 YYYY-MM-DD HH:MM；留空则统计全区间。
        需要"温度最近涨了多少""振动峰值多大"这类精确数字时使用本工具，不要自己估算。
        """
        device = ctx.device_row(device_id.strip())
        if device is None:
            return f"未找到设备 {device_id}。"

        column = metric_for_label(metric.strip()) or metric.strip()
        series = ctx.data[ctx.data["设备ID"] == device_id.strip()].sort_values("时间")
        if column not in series.columns or series[column].notna().sum() == 0:
            available = [b.label for b in bands_for(device["设备类型"], device["设备型号"])]
            return f"{device_id} 没有「{metric}」这一监测项。可用的指标：{'、'.join(available)}"

        series = series.dropna(subset=[column])
        if start:
            series = series[series["时间"] >= pd.to_datetime(start, errors="coerce")]
        if end:
            series = series[series["时间"] <= pd.to_datetime(end, errors="coerce")]
        if series.empty:
            return "指定时段内没有数据，请检查时间格式（YYYY-MM-DD）。"

        values = series[column].astype(float)
        head, tail = float(values.iloc[0]), float(values.iloc[-1])
        direction = "上升" if tail >= head else "下降"
        return (
            f"{device_id} {metric}（{series['时间'].iloc[0]:%Y-%m-%d %H:%M} ~ "
            f"{series['时间'].iloc[-1]:%Y-%m-%d %H:%M}，{len(series)} 个采样点）：\n"
            f"  均值 {values.mean():.3f}｜最小 {values.min():.3f}｜最大 {values.max():.3f}\n"
            f"  标准差 {values.std():.3f}｜首值 {head:.3f}｜末值 {tail:.3f}"
            f"（{direction} {abs(tail - head):.3f}）"
        )

    @tool
    def get_thresholds(device_type: str, model: str = "") -> str:
        """查询某类设备的手册报警阈值。

        device_type: 离心泵 / 旋转设备 / 网络交换机。
        model: 可选，型号如 CXB-200、CXB-350（泵的电流阈值与型号强相关，建议填写）。
        """
        bands = bands_for(device_type.strip(), model.strip())
        if not bands:
            return f"未找到「{device_type}」对应的阈值表。可选：离心泵 / 旋转设备 / 网络交换机。"

        lines = [f"{device_type} 手册报警阈值" + (f"（{model}）" if model else "") + "："]
        for b in bands:
            parts = []
            if b.warn is not None:
                parts.append(f"预警 {b.warn:g}")
            if b.alarm is not None:
                parts.append(f"报警 {b.alarm:g}")
            if b.trip is not None:
                parts.append(f"停机 {b.trip:g}")
            lines.append(f"- {b.label}（{b.unit}）：{'，'.join(parts)}　出处：{b.source}")
        return "\n".join(lines)

    return [query_manual, list_devices, get_device_findings, get_metric_stats, get_thresholds]
