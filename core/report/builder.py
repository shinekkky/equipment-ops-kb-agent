"""诊断报告组装：把检测结论、手册依据、趋势图拼成一份可交付的 Markdown。

报告里每个结论都必须带两样东西
------------------------------
1. **数据依据**——什么时候、哪个指标、越了多少、由哪一层检出；
2. **手册依据**——哪本手册哪一节、原文怎么写的、处置步骤是什么。

只有第 1 条，运维会问"凭什么这么判"；只有第 2 条，会问"我这台到底有没有事"。
两条都在，报告才能直接进检修工单。手册依据由知识库检索得到，检索查询由
检测层按"设备类型 + 指标 + 症状"自动拼出，不需要人工选章节。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import REPORT_OUT_DIR, TOP_K_DEFAULT
from ..diagnostics.detectors import retrieval_hint
from ..diagnostics.engine import Finding, ScanResult
from ..diagnostics.thresholds import ALARM, ATTENTION, SEVERITY_ORDER, TRIP, WARN, band_for
from . import charts

MARKDOWN_ONLY_NOTE = "> 本报告由系统自动生成，结论需经运维人员现场核实后执行。"


@dataclass
class Advice:
    """一条结论 + 它的手册依据。"""

    finding: Finding
    query: str
    retrieved: list = field(default_factory=list)   # list[Retrieved]
    chart: Path | None = None

    @property
    def primary(self):
        """首选手册章节。检索排名第一即最相关，报告正文直接引用它。"""
        return self.retrieved[0] if self.retrieved else None

    @property
    def citation(self) -> str:
        return self.primary.citation if self.primary else "（未检索到对应章节）"

    @property
    def steps_text(self) -> str:
        return self.primary.parent.text if self.primary else ""

    def other_citations(self) -> list[str]:
        return [r.citation for r in self.retrieved[1:]]


@dataclass
class ReportBundle:
    scan: ScanResult
    advices: list[Advice]
    narrative: str = ""
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def charts_dir(self) -> Path:
        return REPORT_OUT_DIR / "charts"


# ---------------------------------------------------------------- 检索依据


def build_advices(
    scan: ScanResult,
    index,
    data: pd.DataFrame | None = None,
    *,
    top_k: int = TOP_K_DEFAULT,
    with_charts: bool = True,
    mode: str = "hybrid",
) -> list[Advice]:
    """为每条结论检索手册章节，并可选地渲染趋势图。

    index 为 core.kb.index.KnowledgeIndex；为 None 时跳过检索，
    报告仍会生成，只是缺少手册依据——离线环境下降级而非报错。
    """
    advices: list[Advice] = []
    for finding in scan.findings:
        query, doc = retrieval_hint(finding.anomalies[0])
        # 设备类型已知，就限定只在该类型对应的手册里检索，避免"轴承温度"
        # 命中的却是网络设备手册里讲光模块温度那一节。
        retrieved = index.search(
            query, top_k=top_k, mode=mode, doc_titles=[doc] if doc else None
        ) if index else []

        chart = None
        if with_charts and data is not None:
            chart = _render_finding_chart(scan, finding, data)

        advices.append(Advice(finding=finding, query=query, retrieved=retrieved, chart=chart))
    return advices


def _render_finding_chart(scan: ScanResult, finding: Finding, data: pd.DataFrame) -> Path | None:
    """给一条结论画趋势图，画在它所属指标上。"""
    if "+" in finding.metric:      # 多参数联合结论没有单一曲线可画
        return None

    series = data[data["设备ID"] == finding.device_id].sort_values("时间")
    if series.empty:
        return None

    band = band_for(finding.device_type, finding.model, finding.metric)
    fig = charts.metric_trend(
        series,
        metric_label=finding.metric_label,
        unit=finding.unit,
        band=band,
        device_label=f"{finding.device_id} {finding.device_name} · {finding.metric_label}",
        windows=[(finding.start, finding.end)],
        severity=finding.severity,
    )
    # 指标名里的 ℃ 与 % 在跨平台打包、下载、URL 编码时都可能出问题，落盘前清掉；
    # 清完可能留下连续下划线（"温度_℃" → "温度_"），一并压掉
    safe_metric = re.sub(r"[\\/:*?\"<>|%℃]", "", finding.metric)
    safe_metric = re.sub(r"_{2,}", "_", safe_metric).strip("_")
    name = f"{finding.device_id}_{safe_metric}_{finding.severity}.png"
    try:
        return charts.save(fig, REPORT_OUT_DIR / "charts" / name)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------- Markdown


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "（无）"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _severity_tag(severity: str) -> str:
    """用文字标注严重度，不依赖颜色——黑白打印的工单上颜色会全部丢失。"""
    return f"【{severity}】"


def render_markdown(
    bundle: ReportBundle,
    *,
    narrative: str = "",
    include_charts: bool = True,
    chart_prefix: str = "",
) -> str:
    """渲染完整报告。narrative 为模型生成的总体分析，为空则只输出检测结论。"""
    scan = bundle.scan
    counts = scan.by_severity()
    parts: list[str] = []

    # ---------------------------------------------------------- 封面
    parts.append("# 工业设备运维与故障诊断报告")
    parts.append("")
    parts.append(
        f"- **数据区间**：{scan.start:%Y-%m-%d %H:%M} 至 {scan.end:%Y-%m-%d %H:%M}"
        f"（{scan.hours:.0f} 小时，{scan.rows} 条采样记录）"
    )
    parts.append(f"- **监测设备**：{len(scan.devices)} 台，覆盖离心泵、旋转设备、网络交换机三类")
    parts.append(f"- **报告生成**：{bundle.generated_at:%Y-%m-%d %H:%M}")
    parts.append(f"- **数据来源**：`{scan.data_path.name}`（仿真数据）")
    parts.append("")

    # ---------------------------------------------------------- 一、总体结论
    parts.append("## 一、总体结论")
    parts.append("")
    total = len(scan.findings)
    alarmed = [f for f in scan.findings if f.is_alarm]
    hidden = scan.hidden_risks
    leads = [f.lead_time_hours for f in scan.findings if f.lead_time_hours]
    max_lead = max(leads) if leads else 0.0

    parts.append(
        f"本次共检出 **{total}** 条异常结论，涉及 **{len({f.device_id for f in scan.findings})}** 台设备；"
        f"其中已越过手册报警线 **{len(alarmed)}** 条，"
        f"尚未越限但已呈现劣化征兆 **{len(hidden)}** 条。"
    )
    parts.append("")
    parts.append(_table(
        ["严重度", "含义", "条数"],
        [
            [TRIP, "必须停机，严禁带病运行", counts.get(TRIP, 0)],
            [ALARM, "需立即处置", counts.get(ALARM, 0)],
            [WARN, "加强监视并安排检查", counts.get(WARN, 0)],
            [ATTENTION, "未越限的早期隐患，建议缩短巡检周期", counts.get(ATTENTION, 0)],
        ],
    ))
    parts.append("")
    if max_lead:
        parts.append(
            f"> **提前量**：其中 {len(leads)} 条结论由统计层先于阈值层发现，"
            f"最长提前 **{max_lead:.0f} 小时**（约 {max_lead / 24:.1f} 天）。"
            "这段窗口就是可以安排检修、而不必被迫停机的余量。"
        )
        parts.append("")

    if narrative:
        parts.append("### 模型综合分析")
        parts.append("")
        parts.append(narrative.strip())
        parts.append("")

    # ---------------------------------------------------------- 二、异常汇总
    parts.append("## 二、异常结论汇总")
    parts.append("")
    parts.append(_table(
        ["#", "设备编号", "设备名称", "监测指标", "严重度", "起始时间", "峰值", "检出层", "提前量(h)", "故障代码"],
        [
            [
                i, f.device_id, f.device_name, f.metric_label,
                _severity_tag(f.severity), f"{f.start:%m-%d %H:%M}", f.peak_text,
                "、".join(f.layers),
                f"{f.lead_time_hours:.0f}" if f.lead_time_hours else "—",
                "/".join(f.fault_codes) or "—",
            ]
            for i, f in enumerate(scan.findings, start=1)
        ],
    ))
    parts.append("")

    # ---------------------------------------------------------- 三、逐条诊断
    parts.append("## 三、逐条诊断与处置建议")
    parts.append("")
    for i, advice in enumerate(bundle.advices, start=1):
        finding = advice.finding
        parts.append(
            f"### 3.{i} {_severity_tag(finding.severity)} "
            f"{finding.device_id} {finding.device_name} —— {finding.metric_label}"
        )
        parts.append("")

        parts.append(_table(
            ["设备型号", "设备类型", "安装位置", "责任人", "异常时段", "峰值"],
            [[
                finding.model, finding.device_type, finding.location, finding.owner or "—",
                f"{finding.start:%Y-%m-%d %H:%M} — {finding.end:%Y-%m-%d %H:%M}",
                finding.peak_text,
            ]],
        ))
        parts.append("")

        if include_charts and advice.chart is not None:
            # 链接写成相对 REPORT_OUT_DIR 的路径。只写文件名的话图在 charts/
            # 子目录里，PDF 渲染时按 base_dir 拼出来找不到文件——图会整批丢失，
            # 而且因为找不到就跳过，连个报错都没有。
            try:
                rel = Path(advice.chart).relative_to(REPORT_OUT_DIR).as_posix()
            except ValueError:
                rel = Path(advice.chart).name
            parts.append(f"![{finding.device_id} {finding.metric_label} 趋势]({chart_prefix}{rel})")
            parts.append("")

        parts.append("**检测依据**")
        parts.append("")
        for line in finding.evidence_lines():
            parts.append(f"- {line}")
        parts.append("")

        parts.append("**手册依据**")
        parts.append("")
        parts.append(f"检索查询：`{advice.query}`")
        parts.append("")
        if advice.primary:
            parts.append(f"命中章节：**{advice.citation}**")
            parts.append("")
            parts.append(advice.steps_text.strip())
            parts.append("")
            others = advice.other_citations()
            if others:
                parts.append("相关章节：" + "；".join(others))
                parts.append("")
        else:
            parts.append("未在知识库中检索到对应章节，请人工查阅手册。")
            parts.append("")

    # ---------------------------------------------------------- 四、排期建议
    parts.append("## 四、处置排期建议")
    parts.append("")
    parts.append(_table(
        ["优先级", "设备编号", "设备名称", "责任人", "待处置问题", "建议动作"],
        [
            [
                "P0" if f.severity == TRIP else "P1" if f.severity == ALARM
                else "P2" if f.severity == WARN else "P3",
                f.device_id, f.device_name, f.owner or "—",
                f.metric_label,
                _action_for(f.severity),
            ]
            for f in scan.findings
        ],
    ))
    parts.append("")

    # ---------------------------------------------------------- 五、方法说明
    parts.append("## 五、检测方法与口径说明")
    parts.append("")
    parts.append(
        "本报告采用四层递进的异常检测：\n\n"
        "1. **阈值规则层**——逐点套用各手册第 2 章的报警阈值表，判定已越限的显性异常；\n"
        "2. **滚动 Z-score 层**——以 7 天滚动中位数为基线，捕捉越限之前、甚至始终不越限的\n"
        "   短时突变（阈值层对此完全无感）；\n"
        "3. **趋势劣化层**——对 5 天窗口做线性回归，捕捉缓慢爬升，并外推预计何时触及报警线；\n"
        "4. **多变量联合层**——用 IsolationForest 对多参数的 24 小时滚动均值打分，\n"
        "   捕捉各指标单独看都正常、但同步偏移的系统性劣化。\n"
    )
    parts.append("")
    parts.append(
        "**阈值口径**：全部取自随附手册，未自行设定。离心泵见《离心泵故障排查手册》2.1，\n"
        "旋转设备振动分区见《旋转设备振动诊断与润滑管理规范》2.2、温升见 4.3，\n"
        "网络设备见《网络设备告警处置指南》2.2。\n"
    )
    parts.append("")
    parts.append(
        "**基线假设**：统计层以序列开头 10 天为正常基线期，假定该期间工况正常。\n"
        "若该期间本身存在异常，或设备刚完成检修，应重新指定基线窗口后复算。\n"
    )
    parts.append("")
    parts.append(MARKDOWN_ONLY_NOTE)
    parts.append("")

    return "\n".join(parts)


def _action_for(severity: str) -> str:
    return {
        TRIP: "立即停机检修，严禁继续运行",
        ALARM: "24 小时内安排处置，处置前加强监视",
        WARN: "纳入本周计划检修窗口，加密巡视频次",
        ATTENTION: "缩短巡检周期至每周，建立趋势跟踪",
    }.get(severity, "持续观察")


def save_markdown(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def default_report_paths(stamp: datetime | None = None) -> tuple[Path, Path]:
    """报告输出路径：(Markdown, PDF)。文件名带时间戳，避免连续导出互相覆盖。"""
    stamp = stamp or datetime.now()
    tag = stamp.strftime("%Y%m%d_%H%M")
    return (
        REPORT_OUT_DIR / f"设备诊断报告_{tag}.md",
        REPORT_OUT_DIR / f"设备诊断报告_{tag}.pdf",
    )
