"""报告配图：指标趋势图与全厂概览图。

图要能直接回答"什么时候开始不对、离红线还有多远"，所以每张趋势图都必须同时
画出三样东西：实测曲线、手册规定的报警/预警线、以及检测到异常的时段。
只画曲线的图在现场没有用——看不出严重程度。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from ..config import setup_chinese_font
from ..diagnostics.thresholds import Band

SEVERITY_COLORS = {
    "停机": "#b42318",
    "报警": "#d92d20",
    "预警": "#dc6803",
    "注意": "#b54708",
    "正常": "#067647",
}


def _new_axes(width: float = 9.0, height: float = 3.0):
    setup_chinese_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    return fig, ax


def metric_trend(
    series: pd.DataFrame,
    *,
    metric_label: str,
    unit: str,
    band: Band | None = None,
    device_label: str = "",
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    severity: str = "注意",
):
    """画单个指标的趋势图，标注手册阈值线与异常时段。

    series 需含「时间」列与 band.metric 列。
    """
    fig, ax = _new_axes()
    metric = band.metric if band else None
    if metric is None or metric not in series.columns:
        return fig

    times = pd.to_datetime(series["时间"])
    values = series[metric].astype(float)

    # 原始采样点太密会把噪声画成一片毛刺，叠加 24 h 滑动平均看清趋势走向
    ax.plot(times, values, color="#98a2b3", linewidth=0.7, alpha=0.8, label="逐时实测")
    ax.plot(times, values.rolling(24, min_periods=6).mean(),
            color="#1570ef", linewidth=1.6, label="24 h 滑动平均")

    if band is not None:
        for edge, text, color in (
            (band.warn, "预警线", "#f79009"),
            (band.alarm, "报警线", "#f04438"),
            (band.trip, "停机线", "#912018"),
        ):
            if edge is None:
                continue
            ax.axhline(edge, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
            ax.annotate(
                f"{text} {edge:g}{unit}",
                xy=(times.iloc[-1], edge), xytext=(4, 2),
                textcoords="offset points", fontsize=8, color=color, va="bottom",
            )

    for start, end in (windows or []):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=SEVERITY_COLORS.get(severity, "#dc6803"), alpha=0.13, linewidth=0)

    ax.set_ylabel(f"{metric_label}（{unit}）" if unit else metric_label, fontsize=9)
    if device_label:
        ax.set_title(device_label, fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def severity_bar(counts: dict[str, int]):
    """全厂结论的严重度分布。用颜色而非仅长度传达紧急程度。"""
    fig, ax = _new_axes(width=5.2, height=2.6)
    order = ["停机", "报警", "预警", "注意"]
    labels = [s for s in order if counts.get(s)]
    values = [counts[s] for s in labels]

    if not labels:
        ax.text(0.5, 0.5, "未检出异常", ha="center", va="center", fontsize=11, color="#667085")
        ax.axis("off")
        fig.tight_layout()
        return fig

    bars = ax.barh(labels, values, color=[SEVERITY_COLORS[s] for s in labels], height=0.55)
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("结论条数", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, max(values) * 1.25)
    fig.tight_layout()
    return fig


def severity_timeline(
    records: pd.DataFrame,
    *,
    start=None,
    end=None,
):
    """按发现日期堆叠展示新增风险，回答“风险是在变多还是变少”。

    records 需含「日期」「严重度」两列。日期按天聚合，没有异常的日子补 0，
    否则柱状图会把两个相隔多日的结论画成相邻两天，趋势会被误读。
    """
    fig, ax = _new_axes(width=8.0, height=2.7)
    order = ["停机", "报警", "预警", "注意"]

    if records is None or records.empty:
        ax.text(0.5, 0.5, "当前区间未检出异常", ha="center", va="center",
                fontsize=11, color="#667085")
        ax.axis("off")
        fig.tight_layout()
        return fig

    dates = pd.to_datetime(records["日期"]).dt.normalize()
    first = pd.Timestamp(start).normalize() if start is not None else dates.min()
    last = pd.Timestamp(end).normalize() if end is not None else dates.max()
    full_range = pd.date_range(first, last, freq="D")

    pivot = (
        pd.DataFrame({"日期": dates, "严重度": records["严重度"].astype(str)})
        .groupby(["日期", "严重度"]).size().unstack(fill_value=0)
        .reindex(full_range, fill_value=0)
        .reindex(columns=order, fill_value=0)
    )

    bottom = np.zeros(len(pivot))
    for severity in order:
        values = pivot[severity].to_numpy()
        if values.sum() == 0:
            continue
        ax.bar(pivot.index, values, bottom=bottom, width=0.78,
               color=SEVERITY_COLORS[severity], label=severity)
        bottom += values

    ax.set_ylabel("新增结论（条）", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=4)
    fig.tight_layout()
    return fig


def save(fig, path: Path) -> Path:
    """存图。PDF 导出与 Streamlit 下载共用同一条渲染路径，保证所见即所得。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return path
