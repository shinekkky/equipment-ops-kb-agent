"""仿真数据的故障注入真值。

真值与 ``scripts/gen_sensor_data.py`` 中硬编码的注入逻辑一一对应。之所以
单独成文件而不是从生成器里"猜"，是因为检测评测的匹配规则必须建立在
**明确、可审计的标注**上：设备、故障类型、时间窗、主/次指标都要写清楚。

约定
----
* ``primary_metrics``：该故障直接注入的指标，参与召回与时间点级评测。
* ``secondary_metrics``：注入时伴随变化、但不应单独代表该故障的指标。
  例如 M-201 振动爬升会因摩擦生热带动温度，温度是伴随现象而非主故障。
* ``start/end``：故障注入的作用区间。趋势类故障的终点是数据结束时刻。
* 真值描述的是"数据里确实被改坏了什么"，不是"检测器应该输出什么"。

如果修改了数据生成脚本，必须同步更新这里，并运行
``tests/test_detection.py`` 中的真值一致性检查。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

DATA_START = pd.Timestamp("2026-03-01 00:00:00")
DATA_END = pd.Timestamp("2026-03-30 23:00:00")


@dataclass(frozen=True)
class FaultEvent:
    event_id: str
    device_id: str
    fault_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    description: str
    expected_layers: tuple[str, ...]
    # 个别伴随指标的作用窗口与主故障不同（例如 M-201 温度比振动晚 2 天起升）。
    # 不把这种差异写清楚，采样点级评测就会把"注入之前"算成漏报。
    metric_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = field(
        default_factory=dict, compare=False, hash=False
    )

    def window_for(self, metric: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.metric_windows.get(metric, (self.start, self.end))


def _at(day: int, hour: int = 0) -> pd.Timestamp:
    return DATA_START + pd.Timedelta(days=day, hours=hour)


FAULT_EVENTS: tuple[FaultEvent, ...] = (
    FaultEvent(
        event_id="E01",
        device_id="P-101",
        fault_type="缓慢趋势劣化",
        start=_at(20),
        end=DATA_END,
        primary_metrics=("温度_℃",),
        secondary_metrics=("振动_mm_s",),
        description="轴承温度自第 20 天起每天约 +1.6 ℃，30 天末未越 85 ℃ 报警线。",
        expected_layers=("trend",),
    ),
    FaultEvent(
        event_id="E02",
        device_id="P-102",
        fault_type="突发尖峰",
        start=_at(15, 6),
        end=_at(15, 16),
        primary_metrics=("温度_℃",),
        secondary_metrics=("振动_mm_s",),
        description="温度在第 15 天 06:00—16:00 冲高至约 92 ℃，越过 85 ℃ 报警线。",
        expected_layers=("threshold", "zscore"),
    ),
    FaultEvent(
        event_id="E03",
        device_id="P-103",
        fault_type="短时尖峰（未越限）",
        start=_at(12, 10),
        end=_at(12, 13),
        primary_metrics=("温度_℃",),
        secondary_metrics=(),
        description="温度在第 12 天 10:00—13:00 冲高 +7.5 ℃，全程未越 75 ℃ 预警线。",
        expected_layers=("zscore",),
    ),
    FaultEvent(
        event_id="E04",
        device_id="M-201",
        fault_type="缓慢趋势劣化",
        start=_at(22),
        end=DATA_END,
        primary_metrics=("振动_mm_s",),
        secondary_metrics=("温度_℃",),
        description="振动自第 22 天起每天约 +0.75 mm/s，第 27 天前后越过报警线；温度因摩擦生热自第 24 天起伴随上升。",
        expected_layers=("trend", "threshold"),
        metric_windows={"温度_℃": (_at(24), DATA_END)},
    ),
    FaultEvent(
        event_id="E05",
        device_id="M-202",
        fault_type="多变量隐性漂移",
        start=_at(16),
        end=DATA_END,
        primary_metrics=("温度_℃", "振动_mm_s", "电流_A"),
        secondary_metrics=(),
        description="温度/振动/电流自第 16 天起同步缓慢偏移，单指标均未越限。",
        expected_layers=("multivariate",),
    ),
    FaultEvent(
        event_id="E06",
        device_id="SW-01",
        fault_type="资源耗尽趋势",
        start=_at(25),
        end=DATA_END,
        primary_metrics=("CPU利用率_%", "端口丢包率_%"),
        secondary_metrics=("内存利用率_%",),
        description="CPU 与端口丢包率自第 25 天起同步攀升，先后越过报警线。",
        expected_layers=("threshold", "trend"),
    ),
)


def ground_truth_dicts() -> list[dict]:
    rows: list[dict] = []
    for event in FAULT_EVENTS:
        row = asdict(event)
        row["start"] = event.start.strftime("%Y-%m-%d %H:%M")
        row["end"] = event.end.strftime("%Y-%m-%d %H:%M")
        row["metric_windows"] = {
            metric: [start.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M")]
            for metric, (start, end) in event.metric_windows.items()
        }
        rows.append(row)
    return rows


def write_ground_truth(path: Path) -> None:
    """把真值写盘，便于人工核对与外部工具复用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ground_truth_dicts(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
