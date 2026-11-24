"""设备异常检测：四层检测（阈值 / Z-score / 趋势 / 多变量）与结果合并。"""

from .engine import (
    Finding,
    ScanResult,
    attach_retrieval_hints,
    load_registry,
    load_sensor_data,
    scan,
)
from .thresholds import ALARM, ATTENTION, NORMAL, SEVERITY_ORDER, TRIP, WARN

__all__ = [
    "Finding",
    "ScanResult",
    "scan",
    "load_sensor_data",
    "load_registry",
    "attach_retrieval_hints",
    "SEVERITY_ORDER",
    "NORMAL",
    "ATTENTION",
    "WARN",
    "ALARM",
    "TRIP",
]
