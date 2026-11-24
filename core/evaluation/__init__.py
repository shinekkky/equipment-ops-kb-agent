"""评测与实验：故障真值、检测/检索/Agent 的自动评分。"""

from .ground_truth import FAULT_EVENTS, FaultEvent, ground_truth_dicts

__all__ = ["FAULT_EVENTS", "FaultEvent", "ground_truth_dicts"]
