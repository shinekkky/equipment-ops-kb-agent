"""评测脚本的输出结构测试（不重跑完整评测）。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_detection  # noqa: E402
import eval_retrieval  # noqa: E402


class RetrievalOutputTest(unittest.TestCase):
    def test_json_payload_is_serializable(self):
        cases = [
            {"query": "P103", "doc": "离心泵故障排查手册", "section_no": "2.2"},
        ]
        result = {
            "mode": "bm25", "n": 1,
            "hit": {"1": 1.0, "3": 1.0, "5": 1.0},
            "mrr": 1.0, "hit_counts": {"1": 1, "3": 1, "5": 1},
            "misses": [], "per_case": [],
        }
        payload = {"config": {"n_queries": len(cases)}, "summary": [result], "runs": [result]}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.assertIn("bm25", text)
        self.assertIn("config", text)


class DetectionReportTest(unittest.TestCase):
    def _payload(self):
        summary = {
            "四层组合": {
                "n_findings": 15,
                "finding_precision": 0.667,
                "finding_recall": 0.846,
                "finding_f1": 0.746,
                "point_precision": 0.930,
                "point_recall": 0.374,
                "point_f1": 0.533,
                "event_recall": 1.0,
                "full_event_recall": 1.0,
            },
        }
        return {
            "config": {
                "generated_at": "2026-11-24T00:00:00+08:00",
                "data": {"sensor_csv": "data.csv", "n_events": 6},
            },
            "summary": summary,
            "methods": {
                "四层组合": {
                    "by_fault_type": {
                        "缓慢趋势劣化": {
                            "n_events": 2, "detected": 2, "fully_detected": 2,
                            "event_recall": 1.0, "full_event_recall": 1.0,
                            "mean_detection_lag_hours": 36.0,
                        },
                    },
                },
            },
            "added": {
                "added_findings": 6,
                "validated_by_ground_truth": 4,
                "unvalidated": 2,
                "items": [{
                    "device_id": "P-103", "metric": "温度_℃",
                    "start": "2026-03-13 12:00", "end": "2026-03-13 12:00",
                    "layers": ["滚动 Z-score"], "validated": True,
                }],
            },
            "lead_times": {
                "positive_leads": 5,
                "mean_hours": 45.0,
                "median_hours": 15.0,
                "max_hours": 103.0,
                "threshold_missed_items": [],
                "rows": [{
                    "event_id": "E01", "device_id": "P-101", "metric": "温度_℃",
                    "four_layer_first": "2026-03-22 20:00",
                    "threshold_first": "2026-03-27 03:00",
                    "lead_hours": 103.0, "note": "",
                }],
            },
        }

    def test_markdown_contains_core_tables(self):
        text = eval_detection.render_markdown(self._payload())
        for expected in ("方法对比", "四层相对阈值层", "提前量", "分故障类型", "复现"):
            self.assertIn(expected, text)
        self.assertIn("100.0%", text)

    def test_json_serializable(self):
        text = json.dumps(self._payload(), ensure_ascii=False, default=str)
        self.assertIn("lead_times", text)


if __name__ == "__main__":
    unittest.main()
