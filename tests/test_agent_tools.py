"""Agent 工具参数校验与异常路径测试。"""

from __future__ import annotations

import unittest

from core.agent import AgentContext, build_tools
from core.diagnostics import load_registry, load_sensor_data, scan


class ToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_sensor_data()
        registry = load_registry()
        result = scan(data=data, registry=registry)
        cls.ctx = AgentContext(scan=result, data=data, index=None)
        cls.tools = {t.name: t for t in build_tools(cls.ctx)}

    def test_unknown_device(self):
        out = self.tools["get_device_findings"].invoke({"device_id": "P-999"})
        self.assertIn("未找到", out)

    def test_unknown_metric_lists_available(self):
        out = self.tools["get_metric_stats"].invoke(
            {"device_id": "P-101", "metric": "轴瓦间隙"}
        )
        self.assertIn("没有", out)
        self.assertIn("轴承温度", out)

    def test_metric_on_wrong_device(self):
        out = self.tools["get_metric_stats"].invoke(
            {"device_id": "SW-01", "metric": "出口压力"}
        )
        self.assertIn("没有", out)

    def test_invalid_date(self):
        out = self.tools["get_metric_stats"].invoke(
            {"device_id": "P-101", "metric": "轴承温度", "start": "2026-13-45"}
        )
        self.assertIn("没有数据", out)

    def test_unknown_threshold_type(self):
        out = self.tools["get_thresholds"].invoke({"device_type": "燃气轮机"})
        self.assertIn("未找到", out)

    def test_metric_stats_has_real_numbers(self):
        out = self.tools["get_metric_stats"].invoke(
            {"device_id": "P-101", "metric": "轴承温度"}
        )
        self.assertIn("均值", out)
        self.assertIn("81.89", out)

    def test_list_devices(self):
        out = self.tools["list_devices"].invoke({})
        for device in ("P-101", "SW-01", "M-202"):
            self.assertIn(device, out)

    def test_query_manual_without_index(self):
        out = self.tools["query_manual"].invoke({"question": "泵温度高怎么办"})
        self.assertIn("知识库未加载", out)


if __name__ == "__main__":
    unittest.main()
