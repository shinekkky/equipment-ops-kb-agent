"""检测规则、风险等级与故障真值一致性测试。"""

from __future__ import annotations

import unittest

from core.diagnostics import load_registry, load_sensor_data, scan
from core.diagnostics.thresholds import (
    ALARM,
    NORMAL,
    TRIP,
    WARN,
    band_for,
    bands_for,
)
from core.evaluation.ground_truth import DATA_END, DATA_START, FAULT_EVENTS
from core.evaluation.matching import event_metrics, finding_metrics


class ThresholdTest(unittest.TestCase):
    def test_pump_temperature_bands(self):
        band = band_for("离心泵", "CXB-200", "温度_℃")
        self.assertIsNotNone(band)
        self.assertEqual(band.evaluate(74.9), NORMAL)
        self.assertEqual(band.evaluate(75.0), NORMAL)   # 预警级从 >75 开始
        self.assertEqual(band.evaluate(80.0), WARN)
        self.assertEqual(band.evaluate(86.0), ALARM)
        self.assertEqual(band.evaluate(96.0), TRIP)
        self.assertEqual(band.evaluate(float("nan")), NORMAL)

    def test_current_threshold_depends_on_model(self):
        cx200 = band_for("离心泵", "CXB-200", "电流_A")
        cx350 = band_for("离心泵", "CXB-350", "电流_A")
        self.assertEqual(cx200.alarm, 52.0)
        self.assertEqual(cx350.alarm, 102.0)
        self.assertEqual(cx350.evaluate(90.0), NORMAL)

    def test_network_thresholds(self):
        bands = {b.metric: b for b in bands_for("网络交换机", "S5720-28X")}
        self.assertEqual(bands["CPU利用率_%"].alarm, 80.0)
        self.assertEqual(bands["端口丢包率_%"].alarm, 1.0)
        self.assertEqual(bands["内存利用率_%"].alarm, 85.0)


class ScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_sensor_data()
        cls.registry = load_registry()
        cls.full = scan(data=cls.data, registry=cls.registry)

    def test_full_scan_returns_findings(self):
        self.assertGreater(len(self.full.findings), 0)
        self.assertEqual(self.full.rows, 5040)
        self.assertEqual(self.full.devices["设备ID"].nunique(), 7)

    def test_all_four_layers_are_exercised(self):
        layers = {a.layer for f in self.full.findings for a in f.anomalies}
        self.assertEqual(
            layers, {"threshold", "zscore", "trend", "multivariate"}
        )

    def test_threshold_only_scan_has_no_statistical_layers(self):
        result = scan(data=self.data, registry=self.registry, layers={"threshold"})
        layers = {a.layer for f in result.findings for a in f.anomalies}
        self.assertEqual(layers, {"threshold"})

    def test_multivariate_only_scan(self):
        result = scan(data=self.data, registry=self.registry, layers={"multivariate"})
        self.assertTrue(result.findings)
        self.assertTrue(all(
            a.layer == "multivariate"
            for f in result.findings for a in f.anomalies
        ))

    def test_unknown_layer_raises(self):
        with self.assertRaises(ValueError):
            scan(data=self.data, registry=self.registry, layers={"nope"})


class GroundTruthTest(unittest.TestCase):
    def test_events_are_within_data(self):
        data = load_sensor_data()
        devices = set(data["设备ID"])
        for event in FAULT_EVENTS:
            self.assertIn(event.device_id, devices)
            self.assertGreaterEqual(event.start, DATA_START)
            self.assertLessEqual(event.end, DATA_END)
            for metric in event.primary_metrics + event.secondary_metrics:
                self.assertIn(metric, data.columns, f"{event.event_id} {metric}")

    def test_four_layers_recall_all_events(self):
        data = load_sensor_data()
        registry = load_registry()
        result = scan(data=data, registry=registry)
        metrics = event_metrics(result.findings, FAULT_EVENTS)
        self.assertEqual(metrics["detected_events"], 6)
        self.assertEqual(metrics["event_recall"], 1.0)

    def test_threshold_only_misses_statistical_events(self):
        data = load_sensor_data()
        registry = load_registry()
        result = scan(data=data, registry=registry, layers={"threshold"})
        metrics = event_metrics(result.findings, FAULT_EVENTS)
        self.assertLess(metrics["detected_events"], 6)
        self.assertIn("E03", [
            row["event_id"] for row in metrics["rows"] if not row["detected"]
        ])

    def test_finding_metrics_are_bounded(self):
        data = load_sensor_data()
        registry = load_registry()
        result = scan(data=data, registry=registry)
        metrics = finding_metrics(result.findings, FAULT_EVENTS)
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertLessEqual(metrics["precision"], 1.0)
        self.assertGreaterEqual(metrics["recall"], 0.0)
        self.assertLessEqual(metrics["recall"], 1.0)
        self.assertGreaterEqual(metrics["f1"], 0.0)
        self.assertLessEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
