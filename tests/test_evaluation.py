"""评测数据集与 Agent 自动评分测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from core.config import AGENT_EVAL_SET_JSONL
from core.evaluation.agent_scoring import (
    aggregate,
    load_dataset,
    score_case,
    validate_dataset,
)

TOOLS = {
    "query_manual", "list_devices", "get_device_findings",
    "get_metric_stats", "get_thresholds",
}


def perfect_record(case: dict) -> dict:
    tools = list(case.get("required_tools") or case.get("required_any")[:1])
    calls = []
    for tool in tools:
        args = dict(case.get("required_args", {}).get(tool, {}))
        ref = " ".join(
            str(v) for v in case.get("expected_numbers", []) + case.get("allowed_numbers", [])
        )
        calls.append({"tool_name": tool, "arguments": args, "result": f"参考数据 {ref}"})
    parts = (
        case.get("must_include", [])
        + case.get("must_include_any", [])[:1]
        + [str(v) for v in case.get("expected_numbers", [])]
        + [str(v) for v in case.get("allowed_numbers", [])]
        + ([case["expected_doc"]] if case.get("expected_doc") else [])
        + case.get("expected_sections", [])
        + [c["result"] for c in calls]
    )
    return {
        "question": case["question"],
        "final_answer": " ".join(parts),
        "tool_calls": calls,
        "error": None,
        "trace_id": "self-test",
    }


class DatasetTest(unittest.TestCase):
    def test_dataset_has_40_cases_and_expected_categories(self):
        cases = load_dataset(AGENT_EVAL_SET_JSONL)
        self.assertEqual(len(cases), 40)
        categories = {c["category"] for c in cases}
        for expected in (
            "设备状态查询", "故障手册检索", "异常详情查询", "多工具组合调用",
            "不存在的设备", "无相关手册内容", "工具执行失败", "精确引用检测数据",
        ):
            self.assertIn(expected, categories)

    def test_dataset_validation(self):
        cases = load_dataset(AGENT_EVAL_SET_JSONL)
        self.assertEqual(validate_dataset(cases, TOOLS), [])

    def test_unknown_tool_is_reported(self):
        cases = [{
            "id": "X1", "category": "x", "question": "q",
            "required_tools": ["not_a_tool"], "required_any": [],
            "forbidden_tools": [], "required_args": {},
        }]
        problems = validate_dataset(cases, TOOLS)
        self.assertTrue(any("not_a_tool" in p for p in problems))


class ScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {c["id"]: c for c in load_dataset(AGENT_EVAL_SET_JSONL)}

    def test_all_cases_pass_with_perfect_record(self):
        scores = [score_case(c, perfect_record(c)) for c in self.cases.values()]
        for score in scores:
            self.assertTrue(score["success"], f"{score['id']}: {score['failure_reasons']}")
        summary = aggregate(scores)
        self.assertEqual(summary["tool_selection_accuracy"], 1.0)
        self.assertEqual(summary["task_success_rate"], 1.0)
        self.assertEqual(summary["numeric_accuracy"], 1.0)
        self.assertEqual(summary["evidence_accuracy"], 1.0)

    def test_wrong_tool_fails_selection(self):
        case = self.cases["A03"]
        record = perfect_record(case)
        record["tool_calls"] = [{"tool_name": "list_devices", "arguments": {}, "result": ""}]
        score = score_case(case, record)
        self.assertFalse(score["tool_selection"]["ok"])
        self.assertIn("缺少工具", "；".join(score["failure_reasons"]))

    def test_wrong_argument_fails(self):
        case = self.cases["A18"]
        record = perfect_record(case)
        record["tool_calls"][0]["arguments"] = {"device_id": "P-101"}
        score = score_case(case, record)
        self.assertFalse(score["tool_args"]["ok"])

    def test_missing_required_text_fails_task(self):
        case = self.cases["A05"]
        record = perfect_record(case)
        record["final_answer"] = "温度到 85 度停机"
        score = score_case(case, record)
        self.assertFalse(score["task"]["ok"])
        self.assertIn("95", score["task"]["missing"])

    def test_missing_evidence_section_fails(self):
        case = self.cases["A05"]
        record = perfect_record(case)
        record["final_answer"] = "95℃ 停机"
        score = score_case(case, record)
        self.assertFalse(score["evidence"]["ok"])

    def test_unsupported_number_is_flagged(self):
        case = self.cases["A01"]
        record = perfect_record(case)
        record["final_answer"] += " 一共 9876 台设备"
        score = score_case(case, record)
        self.assertIn(9876.0, score["unsupported"]["unsupported"])

    def test_nonexistent_device_case_requires_negative_answer(self):
        case = self.cases["A27"]
        record = perfect_record(case)
        record["final_answer"] = "P-999 的温度是 63.2℃"
        score = score_case(case, record)
        self.assertFalse(score["task"]["ok"])


if __name__ == "__main__":
    unittest.main()
