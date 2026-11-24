"""Agent 多轮工具调用与 Trace 持久化测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from core.agent.runtime import run_tool_loop
from core.agent.trace_store import export_trace, list_traces, load_trace


class FakeTool:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def invoke(self, args):
        return self.fn(args)


class FakeLLM:
    model_name = "fake-model"

    def __init__(self, responses, bind_error: Exception | None = None):
        self.responses = list(responses)
        self.bind_error = bind_error

    def bind_tools(self, tools):
        if self.bind_error:
            raise self.bind_error
        return self

    def invoke(self, messages):
        return self.responses.pop(0)


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "traces.jsonl"
        self.old = os.environ.get("AGENT_TRACE_LOG")
        os.environ["AGENT_TRACE_LOG"] = str(self.log)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("AGENT_TRACE_LOG", None)
        else:
            os.environ["AGENT_TRACE_LOG"] = self.old
        self.tmp.cleanup()

    def test_single_tool_call_and_trace(self):
        llm = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "echo", "args": {"text": "hello"}, "id": "c1",
            }]),
            AIMessage(content="最终回答", usage_metadata={
                "input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
            }),
        ])
        tools = [FakeTool("echo", lambda args: f"echo:{args['text']}")]
        run = run_tool_loop(
            llm, tools, [HumanMessage(content="测试")],
            question="测试", model_name="fake-model", persist=True,
        )

        self.assertEqual(run.text, "最终回答")
        self.assertEqual(len(run.tool_calls), 1)
        self.assertEqual(run.tool_calls[0].step, 1)
        self.assertEqual(run.tool_calls[0].status, "success")
        self.assertEqual(run.tool_calls[0].result, "echo:hello")
        self.assertGreaterEqual(run.tool_calls[0].duration_ms, 0)
        self.assertEqual(run.rounds, 2)
        self.assertEqual(run.total_tokens, 5)
        self.assertTrue(run.trace_id)

        record = load_trace(run.trace_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["tool_calls"][0]["tool_name"], "echo")
        self.assertEqual(record["metadata"], {})

        out = Path(self.tmp.name) / "one.json"
        export_trace(run.trace_id, out)
        self.assertIn(run.trace_id, out.read_text(encoding="utf-8"))

    def test_tool_exception_is_recorded_and_loop_continues(self):
        def boom(_):
            raise RuntimeError("坏了")

        llm = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "boom", "args": {}, "id": "c1",
            }]),
            AIMessage(content="已根据失败结果调整"),
        ])
        run = run_tool_loop(
            llm, [FakeTool("boom", boom)], [HumanMessage(content="q")],
            question="q", persist=False,
        )
        self.assertEqual(run.tool_calls[0].status, "error")
        self.assertIn("RuntimeError", run.tool_calls[0].error)
        self.assertIn("工具执行失败", run.tool_calls[0].result)
        self.assertEqual(run.text, "已根据失败结果调整")

    def test_unknown_tool(self):
        llm = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "not_exists", "args": {}, "id": "c1",
            }]),
            AIMessage(content="ok"),
        ])
        run = run_tool_loop(
            llm, [], [HumanMessage(content="q")], question="q", persist=False,
        )
        self.assertEqual(run.tool_calls[0].status, "error")
        self.assertIn("未知工具", run.tool_calls[0].error)

    def test_multi_round(self):
        llm = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "echo", "args": {"text": "1"}, "id": "c1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "echo", "args": {"text": "2"}, "id": "c2",
            }]),
            AIMessage(content="done"),
        ])
        run = run_tool_loop(
            llm, [FakeTool("echo", lambda a: a["text"])],
            [HumanMessage(content="q")], question="q", persist=False,
        )
        self.assertEqual([c.step for c in run.tool_calls], [1, 2])
        self.assertEqual(run.rounds, 3)

    def test_bind_tools_failure_degrades(self):
        llm = FakeLLM([AIMessage(content="纯对话回答")],
                      bind_error=ValueError("no tools"))
        run = run_tool_loop(
            llm, [], [HumanMessage(content="q")], question="q", persist=False,
        )
        self.assertEqual(run.text, "纯对话回答")
        self.assertIn("不支持工具调用", run.error)
        self.assertEqual(run.tool_calls, [])

    def test_max_rounds_returns_last_message(self):
        llm = FakeLLM([
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {}, "id": "c2"}]),
        ])
        run = run_tool_loop(
            llm, [FakeTool("echo", lambda a: "x")],
            [HumanMessage(content="q")], question="q", persist=False,
            max_rounds=2,
        )
        self.assertEqual(run.rounds, 2)
        self.assertEqual(len(run.tool_calls), 2)
        self.assertIn("轮数上限", run.text)

    def test_list_traces_empty(self):
        self.assertEqual(list_traces(path=self.log), [])


if __name__ == "__main__":
    unittest.main()
