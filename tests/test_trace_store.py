"""Trace 存储：追加、查询、导出、统计与脏数据容错。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.agent import trace_store


class TraceStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "traces.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, trace_id="t1", **extra):
        base = {
            "trace_id": trace_id,
            "question": "q",
            "final_answer": "a",
            "duration_ms": 100,
            "tool_call_count": 1,
            "error": None,
            "tool_calls": [],
        }
        base.update(extra)
        return base

    def test_append_and_load(self):
        trace_store.append_trace(self._record(), self.path)
        trace_store.append_trace(self._record("t2"), self.path)
        self.assertEqual(trace_store.load_trace("t1", self.path)["trace_id"], "t1")
        self.assertEqual(len(trace_store.list_traces(path=self.path)), 2)
        self.assertEqual(trace_store.list_traces(path=self.path)[0]["trace_id"], "t2")

    def test_malformed_line_is_skipped(self):
        self.path.write_text(
            json.dumps(self._record("ok"), ensure_ascii=False) + "\n{broken\n",
            encoding="utf-8",
        )
        rows = list(trace_store.iter_traces(self.path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trace_id"], "ok")

    def test_export(self):
        trace_store.append_trace(self._record("t1"), self.path)
        out = Path(self.tmp.name) / "export.json"
        trace_store.export_trace("t1", out, self.path)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["trace_id"], "t1")

    def test_export_missing_raises(self):
        with self.assertRaises(KeyError):
            trace_store.export_trace("missing", Path(self.tmp.name) / "x.json", self.path)

    def test_stats(self):
        trace_store.append_trace(self._record("t1"), self.path)
        trace_store.append_trace(self._record("t2", error="boom", tool_call_count=3), self.path)
        stats = trace_store.trace_stats(trace_store.iter_traces(self.path))
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["error_runs"], 1)
        self.assertEqual(stats["error_rate"], 0.5)
        self.assertEqual(stats["avg_tool_calls"], 2.0)

    def test_env_override(self):
        import os

        old = os.environ.get("AGENT_TRACE_LOG")
        os.environ["AGENT_TRACE_LOG"] = str(self.path)
        try:
            self.assertEqual(trace_store.trace_path(), self.path)
        finally:
            if old is None:
                os.environ.pop("AGENT_TRACE_LOG", None)
            else:
                os.environ["AGENT_TRACE_LOG"] = old


if __name__ == "__main__":
    unittest.main()
