"""RAG 检索与评测函数测试（只跑 BM25，避免测试依赖嵌入模型）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.config import MANUALS_DIR  # noqa: E402
from core.kb.index import build_index  # noqa: E402


class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = build_index(sorted(MANUALS_DIR.glob("*.md")), embedder=None)

    def test_vector_disabled_modes(self):
        self.assertFalse(self.index.vector_enabled)
        # 关键回归：向量不可用时不能把 hybrid 当作独立方法列出来
        self.assertEqual(self.index.modes, ["bm25"])

    def test_fault_code_search(self):
        hits = self.index.search("P103 是什么故障", top_k=3, mode="bm25")
        self.assertTrue(hits)
        self.assertTrue(any(
            hit.chunk.doc_title == "离心泵故障排查手册" and hit.chunk.section_no == "2.2"
            for hit in hits
        ))

    def test_manual_constraint(self):
        hits = self.index.search(
            "CPU 利用率过高怎么处理",
            top_k=3,
            mode="bm25",
            doc_titles=["网络设备告警处置指南"],
        )
        self.assertTrue(hits)
        self.assertTrue(all(hit.chunk.doc_title == "网络设备告警处置指南" for hit in hits))

    def test_hybrid_falls_back_to_bm25_without_vector(self):
        bm25 = self.index.search("泵发烫", top_k=3, mode="bm25")
        hybrid = self.index.search("泵发烫", top_k=3, mode="hybrid")
        self.assertEqual(
            [h.chunk.chunk_id for h in bm25],
            [h.chunk.chunk_id for h in hybrid],
        )

    def test_evaluate_returns_expected_keys(self):
        import eval_retrieval

        cases = [
            {"query": "P103 是什么故障", "doc": "离心泵故障排查手册", "section_no": "2.2"},
            {"query": "CPU 利用率过高怎么处理", "doc": "网络设备告警处置指南", "section_no": "3.1"},
        ]
        result = eval_retrieval.evaluate(self.index, cases, "bm25")
        self.assertEqual(result["n"], 2)
        self.assertEqual(set(result["hit"]), {"1", "3", "5"})
        self.assertGreaterEqual(result["mrr"], 0.0)
        self.assertLessEqual(result["mrr"], 1.0)
        self.assertEqual(len(result["per_case"]), 2)


if __name__ == "__main__":
    unittest.main()
