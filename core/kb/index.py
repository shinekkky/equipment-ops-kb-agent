"""知识库索引：BM25 关键词检索 + 向量语义检索 + RRF 混合融合。

为什么两路都要
--------------
两路的失败模式不重叠，所以混合才有意义：

* **BM25** 精确命中故障码（P103）、告警码（N201）、设备型号（CXB-350）这类
  字符串。稠密向量很容易把 P101 和 P105 视为"语义相近"而混淆，BM25 不会。
* **向量检索** 处理口语化提问。运维人员问"泵发烫是怎么回事"时，手册里写的
  是"轴承温度过高"，字面完全不重合，只有语义检索能召回。

RRF（Reciprocal Rank Fusion）按**排名**而非分数融合，避免了两路分数量纲
不可比的问题，无需调参即可稳定工作。

实现上不引入 chromadb / faiss / rank_bm25：语料规模是几百个块，纯 Python
BM25 与 numpy 余弦相似度已经绰绰有余，少三个依赖换来别人能一键跑起来。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import (
    BGE_QUERY_INSTRUCTION,
    CHUNK_PREVIEW_CHARS,
    EMBEDDING_MODEL,
    INDEX_CACHE_DIR,
    TOP_K_MAX,
)
from .chunker import Chunk
from .loader import load_file

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


# ------------------------------------------------------------------ 分词

_JIEBA_READY = False


def tokenize(text: str) -> list[str]:
    """中文分词。使用 cut_for_search 以提升技术术语的部分匹配召回。

    jieba 首次调用需要加载词典（约 1 s），故只初始化一次。
    """
    global _JIEBA_READY
    import jieba

    if not _JIEBA_READY:
        jieba.initialize()
        _JIEBA_READY = True

    tokens: list[str] = []
    for tok in jieba.cut_for_search(text):
        tok = tok.strip().lower()
        # 丢弃纯空白与纯标点
        if tok and not all(not c.isalnum() for c in tok):
            tokens.append(tok)

    # 故障码/型号原样补一份小写形式，保证 "P103" 与 "p103" 归一
    from .chunker import _FAULT_CODE_RE, _MODEL_RES

    for rx in (_FAULT_CODE_RE, *_MODEL_RES):
        tokens.extend(m.lower() for m in rx.findall(text))
    return tokens


# ------------------------------------------------------------------ BM25


class BM25Index:
    """纯 Python BM25（Okapi），倒排索引实现。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._tfs: list[dict[str, int]] = []
        self._lens: list[int] = []
        df: dict[str, int] = {}

        for chunk in chunks:
            # 与向量索引保持一致，索引 embed_text（含《手册》名与章节路径前缀）。
            # 只索引正文会让"离心泵温度高怎么排查"这类查询丢失文档级信号。
            tokens = tokenize(chunk.embed_text)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tfs.append(tf)
            self._lens.append(len(tokens))
            for t in tf:
                df[t] = df.get(t, 0) + 1

        self._df = df
        self._n = len(chunks)
        self._avgdl = float(np.mean(self._lens)) if self._lens else 1.0

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # 加 1 平滑，保证语料中未出现的词 IDF 非负
        return float(np.log(1.0 + (self._n - df + 0.5) / (df + 0.5)))

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 [(chunk 下标, 分数)]，按分数降序。"""
        if not query.strip() or self._n == 0:
            return []

        scores = np.zeros(self._n, dtype=np.float64)
        for term in set(tokenize(query)):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, tf in enumerate(self._tfs):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * self._lens[i] / self._avgdl)
                scores[i] += idf * (f * (BM25_K1 + 1)) / denom

        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order[:top_k] if scores[i] > 0]


# ------------------------------------------------------------------ 向量


class VectorIndex:
    """基于 sentence-transformers + numpy 余弦相似度的向量检索。

    模型通过 `get_embedder()` 获取（调用方应做进程级缓存，见 app.py 的
    st.cache_resource）；模型不可用时上层应降级为仅 BM25。
    """

    def __init__(self, chunk_ids: list[str], model, embeddings: np.ndarray) -> None:
        self.chunk_ids = chunk_ids
        self.model = model
        self.embeddings = embeddings

    @classmethod
    def build(cls, chunks: list[Chunk], model, cache_path: Path | None = None) -> "VectorIndex":
        """构建向量索引，命中磁盘缓存则直接复用（嵌入是最贵的一步）。"""
        chunk_ids = [c.chunk_id for c in chunks]

        key = _vector_cache_key(chunk_ids)
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path) as data:
                    if str(data["key"]) == key:
                        return cls(chunk_ids, model, data["emb"])
            except Exception:
                pass  # 缓存损坏则重新编码

        embeddings = model.encode(
            [c.embed_text for c in chunks],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, key=key, emb=embeddings)
            except Exception:
                pass  # 缓存写失败不影响功能

        return cls(chunk_ids, model, embeddings)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        if not query.strip() or self.embeddings.size == 0:
            return []

        # BGE 官方建议：检索指令只加在查询侧，文档侧不加
        q = self.model.encode(
            [BGE_QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        sims = (self.embeddings @ q[0]).astype(np.float64)
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order[:top_k]]


def _vector_cache_key(chunk_ids: list[str]) -> str:
    raw = EMBEDDING_MODEL + "|" + "|".join(chunk_ids)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


_EMBEDDER = None


def get_embedder(allow_download: bool = False):
    """加载中文嵌入模型。失败返回 None，调用方据此降级为仅 BM25。

    allow_download=False 时只使用本地缓存，不联网——保证在无网环境下
    仍能启动，也避免 Streamlit 首屏卡在下载上。
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL, local_files_only=not allow_download)
        return _EMBEDDER
    except Exception:
        return None


# ------------------------------------------------------------------ 统一索引


@dataclass
class Retrieved:
    """一条检索结果：命中子块 + 其所属父块（用于回传完整章节）。"""

    chunk: Chunk
    parent: Chunk
    score: float
    retriever: str
    rank: int

    @property
    def citation(self) -> str:
        return self.chunk.citation

    def as_tool_text(self, max_chars: int = CHUNK_PREVIEW_CHARS) -> str:
        """格式化为给模型看的结果文本，带出处、限长。"""
        body = self.parent.text
        if len(body) > max_chars:
            body = body[:max_chars] + " …（后续内容已截断）"
        return (
            f"[{self.rank}] 出处：{self.citation}\n"
            f"（检索方式：{self.retriever}，相关度 {self.score:.3f}）\n"
            f"{body}"
        )


class KnowledgeIndex:
    """知识库索引：持有全部块，并按模式路由到 BM25 / 向量 / 混合检索。"""

    def __init__(
        self,
        children: list[Chunk],
        parents: dict[str, Chunk],
        *,
        embedder=None,
        use_cache: bool = True,
    ) -> None:
        self.children = children
        self.parents = parents
        self.bm25 = BM25Index(children)
        self.vector: VectorIndex | None = None
        if embedder is not None:
            cache = INDEX_CACHE_DIR / f"vec_{_vector_cache_key([c.chunk_id for c in children])}.npz" \
                if use_cache else None
            self.vector = VectorIndex.build(children, embedder, cache_path=cache)

    # -------------------------------------------------- 检索

    @property
    def vector_enabled(self) -> bool:
        return self.vector is not None

    @property
    def modes(self) -> list[str]:
        """真正可评测的检索模式。

        向量不可用时**不返回 hybrid**：此时 hybrid 在 search() 里会退化为
        BM25（这是界面上的降级行为，不是一种独立的检索方法）。若评测脚本
        仍把它列出来，表格里会出现两行完全相同的 BM25 结果，让人误以为
        "混合检索没有增益"。
        """
        if self.vector is None:
            return ["bm25"]
        return ["bm25", "vector", "hybrid"]

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        doc_titles: list[str] | None = None,
    ) -> list[Retrieved]:
        """按指定模式检索，返回带父块的命中列表。

        doc_titles 限定只在指定的若干本手册里检索。诊断层已经知道一台设备该查
        哪本手册（离心泵查泵手册、交换机查网络手册），不把这条已知信息用上，
        就会出现"旋转设备的轴承温度"命中《网络设备告警处置指南》光模块那一节的
        荒唐结果——两者都含"温度"二字，语义检索分辨不了。
        """
        top_k = max(1, min(int(top_k), TOP_K_MAX))
        query = (query or "").strip()
        if not query:
            return []

        allowed: set[int] | None = None
        if doc_titles:
            wanted = set(doc_titles)
            allowed = {i for i, c in enumerate(self.children) if c.doc_title in wanted}
            if not allowed:
                allowed = None  # 指定的手册不在库里，退回全库检索而非返回空

        def keep(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
            if allowed is None:
                return pairs
            return [(i, s) for i, s in pairs if i in allowed]

        # 多取候选再融合，最后截断到 top_k。候选池不能太小，否则两路的
        # 排名信息太少，RRF 退化成随机选择。限定手册时被过滤掉的比例不可控，
        # 故池子再放大一档，保证过滤后仍有足够候选。
        pool = max(top_k * 8, 24) if allowed is not None else max(top_k * 3, 10)
        rankings: dict[str, list[tuple[int, float]]] = {}
        rankings["bm25"] = keep(self.bm25.search(query, pool))
        if self.vector is not None:
            rankings["vector"] = keep(self.vector.search(query, pool))

        if mode == "bm25" or self.vector is None:
            return self._to_results(rankings["bm25"][:top_k], "bm25")

        if mode == "vector":
            return self._to_results(rankings["vector"][:top_k], "vector")

        return self._rrf_fuse(rankings, top_k)

    def _rrf_fuse(
        self, rankings: dict[str, list[tuple[int, float]]], top_k: int
    ) -> list[Retrieved]:
        """Reciprocal Rank Fusion：按排名倒数求和，规避两路分数量纲不可比。"""
        fused: dict[int, float] = {}
        for results in rankings.values():
            for rank, (idx, _score) in enumerate(results):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return self._to_results(ordered, "hybrid")

    def _to_results(self, pairs: list[tuple[int, float]], retriever: str) -> list[Retrieved]:
        out: list[Retrieved] = []
        for rank, (idx, score) in enumerate(pairs, start=1):
            child = self.children[idx]
            parent = self.parents.get(child.parent_id, child)
            out.append(
                Retrieved(chunk=child, parent=parent, score=score, retriever=retriever, rank=rank)
            )
        return out

    # -------------------------------------------------- 目录（供系统提示词注入）

    def catalog(self, max_sections_per_doc: int = 12) -> str:
        """生成紧凑的知识库目录，注入系统提示词让模型知道"有什么可引"。"""
        by_doc: dict[str, list[str]] = {}
        for child in self.children:
            if not child.section_title:
                continue
            label = f"{child.section_no} {child.section_title}".strip()
            key = child.doc_title
            by_doc.setdefault(key, [])
            if label not in by_doc[key]:
                by_doc[key].append(label)

        lines: list[str] = []
        for title, sections in by_doc.items():
            shown = sections[:max_sections_per_doc]
            extra = len(sections) - len(shown)
            tail = f" 等 {len(sections)} 节" if extra > 0 else ""
            lines.append(f"《{title}》：{'、'.join(shown)}{tail}")
        return "\n".join(lines)

    def stats(self) -> dict:
        docs = {c.doc_title for c in self.children}
        return {
            "documents": len(docs),
            "children": len(self.children),
            "parents": len(self.parents),
            "vector_enabled": self.vector is not None,
        }


# ------------------------------------------------------------------ 构建


def build_index(
    paths: list[Path],
    *,
    embedder=None,
    source_type: str = "builtin",
    use_cache: bool = True,
    child_size: int | None = None,
) -> KnowledgeIndex:
    """从一组文件构建索引。单个文件解析失败会被跳过而非中断整批。"""
    children: list[Chunk] = []
    parents: dict[str, Chunk] = {}
    errors: list[str] = []

    for path in paths:
        try:
            kids, folks = load_file(Path(path), source_type=source_type, child_size=child_size)
        except Exception as exc:  # 单个文件失败不应导致整库不可用
            errors.append(f"{Path(path).name}: {exc}")
            continue
        children.extend(kids)
        parents.update(folks)

    index = KnowledgeIndex(children, parents, embedder=embedder, use_cache=use_cache)
    if errors:
        index.load_errors = errors  # type: ignore[attr-defined]
    return index


def save_chunks(index: KnowledgeIndex, path: Path) -> None:
    """把块信息落盘为 JSON（便于人工检查与调试）。"""
    payload = {
        "children": [c.to_dict() for c in index.children],
        "parents": {k: v.to_dict() for k, v in index.parents.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
