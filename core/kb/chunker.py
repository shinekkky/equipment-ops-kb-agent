"""标题感知分块：把手册切成带完整出处信息的父子块。

设计要点
--------
1. **标题感知**：用 MarkdownHeaderTextSplitter 按 H1/H2/H3 分节，章节号与标题
   从标题文本解析，使每个块都能定位到「《手册》3.1 节 高温报警处置」。
2. **块打包而非盲目递归切分**：先按空行切成段落/表格/列表等"块"，再把相邻块
   打包到尺寸上限。这样 Markdown 表格不会被从中间切开——阈值表一旦被切断，
   检索到半张表反而会误导模型。只有单个块自身超长时才退回递归切分。
3. **中文分隔符**：RecursiveCharacterTextSplitter 的默认分隔符是
   `['\\n\\n', '\\n', ' ', '']`，中文句子没有空格，靠默认值根本切不开，
   必须显式补上中文标点。
4. **父子结构**：子块小、用于提升检索精度；命中后回传所属整节（parent），
   保证模型拿到的是完整处置步骤而不是半截流程。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 子块目标尺寸（字符数）。中文约 1 字符 ≈ 1 token。
# 250 是 scripts/eval_retrieval.py 在 27 条评测集上扫出来的：混合检索
# MRR 0.881、Hit@5 100%，优于 150/200/300/450/900 各档。注意块打包器
# 不会切开单个块，所以表格与完整段落不会被这 250 字的预算切碎。
CHILD_SIZE = 250
CHILD_OVERLAP = 40

# 关键：显式传入中文标点分隔符，否则默认值切不动中文
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]

# 章节号形如 "3"、"3.1"、"3.1.2"
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)[\s、.．]*(.*)$")
# 故障/告警码：P101、N201
_FAULT_CODE_RE = re.compile(r"\b[A-Z]{1,2}\d{3,4}\b")
# 设备型号：CXB-200、S5720-28X、Y2-280M、P-101、M-201
_MODEL_RES = [
    re.compile(r"\b[A-Z][A-Z0-9]{1,7}-[A-Z0-9]{2,6}\b"),
    re.compile(r"\b[A-Z]{1,3}-\d{2,3}\b"),
]

_HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]


@dataclass
class Chunk:
    """一个检索单元。子块用于检索，父块用于回传上下文。"""

    chunk_id: str
    parent_id: str
    doc_id: str
    doc_title: str
    source_type: str            # builtin | upload
    file_name: str
    page: int | None
    heading_path: list[str]     # 不含文档标题，如 ['3 故障诊断与处置', '3.1 高温报警处置']
    section_no: str             # '3.1'
    section_title: str          # '高温报警处置'
    text: str                   # 子块正文
    parent_text: str            # 所属整节正文
    embed_text: str             # 向量化用文本（标题路径前缀 + 正文）
    fault_codes: list[str] = field(default_factory=list)
    device_models: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """统一引用串，直接渲染在界面上，让"定位到手册章节"可见。"""
        text = f"《{self.doc_title}》"
        if self.section_no:
            text += f"第 {self.section_no} 节"
        if self.section_title:
            text += f" {self.section_title}"
        # 无书签的 PDF 把页码写进了 section_title，此时不能再追一次页码，
        # 否则会输出"第 1 页（第 1 页）"这种重复。
        if self.page and f"第 {self.page} 页" not in text:
            text += f"（第 {self.page} 页）"
        return text

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["citation"] = self.citation
        return d


def _sha1(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _parse_heading(heading: str) -> tuple[str, str]:
    """从标题文本解析出章节号与标题，'3.1 高温报警处置' -> ('3.1', '高温报警处置')。"""
    if not heading:
        return "", ""
    m = _SECTION_RE.match(heading.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return "", heading.strip()


def _extract_fault_codes(text: str) -> list[str]:
    return sorted(set(_FAULT_CODE_RE.findall(text)))


def _extract_models(text: str) -> list[str]:
    found: set[str] = set()
    for rx in _MODEL_RES:
        found.update(rx.findall(text))
    return sorted(found)


def _split_into_blocks(text: str) -> list[str]:
    """按空行切成块。Markdown 表格、列表、段落各自成为完整块。"""
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def _pack_blocks(blocks: list[str], max_size: int, overlap: int) -> list[str]:
    """把块打包到尺寸上限，不切开任何单个块。

    只有单个块自身超过 max_size 时才用递归切分器处理它，此时才真正需要
    中文分隔符——这是 CHINESE_SEPARATORS 存在的唯一理由。
    """
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
        length_function=len,
    )

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            # 保留尾部一个块作为重叠，避免跨块语义断裂
            tail = buf[-1] if len(buf) > 1 else ""
            buf = [tail] if tail else []
            buf_len = len(tail)

    for block in blocks:
        if len(block) > max_size:
            flush()
            chunks.extend(fallback.split_text(block))
            buf, buf_len = [], 0
            continue
        if buf_len + len(block) + 2 > max_size and buf:
            flush()
        buf.append(block)
        buf_len += len(block) + 2

    if buf:
        chunks.append("\n\n".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def _build_embed_text(heading_path: list[str], doc_title: str, text: str) -> str:
    """向量化前给正文前置标题路径。

    非对称检索的廉价增益：查询通常只给主题（"高温报警怎么处理"），
    把章节路径拼进被检索文本能显著提高主题式查询的召回。
    """
    prefix = "《%s》" % doc_title
    if heading_path:
        prefix += " > " + " > ".join(heading_path)
    return f"{prefix} | {text}"


def chunk_markdown(
    text: str,
    *,
    doc_title: str,
    file_name: str,
    source_type: str = "builtin",
    page: int | None = None,
    child_size: int | None = None,
    child_overlap: int | None = None,
) -> tuple[list[Chunk], dict[str, Chunk]]:
    """把一份 Markdown 文档切成子块，并返回父块索引。

    返回 (children, parents)：
      children —— 用于建立检索索引
      parents  —— {parent_id: 整节 Chunk}，命中子块后据此回传完整章节

    注意：child_size 默认值在**调用时**从模块常量读取，而不是写成默认参数——
    默认参数在函数定义时即绑定，之后修改 CHILD_SIZE 将不生效（这个坑很隐蔽）。
    """
    child_size = CHILD_SIZE if child_size is None else child_size
    child_overlap = CHILD_OVERLAP if child_overlap is None else child_overlap
    doc_id = _sha1(text)
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT_ON)
    sections = splitter.split_text(text)

    children: list[Chunk] = []
    parents: dict[str, Chunk] = {}

    for sec_idx, sec in enumerate(sections):
        meta = sec.metadata
        h1 = meta.get("H1", "")
        heading_path = [meta[k] for k in ("H2", "H3") if meta.get(k)]
        # H1 是文档标题，不作为章节层级；无 H2/H3 时退回 H1
        if not heading_path and h1:
            heading_path = [h1]

        # 章节号取自层级最深的一个标题
        section_no, section_title = _parse_heading(heading_path[-1]) if heading_path else ("", "")
        if not section_no:
            for h in reversed(heading_path):
                no, title = _parse_heading(h)
                if no:
                    section_no, section_title = no, title
                    break

        body = sec.page_content.strip()
        if not body:
            continue

        parent_id = f"{doc_id}#s{sec_idx}"

        # 父块：整节正文，命中后回传给模型
        parent = Chunk(
            chunk_id=parent_id,
            parent_id=parent_id,
            doc_id=doc_id,
            doc_title=doc_title,
            source_type=source_type,
            file_name=file_name,
            page=page,
            heading_path=heading_path,
            section_no=section_no,
            section_title=section_title,
            text=body,
            parent_text=body,
            embed_text=_build_embed_text(heading_path, doc_title, body),
        )
        parent.fault_codes = _extract_fault_codes(body)
        parent.device_models = _extract_models(body)
        parents[parent_id] = parent

        # 子块：检索单元
        for ord_i, piece in enumerate(_pack_blocks(_split_into_blocks(body), child_size, child_overlap)):
            child = Chunk(
                chunk_id=f"{parent_id}c{ord_i}",
                parent_id=parent_id,
                doc_id=doc_id,
                doc_title=doc_title,
                source_type=source_type,
                file_name=file_name,
                page=page,
                heading_path=heading_path,
                section_no=section_no,
                section_title=section_title,
                text=piece,
                parent_text=body,
                embed_text=_build_embed_text(heading_path, doc_title, piece),
            )
            child.fault_codes = _extract_fault_codes(piece)
            child.device_models = _extract_models(piece)
            children.append(child)

    return children, parents


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    doc_title: str,
    file_name: str,
    source_type: str = "upload",
    child_size: int | None = None,
    child_overlap: int | None = None,
    page_sections: dict[int, str] | None = None,
) -> tuple[list[Chunk], dict[str, Chunk]]:
    """把 PDF 的逐页文本切成子块。

    始终以"页"为父块边界：命中后回传整页，既保证上下文完整，也天然带页码出处。

    page_sections 是「页码 → 所属章节标题」，来自 PDF 书签。有它时标注真实章节，
    引用能精确到"《手册》第 3.1 节 高温处置（第 12 页）"；没有书签的 PDF 退回
    标注页码，至少保住出处可追溯。
    """
    child_size = CHILD_SIZE if child_size is None else child_size
    child_overlap = CHILD_OVERLAP if child_overlap is None else child_overlap
    children: list[Chunk] = []
    parents: dict[str, Chunk] = {}
    for page_no, page_text in pages:
        body = (page_text or "").strip()
        if not body:
            continue
        doc_id = _sha1(file_name)
        parent_id = f"{doc_id}#p{page_no}"

        # 有书签就标真实章节，否则退回页码
        heading = (page_sections or {}).get(page_no, "")
        if heading:
            section_no, section_title = _parse_heading(heading)
            heading_path = [heading]
        else:
            section_no, section_title = "", f"第 {page_no} 页"
            heading_path = []

        parent = Chunk(
            chunk_id=parent_id,
            parent_id=parent_id,
            doc_id=doc_id,
            doc_title=doc_title,
            source_type=source_type,
            file_name=file_name,
            page=page_no,
            heading_path=heading_path,
            section_no=section_no,
            section_title=section_title,
            text=body,
            parent_text=body,
            embed_text=_build_embed_text(heading_path, doc_title, body),
        )
        parent.fault_codes = _extract_fault_codes(body)
        parent.device_models = _extract_models(body)
        parents[parent_id] = parent

        for ord_i, piece in enumerate(_pack_blocks(_split_into_blocks(body), child_size, child_overlap)):
            child = Chunk(
                chunk_id=f"{parent_id}c{ord_i}",
                parent_id=parent_id,
                doc_id=doc_id,
                doc_title=doc_title,
                source_type=source_type,
                file_name=file_name,
                page=page_no,
                heading_path=heading_path,
                section_no=section_no,
                section_title=section_title,
                text=piece,
                parent_text=body,
                embed_text=_build_embed_text(heading_path, doc_title, piece),
            )
            child.fault_codes = _extract_fault_codes(piece)
            child.device_models = _extract_models(piece)
            children.append(child)

    return children, parents
