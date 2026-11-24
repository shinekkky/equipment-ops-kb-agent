"""文档加载：把 Markdown / 纯文本 / PDF 统一成 (子块, 父块索引)。

PDF 走 pypdfium2（纯 wheel，无需系统级依赖）。有书签的 PDF 用书签划分章节，
命中后能引用到"《手册》第 3.1 节 高温处置（第 12 页）"；没有书签的退回按页切分，
至少保留页码出处。
"""

from __future__ import annotations

from pathlib import Path

from .chunker import Chunk, chunk_markdown, chunk_pages

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


def _read_text_file(path: Path) -> str:
    """读取文本文件，依次尝试常见中文编码。"""
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """逐页提取 PDF 文本，返回 [(页码, 文本)]，页码从 1 开始。"""
    import pypdfium2 as pdfium

    pages: list[tuple[int, str]] = []
    doc = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(doc)):
            try:
                text = doc[i].get_textpage().get_text_range()
            except Exception:
                text = ""
            pages.append((i + 1, text))
    finally:
        doc.close()
    return pages


def extract_pdf_toc(path: Path) -> list[tuple[int, str]]:
    """读取 PDF 书签目录，返回 [(页码, 标题)]；无书签时返回空列表。

    pypdfium2 不同版本的书签对象接口不一样：新版是 get_title() / get_dest()
    方法，旧版是 title / page_index 属性。只认其中一套会让另一个版本上的
    书签**静默读成空**——不报错，只是所有 PDF 都悄悄退化成按页切分。
    故两套写法都试。
    """
    import pypdfium2 as pdfium

    entries: list[tuple[int, str]] = []
    doc = pdfium.PdfDocument(str(path))
    try:
        try:
            toc = doc.get_toc()  # 注意：返回的是 generator，不是 list
        except Exception:
            return []
        for item in toc or []:
            title = _bookmark_title(item)
            page_index = _bookmark_page_index(item)
            if title and page_index is not None:
                entries.append((page_index + 1, title))
    finally:
        doc.close()
    return sorted(entries, key=lambda kv: kv[0])


def _bookmark_title(item) -> str | None:
    getter = getattr(item, "get_title", None)
    if callable(getter):
        try:
            return str(getter()).strip() or None
        except Exception:
            return None
    title = getattr(item, "title", None)
    return str(title).strip() if title else None


def _bookmark_page_index(item) -> int | None:
    """把书签目标解析成 0 起的页码。"""
    dest_getter = getattr(item, "get_dest", None)
    if callable(dest_getter):
        try:
            dest = dest_getter()
            index_getter = getattr(dest, "get_index", None)
            if callable(index_getter):
                return int(index_getter())
            if getattr(dest, "page_index", None) is not None:
                return int(dest.page_index)
        except Exception:
            pass
    page_index = getattr(item, "page_index", None)
    return int(page_index) if page_index is not None else None


def page_sections(toc: list[tuple[int, str]], total_pages: int) -> dict[int, str]:
    """把书签摊平成"每一页属于哪一节"。

    书签只标在章节起始页上，中间那些页没有书签。向下填充，让每一页都归属到
    最近一个起始于它之前的小节——否则一本书里只有几页能标出章节，其余仍是无名的。
    """
    mapping: dict[int, str] = {}
    current: str | None = None
    marks = dict(toc)
    for page_no in range(1, total_pages + 1):
        if page_no in marks:
            current = marks[page_no]
        if current:
            mapping[page_no] = current
    return mapping


def _pdf_title_from_toc(toc: list[tuple[int, str]], fallback: str) -> str:
    """推断文档标题。

    不能直接拿第一条书签当书名：手册的书签往往从"1 适用范围"这类**章节**开始，
    取第一条会把书名写成章节名，于是整本书的引用都变成"《1 适用范围》第 2 页"。
    只有首条书签不带章节号（形如"离心泵故障排查手册"）时才当作书名，
    否则退回文件名。
    """
    from .chunker import _SECTION_RE

    for _, title in toc:
        if title and not _SECTION_RE.match(title):
            return title
        break  # 只看第一条：第一条就是章节号的话，后面更不可能是书名
    return fallback


def load_file(
    path: Path,
    *,
    source_type: str = "builtin",
    doc_title: str | None = None,
    child_size: int | None = None,
) -> tuple[list[Chunk], dict[str, Chunk]]:
    """加载单个文件，返回 (子块列表, 父块索引)。

    支持 .md / .markdown / .txt / .pdf；不支持的格式抛出 ValueError。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"不支持的文档格式 {suffix!r}，仅支持 {'/'.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        pages = extract_pdf_pages(path)
        toc = extract_pdf_toc(path)
        title = doc_title or _pdf_title_from_toc(toc, path.stem)
        sections = page_sections(toc, len(pages))
        return chunk_pages(
            pages,
            doc_title=title,
            file_name=path.name,
            source_type=source_type,
            child_size=child_size,
            page_sections=sections or None,
        )

    text = _read_text_file(path)
    title = doc_title or _title_from_markdown(text) or path.stem
    return chunk_markdown(
        text,
        doc_title=title,
        file_name=path.name,
        source_type=source_type,
        child_size=child_size,
    )


def _title_from_markdown(text: str) -> str | None:
    """取第一个 H1 作为文档标题。"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line and not line.startswith("#"):
            break
    return None
