"""Markdown → PDF：面向本报告用到的 Markdown 子集，用 reportlab 直接排版。

为什么不上 markdown→HTML→PDF 的通用链路
--------------------------------------
那条链路要么引入 weasyprint / wkhtmltopdf（系统级依赖，评委机器上大概率装不上），
要么对中文字体处理不稳。本报告实际只用到标题、列表、表格、图片、引用五种元素，
写一个覆盖这五种的渲染器不过两百行，且只依赖 requirements.txt 里已有的 reportlab。

中文字体必须**嵌入**：不嵌入的方案在部分 PDF 阅读器上整篇显示为空白方块。
字体文件由 core.config.register_pdf_font 注册。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import register_pdf_font

# ---------------------------------------------------------------- 版式

MARGIN_MM = 18
BODY_SIZE = 9.5
BODY_LEADING = 15

_H1 = re.compile(r"^#\s+(.*)$")
_H2 = re.compile(r"^##\s+(.*)$")
_H3 = re.compile(r"^###\s+(.*)$")
_H4 = re.compile(r"^####\s+(.*)$")
_UL = re.compile(r"^(\s*)-\s+(.*)$")
_OL = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_QUOTE = re.compile(r"^>\s?(.*)$")
_HR = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_IMG = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(text: str) -> str:
    """把行内 Markdown 转成 reportlab 支持的迷你 HTML。"""
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)      # 反引号直接去掉：等宽字体不含中文
    return text


def _font_name() -> str:
    """取已注册的 CJK 字体名；不可用时返回 Helvetica 并提示（会显示为方块）。"""
    name = register_pdf_font()
    if name:
        return name
    print("[警告] 未找到可嵌入的中文字体，PDF 中的中文可能显示为方块。")
    return "Helvetica"


def markdown_to_pdf(markdown_text: str, out_path: Path, *, base_dir: Path | None = None) -> Path:
    """把报告 Markdown 渲染为 PDF。图片路径相对 base_dir 解析。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase.pdfmetrics import registerFontFamily

    font = _font_name()
    # SimHei 没有独立的粗体文件，把 family 的四档全指向同一字体，
    # 否则 <b> 标签会因找不到映射字体而抛错（视觉上加粗本来也不明显）。
    registerFontFamily(font, normal=font, bold=font, italic=font, boldItalic=font)

    base_dir = Path(base_dir) if base_dir else Path.cwd()
    styles = {
        "h1": ParagraphStyle("h1", fontName=font, fontSize=17, leading=24, spaceAfter=8),
        "h2": ParagraphStyle("h2", fontName=font, fontSize=13.5, leading=20,
                             spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName=font, fontSize=11.5, leading=18,
                             spaceBefore=10, spaceAfter=4),
        "h4": ParagraphStyle("h4", fontName=font, fontSize=10, leading=16,
                             spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", fontName=font, fontSize=BODY_SIZE,
                               leading=BODY_LEADING, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", fontName=font, fontSize=BODY_SIZE,
                                 leading=BODY_LEADING, leftIndent=14, bulletIndent=4,
                                 spaceAfter=2),
        "quote": ParagraphStyle("quote", fontName=font, fontSize=BODY_SIZE,
                                leading=BODY_LEADING, leftIndent=14, textColor=colors.HexColor("#344054")),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=8, leading=11.5),
        "cellhead": ParagraphStyle("cellhead", fontName=font, fontSize=8, leading=11.5),
    }

    flow: list = []
    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---------------------------------------------------- 空行 / 分隔线
        if not stripped:
            i += 1
            continue
        if _HR.match(stripped):
            flow.append(Spacer(1, 5))
            i += 1
            continue

        # ---------------------------------------------------- 图片
        m = _IMG.match(stripped)
        if m:
            path = (base_dir / m.group(2)).resolve()
            if path.exists():
                try:
                    iw, ih = ImageReader(str(path)).getSize()
                    width = (A4[0] - 2 * MARGIN_MM * mm)
                    height = ih * width / iw
                    flow.append(Spacer(1, 4))
                    flow.append(Image(str(path), width=width, height=height))
                    flow.append(Spacer(1, 6))
                except Exception as exc:
                    flow.append(Paragraph(
                        f"[图片渲染失败：{_escape(m.group(2))}（{_escape(str(exc))}）]", styles["body"]))
            else:
                # 找不到图必须留下痕迹。曾经因为静默跳过，14 张图全部丢失而
                # 报告看起来"生成成功"——这类失败比报错难查得多。
                flow.append(Paragraph(
                    f"[缺图：{_escape(m.group(2))}（相对 {_escape(str(base_dir))} 未找到）]",
                    styles["body"]))
            i += 1
            continue

        # ---------------------------------------------------- 表格
        if stripped.startswith("|"):
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            flow.extend(_render_table(block, styles, font))
            flow.append(Spacer(1, 6))
            continue

        # ---------------------------------------------------- 标题
        for pattern, key in ((_H4, "h4"), (_H3, "h3"), (_H2, "h2"), (_H1, "h1")):
            m = pattern.match(stripped)
            if m:
                flow.append(Paragraph(_inline(m.group(1)), styles[key]))
                break
        else:
            # ------------------------------------------------ 引用 / 列表 / 正文
            m = _QUOTE.match(stripped)
            if m:
                flow.append(Paragraph(_inline(m.group(1)), styles["quote"]))
            elif (m := _UL.match(line)):
                flow.append(Paragraph(_inline(m.group(2)), styles["bullet"], bulletText="•"))
            elif (m := _OL.match(line)):
                flow.append(Paragraph(_inline(m.group(3)), styles["bullet"],
                                      bulletText=f"{m.group(2)}."))
            else:
                flow.append(Paragraph(_inline(stripped), styles["body"]))
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _page_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawRightString(A4[0] - MARGIN_MM * mm, 10 * mm, f"第 {doc.page} 页")
        canvas.drawString(MARGIN_MM * mm, 10 * mm, "工业设备运维与故障诊断报告")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN_MM * mm, rightMargin=MARGIN_MM * mm,
        topMargin=MARGIN_MM * mm, bottomMargin=16 * mm,
        title="工业设备运维与故障诊断报告",
    )
    doc.build(flow, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return out_path


def _render_table(block: list[str], styles: dict, font: str) -> list:
    """把 Markdown 表格转成 reportlab Table。首行为表头，分隔行跳过。"""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    rows: list[list[str]] = []
    for line in block:
        if _TABLE_SEP.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return []

    head_style = ParagraphStyle("th", parent=styles["cellhead"], textColor=colors.white)
    data = [[Paragraph(_inline(c), head_style if r == 0 else styles["cell"]) for c in row]
            for r, row in enumerate(rows)]

    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475467")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d5dd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f9fafb")]),
    ]))
    return [table]
