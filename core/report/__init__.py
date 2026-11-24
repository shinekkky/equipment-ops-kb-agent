"""诊断报告：手册依据检索、Markdown 组装、趋势配图与 PDF 导出。"""

from . import charts
from .builder import (
    Advice,
    ReportBundle,
    build_advices,
    default_report_paths,
    render_markdown,
    save_markdown,
)
from .pdf import markdown_to_pdf

__all__ = [
    "charts",
    "Advice",
    "ReportBundle",
    "build_advices",
    "render_markdown",
    "save_markdown",
    "markdown_to_pdf",
    "default_report_paths",
]
