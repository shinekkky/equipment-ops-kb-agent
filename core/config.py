"""全局配置：路径常量、中文字体注册、多服务商模型预设、检索参数。

这些配置原先散落在 app.py 里，界面与脚本各维护一份、容易改漏。集中到本模块后，
Streamlit 界面、检索评测脚本与各核心模块共用同一份来源。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------- 路径

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANUALS_DIR = DATA_DIR / "manuals"
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_CACHE_DIR = BASE_DIR / ".index_cache"
REPORT_OUT_DIR = BASE_DIR / "reports_out"

SENSOR_CSV = DATA_DIR / "设备传感器时序数据.csv"
DEVICE_REGISTRY_CSV = DATA_DIR / "设备台账.csv"
EVAL_SET_JSONL = DATA_DIR / "检索评测集.jsonl"

for _d in (DATA_DIR, MANUALS_DIR, UPLOAD_DIR, INDEX_CACHE_DIR, REPORT_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 字体

# SimHei 是单体 TTF，可直接嵌入 PDF；msyh.ttc / simsun.ttc 是 TTC 字体集合，
# reportlab 对 TTC 支持不稳定，故不作为候选。
_CJK_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simkai.ttf"),
    Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
]

_matplotlib_ready = False
_pdf_font_name: str | None = None


def setup_chinese_font() -> str:
    """注册中文字体，返回 matplotlib 可用的字体族名。

    matplotlib 未找到中文字体时返回 'sans-serif'，图表中文会显示为方框但不报错。
    """
    global _matplotlib_ready
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")  # 无 GUI 后端，Streamlit 与脚本导出均可复用

    for path in _CJK_FONT_CANDIDATES:
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                name = font_manager.FontProperties(fname=str(path)).get_name()
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                _matplotlib_ready = True
                return name
            except Exception:
                continue

    matplotlib.rcParams["axes.unicode_minus"] = False
    return "sans-serif"


def register_pdf_font() -> str | None:
    """向 reportlab 注册并嵌入中文字体，返回字体名；无可用字体时返回 None。

    必须嵌入字体而非依赖阅读器自带字体：不嵌入的方案在部分阅读器上
    会整篇显示为空白方块。
    """
    global _pdf_font_name
    if _pdf_font_name is not None:
        return _pdf_font_name

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return None

    for path in _CJK_FONT_CANDIDATES:
        if path.suffix.lower() != ".ttf" or not path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("CJK", str(path)))
            _pdf_font_name = "CJK"
            return _pdf_font_name
        except Exception:
            continue

    return None


# ---------------------------------------------------------------- 模型服务商

# protocol 决定调用协议：openai 兼容走 ChatOpenAI，anthropic 兼容走 ChatAnthropic，
# 两者接口格式不同，不可混用。
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "DeepSeek（深度求索）": {
        "protocol": "openai",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "OpenAI": {
        "protocol": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "通义千问（Qwen）": {
        "protocol": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "智谱 GLM": {
        "protocol": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "月之暗面（Kimi）": {
        "protocol": "openai",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
    "百度文心（千帆）": {
        "protocol": "openai",
        "base_url": "https://qianfan.baidubce.com/v2",
        "model": "ernie-speed-128k",
    },
    "Claude / Anthropic": {
        "protocol": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-5-5",
    },
    "自定义（OpenAI 兼容端点）": {
        "protocol": "openai",
        "base_url": "",
        "model": "",
    },
    "自定义（Anthropic 兼容端点）": {
        "protocol": "anthropic",
        "base_url": "",
        "model": "",
    },
}


# ---------------------------------------------------------------- 检索

RETRIEVAL_MODES = {
    "BM25（关键词）": "bm25",
    "向量（语义）": "vector",
    "混合（RRF 融合）": "hybrid",
}
DEFAULT_RETRIEVAL_MODE = "hybrid"

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# BGE 官方建议的检索指令前缀，仅加在查询侧，文档侧不加。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

TOP_K_DEFAULT = 4
TOP_K_MAX = 5
CHUNK_PREVIEW_CHARS = 600  # 工具返回给模型的单块正文截断长度


def python_version_note() -> str | None:
    """若当前解释器版本不适合本项目，返回提示文本，否则返回 None。"""
    if sys.version_info < (3, 9):
        return f"当前 Python {sys.version.split()[0]} 过低，建议使用 Python 3.12。"
    if sys.version_info >= (3, 14):
        return (
            f"当前 Python {sys.version.split()[0]} 过新，"
            "sentence-transformers / torch 等依赖可能缺少对应 wheel，建议使用 Python 3.12。"
        )
    return None
