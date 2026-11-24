"""设备运维知识库 Agent —— Streamlit 产品原型。

页面按业务角色组织为四个核心工作台：
    设备监控总览 → 管理人员看全局风险、设备健康与趋势
    异常预警列表 → 运维人员按风险处理异常、查看证据链
    智能诊断     → 技术支持查手册依据、追问故障原因
    诊断报告     → 把结论、依据和处置排期整理成可交付报告

底层仍是三层能力：确定性检测（core.diagnostics）→ 知识库检索（core.kb）
→ 模型组织（core.agent）。**没有 API Key 也能用**：检测、检索、图表、
Markdown/PDF 导出全部是确定性计算，模型只负责综合分析与自由问答。
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.agent import AgentContext, answer_question, build_llm, friendly_error, generate_narrative
from core.config import (
    MANUALS_DIR,
    PROVIDER_PRESETS,
    REPORT_OUT_DIR,
    RETRIEVAL_MODES,
    TOP_K_DEFAULT,
    UPLOAD_DIR,
    python_version_note,
)
from core.diagnostics import (
    ALARM,
    ATTENTION,
    TRIP,
    WARN,
    load_registry,
    load_sensor_data,
    scan,
)
from core.kb.index import KnowledgeIndex, get_embedder
from core.kb.loader import load_file
from core.report import (
    ReportBundle,
    build_advices,
    charts,
    default_report_paths,
    markdown_to_pdf,
    render_markdown,
    save_markdown,
)

SEVERITY_ICON = {TRIP: "停机", ALARM: "报警", WARN: "预警", ATTENTION: "注意", "正常": "正常"}

# 业务优先级：与报告「处置排期建议」保持一致，P0 最紧急。
PRIORITY_BY_SEVERITY = {
    TRIP: "P0",
    ALARM: "P1",
    WARN: "P2",
    ATTENTION: "P3",
    "正常": "—",
}

ACTION_BY_SEVERITY = {
    TRIP: "立即停机检修，严禁继续运行",
    ALARM: "24 小时内安排处置，处置前加强监视",
    WARN: "纳入本周计划检修窗口，加密巡视频次",
    ATTENTION: "缩短巡检周期至每周，建立趋势跟踪",
    "正常": "保持常规巡检",
}

ROLE_PROFILES = {
    "设备运维人员": {
        "desc": "快速发现异常，明确先查什么、怎么处理",
        "entry": "异常预警列表 → 智能诊断",
    },
    "设备管理人员": {
        "desc": "掌握设备健康与风险趋势，安排检修计划",
        "entry": "设备监控总览 → 诊断报告",
    },
    "技术支持人员": {
        "desc": "检索手册依据，核对故障原因与处置步骤",
        "entry": "智能诊断 → 知识库",
    },
}

PAGE_LABELS = {
    "overview": "📊 设备监控总览",
    "alerts": "🚨 异常预警列表",
    "diagnosis": "🧠 智能诊断",
    "report": "📄 诊断报告",
    "kb": "📚 知识库",
}

# 角色决定进入工作台时先看到哪一屏；五个页面始终都可访问。
PAGE_ORDER = {
    "设备运维人员": ["alerts", "diagnosis", "overview", "report", "kb"],
    "设备管理人员": ["overview", "alerts", "diagnosis", "report", "kb"],
    "技术支持人员": ["diagnosis", "kb", "alerts", "overview", "report"],
}

st.set_page_config(
    page_title="设备运维知识库 Agent",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2.2rem; }
    .stButton>button { border-radius: 6px; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    .cite { color:#475467; font-size:0.86rem; }
</style>
""", unsafe_allow_html=True)


# ================================================================ 资源加载


@st.cache_resource(show_spinner=False)
def build_knowledge_index(manual_signature: str):
    """构建知识库索引。

    manual_signature 由全部手册文件的路径与修改时间拼成：文件一变，缓存键就变，
    索引自动重建。否则换了手册内容却还在用旧索引，是最难排查的一类问题。
    """
    embedder = get_embedder()
    children, parents = [], {}
    errors: list[str] = []

    sources = [(p, "builtin") for p in sorted(MANUALS_DIR.glob("*.md"))]
    sources += [(p, "upload") for p in sorted(UPLOAD_DIR.glob("*"))
                if p.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}]

    for path, source_type in sources:
        try:
            kids, folks = load_file(path, source_type=source_type)
            children.extend(kids)
            parents.update(folks)
        except Exception as exc:
            errors.append(f"{path.name}：{exc}")

    index = KnowledgeIndex(children, parents, embedder=embedder)
    if errors:
        index.load_errors = errors  # type: ignore[attr-defined]
    return index


def manuals_signature() -> str:
    parts: list[str] = []
    for folder in (MANUALS_DIR, UPLOAD_DIR):
        for path in sorted(folder.glob("*")):
            if path.is_file():
                parts.append(f"{path.name}:{path.stat().st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def derive_registry(data: pd.DataFrame) -> pd.DataFrame:
    """从传感器数据推导设备台账。

    上传的数据未必带台账文件，但报告里要写清安装位置与责任人。缺失时留空并注明
    "未登记"，而不是编造一个假位置——报告上的每个字段都必须能追溯到来源。
    """
    cols = ["设备ID", "设备名称", "设备型号", "设备类型"]
    if not set(cols).issubset(data.columns):
        return pd.DataFrame(columns=cols + ["安装位置", "投运日期", "生产厂商", "责任人"])

    registry = data[cols].drop_duplicates("设备ID").reset_index(drop=True)
    registry["安装位置"] = "未登记"
    registry["投运日期"] = "未登记"
    registry["生产厂商"] = "未登记"
    registry["责任人"] = "未登记"
    return registry


# ================================================================ 侧边栏


with st.sidebar:
    st.header("🎯 角色视角")
    role = st.selectbox("当前角色", list(ROLE_PROFILES), index=1,
                        help="角色只改变进入工作台时先看到哪一屏，五个页面始终都可以访问。")
    profile = ROLE_PROFILES[role]
    st.caption(profile["desc"])
    st.caption(f"建议入口：{profile['entry']}")

    st.divider()
    with st.expander("⚙️ 模型与检索配置", expanded=False):
        provider = st.selectbox("模型服务商", [
            "DeepSeek（深度求索）", "OpenAI", "通义千问（Qwen）", "智谱 GLM",
            "月之暗面（Kimi）", "百度文心（千帆）", "Claude / Anthropic",
            "自定义（OpenAI 兼容端点）", "自定义（Anthropic 兼容端点）",
        ])
        preset = PROVIDER_PRESETS[provider]
        is_custom = provider.startswith("自定义")

        api_key = st.text_input("API Key", type="password", placeholder="sk-...")
        base_url = (
            st.text_input("Base URL", value=preset["base_url"], placeholder="https://api.example.com/v1")
            if is_custom else preset["base_url"]
        )
        model_name = st.text_input(
            "模型名称", value=preset["model"], placeholder="如 deepseek-chat / claude-sonnet-5-5"
        )
        if not is_custom:
            st.caption(f"Base URL：`{base_url}`")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                                help="诊断场景建议 0，保证同一份数据两次分析结论一致。")
        st.caption("API Key 仅保存在当前会话内存中，不落盘。")

        st.markdown("**检索设置**")
        mode_label = st.selectbox("检索模式", list(RETRIEVAL_MODES.keys()), index=2,
                                  help="混合模式用 RRF 融合关键词与语义两路排名，兼顾故障码精确匹配与口语化提问。")
        retrieval_mode = RETRIEVAL_MODES[mode_label]
        top_k = st.slider("引用章节数", 1, 5, TOP_K_DEFAULT)

    st.caption("不填 API Key 也可使用全部检测、检索与报告功能；模型仅用于综合分析与自由问答。")

    note = python_version_note()
    if note:
        st.warning(note)


# ================================================================ 数据源

st.title("设备运维知识库 Agent")
st.caption(
    "从设备数据接入，到异常预警、手册诊断、处置报告的一体化运维工作台"
    "　|　检测与检索为确定性计算，不填 API Key 也可完整使用"
)

with st.expander("📁 数据源（可使用内置仿真数据，或导入自己的数据与手册）", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        sensor_file = st.file_uploader(
            "设备时序数据（CSV / Excel）", type=["csv", "xlsx"],
            help="需包含「时间」与「设备ID」列；其余列按指标名自动识别。",
        )
        registry_file = st.file_uploader(
            "设备台账（可选，CSV / Excel）", type=["csv", "xlsx"],
            help="提供安装位置与责任人，用于报告中的责任划分。缺省时从时序数据推导。",
        )
    with col_b:
        manual_files = st.file_uploader(
            "运维手册（可选，PDF / Markdown / TXT，可多选）", type=["pdf", "md", "markdown", "txt"],
            accept_multiple_files=True,
            help="上传后与内置手册合并建索引。PDF 会优先按书签分节，无书签时按页切分。",
        ) or []

    if manual_files:
        saved = []
        for f in manual_files:
            target = UPLOAD_DIR / f.name
            target.write_bytes(f.getbuffer())
            saved.append(f.name)
        st.success(f"已加入知识库：{'、'.join(saved)}（索引将在下次刷新时重建）")

    if st.button("重建知识库索引"):
        build_knowledge_index.clear()
        st.rerun()


@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes: bytes, name: str) -> pd.DataFrame:
    import io
    buffer = io.BytesIO(file_bytes)
    return pd.read_excel(buffer) if name.endswith(".xlsx") else pd.read_csv(buffer, encoding="utf-8-sig")


def current_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    if sensor_file is not None:
        data = load_uploaded(sensor_file.getvalue(), sensor_file.name)
        data = data.copy()
        data["时间"] = pd.to_datetime(data["时间"], errors="coerce")
        data = data.dropna(subset=["时间"]).sort_values(["设备ID", "时间"]).reset_index(drop=True)
        if "故障代码" in data.columns:
            data["故障代码"] = data["故障代码"].fillna("").astype(str)
        registry = (load_uploaded(registry_file.getvalue(), registry_file.name)
                    if registry_file is not None else derive_registry(data))
        return data, registry
    return load_sensor_data(), load_registry()


try:
    data, registry = current_dataset()
except Exception as exc:
    st.error(f"数据读取失败：{exc}")
    st.stop()

manual_sig = manuals_signature()
index = build_knowledge_index(manual_sig)
stats = index.stats()


# ================================================================ 诊断（缓存）

for key, default in (("scan_result", None), ("bundle", None), ("chat", []), ("scan_sig", None)):
    st.session_state.setdefault(key, default)


def scan_signature() -> str:
    """数据、检索参数或手册任一变化，都应重新跑检测与手册检索。"""
    return "|".join([
        str(len(data)),
        str(data["时间"].min()),
        str(data["时间"].max()),
        str(data["设备ID"].nunique()),
        str(top_k),
        retrieval_mode,
        manual_sig,
    ])


def run_scan():
    with st.spinner("正在执行四层异常检测与手册检索…"):
        result = scan(data=data, registry=registry)
        advices = build_advices(result, index, data, top_k=top_k, mode=retrieval_mode)
        st.session_state.scan_result = result
        st.session_state.bundle = ReportBundle(scan=result, advices=advices)
        st.session_state.scan_sig = scan_signature()
        # 数据或检索条件变了，之前生成的综合分析/PDF 就不能再代表本次结果。
        st.session_state.pop("narrative", None)
        st.session_state.pop("pdf_bytes", None)


if st.session_state.get("scan_sig") != scan_signature():
    run_scan()

result = st.session_state.scan_result
bundle = st.session_state.bundle


def advice_for(finding):
    return next((a for a in bundle.advices if a.finding is finding), None)


# ================================================================ 页面：设备监控总览


def render_overview():
    st.subheader("设备监控总览")
    st.caption("给设备管理人员：先看全局风险，再看设备健康状态与处置优先级。")

    counts = result.by_severity()
    total = len(result.findings)
    alarmed = sum(1 for f in result.findings if f.is_alarm)
    hidden = total - alarmed
    leads = [f.lead_time_hours for f in result.findings if f.lead_time_hours]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("监测设备", f"{len(result.devices)} 台")
    c2.metric("异常结论", f"{total} 条")
    c3.metric("已越限", f"{alarmed} 条")
    c4.metric("未越限隐患", f"{hidden} 条")
    c5.metric("最长提前", f"{max(leads):.0f} 小时" if leads else "—")

    if hidden:
        st.info(
            f"当前 **{hidden}** 条隐患尚未越过手册报警线，"
            f"其中 **{len(leads)}** 条有明确提前量——这段时间就是安排计划检修、避免非计划停机的窗口。"
        )

    left, right = st.columns([1, 2.15])
    with left:
        st.markdown("##### 风险等级分布")
        fig = charts.severity_bar(counts)
        st.pyplot(fig, width="stretch")
        import matplotlib.pyplot as plt
        plt.close(fig)

    with right:
        st.markdown("##### 预警趋势（按发现日期）")
        records = pd.DataFrame([{"日期": f.start, "严重度": f.severity} for f in result.findings])
        fig = charts.severity_timeline(records, start=result.start, end=result.end)
        st.pyplot(fig, width="stretch")
        import matplotlib.pyplot as plt
        plt.close(fig)

    st.markdown("##### 设备健康与处置优先级")
    health = pd.DataFrame(result.device_summary())
    health["处置优先级"] = health["最高严重度"].map(PRIORITY_BY_SEVERITY)
    health["建议动作"] = health["最高严重度"].map(ACTION_BY_SEVERITY)
    health = health.rename(columns={"最高严重度": "风险等级", "问题数": "异常数", "已越限": "已越限条数"})
    health_cols = ["设备ID", "设备名称", "设备类型", "安装位置", "责任人",
                   "风险等级", "异常数", "已越限条数", "处置优先级", "建议动作"]
    st.dataframe(health[health_cols], width="stretch", hide_index=True)

    st.markdown("##### 角色工作台")
    role_cols = st.columns(3)
    for col, (name, meta) in zip(role_cols, ROLE_PROFILES.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(meta["desc"])
                st.markdown(f"建议入口：{meta['entry']}")

    with st.expander("设备台账与指标摘要"):
        st.markdown("**设备台账**")
        st.dataframe(registry, width="stretch", hide_index=True)

        st.markdown("**各设备指标摘要**")
        metric_cols = [c for c in data.columns
                       if c not in ("时间", "设备ID", "设备名称", "设备型号", "设备类型",
                                    "故障代码", "环境温度_℃")
                       and pd.api.types.is_numeric_dtype(data[c])]
        summary = data.groupby("设备ID")[metric_cols].agg(["mean", "max"]).round(2)
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        st.dataframe(summary, width="stretch")

        with st.expander("查看原始数据（前 200 行）"):
            st.dataframe(data.head(200), width="stretch", hide_index=True)


# ================================================================ 页面：异常预警列表


def render_alerts():
    st.subheader("异常预警列表")
    st.caption("给设备运维人员：按风险和责任人筛选异常，展开即可看到趋势、证据链与手册处置步骤。")

    counts = result.by_severity()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("待立即处置（P0/P1）", counts.get(TRIP, 0) + counts.get(ALARM, 0))
    c2.metric("计划检修（P2）", counts.get(WARN, 0))
    c3.metric("缩短巡检（P3）", counts.get(ATTENTION, 0))
    c4.metric("涉及设备", f"{len({f.device_id for f in result.findings})} 台")

    if st.button("重新执行诊断", help="数据、手册或检索设置变化后可手动重算"):
        run_scan()
        st.rerun()

    nos = len(result.findings)
    if nos == 0:
        st.success("本次扫描未检出异常。")
        return

    f1, f2, f3 = st.columns(3)
    severities = ["全部"] + [s for s in (TRIP, ALARM, WARN, ATTENTION) if counts.get(s)]
    pick_sev = f1.selectbox("风险等级", severities, key="alert_sev")
    devices = ["全部"] + sorted({f.device_id for f in result.findings})
    pick_dev = f2.selectbox("设备", devices, key="alert_dev")
    owners = ["全部"] + sorted({f.owner for f in result.findings if f.owner})
    pick_owner = f3.selectbox("责任人", owners, key="alert_owner")

    shown = [
        f for f in result.findings
        if (pick_sev == "全部" or f.severity == pick_sev)
        and (pick_dev == "全部" or f.device_id == pick_dev)
        and (pick_owner == "全部" or f.owner == pick_owner)
    ]

    if not shown:
        st.success("当前筛选条件下没有异常结论。")
        return

    rows = []
    for f in shown:
        row = f.to_dict()
        row["风险等级"] = row.pop("严重度")
        row["处置优先级"] = PRIORITY_BY_SEVERITY[f.severity]
        row["建议动作"] = ACTION_BY_SEVERITY[f.severity]
        rows.append(row)

    table_cols = ["处置优先级", "风险等级", "设备ID", "设备名称", "监测指标",
                  "起始时间", "峰值", "提前量(小时)", "责任人", "检出层"]
    table = pd.DataFrame(rows)[table_cols].copy()
    # None 直接显示成 "None" 会让一线以为系统缺数据；统一成破折号。
    table["提前量(小时)"] = table["提前量(小时)"].map(
        lambda value: f"{value:.0f}" if pd.notna(value) else "—"
    )
    st.markdown(f"#### 待处理异常（{len(shown)} 条）")
    st.dataframe(table, width="stretch", hide_index=True)

    for finding in shown:
        advice = advice_for(finding)
        lead = f"　提前 {finding.lead_time_hours:.0f} 小时" if finding.lead_time_hours else ""
        with st.expander(
            f"[{PRIORITY_BY_SEVERITY[finding.severity]}] "
            f"{finding.device_id} {finding.device_name} — "
            f"{finding.metric_label}　峰值 {finding.peak_text}{lead}"
        ):
            st.markdown(
                f"**处置建议**：{ACTION_BY_SEVERITY[finding.severity]}　|　"
                f"**责任人**：{finding.owner or '未登记'}　|　"
                f"**安装位置**：{finding.location}"
            )
            col_chart, col_text = st.columns([3, 2])
            with col_chart:
                if advice and advice.chart and Path(advice.chart).exists():
                    st.image(str(advice.chart), width="stretch")
                else:
                    st.caption("该结论为多参数联合异常，无单一指标曲线。")
            with col_text:
                st.markdown("**检测依据**")
                for line in finding.evidence_lines():
                    st.markdown(f"- {line}")
                if finding.fault_codes:
                    st.markdown(f"**控制系统故障代码**：`{'/'.join(finding.fault_codes)}`")

            if advice:
                st.markdown("**手册依据**")
                st.caption(f"检索查询：{advice.query}")
                if advice.primary:
                    st.markdown(f"命中章节：**{advice.citation}**")
                    with st.container(border=True):
                        st.markdown(advice.steps_text)
                    others = advice.other_citations()
                    if others:
                        st.caption("相关章节：" + "；".join(others))
                else:
                    st.warning("未检索到对应章节，请人工查阅手册。")


# ================================================================ 页面：智能诊断


def render_diagnosis():
    st.subheader("智能诊断")
    st.caption("给技术支持与运维人员：左侧看确定性诊断结论与手册依据，右侧可就设备数据或手册内容继续追问。")

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("#### 诊断结论与手册依据")
        if result.findings:
            options = {
                f"{f.device_id}｜{f.device_name}｜{f.metric_label}｜{f.severity}": f
                for f in result.findings
            }
            picked = st.selectbox("选择一条异常", list(options), key="diag_finding")
            finding = options[picked]
            advice = advice_for(finding)

            m1, m2, m3 = st.columns(3)
            m1.metric("风险等级", finding.severity)
            m2.metric("峰值", finding.peak_text)
            m3.metric("提前量", f"{finding.lead_time_hours:.0f} 小时" if finding.lead_time_hours else "—")
            st.caption(
                f"设备：{finding.device_id} {finding.device_name}（{finding.model}）　|　"
                f"位置：{finding.location}　|　责任人：{finding.owner or '未登记'}　|　"
                f"异常时段：{finding.start:%m-%d %H:%M} — {finding.end:%m-%d %H:%M}"
            )

            if advice and advice.chart and Path(advice.chart).exists():
                st.image(str(advice.chart), width="stretch")
            elif "+" in finding.metric:
                st.caption("该结论为多参数联合异常，无单一指标曲线。")

            st.markdown("**检测依据**")
            for line in finding.evidence_lines():
                st.markdown(f"- {line}")

            st.markdown("**手册依据**")
            if advice and advice.primary:
                st.markdown(f"命中章节：**{advice.citation}**")
                with st.container(border=True):
                    st.markdown(advice.steps_text)
                others = advice.other_citations()
                if others:
                    st.caption("相关章节：" + "；".join(others))
            else:
                st.warning("未检索到对应章节，请人工查阅手册。")

            st.markdown(f"**建议动作**：{ACTION_BY_SEVERITY[finding.severity]}")
        else:
            st.info("本次扫描未检出异常。可在右侧直接提问手册问题。")

    with right:
        st.markdown("#### 故障问答")
        st.caption("模型会先调用知识库与诊断工具取数，再组织回答；数字不由模型估算。")

        if not api_key or not model_name:
            st.info("未配置模型时，左侧的检测结论与手册依据仍可正常使用；填入 API Key 后可继续追问。")

        with st.container(height=470, border=True):
            if not st.session_state.chat:
                st.caption("可以这样问：")
                st.markdown(
                    "- P-101 最近温度涨了多少？\n"
                    "- 离心泵发生汽蚀有哪些征兆？\n"
                    "- M-202 三个参数都正常，为什么说它有隐患？\n"
                    "- CXB-350 的额定电流是多少，超过多少要停机？"
                )
            for message in st.session_state.chat:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    for call in message.get("tools", []):
                        st.caption(f"🔧 调用 `{call.name}`（{call.args_text}）")

        question = st.chat_input("请输入设备或手册问题…", disabled=not (api_key and model_name),
                                 key="diag_chat")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("正在检索知识库并核查数据…"):
                    try:
                        llm = build_llm(provider, api_key, base_url=base_url,
                                        model=model_name, temperature=temperature)
                        ctx = AgentContext(scan=result, data=data, index=index,
                                           retrieval_mode=retrieval_mode, top_k=top_k)
                        run = answer_question(ctx, llm, question)
                        st.markdown(run.text)
                        for call in run.tool_calls:
                            st.caption(f"🔧 调用 `{call.name}`（{call.args_text}）")
                        st.session_state.chat.append({
                            "role": "assistant", "content": run.text,
                            "tools": run.tool_calls,
                        })
                    except Exception as exc:
                        st.error(f"回答失败：{friendly_error(exc)}")


# ================================================================ 页面：诊断报告


def render_report():
    st.subheader("诊断报告")
    st.caption("把异常证据、手册依据和处置排期整理成可交付报告，支持 Markdown / PDF 导出。")

    total = len(result.findings)
    alarmed = sum(1 for f in result.findings if f.is_alarm)
    cited = sum(1 for a in bundle.advices if a.primary)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("报告结论", f"{total} 条")
    c2.metric("已越限", f"{alarmed} 条")
    c3.metric("未越限隐患", f"{total - alarmed} 条")
    c4.metric("手册依据覆盖", f"{cited}/{total}")

    st.markdown("#### 处置排期建议")
    schedule = pd.DataFrame([{
        "优先级": PRIORITY_BY_SEVERITY[f.severity],
        "设备编号": f.device_id,
        "设备名称": f.device_name,
        "责任人": f.owner or "未登记",
        "待处置问题": f.metric_label,
        "建议动作": ACTION_BY_SEVERITY[f.severity],
    } for f in result.findings])
    if schedule.empty:
        st.info("本次扫描未检出异常，无需排期。")
    else:
        st.dataframe(schedule, width="stretch", hide_index=True)

    st.divider()
    col_gen, col_clear = st.columns([1, 3])
    want_narrative = col_gen.button("生成模型综合分析", type="primary")
    if col_clear.button("清除已生成的分析") and "narrative" in st.session_state:
        del st.session_state["narrative"]

    if want_narrative:
        if not api_key or not model_name:
            st.warning("请先在左侧「模型与检索配置」中填写 API Key 与模型名称。")
        else:
            with st.spinner("模型正在综合分析检测结果…"):
                try:
                    llm = build_llm(provider, api_key, base_url=base_url,
                                    model=model_name, temperature=temperature)
                    ctx = AgentContext(scan=bundle.scan, data=data, index=index,
                                       retrieval_mode=retrieval_mode, top_k=top_k)
                    run = generate_narrative(ctx, llm)
                    st.session_state["narrative"] = run.text
                except Exception as exc:
                    st.error(f"生成失败：{friendly_error(exc)}")

    narrative = st.session_state.get("narrative", "")
    markdown_text = render_markdown(
        ReportBundle(scan=bundle.scan, advices=bundle.advices), narrative=narrative
    )

    if narrative:
        st.markdown("#### 模型综合分析")
        st.markdown(narrative)

    st.markdown("#### 报告预览")
    with st.container(height=460, border=True):
        st.markdown(markdown_text)

    # 落盘用固定文件名覆盖写：报告页每次交互都会重跑脚本，若每次都用时间戳
    # 新建文件，reports_out/ 会被刷成几十份内容相同、只差分钟数的报告。
    stamp = datetime.now()
    timestamped_md, pdf_path = default_report_paths(stamp)
    on_disk_md = REPORT_OUT_DIR / "设备诊断报告.md"
    save_markdown(markdown_text, on_disk_md)

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ 下载 Markdown 报告", markdown_text.encode("utf-8"),
        file_name=timestamped_md.name, mime="text/markdown", width="stretch",
    )
    if d2.button("生成 PDF 报告", width="stretch"):
        with st.spinner("正在排版 PDF…"):
            try:
                markdown_to_pdf(markdown_text, pdf_path, base_dir=REPORT_OUT_DIR)
                st.session_state["pdf_bytes"] = pdf_path.read_bytes()
                st.session_state["pdf_name"] = pdf_path.name
            except Exception as exc:
                st.error(f"PDF 生成失败：{exc}")

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            "⬇️ 下载 PDF 报告", st.session_state["pdf_bytes"],
            file_name=st.session_state.get("pdf_name", "设备诊断报告.pdf"),
            mime="application/pdf", width="stretch",
        )
    st.caption(f"报告与图表同时保存在 `{REPORT_OUT_DIR.name}/` 目录下。")


# ================================================================ 页面：知识库


def render_kb():
    st.subheader("知识库与手册检索")
    st.caption("给技术支持人员：核对知识覆盖范围，直接用一线会问的问题检验检索效果。")

    c1, c2, c3 = st.columns(3)
    c1.metric("手册", f"{stats['documents']} 份")
    c2.metric("章节", f"{stats['parents']} 节")
    c3.metric("检索子块", f"{stats['children']} 块")

    st.markdown("#### 收录的章节")
    st.text(index.catalog(max_sections_per_doc=40))

    errors = getattr(index, "load_errors", None)
    if errors:
        st.warning("以下文件未能解析：\n" + "\n".join(f"- {e}" for e in errors))

    st.divider()
    st.markdown("#### 检索测试")
    st.caption("输入运维人员实际会问的问题，看能否命中正确章节。")
    probe = st.text_input("提问", placeholder="例如：泵发烫是怎么回事 / N203 是什么告警")
    if probe:
        hits = index.search(probe, top_k=top_k, mode=retrieval_mode)
        if not hits:
            st.warning("未召回任何章节。")
        for hit in hits:
            with st.expander(f"[{hit.rank}] {hit.citation}　（{hit.retriever}，{hit.score:.3f}）"):
                st.markdown(hit.parent.text[:1200])

    st.divider()
    st.caption(
        "评测集 `data/检索评测集.jsonl` 含 27 条带标注的查询，"
        "可用 `python scripts/eval_retrieval.py` 量化 Hit@k 与 MRR。"
    )


# ================================================================ 工作台导航

PAGE_RENDERERS = {
    "overview": render_overview,
    "alerts": render_alerts,
    "diagnosis": render_diagnosis,
    "report": render_report,
    "kb": render_kb,
}

order = PAGE_ORDER[role]
tabs = st.tabs([PAGE_LABELS[key] for key in order])
for tab, key in zip(tabs, order):
    with tab:
        PAGE_RENDERERS[key]()
