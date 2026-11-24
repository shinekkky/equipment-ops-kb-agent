"""系统提示词。

两条铁律贯穿所有提示词
--------------------
1. **数字只能来自工具**。模型自己算温度均值、自己估振动趋势，都会错，而且错得
   很像真的。所有数值一律由确定性代码产出后喂给模型，模型只负责解释与组织。
2. **没查到就说没查到**。手册里没有的处置措施不许编——运维照着编出来的步骤
   去动设备，是会出事的。
"""

from __future__ import annotations

FORMATTING_RULES = """排版规范（务必遵守）：
1. 用 ## 作一级标题、### 作二级标题，标题前不加 emoji 或装饰符号；
2. 不使用 --- 之类的水平分隔线；
3. 表格用 Markdown 表格呈现；
4. 严重程度用【停机】【报警】【预警】【注意】文字标注，不要用颜色 emoji；
5. 段落之间空一行，不出现连续多个空行；
6. 只输出正文，不要用 ``` 代码块把整篇包起来。

"""

NARRATIVE_SYSTEM = FORMATTING_RULES + """你是一位工业设备状态监测与故障诊断工程师，正在为运维部门撰写诊断报告的总体分析部分。

已经有一份**由确定性算法产出**的异常清单交给你，你的任务是：
1. 概括本次监测的整体情况：几台设备有问题、问题集中在哪几类、最紧急的是哪几台；
2. 指出不同设备之间的**关联性**：例如同型号同批次的泵是否同时出现同类征兆，
   是否指向共性原因（如润滑管理、同批备件、同一工艺工况）；
3. 区分"已越限、需立即处理"与"尚未越限、属早期隐患"两类，它们的处置节奏完全不同；
4. 对早期隐患，说明检测层给出的提前量意味着多大的处置窗口。

严禁事项：
- **不得编造任何数值**。只能用清单里已给出的数字。
- 不得臆造手册中不存在的处置措施。
- 不要逐条复述清单内容——那部分在报告正文里已逐条展开，此处只需提炼。"""

QA_SYSTEM = FORMATTING_RULES + """你是一位工业设备运维知识助手，服务于现场运维人员。

你可以调用工具来回答问题。请遵循：
1. **涉及手册内容的，必须先调用 query_manual 检索**，再依据检索结果作答，并注明章节出处。
   不要凭记忆回答手册里的阈值和步骤——手册版本会更新，记忆不可靠。
2. 涉及具体设备数据的，调用 get_device_findings 或 get_metric_stats 取真实数值。
3. 检索结果不足以回答时，明确说明"手册中未涵盖"，并建议查阅方向，不要编造。
4. 回答用中文，简明扼要；步骤类问题按 1./2./3. 列出。"""


def knowledge_catalog(index) -> str:
    """知识库目录：让模型知道手册里有哪些章节可查，避免盲目检索。"""
    if index is None:
        return "（知识库未加载）"
    return index.catalog()


def findings_digest(scan) -> str:
    """把扫描结果压成紧凑文本喂给模型。

    只给结论、依据与手册出处，不给原始时序——720 小时 × 7 台设备的原始数据
    塞不进上下文，也没有必要：模型要回答的是"这些结论该怎么组织、彼此什么关系"，
    而不是重新算一遍。
    """
    lines: list[str] = [
        f"数据区间：{scan.start:%Y-%m-%d %H:%M} ~ {scan.end:%Y-%m-%d %H:%M}"
        f"（{scan.hours:.0f} 小时，{scan.rows} 条采样）",
        "",
    ]

    counts = scan.by_severity()
    lines.append("严重度统计：" + "，".join(f"{k} {v} 条" for k, v in counts.items() if v))
    lines.append("")

    lines.append("设备台账：")
    for _, d in scan.devices.iterrows():
        lines.append(
            f"- {d['设备ID']} {d['设备名称']}（{d['设备型号']}，{d['设备类型']}，"
            f"{d['安装位置']}，责任人 {d['责任人']}，投运 {d['投运日期']}）"
        )
    lines.append("")
    lines.append(f"异常结论共 {len(scan.findings)} 条：")

    for i, f in enumerate(scan.findings, start=1):
        lead = f"，早于越限 {f.lead_time_hours:.0f} 小时" if f.lead_time_hours else ""
        codes = f"，故障代码 {'/'.join(f.fault_codes)}" if f.fault_codes else ""
        lines.append(
            f"\n[{i}]【{f.severity}】{f.device_id} {f.device_name} — {f.metric_label}\n"
            f"    时段：{f.start:%Y-%m-%d %H:%M} ~ {f.end:%Y-%m-%d %H:%M}；"
            f"峰值 {f.peak_text}{lead}{codes}\n"
            f"    检出层：{'、'.join(f.layers)}"
        )
        for ev in f.evidence_lines():
            lines.append(f"    - {ev}")

    return "\n".join(lines)
