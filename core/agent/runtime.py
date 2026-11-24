"""工具调用循环与两个入口：生成报告综述、回答运维追问。

用模型原生 tool calling 驱动，而不是解析文本里的代码块：原生调用不依赖输出格式，
模型跑偏时（忘了格式、少写引号）不会整条链路崩掉，OpenAI 与 Anthropic 两套协议
也共用同一段循环代码。

循环里记录每一步调用了什么工具、返回了什么，界面可以直接展示这个过程——
"Agent 先查了知识库、又核了原始数据"这件事看得见，比只给一个最终答案可信得多。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from . import prompts
from .tools import AgentContext, build_tools

MAX_ROUNDS = 8          # 工具调用轮数上限，防止模型陷入死循环
MAX_RESULT_CHARS = 1600  # 回传给模型的单条工具结果上限


@dataclass
class ToolCall:
    """一次工具调用记录，供界面展示。"""

    name: str
    args: dict
    result: str

    @property
    def args_text(self) -> str:
        return "，".join(f"{k}={v}" for k, v in self.args.items() if v)


@dataclass
class AgentRun:
    """一次 Agent 调用的完整结果。"""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    rounds: int = 0
    error: str = ""


def run_tool_loop(llm, tools: list, messages: list, *, max_rounds: int = MAX_ROUNDS) -> AgentRun:
    """执行"模型思考 → 调用工具 → 回传结果 → 继续思考"的循环，直到模型给出终答。"""
    by_name = {t.name: t for t in tools}
    trace: list[ToolCall] = []

    try:
        bound = llm.bind_tools(tools)
    except Exception as exc:
        # 少数端点不支持 tool calling，退回纯对话：仍然能答，只是查不了手册
        reply = llm.invoke(messages)
        return AgentRun(text=str(reply.content), error=f"该模型不支持工具调用，已降级为纯对话：{exc}")

    response = None
    for round_index in range(1, max_rounds + 1):
        response = bound.invoke(messages)
        messages.append(response)

        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            return AgentRun(text=str(response.content or ""), tool_calls=trace, rounds=round_index)

        for call in calls:
            fn = by_name.get(call["name"])
            try:
                result = str(fn.invoke(call.get("args") or {})) if fn else f"未知工具：{call['name']}"
            except Exception as exc:
                # 工具自身异常不该中断整轮对话：把它当作结果回传，模型会据此调整
                result = f"工具执行失败：{exc}"

            trace.append(ToolCall(name=call["name"], args=call.get("args") or {},
                                  result=result))
            messages.append(ToolMessage(content=result[:MAX_RESULT_CHARS],
                                        tool_call_id=call["id"]))

    # 轮数用尽仍未收口：把最后一条有内容的回复交出去，而不是抛错
    text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = str(msg.content)
            break
    return AgentRun(
        text=text or "（已达到工具调用轮数上限，未能形成最终结论，请缩小问题范围后重试。）",
        tool_calls=trace, rounds=max_rounds,
    )


# ---------------------------------------------------------------- 入口


def generate_narrative(ctx: AgentContext, llm) -> AgentRun:
    """生成报告的"模型综合分析"段落。

    这里不挂工具：结论与手册依据都已经由确定性流程算好并塞进上下文，
    再给模型工具只会让它反复检索同样几节，徒增延迟与费用。
    """
    messages = [
        SystemMessage(content=prompts.narrative_system()),
        HumanMessage(content=prompts.findings_digest(ctx.scan)),
    ]
    response = llm.invoke(messages)
    return AgentRun(text=str(response.content or ""))


def answer_question(
    ctx: AgentContext,
    llm,
    question: str,
    history: list | None = None,
) -> AgentRun:
    """回答运维人员的追问，允许调用知识库与诊断工具。"""
    tools = build_tools(ctx)
    catalog = prompts.knowledge_catalog(ctx.index)

    messages: list = [
        SystemMessage(content=(
            f"{prompts.qa_system()}\n\n"
            f"当前知识库收录的手册与章节：\n{catalog}\n\n"
            f"本次监测区间：{ctx.scan.start:%Y-%m-%d} ~ {ctx.scan.end:%Y-%m-%d}，"
            f"共 {len(ctx.scan.devices)} 台设备、{len(ctx.scan.findings)} 条异常结论。"
        )),
    ]
    messages.extend(history or [])
    messages.append(HumanMessage(content=question))

    return run_tool_loop(llm, tools, messages)
