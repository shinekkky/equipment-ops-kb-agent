"""工具调用循环与两个入口：生成报告综述、回答运维追问。

用模型原生 tool calling 驱动，而不是解析文本里的代码块：原生调用不依赖输出格式，
模型跑偏时（忘了格式、少写引号）不会整条链路崩掉，OpenAI 与 Anthropic 两套协议
也共用同一段循环代码。

**结构化 Trace**
----------------
每次工具调用记录 step、工具名、参数、完整返回值、状态、开始时间、耗时与错误；
每次 Agent 运行记录 trace_id、用户问题、最终回答、轮数、模型名、总耗时、Token
用量与错误信息。记录追加写入 ``experiments/traces/agent_traces.jsonl``，可按
trace_id 查询和导出（见 ``core/agent/trace_store.py`` 与 ``scripts/agent_trace.py``）。

这不是为了"看起来可观测"，而是为了回答三个具体问题：
  1. 这次回答到底查了哪些手册、取了哪些数？
  2. 哪个工具慢、哪个工具失败、模型在第几轮收口？
  3. 用户反馈答错时，能否复现当时的完整执行路径？
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from . import prompts
from .tools import AgentContext, build_tools
from .trace_store import append_trace

MAX_ROUNDS = 8          # 工具调用轮数上限，防止模型陷入死循环
MAX_RESULT_CHARS = 1600  # 回传给模型的单条工具结果上限


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _model_name(llm) -> str:
    for attr in ("model_name", "model", "_llm_type"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _usage(response) -> tuple[int | None, int | None, int | None]:
    """尽量从 LangChain 响应里取 Token 用量。取不到就返回 None，不猜。"""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
        completion = usage.get("output_tokens", usage.get("completion_tokens"))
        total = usage.get("total_tokens")
        return prompt, completion, total

    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(usage, dict) and usage:
        prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
        completion = usage.get("output_tokens", usage.get("completion_tokens"))
        total = usage.get("total_tokens")
        if total is None and (prompt is not None or completion is not None):
            total = (prompt or 0) + (completion or 0)
        return prompt, completion, total
    return None, None, None


@dataclass
class ToolCall:
    """一次工具调用的结构化执行记录。"""

    name: str
    args: dict
    result: str
    step: int = 0
    status: str = "success"          # success | error
    start_time: str = ""
    duration_ms: int = 0
    error: str | None = None
    trace_id: str = ""

    @property
    def args_text(self) -> str:
        return "，".join(f"{k}={v}" for k, v in self.args.items() if v)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "step": self.step,
            "tool_name": self.name,
            "arguments": self.args,
            "result": self.result,
            "status": self.status,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class AgentRun:
    """一次 Agent 调用的完整结果与执行元数据。"""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    rounds: int = 0
    error: str = ""
    trace_id: str = ""
    kind: str = "qa"                 # qa | narrative
    question: str = ""
    model_name: str = ""
    started_at: str = ""
    duration_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "question": self.question,
            "final_answer": self.text,
            "tool_call_count": self.tool_call_count,
            "rounds": self.rounds,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error or None,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "metadata": self.metadata,
        }


def _persist(run: AgentRun, enabled: bool) -> None:
    if enabled:
        append_trace(run.to_dict())


def run_tool_loop(
    llm,
    tools: list,
    messages: list,
    *,
    max_rounds: int = MAX_ROUNDS,
    trace_id: str | None = None,
    question: str = "",
    model_name: str = "",
    kind: str = "qa",
    persist: bool = False,
    metadata: dict | None = None,
) -> AgentRun:
    """执行"模型思考 → 调用工具 → 回传结果 → 继续思考"的循环，直到模型给出终答。"""
    started_at = _now_iso()
    clock = time.perf_counter()
    trace_id = trace_id or uuid.uuid4().hex
    model_name = model_name or _model_name(llm)
    by_name = {t.name: t for t in tools}
    trace: list[ToolCall] = []
    prompt_tokens = completion_tokens = total_tokens = 0
    usage_seen = False

    def finish(text: str, rounds: int, error: str = "", token_usage=None) -> AgentRun:
        nonlocal prompt_tokens, completion_tokens, total_tokens, usage_seen
        if token_usage:
            p, c, t = token_usage
            if p is not None:
                prompt_tokens += p; usage_seen = True
            if c is not None:
                completion_tokens += c; usage_seen = True
            if t is not None:
                total_tokens += t; usage_seen = True
        run = AgentRun(
            text=text,
            tool_calls=trace,
            rounds=rounds,
            error=error,
            trace_id=trace_id,
            kind=kind,
            question=question,
            model_name=model_name,
            started_at=started_at,
            duration_ms=int((time.perf_counter() - clock) * 1000),
            prompt_tokens=prompt_tokens if usage_seen else None,
            completion_tokens=completion_tokens if usage_seen else None,
            total_tokens=total_tokens if usage_seen else None,
            metadata=metadata or {},
        )
        _persist(run, persist)
        return run

    try:
        bound = llm.bind_tools(tools)
    except Exception as exc:
        # 少数端点不支持 tool calling，退回纯对话：仍然能答，只是查不了手册
        reply = llm.invoke(messages)
        return finish(
            text=str(reply.content),
            rounds=0,
            error=f"该模型不支持工具调用，已降级为纯对话：{exc}",
            token_usage=_usage(reply),
        )

    response = None
    round_index = 0
    try:
        for round_index in range(1, max_rounds + 1):
            response = bound.invoke(messages)
            usage = _usage(response)
            messages.append(response)

            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                return finish(
                    text=str(response.content or ""),
                    rounds=round_index,
                    token_usage=usage,
                )

            for call_index, call in enumerate(calls, start=1):
                fn = by_name.get(call["name"])
                tool_start = _now_iso()
                tool_clock = time.perf_counter()
                status = "success"
                error = None
                try:
                    result = str(fn.invoke(call.get("args") or {})) if fn else ""
                    if fn is None:
                        status = "error"
                        error = f"未知工具：{call['name']}"
                        result = error
                except Exception as exc:
                    # 工具自身异常不该中断整轮对话：把它当作结果回传，模型会据此调整
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                    result = f"工具执行失败：{exc}"

                record = ToolCall(
                    name=call["name"],
                    args=call.get("args") or {},
                    result=result,
                    step=len(trace) + 1,
                    status=status,
                    start_time=tool_start,
                    duration_ms=int((time.perf_counter() - tool_clock) * 1000),
                    error=error,
                    trace_id=trace_id,
                )
                trace.append(record)
                messages.append(ToolMessage(
                    content=result[:MAX_RESULT_CHARS],
                    tool_call_id=call["id"],
                ))
            # 本轮 Token 在进入下一轮前累加
            if usage:
                p, c, t = usage
                if p is not None:
                    prompt_tokens += p; usage_seen = True
                if c is not None:
                    completion_tokens += c; usage_seen = True
                if t is not None:
                    total_tokens += t; usage_seen = True
    except Exception as exc:
        # 模型侧异常：先落一条失败记录，再把异常交给界面处理
        run = AgentRun(
            text="",
            tool_calls=trace,
            rounds=round_index,
            error=f"{type(exc).__name__}: {exc}",
            trace_id=trace_id,
            kind=kind,
            question=question,
            model_name=model_name,
            started_at=started_at,
            duration_ms=int((time.perf_counter() - clock) * 1000),
            prompt_tokens=prompt_tokens if usage_seen else None,
            completion_tokens=completion_tokens if usage_seen else None,
            total_tokens=total_tokens if usage_seen else None,
            metadata=metadata or {},
        )
        _persist(run, persist)
        raise

    # 轮数用尽仍未收口：把最后一条有内容的回复交出去，而不是抛错
    text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = str(msg.content)
            break
    return finish(
        text=text or "（已达到工具调用轮数上限，未能形成最终结论，请缩小问题范围后重试。）",
        rounds=max_rounds,
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
    started_at = _now_iso()
    clock = time.perf_counter()
    trace_id = uuid.uuid4().hex
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        run = AgentRun(
            text="", error=f"{type(exc).__name__}: {exc}", trace_id=trace_id,
            kind="narrative", question="生成模型综合分析",
            model_name=_model_name(llm), started_at=started_at,
            duration_ms=int((time.perf_counter() - clock) * 1000),
        )
        _persist(run, True)
        raise

    p, c, t = _usage(response)
    run = AgentRun(
        text=str(response.content or ""),
        trace_id=trace_id,
        kind="narrative",
        question="生成模型综合分析",
        model_name=_model_name(llm),
        started_at=started_at,
        duration_ms=int((time.perf_counter() - clock) * 1000),
        prompt_tokens=p,
        completion_tokens=c,
        total_tokens=t,
    )
    _persist(run, True)
    return run


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

    return run_tool_loop(
        llm,
        tools,
        messages,
        question=question,
        kind="qa",
        persist=True,
    )
