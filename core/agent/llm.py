"""模型接入：按服务商预设构造 OpenAI / Anthropic 两种协议的客户端。

两套协议的接口形态不同，不能混用：OpenAI 兼容端点走 ChatOpenAI，
Anthropic 兼容端点走 ChatAnthropic。预设表放在 core.config，界面与脚本共用。
"""

from __future__ import annotations

from ..config import PROVIDER_PRESETS


def provider_names() -> list[str]:
    return list(PROVIDER_PRESETS.keys())


def build_llm(
    provider: str,
    api_key: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
):
    """按服务商构造对话模型。

    temperature 默认 0：诊断结论必须可复现，同一份数据两次分析结果不一致
    是运维场景里不能接受的。需要发散描述时才调高。
    """
    if provider not in PROVIDER_PRESETS:
        raise ValueError(f"未知服务商：{provider}")
    preset = PROVIDER_PRESETS[provider]
    protocol = preset["protocol"]
    base_url = (base_url or preset["base_url"]).strip()
    model = (model or preset["model"]).strip()

    if not api_key:
        raise ValueError("缺少 API Key")
    if not model:
        raise ValueError("缺少模型名称")

    if protocol == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = dict(
            model=model, temperature=temperature,
            anthropic_api_key=api_key, timeout=timeout,
        )
        if base_url:
            kwargs["anthropic_api_url"] = base_url
        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    kwargs = dict(
        model=model, temperature=temperature,
        openai_api_key=api_key, timeout=timeout,
    )
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def friendly_error(exc: Exception) -> str:
    """把各家五花八门的报错翻译成一句人能看懂的话。"""
    msg = str(exc)
    if "Insufficient Balance" in msg or "insufficient_quota" in msg:
        return "账户余额不足或配额已用尽，请检查服务商账户。"
    if "401" in msg or "Unauthorized" in msg or "invalid_api_key" in msg:
        return "API Key 无效或已过期。"
    if "404" in msg or "model_not_found" in msg:
        return "模型名称不存在，请确认该服务商提供的模型名。"
    if "Connection" in msg or "timed out" in msg or "Timeout" in msg:
        return "无法连接到服务商，请检查网络与 Base URL。"
    if "429" in msg or "rate" in msg.lower():
        return "请求过于频繁，请稍后重试。"
    return msg
