"""LangChain 标准模型适配层。

这一层的职责是把项目配置转换成 LangChain 可直接使用的 `BaseChatModel`，
从而让上层节点统一基于标准 Prompt / Tool / Runnable 组件开发。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


@dataclass
class LLMConfig:
    """描述一个模型连接所需的最小配置。"""

    base_url: str
    api_key: str
    model: str
    wire_api: Literal["chat_completions", "responses"] = "chat_completions"
    temperature: float = 0.2
    timeout_sec: int = 60


def build_chat_model(config: LLMConfig) -> BaseChatModel:
    """根据项目配置构建 LangChain 标准聊天模型。

    当前优先走兼容 OpenAI 的接口。
    - `chat_completions` 会使用标准 `/chat/completions`
    - `responses` 会启用 LangChain 对 Responses API 的适配
    """
    kwargs: dict[str, object] = {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.base_url.rstrip("/"),
        "temperature": config.temperature,
        "timeout": config.timeout_sec,
    }
    if config.wire_api == "responses":
        kwargs["use_responses_api"] = True
        kwargs["output_version"] = "responses/v1"

    return ChatOpenAI(**kwargs)
