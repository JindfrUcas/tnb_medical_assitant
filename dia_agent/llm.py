"""Minimal OpenAI-compatible chat client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    wire_api: str = "chat_completions"
    temperature: float = 0.2
    timeout_sec: int = 60


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMConfig):
        self._config = config

    def chat(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if self._config.wire_api == "responses":
            return self._responses(messages=messages, response_format=response_format)

        payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format

        response = requests.post(
            f"{self._config.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._config.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "\n".join(text_parts).strip()
        return ""

    def _responses(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "input": [self._to_responses_message(message) for message in messages],
        }
        if response_format:
            payload["text"] = {"format": response_format}

        response = requests.post(
            f"{self._config.base_url.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._config.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        text_parts: list[str] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content_item in item.get("content") or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if text:
                        text_parts.append(str(text))
        return "\n".join(text_parts).strip()

    def _to_responses_message(self, message: dict[str, Any]) -> dict[str, Any]:
        role = str(message.get("role") or "user")
        if role == "system":
            role = "developer"

        content = message.get("content")
        if isinstance(content, str):
            return {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            }

        converted_content: list[dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"}:
                    converted_content.append(
                        {
                            "type": "input_text",
                            "text": str(item.get("text", "")),
                        }
                    )
                    continue
                if item_type == "image_url":
                    image_url = item.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if url:
                        converted_content.append(
                            {
                                "type": "input_image",
                                "image_url": str(url),
                            }
                        )

        return {
            "type": "message",
            "role": role,
            "content": converted_content or [{"type": "input_text", "text": ""}],
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
