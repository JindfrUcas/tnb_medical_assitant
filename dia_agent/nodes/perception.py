"""Perception node: normalize multimodal input into PatientState."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from dia_agent.llm import OpenAICompatibleChatClient
from dia_agent.schemas import PatientState


INDICATOR_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_\-/]*)\s*[:：=]\s*(-?\d+(?:\.\d+)?)")
INLINE_INDICATOR_PATTERN = re.compile(r"\b(eGFR|HbA1c|ALT|AST|Scr|Cr|BMI)\b\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
SECTION_LABELS = ["疾病", "病史", "诊断", "并发症", "当前用药", "现用药", "用药"]
SECTION_STOP_PATTERN = "|".join(re.escape(label) for label in SECTION_LABELS)


def _split_tokens(text: str) -> list[str]:
    return [token.strip() for token in re.split(r"[，,、;；\n]+", text) if token.strip()]


def _extract_list_by_label(text: str, labels: list[str]) -> list[str]:
    for label in labels:
        pattern = re.compile(rf"{re.escape(label)}\s*[:：]\s*(.+?)(?=(?:{SECTION_STOP_PATTERN})\s*[:：]|$)")
        match = pattern.search(text)
        if match:
            return _split_tokens(match.group(1))
    return []


class PerceptionNode:
    def __init__(self, llm_client: OpenAICompatibleChatClient | None = None):
        self._llm_client = llm_client

    def run(self, raw_input: str | dict[str, Any], history_text: str = "") -> PatientState:
        if isinstance(raw_input, dict):
            if {"indicators", "diseases", "current_drugs"}.issubset(raw_input.keys()):
                return PatientState.model_validate(raw_input)
            return self._run_with_modal_payload(raw_input, history_text)

        text = raw_input.strip()
        payload = self._try_parse_json(text)
        if payload is not None:
            return PatientState.model_validate(payload)

        return self._parse_text_state(text, history_text)

    def _run_with_modal_payload(self, payload: dict[str, Any], history_text: str) -> PatientState:
        text = str(payload.get("text") or "").strip()
        merged_history = str(payload.get("history_text") or history_text or "").strip()
        image_paths = self._normalize_image_paths(payload.get("image_paths") or payload.get("images"))

        extracted_by_vision = self._extract_from_images(text, image_paths, merged_history)
        if extracted_by_vision is not None:
            return PatientState.model_validate(extracted_by_vision)

        state = self._parse_text_state(text, merged_history)
        merged = state.model_dump()
        for key in ["indicators", "diseases", "current_drugs"]:
            value = payload.get(key)
            if value is not None:
                merged[key] = value
        return PatientState.model_validate(merged)

    def _normalize_image_paths(self, value: Any) -> list[Path]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        paths: list[Path] = []
        for item in value:
            path = Path(str(item)).expanduser()
            if path.exists() and path.is_file():
                paths.append(path)
        return paths

    def _extract_from_images(
        self,
        text: str,
        image_paths: list[Path],
        history_text: str,
    ) -> dict[str, Any] | None:
        if not image_paths or self._llm_client is None:
            return None

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "请从图片与补充文本中提取患者结构化状态，输出严格 JSON，字段必须为: "
                    "indicators(字典), diseases(字符串数组), current_drugs(字符串数组)。"
                    "仅输出 JSON，不要额外解释。\n"
                    f"补充文本: {text or '无'}\n"
                    f"病史文本: {history_text or '无'}"
                ),
            }
        ]
        for path in image_paths:
            data_url = self._to_data_url(path)
            if not data_url:
                continue
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = [
            {
                "role": "system",
                "content": "你是医疗结构化抽取助手。必须返回可被 JSON 解析的对象。",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        raw_text = ""
        try:
            raw_text = self._llm_client.chat(messages, response_format={"type": "json_object"})
        except Exception:
            try:
                raw_text = self._llm_client.chat(messages)
            except Exception:
                return None

        parsed = self._parse_json_like(raw_text)
        if not isinstance(parsed, dict):
            return None
        if not {"indicators", "diseases", "current_drugs"}.issubset(parsed.keys()):
            return None
        return parsed

    def _to_data_url(self, path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        mime = mime_type or "image/png"
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _parse_json_like(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    def _parse_text_state(self, text: str, history_text: str) -> PatientState:
        indicators = self._extract_indicators(text)
        diseases = _extract_list_by_label(text, ["疾病", "病史", "诊断"]) or self._extract_history_diseases(history_text)
        current_drugs = _extract_list_by_label(text, ["当前用药", "现用药", "用药"])
        return PatientState(indicators=indicators, diseases=diseases, current_drugs=current_drugs)

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if not {"indicators", "diseases", "current_drugs"}.issubset(payload.keys()):
            return None
        return payload

    def _extract_indicators(self, text: str) -> dict[str, float]:
        indicators: dict[str, float] = {}
        for name, value in INDICATOR_PATTERN.findall(text):
            indicators[name.strip()] = float(value)
        for name, value in INLINE_INDICATOR_PATTERN.findall(text):
            indicators[name.strip()] = float(value)
        return indicators

    def _extract_history_diseases(self, history_text: str) -> list[str]:
        if not history_text:
            return []
        return _extract_list_by_label(history_text, ["病史", "并发症", "诊断"])
