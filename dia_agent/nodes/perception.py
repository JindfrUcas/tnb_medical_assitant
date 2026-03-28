"""Perception 节点：把原始输入转换成标准患者状态。

这一层的目标是把文本、JSON、图片等多种输入形式尽量收口成统一的 `PatientState`。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dia_agent.schemas import PatientState
from dia_agent.utils import normalize_message_content


INDICATOR_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_\-/]*)\s*[:：=]\s*(-?\d+(?:\.\d+)?)")
INLINE_INDICATOR_PATTERN = re.compile(r"\b(eGFR|HbA1c|ALT|AST|Scr|Cr|BMI)\b\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
SECTION_LABELS = ["疾病", "病史", "诊断", "并发症", "当前用药", "现用药", "用药"]
SECTION_STOP_PATTERN = "|".join(re.escape(label) for label in SECTION_LABELS)


def _split_tokens(text: str) -> list[str]:
    """按常见中英文分隔符切分列表内容。"""
    return [token.strip() for token in re.split(r"[，,、;；\n]+", text) if token.strip()]


def _extract_list_by_label(text: str, labels: list[str]) -> list[str]:
    """根据“疾病/病史/用药”等标签从文本中提取列表字段。"""
    for label in labels:
        pattern = re.compile(rf"{re.escape(label)}\s*[:：]\s*(.+?)(?=(?:{SECTION_STOP_PATTERN})\s*[:：]|$)")
        match = pattern.search(text)
        if match:
            return _split_tokens(match.group(1))
    return []


class PerceptionNode:
    """负责输入标准化的节点。"""

    def __init__(self, llm_client: BaseChatModel | None = None):
        """初始化感知节点，可选接入多模态模型。"""
        self._llm_client = llm_client

    def run(self, raw_input: str | dict[str, Any], history_text: str = "") -> PatientState:
        """把各种输入格式统一转换成 `PatientState`。"""
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
        """处理带图片或额外文本字段的多模态输入。"""
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
        """把输入中的图片路径统一归一化成 `Path` 列表。"""
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
        """调用多模态模型，从图片与补充文本中抽取结构化患者状态。"""
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
            SystemMessage(content="你是医疗结构化抽取助手。必须返回结构化患者状态。"),
            HumanMessage(content=user_content),
        ]

        try:
            structured_model = self._llm_client.with_structured_output(PatientState)
            response = structured_model.invoke(messages)
            if isinstance(response, PatientState):
                return response.model_dump()
            if isinstance(response, dict):
                return PatientState.model_validate(response).model_dump()
        except Exception:
            pass

        try:
            raw_message = self._llm_client.invoke(messages)
            raw_text = self._message_to_text(raw_message)
        except Exception:
            return None

        parsed = self._parse_json_like(raw_text)
        if not isinstance(parsed, dict):
            return None
        if not {"indicators", "diseases", "current_drugs"}.issubset(parsed.keys()):
            return None
        return parsed

    def _message_to_text(self, message: Any) -> str:
        """把 LangChain 消息对象中的文本内容提取出来。"""
        return normalize_message_content(getattr(message, "content", ""))

    def _to_data_url(self, path: Path) -> str:
        """把本地图片转成 data URL，便于直接发给兼容 OpenAI 的多模态接口。"""
        mime_type, _ = mimetypes.guess_type(path.name)
        mime = mime_type or "image/png"
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _parse_json_like(self, text: str) -> dict[str, Any] | None:
        """尽量从模型输出中恢复出 JSON 对象。"""
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
        """基于规则从纯文本里抽取指标、疾病和当前用药。"""
        indicators = self._extract_indicators(text)
        diseases = _extract_list_by_label(text, ["疾病", "病史", "诊断"]) or self._extract_history_diseases(history_text)
        current_drugs = _extract_list_by_label(text, ["当前用药", "现用药", "用药"])
        return PatientState(indicators=indicators, diseases=diseases, current_drugs=current_drugs)

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """如果输入本身就是 JSON 字符串，则直接解析。"""
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
        """从文本中抽取常见临床指标。"""
        indicators: dict[str, float] = {}
        for name, value in INDICATOR_PATTERN.findall(text):
            indicators[name.strip()] = float(value)
        for name, value in INLINE_INDICATOR_PATTERN.findall(text):
            indicators[name.strip()] = float(value)
        return indicators

    def _extract_history_diseases(self, history_text: str) -> list[str]:
        """从病史补充信息中提取疾病项。"""
        if not history_text:
            return []
        return _extract_list_by_label(history_text, ["病史", "并发症", "诊断"])
