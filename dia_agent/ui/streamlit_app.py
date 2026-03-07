"""Streamlit UI for transparent Dia-Agent demonstration."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dia_agent.pipeline import DiaAgentPipeline


st.set_page_config(page_title="Dia-Agent", layout="wide")
st.title("Dia-Agent: Guardrail-First 专病智能体")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = DiaAgentPipeline()

pipeline: DiaAgentPipeline = st.session_state.pipeline

raw_input = st.text_area(
    "输入患者信息（支持结构化 JSON 或文本）",
    value='{"indicators": {"eGFR": 25, "HbA1c": 9.3}, "diseases": ["糖尿病酮症酸中毒"], "current_drugs": ["盐酸二甲双胍片 (普通片/缓释片)"]}',
    height=180,
)
rag_query = st.text_input("RAG 检索查询（可留空）", value="1型糖尿病 肾功能不全 用药")
history_text = st.text_area("历史病史（可选）", height=100)
uploaded_images = st.file_uploader(
    "上传化验单图片（可选，多张）",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
)


def parse_raw(value: str) -> str | dict:
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return value


def save_uploaded_images() -> list[str]:
    if not uploaded_images:
        return []
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for file in uploaded_images:
        suffix = Path(file.name).suffix.lower() or ".png"
        output_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
        output_path.write_bytes(file.getbuffer())
        saved_paths.append(str(output_path))
    return saved_paths


if st.button("开始问诊", type="primary"):
    parsed_raw = parse_raw(raw_input)
    image_paths = save_uploaded_images()

    if image_paths:
        if isinstance(parsed_raw, dict) and {"indicators", "diseases", "current_drugs"}.issubset(parsed_raw.keys()):
            consult_payload = parsed_raw
        elif isinstance(parsed_raw, dict):
            consult_payload = {
                "text": json.dumps(parsed_raw, ensure_ascii=False),
                "image_paths": image_paths,
                "history_text": history_text,
            }
        else:
            consult_payload = {
                "text": parsed_raw,
                "image_paths": image_paths,
                "history_text": history_text,
            }
    else:
        consult_payload = parsed_raw
    result = pipeline.consult(consult_payload, rag_query=rag_query, history_text=history_text)

    st.subheader("Agent 输出")
    st.write(result.reasoner_result.recommendation)
