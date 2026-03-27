"""Streamlit 前端入口。

当前页面以“直接体验最终 Agent 输出”为目标，尽量保持界面简单。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # 直接从子目录启动 Streamlit 时，手动把项目根目录加入导包路径。
    sys.path.insert(0, str(ROOT))

from dia_agent.config import get_settings
from dia_agent.pipeline import DiaAgentPipeline

settings = get_settings()

st.set_page_config(page_title=settings.ui_page_title, layout=settings.ui_layout)
st.title(settings.ui_app_title)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = DiaAgentPipeline()

pipeline: DiaAgentPipeline = st.session_state.pipeline

raw_input = st.text_area(
    "输入患者信息（支持结构化 JSON 或文本）",
    value=settings.ui_default_raw_input,
    height=180,
)
rag_query = st.text_input("RAG 检索查询（可留空）", value=settings.default_rag_query)
history_text = st.text_area("历史病史（可选）", value=settings.ui_default_history_text, height=100)
uploaded_images = st.file_uploader(
    "上传化验单图片（可选，多张）",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
)


def parse_raw(value: str) -> str | dict:
    """优先尝试解析成 JSON，失败则保留为普通文本。"""
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return value


def save_uploaded_images() -> list[str]:
    """保存上传图片并返回文件路径列表。"""
    if not uploaded_images:
        return []
    upload_dir = Path(settings.ui_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for file in uploaded_images:
        suffix = Path(file.name).suffix.lower() or ".png"
        output_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
        output_path.write_bytes(file.getbuffer())
        saved_paths.append(str(output_path))
    return saved_paths


if st.button("开始问诊", type="primary"):
    # 前端只负责收集输入与展示结果，核心业务都走 Pipeline。
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
    with st.expander("执行轨迹", expanded=False):
        st.code("\n".join(result.trace) if result.trace else "无轨迹")
    with st.expander("安全白皮书", expanded=False):
        st.text(result.guardrail_report.whitepaper)
    with st.expander("审计结果", expanded=False):
        st.json(result.audit_result.model_dump())
