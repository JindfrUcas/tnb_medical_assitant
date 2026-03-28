"""项目配置模块。

这里集中定义 Dia-Agent 运行时需要的环境变量与默认值。
项目启动时会优先读取根目录 `.env`，并自动把 `DIA_AGENT_` 前缀映射到对应字段。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Dia-Agent 的运行时配置。

    这一层的目标是把“环境变量世界”转换成“强类型 Python 配置对象”，
    这样上层 Pipeline / API / UI 都只依赖 `Settings`，不需要到处直接读环境变量。
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DIA_AGENT_", extra="ignore")

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str | None = Field(default=None)
    neo4j_database: str = Field(default="neo4j")

    graph_json_path: Path = Field(default=Path("dataset/graph.json"))
    guideline_pdf_paths: list[Path] = Field(
        default_factory=lambda: [
            Path("dataset/中国糖尿病防治指南（2024版）.pdf"),
        ]
    )

    chroma_persist_dir: Path = Field(default=Path("data/chroma"))
    chroma_collection: str = Field(default="dia_guidelines")
    embedding_model: str = Field(default="moka-ai/m3e-base")
    embedding_device: str = Field(default="cpu")
    retrieval_k: int = Field(default=4)
    default_rag_query: str = Field(default="1型糖尿病 肾功能不全 用药")

    llm_base_url: str | None = Field(default=None)
    llm_api_key: str | None = Field(default=None)
    llm_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    llm_wire_api: str = Field(default="chat_completions")
    llm_temperature: float = Field(default=0.2)
    llm_timeout_sec: int = Field(default=60)

    vision_base_url: str | None = Field(default=None)
    vision_api_key: str | None = Field(default=None)
    vision_model: str = Field(default="Qwen/Qwen2-VL-7B-Instruct")
    vision_wire_api: str | None = Field(default=None)

    max_reasoner_retries: int = Field(default=3)
    react_max_steps: int = Field(default=4)

    api_title: str = Field(default="Dia-Agent API")
    api_version: str = Field(default="0.1.0")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)

    ui_page_title: str = Field(default="Dia-Agent")
    ui_layout: str = Field(default="wide")
    ui_app_title: str = Field(default="Dia-Agent: Guardrail-First 专病智能体")
    ui_default_raw_input: str = Field(
        default='{"indicators": {"eGFR": 25, "HbA1c": 9.3}, "diseases": ["糖尿病酮症酸中毒"], "current_drugs": ["盐酸二甲双胍片 (普通片/缓释片)"]}'
    )
    ui_default_history_text: str = Field(default="")
    ui_upload_dir: Path = Field(default=Path("data/uploads"))

    @field_validator("guideline_pdf_paths", mode="before")
    @classmethod
    def normalize_guideline_pdf_paths(cls, value: Any) -> list[Path]:
        """支持从列表或逗号分隔字符串读取默认指南 PDF 列表。"""
        if value is None or value == "":
            return [
                Path("dataset/中国糖尿病防治指南（2024版）.pdf"),
            ]
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            return [Path(item) for item in items]
        if isinstance(value, list):
            return [Path(str(item)) for item in value if str(item).strip()]
        raise TypeError("guideline_pdf_paths must be a list or comma-separated string")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回配置对象。

    单独封装成函数，便于后续在测试里替换或注入自定义配置。
    """
    return Settings()
