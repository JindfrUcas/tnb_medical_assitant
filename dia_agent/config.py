"""Project configuration and environment settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for Dia-Agent."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DIA_AGENT_", extra="ignore")

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str | None = Field(default=None)
    neo4j_database: str = Field(default="neo4j")

    graph_json_path: Path = Field(default=Path("dataset/graph.json"))

    chroma_persist_dir: Path = Field(default=Path("data/chroma"))
    chroma_collection: str = Field(default="dia_guidelines")
    embedding_model: str = Field(default="moka-ai/m3e-base")
    retrieval_k: int = Field(default=4)

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


def get_settings() -> Settings:
    return Settings()
