"""FastAPI 接口入口。

这一层把本地 Pipeline 包装成 HTTP 服务，便于前端或第三方系统调用。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from dia_agent.config import get_settings
from dia_agent.pipeline import DiaAgentPipeline


class ConsultRequest(BaseModel):
    """问诊请求体。"""

    raw_input: str | dict[str, Any]
    rag_query: str = ""
    history_text: str = ""


class HealthResponse(BaseModel):
    """健康检查返回体。"""

    status: str = Field(default="ok")


settings = get_settings()
app = FastAPI(title=settings.api_title, version=settings.api_version)
pipeline = DiaAgentPipeline()


@app.on_event("shutdown")
def shutdown_event() -> None:
    """服务关闭时释放底层资源。"""
    pipeline.close()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查接口。"""
    return HealthResponse()


@app.post("/v1/consult")
def consult(payload: ConsultRequest) -> dict[str, Any]:
    """问诊接口。"""
    result = pipeline.consult(
        raw_input=payload.raw_input,
        rag_query=payload.rag_query,
        history_text=payload.history_text,
    )
    return result.model_dump()
