"""FastAPI service for Dia-Agent consultations."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from dia_agent.pipeline import DiaAgentPipeline


class ConsultRequest(BaseModel):
    raw_input: str | dict[str, Any]
    rag_query: str = ""
    history_text: str = ""


class HealthResponse(BaseModel):
    status: str = Field(default="ok")


app = FastAPI(title="Dia-Agent API", version="0.1.0")
pipeline = DiaAgentPipeline()


@app.on_event("shutdown")
def shutdown_event() -> None:
    pipeline.close()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/v1/consult")
def consult(payload: ConsultRequest) -> dict[str, Any]:
    result = pipeline.consult(
        raw_input=payload.raw_input,
        rag_query=payload.rag_query,
        history_text=payload.history_text,
    )
    return result.model_dump()
