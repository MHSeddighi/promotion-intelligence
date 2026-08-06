"""LLM assistant and campaign-explanation routes."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException

from app.agent.llm_agent import TABLE_INFO
from app.api import deps
from app.api.schemas import AssistantRequest, AssistantResponse, ExplainRequest
from app.config import DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MODEL

router = APIRouter()


@router.get("/assistant/tables")
def assistant_tables() -> dict[str, Any]:
    return {"tables": TABLE_INFO}


@router.post("/assistant/ask", response_model=AssistantResponse)
def assistant_ask(payload: AssistantRequest) -> AssistantResponse:
    if deps.DATA_AGENT is None:
        raise HTTPException(status_code=503, detail="Database agent is not ready.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Set OPENAI_API_KEY in the server environment before using this endpoint.",
        )
    try:
        result = deps.DATA_AGENT.ask(
            payload.question,
            api_key=api_key,
            base_url=DEFAULT_OPENAI_BASE_URL,
            model=DEFAULT_OPENAI_MODEL,
            history=[turn.model_dump() for turn in payload.history],
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The LLM or database request failed: {type(exc).__name__}: {exc}",
        ) from exc
    return AssistantResponse(
        answer=result.answer,
        traces=[asdict(trace) for trace in result.traces],
    )


@router.post("/analytics/explain", response_model=AssistantResponse)
def analytics_explain(payload: ExplainRequest) -> AssistantResponse:
    """Run the LLM agent (via MCP analysis tools) over a campaign question."""
    if deps.DATA_AGENT is None:
        raise HTTPException(status_code=503, detail="Analysis agent is not ready.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Set OPENAI_API_KEY in the server environment before using this endpoint.",
        )
    if payload.question:
        question = payload.question
    elif payload.campaign_id is not None:
        question = (
            f"Analyze the effect of campaign {payload.campaign_id} for product {payload.product_id}; "
            "check incremental sales and cannibalization."
            if payload.product_id is not None
            else f"Analyze the effect of campaign {payload.campaign_id}; check incremental sales and cannibalization."
        )
    else:
        raise HTTPException(status_code=422, detail="Provide a question or a campaign_id.")
    try:
        result = deps.DATA_AGENT.ask(
            question,
            api_key=api_key,
            base_url=DEFAULT_OPENAI_BASE_URL,
            model=DEFAULT_OPENAI_MODEL,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The LLM or data request failed: {type(exc).__name__}: {exc}",
        ) from exc
    return AssistantResponse(
        answer=result.answer,
        traces=[asdict(trace) for trace in result.traces],
    )
