"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import init_state, shutdown_state
from app.api.routers import analytics, assistant, campaigns, system


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_state()
    yield
    shutdown_state()


app = FastAPI(
    title="Unified Promotion Intelligence API",
    description=(
        "Baseline hurdle-model inference plus an LLM-powered analytics agent "
        "backed by MCP tools (promotion effect, incremental sales, cannibalization)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(system.router)
app.include_router(campaigns.router)
app.include_router(analytics.router)
app.include_router(assistant.router)
