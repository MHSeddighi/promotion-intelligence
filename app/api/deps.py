"""Shared singletons and lifecycle state for the HTTP API routers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.agent.llm_agent import PromotionDataAgent
from app.models.baseline import BaselinePredictor
from app.models.similarity import ProductSimilarityService
from app.mcp.tools import MCPTools
from app.services.analytics import AnalyticsService
from app.services.attribution import IncrementalSalesAttributionService
from app.services.causal import CausalInferenceService
from app.services.engine import PromotionAnalysisEngine
from app.services.ranking import CampaignRankingService
from app.services.recommendation import RecommendationService

BASELINE = BaselinePredictor()
ANALYTICS = AnalyticsService()
CAUSAL = CausalInferenceService(ANALYTICS)
RANKING = CampaignRankingService(ANALYTICS, CAUSAL)
RECOMMENDATION = RecommendationService(ANALYTICS, CAUSAL, RANKING)
ENGINE = PromotionAnalysisEngine(ANALYTICS, CAUSAL, RECOMMENDATION)
SIMILARITY = ProductSimilarityService()
ATTRIBUTION = IncrementalSalesAttributionService()
MCP = MCPTools(ANALYTICS, attribution=ATTRIBUTION)
DATA_AGENT: PromotionDataAgent | None = None
APP_STATE: dict[str, Any] = {
    "ready": False,
    "loaded_at": None,
    "warmup_prediction": None,
}


def init_state() -> None:
    """Warm up models and build the database/agent singletons at startup."""
    global DATA_AGENT
    DATA_AGENT = PromotionDataAgent()
    warmup = BASELINE.predict(BASELINE.load_sample()).iloc[0]
    APP_STATE.update(
        ready=True,
        loaded_at=datetime.now(UTC).isoformat(),
        warmup_prediction=float(warmup["baseline_prediction"]),
    )


def shutdown_state() -> None:
    """Release process-wide resources at shutdown."""
    global DATA_AGENT
    APP_STATE["ready"] = False
    if DATA_AGENT is not None:
        DATA_AGENT.catalog.close()
        DATA_AGENT = None
