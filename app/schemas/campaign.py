from pydantic import BaseModel
from typing import List, Optional


class CampaignSummary(BaseModel):
    campaign_id: int
    description: str | None = None
    start_day: int | None = None
    end_day: int | None = None
    total_sales: float = 0.0
    total_quantity: int = 0
    total_discount: float = 0.0


class CampaignImpact(BaseModel):
    campaign_id: int
    actual_sales: float
    expected_sales: float
    incremental_sales: float
    incremental_revenue: float
    incremental_profit: float
    promotion_cost: float
    roi: float


class AffectedProduct(BaseModel):
    product_id: int
    product_name: str | None = None
    estimated_lost_sales: float


class CannibalizationResult(BaseModel):
    campaign_id: int
    promoted_product_id: int | None = None
    affected_products: list[AffectedProduct]
    total_lost_sales: float
    cannibalization_score: float


class CampaignRanking(BaseModel):
    campaign_id: int
    roi: float
    incremental_sales: float
    incremental_revenue: float
    reason: str


class ScenarioRequest(BaseModel):
    product_id: int
    budget: float
    discount_range: tuple[float, float] = (0.05, 0.30)
    duration_days: int = 14


class ScenarioResult(BaseModel):
    recommended_discount: float
    expected_revenue: float
    expected_profit: float
    expected_roi: float
    expected_incremental_sales: float
    confidence: str


class RecommendationResponse(BaseModel):
    rankings: list[CampaignRanking]
    total_campaigns_analyzed: int
