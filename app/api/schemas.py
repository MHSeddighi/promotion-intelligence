"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.api.deps import BASELINE


def _feature_field(feature: str) -> tuple[Any, Any]:
    if feature in BASELINE.categorical_features:
        return int | None, Field(..., description="Training encoder category code.")
    return float | None, Field(..., description="Engineered numeric feature.")


BaselineFeatures = create_model(
    "BaselineFeatures",
    __config__=ConfigDict(extra="forbid"),
    **{feature: _feature_field(feature) for feature in BASELINE.features},
)


class BaselineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[BaselineFeatures] = Field(  # type: ignore[valid-type]
        min_length=1, max_length=10_000
    )


class PredictionRow(BaseModel):
    row: int
    probability_positive: float
    predicted_positive_quantity: float
    baseline_prediction: float


class BaselineResponse(BaseModel):
    count: int
    predictions: list[PredictionRow]
    warnings: list[str] = Field(default_factory=list)


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8_000)


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4_000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class AssistantResponse(BaseModel):
    answer: str
    traces: list[dict[str, Any]]


class PromotionEffectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign_id: int = Field(ge=1, le=10_000)
    product_id: int | None = Field(default=None, ge=1)
    start_week: int | None = Field(default=None, ge=1, le=102)
    end_week: int | None = Field(default=None, ge=1, le=102)


class CannibalizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    promoted_product_id: int = Field(ge=1)
    start_week: int | None = Field(default=None, ge=1, le=102)
    end_week: int | None = Field(default=None, ge=1, le=102)


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign_id: int | None = Field(default=None, ge=1, le=10_000)
    product_id: int | None = Field(default=None, ge=1)
    question: str | None = Field(default=None, min_length=1, max_length=4_000)


class DateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_week: int | None = Field(default=None, ge=1, le=102)
    end_week: int | None = Field(default=None, ge=1, le=102)


class PromotionAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(ge=1)
    promotion_id: int | None = Field(default=None, ge=1, le=10_000)
    store_id: int | None = Field(default=None, ge=1)
    date_range: DateRange | None = None


class CampaignCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaigns: list[int] = Field(min_length=1, max_length=100)
    objective: str = Field(default="profit", min_length=1, max_length=50)
    weights: dict[str, float] | None = None
    normalization: Literal["min_max", "z_score", "percentile"] = "min_max"


class CampaignRecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    products: list[int] = Field(min_length=1, max_length=200)
    budget: float | None = Field(default=None, ge=0)
    objective: str | None = Field(default=None, min_length=1, max_length=50)
    constraints: dict[str, Any] | None = None
    date_range: DateRange | None = None


class CampaignSimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(ge=1)
    discount_percentage: float = Field(gt=0, le=90)
    weeks: int = Field(ge=1, le=52)
    start_week: int | None = Field(default=None, ge=1, le=102)
    elasticity: float | None = Field(default=None, ge=0)


class PromotionReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(ge=1)
    promotion_id: int | None = Field(default=None, ge=1, le=10_000)
    date_range: DateRange | None = None
