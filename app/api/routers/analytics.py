"""Promotion effect and cannibalization analytics routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.deps import ANALYTICS
from app.api.schemas import CannibalizationRequest, PromotionEffectRequest

router = APIRouter()


@router.post("/analytics/promotion-effect")
def promotion_effect(payload: PromotionEffectRequest) -> dict[str, Any]:
    result = ANALYTICS.campaign_effect(
        payload.campaign_id,
        product_id=payload.product_id,
        start_week=payload.start_week,
        end_week=payload.end_week,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/analytics/incremental-sales")
def incremental_sales(payload: PromotionEffectRequest) -> dict[str, Any]:
    result = ANALYTICS.incremental_sales(
        payload.campaign_id,
        product_id=payload.product_id,
        start_week=payload.start_week,
        end_week=payload.end_week,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/analytics/cannibalization-effect")
def cannibalization_effect(payload: CannibalizationRequest) -> dict[str, Any]:
    return ANALYTICS.cannibalization_effect(
        payload.promoted_product_id,
        start_week=payload.start_week,
        end_week=payload.end_week,
    )


@router.get("/analytics/promoted-products")
def promoted_products() -> dict[str, Any]:
    return {"products": ANALYTICS.list_promoted_products()}
