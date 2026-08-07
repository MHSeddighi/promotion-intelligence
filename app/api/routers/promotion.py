"""Unified promotion analysis, campaign comparison and recommendation routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api import deps
from app.api.schemas import (
    CampaignCompareRequest,
    CampaignRecommendRequest,
    CampaignSimulateRequest,
    PromotionAnalyzeRequest,
    PromotionReportRequest,
)

router = APIRouter()


@router.post("/promotion/analyze")
def promotion_analyze(payload: PromotionAnalyzeRequest) -> dict[str, Any]:
    """End-to-end analysis: baseline, uplift, incremental sales,
    cannibalization, financial metrics and recommendation features."""
    try:
        result = deps.ENGINE.analyze(
            product_id=payload.product_id,
            promotion_id=payload.promotion_id,
            store_id=payload.store_id,
            start_week=payload.date_range.start_week if payload.date_range else None,
            end_week=payload.date_range.end_week if payload.date_range else None,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@router.post("/promotion/report")
def promotion_report(payload: PromotionReportRequest) -> dict[str, Any]:
    """Executive promotion report: summary, metrics, recommendations, risks."""
    try:
        result = deps.ENGINE.generate_report(
            product_id=payload.product_id,
            promotion_id=payload.promotion_id,
            start_week=payload.date_range.start_week if payload.date_range else None,
            end_week=payload.date_range.end_week if payload.date_range else None,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@router.post("/campaign/compare")
def campaign_compare(payload: CampaignCompareRequest) -> dict[str, Any]:
    """Rank and compare campaigns under an objective with strengths/weaknesses."""
    try:
        return deps.RANKING.compare(
            payload.campaigns,
            objective=payload.objective,
            weights=payload.weights,
            normalize=payload.normalization,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/campaign/recommend")
def campaign_recommend(payload: CampaignRecommendRequest) -> dict[str, Any]:
    """Recommend which products to promote under a strategy and constraints."""
    try:
        return deps.RECOMMENDATION.recommend_products(
            products=payload.products,
            budget=payload.budget,
            objective=payload.objective,
            constraints=payload.constraints,
            start_week=payload.date_range.start_week if payload.date_range else None,
            end_week=payload.date_range.end_week if payload.date_range else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/campaign/simulate")
def campaign_simulate(payload: CampaignSimulateRequest) -> dict[str, Any]:
    """Simulate a hypothetical discount campaign for one product."""
    try:
        return deps.RECOMMENDATION.simulate_campaign(
            product_id=payload.product_id,
            discount_percentage=payload.discount_percentage,
            weeks=payload.weeks,
            start_week=payload.start_week,
            elasticity=payload.elasticity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
