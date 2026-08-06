"""Campaign listing and detail routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.deps import ANALYTICS

router = APIRouter()


@router.get("/campaigns")
def list_campaigns() -> dict[str, Any]:
    return {"campaigns": ANALYTICS.list_campaigns()}


@router.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: int) -> dict[str, Any]:
    campaign = ANALYTICS.get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found.")
    return campaign
