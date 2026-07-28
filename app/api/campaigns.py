from fastapi import APIRouter, HTTPException
from app.services.promotion_service import PromotionService
from app.data.loader import DataLoader

router = APIRouter(prefix="/campaigns", tags=["campaigns"])
loader = DataLoader()
service = PromotionService(loader)


@router.get("")
def list_campaigns():
    return service.get_all_campaigns()


@router.get("/{campaign_id}")
def get_campaign(campaign_id: int):
    result = service.get_campaign(campaign_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return result
