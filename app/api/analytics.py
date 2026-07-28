from fastapi import APIRouter, HTTPException
from app.services.causal_service import CausalService
from app.services.cannibalization_service import CannibalizationService
from app.services.forecasting_service import ForecastingService
from app.data.loader import DataLoader

router = APIRouter(prefix="/analytics", tags=["analytics"])
loader = DataLoader()
causal_service = CausalService(loader)
cannibal_service = CannibalizationService(loader)
forecast_service = ForecastingService(loader)


@router.get("/campaigns/{campaign_id}/impact")
def get_campaign_impact(campaign_id: int):
    result = causal_service.estimate_impact(campaign_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/campaigns/{campaign_id}/cannibalization")
def get_campaign_cannibalization(campaign_id: int):
    result = cannibal_service.detect(campaign_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/campaigns/{campaign_id}/forecast")
def get_campaign_forecast(campaign_id: int):
    actual, expected = forecast_service.predict_baseline(campaign_id)
    return {
        "campaign_id": campaign_id,
        "actual_sales": actual,
        "expected_sales_without_promotion": expected,
        "incremental_sales": round(actual - expected, 2),
    }
