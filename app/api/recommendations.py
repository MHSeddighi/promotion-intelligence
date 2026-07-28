from fastapi import APIRouter, HTTPException
from app.services.recommendation_service import RecommendationService
from app.schemas.campaign import ScenarioRequest
from app.data.loader import DataLoader

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
loader = DataLoader()
service = RecommendationService(loader)


@router.get("")
def get_recommendations(top_n: int = 10):
    rankings = service.rank_campaigns()
    return {
        "rankings": rankings[:top_n],
        "total_campaigns_analyzed": len(rankings),
    }


@router.get("/best")
def get_best_campaigns(top_n: int = 5):
    return {"campaigns": service.get_best_campaigns(top_n)}


@router.get("/worst")
def get_worst_campaigns(top_n: int = 5):
    return {"campaigns": service.get_worst_campaigns(top_n)}


@router.get("/patterns")
def get_patterns():
    return {"patterns": service.get_effective_patterns()}


@router.post("/scenario")
def recommend_scenario(request: ScenarioRequest):
    result = service.recommend_scenario(
        product_id=request.product_id,
        budget=request.budget,
        discount_range=request.discount_range,
        duration_days=request.duration_days,
    )
    return result
