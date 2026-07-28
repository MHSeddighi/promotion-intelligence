from app.data.loader import DataLoader
from app.services.promotion_service import PromotionService
from app.services.causal_service import CausalService
from app.services.cannibalization_service import CannibalizationService
from app.services.recommendation_service import RecommendationService

loader = DataLoader()
promo_service = PromotionService(loader)
causal_service = CausalService(loader)
cannibal_service = CannibalizationService(loader)
rec_service = RecommendationService(loader)


def analyze_campaign(campaign_id: int) -> dict:
    campaign = promo_service.get_campaign(campaign_id)
    if campaign is None:
        return {"error": f"Campaign {campaign_id} not found"}
    impact = causal_service.estimate_impact(campaign_id)
    cannibal = cannibal_service.detect(campaign_id)
    return {
        "campaign": campaign,
        "impact": impact,
        "cannibalization": cannibal,
    }


def calculate_campaign_roi(campaign_id: int) -> dict:
    impact = causal_service.estimate_impact(campaign_id)
    if "error" in impact:
        return impact
    return {
        "campaign_id": campaign_id,
        "promotion_cost": impact["promotion_cost"],
        "incremental_revenue": impact["incremental_revenue"],
        "incremental_profit": impact["incremental_profit"],
        "roi": impact["roi"],
    }


def find_best_campaigns(top_n: int = 5) -> dict:
    return {"campaigns": rec_service.get_best_campaigns(top_n)}


def detect_cannibalization(campaign_id: int) -> dict:
    return cannibal_service.detect(campaign_id)


def recommend_future_campaign(product_id: int, budget: float, duration_days: int = 14) -> dict:
    return rec_service.recommend_scenario(
        product_id=product_id,
        budget=budget,
        discount_range=(0.05, 0.30),
        duration_days=duration_days,
    )
