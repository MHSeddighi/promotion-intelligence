from app.data.loader import DataLoader
from app.services.promotion_service import PromotionService
from app.services.causal_service import CausalService
from app.services.cannibalization_service import CannibalizationService
from app.services.recommendation_service import RecommendationService


def test_promotion_service_list():
    service = PromotionService()
    campaigns = service.get_all_campaigns()
    assert isinstance(campaigns, list)
    assert len(campaigns) > 0
    assert "campaign_id" in campaigns[0]
    assert "total_sales" in campaigns[0]


def test_promotion_service_get():
    service = PromotionService()
    camp = service.get_campaign(1)
    if camp is not None:
        assert camp["campaign_id"] == 1
        assert "start_day" in camp
        assert "end_day" in camp
        assert "total_sales" in camp


def test_causal_service():
    service = CausalService()
    result = service.estimate_impact(1)
    if "error" not in result:
        assert "actual_sales" in result
        assert "incremental_sales_raw" in result
        assert "roi" in result
        assert "promotion_cost" in result


def test_cannibalization_service():
    service = CannibalizationService()
    result = service.detect(1)
    if "error" not in result:
        assert "affected_products" in result
        assert "cannibalization_score" in result
        assert "total_lost_sales" in result


def test_recommendation_service():
    service = RecommendationService()
    rankings = service.rank_campaigns()
    assert isinstance(rankings, list)
    if len(rankings) > 0:
        assert "campaign_id" in rankings[0]
        assert "roi" in rankings[0]
        assert "reason" in rankings[0]


def test_best_campaigns():
    service = RecommendationService()
    best = service.get_best_campaigns(3)
    assert len(best) <= 3
    if len(best) > 1:
        assert best[0]["roi"] >= best[-1]["roi"]


def test_scenario_recommendation():
    service = RecommendationService()
    result = service.recommend_scenario(
        product_id=1004906,
        budget=5000.0,
        discount_range=(0.05, 0.30),
        duration_days=14,
    )
    assert "recommended_discount" in result
    assert "expected_roi" in result
    assert "expected_revenue" in result
