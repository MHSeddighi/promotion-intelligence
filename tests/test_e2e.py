"""End-to-end customer scenario tests (Phase 9).

Each scenario follows a real customer question through the production services
without calling an external LLM (the LLM layer is covered separately by mocked
tests in ``test_analytics.py``/``test_app.py``).
"""

import pytest

from app.services.analytics import AnalyticsService
from app.services.causal import CausalInferenceService
from app.services.engine import PromotionAnalysisEngine
from app.services.recommendation import RecommendationService
from app.mcp.tools import MCPTools
from app.models.similarity import ProductSimilarityService

SERVICE = AnalyticsService()
CAUSAL = CausalInferenceService(SERVICE)
RECOMMENDATION = RecommendationService(SERVICE, CAUSAL)
ENGINE = PromotionAnalysisEngine(SERVICE, CAUSAL, RECOMMENDATION)
TOOLS = MCPTools(
    SERVICE,
    CAUSAL,
    ProductSimilarityService(),
    RECOMMENDATION.ranking,
    RECOMMENDATION,
    ENGINE,
)


def test_scenario1_should_we_promote_coca_cola():
    """Customer: 'Should we promote Coca Cola next month?'"""
    # 1. Find historical promotions for the product (stand-in: campaign 15).
    campaigns = SERVICE.list_campaigns()
    assert campaigns
    # 2. Predict baseline sales.
    baseline = SERVICE.product_baseline(1005637, 89, 101)
    assert baseline["data_sufficient"] and baseline["baseline_qty"] > 0
    # 3. Estimate uplift (causal, control-group based).
    causal = CAUSAL.causal_effect(1005637, 89, 101)
    assert causal["data_sufficient"] is True
    # 4. Detect cannibalization.
    cannibalization = SERVICE.cannibalization_effect(1005637, 89, 101)
    assert "signal" in cannibalization
    # 5. Calculate profit.
    analysis = ENGINE.analyze(1005637, promotion_id=15)
    assert analysis["financial_metrics"]["incremental_profit"] is not None
    # 6. Rank the recommendation.
    ranking = RECOMMENDATION.recommend_products([1005637, 934427, 1004906], objective="profit")
    assert ranking["recommendations"]
    # 7. Explain the decision.
    report = ENGINE.generate_report(1005637, promotion_id=15)
    assert report["recommendations"] and report["summary"]


def test_scenario2_why_did_promotion_fail():
    """Customer: 'Why did this promotion fail?'"""
    effect = SERVICE.campaign_effect(15)
    assert effect["data_sufficient"] is True
    # Expected baseline vs actual sales and the uplift difference.
    assert "actual_sales" in effect and "baseline_sales" in effect
    assert effect["incremental_sales"] == pytest.approx(
        effect["actual_sales"] - effect["baseline_sales"], abs=0.02
    )
    # Competitor impact (cannibalization of other products by this campaign).
    cannibalization = SERVICE.cannibalization_effect(934427, 98, 101)
    assert "affected_products" in cannibalization
    # Profitability.
    report = ENGINE.generate_report(1005637, promotion_id=15)
    roi_metric = next(m for m in report["metrics"] if m["name"] == "roi")
    assert roi_metric["value"] is not None


def test_scenario3_which_products_should_we_promote():
    """Customer: 'Which products should we promote?'"""
    result = RECOMMENDATION.recommend_products(
        [1005637, 934427, 1004906], objective="PROFIT_OPTIMIZATION"
    )
    recommendations = result["recommendations"]
    assert recommendations
    # Evaluates products, predicts incremental sales, calculates profit, ranks.
    scores = [row["score"] for row in recommendations]
    assert scores == sorted(scores, reverse=True)
    for row in recommendations:
        assert row["expected_sales"] >= 0
        assert row["expected_profit"] is not None
        assert row["explanation"]


def test_scenario4_discount_20_percent_three_weeks():
    """Customer: 'What happens if we discount product X by 20% for 3 weeks?'"""
    simulation = RECOMMENDATION.simulate_campaign(
        1005637, discount_percentage=20, weeks=3, start_week=89
    )
    assert simulation["available"] is True
    # Simulates the campaign, predicts demand, calculates ROI, explains risks.
    assert simulation["expected_sales"] > simulation["baseline_sales"]
    assert "roi" in simulation
    assert simulation["risks"]


def test_scenario5_find_similar_products_for_targeting():
    """Customer: 'Find similar products for promotion targeting'"""
    result = TOOLS.find_similar_products(1005637, top_k=5)
    assert len(result["similar_products"]) == 5
    assert result["substitutes"]
    assert result["similar_products"][0]["similarity_score"] >= result["similar_products"][-1]["similarity_score"]


def test_scenario6_compare_last_years_campaigns():
    """Customer: 'Compare last year's Christmas campaigns'"""
    # Historical campaigns (the dataset has no explicit Christmas flag; the
    # same comparison flow applies to any campaign set).
    comparison = RECOMMENDATION.ranking.compare([15, 20, 24], objective="sales")
    assert comparison["ranking"]
    # Explains winners and losers.
    assert comparison["strengths"] and comparison["weaknesses"]
    winner = comparison["ranking"][0]
    loser = comparison["ranking"][-1]
    assert winner["score"] >= loser["score"]
    assert comparison["strengths"][0]["campaign_id"] == winner["campaign_id"]


def test_scenario7_optimize_for_profit_instead_of_sales():
    """Customer: 'Optimize promotions for profit instead of sales'"""
    products = [1005637, 934427, 1004906]
    sales_result = RECOMMENDATION.recommend_products(products, objective="sales")
    profit_result = RECOMMENDATION.recommend_products(products, objective="profit")
    assert sales_result["objective"] == "SALES_GROWTH"
    assert profit_result["objective"] == "PROFIT_OPTIMIZATION"
    # Tradeoffs: different strategies produce different winner/loser narratives.
    sales_best = sales_result["recommendations"][0]
    profit_best = profit_result["recommendations"][0]
    assert sales_best["explanation"] and profit_best["explanation"]
    # The objective weights are recalculated and exposed.
    assert sales_result["weights"] != profit_result["weights"]


def test_scenario8_generate_executive_promotion_report():
    """Customer: 'Generate executive promotion report'"""
    report = TOOLS.generate_promotion_report(1005637, campaign_id=15)
    # Summary, metrics, recommendations and risks.
    assert report["summary"]
    assert report["metrics"]
    assert report["recommendations"]
    assert report["risks"]
    metric_names = {metric["name"] for metric in report["metrics"]}
    assert {
        "baseline_sales",
        "incremental_sales",
        "incremental_profit",
        "roi",
    } <= metric_names
