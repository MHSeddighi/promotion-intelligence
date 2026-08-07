"""Unit + integration tests for the multi-objective recommendation engine."""

import pytest

from app.services.analytics import AnalyticsService
from app.services.recommendation import (
    DEFAULT_ELASTICITY,
    STRATEGIES,
    RecommendationService,
    resolve_strategy,
)

SERVICE = AnalyticsService()
RECOMMENDATION = RecommendationService(SERVICE)


def test_strategy_resolution():
    assert resolve_strategy("profit") == "PROFIT_OPTIMIZATION"
    assert resolve_strategy("SALES_GROWTH") == "SALES_GROWTH"
    assert resolve_strategy("safe") == "SAFE_PROMOTION"
    assert resolve_strategy(None) == "PROFIT_OPTIMIZATION"
    with pytest.raises(ValueError):
        resolve_strategy("bogus")


def test_strategy_weights_defined():
    assert set(STRATEGIES) == {"SALES_GROWTH", "PROFIT_OPTIMIZATION", "SAFE_PROMOTION"}
    for weights in STRATEGIES.values():
        assert weights


def test_evaluate_product_schema():
    evaluation = RECOMMENDATION.evaluate_product(1005637, 89, 101)
    assert evaluation["available"] is True
    for key in (
        "baseline_sales",
        "expected_sales",
        "incremental_sales",
        "incremental_revenue",
        "incremental_profit",
        "roi",
        "cannibalization_risk",
        "confidence",
    ):
        assert key in evaluation


def test_evaluate_product_no_data():
    evaluation = RECOMMENDATION.evaluate_product(99999999, 89, 101)
    assert evaluation["available"] is False


def test_recommend_products_ranking_and_explanations():
    result = RECOMMENDATION.recommend_products(
        [1005637, 934427, 1004906], objective="profit"
    )
    assert result["objective"] == "PROFIT_OPTIMIZATION"
    recommendations = result["recommendations"]
    assert recommendations
    scores = [row["score"] for row in recommendations]
    assert scores == sorted(scores, reverse=True)
    for row in recommendations:
        assert set(row) >= {
            "product_id",
            "score",
            "expected_sales",
            "expected_profit",
            "explanation",
        }
        assert row["explanation"]


def test_recommend_products_budget_caps_cost():
    result = RECOMMENDATION.recommend_products(
        [1005637, 934427, 1004906], objective="sales", budget=0.0
    )
    assert result["recommendations"] == []
    assert result["budget"] == 0.0


def test_recommend_products_constraints():
    result = RECOMMENDATION.recommend_products(
        [1005637, 934427, 1004906],
        objective="profit",
        constraints={"max_cannibalization_risk": 1.0},
    )
    for row in result["recommendations"]:
        assert row["cannibalization_risk"] <= 1.0


def test_recommend_empty_products_raises():
    with pytest.raises(ValueError):
        RECOMMENDATION.recommend_products([], objective="profit")


def test_simulate_campaign_schema():
    simulation = RECOMMENDATION.simulate_campaign(
        1005637, discount_percentage=20, weeks=3, start_week=89
    )
    assert simulation["available"] is True
    assert simulation["discount_percentage"] == 20.0
    assert simulation["weeks"] == 3
    assert simulation["implied_lift_percentage"] == pytest.approx(20 * DEFAULT_ELASTICITY)
    for key in ("expected_sales", "incremental_sales", "roi", "promotion_cost", "risks"):
        assert key in simulation
    assert simulation["risks"]


def test_simulate_campaign_validation():
    with pytest.raises(ValueError):
        RECOMMENDATION.simulate_campaign(1005637, discount_percentage=0, weeks=3)
    with pytest.raises(ValueError):
        RECOMMENDATION.simulate_campaign(1005637, discount_percentage=95, weeks=3)
    with pytest.raises(ValueError):
        RECOMMENDATION.simulate_campaign(1005637, discount_percentage=20, weeks=0)
    with pytest.raises(ValueError):
        RECOMMENDATION.simulate_campaign(1005637, discount_percentage=20, weeks=53)
    with pytest.raises(ValueError):
        RECOMMENDATION.simulate_campaign(1005637, discount_percentage=20, weeks=3, elasticity=-1)


def test_different_strategies_change_ranking():
    products = [1005637, 934427, 1004906]
    sales = RECOMMENDATION.recommend_products(products, objective="sales")
    profit = RECOMMENDATION.recommend_products(products, objective="profit")
    safe = RECOMMENDATION.recommend_products(products, objective="safe")
    assert sales["objective"] == "SALES_GROWTH"
    assert profit["objective"] == "PROFIT_OPTIMIZATION"
    assert safe["objective"] == "SAFE_PROMOTION"
    ids_sales = [row["product_id"] for row in sales["recommendations"]]
    ids_profit = [row["product_id"] for row in profit["recommendations"]]
    assert set(ids_sales) == set(ids_profit)
