"""API tests for the unified promotion endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_promotion_analyze(client):
    response = client.post(
        "/promotion/analyze",
        json={"product_id": 1005637, "promotion_id": 15},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["data_sufficient"] is True
    for key in (
        "baseline_prediction",
        "promotion_prediction",
        "incremental_sales",
        "uplift",
        "cannibalization",
        "financial_metrics",
        "recommendation_features",
    ):
        assert key in payload


def test_promotion_analyze_with_date_range(client):
    response = client.post(
        "/promotion/analyze",
        json={
            "product_id": 1005637,
            "date_range": {"start_week": 89, "end_week": 101},
        },
    )
    assert response.status_code == 200
    assert response.json()["start_week"] == 89
    assert response.json()["end_week"] == 101


def test_promotion_analyze_no_data_product(client):
    response = client.post(
        "/promotion/analyze", json={"product_id": 99999999}
    )
    assert response.status_code == 200
    assert response.json()["data_sufficient"] is False


def test_promotion_analyze_validation(client):
    response = client.post("/promotion/analyze", json={"product_id": 0})
    assert response.status_code == 422


def test_promotion_report(client):
    response = client.post(
        "/promotion/report", json={"product_id": 1005637, "promotion_id": 15}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]
    assert payload["recommendations"]
    assert payload["risks"]


def test_campaign_compare(client):
    response = client.post(
        "/campaign/compare",
        json={"campaigns": [15, 20], "objective": "profit"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ranking"]
    assert payload["strengths"]
    assert payload["weaknesses"]
    scores = [row["score"] for row in payload["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_campaign_compare_zscore_and_custom_weights(client):
    response = client.post(
        "/campaign/compare",
        json={
            "campaigns": [15, 20, 24],
            "objective": "sales",
            "normalization": "z_score",
        },
    )
    assert response.status_code == 200
    assert response.json()["normalization"] == "z_score"


def test_campaign_compare_validation(client):
    response = client.post("/campaign/compare", json={"campaigns": []})
    assert response.status_code == 422
    response = client.post(
        "/campaign/compare", json={"campaigns": [15], "objective": "bogus"}
    )
    assert response.status_code == 422


def test_campaign_recommend(client):
    response = client.post(
        "/campaign/recommend",
        json={
            "products": [1005637, 934427, 1004906],
            "objective": "PROFIT_OPTIMIZATION",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"]
    for row in payload["recommendations"]:
        assert set(row) >= {"product_id", "score", "expected_sales", "expected_profit", "explanation"}


def test_campaign_recommend_with_budget_and_constraints(client):
    response = client.post(
        "/campaign/recommend",
        json={
            "products": [1005637, 934427, 1004906],
            "objective": "safe",
            "budget": 1000.0,
            "constraints": {"max_cannibalization_risk": 20.0},
        },
    )
    assert response.status_code == 200
    assert response.json()["budget"] == 1000.0


def test_campaign_recommend_validation(client):
    response = client.post("/campaign/recommend", json={"products": []})
    assert response.status_code == 422


def test_campaign_simulate(client):
    response = client.post(
        "/campaign/simulate",
        json={
            "product_id": 1005637,
            "discount_percentage": 20,
            "weeks": 3,
            "start_week": 89,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["risks"]


def test_campaign_simulate_validation(client):
    response = client.post(
        "/campaign/simulate",
        json={"product_id": 1005637, "discount_percentage": 0, "weeks": 3},
    )
    assert response.status_code == 422


def test_mcp_tools_endpoint_lists_copilot_tools(client):
    response = client.get("/mcp/tools")
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()["tools"]}
    assert "find_similar_products" in names
    assert "generate_promotion_report" in names
