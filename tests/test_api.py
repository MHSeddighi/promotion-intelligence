from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data


def test_list_campaigns():
    response = client.get("/campaigns")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "campaign_id" in data[0]


def test_get_campaign():
    response = client.get("/campaigns/1")
    if response.status_code == 200:
        data = response.json()
        assert data["campaign_id"] == 1


def test_get_campaign_impact():
    response = client.get("/analytics/campaigns/1/impact")
    if response.status_code == 200:
        data = response.json()
        assert "incremental_sales_raw" in data
        assert "roi" in data
        assert "promotion_cost" in data


def test_get_cannibalization():
    response = client.get("/analytics/campaigns/1/cannibalization")
    if response.status_code == 200:
        data = response.json()
        assert "cannibalization_score" in data


def test_get_recommendations():
    response = client.get("/recommendations")
    assert response.status_code == 200
    data = response.json()
    assert "rankings" in data
    assert "total_campaigns_analyzed" in data


def test_get_best():
    response = client.get("/recommendations/best")
    assert response.status_code == 200
    data = response.json()
    assert "campaigns" in data


def test_get_patterns():
    response = client.get("/recommendations/patterns")
    assert response.status_code == 200
    data = response.json()
    assert "patterns" in data


def test_scenario():
    response = client.post(
        "/recommendations/scenario",
        json={
            "product_id": 1004906,
            "budget": 5000.0,
            "discount_range": [0.05, 0.30],
            "duration_days": 14,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommended_discount" in data
    assert "expected_roi" in data
