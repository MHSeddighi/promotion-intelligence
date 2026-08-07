"""Integration tests for the unified PromotionAnalysisEngine."""

import pytest

from app.services.analytics import AnalyticsService
from app.services.engine import PromotionAnalysisEngine

SERVICE = AnalyticsService()
ENGINE = PromotionAnalysisEngine(SERVICE)


def test_analyze_full_schema():
    result = ENGINE.analyze(1005637, promotion_id=15)
    assert result["data_sufficient"] is True
    for key in (
        "baseline_prediction",
        "promotion_prediction",
        "incremental_sales",
        "uplift",
        "cannibalization",
        "financial_metrics",
        "recommendation_features",
    ):
        assert key in result
    assert result["baseline_prediction"] > 0
    assert result["promotion_prediction"] > 0
    assert result["incremental_sales"] == pytest.approx(
        result["promotion_prediction"] - result["baseline_prediction"], abs=0.02
    )


def test_analyze_financial_consistency():
    result = ENGINE.analyze(1005637, 89, 101)
    financials = result["financial_metrics"]
    expected_revenue = result["incremental_sales"] * financials["average_price"]
    assert financials["incremental_revenue"] == pytest.approx(expected_revenue, abs=0.02)
    assert financials["incremental_units"] == pytest.approx(result["incremental_sales"], abs=0.02)
    assert financials["true_incremental_gain"] <= financials["incremental_units"] + 1e-6


def test_analyze_cannibalization_and_recommendation_present():
    result = ENGINE.analyze(934427, 89, 101)
    assert "signal" in result["cannibalization"]
    assert "recommendation_score" in result["recommendation_features"]
    assert 0.0 <= result["recommendation_features"]["recommendation_score"] <= 100.0


def test_analyze_no_data_product():
    result = ENGINE.analyze(99999999, 89, 101)
    assert result["data_sufficient"] is False
    assert result["baseline_prediction"] is None


def test_analyze_without_campaign_uses_causal_lift():
    result = ENGINE.analyze(1005637, start_week=89, end_week=101)
    assert result["promotion_id"] is None
    assert result["data_sufficient"] is True
    assert result["promotion_prediction"] is not None


def test_generate_report_fields():
    report = ENGINE.generate_report(1005637, promotion_id=15)
    assert report["title"]
    assert report["summary"]
    assert report["metrics"]
    assert report["recommendations"]
    assert report["risks"]
    names = {metric["name"] for metric in report["metrics"]}
    assert "roi" in names and "incremental_profit" in names


def test_generate_report_no_data():
    report = ENGINE.generate_report(99999999)
    assert "Insufficient" in report["summary"] or "coverage" in report["summary"]
