"""Unit tests for promotion accounting / financial metrics."""

import pytest

from app.services.financials import (
    calculate_financial_metrics,
    roi_at_margin,
)


def test_example_incremental_accounting():
    """Task example: baseline=1000, promotion=1400, margin=20%."""
    result = calculate_financial_metrics(
        baseline_sales=1000,
        promotion_sales=1400,
        average_price=2.0,
        margin=0.20,
        promotion_cost=100.0,
    )
    assert result.incremental_units == pytest.approx(400.0)
    assert result.incremental_revenue == pytest.approx(800.0)
    assert result.incremental_profit == pytest.approx(800 * 0.20 - 100)
    assert result.roi == pytest.approx((800 * 0.20 - 100) / 100)
    assert result.breakeven_margin == pytest.approx(100 / 800)


def test_positive_promotion_impact():
    result = calculate_financial_metrics(
        baseline_sales=1000,
        promotion_sales=1500,
        average_price=2.0,
        margin=0.25,
        promotion_cost=50.0,
    )
    assert result.incremental_units == 500.0
    assert result.incremental_revenue == 1000.0
    assert result.incremental_profit == pytest.approx(1000 * 0.25 - 50)
    assert result.roi == pytest.approx((1000 * 0.25 - 50) / 50)
    assert result.true_incremental_gain == result.incremental_units


def test_negative_promotion_impact():
    result = calculate_financial_metrics(
        baseline_sales=1000,
        promotion_sales=800,
        average_price=2.0,
        margin=0.20,
        promotion_cost=100.0,
    )
    assert result.incremental_units == -200.0
    assert result.incremental_revenue == -400.0
    assert result.incremental_profit < 0
    assert any("did not generate incremental units" in note for note in result.notes)


def test_no_effect_promotion():
    result = calculate_financial_metrics(
        baseline_sales=1000,
        promotion_sales=1000,
        average_price=1.0,
        margin=0.20,
    )
    assert result.incremental_units == 0.0
    assert result.roi is None
    assert any("No retailer-funded discount cost" in note for note in result.notes)


def test_cannibalization_reduces_true_gain():
    result = calculate_financial_metrics(
        baseline_sales=1000,
        promotion_sales=1400,
        average_price=2.0,
        margin=0.20,
        promotion_cost=100.0,
        cannibalized_units=120.0,
    )
    assert result.incremental_units == 400.0
    assert result.cannibalized_units == 120.0
    assert result.cannibalized_revenue == pytest.approx(240.0)
    assert result.net_incremental_units == pytest.approx(280.0)
    assert result.true_incremental_gain == pytest.approx(280.0)
    assert result.net_roi < result.roi
    assert any("net of cannibalization" in note for note in result.notes)


def test_roi_at_margin_sensitivity():
    assert roi_at_margin(incremental_revenue=1000, promotion_cost=100, margin=0.2) == pytest.approx(1.0)
    assert roi_at_margin(incremental_revenue=1000, promotion_cost=100, margin=0.1) == pytest.approx(0.0)
    assert roi_at_margin(incremental_revenue=0, promotion_cost=100, margin=0.2) == -1.0
    assert roi_at_margin(incremental_revenue=1000, promotion_cost=0, margin=0.2) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_sales": -1, "promotion_sales": 10, "average_price": 1.0, "margin": 0.2},
        {"baseline_sales": 10, "promotion_sales": -1, "average_price": 1.0, "margin": 0.2},
        {"baseline_sales": 10, "promotion_sales": 10, "average_price": -1.0, "margin": 0.2},
        {"baseline_sales": 10, "promotion_sales": 10, "average_price": 1.0, "margin": 1.5},
        {"baseline_sales": 10, "promotion_sales": 10, "average_price": 1.0, "margin": -0.1},
        {"baseline_sales": 10, "promotion_sales": 10, "average_price": 1.0, "margin": 0.2, "promotion_cost": -1},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        calculate_financial_metrics(**kwargs)


def test_dict_output_is_jsonable():
    result = calculate_financial_metrics(
        baseline_sales=1000,
        promotion_sales=1400,
        average_price=2.0,
        margin=0.2,
        promotion_cost=100.0,
    )
    payload = result.to_dict()
    assert set(payload) == set(result.to_dict())
    assert isinstance(payload["roi"], float)
    assert isinstance(payload["notes"], list)
