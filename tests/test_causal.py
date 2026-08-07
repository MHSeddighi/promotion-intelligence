"""Unit + integration tests for causal inference / promotion impact."""

import numpy as np
import pytest

from app.services.analytics import AnalyticsService
from app.services.causal import (
    CausalInferenceService,
    bootstrap_causal_lift,
)

SERVICE = AnalyticsService()
CAUSAL = CausalInferenceService(SERVICE)


# ------------------------------------------------------------- pure functions
def test_bootstrap_recovers_injected_effect():
    rng = np.random.default_rng(7)
    baseline_t = rng.uniform(5, 20, 400)
    baseline_c = rng.uniform(5, 20, 400)
    units_c = baseline_c * rng.uniform(0.9, 1.1, 400)
    # Inject +50% causal lift on treated cells.
    units_t = baseline_t * 1.5 * rng.uniform(0.9, 1.1, 400)
    lower, upper = bootstrap_causal_lift(units_t, baseline_t, units_c, baseline_c)
    assert lower <= 50.0 <= upper


def test_bootstrap_no_effect_interval_contains_zero():
    rng = np.random.default_rng(11)
    baseline = rng.uniform(5, 20, 300)
    units = baseline * rng.uniform(0.95, 1.05, 300)
    lower, upper = bootstrap_causal_lift(units, baseline, units, baseline)
    assert lower <= 0.0 <= upper


def test_bootstrap_insufficient_data_returns_nan():
    lower, upper = bootstrap_causal_lift(
        np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0])
    )
    assert np.isnan(lower) and np.isnan(upper)


def test_bootstrap_validates_iterations():
    with pytest.raises(ValueError):
        bootstrap_causal_lift(
            np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), n_iter=5
        )


def test_estimate_uplift():
    assert CAUSAL.estimate_uplift(baseline_sales=1000, causal_lift_percentage=10.0) == pytest.approx(100.0)
    assert CAUSAL.estimate_uplift(baseline_sales=1000, causal_lift_percentage=None) == 0.0


# ------------------------------------------------------------- integration
def test_causal_effect_schema():
    result = CAUSAL.causal_effect(1005637, 89, 101)
    assert result["product_id"] == 1005637
    assert result["start_week"] == 89
    assert result["end_week"] == 101
    for key in (
        "data_sufficient",
        "causal_lift_percentage",
        "raw_lift_percentage",
        "control_index",
        "confidence_interval",
        "incremental_units_causal",
        "n_treated_cells",
        "n_control_cells",
    ):
        assert key in result


def test_causal_effect_missing_product():
    result = CAUSAL.causal_effect(99999999, 89, 101)
    assert result["data_sufficient"] is False
    assert result["incremental_units_causal"] == 0.0


def test_campaign_causal_effect():
    result = CAUSAL.campaign_causal_effect(15, product_id=1005637)
    assert result["campaign_id"] == 15
    assert result["data_sufficient"] is True
    assert isinstance(result["causal_lift_percentage"], float)
    assert isinstance(result["incremental_units_causal"], float)


def test_campaign_causal_effect_unknown_campaign():
    result = CAUSAL.campaign_causal_effect(99999)
    assert "error" in result


def test_positive_and_negative_signals_on_synthetic_frame():
    """DID estimator must distinguish positive, negative and null effects."""
    import pandas as pd

    cells = pd.DataFrame(
        {
            "WEEK_NO": [89, 89, 90, 90] * 3,
            "qty": [
                100, 60,  # treated high vs control
                80, 60,   # treated high vs control
                30, 60,   # treated low vs control (negative)
                40, 60,
                60, 60,   # equal (null)
                60, 60,
            ],
            "baseline": [60.0] * 12,
            "treated": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    stats = CAUSAL._did_stats(cells, 89, 90)
    assert stats["data_sufficient"] is True
    assert stats["causal_lift_percentage"] > 0
    assert stats["n_treated_cells"] == 6
    assert stats["n_control_cells"] == 6

    null_cells = cells.copy()
    null_cells["qty"] = 60.0
    null_stats = CAUSAL._did_stats(null_cells, 89, 90)
    assert null_stats["causal_lift_percentage"] == pytest.approx(0.0, abs=1e-6)

    negative_cells = cells.copy()
    negative_cells.loc[negative_cells["treated"] == 1, "qty"] = 20.0
    negative_stats = CAUSAL._did_stats(negative_cells, 89, 90)
    assert negative_stats["causal_lift_percentage"] < 0


def test_did_stats_missing_treatment_group():
    import pandas as pd

    cells = pd.DataFrame(
        {
            "WEEK_NO": [89, 90],
            "qty": [60.0, 60.0],
            "baseline": [60.0, 60.0],
            "treated": [0, 0],
        }
    )
    stats = CAUSAL._did_stats(cells, 89, 90)
    assert stats["data_sufficient"] is False
    assert "No mixed promotion exposure" in stats["message"]
