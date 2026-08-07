"""Unit tests for normalization, weighted scoring and campaign ranking."""

import pytest

from app.services.ranking import (
    OBJECTIVE_PRESETS,
    CampaignRankingService,
    normalize_values,
    weighted_score,
)


def test_min_max_normalization():
    assert normalize_values([10, 20, 30], "min_max") == [0.0, 0.5, 1.0]
    assert normalize_values([5, 5, 5], "min_max") == [0.5, 0.5, 0.5]


def test_z_score_normalization():
    result = normalize_values([10, 20, 30], "z_score")
    assert result[1] == pytest.approx(0.5)
    assert 0.0 <= result[0] <= result[1] <= result[2] <= 1.0


def test_percentile_normalization():
    assert normalize_values([1, 2, 3, 4], "percentile") == [0.25, 0.5, 0.75, 1.0]


def test_inverted_normalization():
    result = normalize_values([1, 2, 3], "min_max", invert=True)
    assert result == [1.0, 0.5, 0.0]


def test_invalid_normalization():
    with pytest.raises(ValueError):
        normalize_values([1, 2], "unknown")


def test_weighted_score():
    scores = {"incremental_profit": 0.8, "roi": 0.4, "cannibalization": 0.2}
    weights = {"incremental_profit": 0.5, "roi": 0.3, "cannibalization": -0.2}
    assert weighted_score(scores, weights) == pytest.approx(0.8 * 0.5 + 0.4 * 0.3 - 0.2 * 0.2)


def test_weighted_score_unknown_objective():
    with pytest.raises(ValueError):
        weighted_score({"a": 1.0}, {"missing": 1.0})


def test_objective_presets_are_valid():
    for weights in OBJECTIVE_PRESETS.values():
        assert weights
        assert all(
            name
            in {
                "incremental_profit",
                "incremental_sales",
                "roi",
                "cannibalization",
                "cost",
            }
            for name in weights
        )
        assert all(isinstance(value, float) for value in weights.values())


def test_rank_validates_inputs():
    service = CampaignRankingService()
    with pytest.raises(ValueError):
        service.rank([])
    with pytest.raises(ValueError):
        service.rank([15], objective="bogus")


def test_rank_real_campaigns_integration():
    service = CampaignRankingService()
    result = service.rank([15, 20], objective="profit")
    assert result["ranking"]
    scores = [row["score"] for row in result["ranking"]]
    assert scores == sorted(scores, reverse=True)
    for row in result["ranking"]:
        assert set(row["scores"]) == {
            "incremental_profit",
            "incremental_sales",
            "roi",
            "cannibalization",
            "cost",
        }


def test_compare_reports_strengths_and_weaknesses():
    service = CampaignRankingService()
    result = service.compare([15, 20], objective="sales")
    assert len(result["strengths"]) == len(result["ranking"])
    assert len(result["weaknesses"]) == len(result["ranking"])
    for item in result["strengths"] + result["weaknesses"]:
        assert item["campaign_id"] in {15, 20}
        assert item["strength_score"] if item in result["strengths"] else item["weakness_score"] >= 0
