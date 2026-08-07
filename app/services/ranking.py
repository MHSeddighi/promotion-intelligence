"""Campaign ranking with normalized multi-objective scores.

Supported normalizations:

- ``min_max``   -> (value - min) / (max - min), inverted for cost metrics
- ``z_score``   -> (value - mean) / std, clipped to [-3, 3] then scaled to [0, 1]
- ``percentile``-> rank-based percentile in [0, 1]

Weighted score (default weights are configurable):

    score = w1*profit + w2*sales + w3*roi - w4*cannibalization - w5*cost
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from app.services.analytics import AnalyticsService
from app.services.causal import CausalInferenceService
from app.services.financials import calculate_financial_metrics

Normalization = Literal["min_max", "z_score", "percentile"]

OBJECTIVE_PRESETS: dict[str, dict[str, float]] = {
    "profit": {
        "incremental_profit": 0.40,
        "incremental_sales": 0.20,
        "roi": 0.20,
        "cannibalization": -0.10,
        "cost": -0.10,
    },
    "sales": {
        "incremental_profit": 0.15,
        "incremental_sales": 0.50,
        "roi": 0.15,
        "cannibalization": -0.10,
        "cost": -0.10,
    },
    "efficiency": {
        "incremental_profit": 0.20,
        "incremental_sales": 0.10,
        "roi": 0.50,
        "cannibalization": -0.10,
        "cost": -0.10,
    },
}


def normalize_values(
    values: list[float], method: Normalization = "min_max", invert: bool = False
) -> list[float]:
    """Normalize a list of values to [0, 1].

    ``invert=True`` reverses the direction so that lower raw values score higher
    (used for cost and cannibalization-risk metrics).
    """
    array = np.asarray([float(value) for value in values], dtype="float64")
    if array.size == 0:
        return []
    if np.allclose(array, array[0]):
        return [0.5 for _ in array]

    if method == "min_max":
        low, high = float(array.min()), float(array.max())
        scaled = (array - low) / (high - low) if high > low else np.zeros_like(array)
    elif method == "z_score":
        mean, std = float(array.mean()), float(array.std())
        z = (array - mean) / std if std > 0 else np.zeros_like(array)
        z = np.clip(z, -3.0, 3.0)
        scaled = (z + 3.0) / 6.0
    elif method == "percentile":
        scaled = pd.Series(array).rank(pct=True).to_numpy(dtype="float64")
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    result = [float(value) for value in scaled]
    if invert:
        result = [1.0 - value for value in result]
    return result


def weighted_score(
    scores: dict[str, float], weights: dict[str, float]
) -> float:
    """Dot product of objective scores and (possibly signed) weights."""
    if not scores or not weights:
        raise ValueError("scores and weights must be non-empty.")
    total = 0.0
    for name, weight in weights.items():
        if name not in scores:
            raise ValueError(f"Weight references unknown objective: {name}")
        total += float(weight) * float(scores[name])
    return total


class CampaignRankingService:
    """Rank historical/simulated campaigns on normalized objectives."""

    def __init__(
        self,
        analytics: AnalyticsService | None = None,
        causal: CausalInferenceService | None = None,
        margin: float = 0.25,
    ) -> None:
        self.analytics = analytics or AnalyticsService()
        self.causal = causal or CausalInferenceService(self.analytics)
        self.margin = float(margin)
        if not 0.0 <= self.margin <= 1.0:
            raise ValueError("margin must be between 0 and 1.")
        self._campaign_metrics_cache: dict[int, dict[str, Any]] = {}
        self._cannibalization_cache: dict[int, dict[str, Any]] = {}
        self._households: pd.Series | None = None

    # ----------------------------------------------------------------- data
    @property
    def households_per_campaign(self) -> pd.Series:
        if self._households is None:
            frame = pd.read_csv(
                self.analytics.data_dir / "campaign_table.csv",
                usecols=["household_key", "CAMPAIGN"],
            )
            self._households = frame.groupby("CAMPAIGN")["household_key"].nunique()
        return self._households

    def _average_price(self, product_id: int, start_week: int, end_week: int) -> float:
        transactions = self.analytics.transactions
        mask = (
            (transactions["PRODUCT_ID"] == int(product_id))
            & transactions["WEEK_NO"].between(start_week, end_week)
            & (transactions["QUANTITY"] > 0)
        )
        rows = transactions.loc[mask]
        if rows.empty:
            return 0.0
        return float(rows["SALES_VALUE"].sum() / rows["QUANTITY"].sum())

    def _promotion_cost(self, product_id: int, start_week: int, end_week: int) -> float:
        transactions = self.analytics.transactions
        mask = (
            (transactions["PRODUCT_ID"] == int(product_id))
            & transactions["WEEK_NO"].between(start_week, end_week)
        )
        rows = transactions.loc[mask]
        if rows.empty:
            return 0.0
        retailer = rows["RETAIL_DISC"].abs() + rows["COUPON_MATCH_DISC"].abs()
        return float(retailer.sum())

    def _cannibalization_summary(self, product_id: int) -> dict[str, Any]:
        if product_id in self._cannibalization_cache:
            return self._cannibalization_cache[product_id]
        result = self.analytics.cannibalization_effect(product_id)
        self._cannibalization_cache[product_id] = result
        return result

    # ------------------------------------------------------------ metrics
    def campaign_metrics(self, campaign_id: int) -> dict[str, Any]:
        """Compute the raw objective metrics for one campaign."""
        if campaign_id in self._campaign_metrics_cache:
            return self._campaign_metrics_cache[campaign_id]

        effect = self.analytics.campaign_effect(campaign_id)
        if "error" in effect or not effect.get("data_sufficient"):
            return {
                "campaign_id": int(campaign_id),
                "available": False,
                "message": effect.get("error") or effect.get("message", "No data."),
            }

        campaign = self.analytics.get_campaign(campaign_id)
        start, end = effect["start_week"], effect["end_week"]
        baseline_sales = float(effect["baseline_sales"])
        promotion_sales = float(effect["actual_sales"])
        incremental_sales = float(effect["incremental_sales"])
        uplift = effect["uplift_percentage"]

        # Average price across analyzed products (panel basis).
        avg_prices = [
            self._average_price(pid, start, end)
            for pid in self._target_products(campaign, effect)
        ]
        average_price = (
            float(np.mean([price for price in avg_prices if price > 0])) if any(avg_prices) else 0.0
        )

        promotion_cost = sum(
            self._promotion_cost(pid, start, end)
            for pid in self._target_products(campaign, effect)
        )
        financials = calculate_financial_metrics(
            baseline_sales=baseline_sales,
            promotion_sales=promotion_sales,
            average_price=average_price,
            margin=self.margin,
            promotion_cost=promotion_cost,
            cannibalized_units=0.0,
        )

        cannibalization = self._campaign_cannibalization(campaign)
        revenue_impact = financials.incremental_revenue
        margin_impact = financials.incremental_profit
        reach = int(self.households_per_campaign.get(campaign_id, 0)) + int(
            campaign["n_products"]
        )
        cost_efficiency = (
            revenue_impact / promotion_cost if promotion_cost > 0 else None
        )

        metrics = {
            "campaign_id": int(campaign_id),
            "available": True,
            "description": campaign["description"],
            "start_week": start,
            "end_week": end,
            "baseline_sales": round(baseline_sales, 2),
            "promotion_sales": round(promotion_sales, 2),
            "incremental_sales": round(incremental_sales, 2),
            "uplift_percentage": uplift,
            "incremental_revenue": round(revenue_impact, 2),
            "incremental_profit": round(margin_impact, 2),
            "roi": financials.roi,
            "promotion_cost": round(promotion_cost, 2),
            "cost_efficiency": round(cost_efficiency, 4) if cost_efficiency is not None else None,
            "customer_reach": reach,
            "cannibalization_risk": cannibalization["risk_percentage"],
            "cannibalized_units": cannibalization["total_lost"],
            "average_price": round(average_price, 4),
            "margin": self.margin,
        }
        self._campaign_metrics_cache[campaign_id] = metrics
        return metrics

    def _target_products(self, campaign: dict[str, Any], effect: dict[str, Any]) -> list[int]:
        panel_products = set(self.analytics.panel["PRODUCT_ID"].unique())
        targets = [int(pid) for pid in campaign["products"] if pid in panel_products]
        if effect.get("product_id") is not None:
            targets = [int(effect["product_id"])]
        return targets

    def _campaign_cannibalization(self, campaign: dict[str, Any]) -> dict[str, Any]:
        total_lost = 0.0
        max_percentage = 0.0
        for product_id in campaign["products"][:20]:
            summary = self._cannibalization_summary(int(product_id))
            total_lost += float(summary["total_lost_quantity"])
            max_percentage = max(max_percentage, float(summary["cannibalization_percentage"]))
        return {
            "total_lost": round(total_lost, 2),
            "risk_percentage": round(max_percentage, 2),
        }

    # ------------------------------------------------------------- ranking
    def rank(
        self,
        campaign_ids: list[int],
        objective: str = "profit",
        weights: dict[str, float] | None = None,
        normalize: Normalization = "min_max",
    ) -> dict[str, Any]:
        """Rank campaigns by weighted normalized objectives."""
        if not campaign_ids:
            raise ValueError("campaign_ids must be non-empty.")
        if objective not in OBJECTIVE_PRESETS and weights is None:
            raise ValueError(
                f"Unknown objective {objective!r}; provide explicit weights or "
                f"use one of {sorted(OBJECTIVE_PRESETS)}."
            )
        effective_weights = weights or OBJECTIVE_PRESETS[objective]

        metrics = [self.campaign_metrics(cid) for cid in campaign_ids]
        available = [m for m in metrics if m.get("available")]
        if not available:
            return {
                "objective": objective,
                "normalization": normalize,
                "ranking": [],
                "scores": [],
                "message": "No campaign has sufficient data to rank.",
            }

        columns = ["incremental_profit", "incremental_sales", "roi", "cannibalization", "cost"]
        raw: dict[str, list[float]] = {name: [] for name in columns}
        for metric in available:
            raw["incremental_profit"].append(metric["incremental_profit"])
            raw["incremental_sales"].append(metric["incremental_sales"])
            raw["roi"].append(metric["roi"] if metric["roi"] is not None else 0.0)
            raw["cannibalization"].append(metric["cannibalization_risk"])
            raw["cost"].append(metric["promotion_cost"])

        normalized: dict[str, list[float]] = {
            "incremental_profit": normalize_values(raw["incremental_profit"], normalize),
            "incremental_sales": normalize_values(raw["incremental_sales"], normalize),
            "roi": normalize_values(raw["roi"], normalize),
            "cannibalization": normalize_values(raw["cannibalization"], normalize, invert=True),
            "cost": normalize_values(raw["cost"], normalize, invert=True),
        }

        rows: list[dict[str, Any]] = []
        for index, metric in enumerate(available):
            scores = {name: normalized[name][index] for name in columns}
            total = weighted_score(scores, effective_weights)
            rows.append(
                {
                    "campaign_id": metric["campaign_id"],
                    "description": metric["description"],
                    "score": round(total, 6),
                    "scores": {name: round(value, 6) for name, value in scores.items()},
                    "incremental_profit": metric["incremental_profit"],
                    "incremental_sales": metric["incremental_sales"],
                    "roi": metric["roi"],
                    "cannibalization_risk": metric["cannibalization_risk"],
                    "promotion_cost": metric["promotion_cost"],
                }
            )
        rows.sort(key=lambda row: row["score"], reverse=True)
        return {
            "objective": objective,
            "weights": effective_weights,
            "normalization": normalize,
            "ranking": rows,
            "scores": [row["scores"] for row in rows],
            "message": "",
        }

    def compare(
        self,
        campaign_ids: list[int],
        objective: str = "profit",
        weights: dict[str, float] | None = None,
        normalize: Normalization = "min_max",
    ) -> dict[str, Any]:
        """Compare campaigns: ranking, strengths and weaknesses per campaign."""
        result = self.rank(campaign_ids, objective, weights, normalize)
        strengths: list[dict[str, Any]] = []
        weaknesses: list[dict[str, Any]] = []
        for row in result["ranking"]:
            scores = row["scores"]
            best = max(scores, key=scores.get)
            worst = min(scores, key=scores.get)
            strengths.append(
                {
                    "campaign_id": row["campaign_id"],
                    "strength": best,
                    "strength_score": round(scores[best], 4),
                }
            )
            weaknesses.append(
                {
                    "campaign_id": row["campaign_id"],
                    "weakness": worst,
                    "weakness_score": round(scores[worst], 4),
                }
            )
        result["strengths"] = strengths
        result["weaknesses"] = weaknesses
        return result
