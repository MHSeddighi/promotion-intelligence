"""Multi-objective promotion recommendation engine.

Strategies (configurable weights):

- ``SALES_GROWTH``:        0.6 incremental_sales + 0.3 revenue + 0.1 reach
- ``PROFIT_OPTIMIZATION``: 0.6 incremental_profit + 0.2 roi - 0.2 cannibalization
- ``SAFE_PROMOTION``:      0.5 roi + 0.3 profit - 0.2 risk

The engine evaluates candidate products, normalizes the objectives across the
candidate set and returns ranked recommendations with explanations.
"""

from __future__ import annotations

from typing import Any

from app.services.analytics import AnalyticsService
from app.services.causal import CausalInferenceService
from app.services.financials import calculate_financial_metrics
from app.services.ranking import CampaignRankingService, normalize_values

STRATEGIES: dict[str, dict[str, float]] = {
    "SALES_GROWTH": {
        "incremental_sales": 0.6,
        "incremental_revenue": 0.3,
        "reach": 0.1,
    },
    "PROFIT_OPTIMIZATION": {
        "incremental_profit": 0.6,
        "roi": 0.2,
        "cannibalization": -0.2,
    },
    "SAFE_PROMOTION": {
        "roi": 0.5,
        "incremental_profit": 0.3,
        "cannibalization": -0.2,
    },
}

STRATEGY_ALIASES: dict[str, str] = {
    "sales": "SALES_GROWTH",
    "sales_growth": "SALES_GROWTH",
    "growth": "SALES_GROWTH",
    "profit": "PROFIT_OPTIMIZATION",
    "profit_optimization": "PROFIT_OPTIMIZATION",
    "safe": "SAFE_PROMOTION",
    "safe_promotion": "SAFE_PROMOTION",
}

DEFAULT_ELASTICITY = 2.0  # % lift per 1% discount when no historical estimate exists


def resolve_strategy(objective: str | None) -> str:
    """Map a user-facing objective to a strategy name."""
    if objective is None:
        return "PROFIT_OPTIMIZATION"
    key = objective.strip().upper()
    if key in STRATEGIES:
        return key
    alias_key = objective.strip().lower()
    if alias_key in STRATEGY_ALIASES:
        return STRATEGY_ALIASES[alias_key]
    raise ValueError(
        f"Unknown objective {objective!r}; choose one of {sorted(STRATEGIES)} "
        "or an alias: sales, profit, safe."
    )


class RecommendationService:
    """Rank product promotion opportunities under a chosen strategy."""

    def __init__(
        self,
        analytics: AnalyticsService | None = None,
        causal: CausalInferenceService | None = None,
        ranking: CampaignRankingService | None = None,
        margin: float = 0.25,
    ) -> None:
        self.analytics = analytics or AnalyticsService()
        self.causal = causal or CausalInferenceService(self.analytics)
        self.ranking = ranking or CampaignRankingService(self.analytics, self.causal, margin)
        self.margin = float(margin)

    # ------------------------------------------------------------ evaluation
    def evaluate_product(
        self,
        product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
        discount_percentage: float | None = None,
    ) -> dict[str, Any]:
        """Build the recommendation feature vector for one product."""
        start, end = self.analytics.resolve_period(
            start_week=start_week, end_week=end_week
        )
        baseline = self.analytics.product_baseline(product_id, start, end)
        if not baseline["data_sufficient"]:
            return {
                "product_id": int(product_id),
                "available": False,
                "message": baseline["message"],
            }

        causal = self.causal.causal_effect(product_id, start, end)
        lift = causal.get("causal_lift_percentage")
        if lift is None:
            lift = 0.0
        baseline_qty = float(baseline["baseline_qty"])
        expected_sales = baseline_qty * (1.0 + lift / 100.0)
        incremental_sales = expected_sales - baseline_qty

        average_price = self._average_price(product_id, start, end)
        promotion_cost = self._promotion_cost(product_id, start, end)
        cannibalization = self.analytics.cannibalization_effect(product_id, start, end)
        cannibalized_units = float(cannibalization["total_lost_quantity"])

        financials = calculate_financial_metrics(
            baseline_sales=baseline_qty,
            promotion_sales=expected_sales,
            average_price=average_price,
            margin=self.margin,
            promotion_cost=promotion_cost,
            cannibalized_units=cannibalized_units,
        )

        reach = self._product_reach(product_id, start, end)
        return {
            "product_id": int(product_id),
            "available": True,
            "start_week": start,
            "end_week": end,
            "baseline_sales": round(baseline_qty, 2),
            "expected_sales": round(expected_sales, 2),
            "incremental_sales": round(incremental_sales, 2),
            "causal_lift_percentage": round(lift, 2),
            "average_price": round(average_price, 4),
            "incremental_revenue": round(financials.incremental_revenue, 2),
            "incremental_profit": round(financials.incremental_profit, 2),
            "roi": financials.roi,
            "promotion_cost": round(promotion_cost, 2),
            "cannibalization_risk": float(cannibalization["cannibalization_percentage"]),
            "cannibalized_units": round(cannibalized_units, 2),
            "reach": reach,
            "confidence": baseline["confidence"],
        }

    def _average_price(self, product_id: int, start: int, end: int) -> float:
        return self.ranking._average_price(product_id, start, end)

    def _promotion_cost(self, product_id: int, start: int, end: int) -> float:
        return self.ranking._promotion_cost(product_id, start, end)

    def _product_reach(self, product_id: int, start: int, end: int) -> int:
        transactions = self.analytics.transactions
        mask = (
            (transactions["PRODUCT_ID"] == int(product_id))
            & transactions["WEEK_NO"].between(start, end)
        )
        if "household_key" not in transactions.columns:
            return 0
        return int(transactions.loc[mask, "household_key"].nunique())

    # ---------------------------------------------------------- recommend
    def recommend_products(
        self,
        products: list[int],
        budget: float | None = None,
        objective: str | None = None,
        constraints: dict[str, Any] | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Rank candidate products under a strategy and constraints."""
        if not products:
            raise ValueError("products must be non-empty.")
        if budget is not None and budget < 0:
            raise ValueError("budget must be non-negative.")
        strategy = resolve_strategy(objective)
        weights = STRATEGIES[strategy]
        constraints = constraints or {}

        candidates = [
            self.evaluate_product(pid, start_week, end_week)
            for pid in products
        ]
        available = [c for c in candidates if c.get("available")]
        if not available:
            return {
                "objective": strategy,
                "recommendations": [],
                "message": "None of the candidate products has sufficient data.",
            }

        # Constraints filtering.
        max_risk = constraints.get("max_cannibalization_risk")
        min_roi = constraints.get("min_roi")
        if max_risk is not None:
            available = [c for c in available if c["cannibalization_risk"] <= float(max_risk)]
        if min_roi is not None:
            available = [c for c in available if (c["roi"] or -1.0) >= float(min_roi)]
        if not available:
            return {
                "objective": strategy,
                "recommendations": [],
                "message": "No candidate satisfies the given constraints.",
            }

        objectives = list(weights)
        raw: dict[str, list[float]] = {name: [] for name in objectives}
        for candidate in available:
            for name in objectives:
                raw[name].append(self._objective_value(candidate, name))

        normalized: dict[str, list[float]] = {
            name: normalize_values(raw[name], "min_max", invert=(name == "cannibalization"))
            for name in objectives
        }

        recommendations: list[dict[str, Any]] = []
        for index, candidate in enumerate(available):
            scores = {name: normalized[name][index] for name in objectives}
            score = sum(float(weights[name]) * scores[name] for name in objectives)
            score = max(0.0, min(100.0, score * 100.0))
            recommendations.append(
                {
                    "product_id": candidate["product_id"],
                    "score": round(score, 1),
                    "objective_scores": {
                        name: round(value, 4) for name, value in scores.items()
                    },
                    "expected_sales": candidate["expected_sales"],
                    "expected_profit": candidate["incremental_profit"],
                    "expected_revenue": candidate["incremental_revenue"],
                    "roi": candidate["roi"],
                    "cannibalization_risk": candidate["cannibalization_risk"],
                    "promotion_cost": candidate["promotion_cost"],
                    "confidence": candidate["confidence"],
                    "explanation": self._explain(candidate, scores, weights),
                }
            )
        recommendations.sort(key=lambda row: row["score"], reverse=True)

        # Budget-constrained selection (greedy by score).
        if budget is not None:
            selected: list[dict[str, Any]] = []
            spent = 0.0
            for row in recommendations:
                cost = float(row["promotion_cost"])
                if spent + cost <= budget:
                    selected.append(row)
                    spent += cost
            recommendations = selected

        return {
            "objective": strategy,
            "weights": weights,
            "recommendations": recommendations,
            "budget": budget,
            "total_promotion_cost": round(
                sum(float(row["promotion_cost"]) for row in recommendations), 2
            ),
            "message": "",
        }

    @staticmethod
    def _objective_value(candidate: dict[str, Any], name: str) -> float:
        if name == "incremental_sales":
            return candidate["incremental_sales"]
        if name == "incremental_revenue":
            return candidate["incremental_revenue"]
        if name == "incremental_profit":
            return candidate["incremental_profit"]
        if name == "roi":
            return candidate["roi"] if candidate["roi"] is not None else 0.0
        if name == "cannibalization":
            return candidate["cannibalization_risk"]
        if name == "reach":
            return float(candidate["reach"])
        raise ValueError(f"Unknown objective for scoring: {name}")

    @staticmethod
    def _explain(
        candidate: dict[str, Any],
        scores: dict[str, float],
        weights: dict[str, float],
    ) -> str:
        parts: list[str] = []
        profit = candidate["incremental_profit"]
        roi = candidate["roi"]
        risk = candidate["cannibalization_risk"]
        if profit > 0:
            parts.append(f"high profit opportunity (${profit:,.0f})")
        elif profit < 0:
            parts.append("profit-negative at the current margin")
        else:
            parts.append("break-even profit potential")
        if roi is not None:
            parts.append(f"ROI of {roi:.2f}")
        if risk <= 5:
            parts.append("limited product substitution")
        elif risk <= 20:
            parts.append("moderate cannibalization risk")
        else:
            parts.append("high cannibalization risk")
        best = max(scores, key=scores.get)
        strategy_name = max(weights, key=weights.get)
        return f"{parts[0].capitalize()}, {', '.join(parts[1:])}. Strongest objective: {best}."

    # ------------------------------------------------------------ simulation
    def simulate_campaign(
        self,
        product_id: int,
        discount_percentage: float,
        weeks: int,
        start_week: int | None = None,
        elasticity: float | None = None,
    ) -> dict[str, Any]:
        """Simulate a hypothetical discount campaign for a product."""
        if not 0.0 < discount_percentage <= 90.0:
            raise ValueError("discount_percentage must be in (0, 90].")
        if weeks < 1 or weeks > 52:
            raise ValueError("weeks must be between 1 and 52.")
        if elasticity is not None and elasticity < 0:
            raise ValueError("elasticity must be non-negative.")

        start, end = self.analytics.resolve_period(start_week=start_week, end_week=None)
        end = min(int(start) + weeks - 1, int(self.analytics.panel["WEEK_NO"].max()))
        baseline = self.analytics.product_baseline(product_id, start, end)
        if not baseline["data_sufficient"]:
            return {
                "product_id": int(product_id),
                "available": False,
                "message": baseline["message"],
            }

        elastic = elasticity if elasticity is not None else DEFAULT_ELASTICITY
        # Clamp the implied lift to a plausible range.
        implied_lift = min(max(discount_percentage * elastic, -25.0), 400.0)
        baseline_qty = float(baseline["baseline_qty"])
        expected_sales = baseline_qty * (1.0 + implied_lift / 100.0)
        incremental_sales = expected_sales - baseline_qty

        average_price = self._average_price(product_id, start, end)
        # Discount spend applies to expected sold units.
        promotion_cost = (
            expected_sales * average_price * (discount_percentage / 100.0)
            if average_price > 0
            else 0.0
        )
        cannibalization = self.analytics.cannibalization_effect(product_id, start, end)
        cannibalized_units = float(cannibalization["total_lost_quantity"])
        financials = calculate_financial_metrics(
            baseline_sales=baseline_qty,
            promotion_sales=expected_sales,
            average_price=average_price,
            margin=self.margin,
            promotion_cost=promotion_cost,
            cannibalized_units=cannibalized_units,
        )

        return {
            "product_id": int(product_id),
            "available": True,
            "start_week": start,
            "end_week": end,
            "discount_percentage": discount_percentage,
            "weeks": weeks,
            "elasticity_assumed": elastic,
            "assumptions": [
                f"Demand elasticity of {elastic:.1f} (% lift per 1% discount); "
                "based on the historical average if no product-specific estimate exists.",
                "Cannibalization carryover uses the saved cannibalization model.",
            ],
            "baseline_sales": round(baseline_qty, 2),
            "expected_sales": round(expected_sales, 2),
            "implied_lift_percentage": round(implied_lift, 2),
            "incremental_sales": round(incremental_sales, 2),
            "incremental_revenue": round(financials.incremental_revenue, 2),
            "incremental_profit": round(financials.incremental_profit, 2),
            "roi": financials.roi,
            "promotion_cost": round(promotion_cost, 2),
            "cannibalization_risk": float(cannibalization["cannibalization_percentage"]),
            "cannibalized_units": round(cannibalized_units, 2),
            "confidence": baseline["confidence"],
            "risks": self._simulation_risks(
                financials.roi, cannibalization["cannibalization_percentage"]
            ),
        }

    @staticmethod
    def _simulation_risks(roi: float | None, cannibalization_percentage: float) -> list[str]:
        risks: list[str] = []
        if roi is not None and roi < 1.0:
            risks.append("ROI is below 1.0; the discount may not pay back its cost.")
        if cannibalization_percentage >= 20:
            risks.append(
                f"Cannibalization risk is high ({cannibalization_percentage:.0f}% "
                "of promoted volume)."
            )
        if not risks:
            risks.append("No major risk flags detected in the simulation.")
        return risks
