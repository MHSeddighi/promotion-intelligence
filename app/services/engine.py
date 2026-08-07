"""Unified promotion analysis engine.

Combines baseline prediction, causal uplift, cannibalization, financial metrics
and recommendation features into one result, so callers (APIs, MCP tools, LLM
agent, frontend) never orchestrate the individual services themselves.
"""

from __future__ import annotations

from typing import Any

from app.services.analytics import AnalyticsService
from app.services.causal import CausalInferenceService
from app.services.financials import calculate_financial_metrics
from app.services.recommendation import RecommendationService


class PromotionAnalysisEngine:
    """Production inference layer over all promotion-intelligence modules."""

    def __init__(
        self,
        analytics: AnalyticsService | None = None,
        causal: CausalInferenceService | None = None,
        recommendation: RecommendationService | None = None,
        margin: float = 0.25,
    ) -> None:
        self.analytics = analytics or AnalyticsService()
        self.causal = causal or CausalInferenceService(self.analytics)
        self.recommendation = recommendation or RecommendationService(
            self.analytics, self.causal, margin=margin
        )
        self.margin = float(margin)

    def analyze(
        self,
        product_id: int,
        promotion_id: int | None = None,
        store_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Full analysis for one product, optionally tied to a campaign."""
        start, end = self.analytics.resolve_period(
            campaign_id=promotion_id,
            start_week=start_week,
            end_week=end_week,
        )
        end = min(end, int(self.analytics.panel["WEEK_NO"].max()))

        baseline = self.analytics.product_baseline(product_id, start, end)
        if not baseline["data_sufficient"]:
            return {
                "product_id": int(product_id),
                "promotion_id": promotion_id,
                "store_id": store_id,
                "start_week": start,
                "end_week": end,
                "data_sufficient": False,
                "confidence": "low",
                "message": baseline["message"],
                "baseline_prediction": None,
                "promotion_prediction": None,
                "incremental_sales": None,
                "uplift": None,
                "cannibalization": None,
                "financial_metrics": None,
                "recommendation_features": None,
            }

        causal = self.causal.causal_effect(product_id, start, end)
        cannibalization = self.analytics.cannibalization_effect(product_id, start, end)

        # Promotion prediction: historical campaign actuals when available,
        # otherwise the causal-lift adjusted baseline.
        promotion_prediction: float | None = None
        actual_sales: float | None = None
        if promotion_id is not None:
            effect = self.analytics.campaign_effect(
                promotion_id, product_id=product_id, start_week=start, end_week=end
            )
            if "error" not in effect and effect.get("data_sufficient"):
                promotion_prediction = float(effect["actual_sales"])
                actual_sales = float(effect["actual_sales"])
        if promotion_prediction is None:
            lift = causal.get("causal_lift_percentage") or 0.0
            promotion_prediction = float(baseline["baseline_qty"]) * (1.0 + lift / 100.0)

        baseline_qty = float(baseline["baseline_qty"])
        incremental_sales = promotion_prediction - baseline_qty
        uplift = (
            (incremental_sales / baseline_qty * 100.0) if baseline_qty > 0 else None
        )

        avg_price = self.recommendation._average_price(product_id, start, end)
        promotion_cost = self.recommendation._promotion_cost(product_id, start, end)
        financials = calculate_financial_metrics(
            baseline_sales=baseline_qty,
            promotion_sales=promotion_prediction,
            average_price=avg_price,
            margin=self.margin,
            promotion_cost=promotion_cost,
            cannibalized_units=float(cannibalization["total_lost_quantity"]),
        )

        recommendation_features = {
            "causal_lift_percentage": causal.get("causal_lift_percentage"),
            "roi": financials.roi,
            "incremental_profit": financials.incremental_profit,
            "cannibalization_risk": float(cannibalization["cannibalization_percentage"]),
            "confidence": baseline["confidence"],
            "recommendation_score": self._score_recommendation(
                financials.incremental_profit,
                financials.roi,
                float(cannibalization["cannibalization_percentage"]),
            ),
        }

        return {
            "product_id": int(product_id),
            "promotion_id": promotion_id,
            "store_id": store_id,
            "start_week": start,
            "end_week": end,
            "data_sufficient": True,
            "confidence": baseline["confidence"],
            "message": baseline.get("message", ""),
            "baseline_prediction": round(baseline_qty, 2),
            "promotion_prediction": round(promotion_prediction, 2),
            "actual_sales": round(actual_sales, 2) if actual_sales is not None else None,
            "incremental_sales": round(incremental_sales, 2),
            "uplift": round(uplift, 2) if uplift is not None else None,
            "cannibalization": {
                "signal": cannibalization["signal"],
                "total_lost_quantity": cannibalization["total_lost_quantity"],
                "cannibalization_percentage": cannibalization["cannibalization_percentage"],
                "affected_products": cannibalization["affected_products"],
                "summary": cannibalization["summary"],
            },
            "financial_metrics": financials.to_dict(),
            "recommendation_features": recommendation_features,
            "weekly": baseline.get("weekly", []),
        }

    @staticmethod
    def _score_recommendation(
        incremental_profit: float, roi: float | None, cannibalization_risk: float
    ) -> float:
        """0-100 heuristic score combining profit, ROI and risk."""
        profit_score = min(50.0, max(0.0, incremental_profit / 100.0))
        roi_score = min(25.0, max(0.0, (roi or 0.0) * 5.0))
        risk_penalty = min(25.0, cannibalization_risk)
        return round(max(0.0, profit_score + roi_score + 25.0 - risk_penalty), 1)

    def generate_report(
        self,
        product_id: int,
        promotion_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Executive report: summary, metrics, data and risks."""
        analysis = self.analyze(product_id, promotion_id, start_week=start_week, end_week=end_week)
        if not analysis["data_sufficient"]:
            return {
                "title": f"Promotion report for product {product_id}",
                "summary": analysis["message"],
                "metrics": [],
                "data": analysis,
                "recommendations": ["Improve demand-history coverage before promoting."],
                "risks": ["Insufficient data to quantify promotion risk."],
            }

        financials = analysis["financial_metrics"]
        cannibalization = analysis["cannibalization"]
        roi = financials["roi"]
        profit = financials["incremental_profit"]

        if analysis["incremental_sales"] > 0 and roi is not None and roi > 1.0:
            recommendation = (
                f"Promote product {product_id}: positive incremental volume "
                f"({analysis['incremental_sales']:,.0f} units) with ROI {roi:.2f}."
            )
        elif profit > 0:
            recommendation = (
                f"Consider promoting product {product_id}: profitable despite "
                "a moderate ROI; monitor cannibalization."
            )
        else:
            recommendation = (
                f"Avoid promoting product {product_id}: incremental profit is "
                "not positive at the assumed margin."
            )

        risks: list[str] = []
        if cannibalization["signal"] == "strong":
            risks.append(
                f"Strong cannibalization signal: {cannibalization['total_lost_quantity']:,.0f} "
                "units lost by other products."
            )
        if roi is not None and roi < 1.0:
            risks.append("ROI is below 1.0; the promotion may destroy value.")
        if not risks:
            risks.append("No major risk flags detected.")

        metrics = [
            {"name": "baseline_sales", "value": analysis["baseline_prediction"]},
            {"name": "promotion_prediction", "value": analysis["promotion_prediction"]},
            {"name": "incremental_sales", "value": analysis["incremental_sales"]},
            {"name": "uplift_percentage", "value": analysis["uplift"]},
            {"name": "incremental_revenue", "value": financials["incremental_revenue"]},
            {"name": "incremental_profit", "value": profit},
            {"name": "roi", "value": roi},
            {"name": "cannibalized_units", "value": cannibalization["total_lost_quantity"]},
        ]
        return {
            "title": f"Executive promotion report — product {product_id}",
            "summary": (
                f"Baseline demand is {analysis['baseline_prediction']:,.0f} units; with "
                f"promotion, expected sales are {analysis['promotion_prediction']:,.0f} units "
                f"({analysis['uplift']:+,.1f}%), delivering ${profit:,.0f} incremental profit "
                f"at a {self.margin:.0%} margin."
            ),
            "metrics": metrics,
            "data": analysis,
            "recommendations": [recommendation],
            "risks": risks,
        }
