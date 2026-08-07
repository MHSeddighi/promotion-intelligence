"""MCP tool registry for promotion analytics.

Each tool maps 1:1 to an ``AnalyticsService`` method. The same registry is
consumed by:

- the stdio MCP server (``mcp_server.py``), and
- the LLM agent (``llm_agent.py``) as OpenAI function-calling tools.

The LLM agent must call these tools instead of raw SQL whenever an analysis
method exists (campaign effect, baseline, incremental sales, cannibalization).
"""

from __future__ import annotations

from typing import Any, Callable

from app.models.similarity import ProductSimilarityService
from app.services.analytics import AnalyticsService
from app.services.attribution import IncrementalSalesAttributionService
from app.services.causal import CausalInferenceService
from app.services.engine import PromotionAnalysisEngine
from app.services.ranking import CampaignRankingService
from app.services.recommendation import RecommendationService


class MCPTools:
    """Read-only analytics tools exposed to LLM agents and the MCP server."""

    def __init__(
        self,
        service: AnalyticsService | None = None,
        causal: CausalInferenceService | None = None,
        similarity: ProductSimilarityService | None = None,
        ranking: CampaignRankingService | None = None,
        recommendation: RecommendationService | None = None,
        engine: PromotionAnalysisEngine | None = None,
        attribution: IncrementalSalesAttributionService | None = None,
    ) -> None:
        self.service = service or AnalyticsService()
        self.causal = causal or CausalInferenceService(self.service)
        self.similarity = similarity or ProductSimilarityService()
        self.ranking = ranking or CampaignRankingService(self.service, self.causal)
        self.recommendation = recommendation or RecommendationService(
            self.service, self.causal, self.ranking
        )
        self.engine = engine or PromotionAnalysisEngine(
            self.service, self.causal, self.recommendation
        )
        self.attribution = attribution or IncrementalSalesAttributionService()

    # ------------------------------------------------------------- promotion
    def list_campaigns(self) -> dict[str, Any]:
        return {"campaigns": self.service.list_campaigns()}

    def get_campaign(self, campaign_id: int) -> dict[str, Any]:
        campaign = self.service.get_campaign(campaign_id)
        if campaign is None:
            return {"error": f"Campaign {campaign_id} not found."}
        return campaign

    # ----------------------------------------------------------------- effect
    def get_campaign_effect(
        self,
        campaign_id: int,
        product_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        return self.service.campaign_effect(campaign_id, product_id, start_week, end_week)

    def get_product_baseline(
        self,
        product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        return self.service.product_baseline(product_id, start_week, end_week)

    def get_incremental_sales(
        self,
        campaign_id: int,
        product_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        return self.service.incremental_sales(campaign_id, product_id, start_week, end_week)

    # ---------------------------------------------------------- cannibalization
    def get_cannibalization_effect(
        self,
        promoted_product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        return self.service.cannibalization_effect(promoted_product_id, start_week, end_week)

    def get_top_impacted_products(
        self,
        promoted_product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
        top_n: int = 10,
    ) -> dict[str, Any]:
        return self.service.top_impacted_products(promoted_product_id, start_week, end_week, top_n)

    # ------------------------------------------------- copilot tools (v2)
    def get_baseline_prediction(
        self,
        product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Alias of get_product_baseline with the copilot-facing name."""
        return self.get_product_baseline(product_id, start_week, end_week)

    def calculate_incremental_sales(
        self,
        campaign_id: int,
        product_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Alias of get_incremental_sales with the copilot-facing name."""
        return self.get_incremental_sales(campaign_id, product_id, start_week, end_week)

    def detect_cannibalization(
        self,
        promoted_product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Alias of get_cannibalization_effect with the copilot-facing name."""
        return self.get_cannibalization_effect(promoted_product_id, start_week, end_week)

    def find_similar_products(
        self, product_id: int, top_k: int = 10
    ) -> dict[str, Any]:
        """Find the most similar products using the embedding model."""
        try:
            similar = self.similarity.find_similar_products(int(product_id), top_k=int(top_k))
            substitutes = self.similarity.find_substitutes(int(product_id), top_k=int(top_k))
        except KeyError as exc:
            return {"error": str(exc)}
        return {
            "product_id": int(product_id),
            "similar_products": similar,
            "substitutes": substitutes,
            "count": len(similar),
        }

    def calculate_financial_metrics(
        self,
        product_id: int,
        campaign_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
        margin: float = 0.25,
    ) -> dict[str, Any]:
        """Financial metrics for a product/campaign: incremental revenue,
        incremental profit, promotion cost and ROI."""
        try:
            analysis = self.engine.analyze(
                int(product_id),
                promotion_id=int(campaign_id) if campaign_id is not None else None,
                start_week=start_week,
                end_week=end_week,
            )
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}
        return {
            "product_id": int(product_id),
            "campaign_id": campaign_id,
            "financial_metrics": analysis.get("financial_metrics"),
            "incremental_sales": analysis.get("incremental_sales"),
            "data_sufficient": analysis.get("data_sufficient"),
        }

    def rank_campaigns(
        self,
        campaign_ids: list[int],
        objective: str = "profit",
        normalization: str = "min_max",
    ) -> dict[str, Any]:
        """Rank campaigns by weighted normalized objectives."""
        try:
            return self.ranking.rank(
                [int(cid) for cid in campaign_ids],
                objective=objective,
                normalize=normalization,  # type: ignore[arg-type]
            )
        except ValueError as exc:
            return {"error": str(exc)}

    def compare_campaigns(
        self,
        campaign_ids: list[int],
        objective: str = "profit",
        normalization: str = "min_max",
    ) -> dict[str, Any]:
        """Compare campaigns and report strengths/weaknesses per campaign."""
        try:
            return self.ranking.compare(
                [int(cid) for cid in campaign_ids],
                objective=objective,
                normalize=normalization,  # type: ignore[arg-type]
            )
        except ValueError as exc:
            return {"error": str(exc)}

    def recommend_campaigns(
        self,
        products: list[int],
        objective: str | None = None,
        budget: float | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Recommend which products to promote under a strategy."""
        try:
            return self.recommendation.recommend_products(
                [int(pid) for pid in products],
                objective=objective,
                budget=budget,
                start_week=start_week,
                end_week=end_week,
            )
        except ValueError as exc:
            return {"error": str(exc)}

    def simulate_campaign_strategy(
        self,
        product_id: int,
        discount_percentage: float,
        weeks: int,
        start_week: int | None = None,
        elasticity: float | None = None,
    ) -> dict[str, Any]:
        """Simulate a hypothetical discount campaign and its ROI/risks."""
        try:
            return self.recommendation.simulate_campaign(
                int(product_id),
                discount_percentage=float(discount_percentage),
                weeks=int(weeks),
                start_week=start_week,
                elasticity=elasticity,
            )
        except ValueError as exc:
            return {"error": str(exc)}

    def generate_promotion_report(
        self,
        product_id: int,
        campaign_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Generate an executive promotion report."""
        try:
            return self.engine.generate_report(
                int(product_id),
                promotion_id=int(campaign_id) if campaign_id is not None else None,
                start_week=start_week,
                end_week=end_week,
            )
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}


    # ---------------------------------------------------------- attribution
    def analyze_incremental_sales_attribution(
        self,
        promotion_id: str,
        product_id: int,
        start_week: int,
        end_week: int,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Explain why a promotion generated incremental sales (actual - baseline)."""
        try:
            return self.attribution.analyze(
                promotion_id=promotion_id,
                product_id=product_id,
                start_week=start_week,
                end_week=end_week,
                top_k=top_k,
            )
        except (FileNotFoundError, ValueError) as exc:
            return {"error": str(exc)}

    # --------------------------------------------------------------- dispatch
    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        tool = TOOL_DEFINITIONS.get(name)
        if tool is None:
            raise ValueError(f"Unknown MCP tool: {name}")
        return tool["callable"](self, **(arguments or {}))


def _tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    callable_: Callable[..., Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "callable": callable_,
        "openai_schema": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": [key for key, spec in parameters.items() if spec.get("__required__")],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    }


def _required(spec: dict[str, Any]) -> dict[str, Any]:
    return {**spec, "__required__": True}


WEEK = {"type": "integer", "minimum": 1, "maximum": 102, "description": "Week number (1-102)."}
PRODUCT = {"type": "integer", "minimum": 1, "description": "Product ID."}
CAMPAIGN = {"type": "integer", "minimum": 1, "description": "Campaign ID."}

TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "list_campaigns": _tool(
        name="list_campaigns",
        description=(
            "List all promotion campaigns with their week ranges and number of "
            "promoted products. Use this to discover valid campaign IDs."
        ),
        parameters={},
        callable_=MCPTools.list_campaigns,
    ),
    "get_campaign": _tool(
        name="get_campaign",
        description=(
            "Load promotion data for one campaign: description, start/end days "
            "and weeks, and the promoted product IDs."
        ),
        parameters={"campaign_id": _required(CAMPAIGN)},
        callable_=MCPTools.get_campaign,
    ),
    "get_campaign_effect": _tool(
        name="get_campaign_effect",
        description=(
            "Analyze the promotion effect of a campaign: actual sales, baseline "
            "predicted sales (without promotion), incremental sales, uplift "
            "percentage and an effectiveness summary. Pass product_id to focus "
            "on one product, or omit it to aggregate all forecastable campaign "
            "products. The response includes data_sufficient/confidence so you "
            "can communicate uncertainty."
        ),
        parameters={
            "campaign_id": _required(CAMPAIGN),
            "product_id": PRODUCT,
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.get_campaign_effect,
    ),
    "get_product_baseline": _tool(
        name="get_product_baseline",
        description=(
            "Load baseline demand predictions (counterfactual sales without "
            "promotion) for a product over a week range, aggregated across the "
            "stores covered by the baseline model. The response includes "
            "confidence and data_sufficient flags."
        ),
        parameters={
            "product_id": _required(PRODUCT),
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.get_product_baseline,
    ),
    "get_incremental_sales": _tool(
        name="get_incremental_sales",
        description=(
            "Calculate incremental sales for a campaign/product: "
            "incremental = actual sales - baseline predicted sales. Returns "
            "actual, baseline, incremental and uplift percentage."
        ),
        parameters={
            "campaign_id": _required(CAMPAIGN),
            "product_id": PRODUCT,
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.get_incremental_sales,
    ),
    "get_cannibalization_effect": _tool(
        name="get_cannibalization_effect",
        description=(
            "Load cannibalization results for a promoted product: affected "
            "products, estimated lost quantity, cannibalization percentage and "
            "signal strength (strong/weak/none). When signal is weak or none, "
            "the model found no strong cannibalization evidence."
        ),
        parameters={
            "promoted_product_id": _required(PRODUCT),
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.get_cannibalization_effect,
    ),
    "get_top_impacted_products": _tool(
        name="get_top_impacted_products",
        description=(
            "Return the top-N products most impacted (largest estimated lost "
            "quantity) by a promoted product over a week range."
        ),
        parameters={
            "promoted_product_id": _required(PRODUCT),
            "start_week": WEEK,
            "end_week": WEEK,
            "top_n": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Number of products to return."},
        },
        callable_=MCPTools.get_top_impacted_products,
    ),
    "get_baseline_prediction": _tool(
        name="get_baseline_prediction",
        description=(
            "Predict baseline sales (counterfactual demand without promotion) "
            "for a product over a week range, aggregated over the model stores. "
            "Returns baseline_qty, confidence and data_sufficient."
        ),
        parameters={
            "product_id": _required(PRODUCT),
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.get_baseline_prediction,
    ),
    "calculate_incremental_sales": _tool(
        name="calculate_incremental_sales",
        description=(
            "Calculate incremental sales for a campaign/product: "
            "incremental = actual sales - baseline predicted sales. Alias of "
            "get_incremental_sales."
        ),
        parameters={
            "campaign_id": _required(CAMPAIGN),
            "product_id": PRODUCT,
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.calculate_incremental_sales,
    ),
    "detect_cannibalization": _tool(
        name="detect_cannibalization",
        description=(
            "Detect cannibalization for a promoted product: affected products, "
            "estimated lost units, cannibalization percentage and signal "
            "(strong/weak/none). Alias of get_cannibalization_effect."
        ),
        parameters={
            "promoted_product_id": _required(PRODUCT),
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.detect_cannibalization,
    ),
    "find_similar_products": _tool(
        name="find_similar_products",
        description=(
            "Find the most similar products to a product using the product "
            "embedding model. Returns similar products with similarity scores "
            "and substitutes (cannibalization candidates)."
        ),
        parameters={
            "product_id": _required(PRODUCT),
            "top_k": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Number of similar products to return."},
        },
        callable_=MCPTools.find_similar_products,
    ),
    "calculate_financial_metrics": _tool(
        name="calculate_financial_metrics",
        description=(
            "Calculate financial metrics for a product/campaign: incremental "
            "revenue, incremental profit, promotion cost, ROI and "
            "cannibalized units. Provide margin as a fraction (default 0.25)."
        ),
        parameters={
            "product_id": _required(PRODUCT),
            "campaign_id": CAMPAIGN,
            "start_week": WEEK,
            "end_week": WEEK,
            "margin": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Gross margin fraction (default 0.25)."},
        },
        callable_=MCPTools.calculate_financial_metrics,
    ),
    "rank_campaigns": _tool(
        name="rank_campaigns",
        description=(
            "Rank campaigns by normalized multi-objective scores. Objectives: "
            "profit, sales, efficiency. Returns ranking with per-objective scores."
        ),
        parameters={
            "campaign_ids": {
                "type": "array",
                "items": CAMPAIGN,
                "minItems": 1,
                "description": "Campaign IDs to rank.",
            },
            "objective": {"type": "string", "enum": ["profit", "sales", "efficiency"], "description": "Ranking objective."},
            "normalization": {"type": "string", "enum": ["min_max", "z_score", "percentile"], "description": "Normalization method."},
        },
        callable_=MCPTools.rank_campaigns,
    ),
    "compare_campaigns": _tool(
        name="compare_campaigns",
        description=(
            "Compare campaigns: ranked scores plus strengths and weaknesses "
            "for each campaign under the chosen objective."
        ),
        parameters={
            "campaign_ids": {
                "type": "array",
                "items": CAMPAIGN,
                "minItems": 2,
                "description": "Campaign IDs to compare.",
            },
            "objective": {"type": "string", "enum": ["profit", "sales", "efficiency"], "description": "Ranking objective."},
            "normalization": {"type": "string", "enum": ["min_max", "z_score", "percentile"], "description": "Normalization method."},
        },
        callable_=MCPTools.compare_campaigns,
    ),
    "recommend_campaigns": _tool(
        name="recommend_campaigns",
        description=(
            "Recommend which products to promote under a strategy. Objective "
            "can be SALES_GROWTH, PROFIT_OPTIMIZATION or SAFE_PROMOTION. "
            "Returns ranked products with expected sales, profit and explanation."
        ),
        parameters={
            "products": {
                "type": "array",
                "items": PRODUCT,
                "minItems": 1,
                "description": "Candidate product IDs.",
            },
            "objective": {"type": "string", "enum": ["SALES_GROWTH", "PROFIT_OPTIMIZATION", "SAFE_PROMOTION"], "description": "Strategy."},
            "budget": {"type": "number", "minimum": 0.0, "description": "Optional promotion budget cap."},
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.recommend_campaigns,
    ),
    "simulate_campaign_strategy": _tool(
        name="simulate_campaign_strategy",
        description=(
            "Simulate a hypothetical discount campaign for a product: expected "
            "demand, incremental sales, ROI and risks. discount_percentage is "
            "the discount depth and weeks is the campaign duration."
        ),
        parameters={
            "product_id": _required(PRODUCT),
            "discount_percentage": _required({"type": "number", "minimum": 0.1, "maximum": 90.0, "description": "Discount depth percent."}),
            "weeks": _required({"type": "integer", "minimum": 1, "maximum": 52, "description": "Campaign duration in weeks."}),
            "start_week": WEEK,
            "elasticity": {"type": "number", "minimum": 0.0, "description": "Optional demand elasticity override."},
        },
        callable_=MCPTools.simulate_campaign_strategy,
    ),
    "generate_promotion_report": _tool(
        name="generate_promotion_report",
        description=(
            "Generate an executive promotion report for a product: summary, "
            "metrics, recommendations and risks."
        ),
        parameters={
            "product_id": _required(PRODUCT),
            "campaign_id": CAMPAIGN,
            "start_week": WEEK,
            "end_week": WEEK,
        },
        callable_=MCPTools.generate_promotion_report,
    ),
    "analyze_incremental_sales_attribution": _tool(
        name="analyze_incremental_sales_attribution",
        description=(
            "Explain WHY a promotion generated incremental sales "
            "(actual - baseline) by attributing the impact to business reasons "
            "(promotion_effect, cannibalization, demand_expansion, "
            "basket_expansion, price_response, unknown) with confidence scores "
            "and explicit uncertainty statements. Call this tool whenever the "
            "user asks to score, analyze, evaluate or explain a promotion and "
            "wants reasons or drivers of the increase, or to know whether the "
            "growth was natural/organic demand or the promotion itself (for "
            "example 'why did sales increase', 'was the growth natural or due "
            "to the promotion', 'score P981760-97-97'). promotion_id uses "
            "format P{product_id}-{start_week}-{end_week}, so P981760-97-97 "
            "means product 981760 in weeks 97-97; you may also pass product_id "
            "with start_week and end_week. Returns ALL reasons plus a Top-K "
            "executive summary."
        ),
        parameters={
            "promotion_id": _required({"type": "string", "description": "Platform promotion identifier (e.g. P1234-80-84)."}),
            "product_id": _required(PRODUCT),
            "start_week": _required(WEEK),
            "end_week": _required(WEEK),
            "top_k": {"type": "integer", "minimum": 1, "maximum": 6, "description": "Reasons in the executive summary; the complete report always includes all reasons."},
        },
        callable_=MCPTools.analyze_incremental_sales_attribution,
    ),
}

ANALYSIS_TOOLS: list[dict[str, Any]] = [
    tool["openai_schema"] for tool in TOOL_DEFINITIONS.values()
]
ANALYSIS_TOOL_NAMES: set[str] = set(TOOL_DEFINITIONS)
