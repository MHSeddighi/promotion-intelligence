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

from app.services.analytics import AnalyticsService


class MCPTools:
    """Read-only analytics tools exposed to LLM agents and the MCP server."""

    def __init__(self, service: AnalyticsService | None = None) -> None:
        self.service = service or AnalyticsService()

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
}

ANALYSIS_TOOLS: list[dict[str, Any]] = [
    tool["openai_schema"] for tool in TOOL_DEFINITIONS.values()
]
ANALYSIS_TOOL_NAMES: set[str] = set(TOOL_DEFINITIONS)
