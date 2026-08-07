"""Tests for the copilot MCP tool registry (v2 tools)."""

import pytest

from app.mcp.server import handle_request
from app.mcp.tools import MCPTools, TOOL_DEFINITIONS

REQUIRED_TOOLS = {
    "get_baseline_prediction",
    "calculate_incremental_sales",
    "detect_cannibalization",
    "find_similar_products",
    "calculate_financial_metrics",
    "rank_campaigns",
    "compare_campaigns",
    "recommend_campaigns",
    "simulate_campaign_strategy",
    "generate_promotion_report",
}


def test_all_copilot_tools_registered():
    assert REQUIRED_TOOLS <= set(TOOL_DEFINITIONS)


def test_tool_discovery_via_stdio_server():
    listing = handle_request({"method": "tools/list", "params": {}})
    names = {tool["name"] for tool in listing["tools"]}
    assert REQUIRED_TOOLS <= names


def test_tool_openai_schemas_strict():
    tools = MCPTools()
    for name, info in TOOL_DEFINITIONS.items():
        schema = info["openai_schema"]
        assert schema["function"]["strict"] is True
        assert schema["function"]["parameters"]["additionalProperties"] is False
        for required in schema["function"]["parameters"]["required"]:
            assert required in schema["function"]["parameters"]["properties"]


def test_find_similar_products_execution():
    tools = MCPTools()
    result = tools.call(
        "find_similar_products", {"product_id": 1005637, "top_k": 3}
    )
    assert "similar_products" in result
    assert len(result["similar_products"]) == 3
    assert "substitutes" in result


def test_find_similar_products_invalid_arguments():
    tools = MCPTools()
    result = tools.call("find_similar_products", {"product_id": 99999999})
    assert "error" in result


def test_calculate_financial_metrics_execution():
    tools = MCPTools()
    result = tools.call(
        "calculate_financial_metrics",
        {"product_id": 1005637, "start_week": 89, "end_week": 101},
    )
    assert result["data_sufficient"] is True
    assert result["financial_metrics"]["incremental_units"] == pytest.approx(
        result["incremental_sales"], abs=0.02
    )


def test_rank_and_compare_campaigns_execution():
    tools = MCPTools()
    ranking = tools.call("rank_campaigns", {"campaign_ids": [15, 20]})
    assert ranking["ranking"]
    comparison = tools.call("compare_campaigns", {"campaign_ids": [15, 20]})
    assert comparison["ranking"]
    assert comparison["strengths"]
    assert comparison["weaknesses"]


def test_recommend_campaigns_execution():
    tools = MCPTools()
    result = tools.call(
        "recommend_campaigns",
        {"products": [1005637, 934427], "objective": "PROFIT_OPTIMIZATION"},
    )
    assert "recommendations" in result
    for row in result["recommendations"]:
        assert "explanation" in row and "expected_profit" in row


def test_simulate_campaign_strategy_execution_and_validation():
    tools = MCPTools()
    simulation = tools.call(
        "simulate_campaign_strategy",
        {"product_id": 1005637, "discount_percentage": 15, "weeks": 2, "start_week": 89},
    )
    assert simulation["available"] is True
    assert simulation["implied_lift_percentage"] > 0
    bad = tools.call(
        "simulate_campaign_strategy",
        {"product_id": 1005637, "discount_percentage": 120, "weeks": 2},
    )
    assert "error" in bad


def test_generate_promotion_report_execution():
    tools = MCPTools()
    report = tools.call("generate_promotion_report", {"product_id": 1005637})
    assert report["title"]
    assert report["metrics"]
    assert report["recommendations"]


def test_get_baseline_prediction_alias():
    tools = MCPTools()
    result = tools.call("get_baseline_prediction", {"product_id": 1005637})
    assert result["data_sufficient"] is True
    assert result["baseline_qty"] > 0


def test_detect_cannibalization_alias():
    tools = MCPTools()
    result = tools.call(
        "detect_cannibalization",
        {"promoted_product_id": 934427, "start_week": 98, "end_week": 101},
    )
    assert result["signal"] in {"strong", "weak", "none"}


def test_unknown_tool_dispatch_raises():
    tools = MCPTools()
    with pytest.raises(ValueError):
        tools.call("missing_tool", {})
