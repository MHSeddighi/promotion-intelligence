"""Tests for incremental-sales attribution (service + MCP tool)."""

import json

import pytest

from app.mcp.server import handle_request
from app.mcp.tools import MCPTools, TOOL_DEFINITIONS
from app.services.attribution import IncrementalSalesAttributionService


def test_service_returns_saved_example_report():
    svc = IncrementalSalesAttributionService()
    result = svc.analyze(promotion_id="P976998-101-101")
    assert result["promotion_id"] == "P976998-101-101"
    assert result["product_id"] == 976998
    assert result["incremental_sales"] == pytest.approx(29.07, abs=0.01)
    assert len(result["reason_attribution"]) == 6
    assert len(result["top_k_reasons"]) == 3
    assert result["executive_summary"].startswith("Promotion P976998-101-101")


def test_service_constructs_promotion_id_from_product_and_weeks():
    svc = IncrementalSalesAttributionService()
    result = svc.analyze(product_id=976998, start_week=101, end_week=101, top_k=6)
    assert result["promotion_id"] == "P976998-101-101"
    assert len(result["top_k_reasons"]) == 6


def test_top_k_is_bounded_to_six():
    svc = IncrementalSalesAttributionService()
    result = svc.analyze(promotion_id="P976998-101-101", top_k=20)
    assert len(result["top_k_reasons"]) == 6


def test_service_coverage_for_non_example_promotion():
    svc = IncrementalSalesAttributionService()
    result = svc.analyze(promotion_id="P28897-82-82")
    assert result["promotion_id"] == "P28897-82-82"
    assert result["incremental_sales"] == pytest.approx(0.922587, abs=1e-3)
    assert "message" in result
    assert result["available_example_reports"]


def test_fallback_responses_are_json_serializable():
    svc = IncrementalSalesAttributionService()
    for result in (
        svc.analyze(promotion_id="P28897-82-82"),
        svc.analyze(promotion_id="P28897-82-82", product_id=28897),
        svc.analyze(promotion_id="P99999999-50-50"),
    ):
        json.dumps(result)  # must not raise


def test_service_unknown_promotion_returns_helpful_error():
    svc = IncrementalSalesAttributionService()
    result = svc.analyze(promotion_id="P99999999-50-50")
    assert "error" in result
    assert result["valid_promotion_ids"]
    assert result["available_example_reports"]


def test_service_rejects_product_mismatch():
    svc = IncrementalSalesAttributionService()
    result = svc.analyze(promotion_id="P976998-101-101", product_id=12345)
    assert "error" in result
    assert "belongs to product" in result["error"]


def test_mcp_tool_registered_and_schema_matches_notebook():
    assert "analyze_incremental_sales_attribution" in TOOL_DEFINITIONS
    schema = TOOL_DEFINITIONS["analyze_incremental_sales_attribution"]["openai_schema"]
    params = schema["function"]["parameters"]
    assert set(params["required"]) == {
        "promotion_id",
        "product_id",
        "start_week",
        "end_week",
    }
    top_k = params["properties"]["top_k"]
    assert top_k["minimum"] == 1
    assert top_k["maximum"] == 6


def test_mcp_tool_execution():
    tools = MCPTools()
    result = tools.call(
        "analyze_incremental_sales_attribution",
        {
            "promotion_id": "P981760-97-97",
            "product_id": 981760,
            "start_week": 97,
            "end_week": 97,
        },
    )
    assert result["promotion_id"] == "P981760-97-97"
    assert result["product_id"] == 981760
    assert len(result["reason_attribution"]) == 6
    assert result["executive_summary"].startswith("Promotion P981760-97-97")


def test_tool_discovery_via_stdio_server():
    listing = handle_request({"method": "tools/list", "params": {}})
    names = {tool["name"] for tool in listing["tools"]}
    assert "analyze_incremental_sales_attribution" in names
