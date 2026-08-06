"""End-to-end analytics, MCP and LLM-agent test scenarios."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import llm_agent
from app.services.analytics import AnalyticsService
from app.main import app
from app.agent.llm_agent import PromotionDataAgent
from app.mcp.tools import MCPTools, ANALYSIS_TOOL_NAMES
from app.mcp.server import handle_request

# Shared service so heavy inputs (transactions, baseline panel) load once.
SERVICE = AnalyticsService()


# Scenario 1: successful campaign analysis
def test_successful_campaign_analysis():
    effect = SERVICE.campaign_effect(15, product_id=1005637)
    assert effect["data_sufficient"] is True
    assert effect["baseline_sales"] > 0
    assert effect["actual_sales"] > 0
    assert effect["incremental_sales"] == pytest.approx(
        effect["actual_sales"] - effect["baseline_sales"], abs=0.01
    )
    assert effect["uplift_percentage"] is not None
    assert "Promotion increased expected sales" in effect["effectiveness_summary"]
    assert effect["weekly"], "weekly actual vs baseline series must be present"


# Scenario 2: cannibalization analysis with known similar products
def test_cannibalization_analysis():
    result = SERVICE.cannibalization_effect(6534178, start_week=98, end_week=101)
    assert result["signal"] == "strong"
    assert result["total_lost_quantity"] > 0
    assert result["affected_products"], "affected products must be listed"
    assert result["top_impacted_products"]
    assert result["affected_products"][0]["lost_quantity"] >= result["affected_products"][-1]["lost_quantity"]
    assert "Product 6534178 promotion caused estimated loss" in result["summary"]


# Scenario 3: low-data product -> uncertainty, no fabricated results
def test_low_data_product_explains_uncertainty():
    low_data_product = SERVICE.get_campaign(24)["products"][0]  # not in the baseline panel
    effect = SERVICE.campaign_effect(24, product_id=low_data_product)
    assert effect["data_sufficient"] is False
    assert effect["confidence"] == "low"
    assert effect["incremental_sales"] == 0.0
    assert "Insufficient historical data" in effect["message"]
    baseline = SERVICE.product_baseline(low_data_product, 95, 101)
    assert baseline["data_sufficient"] is False
    assert baseline["baseline_qty"] == 0.0


# Scenario 4: no cannibalization case
def test_no_cannibalization_signal():
    result = SERVICE.cannibalization_effect(1004906, start_week=98, end_week=101)
    assert result["signal"] == "none"
    assert result["total_lost_quantity"] == 0.0
    assert result["affected_products"] == []
    assert "No strong cannibalization signal" in result["message"]
    assert result["summary"] == result["message"]


# MCP tools: registry, dispatch and stdio protocol
def test_mcp_tools_cover_required_methods():
    required = {
        "get_campaign_effect",
        "get_product_baseline",
        "get_incremental_sales",
        "get_cannibalization_effect",
        "get_top_impacted_products",
    }
    assert required <= ANALYSIS_TOOL_NAMES
    tools = MCPTools(SERVICE)
    assert tools.get_incremental_sales(15, product_id=1005637)["incremental_sales"] > 0
    assert tools.get_product_baseline(1005637)["data_sufficient"] is True
    # top_n caps at what's available: this product has exactly one affected
    # product with evidence, so requesting 2 still returns 1.
    top = tools.get_top_impacted_products(6534178, 98, 101, top_n=2)
    assert len(top["top_impacted_products"]) == 1


def test_mcp_server_stdio_protocol():
    listing = handle_request({"method": "tools/list", "params": {}})
    names = {tool["name"] for tool in listing["tools"]}
    assert "get_campaign_effect" in names
    response = handle_request(
        {
            "method": "tools/call",
            "params": {
                "name": "get_cannibalization_effect",
                "arguments": {"promoted_product_id": 6534178, "start_week": 98, "end_week": 101},
            },
        }
    )
    assert response["result"]["signal"] == "strong"
    unknown = handle_request(
        {
            "method": "tools/call",
            "params": {"name": "missing_tool", "arguments": {}},
        }
    )
    assert unknown["error"].startswith("Unknown tool:")


# API endpoints: Frontend -> Backend -> MCP -> Data/Models
def test_api_promotion_and_cannibalization_endpoints():
    with TestClient(app) as client:
        campaigns = client.get("/campaigns").json()["campaigns"]
        assert any(c["campaign_id"] == 15 for c in campaigns)

        effect = client.post(
            "/analytics/promotion-effect",
            json={"campaign_id": 15, "product_id": 1005637},
        )
        assert effect.status_code == 200
        assert effect.json()["incremental_sales"] > 0

        cannibal = client.post(
            "/analytics/cannibalization-effect",
            json={"promoted_product_id": 6534178, "start_week": 98, "end_week": 101},
        )
        assert cannibal.status_code == 200
        assert cannibal.json()["signal"] == "strong"

        promoted = client.get("/analytics/promoted-products").json()["products"]
        assert promoted and promoted[0]["product_id"] == 6534178


# LLM agent uses MCP analysis tools instead of raw SQL for campaign questions
def test_llm_agent_calls_mcp_tool_for_campaign_effect(monkeypatch):
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                function = SimpleNamespace(
                    name="get_campaign_effect",
                    arguments=json.dumps({"campaign_id": 15, "product_id": 1005637}),
                )
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(id="call-1", function=function)],
                )
            else:
                message = SimpleNamespace(
                    content="Campaign 15 generated 13 units of incremental sales.",
                    tool_calls=[],
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm_agent, "OpenAI", FakeOpenAI)
    agent = PromotionDataAgent()
    try:
        result = agent.ask(
            "Analyze the effect of campaign 15",
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
        )
    finally:
        agent.catalog.close()

    assert result.answer == "Campaign 15 generated 13 units of incremental sales."
    assert result.traces[0].tool == "get_campaign_effect"
    assert result.traces[0].status == "success"
    assert result.traces[0].rows[0]["incremental_sales"] > 0
