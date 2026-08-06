import copy
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.baseline import BaselinePredictor
from app.agent import llm_agent
from app.agent.llm_agent import (
    DatabaseCatalog,
    PromotionDataAgent,
    validate_sql,
)


def test_baseline_predictor_reproduces_sample():
    predictor = BaselinePredictor()
    result = predictor.predict(predictor.load_sample())
    assert result.shape == (1, 3)
    assert result.iloc[0]["probability_positive"] >= 0.0
    assert result.iloc[0]["predicted_positive_quantity"] >= 0.0
    assert result.iloc[0]["baseline_prediction"] >= 0.0


def test_unified_api_health_sample_and_batch():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["models_loaded"] is True
        assert health.json()["database_ready"] is True

        sample = client.get("/baseline/sample").json()["records"][0]
        response = client.post(
            "/baseline/predict",
            json={"records": [sample, copy.deepcopy(sample)]},
        )
        assert response.status_code == 200
        assert response.json()["count"] == 2


def test_sql_validation_and_small_query():
    sql = validate_sql(
        "SELECT CAMPAIGN, DESCRIPTION FROM campaign_desc ORDER BY CAMPAIGN LIMIT 2",
        "campaign_desc",
    )
    catalog = DatabaseCatalog()
    try:
        result = catalog.execute(sql, "campaign_desc")
    finally:
        catalog.close()
    assert len(result) == 2


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE campaign_desc",
        "SELECT * FROM read_csv_auto('secret.csv') JOIN campaign_desc USING (CAMPAIGN)",
        "SELECT * FROM campaign_desc; SELECT 1",
    ],
)
def test_sql_validation_rejects_unsafe_queries(sql):
    with pytest.raises(ValueError):
        validate_sql(sql, "campaign_desc")


def test_assistant_endpoint_requires_server_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/assistant/ask",
            json={"question": "How many campaigns are there?"},
        )
        assert response.status_code == 503
        assert "OPENAI_API_KEY" in response.json()["detail"]


def test_llm_tool_call_flow_without_external_request(monkeypatch):
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                function = SimpleNamespace(
                    name="query_campaign_desc",
                    arguments=json.dumps(
                        {
                            "sql": "SELECT COUNT(*) AS campaign_count FROM campaign_desc",
                            "reason": "Counting campaigns",
                        },
                        ensure_ascii=False,
                    ),
                )
                message = SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(id="call-1", function=function)],
                )
            else:
                message = SimpleNamespace(
                    content="Campaign count computed successfully.",
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
            "How many campaigns are there?",
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
        )
    finally:
        agent.catalog.close()

    assert result.answer == "Campaign count computed successfully."
    assert len(result.traces) == 1
    assert result.traces[0].status == "success"
    assert result.traces[0].rows[0]["campaign_count"] > 0


def test_text_tool_call_markup_detection_and_cleanup():
    from app.agent.llm_agent import (
        _strip_tool_call_markup,
        contains_fake_tool_call,
    )

    assert contains_fake_tool_call("<tool_calls>")
    assert contains_fake_tool_call("<\ufffdtool_calls>")
    assert contains_fake_tool_call("<\u200btool_calls>")
    assert contains_fake_tool_call("<|\uff5c\uff5cDSML\uff5c\uff5c|tool_calls>")
    assert contains_fake_tool_call("<invoke name=\"query_coupon\">")
    assert not contains_fake_tool_call("Total incremental sales increased by 10%.")

    cleaned = _strip_tool_call_markup(
        "Intro.\n<tool_calls>\n<invoke name=\"get_incremental_sales\">\n"
        "<parameter name=\"campaign_id\">8</parameter>\n</invoke>\n</tool_calls>\nOutro."
    )
    assert "Intro." in cleaned
    assert "Outro." in cleaned
    assert not contains_fake_tool_call(cleaned)


def test_trace_rows_nested_detection():
    from app.agent.llm_agent import trace_rows_contain_nested

    assert trace_rows_contain_nested([{"campaign_id": 1, "weekly": []}])
    assert trace_rows_contain_nested([{"campaign_id": 1, "weekly": [{"week": 1}]}])
    assert not trace_rows_contain_nested([{"campaign_id": 1, "actual": 10.0}])
    assert not trace_rows_contain_nested([])
