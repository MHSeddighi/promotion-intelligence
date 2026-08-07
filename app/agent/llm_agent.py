"""Safe DuckDB tool-calling agent extracted from notebooks/llm.ipynb."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import duckdb
import httpx
import pandas as pd
from openai import OpenAI

from app.config import DATA_DIR, DEFAULT_MAX_QUERY_ROWS
from app.mcp.tools import ANALYSIS_TOOL_NAMES, ANALYSIS_TOOLS, MCPTools


TABLE_INFO: dict[str, dict[str, Any]] = {
    "campaign_desc": {
        "columns": ["DESCRIPTION", "CAMPAIGN", "START_DAY", "END_DAY"],
        "description": "Main campaign information: campaign type and its start/end window.",
        "relations": "CAMPAIGN links to campaign_table, coupon and coupon_redempt.",
    },
    "campaign_table": {
        "columns": ["DESCRIPTION", "household_key", "CAMPAIGN"],
        "description": "Household membership or targeting in campaigns.",
        "relations": "Links via CAMPAIGN to campaign_desc, and via household_key to transactions and demographics.",
    },
    "causal_data": {
        "columns": ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"],
        "description": "Promotional display and mailer activities for product, store and week.",
        "relations": "Joining to transaction_data must use PRODUCT_ID, STORE_ID and WEEK_NO together.",
    },
    "coupon": {
        "columns": ["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"],
        "description": "Bridge between coupon, product and campaign.",
        "relations": "Links to coupon_redempt, product and campaign_desc.",
    },
    "coupon_redempt": {
        "columns": ["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"],
        "description": "Record of a coupon redemption by a household on a single day.",
        "relations": "Links via COUPON_UPC and CAMPAIGN to coupon, and via household_key to households.",
    },
    "hh_demographic": {
        "columns": [
            "AGE_DESC",
            "MARITAL_STATUS_CODE",
            "INCOME_DESC",
            "HOMEOWNER_DESC",
            "HH_COMP_DESC",
            "HOUSEHOLD_SIZE_DESC",
            "KID_CATEGORY_DESC",
            "household_key",
        ],
        "description": "Household demographic information.",
        "relations": "household_key links to transaction_data, campaign_table and coupon_redempt.",
    },
    "product": {
        "columns": [
            "PRODUCT_ID",
            "MANUFACTURER",
            "DEPARTMENT",
            "BRAND",
            "COMMODITY_DESC",
            "SUB_COMMODITY_DESC",
            "CURR_SIZE_OF_PRODUCT",
        ],
        "description": "Descriptive product attributes: brand, category and size.",
        "relations": "PRODUCT_ID links to transaction_data, coupon and causal_data.",
    },
    "transaction_data": {
        "columns": [
            "household_key",
            "BASKET_ID",
            "DAY",
            "PRODUCT_ID",
            "QUANTITY",
            "SALES_VALUE",
            "STORE_ID",
            "RETAIL_DISC",
            "TRANS_TIME",
            "WEEK_NO",
            "COUPON_DISC",
            "COUPON_MATCH_DISC",
        ],
        "description": "Transaction line items: quantity, sales, store, week and discounts.",
        "relations": "Links via PRODUCT_ID to product, and via household_key to households and campaigns.",
    },
}


FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|COPY|"
    r"ATTACH|DETACH|INSTALL|LOAD|PRAGMA|CALL|EXPORT|IMPORT|SET|RESET|"
    r"VACUUM|CHECKPOINT)\b",
    flags=re.IGNORECASE,
)
EXTERNAL_ACCESS_PATTERN = re.compile(
    r"\b(read_csv(?:_auto)?|read_json(?:_auto)?|read_parquet|glob|httpfs|sqlite_scan|postgres_scan|"
    r"mysql_scan|delta_scan|iceberg_scan|query_table|query)\s*\(",
    flags=re.IGNORECASE,
)
FAKE_TOOL_CALL_PATTERN = re.compile(
    r"[<]\s*\W*tool_calls\s*>|[<]\s*\W*invoke\s+name=|[<]\s*\W*parameter\s+name=|tool_calls\s*[:=]\s*\[",
    flags=re.IGNORECASE,
)


SYSTEM_PROMPT = """
You are a data analyst and DuckDB expert. Answer user questions by selecting the correct tool. Use read-only SELECT/WITH SQL only when raw data queries are required.

LANGUAGE:
- Always respond in English.
- Explain results clearly for technical and non-technical users.

==================================================
GENERAL RULES
==================================================

1. Never calculate promotion intelligence metrics manually.
For:
- baseline
- incremental sales
- campaign effect
- cannibalization
- similarity
- financial metrics
- campaign ranking/comparison
- recommendations
- attribution

always use the dedicated analysis tools.

2. Use SQL only for raw data questions not covered by tools.

3. Never invent tables, columns, joins, or numbers.

4. Follow documented schema relationships.

5. Required metrics:
- Sales = SUM(SALES_VALUE)
- Quantity = SUM(QUANTITY)
- Basket count = COUNT(DISTINCT BASKET_ID)
- Customers = COUNT(DISTINCT household_key)

6. Campaign transaction attribution requires:
household_key AND DAY between START_DAY and END_DAY.

7. Non-promotion basket:
A basket is non-promoted only if all rows have:
RETAIL_DISC = 0,
COUPON_DISC = 0,
COUPON_MATCH_DISC = 0.
Verify using GROUP BY/HAVING.

8. Use LIMIT for non-aggregate SQL.

9. Use one tool per step only. Never write tool calls as text.

==================================================
PROMOTION ANALYSIS TOOLS
==================================================

Use MCP tools for promotion questions:

- list_campaigns
- get_campaign
- get_campaign_effect
- get_product_baseline
- get_incremental_sales
- get_cannibalization_effect
- detect_cannibalization
- get_top_impacted_products
- calculate_financial_metrics
- compare_campaigns
- rank_campaigns
- recommend_campaigns
- simulate_campaign_strategy
- generate_promotion_report
- analyze_incremental_sales_attribution


Never replace these with SQL.


==================================================
RESULT HANDLING
==================================================

- Use only values returned by tools.
- Never guess metrics.

If:
data_sufficient=false:
Say:
"There is not enough historical data to reliably estimate the promotion effect."

If:
confidence=low:
Say:
"Baseline forecast confidence is limited because the product has a short demand history."

If cannibalization is weak/none:
Say:
"No strong cannibalization signal was detected."


==================================================
PRODUCT INCREMENTAL SALES
==================================================

For product-level incremental sales questions:

1. Use query_product with ILIKE to find product IDs.
2. Use query_coupon to find related campaigns.
3. Call get_incremental_sales for maximum 3 relevant campaigns.
4. Summarize returned results only.


==================================================
CAMPAIGN COMPARISON AND RANKING
==================================================

For campaign scoring, ranking, or comparison:

Do not judge using one metric.

Consider together:

- actual sales
- baseline sales
- incremental sales
- uplift percentage
- cannibalization
- ROI
- incremental profit
- attribution reasons

Use comparison/ranking tools when appropriate.

Explain trade-offs:

Examples:
- "High incremental sales but low ROI because discount cost was high."
- "Positive uplift but strong cannibalization reduces true value."

If results look abnormal, mention possible causes:
- missing stock data
- out-of-stock periods
- date/week mapping problems


==================================================
INCREMENTAL SALES ATTRIBUTION
==================================================

Use analyze_incremental_sales_attribution when the user asks:

- why sales increased
- what drove promotion success
- promotion evaluation with reasons
- natural growth vs promotion effect
- drivers behind incremental sales


Call it before other promotion effect tools.

This tool explains:

incremental_sales =
actual_sales - baseline_prediction


It separates effects into:

1. promotion_effect
   - impact from promotion mechanics

2. cannibalization
   - negative impact from customers switching from similar products

3. demand_expansion
   - natural/category demand growth

4. basket_expansion
   - complementary purchases and larger baskets

5. price_response
   - customer response to price changes

6. unknown
   - unexplained impact caused by missing information or weak evidence


Always:

- report all returned reason effects
- explain positive and negative contributions
- identify strongest drivers
- include confidence and uncertainty


Example:

"The promotion generated +1000 incremental units. Promotion effect contributed +500 units, demand expansion contributed +300 units, and cannibalization reduced impact by -150 units."


Do not assume all growth is caused by promotion.

Separate:
- promotion-driven growth
- natural demand growth
- substitution effects


==================================================
ATTRIBUTION INPUT FORMAT
==================================================

Promotion ID format:

P<product_id>-<start_week>-<end_week>

Example:

P981760-97-97

means:

product_id = 981760
start_week = 97
end_week = 97


Call attribution only when a specific promotion period exists.

Do not call it for:
- generic campaign lists
- similarity questions
- baseline-only questions
- raw SQL questions


If attribution returns only available_example_reports:

- state that full attribution is unavailable
- explain only from returned signals
- do not invent reasons


==================================================
CANNIBALIZATION BETWEEN TWO PRODUCTS
==================================================

When the user asks whether one product cannibalizes another (for example
"is product B losing sales to promoted product A?" or "show cannibalization
between product 1005637 and its substitutes"):

1. Call get_cannibalization_effect and/or get_top_impacted_products for the
   promoted product over the week range.
2. Check whether the other product appears among the affected products.
3. Present the result as a pair:

   promoted product A -> affected product B
   - lost units
   - cannibalization percentage
   - signal strength (strong/weak/none)

4. If the other product is not in the affected list, say no cannibalization was
   detected between them.
5. If the user asks WHY the promotion cannibalized sales, also call
   analyze_incremental_sales_attribution and use its cannibalization reason
   (evidence such as substitute similarity, brand overlap, sub-category overlap).
6. Use find_similar_products to suggest substitute pairs when the user names
   only one product.
7. Never exaggerate: if the signal is weak or none, say "No strong
   cannibalization signal was detected."


==================================================
PRICE CHANGE SITUATIONS
==================================================

When the user asks why price changes affected sales, or wants the different
price situations explained (deep vs shallow discounts, price gaps, elastic vs
inelastic demand, discount-response strength):

1. Call analyze_incremental_sales_attribution and use its price_response reason
   (impact units/percentage, confidence, evidence).
2. Combine it with the period's price features when available:
   - discount_pct (discount depth)
   - price_during and price_before (price level)
   - price_gap_pct (gap to regular price)
   - hist_price_elasticity (elastic vs inelastic demand)
   - hist_discount_response (historical response to discounts)
3. Enumerate each price situation separately with direction and magnitude.
   Examples:
   - "deep discount (-35%) with elastic demand (elasticity -1.1) added +X units"
   - "small price gap left demand nearly unchanged"
   - "historically strong discount response amplified the effect"
   - "price below regular price (-20%) drove a positive price response"
4. Never invent price reasons: report only evidence present in the tool output
   and always include confidence and uncertainty_statements.


==================================================
SHOW AND TELL FROM DATA
==================================================

Every number in your answer must be traceable to a tool result. Show the
evidence and tell the user it was extracted from the data:

1. State which tool/dataset produced each number, for example "extracted from
   get_cannibalization_effect (cannibalization model over weekly product
   sales)" or "from analyze_incremental_sales_attribution (SHAP attribution
   over the promotion-period features)".
2. Quote the exact returned values: product IDs, weeks, units, percentages,
   confidence and signal strength.
3. For cannibalization pairs present them as:

   "Promoted product A -> affected product B: lost X units (Y% of expected
   sales), signal: strong/weak/none"

   and name the data source for the pair.
4. For price situations present each situation separately with its evidence
   (discount depth, price gap, elasticity, discount response) and the data
   source for each number.
5. Never present a number without its source. If a number is not in the tool
   output, do not include it.


==================================================
FINAL RESPONSE STYLE
==================================================

Always provide:
- clear conclusion
- important numbers from tools
- explanation of drivers
- uncertainty/limitations when relevant

Never overstate certainty when stock, calendar, or external market data is missing. 
"""


@dataclass
class QueryTrace:
    tool: str
    reason: str
    sql: str
    status: str
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class AgentResult:
    answer: str
    traces: list[QueryTrace]


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def normalize_mcp_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten common MCP result shapes into a list of display rows."""
    if len(result) == 1:
        value = next(iter(result.values()))
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return value
    return [result]


def trace_rows_contain_nested(rows: list[dict[str, Any]]) -> bool:
    """True when trace rows hold dict/list values a dataframe cannot display."""
    if not rows:
        return False
    return any(
        isinstance(value, (dict, list))
        for value in pd.DataFrame(rows).iloc[0].tolist()
    )


def contains_persian(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" for char in text)


def contains_fake_tool_call(text: str) -> bool:
    return bool(FAKE_TOOL_CALL_PATTERN.search(_strip_dsml_artifacts(text)))


def _strip_dsml_artifacts(text: str) -> str:
    """Remove model-specific tool-call wrapper characters (DSML blocks and junk)."""
    text = text.replace("\ufffd", "")
    return re.sub(r"[|\uFF5C\s]*DSML[|\uFF5C\s]*", "", text, flags=re.IGNORECASE)


def _strip_tool_call_markup(text: str) -> str:
    """Remove any leftover text-form tool-call markup from a final answer."""
    cleaned = _strip_dsml_artifacts(text)
    cleaned = re.sub(
        r"[<]\s*\W*tool_calls\s*>.*?[<]\s*/\s*\W*tool_calls\s*>",
        " ",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"[<]\s*\W*invoke\b.*?(?:[<]\s*/\s*\W*invoke\s*>|$)",
        " ",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"[<]\s*\W*parameter\b.*?(?:[<]\s*/\s*\W*parameter\s*>|$)",
        " ",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def remove_sql_comments(sql: str) -> str:
    without_line_comments = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", without_line_comments, flags=re.DOTALL)


def validate_sql(sql: str, expected_table: str) -> str:
    if expected_table not in TABLE_INFO:
        raise ValueError(f"Unknown tool table: {expected_table}")
    clean_sql = remove_sql_comments(sql).strip().rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", clean_sql, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    if FORBIDDEN_SQL_PATTERN.search(clean_sql):
        raise ValueError("The query contains a forbidden SQL command.")
    if EXTERNAL_ACCESS_PATTERN.search(clean_sql):
        raise ValueError("External file, URL, extension, or dynamic query access is forbidden.")
    if ";" in clean_sql:
        raise ValueError("Multiple SQL statements are not allowed.")
    table_pattern = re.compile(
        rf'\b(?:FROM|JOIN)\s+["`]?{re.escape(expected_table)}["`]?\b',
        flags=re.IGNORECASE,
    )
    if not table_pattern.search(clean_sql):
        raise ValueError(
            f"The selected tool must query its primary table {expected_table!r}."
        )
    return clean_sql


class DatabaseCatalog:
    """Thread-safe DuckDB catalog backed by read-only CSV views."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.connection = duckdb.connect(database=":memory:")
        self.lock = RLock()
        self._register_views()

    def _register_views(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Promotion data directory was not found: {self.data_dir}")
        missing = [
            self.data_dir / f"{table}.csv"
            for table in TABLE_INFO
            if not (self.data_dir / f"{table}.csv").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Required promotion CSV files are missing:\n"
                + "\n".join(f"- {path}" for path in missing)
            )

        with self.lock:
            for table in TABLE_INFO:
                safe_path = (
                    (self.data_dir / f"{table}.csv")
                    .resolve()
                    .as_posix()
                    .replace("'", "''")
                )
                self.connection.execute(
                    f'''CREATE OR REPLACE VIEW "{table}" AS
                        SELECT * FROM read_csv_auto(
                            '{safe_path}', header=TRUE, sample_size=100000
                        )'''
                )

    @property
    def table_names(self) -> list[str]:
        return list(TABLE_INFO)

    def execute(self, sql: str, expected_table: str, max_rows: int = DEFAULT_MAX_QUERY_ROWS) -> pd.DataFrame:
        if not 1 <= max_rows <= 2_000:
            raise ValueError("max_rows must be between 1 and 2000.")
        safe_sql = validate_sql(sql, expected_table)
        limited_sql = f"SELECT * FROM ({safe_sql}) AS query_result LIMIT {int(max_rows)}"
        try:
            with self.lock:
                return self.connection.execute(limited_sql).df()
        except Exception as exc:
            raise RuntimeError(f"DuckDB query failed: {exc}") from exc

    def close(self) -> None:
        with self.lock:
            self.connection.close()


def make_table_tool(table_name: str) -> dict[str, Any]:
    info = TABLE_INFO[table_name]
    description = (
        f"Analysis tool on the main table {table_name}. Use: {info['description']} "
        f"Columns: {', '.join(info['columns'])}. Relations: {info['relations']} "
        "Only generate SELECT or WITH queries; JOINs with related tables are allowed."
    )
    return {
        "type": "function",
        "function": {
            "name": f"query_{table_name}",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": f"A read-only DuckDB query using {table_name} in FROM or JOIN.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "A short English reason for choosing the table and query.",
                    },
                },
                "required": ["sql", "reason"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


TOOLS = [make_table_tool(table_name) for table_name in TABLE_INFO]


class PromotionDataAgent:
    def __init__(
        self,
        catalog: DatabaseCatalog | None = None,
        mcp_tools: MCPTools | None = None,
    ) -> None:
        self.catalog = catalog or DatabaseCatalog()
        self.mcp_tools = mcp_tools or MCPTools()
        self.tools = [*ANALYSIS_TOOLS, *TOOLS]

    def ask(
        self,
        question: str,
        *,
        api_key: str,
        base_url: str,
        model: str,
        history: list[dict[str, str]] | None = None,
        max_steps: int = 8,
    ) -> AgentResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if len(question) > 4_000:
            raise ValueError("Question is too long; maximum length is 4000 characters.")
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required.")
        if not 1 <= max_steps <= 10:
            raise ValueError("max_steps must be between 1 and 10.")

        history = history or []
        if len(history) > 20:
            history = history[-20:]
        validated_history: list[dict[str, str]] = []
        for turn in history:
            role = turn.get("role")
            content = str(turn.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                raise ValueError("Chat history must contain non-empty user/assistant turns.")
            validated_history.append({"role": role, "content": content[:8_000]})

        client = OpenAI(
            base_url=base_url.strip(),
            api_key=api_key.strip(),
            http_client=httpx.Client(trust_env=False),
        )
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *validated_history,
            {"role": "user", "content": question},
        ]
        traces: list[QueryTrace] = []

        for step in range(max_steps):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self.tools,
                tool_choice="required" if step == 0 else "auto",
                parallel_tool_calls=False,
            )
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            tool_calls = assistant_message.tool_calls or []
            if not tool_calls:
                content = assistant_message.content or ""
                if contains_fake_tool_call(content):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You wrote a tool call as plain text. Never write tool calls "
                                "as text or markdown. Use the tool/function-calling interface "
                                "for every tool call, then answer based on the tool results."
                            ),
                        }
                    )
                    continue
                return AgentResult(
                    answer=self._finalize_answer(
                        client, model, messages, initial=content
                    ),
                    traces=traces,
                )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                sql = ""
                reason = ""
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    if tool_name in ANALYSIS_TOOL_NAMES:
                        result = self.mcp_tools.call(tool_name, arguments)
                        if "error" in result:
                            raise RuntimeError(result["error"])
                        sql = json.dumps(arguments, ensure_ascii=False)
                        reason = "MCP analysis tool"
                        rows = normalize_mcp_rows(result)
                        trace = QueryTrace(
                            tool=tool_name,
                            reason=reason,
                            sql=sql,
                            status="success",
                            row_count=len(rows),
                            columns=list(rows[0].keys()) if rows else list(result.keys()),
                            rows=rows,
                        )
                        tool_result: dict[str, Any] = {
                            "status": "success",
                            "tool": tool_name,
                            "arguments": arguments,
                            "result": result,
                        }
                    else:
                        sql = arguments["sql"]
                        reason = arguments["reason"]
                        expected_table = tool_name.removeprefix("query_")
                        result_frame = self.catalog.execute(sql, expected_table)
                        rows = dataframe_to_records(result_frame)
                        trace = QueryTrace(
                            tool=tool_name,
                            reason=reason,
                            sql=sql,
                            status="success",
                            row_count=len(result_frame),
                            columns=result_frame.columns.tolist(),
                            rows=rows,
                        )
                        tool_result = {
                            "status": "success",
                            "tool": tool_name,
                            "sql": sql,
                            "row_count": len(result_frame),
                            "columns": result_frame.columns.tolist(),
                            "rows": rows,
                        }
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    trace = QueryTrace(
                        tool=tool_name,
                        reason=reason,
                        sql=sql,
                        status="error",
                        error=error,
                    )
                    tool_result = {"status": "error", "tool": tool_name, "error": error}
                traces.append(trace)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        return AgentResult(
            answer=self._finalize_answer(client, model, messages),
            traces=traces,
        )

    def _finalize_answer(
        self,
        client: OpenAI,
        model: str,
        messages: list[Any],
        initial: str | None = None,
    ) -> str:
        """Produce a clean final answer, forcing English and rejecting text tool calls."""
        answer = initial
        if answer is None or contains_persian(answer) or contains_fake_tool_call(answer):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self.tools,
                tool_choice="none",
            )
            answer = response.choices[0].message.content or "No final response was generated."

        retries = 0
        while (contains_persian(answer) or contains_fake_tool_call(answer)) and retries < 2:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer contained tool-call markup as plain text "
                            "or was incomplete. Do not run more tools or explore further. "
                            "Provide the final answer NOW, entirely in English, directly "
                            "summarizing the numbers already returned by the tools. No "
                            "tool-call markup and no Persian characters."
                        ),
                    },
                ],
                tools=self.tools,
                tool_choice="none",
            )
            answer = response.choices[0].message.content or "No final response was generated."
            retries += 1
        answer = _strip_tool_call_markup(answer)
        if not answer:
            answer = "The analysis did not produce a complete answer; please try rephrasing your question."
        return answer
