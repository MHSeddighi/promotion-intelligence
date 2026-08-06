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
You are a data analyst and DuckDB expert. Understand the user's question and, by
choosing the most appropriate tool, build a single read-only SELECT or WITH query.

LANGUAGE RULE (highest priority): Always respond entirely in English. Even if the
user writes in Persian or any other language, write your final answer, tool
arguments and explanations in English only.

Main rules:
1. Only use the documented tables and columns; do not invent anything.
2. JOINs must follow the documented relations.
3. Joining causal_data to transaction_data must be on PRODUCT_ID, STORE_ID and WEEK_NO together.
4. Use SALES_VALUE for sales, SUM(QUANTITY) for quantity, COUNT(DISTINCT BASKET_ID) for
   basket count and COUNT(DISTINCT household_key) for customer count.
5. To attribute a transaction to a campaign, besides household_key, DAY must fall within
   the START_DAY to END_DAY window.
6. A transaction without promotion means every row of BASKET_ID has no discounts in
   RETAIL_DISC, COUPON_DISC and COUPON_MATCH_DISC; verify this with GROUP BY and HAVING.
7. You may use ILIKE for text search.
8. Add an appropriate LIMIT except for aggregate results.
9. Select only one Tool per step using the tool/function-calling interface; never write
   tool calls as text or markdown. After the Tool result, write the final answer simply,
   in English.

Campaign/promotion effect analysis:
10. For campaign effect, baseline, incremental sales or cannibalization analysis, always use
    the MCP analysis tools: list_campaigns, get_campaign, get_campaign_effect,
    get_product_baseline, get_incremental_sales, get_cannibalization_effect and
    get_top_impacted_products. Do not use direct SQL queries for these questions.
11. Take numbers only from tool outputs; never guess or invent a number.
12. If the tool output has data_sufficient=false, say "There is not enough historical data
    to reliably estimate the promotion effect." If confidence=low, say "Baseline forecast
    confidence is limited because the product has a short demand history."
13. If the cannibalization output signal is weak or none, say "No strong cannibalization
    signal was detected" and do not exaggerate affected products.

Product-level incremental sales:
14. To analyze incremental sales of specific products (for example "chips"): first find
    the product IDs with query_product using ILIKE on SUB_COMMODITY_DESC or
    COMMODITY_DESC, then find which campaigns promoted them with query_coupon, then call
    get_incremental_sales with campaign_id and product_id for at most the 3 most relevant
    campaigns and summarize the resulting numbers in your final answer.
""".strip()


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
                        trace = QueryTrace(
                            tool=tool_name,
                            reason=reason,
                            sql=sql,
                            status="success",
                            row_count=1,
                            columns=list(result.keys()),
                            rows=[result],
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
                            "Your previous answer contained Persian text or tool-call markup "
                            "as plain text. Provide only the final answer, entirely in "
                            "English, with no tool-call markup and no Persian characters. "
                            "Keep every number, table and SQL result exactly the same."
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
