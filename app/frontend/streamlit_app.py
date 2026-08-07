"""Streamlit UI for the LLM DuckDB analyst and promotion/cannibalization analytics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.agent.llm_agent import (  # noqa: E402
    ANALYSIS_TOOL_NAMES,
    PromotionDataAgent,
    TABLE_INFO,
    trace_rows_contain_nested,
)
from app.config import (  # noqa: E402
    API_BASE_URL,
    DATA_DIR,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
)


st.set_page_config(
    page_title="Promotion Intelligence Hub",
    page_icon="📈",
    layout="wide",
)
st.markdown(
    """
    <style>
    /* Keep Streamlit's icon glyphs functional; these are icons, not text. */
    [data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"] *,
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    }

    /* Remove Streamlit chrome requested for the end-user UI. */
    #MainMenu,
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stMainMenu"],
    [data-testid="stAppDeployButton"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    button[data-testid="stBaseButton-header"],
    button[kind="header"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
    }

    [data-testid="stMetricValue"] {
        direction: ltr;
    }

    /* Technical content stays monospace for readability */
    pre,
    code,
    [data-testid="stCode"],
    [data-testid="stDataFrame"],
    [data-testid="stJson"] {
        direction: ltr;
        text-align: left;
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
            "Liberation Mono", monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing DuckDB and tables...")
def get_data_agent() -> PromotionDataAgent:
    return PromotionDataAgent()


def render_query_traces(traces) -> None:
    if not traces:
        return
    with st.expander(f"Analysis & SQL details ({len(traces)} steps)"):
        for index, trace in enumerate(traces, start=1):
            if hasattr(trace, "tool"):
                tool = trace.tool
                status = trace.status
                reason = trace.reason
                sql = trace.sql
                error = trace.error
                rows = trace.rows
            else:
                tool = trace.get("tool", "")
                status = trace.get("status", "")
                reason = trace.get("reason", "")
                sql = trace.get("sql", "")
                error = trace.get("error")
                rows = trace.get("rows", [])
            st.markdown(f"**Step {index}: `{tool}` — {status}**")
            if reason:
                st.caption(reason)
            language = "json" if tool in ANALYSIS_TOOL_NAMES else "sql"
            st.code(sql or "", language=language)
            if error:
                st.error(error)
            elif rows:
                frame = pd.DataFrame(rows)
                if trace_rows_contain_nested(rows):
                    st.json(rows)
                else:
                    st.dataframe(frame, width="stretch")
            else:
                st.info("The query returned no results.")


def llm_tab() -> None:
    st.subheader("Data Analyst Chatbot")
    st.caption(
        f"Eight CSV tables from {DATA_DIR} are registered as read-only views. "
        "Write a message; the chatbot generates safe SQL, executes it, and answers in English."
    )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = DEFAULT_OPENAI_BASE_URL
    model = DEFAULT_OPENAI_MODEL

    info_column, clear_column = st.columns([4, 1])
    with info_column:
        st.caption(f"Model: `{model}` | Service: `{base_url}`")
    with clear_column:
        if st.button("Clear chat", key="clear_chat", width="stretch"):
            st.session_state["chat_messages"] = []
            st.rerun()

    examples = [
        "What are the total sales and basket counts per week?",
        "Show the top five product categories by sales value.",
        "Compare the number of households per campaign and the campaign duration.",
        "What is the coupon redemption count per campaign?",
    ]
    example_labels = [
        "Weekly sales",
        "Top categories",
        "Campaign comparison",
        "Coupon usage",
    ]

    with st.expander("Analyzable tables and suggested questions"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "table": table,
                        "description": info["description"],
                        "columns": ", ".join(info["columns"]),
                    }
                    for table, info in TABLE_INFO.items()
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        suggestion_columns = st.columns(2)
        selected_example = None
        for index, (label, example) in enumerate(zip(example_labels, examples)):
            with suggestion_columns[index % 2]:
                if st.button(label, key=f"chat_example_{index}", width="stretch"):
                    selected_example = example

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    chat_messages = st.session_state["chat_messages"]
    # Reserve the conversation area before creating the composer. New messages,
    # the spinner and the generated answer are written into this earlier
    # container, so the text input always remains the last visible element.
    conversation_area = st.container()
    typed_question = st.chat_input(
        "e.g. Which product categories had the highest sales?",
        max_chars=4_000,
        key="chat_prompt",
    )
    question = typed_question or selected_example

    with conversation_area:
        if not chat_messages:
            with st.chat_message("assistant", avatar="📊"):
                st.markdown(
                    "Hi! I'm the promotion data analyst. Ask me about sales, products, "
                    "campaigns, customers, or coupons."
                )

        for message in chat_messages:
            avatar = "👤" if message["role"] == "user" else "📊"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_query_traces(message.get("traces", []))

        if question:
            prior_history = [
                {"role": message["role"], "content": message["content"]}
                for message in chat_messages[-20:]
            ]
            chat_messages.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="👤"):
                st.markdown(question)

            if not api_key.strip():
                answer = "API key is not set in the server environment."
                traces = []
            else:
                try:
                    with st.chat_message("assistant", avatar="📊"):
                        with st.spinner("Analyzing data and running safe SQL..."):
                            result = get_data_agent().ask(
                                question,
                                api_key=api_key,
                                base_url=base_url,
                                model=model,
                                history=prior_history,
                            )
                        answer = result.answer
                        traces = result.traces
                        st.markdown(answer)
                        render_query_traces(traces)
                except Exception as exc:
                    answer = f"Analysis failed: {type(exc).__name__}: {exc}"
                    traces = []

            if not api_key.strip() or answer.startswith("Analysis failed"):
                with st.chat_message("assistant", avatar="📊"):
                    st.markdown(answer)
            chat_messages.append(
                {"role": "assistant", "content": answer, "traces": traces}
            )


def _api_get(path: str) -> dict:
    # trust_env=False: ignore HTTP(S)/ALL_PROXY environment variables so a
    # system proxy (e.g. socks://127.0.0.1:12334) never breaks API calls.
    with httpx.Client(base_url=API_BASE_URL, timeout=120, trust_env=False) as client:
        response = client.get(path)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith("application/json") else response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


def _api_post(path: str, payload: dict) -> dict:
    with httpx.Client(base_url=API_BASE_URL, timeout=120, trust_env=False) as client:
        response = client.post(path, json=payload)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith("application/json") else response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


def _api_unavailable(exc: Exception) -> None:
    st.error(f"API server is not available: {exc}")
    st.code("python -m uvicorn app.main:app --host 127.0.0.1 --port 8001")


def promotion_tab() -> None:
    """Scenario 1 — promotion effect analysis (actual vs baseline vs incremental)."""
    st.subheader("Promotion Effect Analysis")
    st.caption(
        f"Data is read through the FastAPI server ({API_BASE_URL}) and the MCP layer. "
        "Incremental sales = actual sales − baseline sales (without promotion)."
    )
    try:
        campaigns = _api_get("/campaigns")["campaigns"]
    except Exception as exc:
        _api_unavailable(exc)
        return

    campaign_ids = [c["campaign_id"] for c in campaigns]
    campaign_labels = {
        c["campaign_id"]: f"Campaign {c['campaign_id']} ({c['description']}) — week {c['start_week']}-{c['end_week']}"
        for c in campaigns
    }

    with st.form("promo_form"):
        campaign_id = st.selectbox(
            "Campaign", campaign_ids, format_func=lambda cid: campaign_labels[cid]
        )
        detail = _api_get(f"/campaigns/{campaign_id}")
        products = detail["products"]
        product_id = st.selectbox(
            "Product (optional)",
            [None, *products],
            format_func=lambda pid: "All campaign-predictable products" if pid is None else f"Product {pid}",
        )
        week_col1, week_col2 = st.columns(2)
        start_week = week_col1.number_input(
            "Start week", min_value=1, max_value=102, value=int(detail["start_week"])
        )
        end_week = week_col2.number_input(
            "End week", min_value=1, max_value=102, value=min(int(detail["end_week"]), 101)
        )
        submitted = st.form_submit_button("Analyze promotion effect", type="primary", width="stretch")

    if not submitted:
        return

    try:
        effect = _api_post(
            "/analytics/promotion-effect",
            {
                "campaign_id": campaign_id,
                "product_id": product_id,
                "start_week": int(start_week),
                "end_week": int(end_week),
            },
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    with st.expander("Campaign info", expanded=True):
        cols = st.columns(4)
        cols[0].write(f"**Campaign:** {campaign_id}")
        cols[1].write(f"**Type:** {detail['description']}")
        cols[2].write(f"**Days:** {detail['start_day']} to {detail['end_day']}")
        cols[3].write(f"**Product count:** {detail['n_products']:,}")

    if not effect.get("data_sufficient"):
        st.warning(effect["message"])
        return

    metrics = st.columns(4)
    metrics[0].metric("Actual sales (units)", f"{effect['actual_sales']:,.0f}")
    metrics[1].metric("Baseline sales (units)", f"{effect['baseline_sales']:,.1f}")
    metrics[2].metric("Incremental sales (units)", f"{effect['incremental_sales']:,.1f}")
    uplift = effect["uplift_percentage"]
    metrics[3].metric("Uplift (%)", f"{uplift:,.1f}%" if uplift is not None else "—")
    st.caption(
        f"Comparison basis: stores covered by the baseline model. "
        f"Actual sales across all stores: {effect['actual_sales_total']:,.0f} units. "
        f"Products analyzed: {effect['n_products_analyzed']}."
    )
    if effect.get("message"):
        st.warning(effect["message"])
    st.success(effect["effectiveness_summary"])

    weekly = pd.DataFrame(effect.get("weekly") or [])
    if not weekly.empty:
        chart = weekly.set_index("week")[["actual", "baseline"]]
        st.line_chart(chart)
        weekly["incremental"] = weekly["actual"] - weekly["baseline"]
        st.bar_chart(weekly.set_index("week")["incremental"])

    with st.spinner("Generating LLM explanation (via MCP)..."):
        try:
            explanation = _api_post(
                "/analytics/explain",
                {"campaign_id": campaign_id, "product_id": product_id},
            )
            st.markdown(explanation["answer"])
            render_query_traces(explanation["traces"])
        except Exception as exc:
            st.info(f"LLM explanation is not available (OPENAI_API_KEY must be set on the server): {exc}")


def render_cannibalization_network(promoted_id: int, affected: list[dict]):
    """Simple directed network view: promoted product -> affected products."""
    import matplotlib.pyplot as plt

    n = len(affected)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    max_lost = max((float(a["lost_quantity"]) for a in affected), default=1.0)
    ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) if n > 1 else [0.0]

    ax.scatter([0.0], [0.0], s=900, c="#1f77b4", edgecolors="black", zorder=3)
    ax.annotate(f"Product {promoted_id}", (0, 0), color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    for affected_item, y in zip(affected, ys):
        x = 1.3
        lost = float(affected_item["lost_quantity"])
        size = 350 + 250 * lost / max_lost
        ax.scatter([x], [y], s=size, c="#d62728", edgecolors="black", zorder=3)
        ax.annotate(f"Product {affected_item['product_id']}\n-{lost:,.0f}", (x, y),
                    color="white", ha="center", va="center", fontsize=8)
        ax.plot([0.12, x - 0.04], [0.0, y], color="#d62728",
                lw=0.8 + 2.2 * lost / max_lost, alpha=0.55, zorder=2)
    ax.set_xlim(-0.45, 2.2)
    ax.set_ylim(-(n - 1) / 2 - 0.7, (n - 1) / 2 + 0.7)
    ax.set_title("Cannibalization network view", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def cannibalization_tab() -> None:
    """Scenario 2 — cannibalization effect analysis."""
    st.subheader("Cannibalization Analysis")
    st.caption(
        f"Cannibalization model output is read through the FastAPI server ({API_BASE_URL}). "
        "Effects are reported only when the model has sufficient evidence."
    )
    try:
        promoted = _api_get("/analytics/promoted-products")["products"]
    except Exception as exc:
        _api_unavailable(exc)
        return

    product_ids = [p["product_id"] for p in promoted]
    product_labels = {
        p["product_id"]: f"Product {p['product_id']} (estimated loss {p['total_lost']:,.0f} units)"
        for p in promoted
    }

    with st.form("cannibal_form"):
        promoted_id = st.selectbox(
            "Promoted product", product_ids, format_func=lambda pid: product_labels[pid]
        )
        week_col1, week_col2 = st.columns(2)
        start_week = week_col1.number_input("Start week", min_value=1, max_value=102, value=98)
        end_week = week_col2.number_input("End week", min_value=1, max_value=102, value=101)
        submitted = st.form_submit_button("Analyze cannibalization", type="primary", width="stretch")

    if not submitted:
        return

    try:
        result = _api_post(
            "/analytics/cannibalization-effect",
            {
                "promoted_product_id": promoted_id,
                "start_week": int(start_week),
                "end_week": int(end_week),
            },
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    metrics = st.columns(3)
    metrics[0].metric("Estimated lost units", f"{result['total_lost_quantity']:,.1f}")
    metrics[1].metric("Cannibalization %", f"{result['cannibalization_percentage']:,.1f}%")
    metrics[2].metric("Weeks with signal", result["n_flagged_weeks"])

    if result["signal"] in ("weak", "none"):
        st.info(result["summary"])
    else:
        st.success(result["summary"])

    affected = result.get("affected_products") or []
    if affected:
        table = pd.DataFrame(affected)
        st.dataframe(table, width="stretch", hide_index=True)
        st.bar_chart(table.set_index("product_id")["lost_quantity"])
        st.pyplot(render_cannibalization_network(promoted_id, affected))
    else:
        st.write("No affected products with sufficient evidence were reported.")

    with st.spinner("Generating LLM explanation (via MCP)..."):
        try:
            explanation = _api_post(
                "/analytics/explain",
                {
                    "question": (
                        f"Analyze the cannibalization of product {promoted_id} in weeks "
                        f"{int(start_week)} to {int(end_week)}."
                    )
                },
            )
            st.markdown(explanation["answer"])
            render_query_traces(explanation["traces"])
        except Exception as exc:
            st.info(f"LLM explanation is not available (OPENAI_API_KEY must be set on the server): {exc}")


def campaign_compare_tab() -> None:
    """Campaign ranking and comparison UI (Phase 4)."""
    st.subheader("Campaign Comparison & Ranking")
    st.caption(
        "Rank historical campaigns on normalized objectives (profit, sales or "
        "efficiency) with configurable weights and normalization."
    )
    try:
        campaigns = _api_get("/campaigns")["campaigns"]
    except Exception as exc:
        _api_unavailable(exc)
        return

    campaign_labels = {
        c["campaign_id"]: f"Campaign {c['campaign_id']} ({c['description']})"
        for c in campaigns
    }
    with st.form("compare_form"):
        selected = st.multiselect(
            "Campaigns to compare",
            [c["campaign_id"] for c in campaigns],
            default=[c["campaign_id"] for c in campaigns[:3]],
            format_func=lambda cid: campaign_labels[cid],
        )
        col1, col2 = st.columns(2)
        objective = col1.selectbox(
            "Objective",
            ["profit", "sales", "efficiency"],
            index=0,
        )
        normalization = col2.selectbox(
            "Normalization", ["min_max", "z_score", "percentile"], index=0
        )
        submitted = st.form_submit_button("Compare campaigns", type="primary", width="stretch")

    if not submitted or not selected:
        return

    try:
        result = _api_post(
            "/campaign/compare",
            {
                "campaigns": [int(cid) for cid in selected],
                "objective": objective,
                "normalization": normalization,
            },
        )
    except Exception as exc:
        st.error(f"Comparison failed: {exc}")
        return

    if result.get("message"):
        st.warning(result["message"])
        return

    ranking = result["ranking"]
    rows = pd.DataFrame(
        [
            {
                "campaign_id": row["campaign_id"],
                "description": row["description"],
                "score": row["score"],
                "incremental_profit": row["incremental_profit"],
                "incremental_sales": row["incremental_sales"],
                "roi": row["roi"] if row["roi"] is not None else 0.0,
                "cannibalization_risk": row["cannibalization_risk"],
                "promotion_cost": row["promotion_cost"],
            }
            for row in ranking
        ]
    )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.bar_chart(rows.set_index("campaign_id")["score"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Strengths**")
        for item in result["strengths"]:
            st.markdown(
                f"- Campaign {item['campaign_id']}: strong on "
                f"`{item['strength']}` ({item['strength_score']:.2f})"
            )
    with col_b:
        st.markdown("**Weaknesses**")
        for item in result["weaknesses"]:
            st.markdown(
                f"- Campaign {item['campaign_id']}: weak on "
                f"`{item['weakness']}` ({item['weakness_score']:.2f})"
            )


def recommendation_tab() -> None:
    """Product recommendation and campaign simulation UI (Phase 5)."""
    st.subheader("Multi-Objective Recommendation")
    st.caption(
        "Evaluate candidate products under a strategy (sales growth, profit "
        "optimization or safe promotion), optionally capped by a budget."
    )
    try:
        campaigns = _api_get("/campaigns")["campaigns"]
    except Exception as exc:
        _api_unavailable(exc)
        return

    candidate_pool: list[int] = []
    candidate_labels: dict[int, str] = {}
    for campaign in campaigns[:8]:
        detail = _api_get(f"/campaigns/{campaign['campaign_id']}")
        for product_id in detail["products"][:5]:
            if product_id not in candidate_labels:
                candidate_labels[product_id] = f"Product {product_id} (campaign {campaign['campaign_id']})"
                candidate_pool.append(product_id)

    with st.form("recommend_form"):
        selected = st.multiselect(
            "Candidate products",
            candidate_pool,
            default=candidate_pool[:3],
            format_func=lambda pid: candidate_labels.get(pid, f"Product {pid}"),
        )
        col1, col2, col3 = st.columns(3)
        objective = col1.selectbox(
            "Strategy",
            ["SALES_GROWTH", "PROFIT_OPTIMIZATION", "SAFE_PROMOTION"],
            index=1,
        )
        budget = col2.number_input("Budget cap ($, 0 = unlimited)", min_value=0.0, value=0.0, step=100.0)
        max_risk = col3.number_input("Max cannibalization risk (%)", min_value=0.0, value=100.0, step=5.0)
        submitted = st.form_submit_button("Recommend products", type="primary", width="stretch")

    if submitted and selected:
        constraints = {"max_cannibalization_risk": float(max_risk)} if max_risk < 100 else None
        try:
            result = _api_post(
                "/campaign/recommend",
                {
                    "products": [int(pid) for pid in selected],
                    "objective": objective,
                    "budget": float(budget) if budget > 0 else None,
                    "constraints": constraints,
                },
            )
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            return
        if result.get("message"):
            st.warning(result["message"])
            return
        recs = result["recommendations"]
        if recs:
            table = pd.DataFrame(
                [
                    {
                        "product_id": row["product_id"],
                        "score": row["score"],
                        "expected_sales": row["expected_sales"],
                        "expected_profit": row["expected_profit"],
                        "roi": row["roi"] if row["roi"] is not None else 0.0,
                        "cannibalization_risk": row["cannibalization_risk"],
                    }
                    for row in recs
                ]
            )
            st.dataframe(table, width="stretch", hide_index=True)
            st.bar_chart(table.set_index("product_id")["score"])
            st.markdown("**Explanations**")
            for row in recs:
                st.markdown(f"- **Product {row['product_id']}** ({row['score']:.0f}/100): {row['explanation']}")

    st.divider()
    st.markdown("**Campaign simulation**")
    st.caption(
        "Simulate a hypothetical discount: baseline demand, expected sales, "
        "ROI and risks."
    )
    with st.form("simulate_form"):
        col1, col2, col3, col4 = st.columns(4)
        sim_product = col1.number_input("Product ID", min_value=1, value=1005637, step=1)
        discount = col2.number_input("Discount (%)", min_value=0.1, max_value=90.0, value=20.0, step=5.0)
        weeks = col3.number_input("Weeks", min_value=1, max_value=52, value=3, step=1)
        start_week = col4.number_input("Start week", min_value=1, max_value=102, value=89, step=1)
        sim_submitted = st.form_submit_button("Simulate campaign", type="primary", width="stretch")

    if sim_submitted:
        try:
            simulation = _api_post(
                "/campaign/simulate",
                {
                    "product_id": int(sim_product),
                    "discount_percentage": float(discount),
                    "weeks": int(weeks),
                    "start_week": int(start_week),
                },
            )
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            return
        if not simulation.get("available"):
            st.warning(simulation.get("message", "Simulation not available."))
            return
        cols = st.columns(4)
        cols[0].metric("Baseline sales", f"{simulation['baseline_sales']:,.1f}")
        cols[1].metric("Expected sales", f"{simulation['expected_sales']:,.1f}")
        cols[2].metric("Incremental sales", f"{simulation['incremental_sales']:,.1f}")
        cols[3].metric("ROI", f"{simulation['roi']:.2f}" if simulation["roi"] is not None else "—")
        st.markdown("**Assumptions**")
        for assumption in simulation["assumptions"]:
            st.caption(f"- {assumption}")
        st.markdown("**Risks**")
        for risk in simulation["risks"]:
            st.markdown(f"- {risk}")


def main() -> None:
    st.title("Promotion Intelligence Hub")
    tab_llm, tab_promo, tab_cannibal, tab_compare, tab_recommend = st.tabs(
        [
            "💬 Analyst Chat",
            "📊 Promotion Analysis",
            "🔄 Cannibalization",
            "🏆 Campaign Compare",
            "🎯 Recommend & Simulate",
        ]
    )
    with tab_llm:
        llm_tab()
    with tab_promo:
        promotion_tab()
    with tab_cannibal:
        cannibalization_tab()
    with tab_compare:
        campaign_compare_tab()
    with tab_recommend:
        recommendation_tab()


if __name__ == "__main__":
    main()
