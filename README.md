# Promotion Intelligence

End-to-end promotion analytics: a Streamlit frontend, a FastAPI backend, an MCP
server and an LLM agent that analyze campaign effects using ML models +
LLM + MCP tools — **baseline demand prediction**, **causal promotion impact**,
**cannibalization detection**, **product similarity**, **financial metrics**,
**campaign ranking**, **multi-objective recommendations** and **executive
reports**, with honest uncertainty handling.

## Architecture

```
Streamlit UI  (app/frontend)
      |
      v
FastAPI      (app/main.py + app/api/routers)
      |
      +------> LLM agent  (app/agent)  ---> MCP tools (app/mcp) ---> PromotionAnalysisEngine
      |                                                                  |
      +------> REST analytics endpoints ----------------------------->   |
                                                                         v
                              +---------------------+  +-----------------------------+
                              | AnalyticsService    |  | CausalInferenceService      |
                              |  (baseline, sales,  |  |  (DID control-store uplift) |
                              |   cannibalization)  |  +-----------------------------+
                              +---------------------+  +-----------------------------+
                              | FinancialMetrics    |  | CampaignRankingService      |
                              |  (ROI/margin/profit)|  |  (normalization + weights)  |
                              +---------------------+  +-----------------------------+
                              +---------------------+  +-----------------------------+
                              | Recommendation      |  | ProductSimilarityService    |
                              |  (strategies/sim)   |  |  (embeddings + substitutes) |
                              +---------------------+  +-----------------------------+
                              | PromotionAnalysisEngine (unified orchestrator)        |
                              +-------------------------------------------------------+
                              | baseline hurdle model (outputs/baseline_engine)       |
                              | raw transactions + campaigns (data/raw)               |
                              | cannibalization predictions (outputs/cannibalization) |
                              | product embeddings (outputs/product_embeddings)       |
                              +-------------------------------------------------------+
```

- **Baseline demand** — two-stage hurdle model in `outputs/baseline_engine`
  scored on the saved feature panel (`panel.parquet`); no retraining.
- **Causal impact** — store-level difference-in-differences: treated stores vs
  non-promoted control stores in the same product-weeks, with bootstrap 95% CI.
- **Incremental sales** — `incremental = actual_sales − baseline_sales` on the
  store basis covered by the model.
- **Cannibalization** — saved model outputs in
  `outputs/cannibalization_detection`; if evidence is insufficient, no effect is
  reported. `true_incremental_gain = incremental − cannibalized_units`.
- **Similarity** — saved embeddings in `outputs/product_embeddings`: cosine
  similarity search, substitutes and pair similarity.
- **Financials** — incremental revenue, incremental profit, promotion cost, ROI,
  breakeven margin, gross/net of cannibalization (margin configurable).
- **Ranking** — min-max / z-score / percentile normalization and configurable
  weighted scoring (`profit`, `sales`, `efficiency` presets).
- **Recommendation** — strategies `SALES_GROWTH`, `PROFIT_OPTIMIZATION`,
  `SAFE_PROMOTION`; budget caps, constraint filtering and discount simulation.
- **Uncertainty** — responses carry `data_sufficient`, `confidence` and `signal`
  flags; the agent phrases uncertainty instead of inventing numbers.

## Project layout

```
app/
  main.py                  # FastAPI entry point
  config.py                # settings (env: BASELINE_ARTIFACT_DIR, ...)
  api/                     # HTTP layer
    deps.py                # shared singletons + lifespan state
    schemas.py             # pydantic models
    routers/               # system, campaigns, analytics, promotion, assistant
  services/
    analytics.py           # AnalyticsService (campaigns, baseline, incremental, cannibalization)
    causal.py              # CausalInferenceService (DID uplift + bootstrap CI)
    financials.py          # FinancialMetricsService (pure ROI/margin accounting)
    ranking.py             # CampaignRankingService (normalization + weights)
    recommendation.py      # RecommendationService (strategies, simulation)
    engine.py              # PromotionAnalysisEngine (unified orchestrator)
  models/
    baseline.py            # BaselinePredictor (two-stage hurdle inference)
    similarity.py          # ProductSimilarityService (embeddings/substitutes)
  agent/llm_agent.py       # LLM agent (SQL tools + MCP analysis tools)
  mcp/                     # MCPTools registry + stdio server
  frontend/streamlit_app.py# Streamlit UI (chat, analysis, cannibalization,
                           #   campaign compare, recommend & simulate)
notebook/                  # experimentation notebooks (percent format)
tests/                     # unit, integration, API, MCP and E2E tests
data/raw/                  # CSV datasets
outputs/                   # model artifacts (baseline, cannibalization, embeddings)
```

## Setup

```bash
cd promotion-intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets must exist under `data/raw/` (`transaction_data.csv`, `product.csv`,
`campaign_desc.csv`, `campaign_table.csv`, `coupon.csv`, `coupon_redempt.csv`,
`hh_demographic.csv`, `causal_data.csv`) and model artifacts under `outputs/`
(generated by the notebooks in `notebook/`).

## Run

**Backend API**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Swagger: `http://127.0.0.1:8001/docs`

**Frontend**

```bash
python -m streamlit run app/frontend/streamlit_app.py
```

Tabs: analyst chat, promotion analysis, cannibalization, campaign comparison,
recommendations & simulation. The UI calls the backend at `API_BASE_URL`
(default `http://127.0.0.1:8001`).

**MCP server (stdio)**

```bash
python -m app.mcp.server
```

```json
{"method": "tools/list", "params": {}}
{"method": "tools/call", "params": {"name": "compare_campaigns", "arguments": {"campaign_ids": [15, 20]}}}
```

**Tests**

```bash
python -m pytest tests/ -q
```

## API overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Models / DB / OpenAI readiness |
| GET | `/campaigns` | List campaigns |
| GET | `/campaigns/{id}` | Campaign detail + promoted products |
| POST | `/baseline/predict` | Score the two-stage baseline model |
| POST | `/analytics/promotion-effect` | Actual, baseline, incremental, uplift |
| POST | `/analytics/incremental-sales` | Incremental sales only |
| POST | `/analytics/cannibalization-effect` | Affected products, lost quantity, signal |
| GET | `/analytics/promoted-products` | Products with cannibalization evidence |
| POST | `/promotion/analyze` | Unified analysis: baseline + uplift + incremental + cannibalization + financials + recommendation |
| POST | `/promotion/report` | Executive promotion report (summary, metrics, recommendations, risks) |
| POST | `/campaign/compare` | Rank/compare campaigns (ranking, scores, strengths, weaknesses) |
| POST | `/campaign/recommend` | Recommend products under a strategy/budget/constraints |
| POST | `/campaign/simulate` | Simulate a discount campaign (demand, ROI, risks) |
| POST | `/assistant/ask` | LLM data agent (SQL + MCP tools) |
| POST | `/analytics/explain` | LLM explanation of a campaign/cannibalization question |
| GET | `/mcp/tools` | List MCP tools |
| POST | `/mcp` | MCP JSON-RPC over HTTP (`tools/list`, `tools/call`) |

## MCP tools

Read-only analytics tools exposed to the LLM agent and the stdio MCP server:

- `list_campaigns`, `get_campaign`
- `get_campaign_effect`, `get_incremental_sales`, `calculate_incremental_sales`
- `get_product_baseline`, `get_baseline_prediction`
- `get_cannibalization_effect`, `detect_cannibalization`, `get_top_impacted_products`
- `find_similar_products`
- `calculate_financial_metrics`
- `rank_campaigns`, `compare_campaigns`, `recommend_campaigns`
- `analyze_incremental_sales_attribution` (why a promotion drove incremental sales)
- `simulate_campaign_strategy`, `generate_promotion_report`

The LLM never calculates metrics itself: it calls these tools and summarizes
their results. SQL tools are reserved for raw-data questions the analysis tools
do not cover.

## Tests

- Unit: financial metrics, normalization, weighted scoring, causal bootstrap,
  cosine similarity, strategy resolution, MCP registry/validation.
- Integration: baseline predictor, causal DID on real data, campaign ranking,
  recommendation engine, unified engine, API endpoints, MCP dispatch.
- E2E: the eight customer scenarios (promote?, why failed?, which products?,
  discount simulation, similar products, campaign comparison, profit vs sales,
  executive report).

LLM endpoints read `OPENAI_API_KEY` from the server environment (never from
requests). Demo inputs: campaign `15` + product `1005637` (positive effect),
product `934427` (strong cannibalization), product `1004906` (no
cannibalization), campaign `24` product `35656` (low data).
