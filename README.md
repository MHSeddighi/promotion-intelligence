# Promotion Intelligence Backend

AI-powered promotion analytics system that analyzes historical marketing campaigns to estimate incremental sales, detect cannibalization, calculate ROI, and recommend future campaign strategies.

## Architecture

```
Client / AI Agent
         |
    FastAPI Application Layer
    /                    \
REST API              MCP Server
    \                    /
      Service Layer
  ----------------------------
  Promotion Analytics Services
  Causal Impact Service
  Cannibalization Service
  Recommendation Service
  Forecasting Service
      |
    ML Models Layer
      |
    MLflow Model Registry
      |
    Static Dataset Assets
```

Both REST API and MCP Server share the same service layer — no logic duplication.

## Data Flow

1. **Static CSV files** (`data/raw/`) are loaded by `DataLoader`
2. **Service layer** reads pre-aggregated data, computes causal impact, cannibalization, and ROI
3. **REST API** exposes results via FastAPI endpoints
4. **MCP Server** exposes the same services as AI agent tools over stdio
5. **MLflow** tracks trained forecasting models

## Project Structure

```
promotion_ai/
├── app/
│   ├── main.py                    # FastAPI application entry
│   ├── api/
│   │   ├── campaigns.py           # GET /campaigns, GET /campaigns/{id}
│   │   ├── analytics.py           # GET /analytics/campaigns/{id}/impact, cannibalization, forecast
│   │   └── recommendations.py     # GET /recommendations, POST /scenario
│   ├── mcp/
│   │   ├── server.py              # MCP stdio server (tools/list, tools/call)
│   │   └── tools.py               # MCP tool implementations (wrap services)
│   ├── services/
│   │   ├── promotion_service.py   # Campaign listing and basic stats
│   │   ├── causal_service.py      # Difference-in-differences causal impact estimation
│   │   ├── cannibalization_service.py  # Product substitution detection
│   │   ├── forecasting_service.py # LightGBM baseline sales prediction
│   │   └── recommendation_service.py  # Campaign ranking and scenario optimization
│   ├── models/
│   │   ├── forecasting.py         # LightGBM training/prediction helpers
│   │   └── optimizer.py           # Discount optimization via grid search
│   ├── mlflow/
│   │   └── registry.py            # Model save/load with MLflow tracking
│   ├── data/
│   │   └── loader.py              # Static CSV loader with validation
│   └── schemas/
│       └── campaign.py            # Pydantic models for API contracts
├── data/
│   └── raw/                       # Static CSV datasets (not committed)
├── tests/
│   ├── test_loader.py             # Data loader unit tests
│   ├── test_services.py           # Service layer unit tests
│   └── test_api.py                # API integration tests
├── requirements.txt
└── README.md
```

## REST API Usage

### Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info and available endpoints |
| GET | `/campaigns` | List all campaigns |
| GET | `/campaigns/{id}` | Get campaign details |
| GET | `/analytics/campaigns/{id}/impact` | Causal impact analysis with ROI |
| GET | `/analytics/campaigns/{id}/cannibalization` | Product cannibalization detection |
| GET | `/analytics/campaigns/{id}/forecast` | Baseline sales forecast |
| GET | `/recommendations` | Ranked campaign performance |
| GET | `/recommendations/best?top_n=5` | Best performing campaigns |
| GET | `/recommendations/worst?top_n=5` | Worst performing campaigns |
| GET | `/recommendations/patterns` | Discover effective campaign patterns |
| POST | `/recommendations/scenario` | Recommend optimal discount scenario |

### Scenario Request Example

```json
{
  "product_id": 1004906,
  "budget": 5000.0,
  "discount_range": [0.05, 0.30],
  "duration_days": 14
}
```

### Scenario Response

```json
{
  "recommended_discount": 0.05,
  "expected_revenue": 183.78,
  "expected_profit": 174.11,
  "expected_roi": 18.0,
  "expected_incremental_sales": 80.12,
  "confidence": "medium"
}
```

## MCP Usage

The MCP server operates over stdio. It implements the [Model Context Protocol](https://modelcontextprotocol.io) for AI agent integration.

### Available MCP Tools

- **`analyze_campaign`** — Full campaign analysis (impact + ROI + cannibalization)
- **`calculate_campaign_roi`** — ROI calculation for a campaign
- **`find_best_campaigns`** — Top N campaigns by ROI
- **`detect_cannibalization`** — Product substitution effects
- **`recommend_future_campaign`** — Optimal discount strategy recommendation

### Starting MCP Server

```bash
python3 -m promotion_ai.app.mcp.server
```

### MCP Communication (JSON over stdio)

```json
{"method": "tools/list", "params": {}}
{"method": "tools/call", "params": {"name": "analyze_campaign", "arguments": {"campaign_id": 1}}}
```

## ML Workflow

### Forecasting Model

Uses LightGBM regression with:
- Lag features (1, 7, 14 days)
- Rolling averages (7, 14 days)
- Calendar features (day_of_week, month, quarter)
- Discount features
- Product-level aggregates

### MLflow Integration

```python
from promotion_ai.app.mlflow.registry import ModelRegistry

registry = ModelRegistry()
registry.save_model(model, "forecast_model", {"n_samples": 10000})
model, metadata = registry.load_model("forecast_model")
```

Models are saved to `models/` directory and tracked in MLflow experiments.

### Causal Impact

Simple difference-in-differences approach:
- Pre-period (28 days before campaign)
- During period
- Post-period (28 days after)
- Adjusts for pre-trend and seasonality effects

### Cannibalization Detection

Compares product sales during vs. before campaign:
- Identifies products with decreased sales during promotion
- Excludes promoted products (top 5 by volume)
- Calculates cannibalization score as lost sales / total sales

## Running Instructions

### Prerequisites

Python 3.10+

### Setup

```bash
# Clone repository
cd promotion-intelligence

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure static datasets exist at:
# data/raw/transaction_data.csv
# data/raw/product.csv
# data/raw/campaign_desc.csv
# data/raw/campaign_table.csv
```

### Run FastAPI Server

```bash
PYTHONPATH=. python3 -m uvicorn promotion_ai.app.main:app --host 0.0.0.0 --port 8000
```

### Run MCP Server

```bash
PYTHONPATH=. python3 -m promotion_ai.app.mcp.server
```

### Run Tests

```bash
PYTHONPATH=. python3 -m pytest promotion_ai/tests/ -v
```

## Key Design Decisions

- **No database** — Everything runs from static CSV files loaded into pandas
- **No authentication** — Hackathon prototype
- **No frontend** — API-only and MCP-only
- **Shared services** — REST API and MCP use the same service layer
- **Cached aggregations** — Services cache aggregated data to avoid repeated expensive joins
- **Simple causal model** — Difference-in-differences instead of complex causal inference
- **Grid search optimization** — Simple discount optimization instead of RL
