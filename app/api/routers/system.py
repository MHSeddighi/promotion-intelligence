"""Root, health, baseline inference and MCP tool-listing routes."""

from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.api import deps
from app.api.deps import BASELINE
from app.api.schemas import BaselineRequest, BaselineResponse, PredictionRow
from app.mcp.tools import TOOL_DEFINITIONS

router = APIRouter()


@router.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Unified Promotion Intelligence",
        "status": "ready" if deps.APP_STATE["ready"] else "starting",
        "docs": "/docs",
        "baseline_predict": "/baseline/predict",
        "assistant_ask": "/assistant/ask",
        "campaigns": "/campaigns",
        "promotion_effect": "/analytics/promotion-effect",
        "cannibalization_effect": "/analytics/cannibalization-effect",
        "explain": "/analytics/explain",
    }


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if deps.APP_STATE["ready"] else "not_ready",
        "models_loaded": deps.APP_STATE["ready"],
        "database_ready": deps.DATA_AGENT is not None,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "loaded_at": deps.APP_STATE["loaded_at"],
        "warmup_prediction": deps.APP_STATE["warmup_prediction"],
        **BASELINE.health(),
    }


@router.get("/baseline/features")
def baseline_features() -> dict[str, Any]:
    return {
        "feature_count": len(BASELINE.features),
        "features_in_required_order": BASELINE.features,
        "categorical_features": BASELINE.categorical_features,
    }


@router.get("/baseline/sample")
def baseline_sample() -> dict[str, Any]:
    row = BASELINE.load_sample().iloc[0]
    record: dict[str, int | float | None] = {}
    for feature in BASELINE.features:
        value = row[feature]
        if pd.isna(value):
            record[feature] = None
        elif feature in BASELINE.categorical_features:
            record[feature] = int(value)
        else:
            record[feature] = float(value)
    return {"records": [record]}


@router.post("/baseline/predict", response_model=BaselineResponse)
def baseline_predict(payload: BaselineRequest) -> BaselineResponse:
    if not deps.APP_STATE["ready"]:
        raise HTTPException(status_code=503, detail="Application is not ready.")
    frame = pd.DataFrame([record.model_dump() for record in payload.records])
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = BASELINE.predict(frame).reset_index(drop=True)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    predictions = [
        PredictionRow(
            row=int(row_number),
            probability_positive=float(row["probability_positive"]),
            predicted_positive_quantity=float(row["predicted_positive_quantity"]),
            baseline_prediction=float(row["baseline_prediction"]),
        )
        for row_number, row in result.iterrows()
    ]
    return BaselineResponse(
        count=len(predictions),
        predictions=predictions,
        warnings=[str(item.message) for item in caught],
    )


@router.get("/mcp/tools")
def mcp_tools() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for name, info in TOOL_DEFINITIONS.items()
        ]
    }
