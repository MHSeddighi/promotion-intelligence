import json
import sys
import numpy as np
from typing import Any
from app.mcp.tools import (
    analyze_campaign,
    calculate_campaign_roi,
    find_best_campaigns,
    detect_cannibalization,
    recommend_future_campaign,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


TOOLS = {
    "analyze_campaign": {
        "function": analyze_campaign,
        "description": "Get full analysis of a campaign including impact, ROI, and cannibalization",
        "parameters": {
            "type": "object",
            "properties": {
                "campaign_id": {"type": "integer", "description": "The campaign ID to analyze"}
            },
            "required": ["campaign_id"],
        },
    },
    "calculate_campaign_roi": {
        "function": calculate_campaign_roi,
        "description": "Calculate ROI for a specific campaign",
        "parameters": {
            "type": "object",
            "properties": {
                "campaign_id": {"type": "integer", "description": "The campaign ID"}
            },
            "required": ["campaign_id"],
        },
    },
    "find_best_campaigns": {
        "function": find_best_campaigns,
        "description": "Find top performing campaigns by ROI",
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "Number of top campaigns to return", "default": 5}
            },
        },
    },
    "detect_cannibalization": {
        "function": detect_cannibalization,
        "description": "Detect product cannibalization effects for a campaign",
        "parameters": {
            "type": "object",
            "properties": {
                "campaign_id": {"type": "integer", "description": "The campaign ID"}
            },
            "required": ["campaign_id"],
        },
    },
    "recommend_future_campaign": {
        "function": recommend_future_campaign,
        "description": "Recommend optimal discount and strategy for a future campaign",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "integer", "description": "The product ID"},
                "budget": {"type": "number", "description": "Budget for the campaign"},
                "duration_days": {"type": "integer", "description": "Campaign duration in days", "default": 14},
            },
            "required": ["product_id", "budget"],
        },
    },
}


def handle_request(request: dict) -> dict:
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": name,
                    "description": info["description"],
                    "input_schema": info["parameters"],
                }
                for name, info in TOOLS.items()
            ]
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in TOOLS:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = TOOLS[tool_name]["function"](**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown method: {method}"}


def run_server():
    encoder = NumpyEncoder()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            sys.stdout.write(json.dumps(response, cls=NumpyEncoder) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            sys.stderr.write(f"Invalid JSON: {line}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    run_server()
