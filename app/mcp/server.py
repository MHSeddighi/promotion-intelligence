"""Minimal stdio MCP server exposing the promotion analytics tools.

Speaks JSON lines over stdin/stdout:

    {"method": "tools/list", "params": {}}
    {"method": "tools/call", "params": {"name": "get_campaign_effect",
                                        "arguments": {"campaign_id": 15}}}

Run with:

    python -m app.mcp.server
"""

from __future__ import annotations

import json
import sys
from typing import Any

from app.mcp.tools import MCPTools, TOOL_DEFINITIONS


def _public_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": info["description"],
            "input_schema": {
                "type": "object",
                "properties": info["parameters"],
                "required": [key for key, spec in info["parameters"].items() if spec.get("__required__")],
            },
        }
        for name, info in TOOL_DEFINITIONS.items()
    ]


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    params = request.get("params") or {}
    tools = MCPTools()

    if method == "tools/list":
        return {"tools": _public_tools()}
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in TOOL_DEFINITIONS:
            return {"error": f"Unknown tool: {name}"}
        try:
            return {"result": tools.call(name, arguments)}
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": f"{type(exc).__name__}: {exc}"}
    return {"error": f"Unknown method: {method}"}


def run_server() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError:
            response = {"error": f"Invalid JSON: {line}"}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_server()
