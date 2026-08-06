"""Paths and environment-backed configuration for the unified application."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")


def _environment_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


ARTIFACT_DIR = _environment_path(
    "BASELINE_ARTIFACT_DIR", PROJECT_ROOT / "outputs" / "baseline_engine"
)
DATA_DIR = _environment_path("PROMOTION_DATA_DIR", PROJECT_ROOT / "data" / "raw")
CANNIBALIZATION_DIR = _environment_path(
    "CANNIBALIZATION_ARTIFACT_DIR",
    PROJECT_ROOT / "outputs" / "cannibalization",
)

DEFAULT_OPENAI_BASE_URL = "https://api.deepseek.com"
DEFAULT_OPENAI_MODEL = "deepseek-chat"
DEFAULT_MAX_QUERY_ROWS = int(os.getenv("LLM_MAX_QUERY_ROWS", "200"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
