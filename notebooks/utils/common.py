"""
Shared infrastructure for the MVP notebooks: paths, logging, seeding and
reusable validation helpers.

Every notebook 04-09 uses these helpers so validation, artifact export and
reload checks stay consistent across the pipeline.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import PROJECT_ROOT, find_project_root  # noqa: F401  (re-export)

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a console handler (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed numpy / python random for reproducible notebooks."""
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

def pipeline_dir(name: str) -> Path:
    """Return (and create) the outputs/<name> directory for a pipeline stage."""
    d = OUTPUTS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(name: str) -> Path:
    """Return (and create) the figures sub-directory for a pipeline stage."""
    d = pipeline_dir(name) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Validation helpers (used in every notebook's validation cells)
# ---------------------------------------------------------------------------

def check_no_nan_inf(df: pd.DataFrame, context: str = "", cols: Sequence[str] | None = None,
                     allow_inf: bool = False,
                     allow_nan_cols: Sequence[str] | None = None) -> None:
    """Assert a frame contains no NaN (and optionally no Inf) values.

    allow_nan_cols: columns where NaN is a legitimate value (e.g. undefined uplift).
    """
    target = df[list(cols)] if cols is not None else df
    allowed = list(allow_nan_cols or [])
    check = target.drop(columns=[c for c in allowed if c in target.columns])
    assert not check.isna().any().any(), f"{context}: NaN values present:\n{check.isna().sum()[check.isna().sum() > 0]}"
    if not allow_inf:
        numeric = target.select_dtypes(include=[np.number]).drop(columns=[c for c in allowed if c in target.columns], errors='ignore')
        if not numeric.empty:
            assert np.isfinite(numeric.to_numpy()).all(), f"{context}: Inf values present"
    print(f"PASS [nan/inf] {context}: no NaN/Inf in {target.shape[0]:,} rows")


def check_schema(df: pd.DataFrame, expected: Sequence[str], context: str = "") -> None:
    """Assert a frame contains at least the expected columns."""
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"{context}: missing columns {missing}"
    print(f"PASS [schema] {context}: {len(df.columns)} columns OK")


def save_parquet(df: pd.DataFrame, path: Path, logger: logging.Logger | None = None,
                 context: str = "") -> Path:
    """Write a parquet file and log the artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    msg = f"WROTE {path} ({df.shape[0]:,} rows x {df.shape[1]} cols)"
    if context:
        msg = f"[{context}] {msg}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return path


def reload_and_check(path: Path, expected: Sequence[str] | None = None,
                     context: str = "", allow_nan_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Reload a parquet artifact and validate it (read-back contract)."""
    path = Path(path)
    assert path.exists(), f"{context}: artifact missing: {path}"
    df = pd.read_parquet(path)
    check_no_nan_inf(df, context=f"reload {path.name}", cols=expected,
                     allow_nan_cols=allow_nan_cols)
    if expected is not None:
        check_schema(df, expected, context=f"reload {path.name}")
    print(f"PASS [reload] {context}: {path.name} -> {df.shape[0]:,} rows")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       y_baseline: np.ndarray | None = None) -> dict:
    """RMSE / MAE / MAPE / SMAPE / R2 for regression predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom)))
    mask = y_true != 0
    mape = float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if mask.any() else float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if len(y_true) > 1 else 1.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape, "r2": r2, "n": int(len(y_true))}
    if y_baseline is not None:
        y_base = np.asarray(y_baseline, dtype=float)
        mae_base = float(np.mean(np.abs(y_true - y_base)))
        rmse_base = float(np.sqrt(np.mean((y_true - y_base) ** 2)))
        out["mae_improvement_vs_naive_pct"] = 100.0 * (mae_base - mae) / max(mae_base, 1e-9)
        out["rmse_improvement_vs_naive_pct"] = 100.0 * (rmse_base - rmse) / max(rmse_base, 1e-9)
    return out


def format_metrics(metrics: dict) -> str:
    """Human-readable one-line metric summary."""
    return " | ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float)))


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def save_json(obj, path: Path, logger: logging.Logger | None = None, context: str = "") -> Path:
    """Serialize an object to JSON (for manifests / LLM-facing exports)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    if logger:
        logger.info("[%s] WROTE %s", context, path)
    return path


def top_categories(df: pd.DataFrame, cat_col: str, value_col: str, k: int = 3) -> list:
    """Top-k categories by summed value (for product-mix summaries)."""
    if df.empty:
        return []
    return df.groupby(cat_col)[value_col].sum().nlargest(k).index.tolist()
