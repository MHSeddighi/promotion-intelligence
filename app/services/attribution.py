"""Read-only access to incremental-sales attribution artifacts.

``notebook/05_incremental_sales_attribution.ipynb`` saved its pipeline results
(period-level features/targets) and three complete per-product attribution
reports to ``outputs/attribution_engine/``.  This service exposes those saved
results to MCP tools and the LLM agent without re-running or modifying the
notebook.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "attribution_engine"

# Feature subsets per business reason, mirroring the notebook's 43-feature groups.
REASON_FEATURES: dict[str, list[str]] = {
    "promotion_effect": [
        "discount_depth",
        "discount_pct",
        "display",
        "mailer",
        "promo_type",
        "duration",
        "hist_promo_uplift_12w",
        "hist_promo_freq_12w",
        "hist_disc_depth_12w",
    ],
    "cannibalization": [
        "sub_sales_ratio_8w",
        "emb_cosine_max",
        "emb_cosine_mean10",
        "sub_commodity_overlap",
        "brand_overlap",
        "n_substitutes",
    ],
    "demand_expansion": [
        "household_growth_8v16",
        "cat_growth_8v16",
        "hist_purchase_freq",
        "hist_households_rmean8",
        "demand_volatility",
    ],
    "basket_expansion": [
        "co_purchase_top5_mean",
        "co_purchase_top10_mean",
        "co_purchase_breadth",
        "hist_basket_size",
        "hist_basket_value",
    ],
    "price_response": [
        "price_gap_pct",
        "price_during",
        "price_before",
        "hist_price_elasticity",
        "hist_discount_response",
    ],
    "unknown": ["stock_missing", "date_limitation", "sparse_product"],
}


class IncrementalSalesAttributionService:
    """Loads and serves the notebook's saved attribution results."""

    def __init__(self, outputs_dir: str | Path | None = None) -> None:
        env_dir = os.environ.get("ATTRIBUTION_OUTPUTS_DIR")
        self.outputs_dir = Path(outputs_dir or env_dir or DEFAULT_OUTPUTS_DIR)
        self._periods: pd.DataFrame | None = None
        self._reports: list[dict[str, Any]] | None = None
        self._report_index: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------- data access
    @property
    def periods(self) -> pd.DataFrame:
        if self._periods is None:
            path = self.outputs_dir / "periods.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Attribution artifacts not found at {self.outputs_dir}. "
                    "Run the incremental-sales attribution notebook first."
                )
            self._periods = pd.read_parquet(path)
        return self._periods

    @property
    def reports(self) -> list[dict[str, Any]]:
        if self._reports is None:
            path = self.outputs_dir / "example_reports.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Attribution example reports not found at {self.outputs_dir}."
                )
            self._reports = json.loads(path.read_text(encoding="utf-8"))
        return self._reports

    @property
    def report_index(self) -> dict[str, dict[str, Any]]:
        if self._report_index is None:
            self._report_index = {r["promotion_id"]: r for r in self.reports}
        return self._report_index

    # ------------------------------------------------------------- discovery
    def coverage(self) -> dict[str, Any]:
        """Summary of what attribution results are available."""
        df = self.periods
        return {
            "periods": int(len(df)),
            "products": int(df["product_id"].nunique()),
            "week_range": [int(df["start_week"].min()), int(df["end_week"].max())],
            "example_reports": [r["promotion_id"] for r in self.reports],
        }

    def list_promotions(
        self, product_id: int | None = None, limit: int = 50
    ) -> list[str]:
        """Valid promotion IDs, optionally filtered to one product."""
        df = self.periods
        if product_id is not None:
            df = df[df["product_id"] == int(product_id)]
        return [str(pid) for pid in df["promotion_id"].head(limit).tolist()]

    # ------------------------------------------------------------- analysis
    def analyze(
        self,
        promotion_id: str | int | None = None,
        product_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Return the saved attribution report for a promotion period.

        A promotion can be identified by ``promotion_id`` (``P1234-80-84``) or by
        ``product_id`` + ``start_week`` + ``end_week``.  Full reason-level
        attribution is only saved for the example promotions in
        ``example_reports.json``; every other period falls back to the saved
        period metrics and feature context.
        """
        pid = self._resolve_promotion_id(promotion_id, product_id, start_week, end_week)
        if isinstance(pid, dict):
            return pid

        report = self.report_index.get(pid)
        if report is not None:
            return self._full_report(report, top_k)
        return self._period_fallback(pid, product_id, start_week, end_week)

    # ------------------------------------------------------------- internals
    def _resolve_promotion_id(
        self,
        promotion_id: str | int | None,
        product_id: int | None,
        start_week: int | None,
        end_week: int | None,
    ) -> str | dict[str, Any]:
        if promotion_id is not None:
            pid = str(promotion_id).strip()
            if not pid.upper().startswith("P"):
                pid = f"P{pid}"
            if product_id is not None:
                embedded = self._product_from_promotion_id(pid)
                if embedded is not None and embedded != int(product_id):
                    return {
                        "error": (
                            f"Promotion {pid} belongs to product {embedded}, "
                            f"not product {product_id}."
                        )
                    }
            return pid
        if product_id is not None and start_week is not None and end_week is not None:
            return f"P{int(product_id)}-{int(start_week)}-{int(end_week)}"
        return {
            "error": (
                "Provide promotion_id (e.g. P1234-80-84), or product_id with "
                "start_week and end_week."
            )
        }

    @staticmethod
    def _product_from_promotion_id(promotion_id: str) -> int | None:
        try:
            return int(promotion_id.lstrip("Pp").split("-", 1)[0])
        except (ValueError, IndexError):
            return None

    def _full_report(self, report: dict[str, Any], top_k: int) -> dict[str, Any]:
        out = deepcopy(report)
        out["top_k_reasons"] = self._top_k(out["reason_attribution"], top_k)
        out["executive_summary"] = self.executive_summary(out)
        return out

    def _period_fallback(
        self,
        promotion_id: str,
        product_id: int | None,
        start_week: int | None,
        end_week: int | None,
    ) -> dict[str, Any]:
        df = self.periods
        rows = df[df["promotion_id"] == promotion_id]
        if product_id is not None:
            rows = rows[rows["product_id"] == int(product_id)]
        if start_week is not None:
            rows = rows[rows["start_week"] == int(start_week)]
        if end_week is not None:
            rows = rows[rows["end_week"] == int(end_week)]
        if rows.empty:
            return self._not_found(promotion_id, product_id)

        row = rows.iloc[0]
        features = {
            reason: {
                col: self._json_value(row.get(col))
                for col in cols
                if col in row.index
            }
            for reason, cols in REASON_FEATURES.items()
        }
        return {
            "promotion_id": promotion_id,
            "product_id": int(row["product_id"]),
            "period": {
                "start_week": int(row["start_week"]),
                "end_week": int(row["end_week"]),
                "duration": int(row.get("duration", 1)),
            },
            "incremental_sales": self._json_value(row.get("incremental_sales")),
            "actual_sales": self._json_value(row.get("actual_qty")),
            "baseline_sales": self._json_value(row.get("baseline_qty")),
            "features": features,
            "message": (
                "Saved attribution results found for this period, but the "
                "reason-level report (SHAP attribution, evidence, confidence) "
                "was only generated for the example promotions listed in "
                "available_example_reports. Use one of those promotion IDs for "
                "the full explainability report."
            ),
            "available_example_reports": [r["promotion_id"] for r in self.reports],
        }

    def _not_found(
        self, promotion_id: str, product_id: int | None
    ) -> dict[str, Any]:
        if product_id is not None:
            valid = self.list_promotions(product_id=product_id, limit=20)
        else:
            valid = self.list_promotions(limit=20)
        return {
            "error": (
                f"No attribution results for promotion {promotion_id}. "
                "Use a promotion_id from valid_promotion_ids or one of the "
                "available_example_reports."
            ),
            "valid_promotion_ids": valid,
            "available_example_reports": [r["promotion_id"] for r in self.reports],
            "coverage": self.coverage(),
        }

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    @staticmethod
    def _top_k(reasons: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        k = max(1, min(int(k), 6))
        ordered = sorted(
            reasons, key=lambda r: abs(float(r.get("impact_units", 0.0))), reverse=True
        )[:k]
        out: list[dict[str, Any]] = []
        for rank, reason in enumerate(ordered, start=1):
            item = dict(reason)
            units = float(item.get("impact_units", 0.0))
            item["rank"] = rank
            item["direction"] = "positive" if units >= 0 else "negative"
            item["summary"] = (
                f"{item['label']} "
                f"{'added' if units >= 0 else 'reduced'} {abs(units):,.0f} units "
                f"({abs(float(item.get('impact_percentage', 0.0))):.0f}%)"
            )
            out.append(item)
        return out

    @staticmethod
    def executive_summary(report: dict[str, Any]) -> str:
        """One-paragraph summary, matching the notebook's helper."""
        top = report["top_k_reasons"]
        line = (
            f"Promotion {report['promotion_id']} generated "
            f"{report['incremental_sales']:,.0f} incremental units "
            f"({report['actual_sales']:,.0f} actual vs "
            f"{report['baseline_sales']:,.0f} baseline). "
        )
        line += "Top drivers: " + "; ".join(
            f"{r['rank']}. {r['label']} {r['impact_units']:+,.0f} units "
            f"({r['impact_percentage']:+.0f}%)"
            for r in top
        ) + ". "
        if report.get("uncertainty_statements"):
            line += "Limitations: " + " ".join(report["uncertainty_statements"])
        return line
