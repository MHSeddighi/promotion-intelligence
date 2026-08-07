"""Causal inference for promotion impact using store-level control groups.

The estimator follows ``notebook/02_promotion_causality.ipynb``: for product-weeks
where the product is promoted in some stores and not in others, the non-promoted
stores are the counterfactual.

    index_treated = sum(units_treated) / sum(baseline_treated)
    index_control = sum(units_control) / sum(baseline_control)
    causal_lift   = index_treated / index_control - 1

The control index is the validation diagnostic: if the baseline predicts the
untreated stores accurately in these exact weeks, the excess in treated stores is
the promotion's own effect. Confidence intervals are computed with a
store-week bootstrap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.models.baseline import BaselinePredictor
from app.services.analytics import AnalyticsService


def bootstrap_causal_lift(
    units_treated: np.ndarray,
    baseline_treated: np.ndarray,
    units_control: np.ndarray,
    baseline_control: np.ndarray,
    n_iter: int = 300,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the causal lift percentage.

    Rows (store-weeks) are resampled with replacement inside the treated and
    control groups; each iteration recomputes the causal lift. Returns
    ``(lower_pct, upper_pct)``.
    """
    if n_iter < 10:
        raise ValueError("n_iter must be at least 10.")
    rng = np.random.default_rng(seed)
    units_t = np.asarray(units_treated, dtype="float64")
    baseline_t = np.asarray(baseline_treated, dtype="float64")
    units_c = np.asarray(units_control, dtype="float64")
    baseline_c = np.asarray(baseline_control, dtype="float64")

    if baseline_t.sum() <= 0 or baseline_c.sum() <= 0:
        return float("nan"), float("nan")

    lifts: list[float] = []
    for _ in range(n_iter):
        idx_t = rng.integers(0, len(units_t), size=len(units_t))
        idx_c = rng.integers(0, len(units_c), size=len(units_c))
        index_t = units_t[idx_t].sum() / max(baseline_t[idx_t].sum(), 1e-9)
        index_c = units_c[idx_c].sum() / max(baseline_c[idx_c].sum(), 1e-9)
        if index_c > 0:
            lifts.append(100.0 * (index_t / index_c - 1.0))
    if not lifts:
        return float("nan"), float("nan")
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


class CausalInferenceService:
    """Store-level difference-in-differences promotion impact estimation."""

    def __init__(
        self,
        analytics: AnalyticsService | None = None,
        predictor: BaselinePredictor | None = None,
    ) -> None:
        self.analytics = analytics or AnalyticsService()
        self._predictor = predictor
        self._cache: dict[int, pd.DataFrame] = {}

    @property
    def predictor(self) -> BaselinePredictor:
        if self._predictor is None:
            self._predictor = self.analytics.baseline
        return self._predictor

    def _product_cells(self, product_id: int) -> pd.DataFrame:
        """Panel rows for one product with baseline predictions and treatment."""
        if product_id in self._cache:
            return self._cache[product_id]
        panel = self.analytics.panel
        rows = panel[panel["PRODUCT_ID"] == int(product_id)].copy()
        if rows.empty:
            self._cache[product_id] = rows
            return rows
        features = self.predictor.features
        prediction = self.predictor.predict(rows[features])
        rows["baseline"] = prediction["baseline_prediction"].to_numpy(dtype="float64")
        rows["treated"] = (rows["promo_week"] > 0).astype("int8")
        self._cache[product_id] = rows
        return rows

    @staticmethod
    def _did_stats(
        cells: pd.DataFrame, start_week: int, end_week: int
    ) -> dict[str, Any]:
        frame = cells[
            cells["WEEK_NO"].between(start_week, end_week)
            & (cells["baseline"] > 0)
        ]
        if frame.empty:
            return {
                "data_sufficient": False,
                "confidence": "low",
                "n_treated_cells": 0,
                "n_control_cells": 0,
                "causal_lift_percentage": None,
                "raw_lift_percentage": None,
                "control_index": None,
                "treated_index": None,
                "confidence_interval": None,
                "incremental_units_causal": 0.0,
                "control_valid": False,
                "message": "Insufficient data to estimate causal promotion impact.",
            }

        treated = frame[frame["treated"] == 1]
        control = frame[frame["treated"] == 0]
        units_t = treated["qty"].to_numpy(dtype="float64")
        baseline_t = treated["baseline"].to_numpy(dtype="float64")
        units_c = control["qty"].to_numpy(dtype="float64")
        baseline_c = control["baseline"].to_numpy(dtype="float64")

        sum_units_t = float(units_t.sum())
        sum_base_t = float(baseline_t.sum())
        sum_units_c = float(units_c.sum())
        sum_base_c = float(baseline_c.sum())

        if len(treated) == 0 or len(control) == 0 or sum_base_t <= 0 or sum_base_c <= 0:
            return {
                "data_sufficient": False,
                "confidence": "low",
                "n_treated_cells": int(len(treated)),
                "n_control_cells": int(len(control)),
                "causal_lift_percentage": None,
                "raw_lift_percentage": None,
                "control_index": None,
                "treated_index": None,
                "confidence_interval": None,
                "incremental_units_causal": 0.0,
                "control_valid": False,
                "message": "No mixed promotion exposure (treated and control stores) in this window.",
            }

        treated_index = sum_units_t / sum_base_t
        control_index = sum_units_c / sum_base_c
        causal_lift = 100.0 * (treated_index / control_index - 1.0)
        raw_lift = 100.0 * (treated_index - 1.0)

        lower, upper = bootstrap_causal_lift(
            units_t, baseline_t, units_c, baseline_c
        )
        control_valid = 0.8 <= control_index <= 1.2

        # Counterfactual for treated cells based on control behaviour.
        counterfactual_treated = (
            sum_units_t / (1.0 + causal_lift / 100.0) if causal_lift > -100.0 else sum_base_t
        )
        incremental_causal = sum_units_t - counterfactual_treated

        if control_valid and len(treated) >= 5:
            confidence = "high" if len(treated) >= 25 else "medium"
            message = ""
        else:
            confidence = "low"
            message = (
                "Causal estimate is limited: the control stores do not track the "
                "baseline closely, or the treated sample is small."
            )

        return {
            "data_sufficient": True,
            "confidence": confidence,
            "message": message,
            "n_treated_cells": int(len(treated)),
            "n_control_cells": int(len(control)),
            "treated_index": round(treated_index, 4),
            "control_index": round(control_index, 4),
            "control_valid": bool(control_valid),
            "raw_lift_percentage": round(raw_lift, 2),
            "causal_lift_percentage": round(causal_lift, 2),
            "confidence_interval": (
                [round(lower, 2), round(upper, 2)]
                if not (np.isnan(lower) or np.isnan(upper))
                else None
            ),
            "incremental_units_causal": round(incremental_causal, 2),
            "treated_units": round(sum_units_t, 2),
            "treated_baseline": round(sum_base_t, 2),
            "control_units": round(sum_units_c, 2),
            "control_baseline": round(sum_base_c, 2),
        }

    def causal_effect(
        self,
        product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Causal promotion impact for one product over a week range."""
        start, end = self.analytics.resolve_period(
            start_week=start_week, end_week=end_week
        )
        cells = self._product_cells(int(product_id))
        if cells.empty:
            return {
                "product_id": int(product_id),
                "start_week": start,
                "end_week": end,
                "data_sufficient": False,
                "confidence": "low",
                "causal_lift_percentage": None,
                "incremental_units_causal": 0.0,
                "message": "Product is not covered by the baseline panel.",
            }
        stats = self._did_stats(cells, start, end)
        return {
            "product_id": int(product_id),
            "start_week": start,
            "end_week": end,
            **stats,
        }

    def campaign_causal_effect(
        self,
        campaign_id: int,
        product_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        """Aggregate causal impact across the campaign's forecastable products."""
        campaign = self.analytics.get_campaign(campaign_id)
        if campaign is None:
            return {"error": f"Campaign {campaign_id} not found."}
        start, end = self.analytics.resolve_period(
            campaign_id, start_week, end_week
        )
        end = min(end, int(self.analytics.panel["WEEK_NO"].max()))
        panel_products = set(self.analytics.panel["PRODUCT_ID"].unique())

        if product_id is not None:
            targets = [int(product_id)]
        else:
            targets = [pid for pid in campaign["products"] if pid in panel_products]

        effects = [self.causal_effect(pid, start, end) for pid in targets]
        available = [e for e in effects if e.get("data_sufficient")]
        if not available:
            return {
                "campaign_id": int(campaign_id),
                "product_id": product_id,
                "start_week": start,
                "end_week": end,
                "data_sufficient": False,
                "confidence": "low",
                "causal_lift_percentage": None,
                "incremental_units_causal": 0.0,
                "n_products_analyzed": 0,
                "products": [],
                "message": "Insufficient data to estimate causal promotion impact.",
            }

        weights = np.array(
            [max(e["treated_baseline"], 0.0) for e in available], dtype="float64"
        )
        if weights.sum() <= 0:
            weights = np.ones(len(available))
        weighted_lift = float(
            np.average(
                [e["causal_lift_percentage"] for e in available],
                weights=weights,
            )
        )
        incremental = sum(e["incremental_units_causal"] for e in available)
        confidence = min(
            (e["confidence"] for e in available),
            key={"high": 0, "medium": 1, "low": 2}.get,
        )
        return {
            "campaign_id": int(campaign_id),
            "product_id": product_id,
            "start_week": start,
            "end_week": end,
            "data_sufficient": True,
            "confidence": confidence,
            "causal_lift_percentage": round(weighted_lift, 2),
            "incremental_units_causal": round(incremental, 2),
            "n_products_analyzed": len(available),
            "message": "",
            "products": [
                {
                    "product_id": e["product_id"],
                    "causal_lift_percentage": e["causal_lift_percentage"],
                    "incremental_units_causal": e["incremental_units_causal"],
                    "control_index": e.get("control_index"),
                    "confidence": e["confidence"],
                }
                for e in available
            ],
        }

    def estimate_uplift(
        self,
        *,
        baseline_sales: float,
        causal_lift_percentage: float | None,
    ) -> float:
        """Expected incremental units for a counterfactual scenario."""
        if causal_lift_percentage is None:
            return 0.0
        return baseline_sales * (causal_lift_percentage / 100.0)
