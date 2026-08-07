"""Financial / accounting metrics for promotion analysis.

The formulas mirror ``notebook/01_promotion_accounting.ipynb`` so the production
layer and the experimentation layer agree:

    incremental_units    = promotion_sales - baseline_sales
    incremental_revenue  = incremental_units * average_unit_price
    promotion_cost       = retailer-funded discount spend during the promotion
    incremental_profit   = incremental_revenue * margin - promotion_cost
    ROI                  = incremental_profit / promotion_cost
    breakeven_margin     = promotion_cost / incremental_revenue

All computation functions are pure: they take explicit numbers and return
deterministic results, which keeps them unit-testable with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FinancialMetrics:
    """Structured financial result for one promotion analysis."""

    baseline_sales: float = 0.0
    promotion_sales: float = 0.0
    incremental_units: float = 0.0
    average_price: float = 0.0
    incremental_revenue: float = 0.0
    margin: float = 0.0
    promotion_cost: float = 0.0
    incremental_profit: float = 0.0
    roi: float | None = None
    breakeven_margin: float | None = None
    cannibalized_units: float = 0.0
    cannibalized_revenue: float = 0.0
    net_incremental_units: float = 0.0
    net_incremental_revenue: float = 0.0
    net_incremental_profit: float = 0.0
    net_roi: float | None = None
    true_incremental_gain: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_sales": round(self.baseline_sales, 2),
            "promotion_sales": round(self.promotion_sales, 2),
            "incremental_units": round(self.incremental_units, 2),
            "average_price": round(self.average_price, 4),
            "incremental_revenue": round(self.incremental_revenue, 2),
            "margin": round(self.margin, 4),
            "promotion_cost": round(self.promotion_cost, 2),
            "incremental_profit": round(self.incremental_profit, 2),
            "roi": round(self.roi, 4) if self.roi is not None else None,
            "breakeven_margin": (
                round(self.breakeven_margin, 4)
                if self.breakeven_margin is not None
                else None
            ),
            "cannibalized_units": round(self.cannibalized_units, 2),
            "cannibalized_revenue": round(self.cannibalized_revenue, 2),
            "net_incremental_units": round(self.net_incremental_units, 2),
            "net_incremental_revenue": round(self.net_incremental_revenue, 2),
            "net_incremental_profit": round(self.net_incremental_profit, 2),
            "net_roi": round(self.net_roi, 4) if self.net_roi is not None else None,
            "true_incremental_gain": round(self.true_incremental_gain, 2),
            "notes": list(self.notes),
        }


def calculate_financial_metrics(
    *,
    baseline_sales: float,
    promotion_sales: float,
    average_price: float,
    margin: float,
    promotion_cost: float = 0.0,
    cannibalized_units: float = 0.0,
) -> FinancialMetrics:
    """Compute promotion accounting metrics from explicit inputs.

    Args:
        baseline_sales: expected sales without the promotion (units).
        promotion_sales: observed sales with the promotion (units).
        average_price: average selling price per unit during the window.
        margin: gross margin as a fraction (e.g. 0.20 for 20%).
        promotion_cost: retailer-funded discount spend (money).
        cannibalized_units: units lost by other products to this promotion.

    Returns:
        A fully populated :class:`FinancialMetrics` result.
    """
    if baseline_sales < 0 or promotion_sales < 0:
        raise ValueError("baseline_sales and promotion_sales must be non-negative.")
    if average_price < 0:
        raise ValueError("average_price must be non-negative.")
    if not 0.0 <= margin <= 1.0:
        raise ValueError("margin must be between 0 and 1.")
    if promotion_cost < 0:
        raise ValueError("promotion_cost must be non-negative.")
    if cannibalized_units < 0:
        raise ValueError("cannibalized_units must be non-negative.")

    incremental_units = promotion_sales - baseline_sales
    incremental_revenue = incremental_units * average_price
    incremental_profit = incremental_revenue * margin - promotion_cost

    roi: float | None = None
    if promotion_cost > 0:
        roi = incremental_profit / promotion_cost

    breakeven_margin: float | None = None
    if incremental_revenue > 0 and promotion_cost > 0:
        breakeven_margin = promotion_cost / incremental_revenue

    cannibalized_revenue = cannibalized_units * average_price
    net_incremental_units = incremental_units - cannibalized_units
    net_incremental_revenue = net_incremental_units * average_price
    net_incremental_profit = net_incremental_revenue * margin - promotion_cost
    net_roi: float | None = None
    if promotion_cost > 0:
        net_roi = net_incremental_profit / promotion_cost

    notes: list[str] = []
    if incremental_units <= 0:
        notes.append("The promotion did not generate incremental units.")
    if promotion_cost <= 0:
        notes.append("No retailer-funded discount cost detected; ROI is undefined.")
    if cannibalized_units > 0:
        notes.append("Results are shown gross and net of cannibalization.")

    return FinancialMetrics(
        baseline_sales=baseline_sales,
        promotion_sales=promotion_sales,
        incremental_units=incremental_units,
        average_price=average_price,
        incremental_revenue=incremental_revenue,
        margin=margin,
        promotion_cost=promotion_cost,
        incremental_profit=incremental_profit,
        roi=roi,
        breakeven_margin=breakeven_margin,
        cannibalized_units=cannibalized_units,
        cannibalized_revenue=cannibalized_revenue,
        net_incremental_units=net_incremental_units,
        net_incremental_revenue=net_incremental_revenue,
        net_incremental_profit=net_incremental_profit,
        net_roi=net_roi,
        true_incremental_gain=net_incremental_units,
        notes=notes,
    )


def roi_at_margin(
    *,
    incremental_revenue: float,
    promotion_cost: float,
    margin: float,
) -> float | None:
    """ROI for a given margin assumption (used for margin sensitivity)."""
    if promotion_cost <= 0:
        return None
    if incremental_revenue <= 0:
        return -1.0
    return (incremental_revenue * margin - promotion_cost) / promotion_cost
