"""
Feature engineering utilities shared across notebooks.

Provides functions for creating lag features, rolling statistics,
calendar features, and discount features.
"""

import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame, day_col: str = "DAY") -> pd.DataFrame:
    """Add day_of_week, month, quarter, week_of_year, is_weekend features."""
    result = df.copy()
    result["day_of_week"] = result[day_col] % 7
    result["month"] = ((result[day_col] / 30).astype(int) % 12) + 1
    result["quarter"] = ((result[day_col] / 30).astype(int) % 12 // 3) + 1
    result["week_of_year"] = (result[day_col] / 7).astype(int) % 52
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    return result


def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "PRODUCT_ID",
    value_col: str = "QUANTITY",
    lags: list[int] = None,
) -> pd.DataFrame:
    """Add lagged values of value_col grouped by group_col."""
    if lags is None:
        lags = [1, 7, 14]
    result = df.copy()
    for lag in lags:
        result[f"lag_{lag}"] = (
            result.groupby(group_col)[value_col].shift(lag).fillna(0)
        )
    return result


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str = "PRODUCT_ID",
    value_col: str = "QUANTITY",
    windows: list[int] = None,
) -> pd.DataFrame:
    """Add rolling mean and std of value_col grouped by group_col."""
    if windows is None:
        windows = [7, 14, 28]
    result = df.copy()
    for window in windows:
        result[f"rolling_mean_{window}"] = (
            result.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            .fillna(0)
        )
        result[f"rolling_std_{window}"] = (
            result.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).std().shift(1))
            .fillna(0)
        )
    return result


def add_discount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add retail_disc_abs, coupon_disc_abs, total_discount, has_discount, discount_rate."""
    result = df.copy()
    result["retail_disc_abs"] = result["RETAIL_DISC"].abs()
    result["coupon_disc_abs"] = result["COUPON_DISC"].abs()
    # result["total_discount"] = result["retail_disc_abs"] + result["coupon_disc_abs"]
    # result["has_discount"] = (result["total_discount"] > 0).astype(int)
    # result["discount_rate"] = np.where(
    #     result["SALES_VALUE"] + result["total_discount"] > 0,
    #     result["total_discount"] / (result["SALES_VALUE"] + result["total_discount"]),
    #     0.0,
    # )
    return result


def build_full_feature_set(
    df: pd.DataFrame,
    lags: list[int] = None,
    roll_windows: list[int] = None,
) -> pd.DataFrame:
    """
    Build a complete feature set from a sorted transaction-level DataFrame.
    
    Expects df to already have PRODUCT_ID, DAY, QUANTITY, SALES_VALUE,
    RETAIL_DISC, COUPON_DISC columns.
    """
    result = df.copy()
    result = add_calendar_features(result)
    result = add_discount_features(result)
    result = add_lag_features(result, lags=lags)
    result = add_rolling_features(result, windows=roll_windows)
    return result


# =============================================================================
# MVP layer: promotion-aware features + incremental sales helpers.
# Used by notebooks 04-09. Existing functions above are left untouched.
# =============================================================================

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

WEEKS_IN_WINDOW = 102


def build_dense_product_day_grid(
    product_ids: Sequence[int],
    max_day: int,
    min_day: int = 1,
) -> pd.DataFrame:
    """Dense (PRODUCT_ID, DAY) grid: every day for every product.

    Needed so lag/rolling features are well-defined even on zero-sale days.
    """
    days = np.arange(min_day, max_day + 1, dtype=np.int16)
    products = np.asarray(product_ids, dtype=np.int32)
    grid = pd.DataFrame(
        {"PRODUCT_ID": np.repeat(products, len(days)),
         "DAY": np.tile(days, len(products))}
    )
    return grid


def merge_promotion_day_features(
    grid: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach per product-day discount features derived from transactions."""
    disc_cols = ["PRODUCT_ID", "DAY", "QUANTITY", "SALES_VALUE", "RETAIL_DISC",
                 "COUPON_DISC", "BASKET_ID"]
    daily = (
        transactions[disc_cols]
        .groupby(["PRODUCT_ID", "DAY"], as_index=False)
        .agg(
            quantity=("QUANTITY", "sum"),
            sales_value=("SALES_VALUE", "sum"),
            retail_disc=("RETAIL_DISC", "sum"),
            coupon_disc=("COUPON_DISC", "sum"),
            baskets=("BASKET_ID", "nunique"),
        )
    )
    daily["has_discount"] = (daily["retail_disc"] < -0.005).astype("int8")
    daily["total_discount"] = (-(daily["retail_disc"] + daily["coupon_disc"])).clip(lower=0)
    daily["discount_rate"] = np.where(
        daily["sales_value"] + daily["total_discount"] > 0,
        daily["total_discount"] / (daily["sales_value"] + daily["total_discount"]),
        0.0,
    )
    out = grid.merge(daily, on=["PRODUCT_ID", "DAY"], how="left")
    for col in ["quantity", "sales_value", "retail_disc", "coupon_disc", "baskets",
                "total_discount", "discount_rate"]:
        out[col] = out[col].fillna(0)
    out["has_discount"] = out["has_discount"].fillna(0).astype("int8")
    return out


def merge_causal_week_features(
    grid: pd.DataFrame,
    causal_weekly: pd.DataFrame,
    week_of_day: dict | None = None,
) -> pd.DataFrame:
    """Attach weekly display/mailer promotion intensity to a product-day grid."""
    g = grid.copy()
    g["WEEK_NO"] = g["DAY"].map(day_to_week if week_of_day is None else week_of_day)
    g = g.merge(
        causal_weekly[["PRODUCT_ID", "WEEK_NO", "display_share", "mailer_share"]],
        on=["PRODUCT_ID", "WEEK_NO"], how="left",
    )
    g["display_share"] = g["display_share"].fillna(0.0)
    g["mailer_share"] = g["mailer_share"].fillna(0.0)
    g["has_display"] = (g["display_share"] > 0).astype("int8")
    g["has_mailer"] = (g["mailer_share"] > 0).astype("int8")
    g["promo_intensity"] = g["display_share"] + g["mailer_share"]
    return g


def day_to_week(day: int) -> int:
    """Map DAY (1-based) to WEEK_NO using the Dunnhumby week cadence."""
    return int(np.ceil(day / 7))


def add_campaign_flags(
    grid: pd.DataFrame,
    campaign_desc: pd.DataFrame,
    prefix: str = "campaign",
) -> pd.DataFrame:
    """Flag rows that fall inside any campaign window."""
    g = grid.copy()
    g["_has_campaign"] = 0
    for _, camp in campaign_desc.iterrows():
        mask = (g["DAY"] >= camp["START_DAY"]) & (g["DAY"] <= camp["END_DAY"])
        g.loc[mask, "_has_campaign"] = 1
    g[f"{prefix}_active"] = g["_has_campaign"].astype("int8")
    return g.drop(columns="_has_campaign")


def add_product_static_features(
    grid: pd.DataFrame,
    product_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach product-level static features (avg price, discount rate, category).

    Static columns that collide with day-level feature names get a ``_product``
    suffix so they never overwrite the dynamic features.
    """
    keep = [c for c in [
        "PRODUCT_ID", "avg_price", "discount_rate", "total_sales", "total_quantity",
        "purchase_frequency", "sales_frequency", "volatility", "DEPARTMENT",
        "COMMODITY_DESC", "BRAND",
    ] if c in product_features.columns]
    if "avg_price" not in keep:
        return grid
    static = product_features[keep].rename(columns={
        "discount_rate": "product_discount_rate",
        "purchase_frequency": "purchase_frequency",
        "sales_frequency": "sales_frequency",
    })
    if "discount_rate" in static.columns:
        static = static.rename(columns={"discount_rate": "product_discount_rate"})
    return grid.merge(static, on="PRODUCT_ID", how="left")


def incremental_stats(
    actual: pd.Series,
    baseline: pd.Series,
    residual_std: float = 0.0,
    z: float = 1.96,
) -> dict:
    """Incremental volume + uplift + residual-based confidence interval.

    actual / baseline are aligned 1-D arrays of quantities or revenue.
    """
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    inc = float(actual.sum() - baseline.sum())
    base_total = float(baseline.sum())
    uplift_pct = 100.0 * inc / base_total if base_total > 0 else 0.0
    if residual_std > 0:
        se = residual_std * np.sqrt(len(actual))
        half_width = z * se
        ci = (inc - half_width, inc + half_width)
        z_score = inc / se if se > 0 else float("nan")
        confidence = min(0.99, max(0.5, 0.5 + 0.5 * (1 - np.exp(-abs(z_score) / 2.0))))
    else:
        ci = (float("nan"), float("nan"))
        z_score = float("nan")
        confidence = 0.5
    return {
        "actual_total": float(actual.sum()),
        "baseline_total": base_total,
        "incremental": inc,
        "uplift_pct": uplift_pct,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "z_score": z_score,
        "confidence": float(confidence),
    }
