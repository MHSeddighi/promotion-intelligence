import pandas as pd
import numpy as np
from app.data.loader import DataLoader


class CausalService:
    def __init__(self, loader: DataLoader = None):
        self.loader = loader or DataLoader()
        self._daily_sales_cache: pd.DataFrame | None = None

    def _get_daily_sales(self):
        if self._daily_sales_cache is not None:
            return self._daily_sales_cache
        self._daily_sales_cache = self.loader.get_product_sales_aggregated()
        return self._daily_sales_cache

    def estimate_impact(self, campaign_id: int) -> dict:
        desc = self.loader.load_campaign_desc()
        row = desc[desc["CAMPAIGN"] == campaign_id]
        if row.empty:
            return {"error": "Campaign not found"}

        row = row.iloc[0]
        start, end = int(row["START_DAY"]), int(row["END_DAY"])

        daily = self._get_daily_sales()

        pre = daily[(daily["DAY"] >= start - 28) & (daily["DAY"] < start)]
        during = daily[(daily["DAY"] >= start) & (daily["DAY"] <= end)]
        post = daily[(daily["DAY"] > end) & (daily["DAY"] <= end + 28)]

        pre_sales = float(pre["sales_value"].sum())
        during_sales = float(during["sales_value"].sum())
        post_sales = float(post["sales_value"].sum()) if not post.empty else 0
        during_days = max(1, end - start + 1)
        daily_baseline = pre_sales / 28.0
        expected_sales = daily_baseline * during_days
        incremental_sales = during_sales - expected_sales

        pre_discount = float(pre["retail_disc"].abs().sum())
        during_discount = float(during["retail_disc"].abs().sum())
        promo_cost = during_discount

        avg_price = during_sales / during["quantity"].sum() if during["quantity"].sum() > 0 else 0
        incremental_revenue = incremental_sales * avg_price
        incremental_profit = incremental_revenue - promo_cost
        roi = incremental_profit / promo_cost if promo_cost > 0 else 0.0

        pre_trend = 0
        if len(pre) > 3:
            pre_daily_vals = pre.groupby("DAY")["sales_value"].sum()
            if len(pre_daily_vals) > 3:
                x = np.arange(len(pre_daily_vals))
                y = pre_daily_vals.values
                coeffs = np.polyfit(x, y, 1)
                pre_trend = float(coeffs[0] * during_days)

        seasonality_effect = 0
        if not post.empty:
            post_days = len(post["DAY"].unique())
        if post_days > 0:
                post_daily_avg = post_sales / post_days
                seasonality_effect = (post_daily_avg - daily_baseline) * during_days

        adjusted_incremental = incremental_sales - pre_trend - seasonality_effect

        return {
            "campaign_id": campaign_id,
            "actual_sales": round(during_sales, 2),
            "expected_sales_without_promo": round(expected_sales, 2),
            "incremental_sales_raw": round(incremental_sales, 2),
            "incremental_sales_adjusted": round(adjusted_incremental, 2),
            "incremental_revenue": round(incremental_revenue, 2),
            "incremental_profit": round(incremental_profit, 2),
            "promotion_cost": round(promo_cost, 2),
            "roi": round(roi, 4),
            "pre_period_sales": round(pre_sales, 2),
            "post_period_sales": round(post_sales, 2),
            "pre_trend_effect": round(pre_trend, 2),
            "seasonality_effect": round(seasonality_effect, 2),
        }
