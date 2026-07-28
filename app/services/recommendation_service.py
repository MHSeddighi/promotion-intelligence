import pandas as pd
import numpy as np
from app.data.loader import DataLoader
from app.services.causal_service import CausalService
from app.services.cannibalization_service import CannibalizationService


class RecommendationService:
    def __init__(self, loader: DataLoader = None):
        self.loader = loader or DataLoader()
        self.causal = CausalService(loader)
        self.cannibalization = CannibalizationService(loader)

    def rank_campaigns(self) -> list[dict]:
        desc = self.loader.load_campaign_desc()
        rankings = []

        for _, row in desc.iterrows():
            camp_id = int(row["CAMPAIGN"])
            try:
                impact = self.causal.estimate_impact(camp_id)
                cannibal = self.cannibalization.detect(camp_id)
            except Exception:
                continue

            roi = impact.get("roi", 0)
            inc_sales = impact.get("incremental_sales_adjusted", 0)
            inc_rev = impact.get("incremental_revenue", 0)
            cannibal_score = cannibal.get("cannibalization_score", 0)

            if roi > 2.0 and inc_sales > 0 and cannibal_score < 0.2:
                reason = "high ROI with strong incremental sales and low cannibalization"
            elif roi > 1.0 and inc_sales > 0:
                reason = "positive ROI with decent incremental sales"
            elif roi > 0 and inc_sales > 0:
                reason = "marginally positive but low impact"
            elif roi <= 0 and inc_sales > 0:
                reason = "positive sales impact but cost exceeds revenue"
            elif roi <= -1:
                reason = "significant losses - campaign cost far exceeds returns"
            else:
                reason = "ineffective - negative or negligible impact"

            rankings.append({
                "campaign_id": camp_id,
                "roi": round(roi, 2),
                "incremental_sales": round(inc_sales, 2),
                "incremental_revenue": round(inc_rev, 2),
                "cannibalization_score": round(cannibal_score, 4),
                "reason": reason,
            })

        rankings.sort(key=lambda x: x["roi"], reverse=True)
        return rankings

    def get_best_campaigns(self, top_n: int = 5) -> list[dict]:
        rankings = self.rank_campaigns()
        return rankings[:top_n]

    def get_worst_campaigns(self, top_n: int = 5) -> list[dict]:
        rankings = self.rank_campaigns()
        return rankings[-top_n:]

    def get_effective_patterns(self) -> list[dict]:
        rankings = self.rank_campaigns()
        desc = self.loader.load_campaign_desc()
        t = self.loader.load_transactions()
        p = self.loader.load_products()

        patterns = []
        for r in rankings[:10]:
            camp_id = r["campaign_id"]
            row = desc[desc["CAMPAIGN"] == camp_id]
            if row.empty:
                continue
            row = row.iloc[0]
            start, end = int(row["START_DAY"]), int(row["END_DAY"])
            duration = max(1, end - start + 1)

            camp_data = t[(t["DAY"] >= start) & (t["DAY"] <= end)]
            if camp_data.empty:
                continue

            total_disc = camp_data["RETAIL_DISC"].abs().sum() + camp_data["COUPON_DISC"].abs().sum()
            total_sales = camp_data["SALES_VALUE"].sum()
            avg_disc_pct = total_disc / total_sales if total_sales > 0 else 0

            top_products = camp_data.groupby("PRODUCT_ID")["QUANTITY"].sum().nlargest(3).index
            product_info = p[p["PRODUCT_ID"].isin(top_products)][["PRODUCT_ID", "COMMODITY_DESC", "DEPARTMENT"]].drop_duplicates()

            patterns.append({
                "campaign_id": camp_id,
                "roi": r["roi"],
                "duration_days": duration,
                "avg_discount_pct": round(float(avg_disc_pct), 4),
                "start_day": int(start),
                "top_categories": product_info["COMMODITY_DESC"].unique().tolist()[:3],
                "departments": product_info["DEPARTMENT"].unique().tolist()[:3],
            })

        return patterns

    def recommend_scenario(self, product_id: int, budget: float, discount_range: tuple[float, float], duration_days: int) -> dict:
        t = self.loader.load_transactions()
        p = self.loader.load_products()

        product_data = t[t["PRODUCT_ID"] == product_id]
        if product_data.empty:
            return self._default_scenario(budget, duration_days)

        avg_weekly_qty = product_data.groupby("WEEK_NO")["QUANTITY"].sum().mean()
        avg_price = (
            product_data["SALES_VALUE"].sum() / product_data["QUANTITY"].sum()
            if product_data["QUANTITY"].sum() > 0 else 10.0
        )

        results = []
        for disc_pct in np.arange(discount_range[0], discount_range[1] + 0.01, 0.05):
            disc_pct = round(disc_pct, 2)
            price_after_disc = avg_price * (1 - disc_pct)
            elasticity = -1.5
            qty_multiplier = 1 + elasticity * disc_pct
            expected_qty = avg_weekly_qty * qty_multiplier * (duration_days / 7)
            expected_qty = max(0, expected_qty)

            revenue = expected_qty * price_after_disc
            cost = expected_qty * avg_price * disc_pct
            profit = revenue - cost
            roi = profit / cost if cost > 0 else 0

            if cost <= budget:
                results.append({
                    "discount": disc_pct,
                    "expected_revenue": round(revenue, 2),
                    "expected_profit": round(profit, 2),
                    "expected_roi": round(roi, 2),
                    "expected_incremental_sales": round(expected_qty, 2),
                    "cost": round(cost, 2),
                })

        if not results:
            return self._default_scenario(budget, duration_days)

        best = max(results, key=lambda x: x["expected_profit"])
        return {
            "recommended_discount": best["discount"],
            "expected_revenue": best["expected_revenue"],
            "expected_profit": best["expected_profit"],
            "expected_roi": best["expected_roi"],
            "expected_incremental_sales": best["expected_incremental_sales"],
            "confidence": "medium" if len(results) > 2 else "low",
        }

    def _default_scenario(self, budget: float, duration_days: int) -> dict:
        return {
            "recommended_discount": 0.15,
            "expected_revenue": budget * 2.5,
            "expected_profit": budget * 1.5,
            "expected_roi": 1.5,
            "expected_incremental_sales": 100.0,
            "confidence": "low",
        }
