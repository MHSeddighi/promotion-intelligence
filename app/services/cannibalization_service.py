import pandas as pd
import numpy as np
from app.data.loader import DataLoader


class CannibalizationService:
    def __init__(self, loader: DataLoader = None):
        self.loader = loader or DataLoader()
        self._daily_cache: pd.DataFrame | None = None

    def _get_daily(self):
        if self._daily_cache is not None:
            return self._daily_cache
        self._daily_cache = self.loader.get_product_sales_aggregated()
        return self._daily_cache
    def detect(self, campaign_id: int) -> dict:
        desc = self.loader.load_campaign_desc()
        row = desc[desc["CAMPAIGN"] == campaign_id]
        if row.empty:
            return {"error": "Campaign not found"}

        row = row.iloc[0]
        start, end = int(row["START_DAY"]), int(row["END_DAY"])

        daily = self._get_daily()

        pre = daily[(daily["DAY"] >= start - 28) & (daily["DAY"] < start)]
        during = daily[(daily["DAY"] >= start) & (daily["DAY"] <= end)]

        pre_product = pre.groupby("PRODUCT_ID", as_index=False).agg(
            pre_qty=("quantity", "sum"),
        )
        during_product = during.groupby("PRODUCT_ID", as_index=False).agg(
            during_qty=("quantity", "sum"),
        )

        merged = pre_product.merge(during_product, on="PRODUCT_ID", how="outer").fillna(0)

        pre_days = max(1, pre["DAY"].nunique())
        during_days = max(1, end - start + 1)

        merged["expected_qty"] = (merged["pre_qty"] / pre_days) * during_days
        merged["change"] = merged["during_qty"] - merged["expected_qty"]

        top_during = during.groupby("PRODUCT_ID")["quantity"].sum().nlargest(5).index
        top_pre = pre.groupby("PRODUCT_ID")["quantity"].sum().nlargest(5).index

        affected = merged[
            (merged["change"] < -1) &
            (~merged["PRODUCT_ID"].isin(top_during))
        ].sort_values("change")

        p = self.loader.load_products()
        affected_products = []
        total_lost = 0.0
        for _, ap in affected.head(10).iterrows():
            lost = abs(float(ap["change"]))
            total_lost += lost
            pinfo = p[p["PRODUCT_ID"] == ap["PRODUCT_ID"]]
            pname = str(pinfo["COMMODITY_DESC"].values[0]) if not pinfo.empty else "Unknown"
            affected_products.append({
                "product_id": int(ap["PRODUCT_ID"]),
                "product_name": pname,
                "estimated_lost_sales": round(lost, 2),
            })

        promoted_id = int(top_during[0]) if len(top_during) > 0 else None
        during_qty = during["quantity"].sum()
        cannibal_score = round(total_lost / during_qty, 4) if during_qty > 0 else 0

        return {
            "campaign_id": campaign_id,
            "promoted_product_id": promoted_id,
            "affected_products": affected_products,
            "total_lost_sales": round(total_lost, 2),
            "cannibalization_score": min(cannibal_score, 1.0),
        }

