import pandas as pd
import numpy as np
from app.data.loader import DataLoader


class PromotionService:
    def __init__(self, loader: DataLoader = None):
        self.loader = loader or DataLoader()

    def get_all_campaigns(self):
        desc = self.loader.load_campaign_desc()
        t = self.loader.load_transactions()
        results = []
        for _, row in desc.iterrows():
            camp_id = row["CAMPAIGN"]
            start, end = row["START_DAY"], row["END_DAY"]
            camp_sales = t[(t["DAY"] >= start) & (t["DAY"] <= end)]
            results.append({
                "campaign_id": int(camp_id),
                "description": row.get("DESCRIPTION", ""),
                "start_day": int(start),
                "end_day": int(end),
                "total_sales": float(camp_sales["SALES_VALUE"].sum()),
                "total_quantity": int(camp_sales["QUANTITY"].sum()),
                "total_discount": float(camp_sales["RETAIL_DISC"].abs().sum()),
            })
        return results

    def get_campaign(self, campaign_id: int):
        desc = self.loader.load_campaign_desc()
        row = desc[desc["CAMPAIGN"] == campaign_id]
        if row.empty:
            return None
        row = row.iloc[0]
        t = self.loader.load_transactions()
        start, end = row["START_DAY"], row["END_DAY"]
        camp_sales = t[(t["DAY"] >= start) & (t["DAY"] <= end)]
        return {
            "campaign_id": int(campaign_id),
            "description": str(row.get("DESCRIPTION", "")),
            "start_day": int(start),
            "end_day": int(end),
            "total_sales": float(camp_sales["SALES_VALUE"].sum()),
            "total_quantity": int(camp_sales["QUANTITY"].sum()),
            "total_discount": float(camp_sales["RETAIL_DISC"].abs().sum()),
        }

    def get_campaign_sales_data(self, campaign_id: int) -> pd.DataFrame:
        desc = self.loader.load_campaign_desc()
        row = desc[desc["CAMPAIGN"] == campaign_id]
        if row.empty:
            return pd.DataFrame()
        row = row.iloc[0]
        t = self.loader.load_transactions()
        start, end = int(row["START_DAY"]), int(row["END_DAY"])
        return t[(t["DAY"] >= start) & (t["DAY"] <= end)].copy()

    def calculate_promotion_cost(self, campaign_id: int) -> float:
        camp_data = self.get_campaign_sales_data(campaign_id)
        if camp_data.empty:
            return 0.0
        return float(camp_data["RETAIL_DISC"].abs().sum() + camp_data["COUPON_DISC"].abs().sum())
