import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


class DataLoader:
    def __init__(self, data_dir: str | Path = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._transactions: pd.DataFrame | None = None
        self._products: pd.DataFrame | None = None
        self._campaigns: pd.DataFrame | None = None
        self._campaign_desc: pd.DataFrame | None = None
        self._coupons: pd.DataFrame | None = None
        self._coupon_redempt: pd.DataFrame | None = None
        self._causal: pd.DataFrame | None = None

    def load_transactions(self) -> pd.DataFrame:
        if self._transactions is None:
            self._transactions = pd.read_csv(self.data_dir / "transaction_data.csv")
            required = {"household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE", "RETAIL_DISC", "WEEK_NO"}
            missing = required - set(self._transactions.columns)
            if missing:
                raise ValueError(f"Missing columns in transactions: {missing}")
        return self._transactions

    def load_products(self) -> pd.DataFrame:
        if self._products is None:
            self._products = pd.read_csv(self.data_dir / "product.csv")
            required = {"PRODUCT_ID", "DEPARTMENT", "BRAND", "COMMODITY_DESC", "SUB_COMMODITY_DESC"}
            missing = required - set(self._products.columns)
            if missing:
                raise ValueError(f"Missing columns in products: {missing}")
        return self._products

    def load_campaigns(self) -> pd.DataFrame:
        if self._campaigns is None:
            self._campaigns = pd.read_csv(self.data_dir / "campaign_table.csv")
        return self._campaigns

    def load_campaign_desc(self) -> pd.DataFrame:
        if self._campaign_desc is None:
            self._campaign_desc = pd.read_csv(self.data_dir / "campaign_desc.csv")
            required = {"CAMPAIGN", "START_DAY", "END_DAY"}
            missing = required - set(self._campaign_desc.columns)
            if missing:
                raise ValueError(f"Missing columns in campaign_desc: {missing}")
        return self._campaign_desc

    def load_coupons(self) -> pd.DataFrame:
        if self._coupons is None:
            self._coupons = pd.read_csv(self.data_dir / "coupon.csv")
        return self._coupons

    def load_coupon_redemptions(self) -> pd.DataFrame:
        if self._coupon_redempt is None:
            self._coupon_redempt = pd.read_csv(self.data_dir / "coupon_redempt.csv")
        return self._coupon_redempt

    def load_causal(self) -> pd.DataFrame:
        if self._causal is None:
            self._causal = pd.read_csv(self.data_dir / "causal_data.csv", nrows=100000)
        return self._causal

    def get_sales_with_products(self) -> pd.DataFrame:
        t = self.load_transactions()
        p = self.load_products()
        df = t.merge(p, on="PRODUCT_ID", how="left")
        return df

    def get_campaign_details(self) -> pd.DataFrame:
        camp = self.load_campaigns()
        desc = self.load_campaign_desc()
        df = camp.merge(desc, on="CAMPAIGN", how="left")
        return df

    def get_product_sales_aggregated(self) -> pd.DataFrame:
        t = self.load_transactions()
        p = self.load_products()
        df = t.merge(p, on="PRODUCT_ID", how="left")
        agg = df.groupby(["DAY", "PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"], as_index=False).agg(
            quantity=("QUANTITY", "sum"),
            sales_value=("SALES_VALUE", "sum"),
            retail_disc=("RETAIL_DISC", "sum"),
            basket_count=("BASKET_ID", "nunique"),
            week_no=("WEEK_NO", "first"),
        )
        return agg

    def get_daily_sales_by_product(self) -> pd.DataFrame:
        return self.get_product_sales_aggregated()

    def get_all_campaign_ids(self) -> list[int]:
        desc = self.load_campaign_desc()
        return sorted(desc["CAMPAIGN"].unique().tolist())
