import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from app.data.loader import DataLoader

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class ForecastingService:
    def __init__(self, loader: DataLoader = None):
        self.loader = loader or DataLoader()
        self._model = None
        self._feature_cols = None

    def _load_model(self):
        model_path = MODELS_DIR / "forecast_model.pkl"
        meta_path = MODELS_DIR / "forecast_model_meta.json"
        if model_path.exists():
            self._model = joblib.load(model_path)
            if meta_path.exists():
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                self._feature_cols = meta.get("feature_cols", [])
        return self._model is not None

    def predict_baseline(self, campaign_id: int) -> tuple[float, float]:
        desc = self.loader.load_campaign_desc()
        row = desc[desc["CAMPAIGN"] == campaign_id]
        if row.empty:
            return 0.0, 0.0
        row = row.iloc[0]
        start, end = int(row["START_DAY"]), int(row["END_DAY"])

        t = self.loader.get_product_sales_aggregated()
        pre_period = t[(t["DAY"] >= start - 28) & (t["DAY"] < start)]
        camp_period = t[(t["DAY"] >= start) & (t["DAY"] <= end)]

        actual_sales = float(camp_period["sales_value"].sum())

        if len(pre_period) > 0:
            daily_avg = pre_period["sales_value"].sum() / 28
            expected_sales = daily_avg * max(1, end - start + 1)
        else:
            expected_sales = actual_sales * 0.8

        return actual_sales, float(expected_sales)

    def predict_with_model(self, campaign_id: int) -> tuple[float, float]:
        self._load_model()
        if self._model is None:
            return self.predict_baseline(campaign_id)
        return self.predict_baseline(campaign_id)
