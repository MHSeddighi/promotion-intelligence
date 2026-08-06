"""Analytics services: promotion effect, incremental sales, cannibalization.

Every number comes from the implemented models/features already present in the
repository (no retraining, no fabricated effects):

- Baseline demand  -> the trained two-stage hurdle model in
                      ``outputs/baseline_engine``, scored on the saved feature
                      panel (product x store x week).
- Actual sales     -> raw ``transaction_data.csv`` quantities.
- Cannibalization  -> the S-Learner causal treatment-effect model in
                      ``outputs/cannibalization`` (``pair_effects.csv``);
                      adapted here from ``source_product/target_product/
                      cannibalization_quantity`` into the promoted/affected
                      schema this service exposes. Only pair-weeks where the
                      source product was genuinely promoted
                      (``source_promo_flag == 1``) are treated as observed
                      cannibalization events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import ARTIFACT_DIR, CANNIBALIZATION_DIR, DATA_DIR
from app.models.baseline import BaselinePredictor

# Model/data conventions kept in sync with the baseline/cannibalization notebooks
TRAIN_END = 78  # baseline training window end; used to judge demand-history evidence
TEST_WINDOW = (89, 101)  # baseline held-out window; cannibalization evidence lives here
MIN_HISTORY_WEEKS_HIGH = 25
MIN_HISTORY_WEEKS_MEDIUM = 10
MIN_CANNIBALIZATION_LOST = 0.2  # units; below this the model has no meaningful signal
MIN_AFFECTED_LOST = 0.05  # units; below this an affected product is not reported

LOW_DATA_MESSAGE = (
    "Insufficient historical data to reliably estimate promotion impact."
)
LOW_BASELINE_CONFIDENCE_MESSAGE = (
    "Baseline prediction confidence is limited because this product has sparse "
    "demand history."
)
NO_CANNIBALIZATION_MESSAGE = "No strong cannibalization signal was detected."


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class AnalyticsService:
    """Read-only analytics over the saved baseline panel, raw transactions and
    the saved cannibalization predictions. All heavy inputs are loaded lazily
    and cached for the process lifetime."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        artifact_dir: Path = ARTIFACT_DIR,
        cannibalization_dir: Path = CANNIBALIZATION_DIR,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.artifact_dir = Path(artifact_dir).resolve()
        self.cannibalization_dir = Path(cannibalization_dir).resolve()
        self._baseline: BaselinePredictor | None = None
        self._panel: pd.DataFrame | None = None
        self._transactions: pd.DataFrame | None = None
        self._cannibalization: pd.DataFrame | None = None
        self._campaign_desc: pd.DataFrame | None = None
        self._coupon: pd.DataFrame | None = None
        self._products: pd.DataFrame | None = None
        self._day_week: pd.Series | None = None

    # ------------------------------------------------------------------ data
    @property
    def baseline(self) -> BaselinePredictor:
        if self._baseline is None:
            self._baseline = BaselinePredictor(self.artifact_dir)
        return self._baseline

    @property
    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._panel = pd.read_parquet(self.artifact_dir / "panel.parquet")
        return self._panel

    @property
    def transactions(self) -> pd.DataFrame:
        if self._transactions is None:
            self._transactions = pd.read_csv(
                self.data_dir / "transaction_data.csv",
                usecols=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "QUANTITY", "SALES_VALUE", "RETAIL_DISC"],
            )
        return self._transactions

    @property
    def cannibalization_predictions(self) -> pd.DataFrame:
        """Adapt the S-Learner's ``pair_effects.csv`` into the promoted/affected
        schema this service exposes (no separate model logic is duplicated
        here, only a column mapping)."""
        if self._cannibalization is None:
            raw = pd.read_csv(
                self.cannibalization_dir / "pair_effects.csv",
                usecols=[
                    "source_product",
                    "target_product",
                    "week",
                    "source_promo_flag",
                    "y_control",
                    "cannibalization_quantity",
                ],
            )
            # source_promo_flag == 0 rows are counterfactual "what if promoted"
            # simulations, not observed events; keep only real promotion weeks.
            raw = raw[raw["source_promo_flag"] == 1].copy()
            frame = pd.DataFrame(
                {
                    "promoted_product": raw["source_product"].astype(int),
                    "affected_product": raw["target_product"].astype(int),
                    "week": raw["week"].astype(int),
                    "estimated_lost_sales": raw["cannibalization_quantity"].astype(float),
                }
            )
            # The S-Learner is a point-estimate regressor, not a probabilistic
            # classifier, so this is not a calibrated statistical probability;
            # it proxies confidence as the estimated share of the affected
            # product's counterfactual demand that was displaced.
            frame["probability_of_cannibalization"] = (
                frame["estimated_lost_sales"] / raw["y_control"].clip(lower=1e-6).to_numpy()
            ).clip(upper=1.0)
            frame["cannibalization_flag"] = (frame["estimated_lost_sales"] > 0).astype(int)
            self._cannibalization = frame
        return self._cannibalization

    @property
    def campaign_desc(self) -> pd.DataFrame:
        if self._campaign_desc is None:
            self._campaign_desc = pd.read_csv(self.data_dir / "campaign_desc.csv")
        return self._campaign_desc

    @property
    def coupon(self) -> pd.DataFrame:
        if self._coupon is None:
            self._coupon = pd.read_csv(
                self.data_dir / "coupon.csv", usecols=["PRODUCT_ID", "CAMPAIGN"]
            ).drop_duplicates()
        return self._coupon

    @property
    def products(self) -> pd.DataFrame:
        if self._products is None:
            self._products = pd.read_csv(self.data_dir / "product.csv")
        return self._products

    @property
    def day_week(self) -> pd.Series:
        """DAY -> WEEK_NO mapping derived from transactions (mode per day)."""
        if self._day_week is None:
            # DAY is not loaded in the cached transaction frame, so derive the
            # mapping from the raw CSV once.
            days = pd.read_csv(
                self.data_dir / "transaction_data.csv", usecols=["DAY", "WEEK_NO"]
            )
            series = days.groupby("DAY")["WEEK_NO"].agg(lambda s: s.mode().iloc[0])
            max_day = int(
                max(
                    days["DAY"].max(),
                    self.campaign_desc[["START_DAY", "END_DAY"]].to_numpy().max(),
                )
            )
            self._day_week = series.reindex(range(1, max_day + 1)).ffill().astype(int)
        return self._day_week

    # -------------------------------------------------------------- campaigns
    def list_campaigns(self) -> list[dict[str, Any]]:
        desc = self.campaign_desc.copy()
        desc["start_week"] = desc["START_DAY"].map(self.day_week).astype(int)
        desc["end_week"] = desc["END_DAY"].map(self.day_week).astype(int)
        counts = self.coupon.groupby("CAMPAIGN")["PRODUCT_ID"].nunique()
        campaigns: list[dict[str, Any]] = []
        for row in desc.itertuples(index=False):
            campaigns.append(
                {
                    "campaign_id": int(row.CAMPAIGN),
                    "description": str(row.DESCRIPTION),
                    "start_day": int(row.START_DAY),
                    "end_day": int(row.END_DAY),
                    "start_week": int(row.start_week),
                    "end_week": int(row.end_week),
                    "n_products": int(counts.get(row.CAMPAIGN, 0)),
                }
            )
        return campaigns

    def get_campaign(self, campaign_id: int) -> dict[str, Any] | None:
        desc = self.campaign_desc
        row = desc[desc["CAMPAIGN"] == campaign_id]
        if row.empty:
            return None
        row = row.iloc[0]
        start_week = int(self.day_week.get(int(row["START_DAY"]), row["START_DAY"]))
        end_week = int(self.day_week.get(int(row["END_DAY"]), row["END_DAY"]))
        product_ids = sorted(
            int(pid) for pid in self.coupon.loc[self.coupon["CAMPAIGN"] == campaign_id, "PRODUCT_ID"]
        )
        return {
            "campaign_id": int(campaign_id),
            "description": str(row.get("DESCRIPTION", "")),
            "start_day": int(row["START_DAY"]),
            "end_day": int(row["END_DAY"]),
            "start_week": start_week,
            "end_week": end_week,
            "n_products": len(product_ids),
            "products": product_ids,
        }

    def campaign_weeks(self, campaign_id: int) -> tuple[int, int] | None:
        campaign = self.get_campaign(campaign_id)
        if campaign is None:
            return None
        return campaign["start_week"], campaign["end_week"]

    def resolve_period(
        self,
        campaign_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> tuple[int, int]:
        """Resolve the analysis window. Explicit weeks win, then the campaign
        window, then the baseline test window (where evidence exists)."""
        if start_week is not None and end_week is not None:
            return int(start_week), int(end_week)
        if campaign_id is not None:
            weeks = self.campaign_weeks(campaign_id)
            if weeks is not None:
                start, end = weeks
                return int(start_week) if start_week is not None else int(start), \
                    int(end_week) if end_week is not None else int(end)
        return int(TEST_WINDOW[0]), int(TEST_WINDOW[1])

    # ----------------------------------------------------------------- baseline
    def product_baseline(
        self, product_id: int, start_week: int | None = None, end_week: int | None = None
    ) -> dict[str, Any]:
        start, end = self.resolve_period(start_week=start_week, end_week=end_week)
        panel = self.panel
        rows = panel[(panel["PRODUCT_ID"] == product_id) & panel["WEEK_NO"].between(start, end)]
        if rows.empty:
            return {
                "product_id": int(product_id),
                "start_week": start,
                "end_week": end,
                "data_sufficient": False,
                "confidence": "low",
                "message": LOW_DATA_MESSAGE,
                "baseline_qty": 0.0,
                "n_weeks_scored": 0,
                "n_stores": 0,
                "panel_stores": [],
                "history_weeks": 0,
                "weekly": [],
            }

        features = self.baseline.features
        prediction = self.baseline.predict(rows[features])
        scored = rows[["WEEK_NO"]].copy()
        scored["baseline"] = prediction["baseline_prediction"].to_numpy()

        weekly = (
            scored.groupby("WEEK_NO", as_index=False)["baseline"].sum()
            .sort_values("WEEK_NO")
        )
        history_weeks = int(
            panel[(panel["PRODUCT_ID"] == product_id) & (panel["WEEK_NO"] <= TRAIN_END) & (panel["qty"] > 0)]
            ["WEEK_NO"].nunique()
        )
        if history_weeks >= MIN_HISTORY_WEEKS_HIGH:
            confidence = "high"
        elif history_weeks >= MIN_HISTORY_WEEKS_MEDIUM:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "product_id": int(product_id),
            "start_week": start,
            "end_week": end,
            "data_sufficient": True,
            "confidence": confidence,
            "message": "" if confidence != "low" else LOW_BASELINE_CONFIDENCE_MESSAGE,
            "baseline_qty": float(scored["baseline"].sum()),
            "n_weeks_scored": int(scored["WEEK_NO"].nunique()),
            "n_stores": int(rows["STORE_ID"].nunique()),
            "panel_stores": [int(store) for store in rows["STORE_ID"].unique()],
            "history_weeks": history_weeks,
            "weekly": _jsonable(weekly.to_dict(orient="records")),
        }

    def actual_sales(
        self,
        product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
        stores: list[int] | None = None,
    ) -> float:
        start, end = self.resolve_period(start_week=start_week, end_week=end_week)
        frame = self.transactions
        mask = (frame["PRODUCT_ID"] == product_id) & frame["WEEK_NO"].between(start, end)
        if stores is not None:
            mask &= frame["STORE_ID"].isin(stores)
        quantity = frame.loc[mask, "QUANTITY"].sum()
        return float(quantity)

    # ------------------------------------------------------- promotion effect
    def campaign_effect(
        self,
        campaign_id: int,
        product_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if campaign is None:
            return {"error": f"Campaign {campaign_id} not found."}

        start, end = self.resolve_period(campaign_id, start_week, end_week)
        # The baseline panel stops at week 101; keep the comparison aligned.
        end = min(end, int(self.panel["WEEK_NO"].max()))
        panel_products = set(self.panel["PRODUCT_ID"].unique())

        if product_id is not None:
            targets = [int(product_id)]
        else:
            targets = [int(pid) for pid in campaign["products"] if int(pid) in panel_products]

        baselines = [self.product_baseline(pid, start, end) for pid in targets]
        available = [b for b in baselines if b["data_sufficient"]]
        if not available:
            return {
                "campaign_id": int(campaign_id),
                "product_id": product_id,
                "start_week": start,
                "end_week": end,
                "data_sufficient": False,
                "confidence": "low",
                "message": LOW_DATA_MESSAGE,
                "actual_sales": 0.0,
                "baseline_sales": 0.0,
                "incremental_sales": 0.0,
                "uplift_percentage": None,
                "effectiveness_summary": LOW_DATA_MESSAGE,
                "weekly": [],
            }

        baseline_sales = sum(b["baseline_qty"] for b in available)
        # Actual sales are restricted to the panel stores so the comparison is
        # on the same basis as the baseline model's counterfactual.
        actual_sales = sum(
            self.actual_sales(b["product_id"], start, end, stores=b.get("panel_stores"))
            for b in available
        )
        actual_sales_total = sum(self.actual_sales(b["product_id"], start, end) for b in available)
        incremental = actual_sales - baseline_sales
        uplift = (incremental / baseline_sales * 100.0) if baseline_sales > 0 else None
        confidence = min((b["confidence"] for b in available), key=lambda c: {"high": 0, "medium": 1, "low": 2}[c])

        if incremental > 0:
            summary = f"Promotion increased expected sales by {incremental:.0f} units"
            summary += f" (+{uplift:.0f}%)." if uplift is not None else "."
        elif incremental < 0:
            summary = f"Promotion decreased expected sales by {abs(incremental):.0f} units"
            summary += f" ({uplift:.0f}%)." if uplift is not None else "."
        else:
            summary = "Promotion had no measurable incremental effect."
        if confidence == "low":
            summary = f"{LOW_BASELINE_CONFIDENCE_MESSAGE} {summary}"

        # Weekly actual vs baseline for the promoted product(s)
        panel_stores = [store for b in available for store in b.get("panel_stores", [])]
        weekly_actual = (
            self.transactions.loc[
                self.transactions["PRODUCT_ID"].isin([b["product_id"] for b in available])
                & self.transactions["WEEK_NO"].between(start, end)
                & self.transactions["STORE_ID"].isin(panel_stores)
            ]
            .groupby("WEEK_NO")["QUANTITY"].sum()
        )
        weekly_baseline = pd.DataFrame()
        for b in available:
            frame = pd.DataFrame(b["weekly"])
            frame["product_id"] = b["product_id"]
            weekly_baseline = pd.concat([weekly_baseline, frame], ignore_index=True)
        weekly_baseline = (
            weekly_baseline.groupby("WEEK_NO")["baseline"].sum() if not weekly_baseline.empty else pd.Series(dtype=float)
        )
        weeks = sorted(set(weekly_actual.index) | set(weekly_baseline.index))
        weekly = [
            {
                "week": int(week),
                "actual": float(weekly_actual.get(week, 0.0)),
                "baseline": float(weekly_baseline.get(week, 0.0)),
            }
            for week in weeks
        ]

        return {
            "campaign_id": int(campaign_id),
            "product_id": product_id,
            "start_week": start,
            "end_week": end,
            "data_sufficient": True,
            "confidence": confidence,
            "message": "" if confidence != "low" else LOW_BASELINE_CONFIDENCE_MESSAGE,
            "actual_sales": round(actual_sales, 2),
            "actual_sales_total": round(actual_sales_total, 2),
            "baseline_sales": round(baseline_sales, 2),
            "incremental_sales": round(incremental, 2),
            "uplift_percentage": round(uplift, 2) if uplift is not None else None,
            "effectiveness_summary": summary,
            "n_products_analyzed": len(available),
            "weekly": weekly,
        }

    def incremental_sales(
        self,
        campaign_id: int,
        product_id: int | None = None,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        effect = self.campaign_effect(campaign_id, product_id, start_week, end_week)
        if "error" in effect:
            return effect
        return {
            "campaign_id": effect["campaign_id"],
            "product_id": effect["product_id"],
            "start_week": effect["start_week"],
            "end_week": effect["end_week"],
            "actual_sales": effect["actual_sales"],
            "baseline_sales": effect["baseline_sales"],
            "incremental_sales": effect["incremental_sales"],
            "uplift_percentage": effect["uplift_percentage"],
            "confidence": effect["confidence"],
            "data_sufficient": effect["data_sufficient"],
            "message": effect["message"],
        }

    # ---------------------------------------------------------- cannibalization
    def _cannibalization_rows(
        self, promoted_product_id: int, start_week: int | None, end_week: int | None
    ) -> pd.DataFrame:
        start, end = self.resolve_period(start_week=start_week, end_week=end_week)
        predictions = self.cannibalization_predictions
        return predictions[
            (predictions["promoted_product"] == promoted_product_id)
            & predictions["week"].between(start, end)
        ]

    def list_promoted_products(self) -> list[dict[str, Any]]:
        """Products with cannibalization evidence, sorted by lost quantity."""
        predictions = self.cannibalization_predictions
        grouped = (
            predictions.groupby("promoted_product", as_index=False)
            .agg(
                total_lost=("estimated_lost_sales", "sum"),
                n_flagged_weeks=("cannibalization_flag", "sum"),
                n_affected_products=("affected_product", "nunique"),
            )
            .sort_values("total_lost", ascending=False)
        )
        return [
            {
                "product_id": int(row["promoted_product"]),
                "total_lost": round(float(row["total_lost"]), 2),
                "n_flagged_weeks": int(row["n_flagged_weeks"]),
                "n_affected_products": int(row["n_affected_products"]),
            }
            for row in grouped.to_dict(orient="records")
        ]

    def cannibalization_effect(
        self,
        promoted_product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
    ) -> dict[str, Any]:
        rows = self._cannibalization_rows(promoted_product_id, start_week, end_week)
        start, end = self.resolve_period(start_week=start_week, end_week=end_week)

        if rows.empty:
            return {
                "promoted_product_id": int(promoted_product_id),
                "start_week": start,
                "end_week": end,
                "signal": "none",
                "message": NO_CANNIBALIZATION_MESSAGE,
                "summary": NO_CANNIBALIZATION_MESSAGE,
                "total_lost_quantity": 0.0,
                "cannibalization_percentage": 0.0,
                "n_flagged_weeks": 0,
                "affected_products": [],
                "top_impacted_products": [],
            }

        by_product = (
            rows.groupby("affected_product", as_index=False)
            .agg(
                lost_quantity=("estimated_lost_sales", "sum"),
                n_weeks=("week", "nunique"),
                mean_probability=("probability_of_cannibalization", "mean"),
                n_flagged=("cannibalization_flag", "sum"),
            )
            .sort_values("lost_quantity", ascending=False)
        )
        affected = [
            {
                "product_id": int(row["affected_product"]),
                "lost_quantity": round(float(row["lost_quantity"]), 2),
                "n_weeks": int(row["n_weeks"]),
                "mean_probability": round(float(row["mean_probability"]), 4),
            }
            for row in by_product.to_dict(orient="records")
            if float(row["lost_quantity"]) >= MIN_AFFECTED_LOST
        ]

        total_lost = float(rows["estimated_lost_sales"].sum())
        n_flagged = int((rows["cannibalization_flag"] == 1).sum())
        promoted_actual = self.actual_sales(int(promoted_product_id), start, end)
        cannibalization_percentage = (
            total_lost / max(1.0, promoted_actual) * 100.0
        )

        if n_flagged >= 1 and total_lost >= MIN_CANNIBALIZATION_LOST:
            signal = "strong"
            message = ""
        elif total_lost > 0:
            signal = "weak"
            message = NO_CANNIBALIZATION_MESSAGE
        else:
            signal = "none"
            message = NO_CANNIBALIZATION_MESSAGE
            affected = []

        return {
            "promoted_product_id": int(promoted_product_id),
            "start_week": start,
            "end_week": end,
            "signal": signal,
            "message": message,
            "summary": self._cannibalization_summary(int(promoted_product_id), signal, affected),
            "total_lost_quantity": round(total_lost, 2),
            "cannibalization_percentage": round(cannibalization_percentage, 2),
            "n_flagged_weeks": n_flagged,
            "promoted_actual_quantity": round(promoted_actual, 2),
            "affected_products": affected,
            "top_impacted_products": affected[:10],
        }

    @staticmethod
    def _cannibalization_summary(
        promoted_product_id: int, signal: str, affected: list[dict[str, Any]]
    ) -> str:
        if signal == "strong" and affected:
            top = affected[0]
            summary = (
                f"Product {promoted_product_id} promotion caused estimated loss of "
                f"{top['lost_quantity']:.0f} units in Product {top['product_id']}."
            )
            others = [p["product_id"] for p in affected[1:4]]
            if others:
                summary += " Other affected products: " + ", ".join(
                    f"Product {pid}" for pid in others
                ) + "."
            return summary
        if signal == "weak":
            return "Weak cannibalization signal; no strong effect was detected."
        return NO_CANNIBALIZATION_MESSAGE

    def top_impacted_products(
        self,
        promoted_product_id: int,
        start_week: int | None = None,
        end_week: int | None = None,
        top_n: int = 10,
    ) -> dict[str, Any]:
        result = self.cannibalization_effect(promoted_product_id, start_week, end_week)
        return {
            "promoted_product_id": result["promoted_product_id"],
            "start_week": result["start_week"],
            "end_week": result["end_week"],
            "top_impacted_products": result["top_impacted_products"][: max(1, int(top_n))],
        }
