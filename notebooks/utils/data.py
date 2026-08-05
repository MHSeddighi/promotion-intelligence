"""
Shared data loading utilities for all notebooks.

Loads raw CSV files from data/raw/ and provides
common preprocessing helpers that avoid code duplication.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_transactions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "transaction_data.csv")


def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "product.csv")


def load_campaign_desc() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "campaign_desc.csv")


def load_campaign_table() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "campaign_table.csv")


def load_coupons() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "coupon.csv")


def load_coupon_redemptions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "coupon_redempt.csv")


def load_causal_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "causal_data.csv")


def load_hh_demographic() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "hh_demographic.csv")

def aggregate_daily_sales(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate transaction data to daily level."""
    return df.groupby("DAY", as_index=False).agg(
        sales=("SALES_VALUE", "sum"),
        quantity=("QUANTITY", "sum"),
        discount=("RETAIL_DISC", lambda x: abs(x).sum()),
        coupon_disc=("COUPON_DISC", lambda x: abs(x).sum()),
        transactions=("BASKET_ID", "nunique"),
    )


def get_product_sales_aggregated(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Merge and aggregate to (DAY, PRODUCT_ID, DEPARTMENT, COMMODITY_DESC) level."""
    df = merge_sales_with_products(transactions, products)
    agg = df.groupby(
        ["DAY", "PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"], as_index=False
    ).agg(
        quantity=("QUANTITY", "sum"),
        sales_value=("SALES_VALUE", "sum"),
        retail_disc=("RETAIL_DISC", "sum"),
        coupon_disc=("COUPON_DISC", "sum"),
        basket_count=("BASKET_ID", "nunique"),
        week_no=("WEEK_NO", "first"),
    )
    return agg


# =============================================================================
# MVP layer: typed loaders, promotion aggregates and campaign helpers.
# These build on the loaders above and are shared by notebooks 04-09.
# =============================================================================

from dataclasses import dataclass
import logging

TX_DTYPES = {
    "household_key": "int32", "BASKET_ID": "int64", "DAY": "int16",
    "PRODUCT_ID": "int32", "QUANTITY": "int32", "SALES_VALUE": "float32",
    "STORE_ID": "int16", "RETAIL_DISC": "float32", "TRANS_TIME": "int32",
    "WEEK_NO": "int16", "COUPON_DISC": "float32", "COUPON_MATCH_DISC": "float32",
}

CAUSAL_DTYPES = {
    "PRODUCT_ID": "int32", "STORE_ID": "int16", "WEEK_NO": "int16",
    "display": "category", "mailer": "category",
}


def find_project_root(start=None) -> Path:
    """Walk up from cwd until the repo's data/raw directory is found."""
    from pathlib import Path as _P
    start = _P(start or __file__).resolve()
    for p in [start, *start.parents]:
        if (p / "data" / "raw" / "transaction_data.csv").exists():
            return p
    return start.parents[0]


def load_transactions_fast(path: Path = None) -> pd.DataFrame:
    """Load transactions with explicit dtypes (fast, low memory)."""
    return pd.read_csv(path or DATA_DIR / "transaction_data.csv", dtype=TX_DTYPES)


def load_causal_data_typed(path: Path = None) -> pd.DataFrame:
    """Load causal (display/mailer) data with explicit dtypes."""
    return pd.read_csv(path or DATA_DIR / "causal_data.csv", dtype=CAUSAL_DTYPES)


def load_or_build_causal_weekly(
    cache_path: Path = None,
    force: bool = False,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """Return product-week display/mailer promotion intensity.

    Builds once from the raw 36.8M-row causal_data.csv and caches the
    product-week aggregate as parquet so downstream notebooks stay fast.
    """
    cache_path = Path(cache_path or PROJECT_ROOT / "data" / "processed" / "causal_weekly.parquet")
    log = logger or logging.getLogger(__name__)
    if cache_path.exists() and not force:
        log.info("Loading cached causal_weekly from %s", cache_path)
        return pd.read_parquet(cache_path)
    log.info("Building causal_weekly from raw causal_data.csv (this can take ~30s)")
    raw = load_causal_data_typed()
    raw["display_on"] = raw["display"].astype(str).ne("0").astype("int8")
    raw["mailer_on"] = raw["mailer"].astype(str).ne("0").astype("int8")
    weekly = (
        raw.groupby(["PRODUCT_ID", "WEEK_NO"], observed=True)
        [["display_on", "mailer_on"]]
        .mean()
        .rename(columns={"display_on": "display_share", "mailer_on": "mailer_share"})
        .reset_index()
    )
    del raw
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(cache_path, index=False)
    log.info("causal_weekly cached: %s rows -> %s", len(weekly), cache_path)
    return weekly


def campaign_days(campaign_desc: pd.DataFrame) -> pd.DataFrame:
    """Add duration_days and mid_day to a campaign description table."""
    df = campaign_desc.copy()
    df["duration_days"] = df["END_DAY"] - df["START_DAY"] + 1
    df["mid_day"] = (df["START_DAY"] + df["END_DAY"]) // 2
    return df


def is_campaign_day(day: int, campaign_desc: pd.DataFrame) -> bool:
    """True if a DAY falls inside any campaign window."""
    return bool(((campaign_desc["START_DAY"] <= day) & (campaign_desc["END_DAY"] >= day)).any())


def campaign_overlap_table(campaign_desc: pd.DataFrame) -> pd.DataFrame:
    """Pairwise overlap (in days) between campaigns."""
    rows = []
    for _, a in campaign_desc.iterrows():
        for _, b in campaign_desc.iterrows():
            if a["CAMPAIGN"] >= b["CAMPAIGN"]:
                continue
            lo = max(a["START_DAY"], b["START_DAY"])
            hi = min(a["END_DAY"], b["END_DAY"])
            overlap = max(0, hi - lo + 1)
            if overlap > 0:
                rows.append({
                    "campaign_a": int(a["CAMPAIGN"]), "campaign_b": int(b["CAMPAIGN"]),
                    "overlap_days": int(overlap),
                    "a_start": int(a["START_DAY"]), "a_end": int(a["END_DAY"]),
                    "b_start": int(b["START_DAY"]), "b_end": int(b["END_DAY"]),
                })
    return pd.DataFrame(rows)


def promoted_products_by_discount(
    transactions: pd.DataFrame,
    campaign_desc: pd.DataFrame,
) -> pd.DataFrame:
    """Product x campaign promotion instances derived from discount activity.

    A product is considered promoted inside a campaign window when it has
    price-discounted transactions (RETAIL_DISC < 0) in that window.
    """
    disc = transactions.loc[transactions["RETAIL_DISC"] < 0,
                            ["PRODUCT_ID", "DAY", "RETAIL_DISC", "SALES_VALUE", "QUANTITY"]]
    frames = []
    for _, camp in campaign_desc.iterrows():
        win = disc[(disc["DAY"] >= camp["START_DAY"]) & (disc["DAY"] <= camp["END_DAY"])]
        if win.empty:
            continue
        agg = win.groupby("PRODUCT_ID", as_index=False).agg(
            promo_days=("DAY", "nunique"),
            promo_quantity=("QUANTITY", "sum"),
            promo_sales=("SALES_VALUE", "sum"),
            promo_discount=("RETAIL_DISC", lambda s: -s.sum()),
        )
        agg["CAMPAIGN"] = int(camp["CAMPAIGN"])
        agg["START_DAY"] = int(camp["START_DAY"])
        agg["END_DAY"] = int(camp["END_DAY"])
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["PRODUCT_ID", "promo_days", "promo_quantity", "promo_sales",
                 "promo_discount", "CAMPAIGN", "START_DAY", "END_DAY"])
    return out.sort_values(["CAMPAIGN", "promo_sales"], ascending=[True, False]).reset_index(drop=True)
