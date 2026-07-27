import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List, Optional, Union
import warnings

warnings.filterwarnings("ignore")

TARGET_COLUMN = "QUANTITY"

META_COLUMNS = [
    "DAY",
    "PRODUCT_ID",
    "WEEK_NO",
]

CATEGORICAL_COLUMNS = [
    "PRODUCT_ID",
    "BRAND",
    "COMMODITY_DESC",
    "SUB_COMMODITY_DESC",
    "DEPARTMENT",
    "MANUFACTURER",
    "CURR_SIZE_OF_PRODUCT",
]

NUMERICAL_FEATURE_COLUMNS = [
    "SALES_VALUE",
    "RETAIL_DISC",
    "COUPON_DISC",
    "BASKET_COUNT",
    "WEEK_NO",
    "discount_amount",
    "has_discount",
    "discount_ratio",
    "day_of_week",
    "month",
    "quarter",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_sum_7",
    "rolling_sum_14",
    "rolling_sum_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_max_28",
    "rolling_std_28",
    "days_since_last_sale",
    "sales_days_last_30",
    "sales_days_last_60",
    "sales_days_last_90",
    "product_age_days",
    "sales_frequency_30",
    "sales_frequency_90",
    "prev_total_quantity",
    "prev_active_days",
    "prev_avg_quantity",
    "prev_avg_sales_value",
]

DROP_COLUMNS = [
    "COUPON_MATCH_DISC",
    "RETAIL_DISC",
    "COUPON_DISC",
    "SALES_VALUE",
    "BASKET_COUNT",
    "avg_price",
    "total_quantity",
    "active_weeks",
    "avg_weekly_quantity",
    "category_frequency",
    "days_until_next_sale",
]


def validate_dataframe(df: pd.DataFrame, stage_name: str) -> None:
    print(f"\n[{stage_name}]")
    print(f"  Shape: {df.shape}")

    if df.empty:
        warnings.warn(f"  WARNING: DataFrame is empty!")
        return

    missing_cols = []
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 90:
            missing_cols.append(f"{col}({missing_pct:.1f}%)")

    if missing_cols:
        warnings.warn(f"  High missing: {', '.join(missing_cols)}")

    dups = df.duplicated().sum()
    if dups:
        print(f"  Duplicated rows: {dups}")

    for col in df.columns:
        if df[col].nunique() <= 1:
            warnings.warn(f"  Column '{col}' is constant")
        elif pd.api.types.is_numeric_dtype(df[col]):
            if (df[col] == 0).all():
                warnings.warn(f"  Column '{col}' contains only zeros")


def aggregate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "QUANTITY": "sum",
        "SALES_VALUE": "sum",
        "RETAIL_DISC": "sum",
        "COUPON_DISC": "sum",
        "BASKET_ID": "nunique",
        "WEEK_NO": "first",
        "BRAND": "first",
        "COMMODITY_DESC": "first",
        "DEPARTMENT": "first",
        "MANUFACTURER": "first",
        "SUB_COMMODITY_DESC": "first",
        "CURR_SIZE_OF_PRODUCT": "first",
    }

    available = {k: v for k, v in agg_dict.items() if k in df.columns}

    sales = df.groupby(["DAY", "PRODUCT_ID"], as_index=False).agg(available)

    sales.rename(columns={"BASKET_ID": "BASKET_COUNT"}, inplace=True)

    validate_dataframe(sales, "AGGREGATION")
    return sales


def create_full_date_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DAY"] = pd.to_datetime(df["DAY"])

    products = df["PRODUCT_ID"].unique()
    date_range = pd.date_range(
        start=df["DAY"].min(),
        end=df["DAY"].max(),
        freq="D"
    )

    product_idx = pd.MultiIndex.from_product(
        [date_range, products],
        names=["DAY", "PRODUCT_ID"]
    )

    full_grid = pd.DataFrame(index=product_idx).reset_index()

    product_info = df.groupby("PRODUCT_ID").first().reset_index()
    product_cols = ["PRODUCT_ID"] + [
        c for c in ["BRAND", "COMMODITY_DESC", "DEPARTMENT",
                    "MANUFACTURER", "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT"]
        if c in product_info.columns
    ]

    full_grid = full_grid.merge(
        product_info[product_cols],
        on="PRODUCT_ID",
        how="left"
    )

    merge_cols = ["DAY", "PRODUCT_ID"]
    value_cols = [
        c for c in ["QUANTITY", "SALES_VALUE", "RETAIL_DISC",
                    "COUPON_DISC", "BASKET_COUNT", "WEEK_NO"]
        if c in df.columns
    ]

    full_grid = full_grid.merge(
        df[merge_cols + value_cols],
        on=merge_cols,
        how="left"
    )

    for col in value_cols:
        if col != "WEEK_NO":
            full_grid[col] = full_grid[col].fillna(0)

    full_grid["WEEK_NO"] = full_grid.groupby("PRODUCT_ID")["WEEK_NO"].ffill()

    if "DEPARTMENT" in full_grid.columns:
        full_grid["DEPARTMENT"] = full_grid.groupby("PRODUCT_ID")["DEPARTMENT"].ffill()

    validate_dataframe(full_grid, "FULL DATE GRID")
    return full_grid


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["DAY"]):
        df["DAY"] = pd.to_datetime(df["DAY"])

    df["day_of_week"] = df["DAY"].dt.dayofweek
    df["month"] = df["DAY"].dt.month
    df["quarter"] = df["DAY"].dt.quarter
    df["week_of_year"] = df["DAY"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    validate_dataframe(df, "CALENDAR FEATURES")
    return df


def create_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "RETAIL_DISC" in df.columns and "COUPON_DISC" in df.columns:
        df["discount_amount"] = df["RETAIL_DISC"].abs() + df["COUPON_DISC"].abs()
    elif "RETAIL_DISC" in df.columns:
        df["discount_amount"] = df["RETAIL_DISC"].abs()
    elif "COUPON_DISC" in df.columns:
        df["discount_amount"] = df["COUPON_DISC"].abs()
    else:
        df["discount_amount"] = 0

    df["has_discount"] = (df["discount_amount"] > 0).astype(int)

    df["discount_ratio"] = 0.0
    if "SALES_VALUE" in df.columns:
        mask = df["SALES_VALUE"] > 0
        df.loc[mask, "discount_ratio"] = (
                df.loc[mask, "discount_amount"] / df.loc[mask, "SALES_VALUE"]
        )
        df["discount_ratio"] = df["discount_ratio"].clip(0, 1).fillna(0)

    validate_dataframe(df, "PROMOTION FEATURES")
    return df


def create_time_aware_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["PRODUCT_ID", "DAY"])

    lags = [1, 7, 14, 28]

    df = df.set_index("DAY")

    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = (
            df.groupby("PRODUCT_ID")["QUANTITY"]
            .shift(lag, freq="D")
            .fillna(0)
            .values
        )

    df = df.reset_index()

    print(f"[LAGS] Created: lag_1, lag_7, lag_14, lag_28")
    validate_dataframe(df, "LAG FEATURES")
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["PRODUCT_ID", "DAY"])

    windows = [7, 14, 28]

    df = df.set_index("DAY")

    shifted = df.groupby("PRODUCT_ID")["QUANTITY"].shift(1, freq="D")

    for window in windows:
        rolling = shifted.groupby("PRODUCT_ID").rolling(
            f"{window}D", min_periods=1
        )

        df[f"rolling_sum_{window}"] = rolling.sum().fillna(0).values
        df[f"rolling_mean_{window}"] = rolling.mean().fillna(0).values

        if window == 28:
            df[f"rolling_max_28"] = rolling.max().fillna(0).values
            df[f"rolling_std_28"] = rolling.std().fillna(0).values

    df = df.reset_index()

    print(f"[ROLLING] Created: sum/mean (7,14,28), max/std (28)")
    validate_dataframe(df, "ROLLING FEATURES")
    return df


def create_sparse_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["PRODUCT_ID", "DAY"])

    df = df.set_index("DAY")

    sale_mask = df["QUANTITY"] > 0

    df["last_sale_date"] = (
        df.groupby("PRODUCT_ID")["DAY"]
        .transform(lambda x: x.where(sale_mask).ffill().shift(1))
    )

    df["days_since_last_sale"] = (
        (df.index - df["last_sale_date"]).dt.days.fillna(9999)
    )

    df["next_sale_date"] = (
        df.groupby("PRODUCT_ID")["DAY"]
        .transform(lambda x: x.where(sale_mask).bfill().shift(-1))
    )

    df["days_until_next_sale"] = (
        (df["next_sale_date"] - df.index).dt.days.fillna(9999)
    )

    for window in [30, 60, 90]:
        rolled = (
            sale_mask.astype(int)
            .groupby("PRODUCT_ID")
            .rolling(f"{window}D", min_periods=1)
            .sum()
        )
        df[f"sales_days_last_{window}"] = rolled.fillna(0).values

    df["first_sale_date"] = (
        df.groupby("PRODUCT_ID")["DAY"]
        .transform(lambda x: x.where(sale_mask).cummin())
    )

    df["product_age_days"] = (
        (df.index - df["first_sale_date"]).dt.days.fillna(0)
    )

    df["sales_frequency_30"] = df["sales_days_last_30"] / 30
    df["sales_frequency_90"] = df["sales_days_last_90"] / 90

    df = df.drop(columns=["last_sale_date", "next_sale_date", "first_sale_date"], errors="ignore")

    df = df.reset_index()

    print(f"[SPARSE] Created: days_since_last_sale, sales_days, product_age, frequencies")
    validate_dataframe(df, "SPARSE DEMAND FEATURES")
    return df


def create_historical_product_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["PRODUCT_ID", "DAY"])

    df = df.set_index("DAY")

    shifted_qty = df.groupby("PRODUCT_ID")["QUANTITY"].shift(1, freq="D")
    shifted_sales = df.groupby("PRODUCT_ID")["SALES_VALUE"].shift(1, freq="D")
    shifted_mask = (shifted_qty > 0).astype(int)

    for label, series, agg_func in [
        ("total_quantity", shifted_qty, "sum"),
        ("active_days", shifted_mask, "sum"),
        ("avg_quantity", shifted_qty, "mean"),
        ("avg_sales_value", shifted_sales, "mean"),
    ]:
        col_name = f"prev_{label}"
        df[col_name] = (
            series.groupby("PRODUCT_ID")
            .expanding()
            .agg(agg_func)
            .fillna(0)
            .values
        )

    df = df.reset_index()

    print(f"[HISTORICAL] Created: prev_total_quantity, prev_active_days, prev_avg_quantity, prev_avg_sales_value")
    validate_dataframe(df, "HISTORICAL PRODUCT FEATURES")
    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            warnings.warn(f"Categorical column '{col}' not found")
            continue

        if df[col].nunique() > 1000:
            warnings.warn(f"High cardinality: '{col}' has {df[col].nunique()} unique values")

        df[col] = df[col].fillna("UNKNOWN").astype(str)

        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    print(f"[ENCODED] {list(encoders.keys())}")
    validate_dataframe(df, "CATEGORICAL ENCODING")
    return df, encoders


def remove_bad_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dropped = []

    protected = META_COLUMNS + [TARGET_COLUMN]

    for col in df.columns:
        if col in protected:
            continue

        if df[col].isna().all():
            df.drop(columns=[col], inplace=True)
            dropped.append(f"{col} (all NaN)")
        elif df[col].nunique() <= 1:
            df.drop(columns=[col], inplace=True)
            dropped.append(f"{col} (constant)")
        elif pd.api.types.is_numeric_dtype(df[col]):
            if (df[col] == 0).all():
                df.drop(columns=[col], inplace=True)
                dropped.append(f"{col} (all zeros)")

    if dropped:
        print(f"\n[DROPPED FEATURES]")
        for d in dropped:
            print(f"  - {d}")

    validate_dataframe(df, "FEATURE CLEANING")
    return df


def build_features(
        df: pd.DataFrame,
        use_full_grid: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    List[str],
    List[str],
    List[str],
    Dict,
]:
    validate_dataframe(df, "RAW DATA")

    df = df.drop(
        columns=[c for c in DROP_COLUMNS if c in df.columns],
        errors="ignore"
    )

    df = aggregate_transactions(df)

    if use_full_grid:
        df = create_full_date_grid(df)

    df = create_calendar_features(df)
    df = create_promotion_features(df)
    df = create_time_aware_lags(df)
    df = create_rolling_features(df)
    df = create_sparse_demand_features(df)
    df = create_historical_product_features(df)

    df, encoders = encode_categoricals(df)

    df = remove_bad_features(df)

    df = df.sort_values(["PRODUCT_ID", "DAY"]).reset_index(drop=True)

    meta_cols = [c for c in META_COLUMNS if c in df.columns]

    feature_cols = [
        c for c in df.columns
        if c != TARGET_COLUMN and c not in meta_cols
    ]

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    meta = df[meta_cols].copy()

    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    temporal_cols = [c for c in feature_cols if "lag" in c or "rolling" in c]
    sparse_cols = [c for c in feature_cols if
                   "days_since" in c or "sales_days" in c or "sales_frequency" in c or "product_age" in c]
    hist_cols = [c for c in feature_cols if c.startswith("prev_")]

    print("\n" + "=" * 60)
    print("FINAL DATASET REPORT")
    print("-" * 60)
    print(f"Rows:                  {len(df):,}")
    print(f"Total Columns:         {len(df.columns)}")
    print(f"Target:                {TARGET_COLUMN}")
    print(f"Feature Columns:       {len(feature_cols)}")
    print(f"  - Numerical:         {len(num_cols)}")
    print(f"  - Categorical:       {len(cat_cols)}")
    print(f"  - Temporal (lags):   {len(temporal_cols)}")
    print(f"  - Sparse demand:     {len(sparse_cols)}")
    print(f"  - Historical:        {len(hist_cols)}")
    print(f"Meta Columns:          {len(meta_cols)}")
    print(f"Encoded Categories:    {len(encoders)}")
    print("=" * 60 + "\n")

    return df, X, y, meta, feature_cols, cat_cols, num_cols, encoders
