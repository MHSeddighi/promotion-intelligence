"""Hybrid prediction-unit helpers for the baseline demand engine (MVP).

The engine forecasts **high-frequency** products at `PRODUCT_ID x STORE_ID x
WEEK_NO` and **sparse/long-tail** products at `CLUSTER_ID x STORE_ID x WEEK_NO`
units, then reconstructs PRODUCT_ID-level predictions with smoothed
training-period contribution shares.

Leakage contract
----------------
Every statistic derived here (demand frequency, segmentation, cluster
embeddings, cluster assignment, allocation shares) is computed **only from the
training-period data the caller passes in**. No validation/test information is
ever used by these helpers.

Design notes
------------
- Demand frequency is measured at PRODUCT x WEEK granularity (demand in any
  store counts as an active week), because store-week rows are overwhelmingly
  zero even for healthy products.
- Clustering uses KMeans over metadata embeddings concatenated with
  standardized training-window demand behavior (no deep learning).
- Allocation shares are Laplace-smoothed so every member of a cluster receives
  a strictly positive, normalized share of cluster demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Demand-frequency segmentation (training window only)
# ---------------------------------------------------------------------------


def demand_frequency(
    panel: pd.DataFrame,
    train_end: int,
    product_col: str = "PRODUCT_ID",
    week_col: str = "WEEK_NO",
    qty_col: str = "qty",
    promo_col: str = "promo_week",
) -> pd.DataFrame:
    """Per-product demand frequency from non-promoted training weeks only.

    Frequency is measured at PRODUCT x WEEK granularity: a product-week counts
    as active when demand is positive in *any* store.
    """
    train = panel[(panel[week_col] <= train_end) & (panel[promo_col] == 0)]
    pw = train.groupby([product_col, week_col])[qty_col].sum().reset_index()
    freq = pw.groupby(product_col).agg(
        train_weeks=(week_col, "nunique"),
        positive_weeks=(qty_col, lambda s: int((s > 0).sum())),
    )
    freq["positive_week_share"] = freq["positive_weeks"] / freq["train_weeks"]
    return freq.reset_index()


def segment_products(
    freq: pd.DataFrame,
    high_frequency_share_min: float = 0.5,
    high_frequency_min_positive_weeks: int = 20,
    product_col: str = "PRODUCT_ID",
) -> pd.DataFrame:
    """Assign every product to a prediction unit.

    - ``high_frequency``: enough active demand history -> keeps its
      PRODUCT_ID-level unit (forecast with the existing product model).
    - ``sparse``: insufficient individual demand observations -> moved to
      cluster-level units downstream.
    """
    hf = (
        (freq["positive_week_share"] >= high_frequency_share_min)
        & (freq["positive_weeks"] >= high_frequency_min_positive_weeks)
    )
    seg = pd.DataFrame(
        {
            product_col: freq[product_col].to_numpy(),
            "positive_week_share": freq["positive_week_share"].to_numpy(),
            "positive_weeks": freq["positive_weeks"].to_numpy(),
            "segment": np.where(hf, "high_frequency", "sparse"),
        }
    )
    return seg


# ---------------------------------------------------------------------------
# 2. Sparse product clustering (metadata + demand behavior embeddings)
# ---------------------------------------------------------------------------

DEMAND_BEHAVIOR_COLS = ["avg_demand", "trend", "vol", "avg_price"]


def choose_n_clusters(
    n_sparse: int,
    products_per_cluster: int = 20,
    min_clusters: int = 6,
    max_clusters: int = 80,
) -> int:
    """Deterministic cluster-count heuristic from the sparse-arm size."""
    if n_sparse <= 0:
        return 0
    if n_sparse == 1:
        return 1
    return int(np.clip(round(n_sparse / products_per_cluster), min_clusters, max_clusters))


def cluster_sparse_products(
    meta_embeddings: np.ndarray,
    demand_stats: pd.DataFrame,
    n_clusters: int,
    seed: int = 42,
    behavior_cols: list[str] | None = None,
) -> tuple[np.ndarray, KMeans, StandardScaler]:
    """Cluster sparse products from metadata embeddings + demand behavior.

    Parameters
    ----------
    meta_embeddings:
        Dense metadata embeddings, one row per sparse product (same order as
        ``demand_stats``). Must be derived without test-period information.
    demand_stats:
        Training-window demand behavior per sparse product
        (``avg_demand``, ``trend``, ``vol``, ``avg_price``).
    """
    if behavior_cols is None:
        behavior_cols = DEMAND_BEHAVIOR_COLS
    behav = demand_stats[behavior_cols].to_numpy(dtype="float64")
    for j, col in enumerate(behavior_cols):
        if col in ("avg_demand", "vol"):  # stabilize skewed demand/volatility
            behav[:, j] = np.log1p(np.maximum(behav[:, j], 0.0))
    behav = np.nan_to_num(behav, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(behav)
    X = np.hstack([meta_embeddings, scaler.transform(behav)]).astype("float32")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans, scaler


# ---------------------------------------------------------------------------
# 3. Cluster-level panel (CLUSTER_ID x STORE_ID x WEEK_NO)
# ---------------------------------------------------------------------------


def build_cluster_panel(
    sparse_panel: pd.DataFrame,
    assignment: pd.DataFrame,
    weeks: list[int],
    product_col: str = "PRODUCT_ID",
    store_col: str = "STORE_ID",
    week_col: str = "WEEK_NO",
    qty_col: str = "qty",
    promo_col: str = "promo_week",
) -> pd.DataFrame:
    """Aggregate sparse PRODUCT x STORE x WEEK rows into CLUSTER x STORE x WEEK.

    ``qty`` is summed over cluster members, ``promo_week`` is the max, and the
    result is a continuous weekly grid (like the product panel) so the same
    lag/rolling feature functions can be applied per cluster-store unit.
    """
    p = sparse_panel.merge(assignment, on=product_col, how="left")
    p = p.dropna(subset=["CLUSTER_ID"])
    p["CLUSTER_ID"] = p["CLUSTER_ID"].astype("int32")
    weekly = (
        p.groupby(["CLUSTER_ID", store_col, week_col], as_index=False)
        .agg(qty=(qty_col, "sum"), promo_week=(promo_col, "max"))
    )
    pairs = weekly[["CLUSTER_ID", store_col]].drop_duplicates()
    grid = pd.MultiIndex.from_arrays(
        [
            np.repeat(pairs["CLUSTER_ID"].to_numpy(), len(weeks)),
            np.repeat(pairs[store_col].to_numpy(), len(weeks)),
            np.tile(np.asarray(weeks, dtype="int16"), len(pairs)),
        ],
        names=["CLUSTER_ID", store_col, week_col],
    )
    cpanel = weekly.set_index(["CLUSTER_ID", store_col, week_col]).reindex(grid, fill_value=0).reset_index()
    cpanel[promo_col] = cpanel[promo_col].astype("int8")
    return cpanel


# ---------------------------------------------------------------------------
# 4. Reconstruction layer (cluster -> PRODUCT_ID allocation)
# ---------------------------------------------------------------------------


def allocation_shares(
    train: pd.DataFrame,
    assignment: pd.DataFrame,
    smooth_lambda: float = 0.5,
    product_col: str = "PRODUCT_ID",
    store_col: str = "STORE_ID",
    qty_col: str = "qty",
    cluster_col: str = "CLUSTER_ID",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Laplace-smoothed product contribution shares within each cluster x store.

    Computed from non-promoted **training rows only**. Returns
    (shares, shares_cluster):

    - ``shares``: one row per (CLUSTER_ID, STORE_ID, PRODUCT_ID) with a
      positive ``share``; shares sum to 1 per (CLUSTER_ID, STORE_ID).
    - ``shares_cluster``: store-agnostic fallback per (CLUSTER_ID, PRODUCT_ID).

    Laplace smoothing guarantees every cluster member (even a product with zero
    training demand) receives a strictly positive, normalized share.
    """
    t = train.merge(assignment, on=product_col, how="left").dropna(subset=[cluster_col])
    t = t.groupby([cluster_col, store_col, product_col], as_index=False)[qty_col].sum()

    def _laplace(group: pd.Series) -> pd.Series:
        total = group.sum()
        return (group + smooth_lambda) / (total + smooth_lambda * len(group))

    t["share"] = t.groupby([cluster_col, store_col])[qty_col].transform(_laplace)
    shares = t.drop(columns=[qty_col])

    t_cluster = t.groupby([cluster_col, product_col], as_index=False)[qty_col].sum()
    t_cluster["share_cluster"] = t_cluster.groupby(cluster_col)[qty_col].transform(_laplace)
    shares_cluster = t_cluster.drop(columns=[qty_col])
    return shares, shares_cluster


def reconstruct_sparse(
    cluster_pred: pd.DataFrame,
    shares: pd.DataFrame,
    shares_cluster: pd.DataFrame,
    cluster_col: str = "CLUSTER_ID",
    store_col: str = "STORE_ID",
    week_col: str = "WEEK_NO",
    product_col: str = "PRODUCT_ID",
    pred_col: str = "pred_cluster",
) -> pd.DataFrame:
    """Allocate cluster-level predictions back to PRODUCT_ID rows.

    ``cluster_pred`` must have one row per (CLUSTER_ID, STORE_ID, WEEK_NO) with
    the column named ``pred_col``. Returns one row per (PRODUCT_ID, STORE_ID,
    WEEK_NO) with the product-level prediction in ``pred_product``.
    """
    pred = cluster_pred.merge(shares, on=[cluster_col, store_col], how="left")
    pred = pred.merge(shares_cluster, on=[cluster_col, product_col], how="left")
    pred["share"] = pred["share"].fillna(pred["share_cluster"])
    pred = pred.drop(columns=["share_cluster"])
    pred["pred_product"] = pred[pred_col] * pred["share"]
    return pred[[product_col, store_col, week_col, "pred_product"]]


# ---------------------------------------------------------------------------
# 5. Sparse-demand validation metrics (original vs hybrid)
# ---------------------------------------------------------------------------


def segment_metrics(
    y_true,
    y_pred,
    threshold: float = 0.3,
    products=None,
) -> dict:
    """Sparse-demand metrics for one prediction series.

    Detection (non-zero demand): precision, recall, F1 plus confusion counts.
    Quantity (rows where actual demand > 0): WMAPE, MAE, RMSE, bias.
    """
    y = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    actual_positive = y > 0
    predicted_positive = pred > threshold

    tn, fp, fn, tp = confusion_matrix(actual_positive, predicted_positive, labels=[False, True]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    roc = roc_auc_score(actual_positive, pred) if actual_positive.any() and (~actual_positive).any() else np.nan
    pr = average_precision_score(actual_positive, pred) if actual_positive.any() else np.nan

    out = {
        "rows": int(len(y)),
        "products": int(pd.Series(products).nunique()) if products is not None else np.nan,
        "actual_zero_%": float((~actual_positive).mean() * 100),
        "actual_nonzero_%": float(actual_positive.mean() * 100),
        "pred_nonzero_%": float(predicted_positive.mean() * 100),
        "TN_zero_correct": int(tn),
        "FP_false_alarm": int(fp),
        "FN_missed_demand": int(fn),
        "TP_detected_demand": int(tp),
        "zero_precision": float(tn / (tn + fn) if (tn + fn) else 0.0),
        "zero_recall": float(tn / (tn + fp) if (tn + fp) else 0.0),
        "nonzero_precision": float(precision),
        "nonzero_recall": float(recall),
        "nonzero_F1": float(f1),
        "ROC_AUC": float(roc),
        "PR_AUC": float(pr),
    }

    positive_errors = pred[actual_positive] - y[actual_positive]
    out["positive_rows"] = int(actual_positive.sum())
    if len(positive_errors) > 0:
        out["positive_MAE"] = float(np.abs(positive_errors).mean())
        out["positive_RMSE"] = float(np.sqrt(np.mean(positive_errors**2)))
        out["positive_WMAPE"] = float(np.abs(positive_errors).sum() / y[actual_positive].sum())
        out["positive_bias"] = float(positive_errors.mean())
    else:
        out.update(
            {
                "positive_MAE": np.nan,
                "positive_RMSE": np.nan,
                "positive_WMAPE": np.nan,
                "positive_bias": np.nan,
            }
        )
    return out


METRIC_COLS = [
    "rows", "products", "actual_zero_%", "actual_nonzero_%", "pred_nonzero_%",
    "TN_zero_correct", "FP_false_alarm", "FN_missed_demand", "TP_detected_demand",
    "zero_precision", "zero_recall", "nonzero_precision", "nonzero_recall", "nonzero_F1",
    "ROC_AUC", "PR_AUC", "positive_rows", "positive_MAE", "positive_RMSE",
    "positive_WMAPE", "positive_bias",
]


def comparison_report(
    df: pd.DataFrame,
    qty_col: str = "qty",
    pred_original: str = "pred_orig_final",
    pred_hybrid: str = "pred_final",
    segment_col: str = "segment",
    product_col: str = "PRODUCT_ID",
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Original-vs-hybrid sparse-demand metrics, one row per segment x model.

    Segments are the distinct values of ``segment_col`` plus an ``ALL`` row;
    each segment is evaluated for both the original and the hybrid prediction
    series.
    """
    rows = []

    def add(segment: str, group: pd.DataFrame) -> None:
        for model, col in (("original", pred_original), ("hybrid", pred_hybrid)):
            m = segment_metrics(group[qty_col], group[col], threshold=threshold, products=group[product_col])
            rows.append({"segment": segment, "model": model, **m})

    add("ALL", df)
    segments = [s for s in pd.unique(df[segment_col].dropna()) if str(s) != "ALL"]
    for segment in segments:
        add(str(segment), df[df[segment_col] == segment])
    return pd.DataFrame(rows)[["segment", "model", *METRIC_COLS]]
