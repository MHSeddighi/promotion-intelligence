"""Unit tests for the hybrid prediction-unit stages (app/models/hybrid_units.py)."""

import numpy as np
import pandas as pd
import pytest

from app.models.hybrid_units import (
    DEMAND_BEHAVIOR_COLS,
    allocation_shares,
    build_cluster_panel,
    choose_n_clusters,
    cluster_sparse_products,
    comparison_report,
    demand_frequency,
    reconstruct_sparse,
    segment_metrics,
    segment_products,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def make_panel():
    """6-week panel, 4 products x 2 stores.

    - products 1, 2: active in most weeks (high frequency)
    - products 3, 4: active in a single week (sparse/long-tail)
    """
    weeks = [1, 2, 3, 4, 5, 6]
    rows = []
    for pid in (1, 2, 3, 4):
        for store in (10, 20):
            for week in weeks:
                qty = 0.0
                if pid in (1, 2) and week <= 5:
                    qty = float(pid * 10 + store + week)
                if pid == 3 and week == 4 and store == 10:
                    qty = 7.0
                if pid == 4 and week == 5 and store == 20:
                    qty = 5.0
                promo = 1 if (pid, week) == (1, 3) else 0
                rows.append({"PRODUCT_ID": pid, "STORE_ID": store, "WEEK_NO": week,
                             "qty": qty, "promo_week": promo})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Segmentation
# ---------------------------------------------------------------------------


def test_demand_frequency_uses_train_window_only():
    panel = make_panel()
    freq = demand_frequency(panel, train_end=5)
    assert list(freq.columns) == ["PRODUCT_ID", "train_weeks", "positive_weeks", "positive_week_share"]
    # product 2 active in weeks 1-5 -> 5/5 = 1.0; product 4 active only in week 5 (store 20)
    by_pid = freq.set_index("PRODUCT_ID")
    assert by_pid.loc[2, "positive_week_share"] == pytest.approx(1.0)
    assert by_pid.loc[4, "positive_week_share"] == pytest.approx(0.2)
    # leakage guard: only weeks <= train_end may contribute
    assert by_pid["train_weeks"].max() <= 5


def test_demand_frequency_excludes_promoted_weeks():
    panel = make_panel()
    freq = demand_frequency(panel, train_end=6)
    by_pid = freq.set_index("PRODUCT_ID")
    # product 1 has a promoted week 3 (excluded) and active weeks 1,2,4,5 -> 4/5
    assert by_pid.loc[1, "positive_week_share"] == pytest.approx(0.8)


def test_segment_products_binary_split():
    panel = make_panel()
    freq = demand_frequency(panel, train_end=5)
    seg = segment_products(freq, high_frequency_share_min=0.5, high_frequency_min_positive_weeks=3)
    by_pid = seg.set_index("PRODUCT_ID")
    assert by_pid.loc[1, "segment"] == "high_frequency"
    assert by_pid.loc[2, "segment"] == "high_frequency"
    assert by_pid.loc[3, "segment"] == "sparse"
    assert by_pid.loc[4, "segment"] == "sparse"


# ---------------------------------------------------------------------------
# 2. Clustering
# ---------------------------------------------------------------------------


def test_choose_n_clusters_bounds():
    assert choose_n_clusters(n_sparse=100, products_per_cluster=20, min_clusters=1) == 5
    assert choose_n_clusters(n_sparse=5000, products_per_cluster=20, max_clusters=80) == 80
    assert choose_n_clusters(n_sparse=10, products_per_cluster=20, min_clusters=6) == 6
    assert choose_n_clusters(n_sparse=0) == 0
    assert choose_n_clusters(n_sparse=1) == 1


def test_cluster_sparse_products_deterministic_and_scaled():
    rng = np.random.RandomState(0)
    meta = rng.normal(size=(30, 8)).astype("float32")
    stats = pd.DataFrame(
        {
            "avg_demand": rng.exponential(3.0, 30),
            "trend": rng.normal(0.0, 0.5, 30),
            "vol": rng.exponential(2.0, 30),
            "avg_price": rng.uniform(1.0, 20.0, 30),
        }
    )
    labels1, kmeans, scaler = cluster_sparse_products(meta, stats, n_clusters=5, seed=42)
    labels2, _, _ = cluster_sparse_products(meta, stats, n_clusters=5, seed=42)
    assert np.array_equal(labels1, labels2)  # deterministic
    assert len(np.unique(labels1)) == 5
    assert kmeans.n_clusters == 5
    # demand stats were standardized (transformed data has ~zero mean, unit std);
    # the module log1p's only the skewed demand/volatility columns
    raw = stats[DEMAND_BEHAVIOR_COLS].to_numpy().copy()
    raw[:, 0] = np.log1p(np.maximum(raw[:, 0], 0.0))
    raw[:, 2] = np.log1p(np.maximum(raw[:, 2], 0.0))
    scaled = scaler.transform(raw)
    assert np.allclose(scaled.mean(axis=0), np.zeros(4), atol=1e-6)
    assert np.allclose(scaled.std(axis=0), np.ones(4), atol=1e-6)


def test_cluster_sparse_products_separates_by_behavior():
    rng = np.random.RandomState(1)
    meta = np.zeros((40, 4), dtype="float32")  # identical metadata -> behavior drives clusters
    stats = pd.DataFrame(
        {
            "avg_demand": np.concatenate([np.full(20, 100.0), np.full(20, 1.0)]),
            "trend": np.zeros(40),
            "vol": np.ones(40),
            "avg_price": np.full(40, 10.0),
        }
    )
    labels, _, _ = cluster_sparse_products(meta, stats, n_clusters=2, seed=7)
    assert set(labels[:20]) == {0} or set(labels[:20]) == {1}
    assert set(labels[20:]) == (set(labels) - set(labels[:20]))


# ---------------------------------------------------------------------------
# 3. Cluster panel
# ---------------------------------------------------------------------------


def test_build_cluster_panel_aggregates_qty_and_grid():
    panel = make_panel()
    assignment = pd.DataFrame({"PRODUCT_ID": [3, 4], "CLUSTER_ID": [0, 0]})
    cpanel = build_cluster_panel(panel, assignment, weeks=list(range(1, 7)))
    assert list(cpanel.columns) == ["CLUSTER_ID", "STORE_ID", "WEEK_NO", "qty", "promo_week"]
    # continuous grid: 1 cluster x 2 stores x 6 weeks
    assert len(cpanel) == 12
    assert set(cpanel["STORE_ID"]) == {10, 20}
    assert set(cpanel["WEEK_NO"]) == {1, 2, 3, 4, 5, 6}
    # qty is summed over members at week 4/store 10: 7 + 0 = 7
    row = cpanel[(cpanel["STORE_ID"] == 10) & (cpanel["WEEK_NO"] == 4)]
    assert row["qty"].iloc[0] == pytest.approx(7.0)
    # no promotion weeks in the sparse fixture -> promo_week all zero
    assert cpanel["promo_week"].eq(0).all()


def test_build_cluster_panel_drops_unassigned_products():
    panel = make_panel()
    assignment = pd.DataFrame({"PRODUCT_ID": [3], "CLUSTER_ID": [0]})
    cpanel = build_cluster_panel(panel, assignment, weeks=[1, 2, 3, 4])
    # only product 3 forms the cluster (product 4 ignored), at both its stores
    assert set(cpanel["STORE_ID"]) == {10, 20}
    assert len(cpanel) == 8


# ---------------------------------------------------------------------------
# 4. Reconstruction / allocation
# ---------------------------------------------------------------------------


def test_allocation_shares_positive_and_partition_of_unity():
    train = pd.DataFrame(
        {
            "PRODUCT_ID": [3, 4, 4],
            "STORE_ID": [10, 10, 20],
            "qty": [7.0, 0.0, 5.0],
        }
    )
    assignment = pd.DataFrame({"PRODUCT_ID": [3, 4], "CLUSTER_ID": [0, 0]})
    shares, shares_cluster = allocation_shares(train, assignment, smooth_lambda=0.5)
    assert {"CLUSTER_ID", "STORE_ID", "PRODUCT_ID", "share"}.issubset(shares.columns)
    assert (shares["share"] > 0).all()
    # shares partition to 1 per (cluster, store), even for zero-qty members
    total = shares.groupby(["CLUSTER_ID", "STORE_ID"])["share"].sum()
    assert np.allclose(total, 1.0)
    assert np.allclose(shares_cluster.groupby("CLUSTER_ID")["share_cluster"].sum(), 1.0)
    # store 10: qty 7 + 0 -> Laplace shares 7.5/8 and 0.5/8
    s10 = shares[shares["STORE_ID"] == 10].set_index("PRODUCT_ID")
    assert s10.loc[3, "share"] == pytest.approx(7.5 / 8.0)
    assert s10.loc[4, "share"] == pytest.approx(0.5 / 8.0)
    # cluster-level fallback: qty 7 + 5 -> shares 7.5/13 and 5.5/13
    sc = shares_cluster.set_index("PRODUCT_ID")
    assert sc.loc[3, "share_cluster"] == pytest.approx(7.5 / 13.0)
    assert sc.loc[4, "share_cluster"] == pytest.approx(5.5 / 13.0)


def test_reconstruct_sparse_preserves_cluster_total():
    cluster_pred = pd.DataFrame(
        {
            "CLUSTER_ID": [0, 0, 0],
            "STORE_ID": [10, 10, 20],
            "WEEK_NO": [4, 5, 5],
            "pred_cluster": [14.0, 10.0, 8.0],
        }
    )
    shares = pd.DataFrame(
        {
            "CLUSTER_ID": [0, 0, 0, 0],
            "STORE_ID": [10, 10, 20, 20],
            "PRODUCT_ID": [3, 4, 3, 4],
            "share": [0.6, 0.4, 0.5, 0.5],
        }
    )
    shares_cluster = pd.DataFrame(
        {"CLUSTER_ID": [0], "PRODUCT_ID": [3], "share_cluster": [0.5]}
    )
    out = reconstruct_sparse(cluster_pred, shares, shares_cluster)
    assert list(out.columns) == ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "pred_product"]
    # (store, week) totals are preserved after allocation
    totals = out.groupby(["STORE_ID", "WEEK_NO"])["pred_product"].sum()
    expected = cluster_pred.groupby(["STORE_ID", "WEEK_NO"])["pred_cluster"].sum()
    pd.testing.assert_series_equal(totals, expected, check_names=False)
    # fallback: product 3 at store 20 uses the cluster-level share 0.5
    row = out[(out["PRODUCT_ID"] == 3) & (out["STORE_ID"] == 20) & (out["WEEK_NO"] == 5)]
    assert row["pred_product"].iloc[0] == pytest.approx(8.0 * 0.5)


# ---------------------------------------------------------------------------
# 5. Sparse-demand metrics
# ---------------------------------------------------------------------------


def test_segment_metrics_detection_and_quantity():
    y = np.array([0.0, 0.0, 4.0, 6.0])
    pred = np.array([0.1, 2.0, 3.0, 8.0])
    m = segment_metrics(y, pred, threshold=0.5, products=[1, 1, 2, 2])
    assert m["rows"] == 4
    assert m["products"] == 2
    # confusion: actual [F,F,T,T], predicted [F,T,T,T] -> tn=1, fp=1, fn=0, tp=2
    assert m["TN_zero_correct"] == 1
    assert m["FP_false_alarm"] == 1
    assert m["FN_missed_demand"] == 0
    assert m["TP_detected_demand"] == 2
    assert m["nonzero_precision"] == pytest.approx(2 / 3)
    assert m["nonzero_recall"] == pytest.approx(1.0)
    assert m["nonzero_F1"] == pytest.approx(0.8)
    # quantity on positive rows: errors -1, +2 -> MAE 1.5, RMSE sqrt(2.5), WMAPE 3/10, bias 0.5
    assert m["positive_MAE"] == pytest.approx(1.5)
    assert m["positive_RMSE"] == pytest.approx(np.sqrt(2.5))
    assert m["positive_WMAPE"] == pytest.approx(0.3)
    assert m["positive_bias"] == pytest.approx(0.5)


def test_comparison_report_rows_and_columns():
    df = pd.DataFrame(
        {
            "PRODUCT_ID": [1, 1, 2, 2, 3, 3],
            "segment": ["long_tail"] * 4 + ["fast"] * 2,
            "qty": [0.0, 4.0, 0.0, 6.0, 5.0, 5.0],
            "pred_orig_final": [0.0, 2.0, 1.0, 5.0, 4.0, 4.0],
            "pred_final": [0.0, 3.0, 1.0, 6.0, 5.0, 5.0],
        }
    )
    rep = comparison_report(df, threshold=0.5)
    # segments: ALL + long_tail + fast, each with original + hybrid
    assert set(rep["model"]) == {"original", "hybrid"}
    assert set(rep["segment"]) == {"ALL", "long_tail", "fast"}
    assert len(rep) == 6
    assert "nonzero_precision" in rep.columns
    assert "positive_WMAPE" in rep.columns
    # hybrid strictly better on fast segment MAE (perfect predictions)
    fast = rep[(rep["segment"] == "fast")]
    hyb = fast[fast["model"] == "hybrid"]["positive_MAE"].iloc[0]
    orig = fast[fast["model"] == "original"]["positive_MAE"].iloc[0]
    assert hyb == pytest.approx(0.0)
    assert orig > hyb
