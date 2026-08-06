# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cannibalization Detection Model — Causal Treatment-Effect (S-Learner)
#
# Estimates the **causal impact of promoting product A on the sales quantity of product B** with an S-Learner treatment-effect model. The previous weak-label approach (deriving `cannibalization_amount = max(0, baseline_B − actual_B)`) is **replaced** by a counterfactual comparison:
#
# ```
# Cannibalization(A, B) =
#     Predicted B sales when A is not promoted
#     − Predicted B sales when A is promoted
# ```
#
# ```
# Promoted Product A ──► embedding relationship ──► Candidate affected Product B
#                              │
#                              └──► S-Learner:  E[qty_B | features, source_promo_flag]
#                                           ├── y_control   (source_promo_flag = 0)
#                                           └── y_treatment (source_promo_flag = 1)
#                                     cannibalization = max(0, y_control − y_treatment)
# ```
#
# ## Pipeline
# 1. **Inputs** — transaction data, promotion/campaign data, product metadata, baseline model predictions (`outputs/baseline_engine/`), and the relationship outputs of `03_product_embeddings.py` (`outputs/product_embeddings/`).
# 2. **Candidate pairs** — **unchanged**: only pairs with embedding similarity, same category/brand, or basket substitution evidence (never all pairs).
# 3. **Pair-week dataset** — every `(source_product, affected_product, week)` row with the target `affected_product_quantity` (actual units) and source / affected / pair features, including the treatment variable `source_promo_flag`.
# 4. **S-Learner** — one LightGBM regression model `qty_B = f(all features + source_promo_flag)` that learns the conditional effect of a source promotion on the affected product's quantity.
# 5. **Treatment-effect estimation** — for every pair-week, predict with `source_promo_flag` forced to 0 (control) and 1 (treatment); `cannibalization_quantity = max(0, y_control − y_treatment)`.
# 6. **Aggregation** — product × product cannibalization matrix (sum / mean predicted cannibalized units).
# 7. **Validation** — time-based split only (train weeks 1–97, test weeks 98–101), prediction error on actual affected-product sales, SHAP explanation of which features drive cannibalization.
#
# **Fast smoke test:** set `CANNIBALIZATION_SMOKE=1` to run a ~10-second end-to-end validation (40 promoted products, 60 trees, small SHAP sample) that writes to `outputs/cannibalization_smoke/` without touching the full artifacts.
#
# ## Leakage strategy
# - The S-Learner is trained on **weeks 1-97** and evaluated on **weeks 98-101** (the baseline engine's held-out window is 89-101).
# - Relationship signals were learned on **weeks 1-88 only** (see notebook 1).
# - Static pair features (demand correlation, scale ratio, promotion frequency) use **weeks 1-78 only**.
# - Rolling demand features use only past weeks (`shift(1)` before rolling).
# - Raw baseline predictions are per-product calibrated on **non-promoted weeks 79-88** (outside the baseline training window 1-78).
# - Rows where the affected product is itself promoted are excluded (its own promotion confounds the A→B effect), and `qty_A` (a post-treatment variable) is deliberately **not** used as a model feature.
# - No weak labels are constructed from `baseline_sales − actual_sales`; the target is the observed affected-product quantity.

# %%
# 0. Environment check (no-op when packages are already installed)
from __future__ import annotations

import subprocess
import sys


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


missing = [pkg for mod, pkg in {"lightgbm": "lightgbm", "shap": "shap", "sklearn": "scikit-learn",
                                "matplotlib": "matplotlib", "seaborn": "seaborn",
                                "pyarrow": "pyarrow", "joblib": "joblib"}.items() if not _importable(mod)]
if missing:
    print("installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *set(missing)])

# %%
# 1. Imports + config
import json
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path.cwd().resolve()
if not (ROOT / "data" / "raw" / "transaction_data.csv").exists():
    ROOT = next(p for p in ROOT.parents if (p / "data" / "raw" / "transaction_data.csv").exists())
RAW = ROOT / "data" / "raw"
BASE = ROOT / "outputs" / "baseline_engine"      # baseline model + features (01_baseline_detection.py)
EMB = ROOT / "outputs" / "product_embeddings"    # embeddings + relationships (03_product_embeddings.py)

# Smoke-test mode: CANNIBALIZATION_SMOKE=1 runs a reduced version (fewer pairs,
# weeks-limited model, smaller SHAP sample) that finishes in seconds and writes
# to outputs/cannibalization_smoke/ so the full artifacts stay untouched.
SMOKE = os.environ.get("CANNIBALIZATION_SMOKE", "") == "1"
OUT = ROOT / "outputs" / ("cannibalization_smoke" if SMOKE else "cannibalization")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Time windows (kept in sync with the baseline engine)
FIRST_TEST_WEEK, LAST_TEST_WEEK = 89, 101   # baseline test window
TRAIN_END = 78                              # static pair features use weeks <= 78 only
RESID_END = 88                              # residual std / learning window used by notebook 1
MODEL_SPLIT_WEEK = 97                       # S-Learner: train weeks 1-97, test weeks 98-101

# Candidate-pair construction (unchanged from the embedding pipeline)
TOP_SUBS = 10            # substitutes per product from notebook 1
MAX_CANDIDATES = 12      # cap on candidates per promoted product
COSINE_MIN = 0.35        # min embedding cosine for same-subcommodity additions
EVIDENCE_COSINE = 0.45   # min cosine alone counts as substitution evidence

# Source-embedding features: first EMB_DIMS SVD components (ordered by variance)
EMB_DIMS = 8 if SMOKE else 64

# Baseline calibration (leakage-free): per-product rescale of raw baseline predictions
CALIB_LO, CALIB_HI = 79, 88   # holdout window the baseline engine never trained on (weeks 1-78)

SEED = 42
np.random.seed(SEED)
if SMOKE:
    print("SMOKE TEST MODE: reduced data/model for a fast end-to-end validation run")

# %% [markdown]
# ## 2. Load data
#
# Raw tables (same columns as the baseline notebook) plus the promotion and campaign tables: `causal_weekly.parquet` carries display/mailer shares per product-week; `coupon.csv` + `campaign_desc.csv` map products to campaign weeks and types.

# %%
# 2a. Transactions + product metadata
tx = pd.read_csv(RAW / "transaction_data.csv", usecols=["household_key", "BASKET_ID", "DAY", "PRODUCT_ID",
                                                        "QUANTITY", "SALES_VALUE", "STORE_ID", "RETAIL_DISC", "WEEK_NO"])
product = pd.read_csv(RAW / "product.csv")
print(f"transactions: {len(tx):,} | weeks {tx.WEEK_NO.min()}-{tx.WEEK_NO.max()}")

# 2b. Promotion intensity (display / mailer shares per product-week)
try:
    causal = pd.read_parquet(ROOT / "data" / "processed" / "causal_weekly.parquet")
    print("loaded causal_weekly.parquet:", causal.shape)
except FileNotFoundError:
    print("causal_weekly.parquet missing -> aggregating from raw causal_data.csv (may take a minute)")
    causal_raw = pd.read_csv(RAW / "causal_data.csv", usecols=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"])
    causal_raw["mailer_on"] = causal_raw["mailer"].astype(str).ne("0").astype(int)
    causal = (causal_raw.groupby(["PRODUCT_ID", "WEEK_NO"], as_index=False)
              .agg(display_share=("display", lambda s: (s > 0).mean()),
                   mailer_share=("mailer_on", "mean")))
    causal.to_parquet(ROOT / "data" / "processed" / "causal_weekly.parquet", index=False)
causal = causal.rename(columns={"display_share": "display", "mailer_share": "mailer"})

# 2c. Campaign data: day -> week mapping (derived from transactions), product -> campaign weeks + type
campaign_desc = pd.read_csv(RAW / "campaign_desc.csv")  # DESCRIPTION, CAMPAIGN, START_DAY, END_DAY
day_week = tx.groupby("DAY")["WEEK_NO"].agg(lambda s: s.mode().iloc[0])
max_day = int(max(tx["DAY"].max(), campaign_desc[["START_DAY", "END_DAY"]].to_numpy().max()))
day_week = day_week.reindex(range(1, max_day + 1)).ffill().astype(int)  # campaigns can run past the tx calendar
coupon = pd.read_csv(RAW / "coupon.csv", usecols=["PRODUCT_ID", "CAMPAIGN"]).drop_duplicates()

campaign_desc["start_week"] = campaign_desc["START_DAY"].map(day_week).astype(int)
campaign_desc["end_week"] = campaign_desc["END_DAY"].map(day_week).astype(int)
pc = coupon.merge(campaign_desc[["CAMPAIGN", "DESCRIPTION", "start_week", "end_week"]], on="CAMPAIGN")
pc = pc[(pc["start_week"] <= LAST_TEST_WEEK) & (pc["end_week"] >= FIRST_TEST_WEEK)]
n_weeks = (pc["end_week"] - pc["start_week"] + 1).to_numpy()
pc_weeks = np.concatenate([np.arange(s, e + 1) for s, e in zip(pc["start_week"], pc["end_week"])])
camp_prod = np.repeat(pc["PRODUCT_ID"].to_numpy(), n_weeks)
camp_type = np.repeat(pc["DESCRIPTION"].to_numpy(), n_weeks)
campaign_weekly = pd.DataFrame({"PRODUCT_ID": camp_prod, "WEEK_NO": pc_weeks, "TYPE": camp_type})
campaign_weekly["has_campaign"] = 1
campaign_weekly["TYPE_code"] = campaign_weekly["TYPE"].map({"TypeA": 1, "TypeB": 2, "TypeC": 3}).fillna(0)
campaign_weekly = (campaign_weekly.groupby(["PRODUCT_ID", "WEEK_NO"], as_index=False)
                   .agg(has_campaign=("has_campaign", "max"), n_campaigns=("TYPE", "count"),
                        campaign_type=("TYPE_code", "max")))
print(f"campaign coverage: {len(campaign_weekly):,} product-weeks | "
      f"campaigns {campaign_desc['CAMPAIGN'].nunique()}")

# %% [markdown]
# ## 3. Baseline demand predictions for every product × week
#
# Reuses the baseline engine's saved models and feature list to score the **full panel** (all weeks, including promoted ones). This gives a counterfactual `baseline_qty` for every product × store × week, which is then aggregated to product × week.

# %%
# 3. Score full baseline panel -> product x week baseline demand
fl = json.loads((BASE / "feature_list.json").read_text())
FEATURES = fl["features"]
CATS = fl["cat_cols_encoded"]

base_weekly_cache = OUT / "base_weekly.parquet"
full_cache = ROOT / "outputs" / "cannibalization" / "base_weekly.parquet"
if SMOKE and full_cache.exists():
    base_weekly = pd.read_parquet(full_cache)
    print("smoke: reused cached base_weekly from the full run:", base_weekly.shape)
else:
    panel = pd.read_parquet(BASE / "panel.parquet")
    X = panel[FEATURES].copy()
    for c in CATS:
        X[c] = X[c].astype("category")
    stage1 = joblib.load(BASE / "model_stage1.pkl")
    stage2 = joblib.load(BASE / "model_stage2.pkl")

    panel["baseline_qty"] = stage1.predict_proba(X)[:, 1] * stage2.predict(X)
    base_weekly = (panel.groupby(["PRODUCT_ID", "WEEK_NO"], as_index=False)["baseline_qty"].sum())
    base_weekly.to_parquet(OUT / "base_weekly.parquet", index=False)
    print("scored panel:", panel.shape, "| product-weeks:", base_weekly.shape,
          "| baseline test-week correlation vs saved test_predictions:")
    tp = pd.read_parquet(BASE / "test_predictions.parquet", columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "pred_final"])
    tp = tp.groupby(["PRODUCT_ID", "WEEK_NO"], as_index=False)["pred_final"].sum()
    chk = base_weekly.merge(tp, on=["PRODUCT_ID", "WEEK_NO"])
    print("   pearson corr =", round(chk["baseline_qty"].corr(chk["pred_final"]), 4), f"(n={len(chk):,})")
    del panel, X, chk

# %% [markdown]
# ## 4. Product × week demand / promotion table
#
# Aggregates actual demand, promotion flags, discount depth, unit price, display/mailer shares, campaign activity, past-only rolling demand, and per-product residual noise (from non-promoted weeks 1-88). All rows are restricted to the baseline panel products (the products we can forecast).

# %%
# 4. Product-week table for baseline-panel products
panel_products = pd.Index(base_weekly["PRODUCT_ID"].unique(), name="PRODUCT_ID")
txw = tx.assign(promo=(tx["RETAIL_DISC"] < 0))
txw["unit_price"] = txw["SALES_VALUE"] / txw["QUANTITY"].where(txw["QUANTITY"] > 0)
txw.loc[txw["unit_price"] < 0, "unit_price"] = np.nan

pw = txw[txw["PRODUCT_ID"].isin(panel_products)].groupby(["PRODUCT_ID", "WEEK_NO"], as_index=False).agg(
    qty=("QUANTITY", "sum"), promo=("promo", "max"), disc_min=("RETAIL_DISC", "min"),
    price=("unit_price", "mean"))
disc_depth = (txw[txw["promo"]][["PRODUCT_ID", "WEEK_NO", "RETAIL_DISC"]]
              .groupby(["PRODUCT_ID", "WEEK_NO"])["RETAIL_DISC"].mean().rename("disc_depth"))
pw = pw.merge(disc_depth, on=["PRODUCT_ID", "WEEK_NO"], how="left")
pw["disc_depth"] = (-pw["disc_depth"]).fillna(0.0).clip(lower=0.0)

pw = pw.merge(base_weekly, on=["PRODUCT_ID", "WEEK_NO"], how="left")
pw = pw.merge(causal, on=["PRODUCT_ID", "WEEK_NO"], how="left")
pw = pw.merge(campaign_weekly, on=["PRODUCT_ID", "WEEK_NO"], how="left")
for c in ["display", "mailer", "has_campaign", "n_campaigns", "campaign_type"]:
    pw[c] = pw[c].fillna(0)

# past-only rolling demand (shift(1) before rolling)
pw = pw.sort_values(["PRODUCT_ID", "WEEK_NO"]).reset_index(drop=True)
pw["qty_lag1"] = pw.groupby("PRODUCT_ID")["qty"].shift(1)
pw["rmean4"] = pw.groupby("PRODUCT_ID")["qty_lag1"].transform(lambda s: s.rolling(4, min_periods=1).mean())

# static per-product stats (weeks <= 78) and residual noise (non-promoted weeks 1-88)
pre = pw[pw["WEEK_NO"] <= TRAIN_END]
prod_stats = pre.groupby("PRODUCT_ID", as_index=False).agg(
    total_qty_78=("qty", "sum"), promo_freq=("promo", "mean"))
pw = pw.merge(prod_stats, on="PRODUCT_ID", how="left")

resid = pw[(pw["WEEK_NO"] <= RESID_END) & (pw["promo"] == 0)].copy()
resid["resid"] = resid["qty"] - resid["baseline_qty"]
resid_std = resid.groupby("PRODUCT_ID")["resid"].std().rename("resid_std")
pw = pw.merge(resid_std, on="PRODUCT_ID", how="left")
pw["resid_std"] = pw["resid_std"].fillna(resid["resid"].std())

# per-product calibration factor: sum(qty)/sum(baseline) on non-promoted holdout weeks 79-88
# (fallback: training window 1-78). Rescales the raw baseline to each product's demand level.
cal_hold = pw[(pw["WEEK_NO"].between(CALIB_LO, CALIB_HI)) & (pw["promo"] == 0)].groupby("PRODUCT_ID").agg(
    actual=("qty", "sum"), pred=("baseline_qty", "sum"))
cal_train = pw[(pw["WEEK_NO"] <= TRAIN_END) & (pw["promo"] == 0)].groupby("PRODUCT_ID").agg(
    actual=("qty", "sum"), pred=("baseline_qty", "sum"))
calib = pd.DataFrame({
    "factor": cal_hold["actual"].div(cal_hold["pred"]),
    "fallback": cal_train["actual"].div(cal_train["pred"]),
})
calib["factor"] = calib["factor"].fillna(calib["fallback"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
pw = pw.merge(calib["factor"].rename("calib_factor"), on="PRODUCT_ID", how="left")
pw["calib_factor"] = pw["calib_factor"].fillna(1.0)
pw["baseline_cal"] = pw["baseline_qty"] * pw["calib_factor"]

# missing unit prices -> product median (training window only, leakage-free)
price_med = pw[pw["WEEK_NO"] <= TRAIN_END].groupby("PRODUCT_ID")["price"].median()
pw["price"] = pw["price"].fillna(pw["PRODUCT_ID"].map(price_med))
pw["price"] = pw["price"].fillna(pw[pw["WEEK_NO"] <= TRAIN_END]["price"].median()).clip(lower=0.0)

pw = pw[pw["WEEK_NO"].between(1, LAST_TEST_WEEK)].reset_index(drop=True)
print("product-week rows:", f"{len(pw):,}", "| products:", pw["PRODUCT_ID"].nunique(),
      "| promoted product-weeks 89-101:", int(pw[(pw["promo"] == 1) & (pw["WEEK_NO"] >= FIRST_TEST_WEEK)].shape[0]))

# %% [markdown]
# ## 5. Candidate product pairs
#
# **Unchanged** — the embedding/candidate-pair pipeline from the previous version. For each baseline-panel product we take:
# - its top `TOP_SUBS` substitutes from notebook 1 (`top_k_substitutes.parquet`), and
# - same-sub-commodity panel products with embedding cosine >= `COSINE_MIN` (from `product_similarity.parquet`),
#
# then keep only pairs with **substitution evidence** (same department, basket co-purchase, or cosine >= `EVIDENCE_COSINE`) and cap at `MAX_CANDIDATES` per promoted product.

# %%
# 5a. Substitutes from notebook 1 (directed, ranked)
top = pd.read_parquet(EMB / "top_k_substitutes.parquet")
subs_frames = []
for r in range(1, TOP_SUBS + 1):
    f = pd.DataFrame({
        "promoted_product": top["PRODUCT_ID"],
        "affected_product": top[f"substitute_{r}"],
        "cosine_sim": top[f"substitute_similarity_{r}"],
        "substitute_score": top[f"substitute_score_{r}"],
        "substitute_rank": r,
    })
    subs_frames.append(f)
subs = pd.concat(subs_frames, ignore_index=True).dropna(subset=["affected_product"])
subs["affected_product"] = subs["affected_product"].astype(np.int64)

# 5b. Per-pair relationship signals from notebook 1
sim = pd.read_parquet(EMB / "product_similarity.parquet",
                      columns=["PRODUCT_ID", "SIMILAR_PRODUCT_ID", "embedding_cosine", "sub_commodity_match",
                               "commodity_match", "department_match", "brand_match", "manufacturer_match",
                               "metadata_similarity", "basket_jaccard", "basket_cosine",
                               "household_jaccard", "store_jaccard"])
sim_wide = sim.drop_duplicates(["PRODUCT_ID", "SIMILAR_PRODUCT_ID"]).rename(
    columns={"PRODUCT_ID": "promoted_product", "SIMILAR_PRODUCT_ID": "affected_product"})
subs = subs.merge(sim_wide, on=["promoted_product", "affected_product"], how="left")

# 5c. Same-sub-commodity additions (same category, decent cosine)
extra = sim_wide[(sim_wide["sub_commodity_match"] == 1) & (sim_wide["embedding_cosine"] >= COSINE_MIN)].copy()
extra["substitute_rank"] = TOP_SUBS + 1
extra["substitute_score"] = extra["embedding_cosine"]
subs = pd.concat([subs, extra], ignore_index=True).drop_duplicates(["promoted_product", "affected_product"])

# 5d. Evidence filter + panel restriction + cap
subs = subs[(subs["promoted_product"].isin(panel_products)) & (subs["affected_product"].isin(panel_products))
            & (subs["promoted_product"] != subs["affected_product"])]
evidence = (subs["department_match"] == 1) | (subs["basket_jaccard"].fillna(0) > 0) | (subs["cosine_sim"] >= EVIDENCE_COSINE)
subs = subs[evidence].copy()
subs = subs.sort_values(["promoted_product", "substitute_score"], ascending=[True, False])
subs["cand_rank"] = subs.groupby("promoted_product").cumcount() + 1
subs = subs[subs["cand_rank"] <= MAX_CANDIDATES].reset_index(drop=True)
if SMOKE:
    keep = subs["promoted_product"].unique()[:40]
    subs = subs[subs["promoted_product"].isin(keep)].reset_index(drop=True)
    print("smoke: restricted to", subs["promoted_product"].nunique(), "promoted products ->", len(subs), "pairs")
print("candidate pairs:", f"{len(subs):,}", "| promoted products:", subs["promoted_product"].nunique(),
      "| avg candidates/promoted product:", round(subs.groupby("promoted_product").size().mean(), 2))

# %% [markdown]
# ## 6. Static pair features (leakage-free)
#
# **Unchanged.** Relationship signals come from notebook 1 (learned on weeks 1-88). Demand correlation and sales-scale ratio are computed on **weeks 1-78 only** (the baseline training window), so they never see the evaluation period.

# %%
# 6. Static pair features
# 6a. Demand correlation over the training window (weeks 1-78)
q78 = pw[pw["WEEK_NO"] <= TRAIN_END].pivot(index="PRODUCT_ID", columns="WEEK_NO", values="qty").reindex(
    index=panel_products, columns=range(1, TRAIN_END + 1)).fillna(0)
Q = q78.to_numpy()
corr_mat = np.corrcoef(Q)
corr_mat[np.isnan(corr_mat)] = 0.0
pid_to_row = {p: i for i, p in enumerate(q78.index)}
subs["demand_corr"] = [corr_mat[pid_to_row[a], pid_to_row[b]]
                       for a, b in zip(subs["promoted_product"], subs["affected_product"])]

# 6b. Sales-scale ratio (log) from training-window totals
tot = q78.sum(axis=1).rename("total_qty_78")
subs = subs.merge(tot.rename("total_a"), left_on="promoted_product", right_index=True, how="left")
subs = subs.merge(tot.rename("total_b"), left_on="affected_product", right_index=True, how="left")
subs["sales_scale_log_ratio"] = np.log((subs["total_b"] + 1) / (subs["total_a"] + 1))
subs = subs.drop(columns=["total_a", "total_b"])

# 6c. Promotion frequency (weeks 1-78) per product
pfreq = pre.groupby("PRODUCT_ID")["promo"].mean()
subs = subs.merge(pfreq.rename("promo_freq_A"), left_on="promoted_product", right_index=True, how="left")
subs = subs.merge(pfreq.rename("promo_freq_B"), left_on="affected_product", right_index=True, how="left")

subs["cosine_sim"] = subs["cosine_sim"].fillna(0.0)
for c in ["basket_jaccard", "basket_cosine", "household_jaccard", "store_jaccard", "metadata_similarity",
          "sub_commodity_match", "commodity_match", "department_match", "brand_match", "manufacturer_match"]:
    subs[c] = subs[c].fillna(0)
print("static pair features:", subs.shape, "| mean cosine:", round(subs["cosine_sim"].mean(), 3))

# %% [markdown]
# ## 7. Source-product embedding features
#
# Each candidate pair carries the **source product's embedding** (first `EMB_DIMS` SVD components, ordered by variance from notebook 3) so the model can condition the promotion effect on *which* product is promoted. The embedding pipeline itself is unchanged; we only read its output.

# %%
# 7. Source-product embedding features
emb_cols = ["PRODUCT_ID"] + [f"dim_{i}" for i in range(EMB_DIMS)]
emb = pd.read_parquet(EMB / "product_embeddings.parquet", columns=emb_cols)
emb = emb.rename(columns={f"dim_{i}": f"src_emb_{i}" for i in range(EMB_DIMS)})
subs = subs.merge(emb, left_on="promoted_product", right_on="PRODUCT_ID", how="left").drop(columns=["PRODUCT_ID"])
n_emb_missing = int(subs["src_emb_0"].isna().sum())
for c in [f"src_emb_{i}" for i in range(EMB_DIMS)]:
    subs[c] = subs[c].fillna(0.0)
print(f"source embeddings: {EMB_DIMS} dims | pairs with embedding: {len(subs) - n_emb_missing:,}/{len(subs):,}")

# %% [markdown]
# ## 8. Pair × week dataset
#
# For every candidate pair and **every week** (1-101) one row is created. The target is the **observed affected-product quantity** (`qty_B`) — no weak labels are derived from `baseline − actual`. Features cover the source product (treatment flag, discount, price, baseline, embedding), the affected product (baseline prediction, historical quantity, price, category/brand context), and the pair (embedding/category/brand similarity, price ratio, historical correlation).
#
# Rows where the affected product is itself promoted are dropped (its own promotion confounds the A→B effect).

# %%
# 8. Pair × week feature table (target: affected product quantity)
weeks_all = pd.DataFrame({"WEEK_NO": range(1, LAST_TEST_WEEK + 1)})
df = subs.merge(weeks_all, how="cross")
print("raw pair-week rows (all weeks):", f"{len(df):,}")

pw_cols = ["PRODUCT_ID", "WEEK_NO", "qty", "promo", "disc_depth", "price", "display", "mailer",
           "has_campaign", "n_campaigns", "campaign_type", "qty_lag1", "rmean4", "baseline_cal", "resid_std"]
pwA = pw[pw_cols].rename(columns={c: c + "_A" for c in pw_cols if c not in ("PRODUCT_ID", "WEEK_NO")})
pwB = pw[pw_cols].rename(columns={c: c + "_B" for c in pw_cols if c not in ("PRODUCT_ID", "WEEK_NO")})
df = df.merge(pwA, left_on=["promoted_product", "WEEK_NO"], right_on=["PRODUCT_ID", "WEEK_NO"], how="left")
df = df.merge(pwB, left_on=["affected_product", "WEEK_NO"], right_on=["PRODUCT_ID", "WEEK_NO"], how="left")
df = df.drop(columns=["PRODUCT_ID_x", "PRODUCT_ID_y"])
df = df[df["promo_B"] == 0].reset_index(drop=True)  # affected product not promoted -> effect not confounded

df["source_promo_flag"] = df["promo_A"].fillna(0).astype(int)   # treatment variable (0/1)
df["price_ratio_log"] = np.log((df["price_B"] + 1e-3) / (df["price_A"] + 1e-3)).clip(-6, 6)
df["baseline_ratio_log"] = np.log((df["baseline_cal_B"] + 1) / (df["baseline_cal_A"] + 1))

# 8b. Competitive pressure: how many of B's (and A's) substitutes are promoted this week, and how deep
subs_B = subs[["promoted_product", "affected_product"]].rename(
    columns={"promoted_product": "B", "affected_product": "substitute"})
subs_A = subs[["promoted_product", "affected_product"]].rename(
    columns={"promoted_product": "A", "affected_product": "substitute"})
promo_pw = pw[pw["promo"] == 1][["PRODUCT_ID", "WEEK_NO", "disc_depth"]]


def substitute_pressure(sub_table, id_col):
    tmp = sub_table.merge(promo_pw, left_on="substitute", right_on="PRODUCT_ID", how="inner")
    return (tmp.groupby([id_col, "WEEK_NO"], as_index=False)
            .agg(**{f"n_sub_promos": ("disc_depth", "count"), f"sub_promo_disc": ("disc_depth", "mean")}))


press_B = substitute_pressure(subs_B, "B").rename(
    columns={"n_sub_promos": "n_sub_promos_B", "sub_promo_disc": "sub_promo_disc_B"})
press_A = substitute_pressure(subs_A, "A").rename(
    columns={"n_sub_promos": "n_sub_promos_A", "sub_promo_disc": "sub_promo_disc_A"})
df = df.merge(press_B, left_on=["affected_product", "WEEK_NO"], right_on=["B", "WEEK_NO"], how="left")
df = df.merge(press_A, left_on=["promoted_product", "WEEK_NO"], right_on=["A", "WEEK_NO"], how="left")
for c in ["n_sub_promos_B", "sub_promo_disc_B", "n_sub_promos_A", "sub_promo_disc_A"]:
    df[c] = df[c].fillna(0)

df["split"] = np.where(df["WEEK_NO"] <= MODEL_SPLIT_WEEK, "train", "test")
print("pair-week rows:", f"{len(df):,}", "| products:", df["promoted_product"].nunique(),
      "| train/test:", int((df["split"] == "train").sum()), "/", int((df["split"] == "test").sum()))
print("treatment rate (source promoted):", round(df["source_promo_flag"].mean(), 4),
      "| mean target qty_B:", round(df["qty_B"].mean(), 2))
df.to_parquet(OUT / "pair_features.parquet", index=False)

# %% [markdown]
# ## 9. S-Learner — conditional-effect regression
#
# A single LightGBM regressor learns
#
# ```
# qty_B = f(static pair features, source features, affected features, source embedding, source_promo_flag)
# ```
#
# The treatment variable `source_promo_flag` is included like any other feature, so the model can learn how a source promotion shifts the affected product's quantity conditional on all context. `qty_A` (the source's own contemporaneous sales) is **excluded** because it is a post-treatment variable that would bias the treatment effect.

# %%
# 9. S-Learner: one LightGBM regression of affected quantity on all features + treatment
STATIC_PAIR_FEATURES = ["cosine_sim", "sub_commodity_match", "commodity_match", "department_match",
                        "brand_match", "manufacturer_match", "metadata_similarity", "basket_jaccard",
                        "basket_cosine", "household_jaccard", "store_jaccard", "demand_corr",
                        "sales_scale_log_ratio", "substitute_rank", "promo_freq_A", "promo_freq_B",
                        "price_ratio_log", "baseline_ratio_log"]
SOURCE_FEATURES = ["source_promo_flag", "disc_depth_A", "price_A", "baseline_cal_A",
                   "rmean4_A", "display_A", "mailer_A", "has_campaign_A", "n_campaigns_A", "campaign_type_A",
                   "n_sub_promos_A", "sub_promo_disc_A"]
AFFECTED_FEATURES = ["qty_lag1_B", "rmean4_B", "price_B", "baseline_cal_B", "resid_std_B",
                     "display_B", "mailer_B", "has_campaign_B", "n_campaigns_B", "campaign_type_B",
                     "n_sub_promos_B", "sub_promo_disc_B"]
EMBEDDING_FEATURES = [f"src_emb_{i}" for i in range(EMB_DIMS)]
MODEL_FEATURES = STATIC_PAIR_FEATURES + SOURCE_FEATURES + AFFECTED_FEATURES + EMBEDDING_FEATURES

train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
print("train rows:", f"{len(train):,}", "| test rows:", f"{len(test):,}", "| features:", len(MODEL_FEATURES))

bad = [c for c in MODEL_FEATURES if not pd.api.types.is_numeric_dtype(train[c])]
assert not bad, f"non-numeric model features: {bad}"

t0 = time.time()
m = lgb.LGBMRegressor(n_estimators=60 if SMOKE else 500, learning_rate=0.05,
                      num_leaves=15 if SMOKE else 63, min_child_samples=10,
                      subsample=0.9, subsample_freq=1, colsample_bytree=1.0,
                      random_state=SEED, n_jobs=-1, verbose=-1)
m.fit(train[MODEL_FEATURES], train["qty_B"])
print(f"S-Learner trained in {time.time() - t0:.1f}s | trees: {m.n_estimators}")
imp = pd.Series(m.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False)
print("top 12 features by split gain:")
print(imp.head(12).to_string())
print("source_promo_flag gain rank:", int(imp.index.get_loc("source_promo_flag")) + 1,
      "| gain:", round(float(imp["source_promo_flag"]), 1))

# %% [markdown]
# ## 10. Validation — time-based split, prediction error on affected-product sales
#
# The model is evaluated **only** on weeks 98-101 (never trained on them) and all features are past/static-only, so no future information reaches the test predictions. We report the prediction error of `affected_product_quantity` overall and split by whether the source product was actually promoted.

# %%
# 10. Validation: prediction error on affected product sales
def qty_metrics(d):
    y = d["qty_B"]
    p = d["pred_qty_B"]
    return dict(mae=float(mean_absolute_error(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                corr=float(np.corrcoef(y, p)[0, 1]),
                mean_actual=float(y.mean()), mean_pred=float(p.mean()))


train["pred_qty_B"] = m.predict(train[MODEL_FEATURES])
test["pred_qty_B"] = m.predict(test[MODEL_FEATURES])

print("train MAE/RMSE/corr:", {k: round(v, 3) for k, v in qty_metrics(train).items()})
print("test  MAE/RMSE/corr:", {k: round(v, 3) for k, v in qty_metrics(test).items()})
for g in (0, 1):
    sub = test[test["source_promo_flag"] == g]
    print(f"test  source_promo_flag={g}:", {k: round(v, 3) for k, v in qty_metrics(sub).items()}, f"(n={len(sub):,})")

# %% [markdown]
# ## 11. Treatment-effect estimation for every pair-week
#
# For every held-out pair-week we create two copies of the evaluation data:
# - **Counterfactual control** — `source_promo_flag = 0`
# - **Treatment** — `source_promo_flag = 1`
#
# then
# ```
# cannibalization_quantity = max(0, y_control − y_treatment)
# ```
#
# (When a week is actually promoted, `disc_depth_A` keeps the real discount; for control weeks the treatment copy has flag=1 at zero depth, so its estimate is a conservative lower bound of a real promotion.)

# %%
# 11. Estimate treatment effect for every pair-week (test window)
eval_ctl = test[MODEL_FEATURES].copy()
eval_ctl["source_promo_flag"] = 0
eval_trt = test[MODEL_FEATURES].copy()
eval_trt["source_promo_flag"] = 1

effects = pd.DataFrame({
    "source_product": test["promoted_product"].to_numpy(),
    "target_product": test["affected_product"].to_numpy(),
    "week": test["WEEK_NO"].to_numpy(),
    "source_promo_flag": test["source_promo_flag"].to_numpy(),
    "cosine_sim": test["cosine_sim"].to_numpy(),
    "demand_corr": test["demand_corr"].to_numpy(),
    "y_control": m.predict(eval_ctl),
    "y_treatment": m.predict(eval_trt),
})
effects["cannibalization_quantity"] = np.maximum(0.0, effects["y_control"] - effects["y_treatment"])
effects.to_csv(OUT / "pair_effects.csv", index=False)
print("pair-week effects:", len(effects), "| positive:", int((effects["cannibalization_quantity"] > 0).sum()),
      "| mean:", round(effects["cannibalization_quantity"].mean(), 3),
      "| total:", round(effects["cannibalization_quantity"].sum(), 1))

# %% [markdown]
# ## 12. Product × product cannibalization matrix
#
# Aggregates the pair-week effects into a directed `source_product → target_product` matrix with the **sum** and **mean** predicted cannibalized quantity over the evaluation window.

# %%
# 12. Aggregate: product-product cannibalization matrix
matrix = effects.groupby(["source_product", "target_product"], as_index=False).agg(
    n_weeks=("cannibalization_quantity", "size"),
    sum_cannibalization=("cannibalization_quantity", "sum"),
    mean_cannibalization=("cannibalization_quantity", "mean"),
    mean_cosine_sim=("cosine_sim", "mean"),
    mean_demand_corr=("demand_corr", "mean"),
)
matrix = matrix.sort_values("sum_cannibalization", ascending=False).reset_index(drop=True)
matrix.to_csv(OUT / "cannibalization_matrix.csv", index=False)
print("cannibalization matrix:", matrix.shape,
      "| pairs with positive effect:", int((matrix["sum_cannibalization"] > 0).sum()))
print(matrix.head(10).round(3).to_string(index=False))

# %% [markdown]
# ## 13. SHAP explainability
#
# Two views:
# 1. **Global model importance** — mean |SHAP| on a held-out sample (what drives `affected_product_quantity`).
# 2. **Cannibalization drivers** — SHAP difference between the treatment and control predictions (`shap_treatment − shap_control`). A negative contribution means the feature pushes `y_treatment` below `y_control`, i.e. it **increases cannibalization**.

# %%
# 13a. SHAP importance (global)
import shap

shap_sample = test.sample(min(200 if SMOKE else 1200, len(test)), random_state=SEED)
explainer = shap.TreeExplainer(m)
sv = explainer.shap_values(shap_sample[MODEL_FEATURES])
shap_imp = pd.DataFrame({"feature": MODEL_FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values(
    "mean_abs_shap", ascending=False).reset_index(drop=True)
shap_imp.to_parquet(OUT / "shap_importance.parquet", index=False)

print("top 15 features by mean |SHAP| (affected-quantity model):")
print(shap_imp.head(15).round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=shap_imp.head(20), x="mean_abs_shap", y="feature", color="#1f77b4", ax=ax)
ax.set_title("SHAP importance — S-Learner (affected quantity model)")
fig.tight_layout()
fig.savefig(OUT / "shap_importance.png", dpi=140)
plt.show()

# %%
# 13b. Cannibalization drivers: SHAP difference (treatment - control)
Xc = shap_sample[MODEL_FEATURES].copy()
Xc["source_promo_flag"] = 0
Xt = shap_sample[MODEL_FEATURES].copy()
Xt["source_promo_flag"] = 1
sv_c = explainer.shap_values(Xc)
sv_t = explainer.shap_values(Xt)
contrib = (sv_t - sv_c).mean(axis=0)
drivers = pd.DataFrame({"feature": MODEL_FEATURES, "mean_effect_contribution": contrib}).sort_values(
    "mean_effect_contribution")
drivers.to_csv(OUT / "shap_cannibalization_drivers.csv", index=False)

mean_effect = float((sv_t - sv_c).sum(axis=1).mean())
print("mean E[y_treatment - y_control] =", round(mean_effect, 4),
      "| sum of feature contributions =", round(float(contrib.sum()), 4))
print("\ntop features that INCREASE cannibalization (negative contribution):")
print(drivers.head(10).round(4).to_string(index=False))

vis = pd.concat([drivers.head(15), drivers.tail(15)]).drop_duplicates("feature").sort_values(
    "mean_effect_contribution")
fig2, ax2 = plt.subplots(figsize=(10, 9))
ax2.barh(vis["feature"], vis["mean_effect_contribution"],
         color=np.where(vis["mean_effect_contribution"] < 0, "#d62728", "#1f77b4"))
ax2.axvline(0, color="grey", lw=0.8)
ax2.set(xlabel="mean contribution to E[y_treatment − y_control] (negative ⇒ more cannibalization)",
        title="SHAP difference (treatment − control): features driving predicted cannibalization")
fig2.tight_layout()
fig2.savefig(FIG / "shap_cannibalization_drivers.png", dpi=140)
plt.show()

# 13c. Worked example: largest predicted cannibalization in the test window
ex_idx = effects["cannibalization_quantity"].nlargest(1).index[0]
ex_row = effects.loc[ex_idx]
print("=" * 100)
print("WORKED EXAMPLE (test window)")
for pid_col, label in [("source_product", "Promoted product"), ("target_product", "Affected product")]:
    pid = int(ex_row[pid_col])
    pinfo = product[product["PRODUCT_ID"] == pid]
    if not pinfo.empty:
        r = pinfo.iloc[0]
        print(f"{label}: {pid} | {r['BRAND']} | {r['COMMODITY_DESC']} | {r['SUB_COMMODITY_DESC']}")
    else:
        print(f"{label}: {pid}")
print(f"Week: {int(ex_row['week'])} | embedding cosine: {ex_row['cosine_sim']:.3f}")
print(f"y_control (A not promoted): {ex_row['y_control']:.2f} | "
      f"y_treatment (A promoted): {ex_row['y_treatment']:.2f} | "
      f"cannibalization_quantity: {ex_row['cannibalization_quantity']:.2f}")
if ex_idx in shap_sample.index:
    pos = list(shap_sample.index).index(ex_idx)
    delta_row = sv_t[pos] - sv_c[pos]
    row_drivers = pd.Series(delta_row, index=MODEL_FEATURES).sort_values()
    print("Main drivers of this effect (SHAP difference, negative => more cannibalization):")
    test_row = test.loc[ex_idx]
    for f in row_drivers.head(5).index:
        print(f"  {f:28s} contrib={row_drivers[f]:+.4f}  (value={test_row[f]:.4f})")
else:
    print("(example row not in SHAP sample; driver detail skipped)")

# %% [markdown]
# ## 14. Save outputs
#
# Saved under `outputs/cannibalization/`:
# - `causal_model.pkl` — the S-Learner LightGBM regressor
# - `cannibalization_matrix.csv` — product × product sum/mean predicted cannibalization
# - `pair_effects.csv` — per pair-week `y_control`, `y_treatment`, `cannibalization_quantity`
# - `shap_importance.png` — global SHAP importance
# - plus `metrics.json`, the pair-feature table, candidate pairs, and SHAP details

# %%
# 14. Save artifacts
joblib.dump(m, OUT / "causal_model.pkl")
subs.to_parquet(OUT / "candidate_pairs.parquet", index=False)

json.dump({
    "model": "S-Learner LightGBM regression (causal treatment-effect)",
    "n_features": len(MODEL_FEATURES),
    "features": MODEL_FEATURES,
    "n_candidates": int(len(subs)),
    "n_pair_weeks": {"train": int(len(train)), "test": int(len(test))},
    "validation": {"train": qty_metrics(train), "test": qty_metrics(test),
                   "test_promo_off": qty_metrics(test[test["source_promo_flag"] == 0]),
                   "test_promo_on": qty_metrics(test[test["source_promo_flag"] == 1])},
    "effects": {"n_pair_weeks": int(len(effects)),
                "n_positive": int((effects["cannibalization_quantity"] > 0).sum()),
                "mean_cannibalization": float(effects["cannibalization_quantity"].mean()),
                "total_cannibalization": float(effects["cannibalization_quantity"].sum())},
    "config": {"first_test_week": FIRST_TEST_WEEK, "last_test_week": LAST_TEST_WEEK,
               "train_end": TRAIN_END, "resid_end": RESID_END, "model_split_week": MODEL_SPLIT_WEEK,
               "top_subs": TOP_SUBS, "max_candidates": MAX_CANDIDATES,
               "cosine_min": COSINE_MIN, "evidence_cosine": EVIDENCE_COSINE,
               "emb_dims": EMB_DIMS, "seed": SEED}},
    open(OUT / "metrics.json", "w"), indent=2)

print("saved ->", OUT)
for f in sorted(OUT.iterdir()):
    if f.is_file():
        print(f"  {f.name:38s} {f.stat().st_size / 1e6:8.2f} MB")

print("\ntop product-product cannibalization pairs (test window):")
print(matrix.head(10).round(3).to_string(index=False))
