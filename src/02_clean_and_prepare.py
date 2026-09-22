"""
Step 2: turn the raw file into a clean, forecast-ready panel plus a
per-SKU feature table.

Every cleaning rule here comes straight out of step 1's findings, nothing
is guessed:
  - dates: ISO YYYY-MM-DD, confirmed
  - duplicates: exact-row dupes, safe to drop
  - future_4wk_avg_demand: leakage, dropped from modeling and kept as a
    separate reference file only
  - demand_qty gaps (~0.2%): interpolated per SKU
  - on_hand_qty negatives: floored at 0
  - unit_cost negative/unstable values: replaced with that SKU's own
    median of valid costs
  - lead_time_days / supplier_id: use the most recent value per SKU (one
    SKU, CB-1050, genuinely switched suppliers mid-series)

Run: python 02_clean_and_prepare.py  (needs component_demand_history.csv)

Outputs: clean_panel.csv, sku_features.csv, leakage_reference.csv
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

CSV_PATH = "component_demand_history.csv"
OUT_PANEL = "clean_panel.csv"
OUT_FEATURES = "sku_features.csv"
OUT_LEAKAGE_REF = "leakage_reference.csv"


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_and_parse(path):
    df = pd.read_csv(path, dtype=str, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    df["_date"] = pd.to_datetime(df["week_start_date"], format="%Y-%m-%d", errors="coerce")
    bad = df["_date"].isna() & df["week_start_date"].notna()
    if bad.any():
        df.loc[bad, "_date"] = pd.to_datetime(df.loc[bad, "week_start_date"], errors="coerce")
    for c in ["demand_qty", "future_4wk_avg_demand", "unit_cost", "on_hand_qty",
              "lead_time_days", "stockout_flag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def sb_class(series):
    n = len(series)
    nz = series[series > 0]
    if len(nz) == 0:
        return pd.Series({"adi": np.nan, "cv2": np.nan, "sb_class": "No demand"})
    adi = n / len(nz)
    cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if nz.mean() > 0 else np.nan
    if pd.isna(cv2):
        cls = "Unknown"
    elif adi < 1.32 and cv2 < 0.49:
        cls = "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        cls = "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        cls = "Erratic"
    else:
        cls = "Lumpy"
    return pd.Series({"adi": adi, "cv2": cv2, "sb_class": cls})


def main():
    banner("LOAD + DEDUP")
    df = load_and_parse(CSV_PATH)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows: {n0:,} -> {len(df):,} after dropping exact duplicates")
    df = df.sort_values(["sku_id", "_date"]).reset_index(drop=True)

    banner("SPLIT OFF THE LEAKAGE COLUMN")
    leakage_ref = df[["sku_id", "week_start_date", "future_4wk_avg_demand"]].copy()
    leakage_ref.to_csv(OUT_LEAKAGE_REF, index=False)
    print(f"Saved future_4wk_avg_demand separately -> {OUT_LEAKAGE_REF} (reference only, never a model input)")
    df = df.drop(columns=["future_4wk_avg_demand"])

    banner("FIX demand_qty MISSING VALUES")
    n_missing = df["demand_qty"].isna().sum()
    print(f"Missing demand_qty rows before fix: {n_missing}")
    df["demand_qty"] = df.groupby("sku_id")["demand_qty"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    print(f"Missing demand_qty rows after per-SKU interpolation: {df['demand_qty'].isna().sum()}")

    banner("FIX on_hand_qty NEGATIVE VALUES")
    n_neg_oh = (df["on_hand_qty"] < 0).sum()
    print(f"Negative on_hand_qty rows: {n_neg_oh} -> floored at 0")
    df["on_hand_qty"] = df["on_hand_qty"].clip(lower=0)

    banner("FIX unit_cost NEGATIVE / UNSTABLE VALUES")
    cost_stats = df.groupby("sku_id")["unit_cost"].agg(["mean", "std"])
    cost_stats["cv"] = (cost_stats["std"] / cost_stats["mean"]).abs()
    bad_cost_skus = cost_stats[(cost_stats["cv"] > 0.05) | (cost_stats["mean"] < 0)].index.tolist()
    print(f"SKUs with unstable/negative unit_cost, replaced with per-SKU median: {bad_cost_skus}")
    median_pos_cost = df[df["unit_cost"] >= 0].groupby("sku_id")["unit_cost"].median()
    n_neg_cost = (df["unit_cost"] < 0).sum()
    df["unit_cost_clean"] = df["unit_cost"]
    for sku in bad_cost_skus:
        med = median_pos_cost.get(sku, np.nan)
        mask = df["sku_id"] == sku
        df.loc[mask, "unit_cost_clean"] = med
    still_neg = df["unit_cost_clean"] < 0
    if still_neg.any():
        df.loc[still_neg, "unit_cost_clean"] = df.loc[still_neg, "sku_id"].map(median_pos_cost)
    print(f"Negative unit_cost rows fixed: {n_neg_cost}")

    banner("LATEST lead_time_days / supplier_id / unit_cost PER SKU")
    latest = df.sort_values("_date").groupby("sku_id").tail(1).set_index("sku_id")
    latest_lookup = latest[["lead_time_days", "supplier_id", "unit_cost_clean"]].rename(
        columns={"lead_time_days": "latest_lead_time_days",
                 "supplier_id": "latest_supplier_id",
                 "unit_cost_clean": "latest_unit_cost"}
    )
    multi = df.groupby("sku_id")["supplier_id"].nunique()
    print(f"SKUs with a mid-series supplier change (using most recent lead time/supplier): "
          f"{multi[multi > 1].index.tolist()}")

    banner("SAVE CLEAN PANEL")
    panel_cols = ["sku_id", "_date", "week_start_date", "product_family", "demand_qty",
                  "unit_cost_clean", "on_hand_qty", "supplier_id", "lead_time_days",
                  "stockout_flag"]
    panel = df[panel_cols].rename(columns={"unit_cost_clean": "unit_cost", "_date": "date"})
    panel.to_csv(OUT_PANEL, index=False)
    print(f"Saved {OUT_PANEL}  ({panel.shape[0]:,} rows, {panel['sku_id'].nunique()} SKUs)")

    banner("BUILD PER-SKU FEATURE TABLE")
    base = df.groupby("sku_id").agg(
        product_family=("product_family", "first"),
        n_weeks=("_date", "nunique"),
        total_demand=("demand_qty", "sum"),
        mean_demand=("demand_qty", "mean"),
        std_demand=("demand_qty", "std"),
        pct_zero_weeks=("demand_qty", lambda s: (s == 0).mean() * 100),
        stockout_rate=("stockout_flag", "mean"),
        latest_on_hand=("on_hand_qty", "last"),
    )
    sb = df.groupby("sku_id")["demand_qty"].apply(sb_class).unstack()
    nz_stats = df[df["demand_qty"] > 0].groupby("sku_id")["demand_qty"].agg(nz_mean="mean", nz_std="std")

    feat = base.join(sb).join(nz_stats).join(latest_lookup)
    feat["inventory_value"] = feat["latest_on_hand"] * feat["latest_unit_cost"]
    feat["weeks_of_supply"] = feat["latest_on_hand"] / feat["mean_demand"].replace(0, np.nan)

    wos_med = feat["weeks_of_supply"].median()
    so_med = feat["stockout_rate"].median()

    def tag(row):
        over = row["weeks_of_supply"] > 2 * wos_med if pd.notna(row["weeks_of_supply"]) else False
        under = row["stockout_rate"] > max(2 * so_med, 0.10)
        if over and under:
            return "Overstocked AND stocking out (worst case)"
        if over:
            return "Overstocked / slow-moving"
        if under:
            return "Chronic stockouts"
        return "Reasonably balanced"

    feat["paradox_tag"] = feat.apply(tag, axis=1)
    feat = feat.reset_index()
    feat.to_csv(OUT_FEATURES, index=False)

    print(f"Saved {OUT_FEATURES}  ({feat.shape[0]} SKUs)")
    print("\nsb_class distribution:")
    print(feat["sb_class"].value_counts().to_string())
    print("\nparadox_tag distribution:")
    print(feat["paradox_tag"].value_counts().to_string())
    print("\nTotal cleaned inventory value:", round(feat["inventory_value"].sum(), 0))

    banner("DONE")


if __name__ == "__main__":
    main()
