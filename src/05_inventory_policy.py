"""
Step 5: turn forecasts into an actual inventory policy -- safety stock,
reorder point, differentiated service levels.

1. Bias correction first. forecast_results.csv gives a per-SKU backtest
   bias (forecast minus actual) -- a known, measured, systematic error.
   Correcting it directly is more honest than padding safety stock to
   cover it, since safety stock should absorb genuine randomness, not a
   directional error that would over- or under-cover depending on which
   way the noise happens to point that week.
       corrected_weekly_forecast = forecast_mean_next_horizon - backtest_bias

2. Two safety-stock methods, matching the demand-pattern split from step 1:
   - Smooth/Erratic (~155 SKUs, ~97% of volume): classic
     SS = z * sigma_weekly * sqrt(lead_time_weeks). sigma_weekly comes
     from each SKU's own backtest MAE (sigma ~= MAE * 1.2533, the usual
     conversion assuming ~normal forecast errors), because it reflects
     forecast uncertainty going forward, not raw historical demand
     variance -- some of that variance is already explained by trend and
     season.
   - Intermittent/Lumpy (~45 SKUs): a normal-distribution formula is the
     wrong tool here, these SKUs are mostly zero with occasional spikes,
     nothing like a bell curve. Instead: block-bootstrap the SKU's own
     history in windows the length of its lead time, build an empirical
     distribution of what lead-time demand could look like, and set
     safety stock as the gap between the target service-level percentile
     of that distribution and its mean. No distributional assumption
     needed, and it's easy to explain in plain language.

3. Service level differentiated by value (ABC), not one blanket number.
   Step 1 found inventory value heavily concentrated. A uniform 95%
   service level on a $7,000 part and a $25 part wastes safety-stock
   dollars on the cheap one and under-protects the expensive one.
   Standard ABC treatment by cumulative value share: A = first 70%,
   B = next 20% (up to 90%), C = remaining 10%; target service levels
   98% / 95% / 90%. This is a stated assumption, not a fact from the
   data -- flag it in the write-up and adjust ABC_SERVICE_LEVELS below
   if the business wants different targets.

Run: python 05_inventory_policy.py  (needs clean_panel.csv, sku_features.csv
from step 2, and forecast_results.csv from step 4)

Outputs: inventory_policy.csv
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

PANEL_PATH = "clean_panel.csv"
FEATURES_PATH = "sku_features.csv"
FORECAST_RESULTS_PATH = "forecast_results.csv"
OUT_POLICY = "inventory_policy.csv"

MAE_TO_STD = 1.2533   # sigma ~= MAE * sqrt(pi/2), standard conversion assuming
                       # ~normally distributed forecast errors
N_BOOTSTRAP = 2000
RNG_SEED = 42

ABC_VALUE_CUTOFFS = (0.70, 0.90)   # cumulative value share: A up to 70%, B up to 90%, C rest
ABC_SERVICE_LEVELS = {"A": 0.98, "B": 0.95, "C": 0.90}
ABC_Z = {"A": 2.054, "B": 1.645, "C": 1.282}


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def assign_abc(feat):
    srt = feat.sort_values("inventory_value", ascending=False).copy()
    srt["cum_share"] = srt["inventory_value"].cumsum() / srt["inventory_value"].sum()

    def tier(share):
        if share <= ABC_VALUE_CUTOFFS[0]:
            return "A"
        elif share <= ABC_VALUE_CUTOFFS[1]:
            return "B"
        return "C"

    srt["abc_tier"] = srt["cum_share"].map(tier)
    return srt[["sku_id", "abc_tier"]]


def bootstrap_safety_stock(weekly_demand, lead_time_weeks, service_level, n=N_BOOTSTRAP, rng=None):
    """Block-bootstrap the SKU's own history in windows of length
    round(lead_time_weeks), sum each block into one simulated lead-time
    demand draw, and set safety stock as the gap between the target
    percentile of that empirical distribution and its mean."""
    rng = rng or np.random.default_rng(RNG_SEED)
    block_len = max(1, round(lead_time_weeks))
    y = np.asarray(weekly_demand, dtype=float)
    if len(y) < block_len:
        block_len = len(y)
    n_blocks_possible = len(y) - block_len + 1
    if n_blocks_possible < 1:
        return 0.0, float(np.mean(y) * lead_time_weeks)
    starts = rng.integers(0, n_blocks_possible, size=n)
    draws = np.array([y[s:s + block_len].sum() for s in starts])
    target = np.percentile(draws, service_level * 100)
    mean_ltd = draws.mean()
    return float(max(target - mean_ltd, 0.0)), float(mean_ltd)


def main():
    banner("LOAD")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    feat = pd.read_csv(FEATURES_PATH)
    fc = pd.read_csv(FORECAST_RESULTS_PATH)
    print(f"Panel: {panel.shape[0]:,} rows, {panel['sku_id'].nunique()} SKUs")
    print(f"Features: {feat.shape[0]} SKUs. Forecasts: {fc.shape[0]} SKUs.")

    # product_family isn't guaranteed to be in sku_features.csv depending on
    # which version of step 2 produced it -- pull it from the panel if it's
    # missing, otherwise leave it alone (merging a second copy in would
    # silently get renamed to product_family_x/_y by pandas and break the
    # column selection at the end).
    has_family = "product_family" in feat.columns
    if not has_family:
        family_lookup = (
            panel.sort_values("date").groupby("sku_id")["product_family"].last().reset_index()
        )

    banner("ASSIGN ABC TIERS (by cumulative inventory value share)")
    abc = assign_abc(feat)
    print(abc["abc_tier"].value_counts().to_string())
    tier_value = feat.merge(abc, on="sku_id").groupby("abc_tier")["inventory_value"].sum()
    print("\nInventory value by tier:")
    print(tier_value.round(0).to_string())

    df = feat.merge(abc, on="sku_id").merge(fc, on="sku_id", suffixes=("", "_fc"))
    if not has_family:
        df = df.merge(family_lookup, on="sku_id", how="left")

    df["target_service_level"] = df["abc_tier"].map(ABC_SERVICE_LEVELS)
    df["z"] = df["abc_tier"].map(ABC_Z)

    lt_weeks = df["latest_lead_time_days"] / 7.0
    df["lead_time_weeks"] = lt_weeks

    # bias correction: forecast_mean_next_horizon is already the point
    # forecast from step 4; subtract the measured backtest bias to correct it
    df["corrected_weekly_forecast"] = (df["forecast_mean_next_horizon"] - df["backtest_bias"]).clip(lower=0)

    banner("SAFETY STOCK -- TWO METHODS BY SEGMENT")
    rng = np.random.default_rng(RNG_SEED)
    ss_list, exp_ltd_list, method_list = [], [], []
    demand_by_sku = {sku: g.sort_values("date")["demand_qty"].to_numpy()
                      for sku, g in panel.groupby("sku_id")}

    for _, row in df.iterrows():
        sku = row["sku_id"]
        ltw = row["lead_time_weeks"]
        if row["sb_class"] in ("Intermittent", "Lumpy"):
            ss, exp_ltd = bootstrap_safety_stock(
                demand_by_sku[sku], ltw, row["target_service_level"], rng=rng
            )
            method_list.append("Empirical bootstrap")
        else:
            sigma_weekly = row["backtest_mae"] * MAE_TO_STD
            ss = row["z"] * sigma_weekly * np.sqrt(ltw)
            exp_ltd = row["corrected_weekly_forecast"] * ltw
            method_list.append("Parametric (z*sigma*sqrt(LT))")
        ss_list.append(ss)
        exp_ltd_list.append(exp_ltd)

    df["safety_stock_method"] = method_list
    df["safety_stock"] = np.round(ss_list, 1)
    df["expected_lead_time_demand"] = np.round(exp_ltd_list, 1)
    df["reorder_point"] = np.round(df["expected_lead_time_demand"] + df["safety_stock"], 1)

    df["current_on_hand"] = df["latest_on_hand"]
    df["gap_vs_reorder_point"] = np.round(df["current_on_hand"] - df["reorder_point"], 1)

    def flag(row):
        if row["current_on_hand"] <= 0:
            return "AT ZERO -- order now"
        if row["gap_vs_reorder_point"] < 0:
            return "Below ROP -- order now"
        if row["gap_vs_reorder_point"] > 2 * row["reorder_point"]:
            return "Well above ROP -- excess"
        return "At/above ROP -- OK"

    df["status"] = df.apply(flag, axis=1)

    banner("SUMMARY")
    print("Status counts:")
    print(df["status"].value_counts().to_string())
    print("\nSafety stock method usage:")
    print(df["safety_stock_method"].value_counts().to_string())
    print("\nMean safety stock (units) by abc_tier x sb_class:")
    print(df.groupby(["abc_tier", "sb_class"])["safety_stock"].mean().round(1).to_string())

    cols = ["sku_id", "product_family", "sb_class", "abc_tier", "target_service_level",
            "lead_time_weeks", "corrected_weekly_forecast", "expected_lead_time_demand",
            "safety_stock_method", "safety_stock", "reorder_point", "current_on_hand",
            "gap_vs_reorder_point", "status", "latest_unit_cost", "inventory_value"]
    df[cols].to_csv(OUT_POLICY, index=False)

    banner("DONE")
    print(f"Saved {OUT_POLICY} ({df.shape[0]} SKUs)")


if __name__ == "__main__":
    main()
