"""
Step 1: data understanding, cleaning checks, and a first look at the
"too much stock AND too many stockouts" paradox.

Loads the raw weekly demand file, runs a bunch of sanity checks (schema,
dates, duplicates, missing/negative values), tests whether one of the
columns is leaking future information, tests whether demand looks censored
during stockout weeks, then builds a per-SKU summary that classifies each
part by demand pattern (Syntetos-Boylan: Smooth / Intermittent / Erratic /
Lumpy) and flags whether it looks overstocked, understocked, both, or fine.

Run: python 01_diagnostics.py   (needs component_demand_history.csv in the
same folder)

Outputs: sku_summary.csv, data_quality_flags.csv
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

CSV_PATH = "component_demand_history.csv"
OUT_SKU_SUMMARY = "sku_summary.csv"
OUT_ROW_FLAGS = "data_quality_flags.csv"

EXPECTED_COLS = [
    "week_start_date", "sku_id", "product_family", "demand_qty",
    "future_4wk_avg_demand", "unit_cost", "on_hand_qty", "supplier_id",
    "lead_time_days", "stockout_flag",
]

VALID_FAMILIES = {"Molded Case", "Trip Unit", "Terminal Block", "Contact Assembly"}


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_raw(path):
    # sep=None + engine="python" auto-detects the delimiter -- real exports
    # aren't always comma-separated just because the extension says .csv
    df = pd.read_csv(path, dtype=str, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def section_schema(df):
    banner("A. SCHEMA CHECK")
    print(f"Rows: {len(df):,}   Columns: {list(df.columns)}")
    missing_cols = set(EXPECTED_COLS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLS)
    print(f"Missing expected columns: {missing_cols or 'none'}")
    print(f"Unexpected extra columns: {extra_cols or 'none'}")
    print(f"Unique sku_id count: {df['sku_id'].nunique()}")
    print(f"Unique product_family values: {sorted(df['product_family'].dropna().unique().tolist())}")
    bad_family = set(df["product_family"].dropna().unique()) - VALID_FAMILIES
    if bad_family:
        print(f"  -> product_family values not in the data dictionary: {bad_family}")
    print(f"Unique supplier_id count: {df['supplier_id'].nunique()}")


def parse_dates(df):
    # dates are ISO YYYY-MM-DD -- confirmed by testing a few candidate
    # formats directly against the raw strings (see exploration/date_format_check.py).
    # First pass assumed DD-MM-YYYY from how a sample looked when pasted into
    # a chat window, which silently mis-parsed ~60% of rows and made every
    # SKU look like it only had 42 weeks of history. Not a data problem, a
    # parsing bug -- worth remembering next time a date column "looks"
    # unambiguous but isn't.
    parsed = pd.to_datetime(df["week_start_date"], format="%Y-%m-%d", errors="coerce")
    fallback_needed = parsed.isna() & df["week_start_date"].notna()
    if fallback_needed.any():
        parsed2 = pd.to_datetime(df.loc[fallback_needed, "week_start_date"], errors="coerce")
        parsed.loc[fallback_needed] = parsed2
    return parsed


def section_dates(df):
    banner("B. DATE PARSING & CALENDAR COMPLETENESS")
    df["_date"] = parse_dates(df)
    bad_dates = df["_date"].isna().sum()
    print(f"Rows where week_start_date failed to parse: {bad_dates}")
    weekdays = df["_date"].dt.day_name().value_counts()
    print("Day-of-week distribution (should be ~100% Monday):")
    print(weekdays.to_string())
    print(f"Date range: {df['_date'].min()} to {df['_date'].max()}")

    full_range = pd.date_range(df["_date"].min().normalize(), df["_date"].max().normalize(), freq="W-MON")
    expected_weeks = len(full_range)
    coverage = df.groupby("sku_id")["_date"].nunique()
    print(f"\nExpected weeks in the full horizon: {expected_weeks}")
    print(f"SKUs with fewer distinct weeks than that: {(coverage < expected_weeks).sum()} / {coverage.shape[0]}")
    print(f"Coverage per SKU: min={coverage.min()}, median={coverage.median()}, max={coverage.max()}")
    return df


def section_duplicates(df):
    banner("C. DUPLICATES")
    dupe_key = df.duplicated(subset=["sku_id", "week_start_date"], keep=False)
    print(f"Rows sharing the same (sku_id, week_start_date): {dupe_key.sum()}")
    full_dupe = df.duplicated(keep=False)
    print(f"Fully identical duplicate rows: {full_dupe.sum()}")
    if dupe_key.sum() and full_dupe.sum() == dupe_key.sum():
        print("  -> every duplicate key is a fully identical row, safe to drop with drop_duplicates()")
    elif dupe_key.sum():
        print("  -> some duplicate keys have conflicting values, needs a tie-break rule before dropping")
    return dupe_key


def to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def section_missing_and_ranges(df):
    banner("D. MISSING VALUES & OUT-OF-RANGE VALUES")
    numeric_cols = ["demand_qty", "future_4wk_avg_demand", "unit_cost",
                     "on_hand_qty", "lead_time_days", "stockout_flag"]
    df = to_numeric(df, numeric_cols)

    miss = df[EXPECTED_COLS].isna().mean().sort_values(ascending=False) * 100
    print("Missing % by column:")
    print(miss.round(2).to_string())

    print("\nNegative-value checks (should never happen):")
    for c in ["demand_qty", "on_hand_qty", "unit_cost", "lead_time_days"]:
        n_neg = (df[c] < 0).sum()
        print(f"  {c}: {n_neg} negative rows" + (f"  (min={df[c].min()})" if n_neg else ""))

    print("\nstockout_flag sanity:")
    print(df["stockout_flag"].value_counts(dropna=False).to_string())

    print("\nunit_cost stability per SKU (a cost shouldn't swing much):")
    cost_stats = df.groupby("sku_id")["unit_cost"].agg(["mean", "std", "min", "max"])
    cost_stats["cv"] = cost_stats["std"] / cost_stats["mean"]
    volatile_cost = cost_stats[cost_stats["cv"] > 0.05].sort_values("cv", ascending=False)
    print(f"  SKUs with unit_cost CV > 5%: {len(volatile_cost)} / {cost_stats.shape[0]}")
    if len(volatile_cost):
        print(volatile_cost.head(10).round(2).to_string())

    print("\nlead_time_days stability per SKU:")
    lt_stats = df.groupby("sku_id")["lead_time_days"].nunique()
    print(f"  SKUs with more than one distinct lead time: {(lt_stats > 1).sum()} / {lt_stats.shape[0]}")

    print("\nsupplier_id changes mid-series:")
    sup_stats = df.groupby("sku_id")["supplier_id"].nunique()
    print(f"  SKUs with more than one supplier over the 2 years: {(sup_stats > 1).sum()} / {sup_stats.shape[0]}")

    return df


def section_leakage_check(df):
    banner("E. IS future_4wk_avg_demand LEAKAGE?")
    print("Testing whether it equals the mean of demand_qty over the next 4 weeks per SKU.")
    df_sorted = df.sort_values(["sku_id", "_date"]).copy()
    df_sorted["_fwd_mean_4"] = (
        df_sorted.groupby("sku_id")["demand_qty"]
        .transform(lambda s: s.shift(-1).rolling(4, min_periods=4).mean())
    )
    comparable = df_sorted.dropna(subset=["_fwd_mean_4", "future_4wk_avg_demand"])
    if len(comparable):
        diff = (comparable["_fwd_mean_4"] - comparable["future_4wk_avg_demand"]).abs()
        close = (diff < 0.51).mean() * 100
        corr = comparable["_fwd_mean_4"].corr(comparable["future_4wk_avg_demand"])
        print(f"  Comparable rows: {len(comparable):,}")
        print(f"  % matching the forward mean: {close:.1f}%")
        print(f"  Correlation: {corr:.4f}")
        if close > 90:
            print("  -> this column is built from future actuals. It can't be used as a model "
                  "input, only as a retrospective benchmark.")
        else:
            print("  -> doesn't cleanly match a forward-4-week mean, worth a closer look.")
    else:
        print("  Not enough per-SKU history to test this.")
    return df_sorted


def section_censoring_check(df):
    banner("F. IS demand_qty CENSORED DURING STOCKOUTS?")
    print("If demand is 'units consumed' rather than 'units wanted', a stockout week could "
          "understate real demand -- which would under-forecast exactly the SKUs that run out, "
          "compounding the problem. Checking whether that's actually happening here.")

    g = df.groupby("stockout_flag")["demand_qty"].agg(["count", "mean", "median", "std"])
    print("\ndemand_qty by stockout_flag:")
    print(g.round(2).to_string())

    zero_oh_and_stockout = ((df["on_hand_qty"] <= 0) & (df["stockout_flag"] == 1)).sum()
    stockout_rows = (df["stockout_flag"] == 1).sum()
    print(f"\nstockout_flag=1 and on_hand_qty<=0: {zero_oh_and_stockout} / {stockout_rows} stockout rows")

    nonzero_oh_and_stockout = ((df["on_hand_qty"] > 0) & (df["stockout_flag"] == 1)).sum()
    print(f"stockout_flag=1 but on_hand_qty>0: {nonzero_oh_and_stockout} "
          "(possible partial stockouts, or a flag/qty mismatch)")

    df2 = df.sort_values(["sku_id", "_date"]).copy()
    df2["_prev_stockout"] = df2.groupby("sku_id")["stockout_flag"].shift(1)
    rebound = df2[df2["_prev_stockout"] == 1]["demand_qty"].mean()
    normal = df2[df2["_prev_stockout"] == 0]["demand_qty"].mean()
    print(f"\nMean demand the week after a stockout: {rebound:.2f}")
    print(f"Mean demand the week after a normal week: {normal:.2f}")
    print("(a rebound after stockouts would support the censoring/pent-up-demand theory)")


def syntetos_boylan_class(series):
    """Classify a SKU's weekly demand into Smooth / Intermittent / Erratic / Lumpy
    using average inter-demand interval (ADI) and squared coefficient of variation
    of nonzero demand (CV2). Standard thresholds: ADI 1.32, CV2 0.49."""
    n = len(series)
    nonzero = series[series > 0]
    if len(nonzero) == 0:
        return pd.Series({"adi": np.nan, "cv2": np.nan, "sb_class": "No demand"})
    adi = n / len(nonzero)
    cv2 = (nonzero.std(ddof=0) / nonzero.mean()) ** 2 if nonzero.mean() > 0 else np.nan
    if pd.isna(cv2):
        sb_class = "Unknown"
    elif adi < 1.32 and cv2 < 0.49:
        sb_class = "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        sb_class = "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        sb_class = "Erratic"
    else:
        sb_class = "Lumpy"
    return pd.Series({"adi": adi, "cv2": cv2, "sb_class": sb_class})


def build_sku_summary(df):
    banner("G. PER-SKU SEGMENTATION & PARADOX TAGGING")
    df = df.sort_values(["sku_id", "_date"])

    base = df.groupby("sku_id").agg(
        product_family=("product_family", "first"),
        n_weeks=("_date", "nunique"),
        total_demand=("demand_qty", "sum"),
        mean_demand=("demand_qty", "mean"),
        pct_zero_weeks=("demand_qty", lambda s: (s == 0).mean() * 100),
        stockout_rate=("stockout_flag", "mean"),
        avg_on_hand=("on_hand_qty", "mean"),
        latest_on_hand=("on_hand_qty", "last"),
        unit_cost=("unit_cost", "mean"),
        lead_time_days=("lead_time_days", "mean"),
        n_suppliers=("supplier_id", "nunique"),
    )

    sb = df.groupby("sku_id")["demand_qty"].apply(syntetos_boylan_class).unstack()

    summary = base.join(sb)
    summary["inventory_value"] = summary["latest_on_hand"] * summary["unit_cost"]
    summary["weeks_of_supply"] = summary["latest_on_hand"] / summary["mean_demand"].replace(0, np.nan)

    # simple within-dataset thresholds, not fixed business rules -- revisit
    # once the real distribution is in front of you
    wos_median = summary["weeks_of_supply"].median()
    stockout_median = summary["stockout_rate"].median()

    def tag(row):
        overstocked = row["weeks_of_supply"] > 2 * wos_median if pd.notna(row["weeks_of_supply"]) else False
        understocked = row["stockout_rate"] > max(2 * stockout_median, 0.10)
        if overstocked and understocked:
            return "Overstocked AND stocking out (worst case)"
        if overstocked:
            return "Overstocked / slow-moving"
        if understocked:
            return "Chronic stockouts"
        return "Reasonably balanced"

    summary["paradox_tag"] = summary.apply(tag, axis=1)

    print(f"Demand-class distribution:\n{summary['sb_class'].value_counts().to_string()}")
    print(f"\nParadox-tag distribution:\n{summary['paradox_tag'].value_counts().to_string()}")
    print(f"\nTotal inventory value, all SKUs: {summary['inventory_value'].sum():,.0f}")
    print("Inventory value by paradox tag:")
    print(summary.groupby("paradox_tag")["inventory_value"].sum().round(0).to_string())
    print("\nDemand class vs paradox tag (the core of the diagnosis -- different SKU types "
          "need different treatment):")
    print(pd.crosstab(summary["sb_class"], summary["paradox_tag"]).to_string())

    return summary.reset_index()


def main():
    banner("LOADING")
    df = load_raw(CSV_PATH)
    section_schema(df)
    df = section_dates(df)
    dupe_mask = section_duplicates(df)

    # confirmed: every duplicate key in this dataset is a fully identical
    # repeated row, so a plain drop is safe. Re-check section C's output
    # above before trusting this on a different extract.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if n_before != len(df):
        print(f"\n[dedup] Dropped {n_before - len(df)} exact duplicate rows ({n_before:,} -> {len(df):,}).")

    df = section_missing_and_ranges(df)
    df = section_leakage_check(df)
    section_censoring_check(df)
    summary = build_sku_summary(df)

    df["_dupe_key"] = dupe_mask.values if len(dupe_mask) == len(df) else False
    flag_cols = ["sku_id", "week_start_date", "demand_qty", "on_hand_qty", "unit_cost",
                 "lead_time_days", "stockout_flag", "_dupe_key"]
    df[flag_cols].to_csv(OUT_ROW_FLAGS, index=False)
    summary.to_csv(OUT_SKU_SUMMARY, index=False)

    banner("DONE")
    print(f"Saved {OUT_SKU_SUMMARY}  ({summary.shape[0]} SKUs)")
    print(f"Saved {OUT_ROW_FLAGS}  ({df.shape[0]} rows)")


if __name__ == "__main__":
    main()
