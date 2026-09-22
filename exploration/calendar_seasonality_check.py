"""
One-off check that led to building the pooled-seasonal forecast in step 4.

The baseline forecast's bias (Smooth SKUs: mean -9.41, ~90% negative) was
too large and too uniform in direction to be per-SKU noise. Every SKU's
holdout is the same last 8 calendar weeks, so if demand across the whole
catalog is genuinely higher in that window than a trend line predicts,
that's a real shared calendar effect, not 200 unrelated estimation errors
-- and the fix isn't "give up on seasonality," it's "borrow strength
across all 200 SKUs to estimate one shared seasonal profile."

This script doesn't assume the answer, it just aggregates the clean panel
so the shape is visible directly.

Run: python calendar_seasonality_check.py  (needs clean_panel.csv from
step 2)
"""

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

PANEL_PATH = "clean_panel.csv"
HOLDOUT_WEEKS = 8


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"]).sort_values("date")

    weekly_total = panel.groupby("date")["demand_qty"].sum().sort_index()
    weeks = weekly_total.index.to_list()

    banner("1. TOTAL WEEKLY DEMAND (ALL SKUs) -- LAST 16 WEEKS")
    print(weekly_total.tail(16).to_string())
    print(f"\nFull-series weekly average: {weekly_total.mean():.1f}")
    print(f"Full-series weekly std:     {weekly_total.std():.1f}")

    banner("2. HOLDOUT WINDOW (LAST 8 WEEKS) VS. REST OF HISTORY")
    holdout = weekly_total.iloc[-HOLDOUT_WEEKS:]
    rest = weekly_total.iloc[:-HOLDOUT_WEEKS]
    print(f"Mean weekly total, last {HOLDOUT_WEEKS} weeks:  {holdout.mean():.1f}")
    print(f"Mean weekly total, all earlier weeks: {rest.mean():.1f}")
    print(f"Difference: {(holdout.mean() / rest.mean() - 1) * 100:+.1f}%")

    banner("3. SAME CALENDAR WEEKS, YEAR OVER YEAR (is it a repeating pattern?)")
    panel["iso_year"] = panel["date"].dt.isocalendar().year
    panel["iso_week"] = panel["date"].dt.isocalendar().week
    yw = panel.groupby(["iso_year", "iso_week"])["demand_qty"].sum().reset_index()
    pivot = yw.pivot(index="iso_week", columns="iso_year", values="demand_qty")
    print("Weekly total demand by ISO week-of-year, one column per year:")
    print(pivot.to_string())

    years = sorted(panel["iso_year"].unique())
    if len(years) >= 2:
        y1, y2 = years[-2], years[-1]
        comp = pivot[[y1, y2]].dropna()
        corr = comp[y1].corr(comp[y2])
        print(f"\nCorrelation between {y1} and {y2}'s week-of-year demand shape: {corr:.3f}")
        print("(close to 1 = the same weeks run high/low both years -> real recurring "
              "seasonality. close to 0 = no repeating pattern -> the bias is something "
              "else, e.g. genuine growth or a quirk of the backtest window.)")

    banner("4. PRODUCT-FAMILY BREAKDOWN OF THE HOLDOUT BUMP (universal or concentrated?)")
    fam = panel.groupby(["product_family", "date"])["demand_qty"].sum().reset_index()
    fam_holdout = fam[fam["date"].isin(weeks[-HOLDOUT_WEEKS:])].groupby("product_family")["demand_qty"].mean()
    fam_rest = fam[fam["date"].isin(weeks[:-HOLDOUT_WEEKS])].groupby("product_family")["demand_qty"].mean()
    fam_cmp = pd.DataFrame({"holdout_wk_mean": fam_holdout, "rest_wk_mean": fam_rest})
    fam_cmp["pct_diff"] = (fam_cmp["holdout_wk_mean"] / fam_cmp["rest_wk_mean"] - 1) * 100
    print(fam_cmp.round(1).to_string())


if __name__ == "__main__":
    main()
