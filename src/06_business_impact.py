"""
Step 6: turn inventory_policy.csv into dollar figures a supply-chain
leader can actually act on.

A couple of framing choices worth explaining:

1. "Target value" uses the reorder point, not a precise target stock
   level. reorder_point (step 5) = expected lead-time demand + safety
   stock -- that's the level that should trigger a fresh order, a floor,
   not the average amount you'd hold across a full order cycle (true
   average holding would also include roughly half an order quantity,
   which sizing an EOQ policy was outside this scope). Every dollar
   figure below is anchored to that floor on purpose -- it's the most
   defensible, conservative reference point available from steps 1-5,
   and it's called out explicitly as an assumption.

2. Excess vs. shortfall, not a single net number. Netting excess against
   shortfall into one "$X more/less needed" figure hides the actual
   finding: the paradox isn't "too much inventory in total," it's "the
   same pool of cash is badly allocated." So this reports excess $
   (freeable, sitting above the floor) and shortfall $ (needed to reach
   the floor) separately, plus what's left after redeploying the excess
   -- that redeployment step is the real recommendation, not the raw
   shortfall number.

3. "Value at risk" is deliberately not a full-year revenue extrapolation.
   An earlier version of this multiplied weekly demand-at-risk by 52
   weeks and landed on a number many times larger than the entire
   inventory book -- not credible, and it overstates the case (being
   below the reorder point doesn't mean 100% of a SKU's demand goes
   unfulfilled every week going forward, only that its buffer is thinner
   than the target service level calls for). Instead this reports two
   bounded, defensible numbers: the dollar value of demand on SKUs that
   are at zero stock right now (an active stockout, not a projection),
   and the count/dollar value of SKUs running below their protective
   floor as a coverage-risk indicator.

Run: python 06_business_impact.py  (needs inventory_policy.csv from step 5)

Outputs: business_impact_summary.csv
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

POLICY_PATH = "inventory_policy.csv"
OUT_SUMMARY = "business_impact_summary.csv"


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    banner("LOAD")
    df = pd.read_csv(POLICY_PATH)
    print(f"{df.shape[0]} SKUs loaded from {POLICY_PATH}")

    df["target_value"] = df["reorder_point"] * df["latest_unit_cost"]
    df["gap_value"] = df["inventory_value"] - df["target_value"]  # + = excess, - = shortfall

    total_current = df["inventory_value"].sum()
    total_target = df["target_value"].sum()
    excess = df.loc[df["gap_value"] > 0, "gap_value"].sum()
    shortfall = -df.loc[df["gap_value"] < 0, "gap_value"].sum()
    n_excess = (df["gap_value"] > 0).sum()
    n_short = (df["gap_value"] < 0).sum()

    banner("HEADLINE $ FIGURES")
    print(f"Total current inventory value (all {df.shape[0]} SKUs):  {total_current:>12,.0f}")
    print(f"Total value at each SKU's reorder point (floor):        {total_target:>12,.0f}")
    print()
    print(f"Freeable cash -- {n_excess} SKUs comfortably above their floor:   {excess:>12,.0f}")
    print(f"Coverage gap  -- {n_short} SKUs below their floor:                {shortfall:>12,.0f}")
    print(f"Net investment needed after redeploying the freed cash:      {shortfall - excess:>12,.0f}")

    banner("ACTIVE STOCKOUTS RIGHT NOW (bounded, not projected)")
    at_zero = df[df["status"] == "AT ZERO -- order now"]
    wk_value = (at_zero["corrected_weekly_forecast"] * at_zero["latest_unit_cost"]).sum()
    print(f"{len(at_zero)} SKUs at zero on-hand today.")
    print(f"Combined weekly demand value on those SKUs: {wk_value:,.0f} "
          f"(what a single week of stockout on these SKUs is worth at cost -- "
          f"a bounded, current-state number, not a full-year claim)")

    banner("WHERE THE EXCESS AND THE SHORTFALL CONCENTRATE (ABC x demand class)")
    seg = df.groupby(["abc_tier", "sb_class"]).agg(
        n_skus=("sku_id", "count"),
        current_value=("inventory_value", "sum"),
        target_value=("target_value", "sum"),
        excess=("gap_value", lambda s: s[s > 0].sum()),
        shortfall=("gap_value", lambda s: -s[s < 0].sum()),
    ).round(0)
    print(seg.to_string())
    seg.to_csv(OUT_SUMMARY)

    banner("BY ABC TIER ONLY (simpler view for the write-up)")
    tier = df.groupby("abc_tier").agg(
        n_skus=("sku_id", "count"),
        current_value=("inventory_value", "sum"),
        target_value=("target_value", "sum"),
        excess=("gap_value", lambda s: s[s > 0].sum()),
        shortfall=("gap_value", lambda s: -s[s < 0].sum()),
    ).round(0)
    print(tier.to_string())

    banner("TOP 10 SKUs HOLDING THE MOST FREEABLE CASH")
    top_excess = df[df["gap_value"] > 0].sort_values("gap_value", ascending=False).head(10)
    print(top_excess[["sku_id", "product_family", "sb_class", "abc_tier",
                       "current_on_hand", "reorder_point", "gap_value"]].to_string(index=False))

    banner("DONE")
    print(f"Saved {OUT_SUMMARY} (per ABC-tier x demand-class segment)")


if __name__ == "__main__":
    main()
