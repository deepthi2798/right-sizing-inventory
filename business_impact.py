"""
PHASE 5 -- Business impact: turn inventory_policy.csv into $ figures a
supply-chain leader can act on.

WHY THIS FRAMING (each number traces back to something upstream):

1. "TARGET VALUE" USES REORDER POINT, NOT A PRECISE TARGET STOCK LEVEL.
   reorder_point (Phase 4) = expected lead-time demand + safety stock -- it's
   the level that should trigger a fresh order, i.e. a FLOOR, not the average
   amount you'd hold across a full order cycle (true average holding would
   also include ~half an order quantity, which this case never asked us to
   size -- no EOQ / order-quantity policy was requested). So every $ figure
   here is anchored to that floor on purpose: it's the most defensible,
   most conservative reference point available from what Phases 1-4 produced,
   and it's stated explicitly as an assumption in the write-up.

2. EXCESS vs SHORTFALL, NOT A SINGLE NET NUMBER.
   Netting excess against shortfall (as a single "$X more/less needed")
   hides the real finding: the paradox isn't "too much inventory in total,"
   it's "the SAME pool of cash is badly allocated." So this script reports
   excess $ (freeable, sitting above the floor) and shortfall $ (needed to
   reach the floor) separately, plus what's left after redeploying the
   excess -- that redeployment step is the actual, actionable recommendation,
   not the raw shortfall number.

3. "VALUE AT RISK" IS DELIBERATELY *NOT* A FULL-YEAR REVENUE EXTRAPOLATION.
   An earlier draft of this analysis multiplied weekly demand-at-risk by 52
   weeks and got a number many times larger than the entire inventory book
   ($50M+ on a $2.5M book) -- that's not credible and overstates the case:
   being below the reorder point does not mean 100% of a SKU's demand goes
   unfulfilled every week from here on, only that the buffer protecting it
   is thinner than the target service level calls for. Instead this script
   reports two bounded, defensible numbers: (a) the $ value of demand for
   SKUs currently AT ZERO -- an active stockout happening right now, not a
   projection, and (b) the count/$  of SKUs running below their protective
   floor, as a coverage-risk indicator, not a lost-sales claim.

OUTPUTS (printed; also written to business_impact_summary.csv, one row per
ABC tier x sb_class segment, for pasting into the write-up)

HOW TO USE: run after phase4_inventory_policy.py has produced
inventory_policy.csv. `python phase5_business_impact.py`.
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
          f"(this is what a single week of stockout on these SKUs is worth "
          f"at cost -- a bounded, current-state number, not a full-year claim)")

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
    print(f"Saved {OUT_SUMMARY} (per ABC-tier x demand-class segment).")
    print("\nPaste back everything above -- these are the numbers the written "
          "summary and $ impact section will be built from.")


if __name__ == "__main__":
    main()