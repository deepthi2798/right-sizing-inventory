"""
One-off check that led to fixing the date parsing in step 1.

First pass through the data assumed week_start_date was DD-MM-YYYY, based
on how a pasted sample happened to display. That silently mis-parsed a
big chunk of rows and made every SKU look like it only had 42 weeks of
history instead of ~104. This script doesn't trust any earlier parsing --
it goes back to the raw strings and tests a few candidate formats directly
so there's no ambiguity about which one is right.

Run: python date_format_check.py  (needs component_demand_history.csv)
"""

import re
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

CSV_PATH = "component_demand_history.csv"


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    df = pd.read_csv(CSV_PATH, dtype=str, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    banner("1. RAW ROW / STRING COUNTS")
    print(f"Total rows in file: {len(df):,}")
    print(f"Unique sku_id: {df['sku_id'].nunique()}")
    raw_dates = df["week_start_date"].astype(str).str.strip()
    print(f"Unique raw week_start_date strings (no parsing): {raw_dates.nunique():,}")
    print(f"Rows per SKU: {len(df)/df['sku_id'].nunique():.1f}")

    banner("2. RAW STRING FORMAT CHECK")
    def shape(s):
        return "NaN" if pd.isna(s) else re.sub(r"\d", "#", s)
    shapes = raw_dates.map(shape).value_counts()
    print("Distinct date-string shapes (### = digit run):")
    print(shapes.to_string())
    print("\nFirst 5:", raw_dates.head(5).tolist())
    print("Last 5:", raw_dates.tail(5).tolist())

    banner("3. PARSE SUCCESS RATE UNDER DIFFERENT FORMAT ASSUMPTIONS")
    for fmt_name, fmt in [("dayfirst DD-MM-YYYY", "%d-%m-%Y"),
                           ("monthfirst MM-DD-YYYY", "%m-%d-%Y"),
                           ("ISO YYYY-MM-DD", "%Y-%m-%d")]:
        parsed = pd.to_datetime(raw_dates, format=fmt, errors="coerce")
        ok = parsed.notna().sum()
        print(f"  {fmt_name:28s}: {ok:,} / {len(df):,} parsed ({ok/len(df)*100:.1f}%)"
              + (f"   range: {parsed.min().date()} to {parsed.max().date()}" if ok else ""))

    flex = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")
    print(f"  {'flexible dayfirst=True':28s}: {flex.notna().sum():,} / {len(df):,} parsed"
          + (f"   range: {flex.min().date()} to {flex.max().date()}" if flex.notna().any() else ""))

    if flex.isna().any():
        bad = raw_dates[flex.isna()]
        print(f"\n  Still unparseable even with flexible dayfirst=True: {len(bad)}")
        print("  Sample:", bad.unique()[:10].tolist())

    banner("4. PER-SKU WEEK COVERAGE USING THE BEST-PARSING FORMAT")
    df["_date"] = flex
    per_sku_weeks = df.groupby("sku_id")["_date"].nunique()
    print(f"Distinct SKUs: {per_sku_weeks.shape[0]}")
    print(f"Weeks per SKU -> min={per_sku_weeks.min()}, median={per_sku_weeks.median()}, max={per_sku_weeks.max()}")
    print(f"\nDate span: {df['_date'].min()} to {df['_date'].max()}")
    full_range = pd.date_range(df["_date"].min(), df["_date"].max(), freq="W-MON")
    print(f"Number of Mondays in that span: {len(full_range)}")

    banner("5. DUPLICATE (sku_id, week_start_date) KEYS -- WITH CONFLICTS SHOWN")
    key_counts = df.groupby(["sku_id", "week_start_date"]).size()
    dupe_keys = key_counts[key_counts > 1]
    print(f"Duplicate keys: {len(dupe_keys)}")
    print(f"Fully identical duplicate rows: {df.duplicated().sum()}")
    if len(dupe_keys):
        print("\nFirst 3 duplicate keys, all their rows:")
        for (sku, wk) in list(dupe_keys.index)[:3]:
            sub = df[(df["sku_id"] == sku) & (df["week_start_date"] == wk)]
            print(f"\n  {sku} / {wk}  ({len(sub)} rows):")
            print(sub.to_string(index=False))

    banner("6. ROWS PER CALENDAR MONTH")
    print(df["_date"].dt.to_period("M").value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
