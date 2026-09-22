# Right-sizing inventory for a component plant

Case: leadership's complaint was "we have too much cash tied up in
inventory, and we still run out of parts too often." This repo is the
analysis behind the answer: it's not one problem, it's two, happening on
different SKUs at the same time, and a single blanket reorder rule was
never going to fix both.

200 circuit-breaker components, ~2 years of weekly demand history
(~20,800 rows).

## TL;DR

- $2.51M is tied up in inventory across the 200 parts.
- Only $336K of that (13%) is genuine excess sitting in slow-moving parts
  that don't need it. That's freeable cash with zero service-level cost.
- 154 of 200 parts (77%) are below the stock level they need to reliably
  cover demand during the supplier lead time, including 17 that are at
  zero stock right now.
- So it's not "too much inventory overall," it's the same money parked in
  the wrong SKUs. Free the $336K, redirect it to the highest-value gaps
  first (A/B tier accounts for $1.7M of the $2.86M shortfall despite being
  under half the SKU count).

Full writeup with the reasoning behind each number is in
`Inventory_Recommendation_Summary.docx` / `Assumptions_and_Next_Steps.docx`


## Why this isn't a "too much vs. too little" problem

The 200 parts don't all behave the same way. About three-quarters sell in
a steady, predictable weekly rhythm. The other quarter sells in bursts —
nothing for weeks, then a spike. A flat reorder rule can't serve both:
set the buffer high enough for the steady sellers and the bursty ones
drown in stock; set it low enough for the bursty ones and the steady
sellers run dry. That's the actual mechanism behind the complaint, and
it's why the fix here is per-SKU treatment, not a bigger or smaller
blanket buffer.

## Pipeline

Run these in order. Each one reads the previous step's output from the
current folder and writes its own CSVs there too.

```
pip install -r requirements.txt

python src/01_diagnostics.py            # data quality checks + paradox diagnosis
python src/02_clean_and_prepare.py       # cleaned panel + per-SKU features
python src/03_forecast_baseline.py       # per-SKU Holt / Croston baseline
python src/04_forecast_pooled_seasonal.py  # final forecast (replaces step 3's Smooth/Erratic numbers)
python src/05_inventory_policy.py        # safety stock, reorder point, service levels
python src/06_business_impact.py         # $ impact summary
```

You need `component_demand_history.csv` in the working directory to start

`exploration/` has two side scripts that aren't part of the main run but
explain two decisions in the pipeline:

- `date_format_check.py` — why step 1 parses dates as ISO `YYYY-MM-DD`
  and not `DD-MM-YYYY` (an early version got this wrong and it silently
  wrecked the week counts — worth a read if you're ever unsure about a
  date column that "looks" unambiguous).
- `calendar_seasonality_check.py` — the check that showed the baseline
  forecast's bias was real seasonality, not noise, which is why step 4
  exists.

## What each step actually does

**Step 1 — diagnostics.** Schema/date/duplicate checks, tests whether
`future_4wk_avg_demand` is leakage (it is — 100% correlation with the
literal forward-looking average, excluded from everything downstream),
tests whether demand looks censored during stockouts (it doesn't —
demand is actually *higher* on stockout weeks, so that theory's out),
classifies every SKU by demand pattern (Syntetos-Boylan: Smooth /
Intermittent / Erratic / Lumpy), and tags each one as overstocked,
understocked, both, or fine.

**Step 2 — cleaning.** Turns the diagnosis into fixes: interpolate small
demand gaps, floor negative on-hand at zero, replace unstable unit costs
with the SKU's own median, use each SKU's most recent lead time/supplier
rather than a historical blend (one part genuinely switched suppliers
mid-series).

**Step 3 — baseline forecast.** Damped-trend Holt for steady/erratic
demand, Croston-SBA for intermittent/lumpy. First attempt at this also
tried year-over-year seasonality whenever a SKU had 2+ years of history,
which was a mistake — 104 weeks is exactly 2 seasonal cycles, not enough
to fit 52 seasonal indices reliably, and it introduced a systematic
negative bias. Fixed by backtesting seasonal vs. non-seasonal per SKU and
only keeping seasonal if it actually wins.

**Step 4 — pooled seasonal forecast.** Even after that fix, the bias
didn't go away — it just stopped being mislabeled. Checked whether it was
real seasonality being missed (see `calendar_seasonality_check.py`):
2024 and 2025's week-of-year demand shapes correlate at 0.986, which is
about as clear a "yes" as you'll get. No single SKU has enough history to
fit its own seasonal curve, but pooled across 200 SKUs there's plenty of
data — so this fits one shared seasonal curve (2 Fourier harmonics) and
applies it per SKU. Cut Smooth-segment bias from -9.4 to -3.6 and MAE by
~29%, improved 87%+ of SKUs against the baseline.

**Step 5 — inventory policy.** Safety stock sized two different ways
depending on demand pattern (parametric z·σ·√LT for steady demand,
block-bootstrap simulation for intermittent/lumpy since it's nowhere near
normally distributed), reorder point = expected lead-time demand + safety
stock, service level target set by ABC value tier (98/95/90%) rather than
one number for everything.

**Step 6 — business impact.** Turns the policy into dollars: what's
sitting in genuine excess vs. what's needed to close the gap, where each
concentrates by tier and demand class, and a bounded (not extrapolated)
estimate of what's at risk from the parts currently out of stock.

## Assumptions worth knowing about before trusting these numbers

- Reorder point is used as a *floor* everywhere in step 6, not a full
  target — the true average inventory would include some extra cycle
  stock on top, which needs an order-quantity policy this case didn't
  ask for.
- The 98/95/90% service-level split by ABC tier is a judgment call, not
  something the data proved. Worth confirming with whoever owns stockout
  risk tolerance.
- Lead time and supplier per SKU are the *latest* observed value, not a
  historical average.
- Current on-hand is a snapshot, not an average. A part flagged "below
  reorder point" might just be mid-cycle and due for its next order, not
  actually broken — the scale of the gap here (154/200 SKUs, 17 at zero)
  goes well past what normal timing would explain, but not every
  borderline case in that group is necessarily a real problem.
- Two years of data is enough to confirm the seasonal pattern is real but
  not enough to fit it per SKU — that gets easier with a third year of
  history.

## Requirements

```
pandas
numpy
statsmodels   # optional — falls back to a hand-rolled Holt implementation if missing
```
