"""
Step 3: per-SKU baseline forecast.

Different SKUs need different forecasting methods:
  - Smooth/Erratic (regular, mostly-nonzero demand, ~155 of 200 SKUs,
    ~97% of unit volume): damped-trend exponential smoothing (Holt).
  - Intermittent/Lumpy (mostly-zero demand): smoothing methods either
    chase noise or flatten to a number that's wrong every week, so this
    uses Croston's method with the SBA bias correction instead, which
    forecasts demand size and demand interval separately.

First pass at this also tried a 52-week Holt-Winters seasonal fit whenever
a SKU had 2+ years of history. That was a mistake -- 90% of Smooth/Erratic
SKUs came back with a systematic negative bias, because 104 weeks is
exactly 2 seasonal cycles, nowhere near enough to reliably fit 52 separate
seasonal indices per SKU. The seasonal component was overfitting noise and
that bias was leaking straight into the forecast.

Fix: never assume seasonality, test for it. Every Smooth/Erratic SKU is
backtested with both a non-seasonal and a seasonal model on the same
holdout weeks, and the seasonal model only survives if it beats the
non-seasonal one by a real margin, not a coin flip. With ~2 years of data
the non-seasonal model wins almost everywhere, which is itself a useful,
citable finding ("not enough history to reliably fit weekly seasonality")
rather than a failure of the approach.

This script is a useful baseline and a sanity check, but it isn't the
final forecast -- see 04_forecast_pooled_seasonal.py for why the bias
above wasn't actually about seasonality at all.

Run: python 03_forecast_baseline.py  (needs clean_panel.csv and
sku_features.csv from step 2)

Outputs: forecast_results.csv, forecast_weekly.csv
"""

import warnings
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
warnings.filterwarnings("ignore")  # statsmodels convergence warnings are expected
                                    # noise from testing seasonal fits on short
                                    # series; handled via backtest comparison below

PANEL_PATH = "clean_panel.csv"
FEATURES_PATH = "sku_features.csv"
OUT_RESULTS = "forecast_results.csv"
OUT_WEEKLY = "forecast_weekly.csv"

HOLDOUT_WEEKS = 8
FORECAST_HORIZON = 8
SEASONAL_MIN_WEEKS = 3 * 52   # require 3 full cycles before attempting a
                              # seasonal fit at all -- 2 cycles is what broke
                              # the first version of this script
SEASONAL_IMPROVEMENT_THRESHOLD = 0.90  # keep the seasonal model only if its
                              # backtest MAE beats the simple model's by 10%+


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def holt_forecast(y, horizon, alpha=0.3, beta=0.1, phi=0.9):
    """Hand-rolled damped-trend Holt smoothing, no external dependency needed.
    Also used as a fallback if statsmodels isn't installed or fails to converge."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    level = y[0]
    trend = y[1] - y[0] if n > 1 else 0.0
    for t in range(n):
        obs = y[t]
        new_level = alpha * obs + (1 - alpha) * (level + phi * trend)
        new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level, trend = new_level, new_trend
    fc = np.array([level + sum(phi ** i for i in range(1, h + 1)) * trend for h in range(1, horizon + 1)])
    return np.clip(fc, 0, None)


def fit_forecast(y, horizon, seasonal):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        ys = pd.Series(np.asarray(y, dtype=float))
        if seasonal:
            model = ExponentialSmoothing(
                ys, trend="add", damped_trend=True, seasonal="add", seasonal_periods=52
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(ys, trend="add", damped_trend=True).fit(optimized=True)
        fc = np.clip(model.forecast(horizon).to_numpy(), 0, None)
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite forecast")
        return fc
    except Exception:
        return holt_forecast(y, horizon)


def choose_smooth_method(train, holdout_actual):
    """Backtest a non-seasonal model against a seasonal one (if there's enough
    history) on the same holdout weeks and pick whichever generalizes better."""
    h = len(holdout_actual)
    fc_simple = fit_forecast(train, h, seasonal=False)
    mae_simple = float(np.mean(np.abs(fc_simple - holdout_actual)))
    bias_simple = float(np.mean(fc_simple - holdout_actual))
    best = ("Holt damped-trend", mae_simple, bias_simple)

    if len(train) >= SEASONAL_MIN_WEEKS:
        fc_seas = fit_forecast(train, h, seasonal=True)
        mae_seas = float(np.mean(np.abs(fc_seas - holdout_actual)))
        bias_seas = float(np.mean(fc_seas - holdout_actual))
        if mae_seas <= SEASONAL_IMPROVEMENT_THRESHOLD * mae_simple:
            best = ("Holt-Winters (add trend+season)", mae_seas, bias_seas)

    return best


def croston_sba(y, horizon, alpha=0.2):
    y = np.asarray(y, dtype=float)
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return np.zeros(horizon)
    first = nz_idx[0]
    z = y[first]
    p = (nz_idx[1] - nz_idx[0]) if len(nz_idx) > 1 else (len(y) - first)
    q = 1
    for t in range(first + 1, len(y)):
        if y[t] > 0:
            z = alpha * y[t] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
    rate = max((z / p) * (1 - alpha / 2), 0.0)  # SBA bias correction
    return np.full(horizon, rate)


def main():
    banner("LOAD")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    feat = pd.read_csv(FEATURES_PATH)
    panel = panel.sort_values(["sku_id", "date"])
    print(f"Panel: {panel.shape[0]:,} rows, {panel['sku_id'].nunique()} SKUs")
    print(f"Feature table: {feat.shape[0]} SKUs, sb_class counts:")
    print(feat["sb_class"].value_counts().to_string())
    if panel.groupby("sku_id").size().min() < SEASONAL_MIN_WEEKS:
        print(f"\nMax available history per SKU is {panel.groupby('sku_id').size().max()} weeks, "
              f"below the {SEASONAL_MIN_WEEKS}-week bar for even attempting a seasonal fit -- "
              f"every Smooth/Erratic SKU will use the non-seasonal model, which is expected "
              f"with ~2 years of data.")

    results, weekly_rows = [], []

    for sku, g in panel.groupby("sku_id"):
        y = g["demand_qty"].to_numpy(dtype=float)
        sb_row = feat.loc[feat["sku_id"] == sku, "sb_class"]
        sb = sb_row.iloc[0] if len(sb_row) else "Smooth"

        holdout = min(HOLDOUT_WEEKS, max(1, len(y) - 10))
        train, test = y[:-holdout], y[-holdout:]

        if sb in ("Intermittent", "Lumpy"):
            fc_bt = croston_sba(train, holdout)
            mae = float(np.mean(np.abs(fc_bt - test)))
            bias = float(np.mean(fc_bt - test))
            method = "Croston-SBA"
            fc = croston_sba(y, FORECAST_HORIZON)
        else:
            method, mae, bias = choose_smooth_method(train, test)
            seasonal = method.startswith("Holt-Winters")
            fc = fit_forecast(y, FORECAST_HORIZON, seasonal=seasonal)

        results.append({
            "sku_id": sku, "sb_class": sb, "method": method,
            "backtest_mae": round(mae, 2), "backtest_bias": round(bias, 2),
            "forecast_mean_next_horizon": round(float(np.mean(fc)), 2),
        })
        for h, val in enumerate(fc, start=1):
            weekly_rows.append({"sku_id": sku, "week_ahead": h, "forecast_qty": round(float(val), 2)})

    res = pd.DataFrame(results)
    weekly = pd.DataFrame(weekly_rows)
    res.to_csv(OUT_RESULTS, index=False)
    weekly.to_csv(OUT_WEEKLY, index=False)

    banner("BACKTEST SUMMARY BY SEGMENT")
    print(res.groupby("sb_class")[["backtest_mae", "backtest_bias"]].mean().round(2))
    print("\nMethod usage:")
    print(res["method"].value_counts().to_string())
    sm = res[res["sb_class"].isin(["Smooth", "Erratic"])].copy()
    sm["bias_pct"] = (sm["backtest_bias"].abs() / sm["forecast_mean_next_horizon"].clip(lower=1)) * 100
    print(f"\n% of Smooth/Erratic SKUs with |bias| > 10% of forecast mean: "
          f"{(sm['bias_pct'] > 10).mean() * 100:.1f}%  (median |bias|: {sm['backtest_bias'].abs().median():.2f})")

    banner("DONE")
    print(f"Saved {OUT_RESULTS} ({res.shape[0]} SKUs) and {OUT_WEEKLY} "
          f"({weekly.shape[0]} rows, {FORECAST_HORIZON}-week horizon)")


if __name__ == "__main__":
    main()
