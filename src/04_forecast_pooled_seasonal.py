"""
Step 4: pooled seasonal forecast -- replaces the baseline's Smooth/Erratic
method. Croston-SBA for Intermittent/Lumpy is unchanged.

The baseline (step 3) still showed a large systematic negative bias for
Smooth SKUs even after the seasonal-overfitting bug was fixed there --
mean bias -9.41, about 90% of SKUs running negative. That ruled out
"overfit seasonality" as the cause and pointed at something real being
missed. Checked it directly (see exploration/calendar_seasonality_check.py):
2024 and 2025's week-of-year demand shapes correlate at 0.986. That's a
genuine, strong, repeating annual cycle plus a ~10-15% year-over-year growth
trend -- not noise, and not something a per-SKU model can ever estimate
reliably with only 104 weeks of history (3+ full cycles is the standard bar
for fitting 52 seasonal indices on one series).

The fix is pooling: 200 SKUs x 104 weeks is 20,800 data points, plenty to
estimate ONE shared seasonal curve even though no single SKU has enough on
its own.

Method:
  1. Fit one seasonal index curve (2 Fourier harmonics, not 52 free
     parameters -- the signal looks like one dominant annual cycle plus a
     milder second wave, not 52 independent effects) using every
     Smooth/Erratic SKU's demand normalized by its own mean, on data
     before a global holdout cutoff (same last-8-weeks window for every
     SKU, so this is a fair backtest, not leakage).
  2. Deseasonalize each SKU's training series, fit a plain damped-trend
     Holt model on what's left (just a trend/level to estimate now),
     forecast forward, then reseasonalize by multiplying back by the index.
  3. Backtest against the same holdout and compare directly to the step-3
     baseline so the improvement is a measured number, not an assumption.
  4. Intermittent/Lumpy SKUs still use Croston-SBA -- pooled seasonality
     for sparse-demand SKUs is a real gap in this pass (only 2.8% of
     volume, but worth flagging as a next step rather than ignoring).

Run: python 04_forecast_pooled_seasonal.py  (needs clean_panel.csv and
sku_features.csv from step 2)

Outputs: seasonal_index.csv, forecast_results.csv, forecast_weekly.csv
(overwrites step 3's forecast_results.csv / forecast_weekly.csv)
"""

import warnings
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
warnings.filterwarnings("ignore")

PANEL_PATH = "clean_panel.csv"
FEATURES_PATH = "sku_features.csv"
OUT_INDEX = "seasonal_index.csv"
OUT_RESULTS = "forecast_results.csv"
OUT_WEEKLY = "forecast_weekly.csv"

HOLDOUT_WEEKS = 8
FORECAST_HORIZON = 8


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fit_seasonal_index(train_panel, sku_class_map):
    """Pooled seasonal curve from Smooth/Erratic SKUs' training data only.
    2 Fourier harmonics (4 regressors + intercept) instead of 52 free
    indices, to keep it smooth and resistant to overfitting."""
    df = train_panel.copy()
    df["sb_class"] = df["sku_id"].map(sku_class_map)
    df = df[df["sb_class"].isin(["Smooth", "Erratic"])]
    sku_mean = df.groupby("sku_id")["demand_qty"].transform("mean")
    df = df[sku_mean > 0]
    df["ratio"] = df["demand_qty"] / sku_mean
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int).clip(upper=52)

    w = df["iso_week"].to_numpy()
    theta = 2 * np.pi * w / 52.0
    X = np.column_stack([
        np.ones(len(df)), np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta)
    ])
    y = df["ratio"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    weeks = np.arange(1, 53)
    theta_all = 2 * np.pi * weeks / 52.0
    X_all = np.column_stack([
        np.ones(52), np.sin(theta_all), np.cos(theta_all), np.sin(2 * theta_all), np.cos(2 * theta_all)
    ])
    index = X_all @ coef
    index = index / index.mean()      # renormalize so the curve averages to 1.0 over the year
    index = np.clip(index, 0.4, 2.0)  # guardrail against the pooled fit extrapolating too far
    return pd.Series(index, index=weeks, name="seasonal_index")


def holt_forecast(y, horizon, alpha=0.3, beta=0.1, phi=0.9):
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
    return fc  # not clipped here -- caller reseasonalizes and clips after


def fit_forecast_deseasonalized(y_dates, y_vals, seasonal_index, horizon, forecast_dates):
    weeks = y_dates.dt.isocalendar().week.astype(int).clip(upper=52).to_numpy()
    idx = seasonal_index.reindex(weeks).to_numpy()
    deseason = y_vals / idx
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(pd.Series(deseason), trend="add", damped_trend=True).fit(optimized=True)
        base_fc = model.forecast(horizon).to_numpy()
        if not np.all(np.isfinite(base_fc)):
            raise ValueError
    except Exception:
        base_fc = holt_forecast(deseason, horizon)

    forecast_dates = pd.Series(forecast_dates).reset_index(drop=True)
    fc_weeks = forecast_dates.dt.isocalendar().week.astype(int).clip(upper=52).to_numpy()
    fc_idx = seasonal_index.reindex(fc_weeks).to_numpy()
    return np.clip(base_fc * fc_idx, 0, None)


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
    rate = max((z / p) * (1 - alpha / 2), 0.0)
    return np.full(horizon, rate)


def plain_holt_baseline(y, horizon):
    """Used only for the side-by-side comparison against step 3, not the
    forecast itself."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(pd.Series(y, dtype=float), trend="add", damped_trend=True).fit(optimized=True)
        fc = np.clip(model.forecast(horizon).to_numpy(), 0, None)
        if np.all(np.isfinite(fc)):
            return fc
    except Exception:
        pass
    return np.clip(holt_forecast(y, horizon), 0, None)


def main():
    banner("LOAD")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"]).sort_values(["sku_id", "date"])
    feat = pd.read_csv(FEATURES_PATH)
    sku_class_map = dict(zip(feat["sku_id"], feat["sb_class"]))
    all_dates = sorted(panel["date"].unique())
    holdout_dates = set(all_dates[-HOLDOUT_WEEKS:])
    train_panel = panel[~panel["date"].isin(holdout_dates)]

    banner("FIT POOLED SEASONAL INDEX (train weeks only, no leakage)")
    seasonal_index = fit_seasonal_index(train_panel, sku_class_map)
    seasonal_index.to_csv(OUT_INDEX, header=True)
    print("Seasonal index by ISO week (1.0 = average week):")
    print(seasonal_index.round(3).to_string())
    print(f"\nPeak week: {seasonal_index.idxmax()} (x{seasonal_index.max():.2f})   "
          f"Trough week: {seasonal_index.idxmin()} (x{seasonal_index.min():.2f})")

    results, weekly_rows, comparison_rows = [], [], []

    for sku, g in panel.groupby("sku_id"):
        sb = sku_class_map.get(sku, "Smooth")
        g = g.sort_values("date")
        y = g["demand_qty"].to_numpy(dtype=float)
        dates = g["date"]
        holdout_n = min(HOLDOUT_WEEKS, max(1, len(y) - 10))
        train_y, test_y = y[:-holdout_n], y[-holdout_n:]
        train_dates, test_dates = dates.iloc[:-holdout_n], dates.iloc[-holdout_n:]

        if sb in ("Intermittent", "Lumpy"):
            fc_bt = croston_sba(train_y, holdout_n)
            method = "Croston-SBA"
            fc = croston_sba(y, FORECAST_HORIZON)
        else:
            fc_bt = fit_forecast_deseasonalized(train_dates, train_y, seasonal_index, holdout_n, test_dates)
            method = "Pooled-seasonal Holt"
            future_dates = pd.Series(pd.date_range(dates.iloc[-1] + pd.Timedelta(weeks=1),
                                                     periods=FORECAST_HORIZON, freq="W-MON"))
            fc = fit_forecast_deseasonalized(dates, y, seasonal_index, FORECAST_HORIZON, future_dates)

            fc_plain = plain_holt_baseline(train_y, holdout_n)
            comparison_rows.append({
                "sku_id": sku,
                "plain_mae": float(np.mean(np.abs(fc_plain - test_y))),
                "plain_bias": float(np.mean(fc_plain - test_y)),
                "pooled_seasonal_mae": float(np.mean(np.abs(fc_bt - test_y))),
                "pooled_seasonal_bias": float(np.mean(fc_bt - test_y)),
            })

        mae = float(np.mean(np.abs(fc_bt - test_y)))
        bias = float(np.mean(fc_bt - test_y))
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

    banner("BACKTEST SUMMARY BY SEGMENT (pooled-seasonal method)")
    print(res.groupby("sb_class")[["backtest_mae", "backtest_bias"]].mean().round(2))

    if comparison_rows:
        cmp = pd.DataFrame(comparison_rows)
        banner("PLAIN HOLT (step 3) vs POOLED-SEASONAL HOLT (this script)")
        print(cmp[["plain_mae", "pooled_seasonal_mae"]].mean().round(2).rename("mean_MAE").to_string())
        print()
        print(cmp[["plain_bias", "pooled_seasonal_bias"]].mean().round(2).rename("mean_bias").to_string())
        improved_mae = (cmp["pooled_seasonal_mae"] < cmp["plain_mae"]).mean() * 100
        improved_bias = (cmp["pooled_seasonal_bias"].abs() < cmp["plain_bias"].abs()).mean() * 100
        print(f"\n% of Smooth/Erratic SKUs where pooled-seasonal MAE beat plain Holt: {improved_mae:.1f}%")
        print(f"% of Smooth/Erratic SKUs where pooled-seasonal |bias| beat plain Holt: {improved_bias:.1f}%")

    banner("DONE")
    print(f"Saved {OUT_INDEX}, {OUT_RESULTS} ({res.shape[0]} SKUs), {OUT_WEEKLY} ({weekly.shape[0]} rows)")


if __name__ == "__main__":
    main()
