"""Phase 2 — The volatility-forecasting engine.

Target: monthly **realized variance** ``RV_{t+1} = sum_{d in month t+1} r_d^2``.

Three one-step-ahead forecasters, evaluated walk-forward (expanding window,
strictly no lookahead — a forecast for month ``t+1`` only ever uses information
available up to the end of month ``t``):

1. **Naive (random walk)** — ``hat{RV}_{t+1} = RV_t``.  Exactly the estimator in
   Moreira & Muir (2017): even the simplest predictor works.
2. **GARCH(1,1)** — fit on daily excess returns up to the end of month ``t``,
   analytically forecast the next ~21 daily variances, and sum them to a monthly
   variance.  (Engle, Nobel 2003.)
3. **HAR-RV** — Corsi (2009) heterogeneous auto-regression, here at monthly
   frequency with 1-/3-/12-month components, fit by expanding-window OLS.

We also build a simple **ensemble** (mean of GARCH and HAR), since averaging good
forecasts often beats either one.

Accuracy is judged with **RMSE** *and* **QLIKE** (Patton 2011, the loss that is
robust to noise in the volatility proxy), and differences are tested with the
**Diebold-Mariano** test (Harvey-Leybourne-Newbold small-sample correction).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src import config

HAR_MIN_TRAIN = 36          # months of OLS history before first HAR forecast
GARCH_MIN_DAILY = 500       # ~2 years of daily obs before first GARCH fit
GARCH_HORIZON = config.DAYS_PER_MONTH


# --------------------------------------------------------------------------- #
# Individual forecasters (each returns a Series indexed by the TARGET month)   #
# --------------------------------------------------------------------------- #
def forecast_naive(monthly: pd.DataFrame) -> pd.Series:
    """Random-walk forecast: variance of month t+1 = realized variance of t."""
    return monthly["rv"].shift(1).rename("naive")


def forecast_har(monthly: pd.DataFrame, min_train: int = HAR_MIN_TRAIN) -> pd.Series:
    """Monthly HAR-RV forecast by expanding-window OLS (no lookahead).

    Features for target month ``i`` use only ``RV`` up to month ``i-1``:
    last month, last-quarter mean, last-year mean.
    """
    rv = monthly["rv"]
    F = pd.DataFrame(
        {
            "m": rv.shift(1),
            "q": rv.rolling(3).mean().shift(1),
            "y": rv.rolling(12).mean().shift(1),
        }
    )
    yv = rv.values.astype(float)
    Fv = F.values.astype(float)
    n = len(rv)
    preds = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(Fv[i]).any():
            continue
        # training rows j < i with complete features
        train = [j for j in range(i) if not np.isnan(Fv[j]).any()]
        if len(train) < min_train:
            continue
        Xtr = np.column_stack([np.ones(len(train)), Fv[train]])
        beta, *_ = np.linalg.lstsq(Xtr, yv[train], rcond=None)
        preds[i] = np.r_[1.0, Fv[i]] @ beta
    preds = np.maximum(preds, 1e-10)  # variance must be positive
    return pd.Series(preds, index=rv.index, name="har")


def forecast_garch(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    refit_every: int = 1,
    horizon: int = GARCH_HORIZON,
    verbose: bool = False,
) -> pd.Series:
    """GARCH(1,1) monthly-variance forecast, refit walk-forward on daily data.

    For target month ``i`` we fit on all daily excess returns up to the end of
    month ``i-1`` and sum the analytic 1..H day-ahead variance forecasts.
    Returns in percent are used for numerical stability, then converted back.
    """
    from arch import arch_model

    r_pct = (daily["mkt_excess"] * 100.0).dropna()
    month_ends = monthly.index
    n = len(month_ends)
    n_days = monthly["n_days"].values  # trading days per month (calendar info)
    preds = np.full(n, np.nan)

    last_fit_idx = -10_000
    cached_params = None
    for i in range(1, n):
        decision_date = month_ends[i - 1]  # info available up to end of month i-1
        hist = r_pct.loc[:decision_date]
        if len(hist) < GARCH_MIN_DAILY:
            continue
        try:
            am = arch_model(hist, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
            if cached_params is None or (i - last_fit_idx) >= refit_every:
                # full re-estimation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = am.fit(disp="off", show_warning=False)
                cached_params = res.params
                last_fit_idx = i
            else:
                # re-filter the latest data with the most recent parameters
                # (no optimisation) -- correct AND fast
                res = am.fix(cached_params)
            fc = res.forecast(horizon=horizon, reindex=False)
            var_pct2 = np.asarray(fc.variance.values[-1, :])
            # Scale the average daily forecast to the target month's actual
            # trading-day count so the monthly forecast matches the realized-
            # variance horizon (sum over the month's days). n_days is calendar
            # information, deterministic and known ex ante -- not return lookahead.
            preds[i] = float(var_pct2.mean()) * float(n_days[i]) / (100.0 ** 2)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeError) as exc:
            # Genuine optimiser/convergence failure: leave NaN so the month is
            # excluded from the common evaluation sample (no silent substitution).
            # Unexpected exceptions (KeyError/IndexError/...) are NOT caught so
            # real bugs surface immediately.
            print(f"WARNING: GARCH fit failed at {decision_date.date()}: {exc}")
            preds[i] = np.nan
    return pd.Series(preds, index=month_ends, name="garch")


# --------------------------------------------------------------------------- #
# Loss functions                                                              #
# --------------------------------------------------------------------------- #
def rmse(realized: pd.Series, forecast: pd.Series) -> float:
    e = (forecast - realized).dropna()
    return float(np.sqrt(np.mean(e.values ** 2)))


def qlike(realized: pd.Series, forecast: pd.Series) -> float:
    """Patton (2011) QLIKE loss: realized/forecast - log(realized/forecast) - 1."""
    df = pd.concat([realized, forecast], axis=1).dropna()
    r, f = df.iloc[:, 0].values, df.iloc[:, 1].values
    ratio = r / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _loss_series(realized: pd.Series, forecast: pd.Series, kind: str) -> pd.Series:
    df = pd.concat([realized, forecast], axis=1).dropna()
    r, f = df.iloc[:, 0], df.iloc[:, 1]
    if kind == "se":
        return (f - r) ** 2
    if kind == "qlike":
        ratio = r / f
        return ratio - np.log(ratio) - 1.0
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Diebold-Mariano test                                                         #
# --------------------------------------------------------------------------- #
def diebold_mariano(loss1: pd.Series, loss2: pd.Series, h: int = 1) -> tuple[float, float]:
    """DM test of equal predictive accuracy with HLN small-sample correction.

    Positive statistic => model 1 has the *larger* loss (model 2 is better).
    Returns (stat, two-sided p-value).
    """
    d = (loss1 - loss2).dropna()
    n = len(d)
    dbar = d.mean()
    dv = (d - dbar).values
    gamma0 = np.dot(dv, dv) / n
    var = gamma0
    for k in range(1, h):
        gk = np.dot(dv[k:], dv[:-k]) / n
        var += 2 * gk
    var_dbar = var / n
    if var_dbar <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var_dbar)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= hln
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run_vol_forecasts(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    refit_every: int = 1,
    verbose: bool = True,
) -> dict:
    """Produce all forecasts, align them, and evaluate (RMSE/QLIKE/DM)."""
    realized = monthly["rv"].rename("realized")

    if verbose:
        print("  - naive forecast ...")
    f_naive = forecast_naive(monthly)
    if verbose:
        print("  - HAR forecast ...")
    f_har = forecast_har(monthly)
    if verbose:
        print("  - GARCH walk-forward (this is the slow step) ...")
    f_garch = forecast_garch(daily, monthly, refit_every=refit_every, verbose=verbose)

    forecasts = pd.concat([realized, f_naive, f_har, f_garch], axis=1)
    forecasts["ensemble"] = forecasts[["har", "garch"]].mean(axis=1)

    # Common evaluation sample: months where every model has a forecast.
    model_cols = ["naive", "har", "garch", "ensemble"]
    common = forecasts.dropna(subset=["realized", *model_cols])

    # Metrics table
    metrics = pd.DataFrame(
        {
            "RMSE": {m: rmse(common["realized"], common[m]) for m in model_cols},
            "QLIKE": {m: qlike(common["realized"], common[m]) for m in model_cols},
        }
    )
    metrics.index.name = "Model"

    # DM tests vs the naive baseline and HAR-vs-GARCH, under both losses.
    pairs = [("garch", "naive"), ("har", "naive"), ("ensemble", "naive"), ("har", "garch")]
    dm_rows = []
    for a, b in pairs:
        for kind, label in [("se", "RMSE/MSE"), ("qlike", "QLIKE")]:
            la = _loss_series(common["realized"], common[a], kind)
            lb = _loss_series(common["realized"], common[b], kind)
            stat, p = diebold_mariano(la, lb, h=1)
            dm_rows.append(
                {
                    "Comparison": f"{a} vs {b}",
                    "Loss": label,
                    "DM_stat": stat,
                    "p_value": p,
                    "better": a if stat < 0 else b,
                }
            )
    dm_table = pd.DataFrame(dm_rows)

    return {
        "forecasts": forecasts,
        "common": common,
        "metrics": metrics,
        "dm_table": dm_table,
        "eval_start": common.index.min(),
        "eval_end": common.index.max(),
        "n_eval": len(common),
    }


if __name__ == "__main__":  # pragma: no cover
    from src.data_loader import load_dataset

    d, m = load_dataset()
    out = run_vol_forecasts(d, m, refit_every=1)
    print(f"\nEval sample: {out['eval_start'].date()} -> {out['eval_end'].date()} "
          f"({out['n_eval']} months)\n")
    print(out["metrics"].round(6))
    print()
    print(out["dm_table"].round(4).to_string(index=False))
