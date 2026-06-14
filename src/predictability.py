"""Phase 1 — The honesty test (the "null result").

We show what *fails* before what works. Using only past information in an
expanding-window, out-of-sample exercise we try to predict:

* next month's **return**     -> out-of-sample R^2 is ~0 or negative (price
  direction is noise / markets are efficient), and
* next month's **volatility** -> out-of-sample R^2 is large (0.3-0.6): risk is
  highly predictable because of volatility clustering.

The benchmark is the prevailing (expanding) historical mean — i.e. the
Campbell & Thompson (2008) out-of-sample R^2.  For the return models we also
report the Clark & West (2007) statistic, the correct test for nested
forecast comparisons.

No future information is ever used to fit a model or build a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_TRAIN = 120  # 10 years of monthly data before the first out-of-sample point


# --------------------------------------------------------------------------- #
# Core walk-forward machinery                                                  #
# --------------------------------------------------------------------------- #
def _expanding_ols_forecasts(
    y: pd.Series, X: pd.DataFrame, min_train: int = MIN_TRAIN
) -> pd.Series:
    """One-step-ahead OLS forecasts from an expanding window (no lookahead).

    To forecast ``y[t]`` we fit on observations ``s < t`` only, using
    predictors ``X`` that are already lagged (known strictly before ``y[t]``).
    """
    yv = y.values.astype(float)
    Xv = X.values.astype(float)
    n = len(y)
    preds = np.full(n, np.nan)
    for t in range(min_train, n):
        Xtr = np.column_stack([np.ones(t), Xv[:t]])
        beta, *_ = np.linalg.lstsq(Xtr, yv[:t], rcond=None)
        preds[t] = np.r_[1.0, Xv[t]] @ beta
    return pd.Series(preds, index=y.index)


def _expanding_mean_forecasts(y: pd.Series, min_train: int = MIN_TRAIN) -> pd.Series:
    """Prevailing historical-mean benchmark forecasts."""
    yv = y.values.astype(float)
    n = len(y)
    preds = np.full(n, np.nan)
    for t in range(min_train, n):
        preds[t] = yv[:t].mean()
    return pd.Series(preds, index=y.index)


def oos_r2(y: pd.Series, f_model: pd.Series, f_bench: pd.Series) -> float:
    """Campbell-Thompson out-of-sample R^2 (model vs prevailing mean)."""
    mask = f_model.notna() & f_bench.notna() & y.notna()
    e_m = (y[mask] - f_model[mask]) ** 2
    e_b = (y[mask] - f_bench[mask]) ** 2
    return float(1.0 - e_m.sum() / e_b.sum())


def clark_west(y: pd.Series, f_model: pd.Series, f_bench: pd.Series) -> tuple[float, float]:
    """Clark-West (2007) statistic for nested OOS comparison; returns (stat, p).

    H0: the larger model does not improve on the historical-mean benchmark.
    One-sided p-value (improvement) based on a HAC-robust mean of the adjusted
    loss differential.
    """
    mask = f_model.notna() & f_bench.notna() & y.notna()
    yv, fm, fb = y[mask].values, f_model[mask].values, f_bench[mask].values
    f_adj = (yv - fb) ** 2 - ((yv - fm) ** 2 - (fb - fm) ** 2)
    fbar = f_adj.mean()
    # Newey-West (lag 0 is fine for monthly 1-step; use small lag for safety).
    se = _nw_se_mean(f_adj, lags=3)
    stat = fbar / se if se > 0 else np.nan
    p = 1.0 - stats.norm.cdf(stat)
    return float(stat), float(p)


def _nw_se_mean(x: np.ndarray, lags: int = 3) -> float:
    """Newey-West HAC standard error of the sample mean."""
    x = x - x.mean()
    n = len(x)
    gamma0 = np.dot(x, x) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        gk = np.dot(x[k:], x[:-k]) / n
        var += 2 * w * gk
    return float(np.sqrt(var / n))


# --------------------------------------------------------------------------- #
# Feature construction (all strictly lagged => known before the target)       #
# --------------------------------------------------------------------------- #
def _return_features(monthly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r = monthly["mkt_excess"]
    rv = monthly["rv"]
    feats = {
        "AR(1)": pd.DataFrame({"r_lag1": r.shift(1)}),
        "Momentum (3m)": pd.DataFrame({"mom3": r.rolling(3).sum().shift(1)}),
        "Lagged variance": pd.DataFrame({"rv_lag1": rv.shift(1)}),
        "Kitchen sink": pd.DataFrame(
            {
                "r_lag1": r.shift(1),
                "mom3": r.rolling(3).sum().shift(1),
                "rv_lag1": rv.shift(1),
            }
        ),
    }
    return feats


def _vol_features(monthly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lrv = np.log(monthly["rv"])
    feats = {
        "AR(1) log-RV": pd.DataFrame({"lrv_lag1": lrv.shift(1)}),
        "HAR (1/3/12m)": pd.DataFrame(
            {
                "lrv_lag1": lrv.shift(1),
                "lrv_q": lrv.rolling(3).mean().shift(1),
                "lrv_y": lrv.rolling(12).mean().shift(1),
            }
        ),
    }
    return feats


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def run_predictability(monthly: pd.DataFrame) -> dict:
    """Run the full return-vs-volatility OOS predictability comparison.

    Returns a dict with a tidy results table and the realized/forecast series
    for the headline volatility model (for plotting).
    """
    rows = []

    # ---- Returns: target = next month's excess return ---------------------- #
    # The prevailing-mean benchmark is recomputed on each feature's `valid`
    # subset so the nested pair (mean vs OLS) shares an identical estimation
    # window at every origin (clean Clark-West comparison).
    y_ret = monthly["mkt_excess"]
    for name, X in _return_features(monthly).items():
        X = X.reindex(y_ret.index)
        valid = X.dropna().index
        fm = _expanding_ols_forecasts(y_ret.loc[valid], X.loc[valid])
        fb = _expanding_mean_forecasts(y_ret.loc[valid])
        r2 = oos_r2(y_ret.loc[valid], fm, fb)
        cw_stat, cw_p = clark_west(y_ret.loc[valid], fm, fb)
        rows.append(
            {
                "Target": "Return",
                "Model": name,
                "R2_OOS": r2,
                "CW_stat": cw_stat,
                "CW_pvalue": cw_p,
            }
        )

    # ---- Volatility: target = log realized variance ------------------------ #
    y_vol = np.log(monthly["rv"])
    vol_series = None
    for name, X in _vol_features(monthly).items():
        X = X.reindex(y_vol.index)
        valid = X.dropna().index
        fm = _expanding_ols_forecasts(y_vol.loc[valid], X.loc[valid])
        fb = _expanding_mean_forecasts(y_vol.loc[valid])
        r2 = oos_r2(y_vol.loc[valid], fm, fb)
        rows.append(
            {
                "Target": "Volatility (log-RV)",
                "Model": name,
                "R2_OOS": r2,
                "CW_stat": np.nan,
                "CW_pvalue": np.nan,
            }
        )
        if name == "HAR (1/3/12m)":
            # store forecast vs realized (in variance units) for the figure
            mask = fm.notna()
            vol_series = pd.DataFrame(
                {
                    "realized_rv": np.exp(y_vol.loc[valid][mask]),
                    "forecast_rv": np.exp(fm[mask]),
                }
            )

    table = pd.DataFrame(rows)
    return {"table": table, "vol_forecast_vs_realized": vol_series}


if __name__ == "__main__":  # pragma: no cover
    from src.data_loader import load_dataset

    _, m = load_dataset()
    out = run_predictability(m)
    print(out["table"].round(4).to_string(index=False))
