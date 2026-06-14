"""Phase 4 — Honest evaluation.

The headline test (Moreira & Muir, 2017) regresses the managed series on
buy-and-hold:

    f_managed[t] = alpha + beta * f[t] + eps[t]

A positive, significant ``alpha`` is return not explained by simple market
exposure; the **appraisal ratio** ``alpha / std(eps)`` says how much the Sharpe
ratio can rise.  We use Newey-West (HAC) standard errors throughout.

We then add the reality checks most LinkedIn projects skip:

* **Transaction costs** on turnover, with a break-even cost level.
* **Leverage caps** (re-normalising vol under each cap).
* **Sub-period analysis** (dot-com, GFC, COVID, calm spells).
* **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014) — corrects the
  Sharpe ratio for non-normality, sample length and the number of strategies
  tried, pre-empting the "you just data-mined" objection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src import config
from src.strategy import apply_costs, build_managed_strategy, turnover

MONTHS = config.MONTHS_PER_YEAR
NW_LAGS = 6  # Newey-West lags for monthly data


# --------------------------------------------------------------------------- #
# Core performance metrics                                                     #
# --------------------------------------------------------------------------- #
def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def performance_metrics(returns: pd.Series, name: str = "strategy") -> dict:
    """Annualised performance statistics for a monthly *excess*-return series."""
    r = returns.dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    # Target downside deviation (MAR = 0): RMS of shortfalls below the target,
    # averaged over ALL periods (Sortino & van der Meer 1991) -- not the std of
    # the negative subset.
    downside = float(np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2)))
    cagr = float((1.0 + r).prod() ** (MONTHS / len(r)) - 1.0)
    mdd = max_drawdown(r)
    return {
        "name": name,
        "ann_return": float(mu * MONTHS),
        "cagr": cagr,
        "ann_vol": float(sd * np.sqrt(MONTHS)),
        "sharpe": float(mu / sd * np.sqrt(MONTHS)) if sd > 0 else np.nan,
        "sortino": float(mu / downside * np.sqrt(MONTHS)) if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
        "n_months": int(len(r)),
    }


# --------------------------------------------------------------------------- #
# Alpha regression (Newey-West)                                               #
# --------------------------------------------------------------------------- #
def alpha_regression(managed: pd.Series, buyhold: pd.Series) -> dict:
    """Regress managed on buy-and-hold with HAC (Newey-West) standard errors."""
    df = pd.concat([managed.rename("y"), buyhold.rename("x")], axis=1).dropna()
    X = sm.add_constant(df["x"])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    alpha_m, beta = res.params["const"], res.params["x"]
    resid_sd = res.resid.std(ddof=1)
    return {
        "alpha_monthly": float(alpha_m),
        "alpha_annual": float(alpha_m * MONTHS),
        "alpha_tstat": float(res.tvalues["const"]),
        "alpha_pvalue": float(res.pvalues["const"]),
        "beta": float(beta),
        "beta_tstat": float(res.tvalues["x"]),
        "r_squared": float(res.rsquared),
        # appraisal / information ratio, annualised
        "appraisal_ratio": float(alpha_m / resid_sd * np.sqrt(MONTHS)) if resid_sd > 0 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Probabilistic / Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)        #
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    sr_per_period: float, T: int, skew: float, ex_kurt: float, sr_benchmark: float = 0.0
) -> float:
    """P(true SR > benchmark) given non-normality and sample length (per-period SR)."""
    kurt = ex_kurt + 3.0
    denom = np.sqrt(1.0 - skew * sr_per_period + (kurt - 1.0) / 4.0 * sr_per_period ** 2)
    z = (sr_per_period - sr_benchmark) * np.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum (per-period) Sharpe across ``n_trials`` independent trials."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    e = np.e
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(sr_variance) * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None
) -> dict:
    """Deflated Sharpe Ratio: PSR against the data-mining-adjusted benchmark.

    ``trial_sharpes`` are the per-period Sharpe ratios of all strategy variants
    tried (used to estimate the cross-trial Sharpe variance ``V``).  If not
    given, a conservative default variance of 1/(T-1) is used.
    """
    r = returns.dropna()
    T = len(r)
    sr = float(r.mean() / r.std(ddof=1))  # per-period Sharpe
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sr_variance = float(np.var(trial_sharpes, ddof=1))
    else:
        sr_variance = 1.0 / (T - 1)
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    dsr = probabilistic_sharpe_ratio(sr, T, r.skew(), r.kurt(), sr_benchmark=sr0)
    psr0 = probabilistic_sharpe_ratio(sr, T, r.skew(), r.kurt(), sr_benchmark=0.0)
    return {
        "sharpe_per_period": sr,
        "sharpe_annual": float(sr * np.sqrt(MONTHS)),
        "n_trials": n_trials,
        "sr_benchmark_per_period": sr0,
        "psr_vs_zero": psr0,
        "deflated_sharpe": dsr,
    }


# --------------------------------------------------------------------------- #
# Robustness: transaction costs, leverage caps, sub-periods                    #
# --------------------------------------------------------------------------- #
def cost_sensitivity(strat: pd.DataFrame, cost_grid=config.COST_GRID_BPS) -> pd.DataFrame:
    """Net Sharpe / annual alpha across a grid of transaction-cost levels."""
    rows = []
    avg_turn = float(turnover(strat["weight"]).mean())
    for bps in cost_grid:
        net = apply_costs(strat, bps)
        m = performance_metrics(net, name=f"{bps}bps")
        a = alpha_regression(net, strat["f"])
        rows.append(
            {
                "cost_bps": bps,
                "avg_turnover": avg_turn,
                "ann_return": m["ann_return"],
                "sharpe": m["sharpe"],
                "alpha_annual": a["alpha_annual"],
                "alpha_tstat": a["alpha_tstat"],
            }
        )
    return pd.DataFrame(rows)


def breakeven_cost(strat: pd.DataFrame, hi_bps: float = 200.0) -> float:
    """Transaction cost (bps) at which managed Sharpe equals buy-and-hold Sharpe."""
    bh = performance_metrics(strat["f"])["sharpe"]
    lo, hi = 0.0, hi_bps
    if performance_metrics(apply_costs(strat, lo))["sharpe"] < bh:
        return 0.0
    if performance_metrics(apply_costs(strat, hi))["sharpe"] > bh:
        return float("inf")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if performance_metrics(apply_costs(strat, mid))["sharpe"] > bh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def leverage_sensitivity(
    monthly: pd.DataFrame, var_forecast: pd.Series, caps=config.LEVERAGE_CAPS
) -> pd.DataFrame:
    """Performance under each leverage cap (vol re-normalised within each cap)."""
    rows = []
    for cap in caps:
        strat, info = build_managed_strategy(monthly, var_forecast, leverage_cap=cap)
        m = performance_metrics(strat["managed"])
        a = alpha_regression(strat["managed"], strat["f"])
        # cap=1 with vol-matching degenerates to buy-and-hold (weight==1 a.s.),
        # making alpha==0 with a meaningless, near-singular t-stat. Suppress it.
        if info["max_weight"] <= 1.0 + 1e-9:
            a = {**a, "alpha_annual": 0.0, "alpha_tstat": np.nan}
        rows.append(
            {
                "leverage_cap": cap,
                "mean_weight": info["mean_weight"],
                "max_weight": info["max_weight"],
                "ann_return": m["ann_return"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "alpha_annual": a["alpha_annual"],
                "alpha_tstat": a["alpha_tstat"],
            }
        )
    return pd.DataFrame(rows)


def subperiod_analysis(
    strat: pd.DataFrame, subperiods=config.SUBPERIODS
) -> pd.DataFrame:
    """Managed vs buy-and-hold Sharpe and return by sub-period."""
    rows = []
    for label, (s, e) in subperiods.items():
        seg = strat.loc[(strat.index >= s) & (strat.index <= e)]
        if len(seg) < 6:
            continue
        mm = performance_metrics(seg["managed"])
        bh = performance_metrics(seg["f"])
        rows.append(
            {
                "period": label,
                "start": seg.index.min().date(),
                "end": seg.index.max().date(),
                "n": len(seg),
                "bh_return": bh["ann_return"],
                "bh_sharpe": bh["sharpe"],
                "mm_return": mm["ann_return"],
                "mm_sharpe": mm["sharpe"],
                "sharpe_gain": mm["sharpe"] - bh["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def compare_forecasters(
    monthly: pd.DataFrame, forecasts: pd.DataFrame, model_cols: list[str]
) -> pd.DataFrame:
    """Managed-strategy performance for each volatility forecaster.

    All strategies are built on the *common* sample where every forecaster is
    available, so the comparison is apples-to-apples.
    """
    common = forecasts.dropna(subset=model_cols)
    rows = []
    for col in model_cols:
        strat, info = build_managed_strategy(
            monthly.loc[common.index], common[col]
        )
        m = performance_metrics(strat["managed"])
        a = alpha_regression(strat["managed"], strat["f"])
        rows.append(
            {
                "forecaster": col,
                "ann_return": m["ann_return"],
                "ann_vol": m["ann_vol"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "alpha_annual": a["alpha_annual"],
                "alpha_tstat": a["alpha_tstat"],
                "appraisal_ratio": a["appraisal_ratio"],
            }
        )
    # buy-and-hold reference on the same sample
    bh = performance_metrics(monthly.loc[common.index, "mkt_excess"].rename("f"))
    rows.append(
        {
            "forecaster": "buy_and_hold",
            "ann_return": bh["ann_return"],
            "ann_vol": bh["ann_vol"],
            "sharpe": bh["sharpe"],
            "max_drawdown": bh["max_drawdown"],
            "alpha_annual": 0.0,
            "alpha_tstat": np.nan,
            "appraisal_ratio": np.nan,
        }
    )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    from src.data_loader import load_dataset
    from src.vol_forecast import forecast_naive

    _, m = load_dataset()
    fc = forecast_naive(m)
    strat, info = build_managed_strategy(m, fc)
    print("Managed:", {k: round(v, 3) for k, v in performance_metrics(strat["managed"]).items() if isinstance(v, float)})
    print("BuyHold:", {k: round(v, 3) for k, v in performance_metrics(strat["f"]).items() if isinstance(v, float)})
    print("Alpha:  ", {k: round(v, 4) for k, v in alpha_regression(strat["managed"], strat["f"]).items()})
    print("DSR:    ", {k: round(v, 4) if isinstance(v, float) else v for k, v in deflated_sharpe_ratio(strat["managed"], 3).items()})
