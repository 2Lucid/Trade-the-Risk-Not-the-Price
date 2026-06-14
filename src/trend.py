"""Trend overlay (extension) — time-series momentum.

Time-series momentum (Moskowitz, Ooi & Pedersen, 2012) is the classic companion
to volatility timing: take a *long* position when the asset's own trailing return
is positive and a *short* one when it is negative. It complements the
volatility-managed sizing — one decides *how big*, the other decides *which way*
— and together they are the core of the managed-futures / CTA recipe.

The signal for month ``t`` uses only returns through the end of month ``t-1``
(strictly lagged), so there is no lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import build_managed_strategy

LOOKBACK = 12  # months of trailing return used for the trend sign


def tsmom_signal(monthly_excess: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    """Time-series-momentum sign in {-1, 0, +1}, known at the end of month t-1.

    ``signal[t] = sign( cumulative excess return over months t-lookback .. t-1 )``.
    """
    trailing = (1.0 + monthly_excess).rolling(lookback).apply(np.prod, raw=True) - 1.0
    return np.sign(trailing.shift(1)).rename("signal")


def build_trend_managed(
    monthly: pd.DataFrame,
    var_forecast: pd.Series,
    leverage_cap: float = 2.0,
    lookback: int = LOOKBACK,
) -> tuple[pd.DataFrame, dict]:
    """Volatility-managed sizing combined with a time-series-momentum direction."""
    sig = tsmom_signal(monthly["mkt_excess"], lookback)
    return build_managed_strategy(
        monthly, var_forecast, leverage_cap=leverage_cap, signal=sig
    )


if __name__ == "__main__":  # pragma: no cover
    from src.data_loader import load_dataset
    from src.evaluation import alpha_regression, performance_metrics
    from src.vol_forecast import forecast_naive

    _, m = load_dataset()
    fc = forecast_naive(m)
    strat, info = build_trend_managed(m, fc, leverage_cap=2.0)
    print("trend+vol-managed:", {k: round(v, 3) for k, v in performance_metrics(strat["managed"]).items() if isinstance(v, float)})
    print("net exposure mean:", round(info["mean_weight"], 3), " gross:", round(info["mean_abs_weight"], 3))
    print("alpha:", {k: round(v, 4) for k, v in alpha_regression(strat["managed"], strat["f"]).items()})
