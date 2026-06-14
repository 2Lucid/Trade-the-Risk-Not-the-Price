"""Phase 3 — The volatility-managed strategy (Moreira & Muir, 2017).

The managed excess return scales next month's market exposure by the inverse of
the *forecast* variance:

    f_managed[t] = ( c / sigma2_hat[t] ) * f[t]

where ``f[t]`` is the market excess return in month ``t`` and ``sigma2_hat[t]``
is the variance forecast for month ``t`` formed at the end of month ``t-1``
(strictly out-of-sample, built in :mod:`src.vol_forecast`).

* Calm market  -> low forecast variance -> large exposure.
* Nervous market -> high forecast variance -> we de-risk.

**The non-negotiable normalisation.**  The constant ``c`` is chosen so the
managed series has the *same full-sample volatility* as buy-and-hold over the
same months.  Without this, comparing Sharpe ratios would compare two different
risk levels and be dishonest.  ``c`` is a single full-sample scalar; it does not
use any cross-sectional future information beyond an overall scale (and a scale
leaves the Sharpe ratio and the regression t-statistics unchanged), exactly as
in Moreira-Muir.  The time-varying *weights* use only past information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _managed_vol(c: float, unscaled_w: np.ndarray, f: np.ndarray, cap: float) -> float:
    w = np.clip(c * unscaled_w, -cap, cap)
    return float(np.std(w * f, ddof=1))


def _solve_c(unscaled_w: np.ndarray, f: np.ndarray, target_vol: float, cap: float) -> float:
    """Solve for the scaling constant ``c`` so the managed vol == target_vol.

    Closed form when there is no leverage cap. With a cap, the search is bounded
    to ``[0, c_sat]`` where ``c_sat`` is the point at which *every* weight has
    saturated (beyond it the managed volatility is constant). For long-only
    positions the capped volatility is monotone in ``c`` and the bisection finds
    the exact root; for signed (trend) positions it can be non-monotone, so if
    the target is not bracketed we fall back to a grid search for the closest
    achievable volatility (no silent blow-up).
    """
    if not np.isfinite(cap):
        raw = unscaled_w * f
        return float(target_vol / np.std(raw, ddof=1))

    nz = np.abs(unscaled_w[unscaled_w != 0.0])
    c_sat = float(cap / nz.min()) if nz.size else 1.0  # all weights clipped beyond this

    if _managed_vol(c_sat, unscaled_w, f, cap) < target_vol:
        # target unreachable even at full saturation (or hidden behind a local
        # peak for signed positions): take the c with the closest achievable vol.
        grid = np.linspace(0.0, c_sat, 512)
        vols = np.array([_managed_vol(c, unscaled_w, f, cap) for c in grid])
        return float(grid[int(np.argmin(np.abs(vols - target_vol)))])

    lo, hi = 0.0, c_sat
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _managed_vol(mid, unscaled_w, f, cap) < target_vol:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_managed_strategy(
    monthly: pd.DataFrame,
    var_forecast: pd.Series,
    leverage_cap: float = np.inf,
    target_vol: float | None = None,
    signal: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the volatility-managed strategy from a variance-forecast series.

    Parameters
    ----------
    monthly        : monthly dataset (needs ``mkt_excess``).
    var_forecast   : variance forecast for month ``t`` formed at end of ``t-1``
                     (index = target month). NaNs (warm-up) are dropped.
    leverage_cap   : maximum |exposure| as a multiple of full market exposure.
    target_vol     : volatility to match; defaults to the buy-and-hold vol over
                     the strategy sample.
    signal         : optional directional signal in {-1, +1} (e.g. a trend
                     overlay) known at the end of ``t-1``; defaults to +1
                     (pure long volatility-managed, Moreira-Muir). With a signal
                     the position is signed and clipped to [-cap, +cap].

    Returns
    -------
    (df, info) where ``df`` has columns
        f            buy-and-hold market excess return
        var_forecast forecast variance used for sizing
        weight       realised (signed) exposure (after the leverage cap)
        managed      managed excess return = weight * f  (gross of costs)
    and ``info`` carries ``c``, ``target_vol``, ``achieved_vol``, ``cap``.
    """
    cols = {"f": monthly["mkt_excess"], "var_forecast": var_forecast}
    if signal is not None:
        cols["signal"] = signal
    df = pd.DataFrame(cols).dropna()
    f = df["f"].values.astype(float)
    sig = df["signal"].values.astype(float) if signal is not None else np.ones(len(df))
    unscaled_w = (sig / df["var_forecast"].values).astype(float)

    if target_vol is None:
        target_vol = float(np.std(f, ddof=1))  # buy-and-hold vol over same months

    c = _solve_c(unscaled_w, f, target_vol, leverage_cap)
    weight = np.clip(c * unscaled_w, -leverage_cap, leverage_cap)
    managed = weight * f

    df["weight"] = weight
    df["managed"] = managed
    info = {
        "c": float(c),
        "target_vol": float(target_vol),
        "achieved_vol": float(np.std(managed, ddof=1)),
        "buyhold_vol": float(np.std(f, ddof=1)),
        "cap": float(leverage_cap),
        "mean_weight": float(np.mean(weight)),          # net (signed) exposure
        "mean_abs_weight": float(np.mean(np.abs(weight))),  # gross exposure
        "max_weight": float(np.max(np.abs(weight))),    # peak |exposure|
        "sample_start": df.index.min(),
        "sample_end": df.index.max(),
        "n_months": int(len(df)),
    }
    return df, info


def turnover(weights: pd.Series) -> pd.Series:
    """Per-period turnover = |w[t] - w[t-1]| (first period vs a 1x start)."""
    prev = weights.shift(1)
    prev.iloc[0] = 1.0  # assume we start from full (1x) market exposure
    return (weights - prev).abs()


def apply_costs(strat: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Net managed return after proportional transaction costs on turnover."""
    tc = cost_bps / 1e4
    cost = tc * turnover(strat["weight"])
    return strat["managed"] - cost


if __name__ == "__main__":  # pragma: no cover
    from src.data_loader import load_dataset
    from src.vol_forecast import forecast_naive

    _, m = load_dataset()
    fc = forecast_naive(m)
    strat, info = build_managed_strategy(m, fc)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})
    print(strat.head())
