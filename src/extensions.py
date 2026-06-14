"""Extension — cross-asset robustness and a diversified trend overlay.

Two questions beyond the single US market:

1. **Does the principle travel?** We re-run the volatility-managed strategy
   (naive realized-variance forecast, 2x cap, vol-matched) on every asset in the
   panel and check whether the Sharpe improvement holds out-of-US and
   cross-asset.
2. **Does combining help?** We build a diversified portfolio by inverse-volatility
   (risk-parity-style) weighting across assets — with weights formed from
   *trailing* volatility, so no lookahead — and compare buy-and-hold,
   volatility-managed, and volatility-managed + time-series-momentum versions,
   all vol-matched to the diversified buy-and-hold benchmark.

Diversification and risk control — not a secret signal — are where the durable
gains come from; this section quantifies that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import alpha_regression, performance_metrics
from src.multi_asset import ASSET_UNIVERSE, build_panel
from src.strategy import apply_costs, build_managed_strategy
from src.trend import build_trend_managed
from src.vol_forecast import forecast_naive

LEVERAGE_CAP = 2.0
COST_BPS = 10.0       # realistic per-unit-turnover cost for the net-of-cost figures
MIN_ASSETS = 3        # require >=3 assets before the diversified portfolio starts
IVOL_LOOKBACK = 12    # months of trailing vol for the inverse-vol weights


def _inverse_vol_weights(
    F: pd.DataFrame, avail: pd.DataFrame | None = None, lookback: int = IVOL_LOOKBACK
) -> pd.DataFrame:
    """Risk-parity-style weights from *trailing* volatility (lagged, no lookahead).

    Trailing vol is always estimated from the buy-and-hold panel ``F``; ``avail``
    (defaults to ``F``) is the sleeve whose actual availability sets which assets
    receive weight, so the weights renormalise to 1 over the assets a given sleeve
    can actually trade (the trend sleeve loses extra months to its 12m warm-up).
    """
    vol = F.rolling(lookback, min_periods=6).std()
    inv = (1.0 / vol).shift(1)          # weight for month t uses vols through t-1
    mask = avail if avail is not None else F
    inv = inv.where(mask.notna())       # only assets the sleeve actually trades
    return inv.div(inv.sum(axis=1), axis=0)


def _combine(weights: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    return (weights * panel).sum(axis=1, min_count=1)


def _vol_match(s: pd.Series, target_vol: float) -> pd.Series:
    return s * (target_vol / s.std(ddof=1))


def run_extensions(
    daily_ff: pd.DataFrame,
    monthly_ff: pd.DataFrame,
    leverage_cap: float = LEVERAGE_CAP,
    cost_bps: float = COST_BPS,
    rebuild: bool = False,
) -> dict:
    panel = build_panel(daily_ff, rebuild=rebuild)
    monthly_assets = panel["monthly"]

    Fc, Mc, MTc, Mnet, MTnet = {}, {}, {}, {}, {}
    cross_rows = []
    for tkr, ma in monthly_assets.items():
        fc = forecast_naive(ma)
        sm, _ = build_managed_strategy(ma, fc, leverage_cap=leverage_cap)
        st, _ = build_trend_managed(ma, fc, leverage_cap=leverage_cap)
        Fc[tkr], Mc[tkr], MTc[tkr] = sm["f"], sm["managed"], st["managed"]
        Mnet[tkr] = apply_costs(sm, cost_bps)
        MTnet[tkr] = apply_costs(st, cost_bps)

        name, klass = ASSET_UNIVERSE[tkr]
        bh, mm, tt = (performance_metrics(sm["f"]), performance_metrics(sm["managed"]),
                      performance_metrics(st["managed"]))
        am = alpha_regression(sm["managed"], sm["f"])
        at = alpha_regression(st["managed"], st["f"])
        cross_rows.append(
            {
                "asset": name, "class": klass, "n_months": len(sm),
                "bh_sharpe": bh["sharpe"], "mm_sharpe": mm["sharpe"], "mt_sharpe": tt["sharpe"],
                "mm_alpha_t": am["alpha_tstat"], "mt_alpha_t": at["alpha_tstat"],
            }
        )
    cross = pd.DataFrame(cross_rows)

    F = pd.DataFrame(Fc)
    M, MT = pd.DataFrame(Mc), pd.DataFrame(MTc)
    Mn, MTn = pd.DataFrame(Mnet), pd.DataFrame(MTnet)

    # Diversified portfolios (same risk-parity weights for every sleeve type,
    # so the only difference is buy-and-hold vs managed vs managed+trend).
    # F and the managed sleeve M share a footprint; the trend sleeves (MT, MTn)
    # lose extra months to the 12m trend warm-up, so they get their own weights
    # renormalised over the assets they can actually trade (no silent under-
    # investment in the months a newly-eligible asset's trend is still warming up).
    W_f = _inverse_vol_weights(F)
    W_mt = _inverse_vol_weights(F, MT)
    n_assets = F.notna().sum(axis=1)
    keep = n_assets[n_assets >= MIN_ASSETS].index

    def combo(weights, p):
        return _combine(weights, p).reindex(keep).dropna()

    div_bh, div_m, div_mt = combo(W_f, F), combo(W_f, M), combo(W_mt, MT)
    div_mn, div_mtn = combo(W_f, Mn), combo(W_mt, MTn)
    idx = div_bh.index.intersection(div_m.index).intersection(div_mt.index)
    div_bh, div_m, div_mt = div_bh[idx], div_m[idx], div_mt[idx]
    div_mn, div_mtn = div_mn.reindex(idx), div_mtn.reindex(idx)

    # Vol-match the gross managed/trend portfolios to the diversified BH.
    tgt = div_bh.std(ddof=1)
    div_m_v, div_mt_v = _vol_match(div_m, tgt), _vol_match(div_mt, tgt)

    # US-only volatility-managed (headline) over its full sample, for reference.
    us_managed, _ = build_managed_strategy(
        monthly_ff, forecast_naive(monthly_ff), leverage_cap=leverage_cap
    )

    # Comparison table
    def _row(label, series, bench=None):
        m = performance_metrics(series)
        r = {"strategy": label, "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
             "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"], "sortino": m["sortino"]}
        if bench is not None:
            a = alpha_regression(series, bench)
            r["alpha_annual"], r["alpha_tstat"] = a["alpha_annual"], a["alpha_tstat"]
        else:
            r["alpha_annual"], r["alpha_tstat"] = np.nan, np.nan
        return r

    comparison = pd.DataFrame(
        [
            _row("US-only managed", us_managed["managed"], us_managed["f"]),
            _row("Diversified buy-and-hold", div_bh),
            _row("Diversified managed", div_m_v, div_bh),
            _row("Diversified managed + trend", div_mt_v, div_bh),
        ]
    )

    net_sharpe = {
        "diversified_managed_net": performance_metrics(div_mn)["sharpe"],
        "diversified_trend_net": performance_metrics(div_mtn)["sharpe"],
        "cost_bps": cost_bps,
    }

    return {
        "cross_asset": cross,
        "comparison": comparison,
        "curves": {
            "Diversified buy-and-hold": div_bh,
            "Diversified managed": div_m_v,
            "Diversified managed + trend": div_mt_v,
        },
        "correlation": F.corr(),
        "meta": panel["meta"],
        "net_sharpe": net_sharpe,
        "n_assets_path": n_assets.reindex(idx),
        "div_start": str(idx.min().date()),
        "div_end": str(idx.max().date()),
        "n_assets_final": int(n_assets.reindex(idx).iloc[-1]),
    }


if __name__ == "__main__":  # pragma: no cover
    from src.data_loader import load_dataset

    d, m = load_dataset()
    out = run_extensions(d, m, rebuild=False)
    print("=== Cross-asset (Sharpe: BH -> managed -> +trend) ===")
    print(out["cross_asset"].round(3).to_string(index=False))
    print("\n=== Diversified comparison ===")
    print(out["comparison"].round(3).to_string(index=False))
    print("\nNet-of-cost Sharpe (@%.0f bps): managed=%.3f  +trend=%.3f"
          % (out["net_sharpe"]["cost_bps"], out["net_sharpe"]["diversified_managed_net"],
             out["net_sharpe"]["diversified_trend_net"]))
    print("Diversified sample:", out["div_start"], "->", out["div_end"],
          " final #assets:", out["n_assets_final"])
