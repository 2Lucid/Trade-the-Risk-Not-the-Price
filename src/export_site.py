"""Export pipeline results to a self-contained ``site/data.js`` for the web app.

The interactive site recomputes the strategy in the browser, so we export the
raw monthly building blocks (excess returns ``f`` and the variance forecasts) per
asset, plus the precomputed showcase series (regimes, diversified portfolios) and
summary tables. Data is written as ``window.VMP_DATA = {...};`` so the site works
by simply opening ``index.html`` (no server, no fetch/CORS).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, regimes as rg
from src.data_loader import load_dataset
from src.extensions import run_extensions
from src.multi_asset import ASSET_UNIVERSE
from src.predictability import run_predictability
from src.vol_forecast import forecast_naive, run_vol_forecasts

SITE = config.ROOT / "site"


def _dates(idx) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in idx]


def _arr(s: pd.Series, nd: int = 8) -> list:
    return [round(float(x), nd) for x in s.values]


def main():
    SITE.mkdir(exist_ok=True)
    daily, monthly = load_dataset()

    # ---- US market: f, naive & GARCH variance forecasts -------------------- #
    vf = run_vol_forecasts(daily, monthly, verbose=False)
    fc = vf["forecasts"]
    us = pd.DataFrame(
        {"f": monthly["mkt_excess"], "varNaive": fc["naive"], "varGarch": fc["garch"]}
    ).dropna(subset=["f", "varNaive"])

    assets = {
        "US market": {
            "cls": "Equity — US factor (Ken French)",
            "dates": _dates(us.index),
            "f": _arr(us["f"], 6),
            "varNaive": _arr(us["varNaive"]),
            "varGarch": [None if pd.isna(x) else round(float(x), 8) for x in us["varGarch"]],
        }
    }
    asset_order = ["US market"]

    # ---- Cross-asset panel: per-asset f + naive variance forecast ---------- #
    xt = run_extensions(daily, monthly)
    from src.multi_asset import build_panel

    panel = build_panel(daily)
    for tkr, ma in panel["monthly"].items():
        name = ASSET_UNIVERSE[tkr][0]
        d = pd.DataFrame({"f": ma["mkt_excess"], "varNaive": ma["rv"].shift(1)}).dropna()
        assets[name] = {
            "cls": ASSET_UNIVERSE[tkr][1] + f" ({tkr})",
            "dates": _dates(d.index),
            "f": _arr(d["f"], 6),
            "varNaive": _arr(d["varNaive"]),
            "varGarch": None,
        }
        asset_order.append(name)

    # ---- Regimes (smoothed) ------------------------------------------------ #
    hmm = rg.fit_hmm_smoothed(monthly, n_states=2)
    price = (1 + monthly["mkt"]).cumprod()
    regimes = {
        "dates": _dates(monthly.index),
        "price": _arr(price, 4),
        "state": [int(x) for x in hmm["states"].values],
        "labels": hmm["labels"],
    }

    # ---- Diversified portfolios (monthly returns) -------------------------- #
    cur = xt["curves"]
    div_idx = cur["Diversified buy-and-hold"].index
    diversified = {
        "dates": _dates(div_idx),
        "bh": _arr(cur["Diversified buy-and-hold"], 6),
        "managed": _arr(cur["Diversified managed"], 6),
        "trend": _arr(cur["Diversified managed + trend"], 6),
    }

    # ---- Summary tables ---------------------------------------------------- #
    predictability = [
        {"target": r["Target"], "model": r["Model"], "r2": round(float(r["R2_OOS"]), 4)}
        for r in run_predictability(monthly)["table"].to_dict("records")
    ]
    vol_metrics = [
        {"model": k, "rmse": round(float(v["RMSE"]), 6), "qlike": round(float(v["QLIKE"]), 4)}
        for k, v in vf["metrics"].to_dict("index").items()
    ]
    cross_asset = [
        {"asset": r["asset"], "cls": r["class"], "bh": round(float(r["bh_sharpe"]), 3),
         "managed": round(float(r["mm_sharpe"]), 3), "trend": round(float(r["mt_sharpe"]), 3)}
        for r in xt["cross_asset"].to_dict("records")
    ]
    comparison = [
        {"strategy": r["strategy"], "sharpe": round(float(r["sharpe"]), 3),
         "annReturn": round(float(r["ann_return"]), 4), "maxDD": round(float(r["max_drawdown"]), 4),
         "alphaT": (None if pd.isna(r["alpha_tstat"]) else round(float(r["alpha_tstat"]), 2))}
        for r in xt["comparison"].to_dict("records")
    ]

    payload = {
        "meta": {
            "dataStart": str(monthly.index.min().date()),
            "dataEnd": str(monthly.index.max().date()),
            "nMonthly": int(len(monthly)),
            "nAssets": xt["n_assets_final"],
            "tradingMonths": config.MONTHS_PER_YEAR,
        },
        "assetOrder": asset_order,
        "assets": assets,
        "regimes": regimes,
        "diversified": diversified,
        "predictability": predictability,
        "volMetrics": vol_metrics,
        "crossAsset": cross_asset,
        "comparison": comparison,
        "netSharpe": _r(xt["net_sharpe"]),
    }

    out = SITE / "data.js"
    out.write_text("window.VMP_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({kb:.0f} KB) — {len(assets)} assets, {len(monthly)} months.")


def _r(d):
    return {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in d.items()}


if __name__ == "__main__":
    main()
