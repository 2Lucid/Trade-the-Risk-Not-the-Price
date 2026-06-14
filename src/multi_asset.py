"""Multi-asset data (extension).

Builds a consistent, USD total-return panel across asset classes from Yahoo
Finance (auto-adjusted ETFs/series), aligns it to the Ken French daily trading
calendar and risk-free rate, and produces per-asset monthly excess returns and
realized variance in the *same schema* as :mod:`src.data_loader` so the existing
strategy/evaluation machinery works unchanged.

Using USD total-return instruments (rather than mixing price and total-return
local indices) keeps the cross-asset comparison internally consistent: every
series is what a USD investor actually earns, and excess returns use one common
(US) risk-free rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# ticker -> (display name, asset class). USD total-return, liquid, long history.
ASSET_UNIVERSE = {
    "SPY": ("US equity", "Equity"),
    "EWU": ("UK equity", "Equity"),
    "EWG": ("Germany equity", "Equity"),
    "EWJ": ("Japan equity", "Equity"),
    "EWQ": ("France equity", "Equity"),
    "EEM": ("EM equity", "Equity"),
    "TLT": ("US long Treasuries", "Bonds"),
    "IEF": ("US 7-10y Treasuries", "Bonds"),
    "GLD": ("Gold", "Commodity"),
    "DBC": ("Commodities", "Commodity"),
    "BTC-USD": ("Bitcoin", "Crypto"),
}

PRICES_PARQUET = config.DATA_RAW / "multiasset_prices.parquet"
MIN_MONTHS = 36  # require >=3y of monthly history before an asset enters the study


def download_prices(rebuild: bool = False) -> pd.DataFrame:
    """Download (and cache) adjusted close prices for the asset universe."""
    if PRICES_PARQUET.exists() and not rebuild:
        return pd.read_parquet(PRICES_PARQUET)

    import yfinance as yf

    tickers = list(ASSET_UNIVERSE)
    raw = yf.download(
        tickers, start="1999-01-01", end="2026-05-01",
        progress=False, auto_adjust=True,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close[tickers]  # keep order
    close.to_parquet(PRICES_PARQUET)
    return close


def _asset_monthly(daily_ret: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Monthly excess return + realized variance for one asset (data_loader schema)."""
    d = pd.DataFrame({"ret": daily_ret, "rf": rf}).dropna()
    d["excess"] = d["ret"] - d["rf"]
    grp = d.resample("ME")

    def _compound(s):
        return float(np.prod(1.0 + s.values) - 1.0)

    m = pd.DataFrame(index=grp["ret"].apply(_compound).index)
    m["mkt"] = grp["ret"].apply(_compound)
    m["rf"] = grp["rf"].apply(_compound)
    m["mkt_excess"] = m["mkt"] - m["rf"]
    m["rv"] = grp["excess"].apply(lambda s: float(np.sum(s.values ** 2)))
    m["rvol"] = np.sqrt(m["rv"])
    m["n_days"] = grp["excess"].count().astype(int)
    return m[m["n_days"] >= 5]


def build_panel(daily_ff: pd.DataFrame, rebuild: bool = False) -> dict:
    """Build the aligned multi-asset panel.

    Returns a dict with:
      ``daily_excess`` : DataFrame (date x asset) of daily excess returns,
      ``monthly``      : dict {ticker: monthly DataFrame} (data_loader schema),
      ``meta``         : DataFrame of name/class/start/n_months per asset.
    """
    prices = download_prices(rebuild=rebuild)

    # Align to the Ken French trading calendar; carry prices over foreign
    # holidays (return 0 that day). This also samples crypto at business-day
    # closes so it mixes cleanly with the rest of the panel.
    ff_index = daily_ff.index
    rf = daily_ff["rf"]
    prices = prices.reindex(ff_index).ffill(limit=5)
    rets = prices.pct_change(fill_method=None)

    daily_excess = rets.sub(rf, axis=0)

    monthly, meta_rows = {}, []
    for tkr in ASSET_UNIVERSE:
        if tkr not in rets.columns:
            continue
        m = _asset_monthly(rets[tkr], rf)
        if len(m) < MIN_MONTHS:
            continue
        monthly[tkr] = m
        name, klass = ASSET_UNIVERSE[tkr]
        meta_rows.append(
            {"ticker": tkr, "name": name, "class": klass,
             "start": m.index.min().date(), "n_months": len(m)}
        )

    meta = pd.DataFrame(meta_rows)
    return {"daily_excess": daily_excess, "monthly": monthly, "meta": meta}


if __name__ == "__main__":  # pragma: no cover
    from src.data_loader import load_dataset

    d, _ = load_dataset()
    panel = build_panel(d, rebuild=True)
    print(panel["meta"].to_string(index=False))
    print("\nassets in study:", list(panel["monthly"]))
