"""Phase 0 — Data.

Primary source: **Ken French Data Library** daily research factors
(`Mkt-RF`, `SMB`, `HML`, `RF`). We download the official CSV directly (no
dependency on the `pandas-datareader` package, which is broken on Python 3.13),
parse it, and produce two clean, reproducible datasets saved to parquet:

* ``daily``   — daily excess-return series and the risk-free rate.
* ``monthly`` — monthly market excess return and monthly **realized variance**
  ``RV_t = sum_{d in month t} r_d^2`` (the standard, robust within-month
  realized-variance proxy used by Moreira & Muir, 2017).

Working in *excess* returns from the start makes every Sharpe ratio downstream
correct by construction.

Data contract (column names other modules rely on):

``daily``  index: DatetimeIndex (business days)
    mkt_excess : market excess return  (Mkt-RF, decimal)
    rf         : risk-free rate        (decimal)
    mkt        : total market return   (mkt_excess + rf, decimal)
    smb, hml   : size / value factors  (decimal, kept for completeness)

``monthly`` index: DatetimeIndex (month-end)
    mkt_excess : monthly market excess return (decimal, compounded)
    rf         : monthly risk-free rate       (decimal, compounded)
    mkt        : monthly total market return  (decimal, compounded)
    rv         : realized variance  = sum of daily squared excess returns
    rvol       : realized volatility = sqrt(rv) (monthly, not annualised)
    n_days     : number of trading days in the month
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

RAW_ZIP = config.DATA_RAW / "F-F_Research_Data_Factors_daily_CSV.zip"
DAILY_PARQUET = config.DATA_PROC / "daily_returns.parquet"
MONTHLY_PARQUET = config.DATA_PROC / "monthly_returns.parquet"


# --------------------------------------------------------------------------- #
# Download + parse                                                             #
# --------------------------------------------------------------------------- #
def _download_raw(force: bool = False) -> bytes:
    """Download (and cache) the Ken French daily factors zip; return raw bytes."""
    if RAW_ZIP.exists() and not force:
        return RAW_ZIP.read_bytes()

    import urllib.request

    req = urllib.request.Request(
        config.KEN_FRENCH_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted URL)
        raw = resp.read()
    RAW_ZIP.write_bytes(raw)
    return raw


def _parse_ff_csv(raw_zip: bytes) -> pd.DataFrame:
    """Parse the Ken French daily factors CSV out of its zip into a DataFrame.

    The file has a free-text header, then a row ``,Mkt-RF,SMB,HML,RF``, then
    daily rows keyed by an 8-digit ``YYYYMMDD`` date, then a copyright footer.
    We locate the header row and keep only rows whose key is an 8-digit date.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(csv_name).decode("utf-8", errors="replace")

    lines = text.splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if "Mkt-RF" in ln)

    rows: list[list[str]] = []
    for ln in lines[header_idx + 1 :]:
        parts = [p.strip() for p in ln.split(",")]
        key = parts[0]
        if len(key) == 8 and key.isdigit():  # YYYYMMDD daily row
            rows.append(parts)
        elif rows:  # reached the footer after the daily block
            break

    cols = ["Mkt-RF", "SMB", "HML", "RF"]
    df = pd.DataFrame(rows, columns=["date", *cols])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)

    # Ken French publishes percentages; convert to decimal returns.
    df = df / 100.0
    df = df.rename(columns={"Mkt-RF": "mkt_excess", "SMB": "smb", "HML": "hml", "RF": "rf"})
    df["mkt"] = df["mkt_excess"] + df["rf"]
    return df[["mkt_excess", "rf", "mkt", "smb", "hml"]]


# --------------------------------------------------------------------------- #
# Aggregation to monthly + realized variance                                  #
# --------------------------------------------------------------------------- #
def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily excess returns to monthly returns and realized variance."""
    grp = daily.resample("ME")

    # Compounded monthly returns (geometric), then excess = mkt_m - rf_m.
    def _compound(s: pd.Series) -> float:
        return float(np.prod(1.0 + s.values) - 1.0)

    mkt_m = grp["mkt"].apply(_compound)
    rf_m = grp["rf"].apply(_compound)
    monthly = pd.DataFrame(index=mkt_m.index)
    monthly["mkt"] = mkt_m
    monthly["rf"] = rf_m
    monthly["mkt_excess"] = mkt_m - rf_m

    # Realized variance proxy: sum of squared daily *excess* returns in the month.
    monthly["rv"] = grp["mkt_excess"].apply(lambda s: float(np.sum(s.values ** 2)))
    monthly["rvol"] = np.sqrt(monthly["rv"])
    monthly["n_days"] = grp["mkt_excess"].count().astype(int)

    # Drop trailing partial month if it has too few trading days to be reliable.
    monthly = monthly[monthly["n_days"] >= 5]
    return monthly


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def build_dataset(
    start: str | None = config.START_DATE,
    end: str | None = config.END_DATE,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download, parse, aggregate and cache the dataset. Returns (daily, monthly)."""
    raw = _download_raw(force=force_download)
    daily = _parse_ff_csv(raw)

    if start is not None:
        daily = daily.loc[daily.index >= pd.Timestamp(start)]
    if end is not None:
        daily = daily.loc[daily.index <= pd.Timestamp(end)]

    monthly = _to_monthly(daily)

    daily.to_parquet(DAILY_PARQUET)
    monthly.to_parquet(MONTHLY_PARQUET)
    return daily, monthly


def load_dataset(rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached (daily, monthly); build from source if missing or ``rebuild``."""
    if rebuild or not (DAILY_PARQUET.exists() and MONTHLY_PARQUET.exists()):
        return build_dataset()
    return pd.read_parquet(DAILY_PARQUET), pd.read_parquet(MONTHLY_PARQUET)


def describe(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics table for the Data section of the paper."""
    ann = config.TRADING_DAYS
    me = daily["mkt_excess"]
    desc = pd.DataFrame(
        {
            "Daily Mkt-RF": [
                len(me),
                me.mean() * ann,
                me.std() * np.sqrt(ann),
                (me.mean() * ann) / (me.std() * np.sqrt(ann)),
                me.skew(),
                me.kurt(),
                me.min(),
                me.max(),
            ],
            "Monthly Mkt-RF": [
                len(monthly),
                monthly["mkt_excess"].mean() * 12,
                monthly["mkt_excess"].std() * np.sqrt(12),
                (monthly["mkt_excess"].mean() * 12) / (monthly["mkt_excess"].std() * np.sqrt(12)),
                monthly["mkt_excess"].skew(),
                monthly["mkt_excess"].kurt(),
                monthly["mkt_excess"].min(),
                monthly["mkt_excess"].max(),
            ],
        },
        index=[
            "N obs",
            "Ann. mean",
            "Ann. vol",
            "Sharpe",
            "Skewness",
            "Excess kurtosis",
            "Min",
            "Max",
        ],
    )
    return desc


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    d, m = build_dataset()
    print(f"daily   {d.shape}  {d.index.min().date()} -> {d.index.max().date()}")
    print(f"monthly {m.shape}  {m.index.min().date()} -> {m.index.max().date()}")
    print(d.head())
    print(m.head())
    print(describe(d, m).round(4))
