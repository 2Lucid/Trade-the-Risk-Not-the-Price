# Data

The data used by the project is committed here for full reproducibility (it is
small, ~0.9 MB total). It can also be regenerated from source at any time with
`python -m src.run_all`.

## `raw/`
| File | Source | Notes |
|---|---|---|
| `F-F_Research_Data_Factors_daily_CSV.zip` | **Ken French Data Library** | Daily research factors (`Mkt-RF`, `SMB`, `HML`, `RF`). Freely available for research; downloaded by `src/data_loader.py`. |
| `multiasset_prices.parquet` | **Yahoo Finance** (via `yfinance`) | Adjusted daily closes for the 11-asset panel. Cached for reproducibility / educational use only; downloaded by `src/multi_asset.py`. |

## `processed/`
Clean, analysis-ready tables built from `raw/`:

| File | Built by | Contents |
|---|---|---|
| `daily_returns.parquet` | `data_loader.py` | Daily excess returns + risk-free rate |
| `monthly_returns.parquet` | `data_loader.py` | Monthly excess returns + realized variance |
| `forecasts.parquet` | `vol_forecast.py` | Naive / GARCH / HAR / ensemble variance forecasts |

## Provenance & licensing

Ken French data is publicly distributed for research. Yahoo Finance data is
included only as a reproducibility cache for this educational project; if you
reuse it, check Yahoo's terms. Nothing here is investment advice.
