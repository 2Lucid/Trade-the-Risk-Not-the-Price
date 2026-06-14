# Volatility-Managed Portfolios — replication & extension of Moreira & Muir (2017)

> **The pitch:** *You can't predict a stock's **price**. But you can predict its **risk** — and that's enough to beat the market on a risk-adjusted basis.*

A clean, fully reproducible implementation of a real buy-side technique
(volatility timing / volatility-managed portfolios, Moreira & Muir, *Journal of
Finance* 2017), validated with the rigour most portfolio projects skip
(walk-forward evaluation, QLIKE loss, Diebold-Mariano tests, transaction costs,
leverage caps, the Deflated Sharpe Ratio), and pushed further with a Hidden
Markov regime layer that explains *where* the edge comes from.

---

## Headline results (US market factor, 2000–2026)

| Metric | Buy-and-hold | Volatility-managed* |
|---|---:|---:|
| Sharpe ratio | **0.48** | **0.61** |
| Annualised return | 7.6% | 9.6% |
| Annualised vol (matched) | 15.8% | 15.8% |
| Max drawdown | −54% | −43% |
| Annualised α (vs market) | — | **3.6%** (*t* = 1.9) |
| Deflated Sharpe Ratio | — | **0.999** |
| Break-even transaction cost | — | ≈ 39 bps |

\* Headline = naive realized-variance forecast (the exact estimator in
Moreira–Muir) with a realistic 2× leverage cap, volatility-matched to
buy-and-hold. On the common 2004–2026 sample where all forecasters are available,
a **GARCH(1,1)** forecaster (the most accurate out-of-sample, by the
Diebold-Mariano/QLIKE test) lifts the Sharpe ratio to **0.75** (vs 0.60 naive,
0.66 buy-and-hold on that sample).

Two findings stated up front, in the spirit of honesty:

1. **The null result that frames everything.** Forecasting next month's *return*
   out-of-sample gives a **negative** R² (≈ −12%): price direction is noise.
   Forecasting next month's *volatility* gives **+28%**: risk is highly
   predictable. That asymmetry is the whole thesis.
2. **It is not free money.** The uncapped naive strategy inherits fat left tails
   from occasional high leverage; the edge is real but modest in this sample
   (two fast crashes), strengthens with a sensible leverage cap and a better
   variance forecast, and survives realistic trading costs.

*(All numbers are regenerated from source by `python -m src.run_all`; the
figures and the table above come straight from `results/results.json`.)*

### Going further — multi-asset + trend (the powerful part)

Run across **11 assets / 4 classes** (6 equity regions, bonds, gold, commodities,
Bitcoin), volatility timing *alone* does **not** travel uniformly — it mainly
helps equity indices (an honest finding, matching Cederburg et al. 2020). But a
**diversified, risk-parity portfolio that adds a time-series-momentum overlay** is
where the real edge is:

| Diversified portfolio (vol-matched) | Sharpe | Max DD | α (vs div. BH) |
|---|---:|---:|---:|
| Buy-and-hold | 0.39 | −45% | — |
| Volatility-managed | 0.41 | −41% | — |
| **Managed + trend** | **0.72** | **−24%** | **8.0%** (*t* = 3.6) |

Net of 10 bps costs the managed+trend Sharpe is still **0.64**. The lesson is the
classic one: durable gains come from **diversification and risk control**, not a
secret signal.

### Interactive site

A self-contained, **beautiful interactive site** lives in [site/](site/): open
[site/index.html](site/index.html) in any browser (no server needed) and re-run
the whole strategy live — pick the asset, the volatility forecaster, the trend
overlay, the leverage cap and the trading cost, and watch the equity curve,
Sharpe, return and drawdown recompute instantly. The in-browser engine mirrors
the Python exactly (the US headline reproduces 0.48 → 0.61). Regenerate its data
with `python -m src.export_site`.

---

## The pipeline (Phases 0–5)

| Phase | Module | What it does |
|---|---|---|
| 0 — Data | [src/data_loader.py](src/data_loader.py) | Downloads Ken French daily factors directly; builds excess returns + monthly realized variance; caches to parquet. |
| 1 — Null result | [src/predictability.py](src/predictability.py) | Out-of-sample R² for forecasting *return* vs *volatility* (expanding window, Campbell-Thompson R², Clark-West test). |
| 2 — Forecasting | [src/vol_forecast.py](src/vol_forecast.py) | Naive / GARCH(1,1) / HAR-RV / ensemble, walk-forward; RMSE, **QLIKE** (Patton 2011), **Diebold-Mariano** (HLN correction). |
| 3 — Strategy | [src/strategy.py](src/strategy.py) | `f_managed = (c/σ̂²)·f`, with the vol-matching normalisation `c`, leverage caps, turnover/costs, optional trend signal. |
| 4 — Evaluation | [src/evaluation.py](src/evaluation.py) | α-regression (Newey-West), Sharpe/Sortino/Calmar/maxDD, **Deflated Sharpe Ratio**, cost & leverage robustness, sub-periods. |
| 5 — Regimes | [src/regimes.py](src/regimes.py) | Gaussian HMM; **smoothed** states for visuals, **filtered** (walk-forward) for any in-backtest use; regime decomposition. |
| 6 — Multi-asset | [src/multi_asset.py](src/multi_asset.py) · [src/trend.py](src/trend.py) · [src/extensions.py](src/extensions.py) | 11-asset USD panel (yfinance); cross-asset robustness; **time-series-momentum** overlay; inverse-vol diversified portfolio. |
| — Figures | [src/plots.py](src/plots.py) | All figures, one unified style. |
| — Orchestration | [src/run_all.py](src/run_all.py) | Runs everything; emits figures, `results/results.json`, and LaTeX tables/macros. |

---

## How the rigour traps are handled

- **No lookahead bias.** Every forecast for month *t* uses only information up to
  the end of month *t−1* (expanding-window refits for GARCH/HAR/OLS; `shift(1)`
  on all features; the HMM filtered signal refits walk-forward).
- **Volatility normalisation.** `c` is calibrated so the managed series has the
  same full-sample volatility as buy-and-hold — otherwise the Sharpe comparison
  would secretly compare two risk levels.
- **Transaction costs.** Costs scale with turnover; we report the break-even cost
  at which the edge disappears.
- **Multiple testing.** The Deflated Sharpe Ratio penalises the number of
  forecasters tried and corrects for non-normality and sample length.
- **Filtered vs smoothed HMM.** Smoothed (full-sample) states are used *only* to
  describe/visualise; the filtered (online) signal is the only thing admissible
  inside a backtest. The distinction is implemented, not just claimed.

---

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.run_all              # ~30s; writes figures/, results/, paper/tables/
```

Outputs:
- `figures/fig01..fig08` (PNG + PDF) — the eight figures.
- `results/results.json` — every number.
- `paper/tables/*.tex` — LaTeX tables + `results_macros.tex` (auto-filled paper).

Individual phases can be run standalone, e.g. `python -m src.vol_forecast`.

The narrative walkthrough is in [notebooks/analysis.ipynb](notebooks/analysis.ipynb).

---

## Repository layout

```
vol-managed-portfolio/
├── README.md                 # this file
├── requirements.txt
├── data/{raw,processed}/     # Ken French + yfinance downloads + parquet cache
├── src/                      # the pipeline (see table above) + multi_asset / trend
│                             #   / extensions / export_site
├── notebooks/analysis.ipynb  # narrative, end-to-end
├── figures/                  # exported PNG/PDF (11 figures)
├── results/results.json      # machine-readable results
├── site/                     # interactive web app (index.html, app.js, data.js)
└── paper/                    # LaTeX working paper + auto-generated tables
```

---

## Methodological notes & limitations

- **Realized variance** uses the sum of squared *daily* excess returns within the
  month (no intraday data), the standard daily-data proxy. HAR is therefore a
  monthly heterogeneous adaptation (1/3/12-month components) rather than the
  intraday-RV original.
- This is a **research study, not investment advice.** A backtest is not live
  trading: results depend on the (low-rate, two-fast-crashes) 2000–2026 sample,
  ignore implementation frictions beyond a turnover cost, and assume frictionless
  leverage up to the cap.

## References

Moreira & Muir (2017, *JF*); Engle (1982); Bollerslev (1986); Corsi (2009);
Patton (2011); Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997);
Campbell & Thompson (2008); Clark & West (2007); Bailey & López de Prado (2014);
Hamilton (1989); Moskowitz, Ooi & Pedersen (2012); Cederburg et al. (2020).
Full BibTeX in [paper/references.bib](paper/references.bib).
