"""End-to-end pipeline orchestrator.

Runs every phase, writes all figures to ``figures/``, dumps every result to
``results/results.json``, and emits LaTeX tables + a macro file to
``paper/tables/`` so the report fills itself in with the real numbers (zero
manual copying).

Run from the repo root:

    python -m src.run_all
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root on path even if invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, evaluation as ev, extensions as ext, plots, regimes as rg  # noqa: E402
from src.data_loader import RAW_ZIP, describe, load_dataset  # noqa: E402
from src.multi_asset import ASSET_UNIVERSE, PRICES_PARQUET  # noqa: E402
from src.predictability import run_predictability  # noqa: E402
from src.strategy import build_managed_strategy  # noqa: E402
from src.vol_forecast import run_vol_forecasts  # noqa: E402

# Headline configuration: faithful Moreira-Muir estimator (naive realized
# variance) plus a realistic 2x leverage cap.  The cap is an a-priori risk
# limit, not chosen to maximise performance.
HEADLINE_CAP = 2.0
N_STATES = 2


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (isinstance(obj, float) and np.isnan(obj)) else round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, _dt.date, _dt.datetime)):
        return str(obj if isinstance(obj, _dt.date) and not isinstance(obj, _dt.datetime) else pd.Timestamp(obj).date())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, **kw):
    path = Path(path)
    body = df.to_latex(index=kw.pop("index", False), escape=True, na_rep="--",
                       float_format=kw.pop("float_format", "%.3f"), **kw)
    tex = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\small\n" + body + "\\end{table}\n"
    )
    path.write_text(tex)


def main():
    config.set_plot_style()
    results: dict = {}

    # ---------------- Phase 0: data --------------------------------------- #
    print("[0] Loading data (Ken French) ...")
    daily, monthly = load_dataset(rebuild=True)
    desc = describe(daily, monthly)
    raw_hash = hashlib.sha256(RAW_ZIP.read_bytes()).hexdigest()[:16] if RAW_ZIP.exists() else "n/a"
    results["data"] = {
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "n_daily": int(len(daily)),
        "n_monthly": int(len(monthly)),
        "raw_sha256_16": raw_hash,  # vintage of the Ken French download
    }
    _to_latex(desc.reset_index().rename(columns={"index": "Statistic"}),
              config.PAPER_TABLES / "tab_descriptive.tex",
              "Descriptive statistics, US market excess return (Ken French).",
              "tab:descriptive", float_format="%.4f")

    # ---------------- Phase 1: predictability (null result) --------------- #
    print("[1] Predictability: return vs volatility (OOS R^2) ...")
    pred = run_predictability(monthly)
    results["predictability"] = pred["table"].to_dict(orient="records")
    plots.fig_predictability_r2(pred["table"], config.FIGURES / "fig01_predictability_r2.png")
    _to_latex(pred["table"], config.PAPER_TABLES / "tab_predictability.tex",
              "Out-of-sample $R^2$: forecasting the return vs forecasting the "
              "volatility (expanding window, vs the prevailing mean).",
              "tab:predictability", float_format="%.4f")

    # ---------------- Phase 2: volatility forecasting --------------------- #
    print("[2] Volatility forecasting (naive / GARCH / HAR / ensemble) ...")
    vf = run_vol_forecasts(daily, monthly, refit_every=1, verbose=True)
    forecasts = vf["forecasts"]
    results["vol_forecast"] = {
        "metrics": vf["metrics"].reset_index().to_dict(orient="records"),
        "dm_table": vf["dm_table"].to_dict(orient="records"),
        "eval_start": str(vf["eval_start"].date()),
        "eval_end": str(vf["eval_end"].date()),
        "n_eval": vf["n_eval"],
    }
    vol_df = (forecasts[["realized", "garch"]]
              .rename(columns={"realized": "realized_rv", "garch": "forecast_rv"}).dropna())
    plots.fig_vol_forecast(vol_df, config.FIGURES / "fig02_vol_forecast.png",
                           title="GARCH(1,1) one-step-ahead forecast vs realized volatility")
    plots.fig_forecast_accuracy(vf["metrics"], config.FIGURES / "fig03_forecast_accuracy.png")
    _to_latex(vf["metrics"].reset_index(), config.PAPER_TABLES / "tab_vol_metrics.tex",
              "Volatility-forecast accuracy (RMSE and QLIKE, common sample).",
              "tab:volmetrics", float_format="%.5f")
    _to_latex(vf["dm_table"], config.PAPER_TABLES / "tab_dm.tex",
              "Diebold-Mariano tests of equal predictive accuracy "
              "(HLN small-sample correction).", "tab:dm", float_format="%.3f")

    # ---------------- Phase 3 & 4: strategy + evaluation ------------------ #
    print("[3-4] Strategy construction + evaluation ...")
    naive_fc = forecasts["naive"]

    # MM literal baseline (uncapped) and the headline (2x cap), both naive.
    strat_uncapped, info_unc = build_managed_strategy(monthly, naive_fc)
    strat, info = build_managed_strategy(monthly, naive_fc, leverage_cap=HEADLINE_CAP)

    bh_metrics = ev.performance_metrics(strat["f"], name="buy_and_hold")
    mm_metrics = ev.performance_metrics(strat["managed"], name="managed_headline")
    mm_alpha = ev.alpha_regression(strat["managed"], strat["f"])
    unc_metrics = ev.performance_metrics(strat_uncapped["managed"])
    unc_alpha = ev.alpha_regression(strat_uncapped["managed"], strat_uncapped["f"])

    # Forecaster comparison (common sample, uncapped) -- shows better forecasts help.
    fc_cmp = ev.compare_forecasters(monthly, forecasts, ["naive", "har", "garch", "ensemble"])

    # Deflated Sharpe ratio: candidate and trials share ONE configuration
    # (common evaluation sample, 2x cap, the three independent forecasters --
    # ensemble is a deterministic average, not an independent search), and the
    # candidate is one of the trials. A self-consistent multiple-testing test.
    dsr_common = forecasts.dropna(subset=["naive", "har", "garch"])
    dsr_strats = {
        col: build_managed_strategy(
            monthly.loc[dsr_common.index], dsr_common[col], leverage_cap=HEADLINE_CAP
        )[0]["managed"]
        for col in ["naive", "har", "garch"]
    }
    trial_sharpes = [float(r.mean() / r.std(ddof=1)) for r in dsr_strats.values()]
    dsr = ev.deflated_sharpe_ratio(dsr_strats["naive"], len(trial_sharpes), trial_sharpes)

    # Robustness
    cost_tab = ev.cost_sensitivity(strat)
    be_cost = ev.breakeven_cost(strat)
    lev_tab = ev.leverage_sensitivity(monthly, naive_fc)
    sub_tab = ev.subperiod_analysis(strat)

    results["strategy"] = {
        "headline_config": {"forecaster": "naive", "leverage_cap": HEADLINE_CAP, **_round(info)},
        "buy_and_hold": _round(bh_metrics),
        "managed_headline": _round(mm_metrics),
        "managed_headline_alpha": _round(mm_alpha),
        "mm_baseline_uncapped": {**_round(unc_metrics), **_round(unc_alpha)},
        "forecaster_comparison": fc_cmp.to_dict(orient="records"),
        "deflated_sharpe": _round(dsr),
        "breakeven_cost_bps": _round(be_cost),
        "cost_sensitivity": cost_tab.to_dict(orient="records"),
        "leverage_sensitivity": lev_tab.to_dict(orient="records"),
        "subperiods": sub_tab.to_dict(orient="records"),
    }

    # Figures
    plots.fig_weights(strat, config.FIGURES / "fig04_exposure.png")
    plots.fig_equity_curve(strat, config.FIGURES / "fig05_equity_curve.png")
    plots.fig_cost_sensitivity(cost_tab, be_cost, bh_metrics["sharpe"],
                               config.FIGURES / "fig06_cost_sensitivity.png")

    # Tables
    metrics_tab = pd.DataFrame([bh_metrics, mm_metrics]).set_index("name").T
    _to_latex(metrics_tab.reset_index().rename(columns={"index": "Metric"}),
              config.PAPER_TABLES / "tab_metrics.tex",
              "Headline performance: volatility-managed (naive forecast, 2$\\times$ cap) "
              "vs buy-and-hold, vol-matched.", "tab:metrics", float_format="%.3f")
    _to_latex(fc_cmp, config.PAPER_TABLES / "tab_forecasters.tex",
              "Managed-strategy performance by volatility forecaster "
              "(common sample, uncapped).", "tab:forecasters", float_format="%.3f")
    _to_latex(cost_tab, config.PAPER_TABLES / "tab_costs.tex",
              "Transaction-cost sensitivity of the headline strategy.",
              "tab:costs", float_format="%.3f")
    _to_latex(lev_tab, config.PAPER_TABLES / "tab_leverage.tex",
              "Leverage-cap sensitivity (volatility re-normalised within each cap).",
              "tab:leverage", float_format="%.3f")
    _to_latex(sub_tab, config.PAPER_TABLES / "tab_subperiods.tex",
              "Sub-period analysis: managed vs buy-and-hold.", "tab:subperiods",
              float_format="%.3f")

    # ---------------- Phase 5: regimes ------------------------------------ #
    print("[5] HMM regimes ...")
    hmm = rg.fit_hmm_smoothed(monthly, n_states=N_STATES)
    filt = rg.filtered_regimes(monthly, n_states=N_STATES, min_train=60, refit_every=3)
    decomp = rg.regime_decomposition(strat, hmm["states"], hmm["labels"])
    results["regimes"] = {
        "state_stats": hmm["state_stats"].to_dict(orient="records"),
        "decomposition": decomp.to_dict(orient="records"),
        "filtered_available_from": str(filt.dropna().index.min().date()),
    }
    plots.fig_regimes(monthly, hmm, config.FIGURES / "fig07_regimes.png")
    plots.fig_regime_decomposition(decomp, config.FIGURES / "fig08_regime_decomposition.png")
    _to_latex(hmm["state_stats"], config.PAPER_TABLES / "tab_regime_stats.tex",
              "Estimated HMM regimes (states ordered by variance).",
              "tab:regimestats", float_format="%.3f")
    _to_latex(decomp, config.PAPER_TABLES / "tab_regime_decomp.tex",
              "Performance decomposition by (smoothed) regime.",
              "tab:regimedecomp", float_format="%.3f")

    # ---------------- Phase 6: multi-asset robustness + trend ------------- #
    print("[6] Multi-asset robustness + trend overlay (downloads yfinance panel) ...")
    xt = ext.run_extensions(daily, monthly, leverage_cap=HEADLINE_CAP)
    prices_hash = (hashlib.sha256(PRICES_PARQUET.read_bytes()).hexdigest()[:16]
                   if PRICES_PARQUET.exists() else "n/a")
    results["extensions"] = {
        "cross_asset": xt["cross_asset"].to_dict(orient="records"),
        "comparison": xt["comparison"].to_dict(orient="records"),
        "net_sharpe": _round(xt["net_sharpe"]),
        "div_start": xt["div_start"],
        "div_end": xt["div_end"],
        "n_assets_final": xt["n_assets_final"],
        "universe": xt["meta"].to_dict(orient="records"),
        "prices_sha256_16": prices_hash,  # vintage of the yfinance panel
    }
    plots.fig_cross_asset_sharpe(xt["cross_asset"], config.FIGURES / "fig09_cross_asset_sharpe.png")
    plots.fig_diversified_equity(xt["curves"], config.FIGURES / "fig10_diversified_equity.png")
    plots.fig_correlation(xt["correlation"], ASSET_UNIVERSE, config.FIGURES / "fig11_correlation.png")
    cross_tab = xt["cross_asset"].rename(columns={
        "bh_sharpe": "BH SR", "mm_sharpe": "Managed SR", "mt_sharpe": "Mgd+Trend SR",
        "mm_alpha_t": "Mgd alpha t", "mt_alpha_t": "Trend alpha t"})
    _to_latex(cross_tab, config.PAPER_TABLES / "tab_cross_asset.tex",
              "Volatility management and a trend overlay across assets "
              "(Sharpe ratios and managed-alpha $t$-statistics).",
              "tab:crossasset", float_format="%.3f")
    _to_latex(xt["comparison"], config.PAPER_TABLES / "tab_diversified.tex",
              "Diversified portfolios vs the single-market strategy "
              "(all vol-matched to the diversified buy-and-hold).",
              "tab:diversified", float_format="%.3f")

    # ---------------- Persist + macros ------------------------------------ #
    (config.RESULTS / "results.json").write_text(json.dumps(_round(results), indent=2))
    _write_macros(results, config.PAPER_TABLES / "results_macros.tex")
    forecasts.to_parquet(config.DATA_PROC / "forecasts.parquet")

    _print_summary(results)
    print(f"\nAll outputs written. Figures -> {config.FIGURES}, "
          f"tables -> {config.PAPER_TABLES}, results -> {config.RESULTS}")
    return results


def _write_macros(r: dict, path: Path):
    """Emit \\newcommand macros so the LaTeX paper uses the real numbers."""
    bh, mm = r["strategy"]["buy_and_hold"], r["strategy"]["managed_headline"]
    a = r["strategy"]["managed_headline_alpha"]
    dsr = r["strategy"]["deflated_sharpe"]
    unc = r["strategy"]["mm_baseline_uncapped"]
    fc_cmp = r["strategy"]["forecaster_comparison"]
    best = max(
        [x for x in fc_cmp if x["forecaster"] != "buy_and_hold"],
        key=lambda x: x["sharpe"],
    )
    naive_common = next(x for x in fc_cmp if x["forecaster"] == "naive")
    bh_common = next(x for x in fc_cmp if x["forecaster"] == "buy_and_hold")
    ret_r2 = min(x["R2_OOS"] for x in r["predictability"] if x["Target"] == "Return")
    vol_r2 = max(x["R2_OOS"] for x in r["predictability"] if x["Target"].startswith("Vol"))

    def cmd(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}\n"

    txt = "% Auto-generated by src/run_all.py -- do not edit by hand.\n"
    txt += cmd("dataStart", r["data"]["start"])
    txt += cmd("dataEnd", r["data"]["end"])
    txt += cmd("nMonthly", r["data"]["n_monthly"])
    txt += cmd("retRtwo", f"{ret_r2*100:.1f}\\%")
    txt += cmd("volRtwo", f"{vol_r2*100:.1f}\\%")
    txt += cmd("sharpeBH", f"{bh['sharpe']:.2f}")
    txt += cmd("sharpeManaged", f"{mm['sharpe']:.2f}")
    txt += cmd("sharpeBest", f"{best['sharpe']:.2f}")
    txt += cmd("bestForecaster", best["forecaster"].upper())
    # common-sample, uncapped figures for an apples-to-apples forecaster comparison
    txt += cmd("sharpeBHcommon", f"{bh_common['sharpe']:.2f}")
    txt += cmd("sharpeNaiveCommon", f"{naive_common['sharpe']:.2f}")
    txt += cmd("annRetBH", f"{bh['ann_return']*100:.1f}\\%")
    txt += cmd("annRetManaged", f"{mm['ann_return']*100:.1f}\\%")
    txt += cmd("maxddBH", f"{bh['max_drawdown']*100:.1f}\\%")
    txt += cmd("maxddManaged", f"{mm['max_drawdown']*100:.1f}\\%")
    txt += cmd("alphaManaged", f"{a['alpha_annual']*100:.1f}\\%")
    txt += cmd("alphaTstat", f"{a['alpha_tstat']:.2f}")
    txt += cmd("appraisal", f"{a['appraisal_ratio']:.2f}")
    txt += cmd("betaManaged", f"{a['beta']:.2f}")
    txt += cmd("alphaUncapped", f"{unc['alpha_annual']*100:.1f}\\%")
    txt += cmd("alphaUncappedT", f"{unc['alpha_tstat']:.2f}")
    txt += cmd("deflatedSharpe", f"{dsr['deflated_sharpe']:.3f}")
    txt += cmd("breakevenCost", f"{r['strategy']['breakeven_cost_bps']:.0f}")

    # ---- extension (multi-asset + trend) macros ---------------------------- #
    ext_cmp = {row["strategy"]: row for row in r["extensions"]["comparison"]}
    div_bh = ext_cmp["Diversified buy-and-hold"]
    div_mt = ext_cmp["Diversified managed + trend"]
    ns = r["extensions"]["net_sharpe"]
    txt += cmd("nAssets", f"{r['extensions']['n_assets_final']}")
    txt += cmd("sharpeDivBH", f"{div_bh['sharpe']:.2f}")
    txt += cmd("sharpeDivManaged", f"{ext_cmp['Diversified managed']['sharpe']:.2f}")
    txt += cmd("sharpeDivTrend", f"{div_mt['sharpe']:.2f}")
    txt += cmd("maxddDivBH", f"{div_bh['max_drawdown']*100:.1f}\\%")
    txt += cmd("maxddDivTrend", f"{div_mt['max_drawdown']*100:.1f}\\%")
    txt += cmd("alphaDivTrend", f"{div_mt['alpha_annual']*100:.1f}\\%")
    txt += cmd("alphaDivTrendT", f"{div_mt['alpha_tstat']:.2f}")
    txt += cmd("sharpeDivTrendNet", f"{ns['diversified_trend_net']:.2f}")
    txt += cmd("costBpsExt", f"{ns['cost_bps']:.0f}")
    path.write_text(txt)


def _print_summary(r: dict):
    bh, mm = r["strategy"]["buy_and_hold"], r["strategy"]["managed_headline"]
    a = r["strategy"]["managed_headline_alpha"]
    print("\n" + "=" * 64)
    print("HEADLINE RESULTS (naive forecast, 2x cap, vol-matched)")
    print("=" * 64)
    print(f"  Sample          : {r['data']['start']} -> {r['data']['end']}")
    print(f"  Sharpe  BH  ->  Managed : {bh['sharpe']:.3f} -> {mm['sharpe']:.3f}")
    print(f"  Ann.ret BH  ->  Managed : {bh['ann_return']*100:.1f}% -> {mm['ann_return']*100:.1f}%")
    print(f"  MaxDD   BH  ->  Managed : {bh['max_drawdown']*100:.1f}% -> {mm['max_drawdown']*100:.1f}%")
    print(f"  Alpha (ann)             : {a['alpha_annual']*100:.2f}%  (t = {a['alpha_tstat']:.2f})")
    print(f"  Appraisal ratio         : {a['appraisal_ratio']:.3f}")
    print(f"  Deflated Sharpe         : {r['strategy']['deflated_sharpe']['deflated_sharpe']:.3f}")
    print(f"  Break-even cost         : {r['strategy']['breakeven_cost_bps']:.0f} bps")
    best = max([x for x in r["strategy"]["forecaster_comparison"] if x["forecaster"] != "buy_and_hold"],
               key=lambda x: x["sharpe"])
    print(f"  Best forecaster         : {best['forecaster']} (Sharpe {best['sharpe']:.3f})")


if __name__ == "__main__":
    main()
