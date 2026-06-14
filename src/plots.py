"""Figures — one unified, publication-quality style for the whole project.

Every function takes already-computed results and an output path, writes a PNG
(and PDF) and returns the path.  No computation or data loading happens here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

config.set_plot_style()
P = config.PALETTE


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Phase 1 — predictability                                                     #
# --------------------------------------------------------------------------- #
def fig_predictability_r2(table: pd.DataFrame, path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [P["accent"] if "Return" in t else P["good"] for t in table["Target"]]
    labels = [f"{r.Model}\n({r.Target.split(' ')[0]})" for r in table.itertuples()]
    bars = ax.bar(range(len(table)), table["R2_OOS"] * 100, color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Out-of-sample $R^2$ (%)")
    ax.set_title("You cannot predict the return — but you can predict the risk")
    for b, v in zip(bars, table["R2_OOS"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + (1 if v >= 0 else -1),
                f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=P["accent"]),
               plt.Rectangle((0, 0), 1, 1, color=P["good"])]
    ax.legend(handles, ["Target: next-month return", "Target: next-month volatility"],
              loc="upper left")
    return _save(fig, path)


def fig_vol_forecast(vol_df: pd.DataFrame, path, title="Realized vs forecast volatility") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    rvol = np.sqrt(vol_df["realized_rv"]) * 100
    fvol = np.sqrt(vol_df["forecast_rv"]) * 100
    ax.plot(rvol.index, rvol, color=P["buyhold"], lw=1.1, label="Realized")
    ax.plot(fvol.index, fvol, color=P["managed"], lw=1.4, label="Forecast")
    ax.set_ylabel("Monthly volatility (%)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, path)


def fig_forecast_accuracy(metrics: pd.DataFrame, path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, ttl in zip(axes, ["RMSE", "QLIKE"], ["RMSE (lower = better)", "QLIKE (lower = better)"]):
        order = metrics[col].sort_values()
        cols = [P["good"] if m == order.index[0] else P["buyhold"] for m in order.index]
        ax.bar(order.index, order.values, color=cols, edgecolor="white")
        ax.set_title(ttl)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Volatility-forecast accuracy", fontweight="bold")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Phase 3 — exposure                                                           #
# --------------------------------------------------------------------------- #
def fig_weights(strat: pd.DataFrame, path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(strat.index, strat["weight"], color=P["managed"], lw=1.1)
    ax.axhline(1.0, color=P["buyhold"], ls="--", lw=1, label="Buy-and-hold (1x)")
    ax.fill_between(strat.index, 1.0, strat["weight"],
                    where=strat["weight"] >= 1.0, color=P["good"], alpha=0.25)
    ax.fill_between(strat.index, 1.0, strat["weight"],
                    where=strat["weight"] < 1.0, color=P["accent"], alpha=0.25)
    ax.set_ylabel("Exposure (× market)")
    ax.set_title("Volatility-managed exposure: lever up when calm, de-risk when nervous")
    ax.legend(loc="upper right")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Phase 4 — equity curve + drawdowns                                          #
# --------------------------------------------------------------------------- #
def fig_equity_curve(strat: pd.DataFrame, path, label_managed="Volatility-managed") -> Path:
    from src.evaluation import drawdown_series

    eq_m = (1 + strat["managed"]).cumprod()
    eq_b = (1 + strat["f"]).cumprod()
    dd_m = drawdown_series(strat["managed"]) * 100
    dd_b = drawdown_series(strat["f"]) * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    ax1.plot(eq_b.index, eq_b, color=P["buyhold"], lw=1.4, label="Buy-and-hold")
    ax1.plot(eq_m.index, eq_m, color=P["managed"], lw=1.7, label=label_managed)
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (excess, log)")
    ax1.set_title("Volatility-managed vs buy-and-hold (vol-matched)")
    ax1.legend(loc="upper left")

    ax2.fill_between(dd_b.index, dd_b, 0, color=P["buyhold"], alpha=0.45, label="Buy-and-hold")
    ax2.fill_between(dd_m.index, dd_m, 0, color=P["managed"], alpha=0.5, label=label_managed)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", ncol=2, fontsize=8)
    return _save(fig, path)


def fig_cost_sensitivity(cost_table: pd.DataFrame, breakeven: float, bh_sharpe: float, path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(cost_table["cost_bps"], cost_table["sharpe"], "o-", color=P["managed"],
            label="Managed (net of costs)")
    ax.axhline(bh_sharpe, color=P["buyhold"], ls="--", label="Buy-and-hold Sharpe")
    if np.isfinite(breakeven):
        ax.axvline(breakeven, color=P["accent"], ls=":", lw=1.4,
                   label=f"Break-even ≈ {breakeven:.0f} bps")
    ax.set_xlabel("Transaction cost (bps per unit turnover)")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Does the edge survive trading costs?")
    ax.legend(loc="upper right")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Phase 5 — regimes                                                            #
# --------------------------------------------------------------------------- #
def _shade_regimes(ax, states: pd.Series, n_states: int):
    """Shade the background by the (smoothed) HMM state."""
    colors = {0: P["calm"], n_states - 1: P["crisis"]}
    if n_states == 3:
        colors[1] = P["intermediate"]
    s = states.dropna()
    idx = s.index
    start = idx[0]
    cur = s.iloc[0]
    for i in range(1, len(s)):
        if s.iloc[i] != cur:
            ax.axvspan(start, idx[i], color=colors.get(cur, P["buyhold"]), alpha=0.16, lw=0)
            start, cur = idx[i], s.iloc[i]
    ax.axvspan(start, idx[-1], color=colors.get(cur, P["buyhold"]), alpha=0.16, lw=0)


def fig_regimes(monthly: pd.DataFrame, hmm: dict, path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    price = (1 + monthly["mkt"]).cumprod()
    ax.plot(price.index, price, color="black", lw=1.4)
    ax.set_yscale("log")
    _shade_regimes(ax, hmm["states"], hmm["n_states"])
    ax.set_ylabel("Market total-return index (log)")
    ax.set_title("The HMM rediscovers the crises on its own — no dates supplied")

    labels = hmm["labels"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4)
               for c in ([P["calm"], P["crisis"]] if hmm["n_states"] == 2
                         else [P["calm"], P["intermediate"], P["crisis"]])]
    ax.legend(handles, labels, loc="upper left", title="HMM regime")
    return _save(fig, path)


def fig_cross_asset_sharpe(cross: pd.DataFrame, path) -> Path:
    """Per-asset Sharpe: buy-and-hold vs managed vs managed+trend."""
    c = cross.copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(c))
    w = 0.27
    ax.bar(x - w, c["bh_sharpe"], w, color=P["buyhold"], label="Buy-and-hold")
    ax.bar(x, c["mm_sharpe"], w, color=P["managed"], label="Vol-managed")
    ax.bar(x + w, c["mt_sharpe"], w, color=P["good"], label="Vol-managed + trend")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(c["asset"], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Annualised Sharpe ratio")
    top = float(np.nanmax(c[["bh_sharpe", "mm_sharpe", "mt_sharpe"]].values))
    ax.set_ylim(top=top * 1.28)  # headroom for the legend
    ax.set_title("The principle travels — and the trend overlay does the heavy lifting")
    ax.legend(loc="upper center", ncol=3)
    return _save(fig, path)


def fig_diversified_equity(curves: dict, path) -> Path:
    """Cumulative growth of the diversified buy-and-hold / managed / +trend portfolios."""
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = {"Diversified buy-and-hold": P["buyhold"],
              "Diversified managed": P["managed"],
              "Diversified managed + trend": P["good"]}
    for name, ser in curves.items():
        eq = (1 + ser).cumprod()
        ax.plot(eq.index, eq, color=colors.get(name, P["neutral"]),
                lw=1.8 if "trend" in name else 1.4, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (excess, log)")
    ax.set_title("Diversification + volatility management + trend (vol-matched)")
    ax.legend(loc="upper left")
    return _save(fig, path)


def fig_correlation(corr: pd.DataFrame, names: dict, path) -> Path:
    """Cross-asset monthly-return correlation heatmap (the diversification picture)."""
    labels = [names.get(c, (c,))[0] if isinstance(names.get(c), tuple) else c for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    try:
        import seaborn as sns

        sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": 6.5},
                    xticklabels=labels, yticklabels=labels, cbar_kws={"shrink": 0.8})
    except Exception:
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        fig.colorbar(im, shrink=0.8)
    ax.set_title("Cross-asset return correlations (why diversification helps)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7.5)
    plt.setp(ax.get_yticklabels(), fontsize=7.5)
    return _save(fig, path)


def fig_regime_decomposition(decomp: pd.DataFrame, path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(decomp))
    w = 0.38
    ax1.bar(x - w / 2, decomp["bh_ann_vol"] * 100, w, color=P["buyhold"], label="Buy-and-hold")
    ax1.bar(x + w / 2, decomp["mm_ann_vol"] * 100, w, color=P["managed"], label="Managed")
    ax1.set_xticks(x); ax1.set_xticklabels(decomp["regime"])
    ax1.set_ylabel("Annualised volatility (%)")
    ax1.set_title("Risk is cut where it matters")
    ax1.legend()

    ax2.bar(x - w / 2, decomp["bh_ann_return"] * 100, w, color=P["buyhold"], label="Buy-and-hold")
    ax2.bar(x + w / 2, decomp["mm_ann_return"] * 100, w, color=P["managed"], label="Managed")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(decomp["regime"])
    ax2.set_ylabel("Annualised return (%)")
    ax2.set_title("Return by regime")
    ax2.legend()
    fig.suptitle("Where the edge comes from: exposure up in calm, risk down in turbulence",
                 fontweight="bold", y=1.02)
    return _save(fig, path)
