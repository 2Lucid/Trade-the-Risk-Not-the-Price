"""Phase 5 — The HMM regime layer (explaining *why* it works).

A Gaussian Hidden Markov Model is fit to monthly market returns and discovers
market regimes on its own — no crisis dates are ever supplied.  States are
relabelled by variance so the labelling is stable: 0 = calm, ... , k-1 = crisis.

**The methodological distinction that separates a real quant from a beginner**
(stated explicitly because it matters):

* **Smoothed** inference (full-sample forward-backward posteriors) is used only
  to *visualise* and *describe* the regime structure.  This is legitimate
  because it is descriptive, not a trading decision.
* **Filtered** inference (online, past information only) is the *only* thing
  admissible if the regime is used inside a strategy.  We implement it the
  rigorous way: at each month we refit the HMM on data up to that month and take
  the end-point posterior (which, having no future data, equals the filtered
  probability).  Using smoothed states in a backtest would be lookahead bias.

The strategy itself is *not* regime-conditioned; the HMM is used to (1) produce
the regime-coloured figure and (2) decompose where the out-performance is
earned.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src import config

# hmmlearn logs harmless "Model is not converging" notes (tiny EM deltas at the
# tolerance boundary) via the logging module; quiet them.
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

# Labels are ordered by ascending variance (calm -> turbulent). The high-variance
# state contains the recognised crises (2008, 2020, 2022) but also other
# elevated-volatility spells, so "Turbulent" is the honest name for it.
STATE_NAMES = {2: ["Calm", "Turbulent"], 3: ["Calm", "Stressed", "Turbulent"]}


def _fit_one(X: np.ndarray, n_states: int, seed: int):
    """Fit a GaussianHMM and return (model, variance-ascending relabel map)."""
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        tol=1e-4,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    variances = model.covars_.reshape(n_states, -1).sum(axis=1)
    order = np.argsort(variances)            # ascending variance => calm..crisis
    relabel = {old: new for new, old in enumerate(order)}
    return model, relabel


def fit_hmm_smoothed(
    monthly: pd.DataFrame, n_states: int = 2, seed: int = config.RANDOM_SEED
) -> dict:
    """Fit the HMM on the full sample; return SMOOTHED states/probs (for figures)."""
    X = monthly["mkt_excess"].values.reshape(-1, 1)
    model, relabel = _fit_one(X, n_states, seed)

    raw_states = model.predict(X)                  # Viterbi path
    raw_probs = model.predict_proba(X)             # smoothed posteriors
    states = np.array([relabel[s] for s in raw_states])
    # reorder probability columns to the calm..crisis labelling
    inv = {new: old for old, new in relabel.items()}
    probs = raw_probs[:, [inv[k] for k in range(n_states)]]

    idx = monthly.index
    state_ser = pd.Series(states, index=idx, name="state")
    prob_df = pd.DataFrame(
        probs, index=idx, columns=[f"p_{STATE_NAMES[n_states][k]}" for k in range(n_states)]
    )

    # per-state descriptive statistics
    stats_rows = []
    for k in range(n_states):
        mask = state_ser == k
        r = monthly.loc[mask, "mkt_excess"]
        stats_rows.append(
            {
                "state": k,
                "label": STATE_NAMES[n_states][k],
                "n_months": int(mask.sum()),
                "frequency": float(mask.mean()),
                "ann_mean": float(r.mean() * 12),
                "ann_vol": float(r.std(ddof=1) * np.sqrt(12)),
                "avg_duration": _avg_duration(state_ser, k),
            }
        )
    return {
        "model": model,
        "states": state_ser,
        "probs": prob_df,
        "state_stats": pd.DataFrame(stats_rows),
        "n_states": n_states,
        "labels": STATE_NAMES[n_states],
    }


def _avg_duration(states: pd.Series, k: int) -> float:
    """Average number of consecutive months spent in state k."""
    runs, cur = [], 0
    for s in states.values:
        if s == k:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def filtered_regimes(
    monthly: pd.DataFrame,
    n_states: int = 2,
    min_train: int = 60,
    refit_every: int = 1,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Lookahead-free FILTERED regime probabilities (walk-forward refit).

    At each month ``t`` we refit on data up to ``t`` and take the end-point
    posterior, which equals the filtered probability (no future data).  States
    are relabelled by variance at every refit to stay aligned.  This is the only
    regime signal admissible inside a backtest.
    """
    X_all = monthly["mkt_excess"].values.reshape(-1, 1)
    n = len(X_all)
    out = np.full((n, n_states), np.nan)
    model = relabel = None
    for t in range(min_train, n):
        if model is None or (t - min_train) % refit_every == 0:
            model, relabel = _fit_one(X_all[: t + 1], n_states, seed)
        post = model.predict_proba(X_all[: t + 1])[-1]  # filtered prob at time t
        inv = {new: old for old, new in relabel.items()}
        out[t] = post[[inv[k] for k in range(n_states)]]
    cols = [f"filt_{STATE_NAMES[n_states][k]}" for k in range(n_states)]
    return pd.DataFrame(out, index=monthly.index, columns=cols)


def regime_decomposition(strat: pd.DataFrame, states: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Decompose managed vs buy-and-hold performance by (smoothed) regime.

    Reports, per regime, the average managed and buy-and-hold returns and the
    share of the strategy's total cumulative out-performance earned there.
    """
    df = strat.join(states.rename("state"), how="inner").dropna(subset=["state"])
    df["excess_over_bh"] = df["managed"] - df["f"]
    total_outperf = df["excess_over_bh"].sum()

    rows = []
    for k, label in enumerate(labels):
        seg = df[df["state"] == k]
        if len(seg) == 0:
            continue
        contrib = seg["excess_over_bh"].sum()
        rows.append(
            {
                "regime": label,
                "n_months": int(len(seg)),
                "frequency": float(len(seg) / len(df)),
                "avg_exposure": float(seg["weight"].mean()),
                "mm_ann_return": float(seg["managed"].mean() * 12),
                "bh_ann_return": float(seg["f"].mean() * 12),
                "mm_ann_vol": float(seg["managed"].std(ddof=1) * np.sqrt(12)),
                "bh_ann_vol": float(seg["f"].std(ddof=1) * np.sqrt(12)),
                "monthly_outperf": float(seg["excess_over_bh"].mean()),
                "share_of_total_outperf": float(contrib / total_outperf) if total_outperf != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    from src.data_loader import load_dataset
    from src.strategy import build_managed_strategy
    from src.vol_forecast import forecast_naive

    _, m = load_dataset()
    hmm = fit_hmm_smoothed(m, n_states=2)
    print(hmm["state_stats"].round(4).to_string(index=False))
    strat, _ = build_managed_strategy(m, forecast_naive(m), leverage_cap=2.0)
    print()
    print(regime_decomposition(strat, hmm["states"], hmm["labels"]).round(4).to_string(index=False))
