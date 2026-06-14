"""Shared configuration: paths, constants, sub-periods, and a unified plot style.

Everything that other modules need to agree on lives here so the pipeline stays
internally consistent (single source of truth for annualisation factors,
sub-period dates, random seeds, file locations and figure styling).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
PAPER_TABLES = PAPER / "tables"

for _p in (DATA_RAW, DATA_PROC, FIGURES, RESULTS, PAPER_TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Sample / frequency constants                                                 #
# --------------------------------------------------------------------------- #
START_DATE = "2000-01-01"          # primary sample start (covers dot-com, GFC, COVID)
END_DATE = "2026-04-30"            # pinned snapshot for reproducibility (set None for latest)
TRADING_DAYS = 252                 # annualisation for daily series
MONTHS_PER_YEAR = 12               # annualisation for monthly series
DAYS_PER_MONTH = 21                # nominal trading days per month (GARCH aggregation)
RANDOM_SEED = 42

# Ken French daily research factors (Mkt-RF, SMB, HML, RF), direct CSV download.
KEN_FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

# --------------------------------------------------------------------------- #
# Sub-periods for the robustness / regime analysis (Phase 4 & 5)              #
# --------------------------------------------------------------------------- #
SUBPERIODS = {
    "Dot-com bust": ("2000-01-01", "2002-12-31"),
    "Mid-2000s calm": ("2003-01-01", "2007-06-30"),
    "Global Financial Crisis": ("2007-07-01", "2009-06-30"),
    "Post-GFC bull": ("2009-07-01", "2019-12-31"),
    "COVID crash & rebound": ("2020-01-01", "2020-12-31"),
    "Post-COVID": ("2021-01-01", "2100-01-01"),
}

# Transaction-cost grid (bps per unit of turnover) used in the robustness section.
COST_GRID_BPS = [0, 1, 2, 5, 10, 20]
# Leverage caps (multiples of full exposure) used in the robustness section.
LEVERAGE_CAPS = [1.0, 1.5, 2.0, 3.0, np.inf]

# Number of distinct strategy/model configurations effectively searched.
# Used by the Deflated Sharpe Ratio to penalise multiple testing.
N_TRIALS_TESTED = 3  # naive, GARCH, HAR volatility forecasters

# --------------------------------------------------------------------------- #
# Unified plot style                                                           #
# --------------------------------------------------------------------------- #
PALETTE = {
    "managed": "#1f6feb",     # blue
    "buyhold": "#6e7781",     # grey
    "accent": "#d1242f",      # red
    "good": "#1a7f37",        # green
    "neutral": "#8250df",     # purple
    "calm": "#2da44e",        # regime: calm (green)
    "crisis": "#cf222e",      # regime: crisis (red)
    "intermediate": "#bf8700",  # regime: intermediate (amber)
}
REGIME_COLORS = [PALETTE["calm"], PALETTE["crisis"], PALETTE["intermediate"]]


def set_plot_style() -> None:
    """Apply a clean, publication-quality matplotlib/seaborn style."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    except Exception:  # pragma: no cover - seaborn optional at style time
        pass

    mpl.rcParams.update(
        {
            "figure.figsize": (9, 5.2),
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    _ = plt  # keep import used


def annualization_factor(freq: str) -> int:
    """Return the annualisation factor for a return frequency ('D' or 'M')."""
    f = freq.upper()
    if f.startswith("D"):
        return TRADING_DAYS
    if f.startswith("M"):
        return MONTHS_PER_YEAR
    raise ValueError(f"Unknown frequency {freq!r}")
