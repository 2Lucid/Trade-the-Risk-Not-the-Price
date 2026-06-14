"""Generate a LinkedIn-ready carousel from the project figures.

Produces square-ish portrait (1080x1350) slides: a designed dark cover, one slide
per key figure (with a clean header/footer wrapper), and a dark call-to-action
slide. Saves PNGs to ``linkedin/slides/`` and a single ``linkedin/carousel.pdf``
ready to upload as a LinkedIn *document* post.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src import config  # noqa: E402

OUT = config.ROOT / "linkedin"
SLIDES = OUT / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1350
DPI = 100
INK = "#0a0f22"
NAME = "Clément Bellet-Odent"
REPO = "github.com/2Lucid/Trade-the-Risk-Not-the-Price"
BLUE, GREEN, RED, GREY = "#5b8bff", "#3ddc84", "#ff6b6b", "#9fb0e0"


def _canvas(dark=True):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    if dark:
        grad = np.linspace(0, 1, 256).reshape(-1, 1)
        bg.imshow(grad, extent=[0, 1, 0, 1], aspect="auto", cmap=_cmap(), zorder=-1)
    else:
        bg.add_patch(plt.Rectangle((0, 0), 1, 1, color="#ffffff", zorder=-1))
    return fig


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("hero", ["#131a3a", "#0a0f22"])


def _pill(fig, text, y=0.90, color=GREY):
    fig.text(0.5, y, text, ha="center", va="center", color=color, fontsize=15,
             fontweight="bold", fontfamily="DejaVu Sans",
             bbox=dict(boxstyle="round,pad=0.6", fc="none", ec=color, lw=1.2))


def cover():
    fig = _canvas(dark=True)
    _pill(fig, "VOLATILITY-MANAGED PORTFOLIOS", 0.90)
    fig.text(0.08, 0.74, "You can't predict", color="#ffffff", fontsize=46, fontweight="bold")
    fig.text(0.08, 0.675, "the price.", color="#ffffff", fontsize=46, fontweight="bold")
    fig.text(0.08, 0.575, "You can predict", color=BLUE, fontsize=46, fontweight="bold")
    fig.text(0.08, 0.51, "the risk.", color=BLUE, fontsize=46, fontweight="bold")
    fig.text(0.08, 0.40,
             "A Journal of Finance strategy (Moreira–Muir, 2017),\n"
             "rebuilt and stress-tested on 26 years of global markets.",
             color="#c4cee8", fontsize=18, linespacing=1.5)
    fig.text(0.08, 0.25, "Sharpe  0.48 → 0.72      Drawdown halved      α  t = 3.6",
             color=GREEN, fontsize=17, fontweight="bold")
    fig.text(0.08, 0.085, NAME, color="#ffffff", fontsize=18, fontweight="bold")
    fig.text(0.92, 0.085, "swipe →", color=GREY, fontsize=16, ha="right")
    return fig


def figure_slide(img_path, eyebrow, caption):
    fig = _canvas(dark=False)
    fig.text(0.07, 0.93, eyebrow, color=BLUE, fontsize=16, fontweight="bold")
    fig.text(0.07, 0.90, caption, color=INK, fontsize=23, fontweight="bold", va="top", wrap=True)
    img = plt.imread(str(img_path))
    ar = img.shape[0] / img.shape[1]
    fw = 0.90
    fh = fw * (W / H) * ar
    fh = min(fh, 0.62)
    y0 = 0.46 - fh / 2
    ax = fig.add_axes([0.05, y0, fw, fh]); ax.axis("off")
    ax.imshow(img)
    fig.text(0.07, 0.06, "Trade the Risk, Not the Price", color="#5b667d", fontsize=13, fontweight="bold")
    fig.text(0.93, 0.06, NAME, color="#5b667d", fontsize=13, ha="right")
    return fig


def cta():
    fig = _canvas(dark=True)
    _pill(fig, "EXPLORE IT YOURSELF", 0.90)
    fig.text(0.08, 0.74, "Want the details?", color="#ffffff", fontsize=42, fontweight="bold")
    lines = [
        (BLUE, "Interactive site — move the sliders, re-run it live"),
        (GREEN, "Full research paper (PDF, 16 pages)"),
        (GREY, "Open-source code & data pipeline"),
    ]
    y = 0.60
    for col, txt in lines:
        fig.text(0.09, y, "●", color=col, fontsize=18, va="center")
        fig.text(0.14, y, txt, color="#dce4f7", fontsize=20, va="center")
        y -= 0.085
    fig.text(0.08, 0.27, REPO, color=BLUE, fontsize=18, fontweight="bold")
    fig.text(0.08, 0.085, NAME, color="#ffffff", fontsize=18, fontweight="bold")
    fig.text(0.92, 0.085, "Research & education — not investment advice.",
             color=GREY, fontsize=12, ha="right")
    return fig


SLIDE_FIGS = [
    ("fig01_predictability_r2.png", "01 · The honest test",
     "Forecasting returns fails. Forecasting\nvolatility works."),
    ("fig05_equity_curve.png", "02 · The result",
     "Same risk as the market — higher return,\nsmaller crashes."),
    ("fig07_regimes.png", "03 · It explains itself",
     "An algorithm rediscovered every crisis —\nwith no dates given."),
    ("fig09_cross_asset_sharpe.png", "04 · Does it travel?",
     "Across 11 markets: the trend overlay does\nthe heavy lifting."),
    ("fig10_diversified_equity.png", "05 · The powerful part",
     "Diversify + manage risk + trend →\nSharpe 0.39 to 0.72."),
]


def main():
    figs = [("cover", cover())]
    for fname, eye, cap in SLIDE_FIGS:
        p = config.FIGURES / fname
        if p.exists():
            figs.append((fname.split(".")[0], figure_slide(p, eye, cap)))
    figs.append(("cta", cta()))

    pdf_path = OUT / "carousel.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, (name, fig) in enumerate(figs):
            fig.savefig(SLIDES / f"slide{i:02d}_{name}.png", dpi=DPI)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {len(figs)} slides to {SLIDES} and {pdf_path}")


if __name__ == "__main__":
    main()
