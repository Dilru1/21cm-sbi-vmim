"""Shared publication style for MSc report Chapter 3 figures.

Design decisions (kept consistent across every figure):
  * serif / TeX-like fonts (STIX) so figures blend with the LaTeX body text
  * a single colour language:
      - continuous parameter colouring  -> 'viridis' (colour-blind safe)
      - the three redshifts             -> fixed Tol-muted triplet Z_COLORS
      - diverging (correlations)        -> 'RdBu_r', symmetric about 0
  * vector PDF output (+ optional 300 dpi PNG for quick slides)
  * figure widths matched to a thesis \textwidth of ~6.3 in
"""
from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------- rcParams
TEXTWIDTH_IN = 6.3          # full \textwidth figure
COLWIDTH_IN = 4.5           # ~0.7\textwidth figure

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,      # embed TrueType -> text stays editable/searchable
    "axes.prop_cycle": mpl.cycler(
        color=["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
               "#AA3377", "#BBBBBB"]),  # Paul Tol 'bright' (CB-safe)
}

# fixed colours for the three observation redshifts (Tol muted, CB-safe)
Z_COLORS = {0: "#332288", 1: "#117733", 2: "#CC6677"}
Z_LABELS = {0: r"$z=8.18$", 1: r"$z=10.32$", 2: r"$z=12.06$"}
REDSHIFTS = [8.18, 10.32, 12.06]

# LaTeX-consistent parameter labels, in param.dat column order
PARAM_TEX = [r"$f_X$", r"$\tau$ [Myr]", r"$r_{H/S}\ (\theta_2)$",
             r"$\log_{10} M_{\min}$", r"$f_{\mathrm{esc}}$"]
PARAM_SHORT = ["fX", "tau", "rHS", "Mmin", "fesc"]
PARAM_LOG = [True, True, False, False, False]   # plot on log axis?


def apply_style():
    mpl.rcParams.update(RC)


def save(fig, outdir: str, name: str, png: bool = True):
    """Save vector PDF (for LaTeX) and optionally a 300 dpi PNG preview."""
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(pdf)
    if png:
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf}")


def despine(ax, top=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.tick_params(top=top, right=right)


def panel_label(ax, text, loc="upper left", fs=9):
    """Small bold panel tag like (a), (b) ..."""
    xy = {"upper left": (0.03, 0.95), "upper right": (0.97, 0.95),
          "lower left": (0.03, 0.06), "lower right": (0.97, 0.06)}[loc]
    ha = "left" if "left" in loc else "right"
    ax.text(*xy, text, transform=ax.transAxes, fontsize=fs,
            fontweight="bold", va="top", ha=ha)


def symlog_hist_bins(x, n=80, lo_q=0.05, hi_q=99.95):
    lo, hi = np.percentile(x, [lo_q, hi_q])
    return np.linspace(lo, hi, n)
