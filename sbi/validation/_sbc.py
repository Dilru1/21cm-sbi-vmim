# _sbc.py
import os
import math
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from _palette import arm_color, arm_linestyle, FIGFMT

# --- cosmetic settings ------------------------------------------------------
BAND_COLOR = "#4C72B0"      # 68% / 95% envelope
REF_COLOR  = "#3B3B3B"      # miscalibration reference curves
UNIT_COLOR = "#8C8C8C"      # "perfect calibration" line at 1

_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.9,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
# ---------------------------------------------------------------------------


def _sbc_reference_curves(nbins, n_used):
    denom = float(n_used) / float(nbins)
    s = 1.0 / math.sqrt(denom) if denom > 0 else 0.0
    bands = {"low1": 1 - s, "high1": 1 + s, "low2": 1 - 2 * s, "high2": 1 + 2 * s}

    def _cdf_hist(loc, scale, n=400000):
        d = np.random.normal(loc, scale, n)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(d / np.sqrt(2.0)))
        idx = np.clip((cdf * nbins).astype(int), 0, nbins - 1)
        return np.bincount(idx, minlength=nbins).astype(float) / (n / nbins)

    return bands, _cdf_hist(0.2, 1.0), _cdf_hist(0.0, 0.85)


def plot_sbc(report, out, nbins, common_params, labels):
    xval = np.arange(0.5 / nbins, 1.0, 1.0 / nbins)

    for mode, arms in report.items():
        n_used = max((r["used"] for r in arms), default=0)
        bands, bias_hist, under_hist = _sbc_reference_curves(nbins, n_used)

        with plt.rc_context(_RC):
            fig, axes = plt.subplots(
                len(common_params), 1,
                figsize=(12, 2.5 * len(common_params)),
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)

            for jj, j in enumerate(common_params):
                ax = axes[jj]
                ax.fill_between([0, 1], bands["low2"], bands["high2"],
                                color=BAND_COLOR, alpha=0.13, lw=0, zorder=0)
                ax.fill_between([0, 1], bands["low1"], bands["high1"],
                                color=BAND_COLOR, alpha=0.22, lw=0, zorder=0)
                ax.axhline(1.0, color=UNIT_COLOR, lw=0.8, ls=(0, (5, 4)),
                           xmax=1.0 / 1.25, zorder=1)

                for i, r in enumerate(arms):
                    rh = np.asarray(r["rank_hist"])[jj]
                    ax.plot(xval, rh, drawstyle="steps-mid", lw=1.6,
                            color=arm_color(i), ls=arm_linestyle(i),
                            solid_joinstyle="miter", solid_capstyle="butt",
                            zorder=4 + i, label=r["name"])

                ax.plot(xval, bias_hist, ls="--", lw=1.6, color=REF_COLOR,
                        zorder=3, label=r"0.2$\sigma$ bias")
                ax.plot(xval, under_hist, ls="dotted", lw=1.8, color=REF_COLOR,
                        zorder=3, label=r"0.15$\sigma$ under-confidence")

                ax.set_ylim(0, 2); ax.set_xlim(0, 1.25)
                ax.set_xticks(np.arange(0, 1.01, 0.2))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(which="both", top=True, right=False)
                ax.set_xlabel(f"CDF value of {labels[j]}")

                # same legend box, cleaner handles
                handles, leg_labels = ax.get_legend_handles_labels()
                handles = [
                    Line2D([], [], color=h.get_color(), ls=h.get_linestyle(),
                           lw=1.8, solid_capstyle="butt")
                    for h in handles
                ]
                handles += [
                    Patch(facecolor=BAND_COLOR, alpha=0.22, lw=0),
                    Patch(facecolor=BAND_COLOR, alpha=0.13, lw=0),
                ]
                leg_labels += ["68% interval", "95% interval"]
                leg = ax.legend(
                    handles, leg_labels,
                    loc="center",                           
                    bbox_to_anchor=(1.0, 0.0, 0.25, 1.8),
                    bbox_transform=ax.transData,            # data coordinates
                    mode="expand",                          # fill the 0.24 width
                    borderaxespad=0.0,
                    fontsize=9,
                    framealpha=1.0,
                    handlelength=2.4,
                    handleheight=1.0,
                    handletextpad=0.7,
                    labelspacing=0.7,
                    borderpad=0.7,
                    fancybox=False,
                    edgecolor="none",
                )
                leg.get_frame().set_linewidth(0.8)

            safe = re.sub(r"[^A-Za-z0-9]+", "_", mode)
            path = os.path.join(out, f"sbc_overlay_{safe}.{FIGFMT}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[sbc] wrote {path}")