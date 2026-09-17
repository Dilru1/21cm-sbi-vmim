#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master 1x3 SBC diagnostic figure: posterior volume, calibration ratio, marginal widths.

Two publication targets share one code path:
    --style beamer   (default)  large type, sans-serif, thick strokes, vivid fills -> talks
    --style paper               compact, serif, hairline strokes, restrained fills -> journal

Axis convention requested here: tick labels stay plain decimals (0.5, 1, 10.2, ...) and
the common power of ten is factored out and printed once, above the axis, as x10^k.

Usage
-----
    python make_master_figure.py                          # beamer, jitter experiment
    python make_master_figure.py --exp jitter --style paper
    python make_master_figure.py --style paper --out fig/master.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import (AutoMinorLocator, FuncFormatter, LogLocator,
                               MaxNLocator, NullFormatter, NullLocator)

# ===========================================================================
# 1. Workspace
# ===========================================================================
REPO = Path.cwd().resolve()
if REPO.name == "notebook":
    REPO = REPO.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# ===========================================================================
# 2. Report sets
# ===========================================================================
REPORTS_NOISE = {
    "cnn vmim n1": "sbc_out/pathB/noise/vmim_n1",
    "cnn vmim n2": "sbc_out/pathB/noise/vmim_n2",
    "cnn vmim n3": "sbc_out/pathB/noise/vmim_n3",
    "cnn mse n1":  "sbc_out/pathB/noise/mse_n1",
    "cnn mse n2":  "sbc_out/pathB/noise/mse_n2",
    "cnn mse n3":  "sbc_out/pathB/noise/mse_n3",
}

REPORTS_JITTER = {
    "no jit no floor":     "sbc_out/pathB/dequant/jit0_floor0",
    "no jit 1e-2 floor":   "sbc_out/pathB/dequant/jit0_floor_p_01",
    "0p05 jit 1e-2 floor": "sbc_out/pathB/dequant/jit0p05_floor_p_01",
    "0p10 jit 1e-2 floor": "sbc_out/pathB/dequant/jit0p10_floor_p_01",
}

REPORT_SETS = {"jitter": REPORTS_JITTER, "noise": REPORTS_NOISE}

# Optional hand-written x-tick labels; anything absent is prettified automatically.
LABEL_OVERRIDES: dict[str, str] = {
    # "no jit no floor": "no jitter\nno floor",
}

PARAMS = [r"$\log_{10}(F_x)$", r"$\tau$", r"$r_{H/S}$", r"$\log_{10}(M_{min})$"]
PSHORT = {
    PARAMS[0]: r"$\log_{10}F_x$",
    PARAMS[1]: r"$\tau$",
    PARAMS[2]: r"$r_{H/S}$",
    PARAMS[3]: r"$\log_{10}M_{\min}$",
}

TOL = np.sqrt(2 / 29)          # 1-sigma tolerance on the SBC chi^2 ratio
JITTER = 0.09                  # horizontal spread of the per-arm dots
RNG = np.random.default_rng(42)
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# ===========================================================================
# 3. Palettes
# ===========================================================================
# Groups (panel 1). Maximally separated hues, all dark enough for white markers
# to read on top and distinguishable in grayscale by luminance ordering.
GROUP_COLORS = [
    "#E6194B",  # crimson
    "#0B5FD9",  # ultramarine
    "#0F9D58",  # emerald
    "#FF8C00",  # vivid orange
    "#8B1FC9",  # violet
    "#00B7C7",  # cyan
    "#E01FA8",  # magenta
    "#A8901A",  # ochre
]

# Parameters (panels 2 and 3). A deliberately different family from the group
# colours so a red dot in panel 2 is never confused with a red dot in panel 1.
PARAM_COLORS = [
    "#D81B60",  # raspberry
    "#1E5AE8",  # cobalt
    "#00875A",  # deep green
    "#F27E00",  # amber
    "#7B2FF7",  # purple
]

C_SAFE = "#00A07A"   # calibrated band
C_PERF = "#111111"   # ideal ratio = 1
C_MARG = "#5A5A5A"   # 1 + tol

# ===========================================================================
# 4. Style presets
# ===========================================================================
PRESETS = {
    "beamer": dict(
        family="sans-serif",
        fonts=["DejaVu Sans", "Calibri", "Arial"],
        figsize=(15.5, 5.4),
        fs_label=13.5, fs_tick=11.5, fs_legend=11.0, fs_title=14.5, fs_tag=13.0,
        lw_axes=1.2, lw_lines=2.4, ms_gv=9.5, ms_sig=6.2,
        pt_alpha=0.85, fill_alpha=0.14, band_alpha=0.16, grid_alpha=0.40,
        box_w=0.30, dpi=600, tick_len=5.0,
    ),
    "paper": dict(
        family="serif",
        fonts=["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        figsize=(13.2, 4.35),
        fs_label=10.5, fs_tick=9.0, fs_legend=9.0, fs_title=11.0, fs_tag=10.0,
        lw_axes=0.8, lw_lines=1.5, ms_gv=6.0, ms_sig=4.2,
        pt_alpha=0.75, fill_alpha=0.10, band_alpha=0.11, grid_alpha=0.28,
        box_w=0.26, dpi=600, tick_len=3.5,
    ),
}


def _rc(P):
    rc = {
        "font.family": P["family"],
        ("font.sans-serif" if P["family"] == "sans-serif" else "font.serif"): P["fonts"],
        "font.size": P["fs_tick"],
        "axes.labelsize": P["fs_label"],
        "axes.titlesize": P["fs_title"],
        "xtick.labelsize": P["fs_tick"],
        "ytick.labelsize": P["fs_tick"],
        "legend.fontsize": P["fs_legend"],
        "axes.linewidth": P["lw_axes"],
        "axes.axisbelow": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": P["tick_len"], "ytick.major.size": P["tick_len"],
        "xtick.minor.size": P["tick_len"] * 0.55, "ytick.minor.size": P["tick_len"] * 0.55,
        "xtick.major.width": P["lw_axes"], "ytick.major.width": P["lw_axes"],
        "xtick.minor.width": P["lw_axes"] * 0.75, "ytick.minor.width": P["lw_axes"] * 0.75,
        "legend.frameon": False,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.3,
        "lines.solid_capstyle": "round",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if P["family"] == "sans-serif":
        rc.update({
            "mathtext.fontset": "custom",
            "mathtext.rm": "DejaVu Sans",
            "mathtext.it": "DejaVu Sans:italic",
            "mathtext.bf": "DejaVu Sans:bold",
        })
    else:
        rc["mathtext.fontset"] = "dejavuserif"
    return rc


# ===========================================================================
# 5. Axis helpers  (plain decimal ticks + one factored power of ten on top)
# ===========================================================================
def plain_fmt(v, pos=None):
    """0.05 / 0.5 / 1 / 10.2 / 250 -- never 1e-9, never 10^0."""
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        s = f"{v:,.0f}"
    elif a >= 100:
        s = f"{v:.0f}"
    elif a >= 10:
        s = f"{v:.1f}"
    elif a >= 1:
        s = f"{v:.2f}"
    elif a >= 0.1:
        s = f"{v:.2f}"
    else:
        s = f"{v:.3f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def common_exponent(vals, lo_lim=-2, hi_lim=3):
    """Power of ten to factor out of the data. 0 means 'leave it alone'."""
    v = np.abs(np.asarray(vals, float))
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 0
    e = int(np.floor(np.log10(np.median(v))))
    return 0 if lo_lim <= e <= hi_lim else e


def put_exponent(ax, exp, P, axis="y"):
    """Print the factored multiplier once, above the axis, top-left."""
    if exp == 0:
        return
    ax.text(0.0, 1.015, rf"$\times 10^{{{exp}}}$",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=P["fs_tick"], color="0.20")


def set_scale(ax, lo, hi, log_when_range_exceeds=12.0, force=None):
    """Log only when the dynamic range earns it; plain decimal ticks either way."""
    use_log = force if force is not None else (
        lo > 0 and np.isfinite(hi / lo) and (hi / lo) > log_when_range_exceeds
    )
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    return use_log


def style_axis(ax, P):
    ax.tick_params(which="both", top=True, right=True, direction="in")
    for s in ax.spines.values():
        s.set_linewidth(P["lw_axes"])
        s.set_color("black")
        s.set_zorder(6)


def panel_tag(ax, text, P):
    ax.text(0.012, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=P["fs_tag"], fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.65",
                      lw=P["lw_axes"] * 0.8, alpha=0.92))


def column_bands(ax, n, P):
    """Alternating shading so the eye tracks a category across all three panels."""
    for i in range(n):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color="0.5", alpha=0.055, lw=0, zorder=0)


def draw_summary(ax, x, v, color, P):
    """Min-max whisker + IQR box + median bar for one category."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    w = P["box_w"]
    lo, hi = v.min(), v.max()
    if v.size > 1:
        ax.vlines(x, lo, hi, color=color, lw=P["lw_lines"] * 0.7, alpha=0.55, zorder=2)
        ax.hlines([lo, hi], x - w * 0.35, x + w * 0.35, color=color,
                  lw=P["lw_lines"] * 0.7, alpha=0.55, zorder=2)
    if v.size > 2:
        q1, q3 = np.percentile(v, [25, 75])
        ax.add_patch(Rectangle((x - w, q1), 2 * w, max(q3 - q1, 1e-300),
                               facecolor=color, alpha=P["fill_alpha"] + 0.04,
                               edgecolor=color, lw=P["lw_axes"] * 0.9, zorder=2))
    ax.hlines(np.median(v), x - w, x + w, color=color,
              lw=P["lw_lines"] * 1.35, zorder=5)


# ===========================================================================
# 6. Labels
# ===========================================================================
def pretty_label(g):
    if g in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[g]
    s = str(g)
    s = re.sub(r"(?<=\d)p(?=\d)", ".", s)                      # 0p05 -> 0.05
    s = re.sub(r"\b1e-(\d+)\b", r"$10^{-\1}$", s)              # 1e-2 -> 10^-2
    s = s.replace("jit", "jitter").replace("no jitter", "no jitter")
    s = s.replace("cnn ", "").replace("vmim", "VMIM").replace("mse", "MSE")
    return s


def wrap_label(g, maxlen=13):
    s = pretty_label(g)
    if "\n" in s:
        return s
    words = s.split()
    if len(s) <= maxlen or len(words) == 1:
        return s
    # balanced two-line break
    best, best_cost = 1, None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        cost = max(len(a), len(b)) + abs(len(a) - len(b)) * 0.4
        if best_cost is None or cost < best_cost:
            best, best_cost = k, cost
    return " ".join(words[:best]) + "\n" + " ".join(words[best:])


# ===========================================================================
# 7. Data loading
# ===========================================================================
def load_reports(reports, mode="filtered_dlogp10", root=None, verbose=True):
    frames, missing = [], []
    for label, p in reports.items():
        p = Path(p)
        if root is not None and not p.is_absolute():
            p = Path(root) / p
        f = p if p.suffix == ".csv" else p / "metrics.csv"
        if not f.exists():
            missing.append(f"{label}  ->  {f}")
            continue
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        if "mode" in d.columns:
            d = d[d["mode"] == mode].copy()
        else:
            d = d.copy()
        if d.empty:
            missing.append(f"{label}  ->  no rows with mode == {mode!r}")
            continue
        d["group"] = label
        if "arm" in d.columns:
            s = d["arm"].astype(str).str.extract(r"\bc(\d+)\s+n(\d+)\s*$")
            d["cs"] = pd.to_numeric(s[0], errors="coerce").astype("Int64")
            d["ns"] = pd.to_numeric(s[1], errors="coerce").astype("Int64")
        else:
            d["cs"], d["ns"] = pd.NA, pd.NA
        d["report"] = str(f.parent)
        frames.append(d)

    if verbose and missing:
        print("[warn] skipped:\n  " + "\n  ".join(missing), file=sys.stderr)
    if not frames:
        raise FileNotFoundError(
            "No metrics.csv found for any report. Run from the repo root, or pass "
            "--root /path/to/repo."
        )
    return pd.concat(frames, ignore_index=True)


def have(DF, col):
    return col in DF.columns and DF[col].notna().any()


# ===========================================================================
# 8. The figure
# ===========================================================================
def plot_master_1x3(DF, ORDER, COL, PARAMS, PSHORT, REF,
                    out="master_1x3_plot.pdf", style="beamer",
                    force_log_gv=None, force_log_wid=None,
                    guides=True, envelopes=True, note=None, suptitle=None):
    P = PRESETS[style]
    xt = np.arange(len(ORDER))
    xlabels = [wrap_label(g) for g in ORDER]
    params = [p for p in PARAMS if have(DF, f"sigma_{p}") or have(DF, f"calibnsig_{p}")]

    with plt.rc_context(_rc(P)):
        fig, axes = plt.subplots(1, 3, figsize=P["figsize"])
        ax_gv, ax_cal, ax_wid = axes

        # -------------------------------------------------------------------
        # Panel a: posterior volume
        # -------------------------------------------------------------------
        seeds = sorted(int(s) for s in DF["cs"].dropna().unique()) if "cs" in DF.columns else []
        mk = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(seeds)}

        gv_all = DF["GV_4x4"].values.astype(float)
        e_gv = common_exponent(gv_all)
        sc = 10.0 ** (-e_gv)

        lo_gv, hi_gv = np.inf, -np.inf
        for i, g in enumerate(ORDER):
            s = DF[DF["group"] == g]
            if s.empty:
                continue
            gv = s["GV_4x4"].values.astype(float) * sc
            lo_gv, hi_gv = min(lo_gv, gv.min()), max(hi_gv, gv.max())
            draw_summary(ax_gv, i, gv, COL[g], P)
            if seeds:
                for c_seed, sub in s.groupby("cs"):
                    v = sub["GV_4x4"].values.astype(float) * sc
                    ax_gv.plot([i] * len(v), v, ls="none",
                               marker=mk.get(int(c_seed) if pd.notna(c_seed) else -1, "o"),
                               ms=P["ms_gv"], color=COL[g], mfc=COL[g],
                               mec="white", mew=P["lw_axes"] * 0.7,
                               alpha=P["pt_alpha"], zorder=6)
            else:
                ax_gv.plot([i] * len(gv), gv, ls="none", marker="o", ms=P["ms_gv"],
                           color=COL[g], mec="white", mew=P["lw_axes"] * 0.7,
                           alpha=P["pt_alpha"], zorder=6)

        gv_log = set_scale(ax_gv, lo_gv, hi_gv, force=force_log_gv)
        if gv_log:
            pad = (hi_gv / lo_gv) ** 0.07
            ax_gv.set_ylim(lo_gv / pad ** 1.4, hi_gv * pad ** 2.0)
        else:
            span = max(hi_gv - lo_gv, 1e-12)
            ax_gv.set_ylim(max(0.0, lo_gv - 0.14 * span), hi_gv + 0.18 * span)

        put_exponent(ax_gv, e_gv, P)
        ax_gv.set_ylabel(r"$\mathrm{GV}=\det(\Sigma_{4\times4})^{1/2}$", labelpad=4)
        ax_gv.set_title("Posterior volume", pad=16, fontweight="bold")
        panel_tag(ax_gv, "a", P)

        if guides:
            ax_gv.annotate("", xy=(0.030, 0.14), xytext=(0.030, 0.40),
                           xycoords="axes fraction",
                           arrowprops=dict(arrowstyle="-|>", color="0.45",
                                           lw=P["lw_axes"] * 1.1,
                                           shrinkA=0, shrinkB=0))
            ax_gv.text(0.055, 0.27, "tighter", rotation=90, transform=ax_gv.transAxes,
                       ha="left", va="center", color="0.45", style="italic",
                       fontsize=P["fs_legend"] - 1)

        seed_handles = [Line2D([], [], ls="none", marker=mk[s], ms=P["ms_gv"] - 2.0,
                               color="0.30", label=f"seed {s}") for s in seeds]

        # -------------------------------------------------------------------
        # Panel b: calibration ratio
        # -------------------------------------------------------------------
        column_bands(ax_cal, len(ORDER), P)
        ax_cal.axhspan(0, 1.0 + TOL, color=C_SAFE, alpha=P["band_alpha"], lw=0, zorder=0)
        ax_cal.axhline(1.0, color=C_PERF, lw=P["lw_lines"] * 0.8, zorder=1)
        ax_cal.axhline(1.0 + TOL, color=C_MARG, ls=(0, (5, 2.5)),
                       lw=P["lw_lines"] * 0.7, zorder=1)

        hi_cal, lo_cal = 1.0 + 2 * TOL, 1.0 - 2 * TOL
        for pi, p in enumerate(params):
            col = f"calibnsig_{p}"
            if not have(DF, col):
                continue
            c = PARAM_COLORS[pi % len(PARAM_COLORS)]
            med, vmin, vmax = [], [], []
            for gi, g in enumerate(ORDER):
                v = DF[DF["group"] == g][col].values.astype(float) * TOL + 1.0
                v = v[np.isfinite(v)]
                if v.size == 0:
                    med.append(np.nan); vmin.append(np.nan); vmax.append(np.nan); continue
                med.append(np.median(v)); vmin.append(v.min()); vmax.append(v.max())
                jit = RNG.uniform(-JITTER, JITTER, size=len(v))
                ax_cal.plot(gi + jit, v, ls="none", marker="o", ms=P["ms_sig"],
                            color=c, alpha=0.55, mec="white", mew=P["lw_axes"] * 0.5, zorder=3)
            hi_cal = max(hi_cal, np.nanmax(vmax)); lo_cal = min(lo_cal, np.nanmin(vmin))
            if envelopes:
                ax_cal.fill_between(xt, vmin, vmax, color=c, alpha=P["fill_alpha"], lw=0, zorder=2)
            ax_cal.plot(xt, med, "-", color=c, lw=P["lw_lines"], zorder=4,
                        marker="o", ms=P["ms_sig"] * 0.75, mec="white",
                        mew=P["lw_axes"] * 0.6, label=PSHORT.get(p, p))

        span = max(hi_cal - min(lo_cal, 0.0), 1e-6)
        ax_cal.set_ylim(max(0.0, min(lo_cal, 1.0 - TOL) - 0.10 * span), hi_cal + 0.26 * span)
        set_scale(ax_cal, 1.0, 1.0, force=False)
        ax_cal.set_ylabel(r"SBC ratio  $\hat{\chi}^2/\chi^2_{\mathrm{exp}}$", labelpad=4)
        ax_cal.set_title("Calibration", pad=16, fontweight="bold")
        panel_tag(ax_cal, "b", P)

        y0 = ax_cal.get_ylim()[0]
        ax_cal.text(0.985, 0.5 * (y0 + 1.0 + TOL), "calibrated",
                    transform=ax_cal.get_yaxis_transform(),
                    ha="right", va="center", color=C_SAFE, style="italic",
                    fontsize=P["fs_legend"] - 0.5, zorder=7)
        ax_cal.annotate(r"$1+\sqrt{2/29}$", xy=(-0.55, 1.0 + TOL),
                        xytext=(1, 3), textcoords="offset points",
                        ha="left", va="bottom", color=C_MARG,
                        fontsize=P["fs_legend"] - 1.5, zorder=7)

        # -------------------------------------------------------------------
        # Panel c: marginal widths
        # -------------------------------------------------------------------
        column_bands(ax_wid, len(ORDER), P)
        lo_wid, hi_wid = np.inf, -np.inf
        col_lo = np.full(len(ORDER), np.inf)
        col_hi = np.full(len(ORDER), -np.inf)
        for pi, p in enumerate(params):
            col = f"sigma_{p}"
            if not have(DF, col):
                continue
            c = PARAM_COLORS[pi % len(PARAM_COLORS)]
            ref_vals = DF[DF["group"] == REF][col].values.astype(float)
            ref_med = np.median(ref_vals) if ref_vals.size else np.nan
            if not np.isfinite(ref_med) or ref_med == 0:
                continue
            med, vmin, vmax = [], [], []
            for gi, g in enumerate(ORDER):
                v = DF[DF["group"] == g][col].values.astype(float) / ref_med
                v = v[np.isfinite(v)]
                if v.size == 0:
                    med.append(np.nan); vmin.append(np.nan); vmax.append(np.nan); continue
                med.append(np.median(v)); vmin.append(v.min()); vmax.append(v.max())
                col_lo[gi] = min(col_lo[gi], v.min()); col_hi[gi] = max(col_hi[gi], v.max())
                jit = RNG.uniform(-JITTER, JITTER, size=len(v))
                ax_wid.plot(gi + jit, v, ls="none", marker="D", ms=P["ms_sig"] * 0.9,
                            color=c, alpha=0.55, mec="white", mew=P["lw_axes"] * 0.5, zorder=3)
            lo_wid = min(lo_wid, np.nanmin(vmin)); hi_wid = max(hi_wid, np.nanmax(vmax))
            if envelopes:
                ax_wid.fill_between(xt, vmin, vmax, color=c, alpha=P["fill_alpha"], lw=0, zorder=2)
            ax_wid.plot(xt, med, "-", color=c, lw=P["lw_lines"], zorder=4,
                        marker="D", ms=P["ms_sig"] * 0.7, mec="white",
                        mew=P["lw_axes"] * 0.6, label=PSHORT.get(p, p))

        ax_wid.axhline(1.0, color="0.15", ls=(0, (1, 1.8)), lw=P["lw_lines"] * 0.7, zorder=2)
        wid_log = set_scale(ax_wid, lo_wid, hi_wid, force=force_log_wid)
        if wid_log:
            pad = (hi_wid / lo_wid) ** 0.06
            ax_wid.set_ylim(lo_wid / pad, hi_wid * pad ** 2.6)
        else:
            span = max(hi_wid - lo_wid, 1e-9)
            ax_wid.set_ylim(lo_wid - 0.14 * span, hi_wid + 0.30 * span)

        # park the "reference" tag at whichever end of the y=1 line is free
        ref_i = ORDER.index(REF) if REF in ORDER else 0
        at_right = ref_i <= (len(ORDER) - 1) / 2
        end = len(ORDER) - 1 if at_right else 0
        x_ref = (len(ORDER) - 0.45) if at_right else -0.45
        ha_ref = "right" if at_right else "left"
        va_ref = "top" if col_lo[end] > 1.06 else "bottom"
        ax_wid.annotate(f"reference: {pretty_label(REF)}".replace("\n", " "),
                        xy=(x_ref, 1.0), xytext=(0, 3 if va_ref == "bottom" else -3),
                        textcoords="offset points", ha=ha_ref, va=va_ref,
                        color="0.30", style="italic", fontsize=P["fs_legend"] - 1.5, zorder=7)
        ax_wid.set_ylabel(r"relative width  $\sigma/\sigma_{\mathrm{ref}}$", labelpad=4)
        ax_wid.set_title("Marginal widths", pad=16, fontweight="bold")
        panel_tag(ax_wid, "c", P)

        # -------------------------------------------------------------------
        # Shared formatting
        # -------------------------------------------------------------------
        column_bands(ax_gv, len(ORDER), P)
        ref_i = ORDER.index(REF) if REF in ORDER else None
        for ax in axes:
            ax.set_xlim(-0.6, len(ORDER) - 0.4)
            ax.set_xticks(xt)
            ax.set_xticklabels(xlabels, linespacing=1.0, rotation=30,
                               ha="right", rotation_mode="anchor")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(True, axis="y", which="major", ls=(0, (2, 2.6)),
                    lw=P["lw_axes"] * 0.6, alpha=P["grid_alpha"], color="0.55", zorder=0)
            style_axis(ax, P)
            if ref_i is not None:
                ax.get_xticklabels()[ref_i].set_fontweight("bold")

        # one legend for both parameter panels, under the figure
        handles, labels = ax_cal.get_legend_handles_labels()
        if not handles:
            handles, labels = ax_wid.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                       bbox_to_anchor=(0.5, 0.0), frameon=False,
                       fontsize=P["fs_legend"], handlelength=2.0,
                       columnspacing=2.0, borderaxespad=0.0)
        if seed_handles:
            fig.legend(handles=seed_handles, loc="lower left", ncol=len(seed_handles),
                       bbox_to_anchor=(0.008, 0.0), frameon=False,
                       fontsize=P["fs_legend"] - 1, handletextpad=0.35,
                       columnspacing=1.1, borderaxespad=0.0)

        if suptitle:
            fig.suptitle(suptitle, fontsize=P["fs_title"] + 1.5, fontweight="bold", y=0.995)

        bottom_pad = 0.085 if handles else 0.02
        fig.tight_layout(rect=[0.0, bottom_pad, 1.0, 0.965 if suptitle else 1.0])
        fig.subplots_adjust(wspace=0.26)

        if note:
            fig.text(0.995, 0.008, note, ha="right", va="bottom",
                     fontsize=P["fs_legend"] - 2, color="0.45")

        out = str(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        fig.savefig(re.sub(r"\.pdf$", ".png", out), dpi=P["dpi"])
        plt.close(fig)
    return out


# ===========================================================================
# 9. Console summary
# ===========================================================================
def summarise(DF, ORDER, REF, params):
    rows = []
    for g in ORDER:
        s = DF[DF["group"] == g]
        row = {"group": g, "n_arms": len(s), "GV_median": np.median(s["GV_4x4"].values)}
        for p in params:
            c = f"calibnsig_{p}"
            if have(DF, c):
                row[f"cal[{PSHORT.get(p, p)}]"] = np.median(s[c].values * TOL + 1.0)
        rows.append(row)
    t = pd.DataFrame(rows).set_index("group")
    with pd.option_context("display.width", 160, "display.float_format", lambda v: f"{v:.4g}"):
        print(t)


# ===========================================================================
# 10. CLI
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=list(PRESETS), default="beamer")
    ap.add_argument("--exp", choices=list(REPORT_SETS), default="jitter")
    ap.add_argument("--mode", default="filtered_dlogp10")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None, help="reference group for panel c (default: first)")
    ap.add_argument("--suptitle", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--no-guides", action="store_true",
                    help="drop the 'tighter' direction arrow in panel a")
    ap.add_argument("--no-envelope", action="store_true",
                    help="drop the min-max bands in panels b and c (busy figures)")
    ap.add_argument("--log-gv", dest="log_gv", action="store_true", default=None)
    ap.add_argument("--no-log-gv", dest="log_gv", action="store_false")
    ap.add_argument("--log-width", dest="log_wid", action="store_true", default=None)
    ap.add_argument("--no-log-width", dest="log_wid", action="store_false")
    a = ap.parse_args(argv)

    reports = REPORT_SETS[a.exp]
    DF = load_reports(reports, mode=a.mode, root=a.root)
    ORDER = [g for g in reports if g in set(DF["group"])]
    if not ORDER:
        raise SystemExit("No groups survived loading.")
    COL = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(ORDER)}
    REF = a.ref if a.ref in ORDER else ORDER[0]

    out = a.out or f"sbc_out/_img/master_1x3_{a.exp}_{a.style}.pdf"
    note = a.note or f"{a.mode} - {len(DF)} arms - ref: {REF}"

    params = [p for p in PARAMS if have(DF, f"sigma_{p}") or have(DF, f"calibnsig_{p}")]
    summarise(DF, ORDER, REF, params)

    path = plot_master_1x3(DF, ORDER, COL, PARAMS, PSHORT, REF, out=out, style=a.style,
                           force_log_gv=a.log_gv, force_log_wid=a.log_wid,
                           guides=not a.no_guides, envelopes=not a.no_envelope,
                           note=note, suptitle=a.suptitle)
    print(f"\nSaved: {path}\n       {re.sub(r'.pdf$', '.png', path)}")


if __name__ == "__main__":
    main()