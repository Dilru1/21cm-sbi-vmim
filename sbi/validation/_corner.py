# _corner.py -- publication / beamer quality corner plots
import os
import sys
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# 1. Palettes & Settings
# ---------------------------------------------------------------------------
VIVID = ["#2E6FD9", "#E24A33", "#00A878", "#8E5AE8", "#F2A900", "#00B7C7"]
TRUTH_COLOR = "#2B2B2B"
try:                                    # keep the project palette available
    from _palette import OKABE_ITO
except Exception:                       # noqa: BLE001
    OKABE_ITO = ["#0072B2", "#D55E00", "#009E73",
                 "#CC79A7", "#E69F00", "#56B4E9"]
 
PALETTES = {"vivid": VIVID, "okabe": OKABE_ITO}
 

# ---------------------------------------------------------------------------
# 2. Style presets
# ---------------------------------------------------------------------------
STYLE = "beamer"

PRESETS = {
    "beamer": dict(
        fig_width_multiplier=1.2,  # slightly larger panels for slides
        family="sans-serif", fonts=["DejaVu Sans", "Calibri"],
        fs_label=12, fs_tick=9.5, fs_legend=10.5, fs_title=11.0,
        lw_axes=1.0, lw_contour=1.4, lw_hist=1.8, lw_truth=1.2,
        fill_alpha=(0.14, 0.35),   # slightly more vivid for projectors
        pad_label=4.0, panel_space=0.06,
    ),
    "paper": dict(
        fig_width_multiplier=1.0,  # fits two-column journal standard
        family="serif", fonts=["DejaVu Serif", "Times New Roman"],
        fs_label=10, fs_tick=7.5, fs_legend=8.5, fs_title=9.5,
        lw_axes=0.8, lw_contour=1.0, lw_hist=1.3, lw_truth=0.9,
        fill_alpha=(0.10, 0.26),   # (95%, 68%) contour fill opacity
        pad_label=3.0, panel_space=0.06,
    ),
}

def _rc(P):
    return {
        "font.family": P["family"],
        ("font.sans-serif" if P["family"] == "sans-serif" else "font.serif"): P["fonts"],
        "mathtext.fontset": "custom" if P["family"] == "sans-serif" else "dejavuserif",
        "font.size": P["fs_tick"],
        "axes.labelsize": P["fs_label"],
        "axes.titlesize": P["fs_title"],
        "xtick.labelsize": P["fs_tick"],
        "ytick.labelsize": P["fs_tick"],
        "legend.fontsize": P["fs_legend"],
        "axes.linewidth": P["lw_axes"],
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": P["lw_axes"], "ytick.major.width": P["lw_axes"],
        "xtick.minor.width": P["lw_axes"] * 0.75,
        "ytick.minor.width": P["lw_axes"] * 0.75,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def credible_levels(H, fracs=(0.68, 0.95)):
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    out = []
    for f in fracs:
        i = np.searchsorted(csum, f)
        out.append(flat[i] if i < len(flat) else flat[-1])
    return out[::-1]

def _smooth2d(H, sigma):
    if sigma <= 0:
        return H
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(H, sigma)
    except ImportError:
        return H

# ---------------------------------------------------------------------------
# 4. Rendering Engine
# ---------------------------------------------------------------------------
def render_corner(arm_samples, truth, common_params, labels, out_pdf,
                  style=None, bins=30, smooth=1.0, title=None, also_png=False):
    P = dict(PRESETS[style or STYLE])
    cols = VIVID
    d = len(common_params)
    
    # Calculate physical width (max 7.1 inches for paper)
    base_width = min(7.1, 1.55 * d + 0.9)
    width = base_width * P["fig_width_multiplier"]

    with plt.rc_context(_rc(P)):
        fig, axes = plt.subplots(d, d, figsize=(width, width), squeeze=False)
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.10, top=0.985,
                            wspace=P["panel_space"], hspace=P["panel_space"])

        # ---- shared limits -------------------------------------------------
        lims = []
        for k in range(d):
            lo = min(np.percentile(s[:, common_params[k]], 0.3) for _, s in arm_samples)
            hi = max(np.percentile(s[:, common_params[k]], 99.7) for _, s in arm_samples)
            pad = 0.06 * (hi - lo)
            lims.append((lo - pad, hi + pad))

        for r in range(d):
            for c in range(d):
                ax = axes[r][c]
                if c > r:
                    ax.axis("off")
                    continue
                jr, jc = common_params[r], common_params[c]

                # ---------------- diagonal: 1D step histograms -------------
                if r == c:
                    for k, (_lab, s) in enumerate(arm_samples):
                        ax.hist(s[:, jr], bins=bins, range=lims[r], density=True,
                                histtype="step", lw=P["lw_hist"],
                                color=cols[k % len(cols)], zorder=2 + k)
                        
                    ax.axvline(truth[jr], color=TRUTH_COLOR, lw=P["lw_truth"], 
                               ls=(0, (4, 2.5)), zorder=10)
                    ax.set_xlim(*lims[r])
                    ax.set_ylim(bottom=0)

                # ---------------- off-diagonal: 2D contours ----------------
                else:
                    for k, (_lab, s) in enumerate(arm_samples):
                        H, xe, ye = np.histogram2d(s[:, jc], s[:, jr], bins=bins, 
                                                   range=[lims[c], lims[r]])
                        H = _smooth2d(H.astype(float), smooth)
                        
                        if H.sum() <= 0:
                            continue

                        lv = sorted(set(credible_levels(H)))
                        if not lv:
                            continue

                        max_val = H.max()
                        if max_val <= lv[-1]:
                            max_val = lv[-1] + 1e-5

                        col = cols[k % len(cols)]
                        levels_f = lv + [max_val]
                        
                        n_fill = len(levels_f) - 1
                        alphas = P["fill_alpha"][-n_fill:] if n_fill <= len(P["fill_alpha"]) \
                            else (P["fill_alpha"][0],) * n_fill
                        colors_f = [to_rgba(col, a) for a in alphas]

                        xc_mid = 0.5 * (xe[1:] + xe[:-1])
                        yc_mid = 0.5 * (ye[1:] + ye[:-1])

                        ax.contourf(xc_mid, yc_mid, H.T, levels=levels_f,
                                    colors=colors_f, zorder=2 + k)
                        ax.contour(xc_mid, yc_mid, H.T, levels=lv,
                                   colors=[col] * len(lv), linewidths=P["lw_contour"],
                                   zorder=10 + k)

                    ax.axvline(truth[jc], color=TRUTH_COLOR, lw=P["lw_truth"] * 0.9,
                               ls=(0, (4, 2.5)), alpha=0.8, zorder=1)
                    ax.axhline(truth[jr], color=TRUTH_COLOR, lw=P["lw_truth"] * 0.9,
                               ls=(0, (4, 2.5)), alpha=0.8, zorder=1)
                    
                    # Truth marker
                    ax.plot(truth[jc], truth[jr], marker="*", ms=5.5,
                            color=TRUTH_COLOR, mec="white", mew=0.5, zorder=20)
                    
                    ax.set_xlim(*lims[c])
                    ax.set_ylim(*lims[r])

                # ---------------- ticks, labels ----------------------------
                ax.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                
                if r == c:
                    ax.set_yticks([])
                    ax.tick_params(axis="y", which="both", left=False, right=False)
                else:
                    ax.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

                if r == d - 1:
                    ax.set_xlabel(labels[jc], labelpad=P["pad_label"])
                    for t in ax.get_xticklabels():
                        t.set_rotation(45)
                        t.set_horizontalalignment("right")
                        t.set_rotation_mode("anchor")
                else:
                    ax.set_xticklabels([])

                if c == 0 and r > 0:
                    ax.set_ylabel(labels[jr], labelpad=P["pad_label"])
                    for t in ax.get_yticklabels():
                        t.set_rotation(45)
                        t.set_verticalalignment("top")
                        t.set_rotation_mode("anchor")
                elif c != 0:
                    ax.set_yticklabels([])

        # ---- Legend ---------------------------
        handles = [Line2D([0], [0], color=cols[k % len(cols)], lw=P["lw_hist"], label=lab)
                   for k, (lab, _) in enumerate(arm_samples)]
        handles.append(Line2D([0], [0], color=TRUTH_COLOR, lw=P["lw_truth"],
                              ls=(0, (4, 2.5)), marker="*", ms=5.5,
                              mec="white", mew=0.5, label="truth"))

        if d > 1:
            x1 = axes[0][d - 1].get_position().x1
            y1 = axes[0][0].get_position().y1
        else:
            x1, y1 = 0.985, 0.985
            
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(x1, y1),
                   bbox_transform=fig.transFigure, borderaxespad=0.0,
                   handlelength=1.9, handletextpad=0.6, labelspacing=0.45)

        if title:
            fig.text(x1, y1 - 0.012 - 0.045 * len(handles), title,
                     ha="right", va="top", fontsize=P["fs_title"], color="0.25")

        with PdfPages(out_pdf) as pdf:
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
        if also_png:
            fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
            
        plt.close(fig)
    return out_pdf

# ---------------------------------------------------------------------------
# 5. Pipeline wrappers
# ---------------------------------------------------------------------------
def pick_corner_sim(arms, maps, common_params, thin, burnin, dlogp, n_corner):
    from eval import load_chain
    shared = set(maps[0])
    for m in maps[1:]:
        shared &= set(m)
    shared = np.array(sorted(shared))

    if len(shared) == 0:
        print("[corner] selected arms share no simulations", file=sys.stderr)
        return None

    print(f"[corner] shared sims: {len(shared)}")

    ref, rmap = arms[0], maps[0]
    gvs = {}
    for s in shared:
        ld = load_chain(rmap[s], ref["n_params"], burnin, thin, dlogp)
        if ld is None:
            continue
        cov = np.cov(ld[0][:, common_params], rowvar=False)
        sign, ldet = np.linalg.slogdet(cov)
        if sign > 0 and np.isfinite(ldet):
            gvs[int(s)] = np.exp(0.5 * ldet)

    if not gvs:
        return None

    med = np.median(list(gvs.values()))
    sorted_sims = sorted(gvs.keys(), key=lambda s: abs(gvs[s] - med))
    best_sims = sorted_sims[:n_corner]
    
    print(f"[corner] selected {len(best_sims)} sims closest to median GV: {best_sims}")
    return best_sims

def plot_corner(report, out, burnin_frac, thin, dlogp, n_corner, common_params, labels, 
                style=None, bins=30, smooth=1.0, title_sim=True):
    from eval import list_chains, chain_sim, load_chain

    for mode, arms in report.items():
        if not arms:
            continue

        maps = [{chain_sim(f): f for f in list_chains(a["dir"]) if chain_sim(f) is not None} for a in arms]
        sims_to_plot = pick_corner_sim(arms, maps, common_params, thin, burnin_frac, dlogp, n_corner)

        if not sims_to_plot:
            continue

        for sim in sims_to_plot:
            safe_mode = re.sub(r'[^A-Za-z0-9]+', '_', mode)
            out_pdf = os.path.join(out, f"corner_sim{sim}_{safe_mode}.pdf")

            arm_samples, truth = [], None
            for a, m in zip(arms, maps, strict=False):
                if sim not in m:
                    continue
                ld = load_chain(m[sim], a["n_params"], burnin_frac, thin, dlogp)
                if ld is None:
                    continue
                s, tr = ld
                if truth is None:
                    truth = tr
                arm_samples.append((a["name"], s))

            if not arm_samples:
                print(f"[corner] no usable chains for sim {sim}", file=sys.stderr)
                continue

            render_corner(arm_samples, truth, common_params, labels, out_pdf,
                          style=style, bins=bins, smooth=smooth, 
                          title=(f"sim {sim}" if title_sim else None))
            print(f"[corner] wrote {out_pdf}")