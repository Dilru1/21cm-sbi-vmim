#!/usr/bin/env python3
"""
Publication-quality compressor training diagnostics -> vector PDF.
Focused purely on the RF R^2 probe to demonstrate the "Greedy Allocation" effect.
Dynamically maps layouts: 1x4 for a single group, N x 4 for multiple groups.
Uses square subplots, clean panel styling, and highlights the peak R^2 value.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===========================================================================
# 1. Palettes
# ===========================================================================
GROUP_COLORS = [
    "#E6194B",  # crimson
    "#0B5FD9",  # ultramarine
    "#0F9D58",  # emerald
    "#FF8C00",  # vivid orange
    "#8B1FC9",  # violet
    "#00B7C7",  # cyan
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# ===========================================================================
# 2. Style presets
# ===========================================================================
PRESETS = {
    "beamer": dict(
        family="sans-serif",
        fonts=["DejaVu Sans", "Calibri", "Arial"],
        fs_label=11.0, fs_tick=9.5, fs_legend=9.5, fs_title=12.0, fs_tag=11.0,
        lw_axes=1.2, lw_lines=1.8, ms_gv=6.0,
        pt_alpha=0.85, grid_alpha=0.40,
        dpi=600, tick_len=4.0,
    ),
}

def _rc(P):
    return {
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
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "lines.solid_capstyle": "round",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "DejaVu Sans",
    }

def panel_tag(ax, text, P):
    ax.text(0.03, 0.95, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=P["fs_tag"], fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="0.65",
                      lw=P["lw_axes"] * 0.7, alpha=0.92))

# ===========================================================================
# 3. Data Loading
# ===========================================================================
def load_arm(spec, P):
    if "=" in spec and not os.path.exists(spec):
        label, path = spec.rsplit("=", 1)
    else:
        path, label = spec, os.path.basename(os.path.abspath(spec.rstrip("/")))
    nle = path if os.path.isdir(path) else os.path.dirname(path)

    p = os.path.join(nle, "rf_r2_history.npy")
    if not os.path.exists(p):
        sys.exit(f"[FATAL] '{label}': missing {p}")
        
    rf = np.load(p)
    rf = rf[None, :] if rf.ndim == 1 else rf
    
    if rf.shape[1] < P + 1:
        sys.exit(f"[FATAL] '{label}': rf_r2_history has {rf.shape[1]} cols, need >= {P + 1}")

    config_name = re.sub(r"\s+s\d+\s*$", "", label).strip()
    seed_match = re.search(r"s(\d+)\s*$", label)
    seed_id = seed_match.group(1) if seed_match else "1"

    return dict(
        label=label,
        config=config_name,
        seed=seed_id,
        ep=rf[:, 0],
        r2=rf[:, -P:],
        phase=(rf[:, 1] if rf.shape[1] > P + 1 else None),
    )

def switches(ep, phase):
    if phase is None:
        return []
    return [ep[i + 1] for i in np.where(np.diff(phase) != 0)[0]]

# ===========================================================================
# 4. Main Plotting Routine
# ===========================================================================
def main():
    
    # ---------------------------------------------------------
    # HARDCODED CONFIGURATIONS
    # ---------------------------------------------------------
    input_arms = [
        # Group 1: No jitter, no floor (3 seeds)
        "no jit no floor s43=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter_floor_p0/seed_n1_s43/nle",
        "no jit no floor s44=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter_floor_p0/seed_n1_s44/nle",
        "no jit no floor s45=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter_floor_p0/seed_n1_s45/nle",
    
        # Group 2: No jitter, 0.01 floor (3 seeds)
        "no jit 0.01 floor s43=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter/seed_n2_s43/nle",
        "no jit 0.01 floor s44=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter/seed_n2_s44/nle",
        "no jit 0.01 floor s45=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter/seed_n2_s45/nle"

        # Group 3: 0.10 jitter, 0.01 floor (3 seeds)
        # "p01 jit 0.01 floor s43=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/jitter/seed_n1_s43/nle",
        # "p01 jit 0.01 floor s44=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/jitter/seed_n1_s44/nle",
        # "p01 jit 0.01 floor s45=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/jitter/seed_n1_s45/nle"    
    ]
    names = ["Fx", "tau", "rHS", "Mmin"]
    out_path = "sbc_out/_img/r2_training_dynamics_n2vmim.pdf"
    style = "beamer"
    r2_ylim = (-0.15, 1.15)
    # ---------------------------------------------------------

    PR = PRESETS[style]
    P = len(names)
    arms = [load_arm(s, P) for s in input_arms]

    unique_configs = []
    for a in arms:
        if a["config"] not in unique_configs:
            unique_configs.append(a["config"])

    config_to_color = {
        cfg: GROUP_COLORS[i % len(GROUP_COLORS)] 
        for i, cfg in enumerate(unique_configs)
    }
    

    unique_seeds = sorted(list(set(a["seed"] for a in arms)))
    seed_to_marker = {
        sd: MARKERS[i % len(MARKERS)] 
        for i, sd in enumerate(unique_seeds)
    }

    PSHORT = {
        "Fx": r"$\log_{10}F_x$", "tau": r"$\tau$",
        "rHS": r"$r_{H/S}$", "Mmin": r"$\log_{10}M_{\min}$"
    }

    n_groups = len(unique_configs)
    
    with plt.rc_context(_rc(PR)):
        # Calculate figure size to enforce square-ish panels (4 columns by N rows)
        fig_width = 13.5
        panel_size = fig_width / 4.0
        fig_height = panel_size * n_groups + 1.2
        
        fig, axes = plt.subplots(n_groups, 4, figsize=(fig_width, fig_height), sharex=True, sharey=True)
        if n_groups == 1:
            axes = np.expand_dims(axes, axis=0)

        for g_idx, cfg in enumerate(unique_configs):
            cfg_arms = [a for a in arms if a["config"] == cfg]
            c = config_to_color[cfg]

            for p in range(P):
                ax = axes[g_idx, p] if n_groups > 1 else axes[p]
                
                ax.axhline(0.0, color="0.15", lw=PR["lw_axes"]*1.2, zorder=1)
                ax.axhline(1.0, color="0.65", lw=PR["lw_axes"]*1.2, ls="--", zorder=1)
                
                # Track max value across seeds in this panel to star/annotate it
                max_val = -np.inf
                max_pos = None

                for a in cfg_arms:
                    mk = seed_to_marker[a["seed"]]
                    ep_arr = a["ep"]
                    r2_arr = a["r2"][:, p]
                    
                    ax.plot(ep_arr, r2_arr, ls="-", marker=mk, color=c, 
                            mfc="white", ms=PR["ms_gv"]*1, mew=PR["lw_axes"]*0.7, 
                            lw=PR["lw_lines"], zorder=3, alpha=1) # PR["pt_alpha"])
                    
                    # Find local peak for star annotation
                    local_max_idx = np.argmax(r2_arr)
                    if r2_arr[local_max_idx] > max_val:
                        max_val = r2_arr[local_max_idx]
                        max_pos = (ep_arr[local_max_idx], r2_arr[local_max_idx])

                    for x in switches(ep_arr, a["phase"]):
                        ax.axvline(x, color=c, lw=PR["lw_axes"], ls=":", alpha=0.55, zorder=1)

                # Annotate highest value with a star
                if max_pos is not None:
                    ax.plot(max_pos[0], max_pos[1], marker="*", color="#D4AF37", ms=9, mec="black", mew=0.5, zorder=10)
                    ax.annotate(f"{max_pos[1]:.2f}", xy=max_pos, xytext=(0, 6), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333333", zorder=10)

                ax.set_ylim(*r2_ylim)
                
                if g_idx == 0:
                    ax.set_title(PSHORT.get(names[p], names[p]), pad=6, fontweight="bold")
                
                if p == 3:
                    ax.annotate(cfg, xy=(1.03, 0.5), xycoords="axes fraction",
                                rotation=270, ha="left", va="center", fontsize=PR["fs_legend"]-0.5, fontweight="bold", color=c)

                ax.grid(True, axis="y", which="major", ls=(0, (2, 2.6)), lw=PR["lw_axes"]*0.5, alpha=PR["grid_alpha"], color="0.55", zorder=0)
                ax.tick_params(which="both", top=True, right=True, direction="in")
                for s in ax.spines.values():
                    s.set_linewidth(PR["lw_axes"]); s.set_color("black"); s.set_zorder(6)
                
                if g_idx == n_groups - 1:
                    ax.set_xlabel("Epoch", fontweight="bold", labelpad=3)
                if p == 0:
                    ax.set_ylabel(r"Predictive Power ($R^2$)", labelpad=4)
                
                panel_tag(ax, f"{chr(97 + g_idx*4 + p)}", PR)

        # Legend handles
        legend_handles = []
        for cfg, col in config_to_color.items():
            legend_handles.append(Line2D([0], [0], color=col, lw=2.5, label=cfg))
        for sd, mk in seed_to_marker.items():
            legend_handles.append(Line2D([0], [0], color="0.3", marker=mk, ls="none", 
                                          mfc="white", mew=0.8, ms=5, label=f"seed {sd}"))

        if legend_handles:
            fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
                       bbox_to_anchor=(0.5, 0.0), frameon=False,
                       fontsize=PR["fs_legend"], handlelength=1.8,
                       columnspacing=1.2, borderaxespad=0.0)

        fig.suptitle("The Greedy Allocation Effect (Per Configuration)", fontsize=PR["fs_title"] + 1.0, fontweight="bold", y=0.98)

        fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.95])
        
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        fig.savefig(re.sub(r"\.pdf$", ".png", out_path), dpi=PR["dpi"])
        print(f"Saved: {out_path}")
        print(f"       {re.sub(r'.pdf$', '.png', out_path)}")

if __name__ == "__main__":
    main()