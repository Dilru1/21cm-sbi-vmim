#!/usr/bin/env python3
"""Latent 'island' figure: noise smear vs parameter separation, Beamer-sized.

    python tools/plot_latent_islands.py --out figs/latent_islands
    python tools/plot_latent_islands.py --out figs/x --n-islands 6 --width 5.8

WHAT IT SHOWS
-------------
Each panel is one arm. Coloured clouds are individual SIMULATIONS -- every point
is the same theta with a different noise realisation, so the cloud size IS the
noise smear and the distance between clouds IS the parameter signal. A
compressor is good when the islands are small and far apart.

WHY SMALL MULTIPLES AND NOT ONE OVERLAID PANEL
----------------------------------------------
VMIM maximises I(t; theta), which is invariant under any invertible
reparameterisation of t, so two seeds converge to latents related by an
arbitrary rotation: seed 43's t_0 is not seed 44's t_0. Plotting them on shared
axes makes seeds look inconsistent when they may carry identical information,
and it is worse across objectives (an MSE arm with direct_regression=True has
t ~ theta, a specific basis).

So every panel gets its OWN basis, and the comparison is carried by
  * the SAME simulations in the SAME colours in every panel, so the reader
    tracks a simulation across arms, and
  * the separation ratio S printed on each panel: median between-island
    distance divided by median within-island RMS, in that panel's own
    standardised coordinates. S is dimensionless and basis-free, so unlike the
    axes it IS comparable between panels. Bigger = better compressor.

The projection is the top-2 PCA of the PER-SIM MEANS (not of all rows), i.e.
the plane that maximises between-simulation structure -- the plane where the
islands are most separable. Using raw t_0/t_1 shows an arbitrary slice.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --------------------------------------------------------------------------
ARMS = [
    ("VMIM s43", "/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter/seed_n1_s43/summaries"),
    ("VMIM s44", "/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter/seed_n1_s44/summaries"),
    ("VMIM s45", "/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter/seed_n1_s45/summaries"),
    ("MSE s43",  "/gscratch/ddehiwalage-don/sbi/cnn_mse_up/seed_n1_s43/summaries"),
    ("MSE s44",  "/gscratch/ddehiwalage-don/sbi/cnn_mse_up/seed_n1_s44/summaries"),
    ("MSE s45",  "/gscratch/ddehiwalage-don/sbi/cnn_mse_up/seed_n1_s45/summaries"),
]

# Okabe-Ito: colour-blind safe, and it survives greyscale printing
ISLAND_C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def rc(fs):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.4,
        "figure.dpi": 150, "savefig.dpi": 400,
    })


def load(path, max_rows):
    t = np.load(os.path.join(path, "t.npy"), mmap_mode="r")
    s = np.load(os.path.join(path, "original_sim_ids.npy"), mmap_mode="r")
    n = min(len(t), max_rows)
    return np.asarray(t[:n], np.float64), np.asarray(s[:n])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="latent_islands")
    ap.add_argument("--n-islands", type=int, default=5)
    ap.add_argument("--ncol", type=int, default=3)
    ap.add_argument("--width", type=float, default=5.6,
                    help="inches; 5.6 fills a 16:9 Beamer body with room for a title")
    ap.add_argument("--panel-aspect", type=float, default=0.92)
    ap.add_argument("--fontsize", type=float, default=7.0)
    ap.add_argument("--max-rows", type=int, default=1_500_000)
    ap.add_argument("--bg", type=int, default=1500, help="background points per panel")
    ap.add_argument("--zoom", type=float, default=1.35,
                    help="axis half-range as a multiple of the island spread; "
                         "smaller = tighter crop on the islands")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    rc(args.fontsize)

    arms = [(l, p) for l, p in ARMS if os.path.isdir(p)]
    for l, p in ARMS:
        if not os.path.isdir(p):
            print(f"[skip] {l}: no such dir {p}")
    if not arms:
        raise SystemExit("no arms found")

    # the SAME simulations in every panel -- that is what carries the comparison
    common = None
    for _, p in arms:
        s = set(np.unique(np.load(os.path.join(p, "original_sim_ids.npy"), mmap_mode="r")))
        common = s if common is None else (common & s)
    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(sorted(common), args.n_islands, replace=False))
    print(f"{len(arms)} arms, {len(common):,} shared sims, islands = {picks.tolist()}")

    nrow = int(np.ceil(len(arms) / args.ncol))
    pw = args.width / args.ncol
    fig, axes = plt.subplots(nrow, args.ncol,
                             figsize=(args.width, pw * args.panel_aspect * nrow),
                             squeeze=False)

    stats = []
    for i, (label, path) in enumerate(arms):
        ax = axes[i // args.ncol][i % args.ncol]
        t, sims = load(path, args.max_rows)

        # standardise, then project on the top-2 PCA of the PER-SIM MEANS: the
        # plane that maximises between-simulation structure, i.e. where the
        # islands are most separable. Raw t0/t1 is an arbitrary slice.
        mu, sd = t.mean(0), t.std(0) + 1e-12
        Z = (t - mu) / sd
        uniq, inv = np.unique(sims, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        M = np.stack([np.bincount(inv, weights=Z[:, j]) for j in range(Z.shape[1])], 1) / cnt[:, None]
        P = np.linalg.svd(M - M.mean(0), full_matrices=False)[2][:2].T
        Zp, Mp = Z @ P, M @ P

        bg = rng.choice(len(Zp), min(args.bg, len(Zp)), replace=False)
        ax.scatter(Zp[bg, 0], Zp[bg, 1], s=1.2, c="0.72", alpha=.30,
                   linewidths=0, rasterized=True, zorder=1)

        cen, wrms = [], []
        for k, sid in enumerate(picks):
            blk = Zp[sims == sid]
            c = ISLAND_C[k % len(ISLAND_C)]
            ax.scatter(blk[:, 0], blk[:, 1], s=2.2, color=c, alpha=.45,
                       linewidths=0, rasterized=True, zorder=3)
            m = blk.mean(0)
            ax.plot(m[0], m[1], "*", ms=6.5, color=c, mec="k", mew=.45, zorder=5)
            cen.append(m)
            wrms.append(np.sqrt(((blk - m) ** 2).sum(1).mean()))
        cen = np.asarray(cen)

        # S = median between-island distance / median within-island RMS.
        # Dimensionless and basis-free, so it IS comparable across panels.
        dij = [np.linalg.norm(cen[a] - cen[b])
               for a in range(len(cen)) for b in range(a + 1, len(cen))]
        S = np.median(dij) / max(np.median(wrms), 1e-12)
        stats.append((label, S, np.median(wrms), np.median(dij)))

        # crop on the islands, not the full cloud
        ctr = cen.mean(0)
        r = args.zoom * max(np.abs(cen - ctr).max(), 3 * np.median(wrms))
        ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, pad=2.5, fontweight="bold")
        ax.text(.035, .965, f"$S={S:.1f}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=args.fontsize - 0.5,
                bbox=dict(fc="white", ec="0.7", lw=.4, pad=1.3, alpha=.85), zorder=8)
        if i % args.ncol == 0:
            ax.set_ylabel("PC2", labelpad=1.5)
        if i // args.ncol == nrow - 1:
            ax.set_xlabel("PC1", labelpad=1.5)

    for k in range(len(arms), nrow * args.ncol):
        axes[k // args.ncol][k % args.ncol].axis("off")

    h = [mlines.Line2D([], [], ls="", marker="o", ms=3.2,
                       color=ISLAND_C[k % len(ISLAND_C)], label=f"sim {s}")
         for k, s in enumerate(picks)]
    h.append(mlines.Line2D([], [], ls="", marker="*", ms=6, color="0.35",
                           mec="k", mew=.45, label="island mean"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False,
               handletextpad=.25, columnspacing=.9,
               bbox_to_anchor=(.5, -0.012))

    if args.title:
        fig.suptitle(args.title, y=.995, fontweight="bold")
    fig.tight_layout(rect=(0, .045, 1, .995 if args.title else 1), h_pad=.55, w_pad=.55)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'arm':<12}{'S':>7}{'within RMS':>12}{'between':>10}")
    for l, S, w, d in stats:
        print(f"{l:<12}{S:7.2f}{w:12.3f}{d:10.3f}")
    print("\nS = median between-island distance / median within-island RMS,")
    print("in each panel's own standardised PCA plane. Dimensionless, so it is")
    print("comparable ACROSS panels even though the axes are not.")
    print(f"\nwrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()