#!/usr/bin/env python3
"""What the compression actually buys: latent islands + the RESOLUTION LIMIT.

    python tools/plot_res_sep7.py \
        "SKA 100h"=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter/seed_n1_s43/summaries \
        "SKA 25h"=/gscratch/ddehiwalage-don/sbi/cnn_vmim_up/no_jitter/seed_n2_s43/summaries \
        --names Fx tau rHS Mmin --out notebook/figs/resolution \
        --sigma-measured 4.06e-3,9.77e-3,17.33e-3,8.20e-3 \
                         6.07e-3,9.29e-3,28.92e-3,7.53e-3

THE IDEA
--------
The compressor maps 3 x 32^3 = 98,304 voxels to 4 numbers. Whether that is
lossy in a way that MATTERS is a question about one length scale: how far apart
must two simulations be in theta before their latent islands stop overlapping?

Linearising t(theta) about the mean, a change dtheta_j moves the latent by
||J_j|| dtheta_j, where J_j is the j-th column of the Jacobian estimated by
least squares from the per-simulation latent means. Two simulations become
distinguishable when that displacement exceeds the noise smear s (the RMS
spread of a single simulation's island). So

    dtheta_j^min  =  s / ||J_j||

is a RESOLUTION LIMIT in physical parameter units, derived from the latent
alone -- before any NLE, before any MCMC.

Both s and ||J_j|| are computed in standardised latent coordinates, so their
ratio is invariant under the arbitrary rotation VMIM leaves in t. That is what
makes this comparable across seeds and objectives, unlike the island picture
itself.

Pass --sigma-measured to overlay the posterior widths stage 4 measured. If the
prediction tracks the measurement, the chain latent -> NLE -> posterior is
doing what it should, and any parameter where they DIVERGE is one where the
NLE, not the compressor, is the bottleneck.

Figures (prefix = --out):
  _islands.png   the intuition: island size vs island separation, per arm
  _resolution.png  the claim: predicted dtheta^min, optionally vs measured sigma
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def rc(fs):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif", "font.size": fs,
        "axes.labelsize": fs, "axes.titlesize": fs,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1, "legend.fontsize": fs - 1,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
        "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .4,
        "figure.dpi": 150, "savefig.dpi": 400})


def load(spec, max_rows):
    lab, path = spec.split("=", 1)
    t = np.load(os.path.join(path, "t.npy"), mmap_mode="r")
    th = np.load(os.path.join(path, "theta.npy"), mmap_mode="r")
    s = np.load(os.path.join(path, "original_sim_ids.npy"), mmap_mode="r")
    n = min(len(t), max_rows)
    t, th, s = np.asarray(t[:n], np.float64), np.asarray(th[:n], np.float64), np.asarray(s[:n])

    # standardise, then per-sim means and within-sim variance in ONE pass
    mu, sd = t.mean(0), t.std(0) + 1e-12
    Z = (t - mu) / sd
    uniq, inv = np.unique(s, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    d = Z.shape[1]
    s1 = np.stack([np.bincount(inv, weights=Z[:, j]) for j in range(d)], 1)
    s2 = np.stack([np.bincount(inv, weights=Z[:, j] ** 2) for j in range(d)], 1)
    M = s1 / cnt[:, None]
    V = np.maximum(s2 / cnt[:, None] - M ** 2, 0) * (cnt / np.maximum(cnt - 1, 1))[:, None]
    first = np.zeros(len(uniq), int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]

    # smear = RMS radius of one island in the full latent space
    smear = float(np.sqrt(V.sum(1).mean()))
    TH = th[first]
    # Jacobian by least squares: M ~ TH.  ||J_j|| is how far a unit change in
    # theta_j moves the latent.
    A = np.c_[TH - TH.mean(0), np.ones(len(TH))]
    J = np.linalg.lstsq(A, M - M.mean(0), rcond=None)[0][:-1]      # (n_params, t_dim)
    Jn = np.linalg.norm(J, axis=1)
    return dict(label=lab, Z=Z, sims=s, M=M, TH=TH, sim_ids=uniq,
                smear=smear, Jnorm=Jn, dmin=smear / np.maximum(Jn, 1e-12))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="'label'=summaries_dir")
    ap.add_argument("--names", nargs="+", default=["Fx", "tau", "rHS", "Mmin"])
    ap.add_argument("--out", default="resolution")
    ap.add_argument("--max-rows", type=int, default=1_500_000)
    ap.add_argument("--n-islands", type=int, default=5)
    ap.add_argument("--sigma-measured", nargs="+", default=None,
                    help="one comma list per arm, in the same order: the posterior "
                         "sigmas from stage 4 (metrics.csv)")
    ap.add_argument("--grid", nargs="+", type=float, default=None,
                    help="theta grid spacing per parameter, to mark the "
                         "quantisation floor (e.g. 0.2 for rHS on 6 points)")
    ap.add_argument("--width", type=float, default=5.6)
    ap.add_argument("--fontsize", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rc(args.fontsize)

    arms = [load(s, args.max_rows) for s in args.arms]
    P = len(args.names)
    for a in arms:
        print(f"[{a['label']:12s}] smear={a['smear']:.4f}  "
              f"|J_j|={np.round(a['Jnorm'],2)}  dmin={np.round(a['dmin']*1e3,2)} (x1e-3)")

    sig = None
    if args.sigma_measured:
        sig = [np.array([float(x) for x in s.split(",")]) for s in args.sigma_measured]

    # ---------- 1. islands: the intuition ----------
    rng = np.random.default_rng(args.seed)
    common = set(arms[0]["sim_ids"])
    for a in arms[1:]:
        common &= set(a["sim_ids"])
    picks = np.sort(rng.choice(sorted(common), args.n_islands, replace=False))

    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(args.width, args.width / n * 1.06),
                             squeeze=False)
    for i, a in enumerate(arms):
        ax = axes[0][i]
        Pj = np.linalg.svd(a["M"] - a["M"].mean(0), full_matrices=False)[2][:2].T
        Zp = a["Z"] @ Pj
        bg = rng.choice(len(Zp), min(1500, len(Zp)), replace=False)
        ax.scatter(Zp[bg, 0], Zp[bg, 1], s=1.1, c="0.75", alpha=.30,
                   linewidths=0, rasterized=True)
        cen = []
        for k, sid in enumerate(picks):
            b = Zp[a["sims"] == sid]
            ax.scatter(b[:, 0], b[:, 1], s=2.0, color=C[k % len(C)], alpha=.45,
                       linewidths=0, rasterized=True)
            m = b.mean(0); cen.append(m)
            ax.plot(*m, "*", ms=6, color=C[k % len(C)], mec="k", mew=.4, zorder=5)
        cen = np.asarray(cen)
        ctr = cen.mean(0); r = 1.35 * max(np.abs(cen - ctr).max(), 3 * a["smear"])
        ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r)
        # scale bar = the noise smear, the thing the separation must beat
        ax.plot([ctr[0] - .8 * r, ctr[0] - .8 * r + a["smear"]],
                [ctr[1] - .85 * r] * 2, "k-", lw=2, solid_capstyle="butt")
        ax.text(ctr[0] - .8 * r + a["smear"] / 2, ctr[1] - .78 * r,
                "noise smear", ha="center", va="bottom", fontsize=args.fontsize - 1.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(a["label"], fontweight="bold", pad=2.5)
    fig.suptitle("latent islands: one colour = one simulation, all noise realisations",
                 y=1.005, fontsize=args.fontsize)
    fig.tight_layout(); fig.savefig(f"{args.out}_islands.png", bbox_inches="tight")
    fig.savefig(f"{args.out}_islands.pdf", bbox_inches="tight"); plt.close(fig)

    # ---------- 2. resolution limit: the claim ----------
    fig, ax = plt.subplots(figsize=(args.width, args.width * .52))
    W = .8 / max(n, 1)
    x = np.arange(P)
    for i, a in enumerate(arms):
        off = (i - (n - 1) / 2) * W
        ax.bar(x + off, a["dmin"][:P], W * .86, color=C[i], alpha=.55,
               edgecolor=C[i], lw=.8,
               label=f"{a['label']}  (predicted from latent)")
        if sig is not None:
            ax.plot(x + off, sig[i][:P], "k_", ms=W * 62, mew=1.6, zorder=6)
    if args.grid:
        for j, g in enumerate(args.grid[:P]):
            ax.hlines(g / np.sqrt(12), j - .42, j + .42, color="0.35", ls=":", lw=1.1,
                      zorder=7)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(args.names[:P], fontsize=args.fontsize + .5)
    ax.set_ylabel(r"$\delta\theta_j^{\min}=s/\|J_j\|$   (normalised $\theta$)")
    ax.set_title("parameter resolution limit implied by the latent space", fontsize=args.fontsize)
    h, l = ax.get_legend_handles_labels()
    if sig is not None:
        h.append(Line2D([], [], color="k", marker="_", ls="", ms=9, mew=1.6,
                        label="measured posterior $\\sigma$ (stage 4)"))
    if args.grid:
        h.append(Line2D([], [], color="0.35", ls=":", lw=1.1,
                        label=r"grid floor $\Delta\theta/\sqrt{12}$"))
    ax.legend(handles=h, fontsize=args.fontsize - 1.3, framealpha=.9, loc="best")
    fig.tight_layout(); fig.savefig(f"{args.out}_resolution.png", bbox_inches="tight")
    fig.savefig(f"{args.out}_resolution.pdf", bbox_inches="tight"); plt.close(fig)

    print(f"\n{'param':<10}" + "".join(f"{a['label']:>22}" for a in arms))
    for j in range(P):
        row = f"{args.names[j]:<10}"
        for i, a in enumerate(arms):
            s = f"{a['dmin'][j]*1e3:8.2f}"
            if sig is not None:
                s += f" / {sig[i][j]*1e3:6.2f}"
            row += f"{s:>22}"
        print(row)
    print("\n(x1e-3, normalised theta. 'predicted / measured' when --sigma-measured given.)")
    if len(arms) == 2:
        r = arms[1]["dmin"][:P] / arms[0]["dmin"][:P]
        print(f"\nresolution ratio {arms[1]['label']}/{arms[0]['label']}: "
              + " ".join(f"{v:.2f}" for v in r))
        print("naive noise scaling would give 2.00 for a 4x shorter integration.")
    print(f"\nwrote {args.out}_islands.pdf/.png and {args.out}_resolution.pdf/.png")


if __name__ == "__main__":
    main()