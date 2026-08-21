#!/usr/bin/env python3
"""Compare exported latent summaries t across noise levels (or any two arms).

Reads theta.npy, t.npy, original_sim_ids.npy (+ export_noise_scale.npy for the
label) from each summaries/ dir.

Figures written (prefix = --out):
    _snr.png      per-latent-dim within/between spread + latent SNR, both arms
    _pca.png      PCA of t (fitted on arm 1, applied to both), coloured by theta
    _smear.png    noise-smear clouds for a handful of individual simulations
    _corr.png     |corr(t_i, theta_j)| heatmaps -- which dims encode which param
    _gridband.png visualizes the jitter effect on a discrete parameter
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def diverging_arm_colors(n, cmap="coolwarm", lo=0.12, hi=0.88):
    cm = plt.get_cmap(cmap)
    if n == 1:
        return [cm(0.5)]
    return [cm(x) for x in np.linspace(lo, hi, n)]


def load_arm(spec):
    label, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
    t = np.load(os.path.join(path, "t.npy"), mmap_mode="r")
    theta = np.load(os.path.join(path, "theta.npy"), mmap_mode="r")
    sims = np.load(os.path.join(path, "original_sim_ids.npy"), mmap_mode="r")
    ns_path = os.path.join(path, "export_noise_scale.npy")
    ns = float(np.load(ns_path)[0]) if os.path.exists(ns_path) else float("nan")
    print(
        f"[{label}] rows={len(t):,} t_dim={t.shape[1]} "
        f"n_sims={len(np.unique(np.asarray(sims))):,} noise_scale={ns:g}",
        flush=True,
    )
    return dict(
        label=label,
        t=np.asarray(t, np.float64),
        theta=np.asarray(theta, np.float64),
        sims=np.asarray(sims),
        noise_scale=ns,
    )


def decompose(arm, max_sims=1500, rng=None):
    rng = rng or np.random.default_rng(0)
    t, sims, theta = arm["t"], arm["sims"], arm["theta"]
    uniq = np.unique(sims)
    if len(uniq) > max_sims:
        uniq = rng.choice(uniq, max_sims, replace=False)

    means, wvars, th = [], [], []
    for s in uniq:
        m = sims == s
        block = t[m]
        if block.shape[0] < 2:
            continue
        means.append(block.mean(0))
        wvars.append(block.var(0, ddof=1))
        th.append(theta[m][0])
    means = np.asarray(means)
    wvars = np.asarray(wvars)
    th = np.asarray(th)

    within = np.sqrt(wvars.mean(0))
    between = means.std(0)
    snr = between / np.maximum(within, 1e-12)
    arm.update(
        sim_means=means,
        sim_theta=th,
        within=within,
        between=between,
        snr=snr,
        sim_ids_used=uniq[: len(means)],
    )
    return arm


def param_snr(arm, n_params):
    T = arm["sim_means"]
    TH = arm["sim_theta"][:, :n_params]
    Ts = (T - T.mean(0)) / (T.std(0) + 1e-12)
    out = []
    for j in range(n_params):
        y = TH[:, j] - TH[:, j].mean()
        w, *_ = np.linalg.lstsq(Ts, y, rcond=None)
        d = Ts @ w
        signal = d.std()
        w_unit = w / (np.linalg.norm(w) + 1e-12)
        smear = np.sqrt(np.sum((arm["within"] / (T.std(0) + 1e-12) * w_unit) ** 2))
        out.append(signal / max(smear, 1e-12))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="label=summaries_dir (2 or more)")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--out", default="latent_noise")
    ap.add_argument("--max-sims", type=int, default=1500)
    ap.add_argument("--n-smear", type=int, default=6)
    ap.add_argument("--smear-overlay", action="store_true")
    ap.add_argument("--smear-alpha", type=float, default=0.30)
    ap.add_argument("--center-islands", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--corner", action="store_true")
    ap.add_argument("--color-by", default=None)
    ap.add_argument("--grid-param", default=None, metavar="NAME")
    args = ap.parse_args()

    arms = [decompose(load_arm(s), args.max_sims) for s in args.arms]
    t_dim = arms[0]["t"].shape[1]
    n_params = arms[0]["theta"].shape[1]
    names = args.names or [f"t{j}" for j in range(n_params)]
    cmap = plt.get_cmap("tab10")

    # ---------- 1. within/between/SNR per latent dim ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    w = 0.8 / len(arms)
    xs = np.arange(t_dim)
    for i, a in enumerate(arms):
        off = (i - (len(arms) - 1) / 2) * w
        lab = f"{a['label']} (ns={a['noise_scale']:g})"
        axes[0].bar(xs + off, a["within"], w, color=cmap(i), label=lab)
        axes[1].bar(xs + off, a["between"], w, color=cmap(i), label=lab)
        axes[2].bar(xs + off, a["snr"], w, color=cmap(i), label=lab)
    for ax, ttl, yl in zip(
        axes,
        [
            "within-sim spread (noise smear)",
            "between-sim spread (theta signal)",
            "latent SNR = between / within",
        ],
        ["std of t at fixed theta", "std of per-sim mean t", "ratio"],
        strict=False,
    ):
        ax.set_title(ttl)
        ax.set_xlabel("latent dim")
        ax.set_ylabel(yl)
        ax.set_xticks(xs)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
    axes[2].axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    fig.tight_layout()
    fig.savefig(f"{args.out}_snr.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- 2. PCA clouds ----------
    T0 = arms[0]["t"]
    mu, sd = T0.mean(0), T0.std(0) + 1e-12
    X0 = (T0[:200000] - mu) / sd
    U, S, Vt = np.linalg.svd(X0 - X0.mean(0), full_matrices=False)
    P = Vt[:2].T

    fig, axes = plt.subplots(
        len(arms), n_params, figsize=(4.2 * n_params, 3.8 * len(arms)), squeeze=False
    )
    for i, a in enumerate(arms):
        Z = ((a["sim_means"] - mu) / sd) @ P
        for j in range(n_params):
            ax = axes[i][j]
            sc = ax.scatter(
                Z[:, 0], Z[:, 1], c=a["sim_theta"][:, j], s=6, alpha=0.6, cmap="viridis"
            )
            ax.set_title(f"{a['label']} (ns={a['noise_scale']:g}) — {names[j]}", fontsize=9)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2" if j == 0 else "")
            plt.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle("Latent space (per-sim means), shared PCA basis, coloured by parameter", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{args.out}_pca.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- 3. noise-smear clouds for individual sims ----------
    rng = np.random.default_rng(args.seed)
    shared = set(arms[0]["sim_ids_used"])
    for a in arms[1:]:
        shared &= set(a["sim_ids_used"])
    pick = rng.choice(sorted(shared), min(args.n_smear, len(shared)), replace=False)

    if args.smear_overlay:
        arm_colors = diverging_arm_colors(len(arms))
        ncol = min(len(pick), 5)
        nrow = int(np.ceil(len(pick) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
        for idx, s in enumerate(pick):
            ax = axes[idx // ncol][idx % ncol]
            for i, a in enumerate(arms):
                blk = a["t"][a["sims"] == s]
                Z = ((blk - mu) / sd) @ P
                if args.center_islands:
                    Z = Z - Z.mean(0)
                ax.scatter(
                    Z[:, 0],
                    Z[:, 1],
                    s=5,
                    alpha=args.smear_alpha,
                    color=arm_colors[i],
                    edgecolors="none",
                    zorder=2 + i,
                )
                if not args.center_islands:
                    ax.scatter(
                        Z[:, 0].mean(),
                        Z[:, 1].mean(),
                        s=110,
                        marker="*",
                        edgecolor="k",
                        linewidth=0.6,
                        color=arm_colors[i],
                        zorder=10 + i,
                    )
            if args.center_islands:
                ax.axhline(0, color="0.8", lw=0.5, zorder=0)
                ax.axvline(0, color="0.8", lw=0.5, zorder=0)
            th = (
                arms[0]["sim_theta"][list(arms[0]["sim_ids_used"]).index(s)]
                if s in arms[0]["sim_ids_used"]
                else None
            )
            ttl = f"sim {s}"
            if th is not None:
                ttl += "\n" + ", ".join(f"{names[j]}={th[j]:.2f}" for j in range(n_params))
            ax.set_title(ttl, fontsize=7.5)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(alpha=0.25)
        for k in range(len(pick), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=arm_colors[i],
                label=f"{a['label']} (ns={a['noise_scale']:g})",
            )
            for i, a in enumerate(arms)
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(arms),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.suptitle(
            "Noise smear at fixed theta (stars = per-sim means)", y=1.05 if nrow == 1 else 1.02
        )
    else:
        fig, axes = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 5.0), squeeze=False)
        for i, a in enumerate(arms):
            ax = axes[0][i]
            for k, s in enumerate(pick):
                blk = a["t"][a["sims"] == s]
                Z = ((blk - mu) / sd) @ P
                ax.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.25, color=cmap(k % 10))
                ax.scatter(
                    Z[:, 0].mean(),
                    Z[:, 1].mean(),
                    s=90,
                    marker="*",
                    edgecolor="k",
                    color=cmap(k % 10),
                    zorder=5,
                )
            ax.set_title(
                f"{a['label']} (ns={a['noise_scale']:g})\n{len(pick)} sims, all noise reps"
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(alpha=0.25)
        fig.suptitle("Noise smear at fixed theta (stars = per-sim means)", y=1.0)

    fig.tight_layout()
    fig.savefig(f"{args.out}_smear.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- 3b. raw-t corner plots ----------
    if args.corner:
        cb_idx = names.index(args.color_by) if args.color_by in names else n_params - 1
        stds = np.array([a["sim_means"].std(0) for a in arms])
        if len(arms) > 1:
            ratio = stds.max(0) / np.maximum(stds.min(0), 1e-12)
            print("\n[corner] per-dim latent std by arm:")
            for i, a in enumerate(arms):
                print(f"  {a['label']:16s} " + " ".join(f"{v:7.4f}" for v in stds[i]))
            print(f"  {'max/min ratio':16s} " + " ".join(f"{v:7.2f}" for v in ratio))

        pairs = [(i, j) for j in range(t_dim) for i in range(j)]
        ncols = t_dim - 1
        fig, axes = plt.subplots(
            len(arms) * (t_dim - 1),
            ncols,
            figsize=(2.4 * ncols, 2.4 * len(arms) * (t_dim - 1)),
            squeeze=False,
        )
        for r in range(axes.shape[0]):
            for c in range(ncols):
                axes[r][c].axis("off")

        for ai, a in enumerate(arms):
            T = a["sim_means"]
            col = a["sim_theta"][:, cb_idx]
            r0 = ai * (t_dim - 1)
            for i, j in pairs:
                ax = axes[r0 + (j - 1)][i]
                ax.axis("on")
                sc = ax.scatter(
                    T[:, i], T[:, j], c=col, s=7, alpha=0.7, cmap="viridis", edgecolors="none"
                )
                if i == 0:
                    ax.set_ylabel(f"$t_{j}$", fontsize=9)
                if j == t_dim - 1:
                    ax.set_xlabel(f"$t_{i}$", fontsize=9)
                ax.tick_params(labelsize=6)
            axes[r0][0].set_title(
                f"{a['label']} (ns={a['noise_scale']:g})  — coloured by {names[cb_idx]}",
                fontsize=9,
                loc="left",
            )
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        plt.colorbar(sc, cax=cax, label=f"{names[cb_idx]} (normalised)")
        fig.savefig(f"{args.out}_corner.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # ---------- 3c. GRID BANDING VISUALIZATION (POLISHED JITTER STORY) -------
    # =========================================================================
    gp_name = args.grid_param or ("rHS" if "rHS" in names else None)
    if gp_name is not None and gp_name in names:
        gp = names.index(gp_name)

        def best_dim_for(a, j):
            T = a["sim_means"]
            c = [abs(np.corrcoef(T[:, i], a["sim_theta"][:, j])[0, 1]) for i in range(T.shape[1])]
            return int(np.nanargmax(c)), float(np.nanmax(c))

        cmap_lvl = plt.get_cmap("viridis")
        fig, axes = plt.subplots(2, len(arms), figsize=(5.2 * len(arms), 7.2), squeeze=False)
        sep_ratios = {}

        for i, a in enumerate(arms):
            d, cval = best_dim_for(a, gp)
            y = a["sim_means"][:, d]
            true = a["sim_theta"][:, gp]
            levels = np.unique(np.round(true, 6))
            lvl_col = {lv: cmap_lvl(k / max(len(levels) - 1, 1)) for k, lv in enumerate(levels)}
            noise = float(a["within"][d])

            # Calculate separation metrics
            lvl_means = []
            bins = np.linspace(y.min(), y.max(), 40)

            # --- TOP ROW: 1D Latent Density ---
            axt = axes[0][i]

            # Aesthetic layout adjustments
            axt.spines["top"].set_visible(False)
            axt.spines["right"].set_visible(False)
            axt.grid(True, linestyle="--", alpha=0.3, zorder=0)

            for lv in levels:
                yl = y[np.isclose(true, lv)]
                if len(yl) == 0:
                    continue
                lvl_means.append(yl.mean())
                axt.hist(
                    yl,
                    bins=bins,
                    histtype="stepfilled",
                    alpha=0.45,
                    color=lvl_col[lv],
                    edgecolor="none",
                    zorder=3,
                    label=f"{gp_name}={lv:.2f}",
                )
                axt.hist(yl, bins=bins, histtype="step", lw=1.2, color=lvl_col[lv], zorder=4)

            lvl_means = np.sort(np.asarray(lvl_means))
            gap = float(np.median(np.diff(lvl_means))) if len(lvl_means) > 1 else np.nan
            sep = gap / max(noise, 1e-12)
            sep_ratios[a["label"]] = sep

            # Dynamic Annotation Box
            if sep > 1.8:
                box_color = "#FFE6E6"  # Light red for bad/quantized status
                border_color = "#D9534F"
                text_label = f"⚠️ QUANTIZED SPACE\nSeparation Ratio: {sep:.1f}\n(Empty voids exist between peaks)"
            else:
                box_color = "#E6F4EA"  # Light green for healthy/smooth status
                border_color = "#5CB85C"
                text_label = f"✅ CONTINUOUS MANIFOLD\nSeparation Ratio: {sep:.1f}\n(No gaps; peaks overlap smoothly)"

            axt.text(
                0.05,
                0.92,
                text_label,
                transform=axt.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.5", facecolor=box_color, edgecolor=border_color, alpha=0.9
                ),
                zorder=10,
            )

            axt.set_title(
                f"{a['label']} (ns={a['noise_scale']:g})\n"
                f"Latent $t_{{{d}}}$  (|corr|={cval:.2f} with {gp_name})",
                fontsize=10,
                fontweight="bold",
                pad=10,
            )
            axt.set_xlabel(f"Latent coordinate $t_{{{d}}}$", fontsize=9)
            axt.set_ylabel("Simulation Count", fontsize=9)
            if i == len(arms) - 1:
                axt.legend(
                    fontsize=7, loc="upper right", frameon=True, facecolor="white", framealpha=0.8
                )

            # --- BOTTOM ROW: Latent Coordinate vs. True Parameter ---
            axb = axes[1][i]
            axb.spines["top"].set_visible(False)
            axb.spines["right"].set_visible(False)
            axb.grid(True, linestyle="--", alpha=0.3, zorder=0)

            # Subtle jitter on horizontal axis purely for visualization clarity
            jitterx = (
                np.random.default_rng(0).uniform(-1, 1, len(true))
                * 0.01
                * (levels.max() - levels.min() + 1e-9)
            )

            axb.scatter(
                true + jitterx,
                y,
                s=12,
                alpha=0.4,
                c=[lvl_col[lv] for lv in np.round(true, 6)],
                edgecolors="none",
                zorder=3,
            )

            # Overlay clean means and explicit shaded error bands representing the actual noise smear
            for lv, m in zip(
                levels, [y[np.isclose(true, lv)].mean() for lv in levels], strict=False
            ):
                # Solid line showing the center mean
                axb.hlines(m, lv - 0.04, lv + 0.04, color="black", lw=1.5, zorder=5)
                # Shaded vertical band representing +/- 1sigma noise smear width
                axb.fill_between(
                    [lv - 0.04, lv + 0.04],
                    m - noise,
                    m + noise,
                    color=lvl_col[lv],
                    alpha=0.2,
                    edgecolor=lvl_col[lv],
                    linewidth=0.8,
                    linestyle="--",
                    zorder=2,
                )

            axb.set_title(f"Grid Gap vs. Noise Smear (σ_smear = {noise:.3f})", fontsize=9.5)
            axb.set_xlabel(f"True Physical {gp_name} Grid", fontsize=9)
            axb.set_ylabel(f"Latent coordinate $t_{{{d}}}$", fontsize=9)

        fig.suptitle(
            f"The Jitter Dequantization Effect on '{gp_name}' in Latent Space",
            y=1.01,
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(f"{args.out}_gridband.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(
            f"\n[gridband] latent grid-separation ratio for '{gp_name}' "
            f"(gap between grid levels / noise smear; higher = more quantized):"
        )
        for k, v in sep_ratios.items():
            print(f"  {k:16s} {v:6.1f}")

    # ---------- 4. |corr(t_i, theta_j)| ----------
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 3.8), squeeze=False)
    for i, a in enumerate(arms):
        C = np.zeros((t_dim, n_params))
        for ti in range(t_dim):
            for pj in range(n_params):
                C[ti, pj] = abs(np.corrcoef(a["sim_means"][:, ti], a["sim_theta"][:, pj])[0, 1])
        ax = axes[0][i]
        im = ax.imshow(C, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set_xticks(range(n_params))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(t_dim))
        ax.set_yticklabels([f"t{k}" for k in range(t_dim)])
        ax.set_title(f"{a['label']} (ns={a['noise_scale']:g})")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("|corr(latent dim, parameter)|", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{args.out}_corr.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- print console table ----------
    print("\nlatent SNR per PARAMETER (theta-driven displacement / noise smear):")
    hdr = "  " + " ".join(f"{n:>8}" for n in names)
    print(f"  {'arm':22s}" + hdr)
    snrs = {}
    for a in arms:
        p = param_snr(a, n_params)
        snrs[a["label"]] = p
        label_str = f"{a['label']} (ns={a['noise_scale']:g})"
        print(f"  {label_str:22s}  " + " ".join(f"{v:8.2f}" for v in p))

    if len(arms) == 2:
        a0, a1 = arms[0]["label"], arms[1]["label"]
        ratio = snrs[a1] / np.maximum(snrs[a0], 1e-12)
        print(f"\n  ratio {a1}/{a0}:      " + " ".join(f"{v:8.2f}" for v in ratio))

    extra = ", _corner.png" if args.corner else ""
    if gp_name is not None and gp_name in names:
        extra += ", _gridband.png"
    print(f"\nwrote {args.out}_snr.png, _pca.png, _smear.png, _corr.png{extra}")


if __name__ == "__main__":
    main()
