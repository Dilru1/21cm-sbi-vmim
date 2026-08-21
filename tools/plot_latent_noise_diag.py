#!/usr/bin/env python3
"""Compare exported latent summaries t across noise levels (or any two arms).

    python tools/latent_noise_compare.py \
        n1=/gscratch/.../cnn_mse_up/exp1/summaries \
        n2=/gscratch/.../cnn_mse_up/exp2/summaries \
        --names Fx tau rHS Mmin --out latent_noise

Reads theta.npy, t.npy, original_sim_ids.npy (+ export_noise_scale.npy for the
label) from each summaries/ dir. Because t.npy holds ~1000 noise realisations
PER simulation, the latent scatter decomposes into

    within-sim  spread : same theta, different noise draw  -> the SMEAR
    between-sim spread : across theta                      -> the SIGNAL

and their ratio is a latent signal-to-noise that predicts posterior width
before any MCMC is run. That decomposition is the point of this script; a bare
"are the clouds bigger" scatter cannot separate the two.

Figures written (prefix = --out):
  _snr.png      per-latent-dim within/between spread + latent SNR, both arms
  _pca.png      PCA of t (fitted on arm 1, applied to both), coloured by theta
  _smear.png    noise-smear clouds for a handful of individual simulations
  _corr.png     |corr(t_i, theta_j)| heatmaps -- which dims encode which param
Prints a table of latent SNR and a mutual-information-style proxy per parameter.
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    """Split latent scatter into within-sim (noise) and between-sim (signal).

    within_j  = sqrt( mean_over_sims Var_over_noise( t_j ) )
    between_j = std_over_sims( mean_over_noise( t_j ) )
    snr_j     = between_j / within_j
    Also returns per-sim latent means (the 'clean' latent position) and the
    per-sim theta, for the PCA/correlation panels.
    """
    rng = rng or np.random.default_rng(0)
    t, sims, theta = arm["t"], arm["sims"], arm["theta"]
    uniq = np.unique(sims)
    if len(uniq) > max_sims:  # subsample sims, keep ALL noise reps
        uniq = rng.choice(uniq, max_sims, replace=False)

    means, wvars, th = [], [], []
    for s in uniq:
        m = sims == s
        block = t[m]  # (n_noise, t_dim)
        if block.shape[0] < 2:
            continue
        means.append(block.mean(0))
        wvars.append(block.var(0, ddof=1))  # noise-induced variance at fixed theta
        th.append(theta[m][0])
    means = np.asarray(means)
    wvars = np.asarray(wvars)
    th = np.asarray(th)

    within = np.sqrt(wvars.mean(0))  # (t_dim,)
    between = means.std(0)  # (t_dim,)
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
    """Latent SNR *per parameter*: how well theta_j is resolvable in t.

    Regress the per-sim latent means on theta_j (linear, on standardized t),
    then compare the theta-driven latent displacement to the noise smear along
    the same direction. Crude but interpretable, and it needs no extra model.
    """
    T = arm["sim_means"]
    TH = arm["sim_theta"][:, :n_params]
    Ts = (T - T.mean(0)) / (T.std(0) + 1e-12)
    out = []
    for j in range(n_params):
        y = TH[:, j] - TH[:, j].mean()
        w, *_ = np.linalg.lstsq(Ts, y, rcond=None)  # theta_j ~ w . t
        d = Ts @ w  # latent projection of theta_j
        signal = d.std()
        # noise smear projected on the SAME direction, in standardized units
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
    ap.add_argument("--n-smear", type=int, default=6, help="sims to draw noise clouds for")
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

    # ---------- 2. PCA clouds (fit on arm 0, apply to all) ----------
    T0 = arms[0]["t"]
    mu, sd = T0.mean(0), T0.std(0) + 1e-12
    X0 = (T0[:200000] - mu) / sd
    U, S, Vt = np.linalg.svd(X0 - X0.mean(0), full_matrices=False)
    P = Vt[:2].T  # shared basis -> comparable panels

    fig, axes = plt.subplots(
        len(arms), n_params, figsize=(4.2 * n_params, 3.8 * len(arms)), squeeze=False
    )
    for i, a in enumerate(arms):
        Z = ((a["sim_means"] - mu) / sd) @ P  # per-sim latent means in PC space
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
    rng = np.random.default_rng(1)
    shared = set(arms[0]["sim_ids_used"])
    for a in arms[1:]:
        shared &= set(a["sim_ids_used"])
    pick = rng.choice(sorted(shared), min(args.n_smear, len(shared)), replace=False)

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
        ax.set_title(f"{a['label']} (ns={a['noise_scale']:g})\n{len(pick)} sims, all noise reps")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Noise smear at fixed theta (stars = per-sim means). Wider clouds -> wider posteriors.",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(f"{args.out}_smear.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

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
    fig.suptitle("|corr(latent dim, parameter)| — which dims encode which parameter", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{args.out}_corr.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- table ----------
    print("\nlatent SNR per PARAMETER (theta-driven displacement / noise smear):")
    hdr = "  " + " ".join(f"{n:>8}" for n in names)
    print(f"  {'arm':22s}" + hdr)
    snrs = {}
    for a in arms:
        p = param_snr(a, n_params)
        snrs[a["label"]] = p
        # Fix: Construct the label string cleanly first to avoid quote collision
        label_str = f"{a['label']} (ns={a['noise_scale']:g})"
        print(f"  {label_str:22s}  " + " ".join(f"{v:8.2f}" for v in p))

    if len(arms) == 2:
        a0, a1 = arms[0]["label"], arms[1]["label"]
        ratio = snrs[a1] / np.maximum(snrs[a0], 1e-12)
        print(f"\n  ratio {a1}/{a0}:      " + " ".join(f"{v:8.2f}" for v in ratio))
        print(
            "  (<1 = information lost at the higher noise level; the parameter whose\n"
            "   ratio falls furthest is the one whose signal was most noise-dominated)"
        )
    print(f"\nwrote {args.out}_snr.png, _pca.png, _smear.png, _corr.png")


if __name__ == "__main__":
    main()
