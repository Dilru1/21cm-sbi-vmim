# _latent.py
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _load_summaries(arm):
    sdir = os.path.join(os.path.dirname(arm["dir"].rstrip("/")), "summaries")
    tp, thp = os.path.join(sdir, "t.npy"), os.path.join(sdir, "theta.npy")
    if not (os.path.exists(tp) and os.path.exists(thp)):
        return None
    t = np.load(tp).astype(np.float64)
    theta = np.load(thp).astype(np.float64)
    sim_p = os.path.join(sdir, "original_sim_ids.npy")
    sims = np.load(sim_p) if os.path.exists(sim_p) else None
    return {"t": t, "theta": theta, "sims": sims, "dir": sdir}

def latent_tt_colored(t, theta, out_path, name, pnames):
    n_t, n_p = t.shape[1], theta.shape[1]
    pairs = [(i, j) for i in range(n_t) for j in range(i + 1, n_t)]
    if not pairs: return
    
    m = min(len(t), 20000)
    idx = np.random.default_rng(0).choice(len(t), m, replace=False)
    ts, ths = t[idx], theta[idx]
    nrow, ncol = len(pairs), n_p
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), squeeze=False)
    
    for ri, (i, j) in enumerate(pairs):
        for ci in range(n_p):
            ax = axes[ri][ci]
            ax.scatter(ts[:, i], ts[:, j], c=ths[:, ci], cmap="Spectral", s=3, alpha=0.5, rasterized=True)
            ax.set_xlabel(rf"$t_{{{i+1}}}$", fontsize=8)
            ax.set_ylabel(rf"$t_{{{j+1}}}$", fontsize=8)
            if ri == 0: ax.set_title(f"colored by {pnames[ci]}", fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
            
    fig.suptitle(f"{name}: latent t-t scatter colored by parameter", y=1.002, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

def latent_persim_clouds(t, sims, out_path, name, nb_noise_guess=1000, n_hi=8):
    n_t = t.shape[1]
    pairs = [(i, j) for i in range(n_t) for j in range(i + 1, n_t)]
    if not pairs: return
    
    if sims is not None:
        uniq = np.unique(sims)
        pick = uniq[np.linspace(0, len(uniq) - 1, min(n_hi, len(uniq))).astype(int)]
        groups = [np.where(sims == s)[0] for s in pick]
        labels = [f"sim {s}" for s in pick]
    else:
        nb = nb_noise_guess
        pick = range(min(n_hi, len(t) // nb))
        groups = [np.arange(k * nb, (k + 1) * nb) for k in pick]
        labels = [f"blk {k}" for k in pick]
        
    cmap = plt.get_cmap("tab10")
    ncol = min(len(pairs), 3)
    nrow = (len(pairs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    
    for pi, (i, j) in enumerate(pairs):
        ax = axes[pi // ncol][pi % ncol]
        ax.scatter(t[::5, i], t[::5, j], color="#ddd", s=2, alpha=0.2, rasterized=True)
        for k, g in enumerate(groups):
            ax.scatter(t[g, i], t[g, j], color=cmap(k % 10), s=10, alpha=0.7,
                       marker="s", edgecolors="white", linewidth=0.2, label=labels[k])
            ax.scatter(t[g, i].mean(), t[g, j].mean(), marker="x", color="k", s=45, zorder=5)
        ax.set_xlabel(rf"$t_{{{i+1}}}$"); ax.set_ylabel(rf"$t_{{{j+1}}}$")
        ax.set_aspect("equal", adjustable="datalim")
        if pi == 0: ax.legend(fontsize=6, loc="best")
        
    for k in range(len(pairs), nrow * ncol): axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{name}: per-sim latent clouds", y=1.002, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

def latent_t_theta_corr(t, theta, out_path, name, pnames):
    n_t, n_p = t.shape[1], theta.shape[1]
    C = np.zeros((n_t, n_p))
    for i in range(n_t):
        for j in range(n_p):
            C[i, j] = abs(np.corrcoef(t[:, i], theta[:, j])[0, 1])
            
    fig, ax = plt.subplots(figsize=(1.3 * n_p + 3, 0.7 * n_t + 2))
    im = ax.imshow(C, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_p)); ax.set_xticklabels(pnames, rotation=30, ha="right")
    ax.set_yticks(range(n_t)); ax.set_yticklabels([rf"$t_{{{i+1}}}$" for i in range(n_t)])
    
    for i in range(n_t):
        for j in range(n_p):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] > 0.5 else "black", fontsize=8)
            
    ax.set_title(f"{name}: |corr(t, theta)|")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

def plot_latent(arm, out_dir, labels):
    s = _load_summaries(arm)
    ldir = os.path.join(out_dir, "latent", re.sub(r"[^A-Za-z0-9]+", "_", arm["name"]))
    if s is None:
        print(f"[latent] {arm['name']}: no summaries/ (arm not exported?) -- skipping")
        return
        
    os.makedirs(ldir, exist_ok=True)
    t, theta, sims = s["t"], s["theta"], s["sims"]
    pnames = [labels[j] for j in range(min(theta.shape[1], len(labels)))]

    stds = t.std(0)
    print(f"[latent] {arm['name']}: t.shape={t.shape}  per-dim std={np.round(stds,4).tolist()}")

    latent_tt_colored(t, theta, os.path.join(ldir, "tt_colored.png"), arm["name"], pnames)
    latent_persim_clouds(t, sims, os.path.join(ldir, "persim_clouds.png"), arm["name"])
    latent_t_theta_corr(t, theta, os.path.join(ldir, "t_theta_corr.png"), arm["name"], pnames)
    print(f"[latent] {arm['name']}: wrote 3 latent figures -> {ldir}")