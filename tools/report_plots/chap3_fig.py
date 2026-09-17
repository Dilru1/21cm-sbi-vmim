#!/usr/bin/env python3
"""
Generate all Chapter 3 ("Issue and the data") figures, publication quality.

Figures produced (names match the \label{fig:...} in 3_issue_and_data.tex):

  voxel_pdf.pdf         per-simulation voxel intensity PDFs, one panel per z,
                        coloured by f_X                      -> fig:voxel_pdf
  prior_design.pdf      corner plot of the 5-parameter design, gridded values
                        and the censored (tau, Mmin) corner  -> fig:prior_design
  moments_vs_theta.pdf  std / skewness of each cube vs each parameter
                                                             -> fig:moments_vs_theta
  cube_slices.pdf       clean vs noisy mid-slices at the 3 redshifts
                        (+ snr_table.tex)                    -> fig:cube_slices
  skew_kurt.pdf         per-sim skewness vs excess kurtosis  -> fig:skew_kurt
  corr_structure.pdf    correlation matrix of spherically-averaged power
                        spectrum bins across redshifts       -> fig:corr_structure

Usage on the cluster (real data; paths default to the ones in your configs):

  python chapter3_figs.py \
      --config /path/to/configs/vmim_configs/arm_cnn_vmim.yaml \
      --out figures --max-sims 1200

or fully explicit:

  python chapter3_figs.py \
      --clean "/data/ddehiwalage-don/data/dtb_data/clean_cubes_z={z}.dat" \
      --noise "/data/ddehiwalage-don/data/dtb_data/noise_cubes_100h_z={z}.dat" \
      --params /data/ddehiwalage-don/data/astro_params_masked_from_original.npy \
      --sim-ids /data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy \
      --out figures


  python chapter3_figs.py --demo --out demo_out

Cubes are read through np.memmap, so only the sub-sampled simulations are
ever pulled into RAM; safe to run on a login node.
"""


from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (apply_style, save, Z_COLORS, Z_LABELS, REDSHIFTS,
                   PARAM_TEX, PARAM_LOG, TEXTWIDTH_IN, COLWIDTH_IN,
                   panel_label)

N = 32                      # cube side
VOX = N ** 3
NB_SIMU_TOTAL = 9827        # matches tools/build_masked_params.py


# ============================================================ data loading
def load_real(clean_tpl, noise_tpl, params_path, simids_path,
              max_sims, noise_scale, rng):
    """Memmap the .dat cube files and load only a random subsample of sims.

    Returns dict with: params (n,5) raw, cubes list[3] of (n,32,32,32) clean,
    noise list[3] of (m,32,32,32), sub-sampling indices.
    """
    params = np.load(params_path).astype(np.float64)        # raw units
    sim_ids = np.load(simids_path).astype(int)
    assert len(params) == len(sim_ids)

    n_avail = len(sim_ids)
    if max_sims and max_sims < n_avail:
        keep = np.sort(rng.choice(n_avail, size=max_sims, replace=False))
    else:
        keep = np.arange(n_avail)
    params = params[keep]
    sel_ids = sim_ids[keep]

    cubes, noise = [], []
    for z in REDSHIFTS:
        cpath = clean_tpl.format(z=z)
        npath = noise_tpl.format(z=z)
        cm = np.memmap(cpath, dtype=np.float64, mode="r")
        cm = cm.reshape(-1, N, N, N)[:NB_SIMU_TOTAL]
        cubes.append(np.asarray(cm[sel_ids], dtype=np.float32))
        nm = np.memmap(npath, dtype=np.float64, mode="r").reshape(-1, N, N, N)
        m = min(200, nm.shape[0])                 # 200 noise cubes is plenty
        noise.append(np.asarray(nm[:m], dtype=np.float32) * noise_scale)

    return {"params": params, "cubes": cubes, "noise": noise,
            "params_full": np.load(params_path).astype(np.float64)}


def make_demo(n_sims, rng):
    """Synthetic dataset mimicking the LoReLi II design & 21cm dTb statistics.

    Same shapes/units conventions as the real loader so every figure function
    is agnostic to the data source. Purely for previewing figure style.
    """
    # --- gridded prior design ------------------------------------------------
    fX_grid = np.geomspace(0.1, 10.0, 10)
    tau_grid = np.geomspace(738.9, 10504.7, 10)              # Myr, code units
    th2_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    mmin_grid = np.linspace(8.0, 9.6, 8)
    fesc_grid = np.array([0.05, 0.2, 0.5])

    rows = []
    while len(rows) < n_sims:
        p = [rng.choice(fX_grid), rng.choice(tau_grid), rng.choice(th2_grid),
             rng.choice(mmin_grid), rng.choice(fesc_grid)]
        # censor the (tau, Mmin) corner: fast star formation impossible for
        # heavy Mmin (mimics the observed-LF compatibility cut)
        if p[1] > 4000.0 and p[3] > 8.9:
            continue
        rows.append(p)
    params = np.array(rows)

    # --- mock cubes: lognormal-ish field, statistics driven by (fX, Mmin, z) --
    kx = np.fft.fftfreq(N)[:, None, None]
    ky = np.fft.fftfreq(N)[None, :, None]
    kz = np.fft.fftfreq(N)[None, None, :]
    kk = np.sqrt(kx**2 + ky**2 + kz**2); kk[0, 0, 0] = 1e-6

    cubes, noise = [], []
    for iz, z in enumerate(REDSHIFTS):
        sig = np.empty((n_sims, N, N, N), np.float32)
        for i, (fx, tau, th2, mm, fe) in enumerate(params):
            amp = 8.0 * (fx ** 0.4) * (1 + 0.12 * iz) * (mm / 8.8) ** -1.5
            a = 0.55 + 0.3 * np.log10(fx) + 0.10 * iz        # skew driver
            pk = kk ** (-2.4 + 0.15 * iz)
            pk[0, 0, 0] = 0.0                                # kill DC mode
            g = np.fft.ifftn(np.fft.fftn(rng.standard_normal((N, N, N)))
                             * np.sqrt(pk)).real
            g = (g - g.mean()) / g.std()
            f = np.exp(np.clip(a * g, -20, 20))
            f = f / f.mean() - 1.0                           # lognormal, mean 0
            sig[i] = (amp * f - 0.4 * amp * th2 * g).astype(np.float32)
        cubes.append(sig)
        nz = np.empty((60, N, N, N), np.float32)
        s_n = 4.0 * (1 + 0.9 * iz)                           # noise grows with z
        pk_n = kk ** 0.5
        pk_n[0, 0, 0] = 0.0
        for j in range(60):
            g = np.fft.ifftn(np.fft.fftn(rng.standard_normal((N, N, N)))
                             * np.sqrt(pk_n)).real
            nz[j] = (s_n * (g - g.mean()) / g.std()).astype(np.float32)
        noise.append(nz)

    return {"params": params, "cubes": cubes, "noise": noise,
            "params_full": params}


# ====================================================== summary statistics
def per_sim_moments(cubes):
    """(mean, std, skew, excess kurtosis) per sim per z, vectorised."""
    out = {}
    for iz, sig in enumerate(cubes):
        x = sig.reshape(len(sig), -1).astype(np.float64)
        mu = x.mean(1, keepdims=True)
        d = x - mu
        m2 = (d**2).mean(1); m3 = (d**3).mean(1); m4 = (d**4).mean(1)
        out[iz] = dict(mean=mu[:, 0], std=np.sqrt(m2),
                       skew=m3 / m2**1.5, kurt=m4 / m2**2 - 3.0)
    return out


def power_spectrum_bins(cube, n_bins=12):
    """Spherically averaged dimensionless P(k) of one cube, log-spaced bins."""
    fk = np.fft.fftn(cube)
    pk3 = (fk * np.conj(fk)).real / VOX
    kx = np.fft.fftfreq(N)[:, None, None]
    ky = np.fft.fftfreq(N)[None, :, None]
    kz = np.fft.fftfreq(N)[None, None, :]
    kk = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    pk3 = pk3.ravel()
    edges = np.geomspace(1.0 / N, 0.5, n_bins + 1)
    idx = np.digitize(kk, edges) - 1
    ps = np.array([pk3[idx == b].mean() if np.any(idx == b) else np.nan
                   for b in range(n_bins)])
    return ps


# ================================================================= figures
def fig_voxel_pdf(d, outdir, max_lines=200, rng=None):
    """Voxel PDF of each simulation: 3 rows (redshifts) x 5 cols (parameters).
    Each panel shows all simulations coloured by one parameter at one redshift.
    Transposed layout saves vertical space while keeping all 15 panels."""

    from cmastro import cmaps
    cmap = cmaps["cma:unph"]

    params, cubes = d["params"], d["cubes"]
    n = len(params)
    sel = (np.sort(rng.choice(n, min(max_lines, n), replace=False))
           if n > max_lines else np.arange(n))

    n_params = params.shape[1]
    n_z = len(cubes)

    fig, axes = plt.subplots(n_z, n_params,
                             figsize=(TEXTWIDTH_IN, n_z * 1.8),
                             sharey=True, constrained_layout=True)

    for col in range(n_params):
        raw = params[sel, col]
        cval = np.log10(raw) if PARAM_LOG[col] else raw
        norm = mpl.colors.Normalize(cval.min(), cval.max())
        order = np.argsort(cval)

        for row in range(n_z):
            ax = axes[row, col]
            x = cubes[row][sel].reshape(len(sel), -1)
            lo, hi = np.percentile(x, [0.02, 99.98])
            bins = np.linspace(lo, hi, 90)
            ctr = 0.5 * (bins[1:] + bins[:-1])
            for j in order:
                h, _ = np.histogram(x[j], bins=bins, density=True)
                ax.plot(ctr, h, lw=0.7, alpha=0.8,
                        color=cmap(norm(cval[j])))
            ax.set_yscale("log")

            if row == 0:
                label = (rf"$\log_{{10}}$ {PARAM_TEX[col]}"
                         if PARAM_LOG[col] else PARAM_TEX[col])
                ax.set_title(label, fontsize=8)
            if row == n_z - 1:
                ax.set_xlabel(r"$\delta T_b$ [mK]", fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(Z_LABELS[row], fontsize=8)

        # one colorbar per column, below the bottom row
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes[:, col], pad=0.015, aspect=20,
                          shrink=0.85, location="bottom")
        cb.ax.tick_params(labelsize=6)

    save(fig, outdir, "voxel_pdf_all_params")

def fig_prior_design(d, outdir):
    p = d["params_full"]
    k = p.shape[1]
    fig, axes = plt.subplots(k, k, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.95))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False); continue
            if i == j:
                x = np.log10(p[:, i]) if PARAM_LOG[i] else p[:, i]
                ax.hist(x, bins=40, color="#4477AA", alpha=0.85)
                ax.set_yticks([])
            else:
                x = np.log10(p[:, j]) if PARAM_LOG[j] else p[:, j]
                y = np.log10(p[:, i]) if PARAM_LOG[i] else p[:, i]
                ax.scatter(x, y, s=1.2, alpha=0.15, color="#4477AA",
                           rasterized=True, linewidths=0)
                # highlight the censored (tau, Mmin) corner: tau=col1, Mmin=col3
                if {i, j} == {1, 3}:
                    xt = np.log10(p[:, 1]); ym = p[:, 3]
                    # empty box = region above both 75th percentiles w/ no pts
                    bx = (xt.min(), xt.max()); by = (ym.min(), ym.max())
                    ax.add_patch(Rectangle(
                        (np.percentile(xt, 60), np.percentile(ym, 60)),
                        bx[1] - np.percentile(xt, 60),
                        by[1] - np.percentile(ym, 60),
                        fill=False, ls="--", lw=0.9, ec="#CC3311"))
                    ax.text(0.97, 0.95, "censored", transform=ax.transAxes,
                            ha="right", va="top", fontsize=7, color="#CC3311")
            def _lab(k_):
                return (rf"$\log_{{10}}$ {PARAM_TEX[k_]}"
                        if PARAM_LOG[k_] else PARAM_TEX[k_])
            lab_i, lab_j = _lab(i), _lab(j)
            if i == k - 1:
                ax.set_xlabel(lab_j, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(lab_i, fontsize=8)
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle("Prior design: gridded sampling and censoring", y=0.93,
                 fontsize=10)
    save(fig, outdir, "prior_design")


def fig_moments_vs_theta(d, outdir, moms):
    from scipy import stats as sps
    p = d["params"]
    rows = [("std", r"$\sigma(\delta T_b)$ [mK]"),
            ("skew", r"skewness")]
    fig, axes = plt.subplots(2, 5, figsize=(TEXTWIDTH_IN, 3.4),
                             constrained_layout=True)
    for r, (key, ylab) in enumerate(rows):
        for c in range(5):
            ax = axes[r, c]
            x = p[:, c]
            for iz in range(3):
                ax.scatter(x, moms[iz][key], s=2.5, alpha=0.25,
                           color=Z_COLORS[iz], rasterized=True, linewidths=0)
            rho = sps.spearmanr(x, moms[0][key]).statistic
            ax.text(0.96, 0.94, rf"$\rho_S={rho:+.2f}$",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
            if PARAM_LOG[c]:
                ax.set_xscale("log")
            if r == 1:
                ax.set_xlabel(PARAM_TEX[c])
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_yticklabels([])
        ymin = min(axes[r, c].get_ylim()[0] for c in range(5))
        ymax = max(axes[r, c].get_ylim()[1] for c in range(5))
        for c in range(5):
            axes[r, c].set_ylim(ymin, ymax)
    handles = [Line2D([], [], marker="o", ls="", color=Z_COLORS[iz],
                      label=Z_LABELS[iz], markersize=4) for iz in range(3)]
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.07))
    save(fig, outdir, "moments_vs_theta")


def fig_cube_slices(d, outdir, sim_index=0, noise_scale=1.0):
    from cmastro import cmaps
    print(cmaps.keys())
    #dict_keys(['cma:hesperia', 'cma:hesperia_r', 'cma:lacerta', 
    #'cma:lacerta_r', 'cma:laguna', 'cma:laguna_r', 'cma:emph', 
    #'cma:emph_r', 'cma:unph', 'cma:unph_r'])
    

    cubes, noise = d["cubes"], d["noise"]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 4.1),
                             constrained_layout=True)
    snr = []
    for iz in range(3):
        clean = cubes[iz][sim_index]
        noisy = clean + noise_scale * noise[iz][0]
        sl_c, sl_n = clean[N // 2], noisy[N // 2]
        vmin, vmax = np.percentile(np.stack([sl_c, sl_n]), [0.5, 99.5])
        for r, (sl, ax) in enumerate(zip([sl_c, sl_n], axes[:, iz])):
            im = ax.imshow(sl, cmap="cma:emph_r" , vmin=vmin, vmax=vmax,
                           origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, iz].set_title(Z_LABELS[iz])
        s = clean.std() / (noise_scale * noise[iz].std())
        snr.append(s)
        axes[1, iz].text(0.04, 0.05, rf"SNR$\,\approx {s:.2f}$",
                         transform=axes[1, iz].transAxes, color="w",
                         fontsize=8, va="bottom",
                         bbox=dict(fc="k", alpha=0.45, pad=1.5, lw=0))
        cb = fig.colorbar(im, ax=axes[:, iz], location="bottom",
                          shrink=0.9, pad=0.02, aspect=25)
        cb.set_label(r"$\delta T_b$ [mK]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("clean signal", fontsize=9)
    axes[1, 0].set_ylabel("signal + noise (100 h)", fontsize=9)
    save(fig, outdir, "cube_slices")

    # LaTeX SNR table for the data-quality paragraph / caption
    with open(os.path.join(outdir, "snr_table.tex"), "w") as f:
        f.write("% auto-generated by chapter3_figs.py\n"
                "\\begin{tabular}{lccc}\n\\toprule\n"
                " & " + " & ".join(Z_LABELS[i] for i in range(3)) +
                " \\\\\n\\midrule\n"
                "voxel SNR $\\sigma_{\\rm sig}/\\sigma_{\\rm noise}$ & " +
                " & ".join(f"{s:.2f}" for s in snr) +
                " \\\\\n\\bottomrule\n\\end{tabular}\n")
    print("  wrote snr_table.tex")

from scipy.stats import gaussian_kde

def fig_skew_kurt(d, outdir, moms):
    # squarer aspect than a full-width figure: better fit in a wrapfigure column
    fig, ax = plt.subplots(figsize=(3.4, 3.15), constrained_layout=True)

    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 60.0  # tune to your data's kurtosis range
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for iz in range(3):
        x, y = np.asarray(moms[iz]["skew"]), np.asarray(moms[iz]["kurt"])

        # sparse background scatter to show tails/outliers the KDE would hide
        ax.scatter(x, y, s=5.5, alpha=0.50, color=Z_COLORS[iz],
                   rasterized=True, linewidths=0, zorder=1)

        # density contours carry the "shape" of each redshift's population
        kde = gaussian_kde(np.vstack([x, y]))
        zz = np.reshape(kde(grid), xx.shape)
        levels = np.linspace(zz.max() * 0.15, zz.max() * 0.85, 3)
        #ax.contour(xx, yy, zz, levels=levels, colors=[Z_COLORS[iz]],
        #           linewidths=1.1, zorder=3)
        # dummy handle for a clean legend entry (contour sets don't legend well)
        ax.plot([], [], color=Z_COLORS[iz], lw=1.6, label=Z_LABELS[iz])

    ax.scatter([0], [0], marker="s", s=50, color="#CC3311",
               edgecolors="black", linewidths=0.5, zorder=5, label="Gaussian")

    ax.axhline(0, lw=0.6, color="0.65", zorder=0)
    ax.axvline(0, lw=0.6, color="0.65", zorder=0)

    ax.set_xlabel("skewness")
    ax.set_ylabel("excess kurtosis")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlim(xmin, xmax)

    ax.tick_params(direction="in", which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.legend(fontsize=7.5, markerscale=1.4, handletextpad=0.4,
              labelspacing=0.3, borderpad=0.4, frameon=True,
              framealpha=0.85, loc="upper left")

    save(fig, outdir, "skew_kurt")


def fig_corr_structure(d, outdir, n_bins=12, max_sims=400, rng=None):
    cubes = d["cubes"]
    n = len(cubes[0])
    sel = (np.sort(rng.choice(n, min(max_sims, n), replace=False))
           if n > max_sims else np.arange(n))
    feats = []
    for iz in range(3):
        ps = np.array([power_spectrum_bins(cubes[iz][i], n_bins) for i in sel])
        feats.append(np.log10(np.clip(ps, 1e-12, None)))
    X = np.concatenate(feats, axis=1)
    X = X[:, ~np.isnan(X).any(0)]
    C = np.corrcoef(X, rowvar=False)
    m = C.shape[0]

    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, COLWIDTH_IN * 0.92),
                           constrained_layout=True)
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    per = m // 3
    for b in (per, 2 * per):
        ax.axhline(b - 0.5, color="k", lw=0.8)
        ax.axvline(b - 0.5, color="k", lw=0.8)
    ticks = [per / 2 - 0.5, 1.5 * per - 0.5, 2.5 * per - 0.5]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([Z_LABELS[i] for i in range(3)])
    ax.set_yticklabels([Z_LABELS[i] for i in range(3)], rotation=90,
                       va="center")
    ax.tick_params(which="minor", length=0)
    ax.set_title(r"correlation of $\log P(k)$ bins "
                 rf"(${per}$ $k$-bins $\times$ 3 redshifts)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Pearson correlation")
    save(fig, outdir, "corr_structure")


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic mock data (no cluster files needed)")
    ap.add_argument("--config", help="pipeline yaml; data paths read from it")
    ap.add_argument("--clean", default="/data/ddehiwalage-don/data/dtb_data/clean_cubes_z={z}.dat")
    ap.add_argument("--noise", default="/data/ddehiwalage-don/data/dtb_data/noise_cubes_100h_z={z}.dat")
    ap.add_argument("--params", default="/data/ddehiwalage-don/data/astro_params_masked_from_original.npy")
    ap.add_argument("--sim-ids", default="/data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--max-sims", type=int, default=1200,
                    help="sub-sample size for cube-level statistics")
    ap.add_argument("--out", default="tools/report_plots/figures")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--figs", nargs="*", default=None,
                    help="subset: voxel_pdf prior_design moments cube_slices "
                         "skew_kurt corr")
    args = ap.parse_args()

    apply_style()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.config:
        import yaml
        with open(args.config) as f:
            dcfg = yaml.safe_load(f)["data"]
        args.clean = dcfg["s_paths"][0].replace("8.18", "{z}")
        args.noise = dcfg["n_paths"][0].replace("8.18", "{z}")
        args.params, args.sim_ids = dcfg["params_path"], dcfg["sim_ids_path"]

    if args.demo:
        print("[demo] generating synthetic mock dataset ...")
        d = make_demo(min(args.max_sims, 400), rng)
    else:
        print("[real] memmapping cluster data ...")
        d = load_real(args.clean, args.noise, args.params, args.sim_ids,
                      args.max_sims, args.noise_scale, rng)

    print("computing per-sim moments ...")
    moms = per_sim_moments(d["cubes"])

    want = set(args.figs) if args.figs else None
    def go(name):
        return want is None or name in want

    if go("voxel_pdf"):
        print("fig 1: voxel_pdf");        fig_voxel_pdf(d, args.out, rng=rng)
    if go("prior_design"):
        print("fig 2: prior_design");     fig_prior_design(d, args.out)
    if go("moments"):
        print("fig 3: moments_vs_theta"); fig_moments_vs_theta(d, args.out, moms)
    if go("cube_slices"):
        print("fig 4: cube_slices");      fig_cube_slices(d, args.out,
                                              noise_scale=(1.0 if args.demo
                                                           else args.noise_scale))
    if go("skew_kurt"):
        print("fig 5: skew_kurt");        fig_skew_kurt(d, args.out, moms)
    if go("corr"):
        print("fig 6: corr_structure");   fig_corr_structure(d, args.out, rng=rng)

    print(f"\nAll done -> {args.out}/")


if __name__ == "__main__":
    main()