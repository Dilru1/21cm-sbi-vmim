#!/usr/bin/env python3
r"""Publication-quality compressor training diagnostics -> vector PDF.

ONE FIGURE PER OBJECTIVE. Overlay the arms that share an objective, because
only then are the loss curves on a commensurable scale.

  # MSE noise ladder (aux_only -> no VMIM head -> no sigma row appears)
  python tools/plot_compressor_diag.py \
      "MSE $n_s{=}1$"=/gscratch/.../cnn_mse_up/exp1/nle \
      "MSE $n_s{=}2$"=/gscratch/.../cnn_mse_up/exp2/nle \
      --names Fx tau rHS Mmin --loss-log --out fig_comp_mse.pdf

  # VMIM (sigma row appears automatically; pass the dequant half-widths so the
  # information-theoretic floor is drawn)
  python tools/plot_compressor_diag.py \
      "VMIM seed A"=/gscratch/.../cnn_vmim_up/exp1/nle \
      "VMIM seed B"=/gscratch/.../cnn_vmim_up/exp2/nle \
      --names Fx tau rHS Mmin --dequant 2:0.1000 3:0.0813 \
      --out fig_comp_vmim.pdf

WHAT THE PANELS MEAN  (read this before writing the caption)
------------------------------------------------------------
probe_r2()    fits a RandomForest from the FULL latent vector t to EACH
              parameter theta_p separately. So R^2 is indexed by PARAMETER,
              not by latent dimension. The trainer prints "t0..t3", which is a
              misnomer -- they are theta_0..theta_3. This script labels them
              with --names, which is what your report should say.
probe_sigma() returns the per-parameter conditional sigma of the GMM head
              q(theta_p | t). Same indexing: PARAMETER, not latent dim.

Three reference levels make the sigma panel readable:
  prior sigma  = 1/sqrt(12) = 0.2887   theta ~ U[0,1]; sigma -> here means the
                                       head learned NOTHING about theta_p.
  dequant floor = half/sqrt(3)         uniform jitter on +/-half has this std;
                                       the head CANNOT resolve theta_p below it.
                                       Only for dims listed in `dequant:`.
  sigma_floor                          the head's parametrisation clamp. Sigma
                                       sitting ON it = saturated/overconfident.
NOTE: the floors above are valid only because your configs do NOT set
`margin: true` in the dequant block, so the head sees raw theta in [0,1].
If you ever enable margin (a, b), every level above scales by b.

File schemas handled (both trainer generations):
  loss_history.npy   [epoch, train, val, phase]                      (4 col)
  rf_r2_history.npy  [epoch, R2 x P]                                 (old, 5 col)
                     [epoch, phase, aux_coef, R2 x P]                (new, 7 col)
  sigma_history.npy  [epoch, sigma x P]                              (5 col)
R^2 columns are always taken as the LAST P columns, so old and new arms can be
overlaid in one call.
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

PRIOR_SIGMA = 1.0 / np.sqrt(12.0)  # 0.2887, std of U[0,1]


def diverging_colors(n, cmap="coolwarm", lo=0.10, hi=0.90):
    """Ordered hues from a diverging map: cool -> warm.

    For a NOISE LADDER this is semantically honest -- ns=1 reads cool, ns=2
    reads warm, and the ordering is visible pre-attentively. For unordered arms
    it is just two opposed hues, which is fine but claims nothing. The endpoints
    of coolwarm/RdBu are colourblind-safe and survive greyscale printing; the
    pale middle does not, so we never sample it (lo/hi clip it out).
    """
    cm = plt.get_cmap(cmap)
    if n == 1:
        return [cm(lo)]
    return [cm(x) for x in np.linspace(lo, hi, n)]


def set_pub_style(fs=9.0):
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": fs,
            "axes.titlesize": fs + 0.5,
            "axes.labelsize": fs,
            "xtick.labelsize": fs - 1.5,
            "ytick.labelsize": fs - 1.5,
            "legend.fontsize": fs - 1.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.2,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.dpi": 400,
            # Type-42 = TrueType. Matplotlib defaults to Type-3, which several
            # journals and arXiv's PDF checker reject outright.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_arm(spec, P):
    """spec = 'label=path'; path is the arm's nle/ dir (or a .npy inside it)."""
    if "=" in spec and not os.path.exists(spec):
        label, path = spec.rsplit("=", 1)  # rsplit: labels may contain '='
    else:
        path, label = spec, os.path.basename(os.path.abspath(spec.rstrip("/")))
    nle = path if os.path.isdir(path) else os.path.dirname(path)

    def _load(fname, required=False):
        p = os.path.join(nle, fname)
        if not os.path.exists(p):
            if required:
                sys.exit(f"[FATAL] '{label}': missing {p}")
            print(f"[warn] '{label}': no {fname}", file=sys.stderr)
            return None
        a = np.load(p)
        return a[None, :] if a.ndim == 1 else a

    rf = _load("rf_r2_history.npy", required=True)
    if rf.shape[1] < P + 1:
        sys.exit(
            f"[FATAL] '{label}': rf_r2_history has {rf.shape[1]} cols, "
            f"need >= {P + 1} for {P} parameters"
        )
    sig = _load("sigma_history.npy")
    if sig is not None and sig.shape[1] < P + 1:
        print(f"[warn] '{label}': sigma_history has {sig.shape[1]} cols; ignoring", file=sys.stderr)
        sig = None

    return dict(
        label=label,
        ep=rf[:, 0],
        r2=rf[:, -P:],  # last P cols: shape-agnostic
        phase=(rf[:, 1] if rf.shape[1] > P + 1 else None),
        loss=_load("loss_history.npy"),
        sigma=sig,
    )


def switches(ep, phase):
    """Epochs where the objective/phase flips (warmup -> anneal -> pure VMIM)."""
    if phase is None:
        return []
    return [ep[i + 1] for i in np.where(np.diff(phase) != 0)[0]]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("arms", nargs="+", help="label=path/to/arm/nle")
    ap.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="PARAMETER names in theta order, e.g. Fx tau rHS Mmin",
    )
    ap.add_argument("--out", default="fig_compressor.pdf")
    ap.add_argument(
        "--dequant",
        nargs="*",
        default=[],
        metavar="IDX:HALF",
        help="dequant half-widths, e.g. 2:0.1000 3:0.0813",
    )
    ap.add_argument("--sigma-floor", type=float, default=1e-2)
    ap.add_argument(
        "--loss-log",
        action="store_true",
        help="log-scale the loss axis. Use for MSE. NEVER for VMIM "
        "(the NLL is negative and log() will silently drop it).",
    )
    ap.add_argument("--r2-ylim", type=float, nargs=2, default=(-0.25, 1.03))
    ap.add_argument(
        "--cmap", default="coolwarm", help="diverging colormap: coolwarm, RdBu_r, bwr, PuOr"
    )
    ap.add_argument(
        "--colwidth",
        type=float,
        default=1.95,
        help="inches per parameter column (4 params ~ 7.8in, full page)",
    )
    ap.add_argument("--png", action="store_true", help="also emit a PNG preview")
    args = ap.parse_args()

    set_pub_style()
    names, P = args.names, len(args.names)
    arms = [load_arm(s, P) for s in args.arms]

    from cmastro import cmaps
    # dict_keys(['cma:hesperia', 'cma:hesperia_r', 'cma:lacerta',
    #'cma:lacerta_r', 'cma:laguna', 'cma:laguna_r', 'cma:emph',
    #'cma:emph_r', 'cma:unph', 'cma:unph_r'])

    cmaps["cma:laguna"]

    colors = diverging_colors(len(arms), "jet_r")

    dq = {}
    for tok in args.dequant:
        i, h = tok.split(":")
        dq[int(i)] = float(h)

    have_sigma = any(a["sigma"] is not None for a in arms)
    have_loss = any(a["loss"] is not None for a in arms)

    rows = ["r2"] + (["sigma"] if have_sigma else []) + (["loss"] if have_loss else [])
    hgt = {"r2": 1.20, "sigma": 1.20, "loss": 1.50}
    fig = plt.figure(figsize=(args.colwidth * P, sum(hgt[r] for r in rows) + 0.5))
    gs = GridSpec(
        len(rows), P, figure=fig, height_ratios=[hgt[r] for r in rows], hspace=0.45, wspace=0.12
    )
    R = {r: i for i, r in enumerate(rows)}

    # ---------- row 1: RF-R^2, one panel per PARAMETER, arms overlaid ----------
    for p in range(P):
        ax = fig.add_subplot(gs[R["r2"], p])
        for a, c in zip(arms, colors, strict=False):
            ax.plot(
                a["ep"],
                a["r2"][:, p],
                "-o",
                color=c,
                mfc="white",
                mew=0.9,
                label=a["label"],
                zorder=3,
                clip_on=False,
            )
            # Mark the epoch that produced learned_compressor_bestprobe.pt --
            # selection is on the MEAN R2 across parameters, so the star can sit
            # below a panel's own peak. That is correct, and worth showing: it
            # tells the reader which checkpoint actually went downstream.
            b = int(np.argmax(a["r2"].mean(axis=1)))
            ax.plot(
                a["ep"][b],
                a["r2"][b, p],
                "*",
                color=c,
                ms=9,
                mec="0.15",
                mew=0.5,
                zorder=4,
                clip_on=False,
            )
            for x in switches(a["ep"], a["phase"]):
                ax.axvline(x, color=c, lw=0.7, ls=":", alpha=0.55, zorder=1)
        ax.axhline(0.0, color="0.35", lw=0.6, zorder=1)  # R2=0: no information
        ax.axhline(1.0, color="0.75", lw=0.6, ls="--", zorder=1)
        ax.set_ylim(*args.r2_ylim)
        ax.set_xlabel("epoch")
        ax.set_title(names[p])
        ax.grid(True, zorder=0)
        if p == 0:
            ax.set_ylabel(r"RF $R^2(\theta_p \,|\, \mathbf{t})$")
            ax.legend(loc="lower right", borderaxespad=0.3)
        else:
            ax.tick_params(labelleft=False)

    # ---------- row 2: conditional sigma of q(theta_p | t) --------------------
    if have_sigma:
        smax = max(np.nanmax(a["sigma"][:, 1 : 1 + P]) for a in arms if a["sigma"] is not None)
        top = max(1.12 * smax, 1.30 * PRIOR_SIGMA)
        for p in range(P):
            ax = fig.add_subplot(gs[R["sigma"], p])
            for a, c in zip(arms, colors, strict=False):
                if a["sigma"] is None:
                    continue
                ax.plot(
                    a["sigma"][:, 0],
                    a["sigma"][:, 1 + p],
                    "-o",
                    color=c,
                    mfc="white",
                    mew=0.9,
                    label=a["label"],
                    zorder=3,
                    clip_on=False,
                )
            # prior: sigma at/above this = t carries no information on theta_p
            ax.axhline(PRIOR_SIGMA, color="0.25", lw=0.9, ls="--", zorder=2)
            # dequant floor: hard resolution limit set by the jitter you injected
            if p in dq:
                ax.axhline(dq[p] / np.sqrt(3.0), color="#B2182B", lw=0.9, ls="-.", zorder=2)
            # head clamp: sigma pinned here = saturated, the collapse you fear
            ax.axhline(args.sigma_floor, color="0.55", lw=0.9, ls=":", zorder=2)
            ax.set_ylim(0.0, top)
            ax.set_xlabel("epoch")
            ax.grid(True, zorder=0)
            if p == 0:
                ax.set_ylabel(r"$\sigma\!\left(\theta_p \,|\, \mathbf{t}\right)$")
                tr = ax.get_yaxis_transform()
                ax.text(
                    0.02,
                    PRIOR_SIGMA,
                    "prior",
                    transform=tr,
                    va="bottom",
                    fontsize=6.5,
                    color="0.25",
                )
                ax.text(
                    0.02,
                    args.sigma_floor,
                    "floor",
                    transform=tr,
                    va="bottom",
                    fontsize=6.5,
                    color="0.55",
                )
            else:
                ax.tick_params(labelleft=False)
            if p in dq:
                ax.text(
                    0.02,
                    dq[p] / np.sqrt(3.0),
                    "dequant",
                    transform=ax.get_yaxis_transform(),
                    va="bottom",
                    fontsize=6.5,
                    color="#B2182B",
                )

    # ---------- row 3: train/val loss, ONE wide panel -------------------------
    if have_loss:
        ax = fig.add_subplot(gs[R["loss"], :])
        for a, c in zip(arms, colors, strict=False):
            if a["loss"] is None:
                continue
            L = a["loss"]
            ax.plot(L[:, 0], L[:, 1], "-", color=c, lw=1.4)  # train
            ax.plot(L[:, 0], L[:, 2], "--", color=c, lw=1.2, alpha=0.85)  # val
            if L.shape[1] > 3:
                for x in switches(L[:, 0], L[:, 3]):
                    ax.axvline(x, color=c, lw=0.7, ls=":", alpha=0.55)
        if args.loss_log:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE" if args.loss_log else r"loss  ($-\log q(\theta\,|\,t)$)")
        ax.grid(True, zorder=0)
        # Two legends: colour = arm, dash = train/val. A single 4-entry legend
        # forces the reader to parse strings like "MSE ns=1 (validation)".
        h_arm = [Line2D([], [], color=c, lw=1.4) for c in colors]
        h_spl = [
            Line2D([], [], color="0.3", lw=1.4, ls="-"),
            Line2D([], [], color="0.3", lw=1.2, ls="--"),
        ]
        l1 = ax.legend(h_arm, [a["label"] for a in arms], loc="upper right", borderaxespad=0.4)
        ax.add_artist(l1)
        ax.legend(
            h_spl,
            ["train", "validation"],
            loc="upper right",
            bbox_to_anchor=(1.0, 0.70),
            borderaxespad=0.4,
        )

    out = args.out if args.out.endswith(".pdf") else args.out + ".pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    if args.png:
        fig.savefig(out[:-4] + ".png")
        print(f"wrote {out[:-4]}.png")

    # ---------- numbers + \input{}-ready booktabs table -----------------------
    def table(title, rowfn):
        print(f"\n{title}")
        print(f"  {'arm':20s}" + " ".join(f"{n:>9}" for n in names))
        for a in arms:
            v = rowfn(a)
            if v is None:
                continue
            print(f"  {a['label']:20s}" + " ".join(f"{x:9.3f}" for x in v))

    table("best RF-R2 (per parameter):", lambda a: a["r2"].max(axis=0))
    table("final RF-R2 (per parameter):", lambda a: a["r2"][-1])
    table(
        "RF-R2 at best-probe checkpoint (the one exported):",
        lambda a: a["r2"][int(np.argmax(a["r2"].mean(axis=1)))],
    )
    if have_sigma:
        table(
            f"final sigma(theta_p|t)  [prior {PRIOR_SIGMA:.3f}]:",
            lambda a: None if a["sigma"] is None else a["sigma"][-1, 1 : 1 + P],
        )

    tex = out[:-4] + ".tex"
    with open(tex, "w") as fh:
        fh.write("% auto-generated by tools/plot_compressor_diag.py -- do not edit\n")
        fh.write("\\begin{tabular}{l" + "r" * P + "}\n\\toprule\n")
        fh.write("Arm & " + " & ".join(f"${n}$" for n in names) + r" \\" + "\n\\midrule\n")
        for a in arms:
            v = a["r2"][int(np.argmax(a["r2"].mean(axis=1)))]
            fh.write(f"{a['label']} & " + " & ".join(f"{x:.3f}" for x in v) + r" \\" + "\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nwrote {tex}  (RF-R2 at the exported checkpoint, \\input{{}}-ready)")


if __name__ == "__main__":
    main()
