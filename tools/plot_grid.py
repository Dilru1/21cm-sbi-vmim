#!/usr/bin/env python3
"""Read the VMIM sweep CONFIG files, resolve each run's output dir from
(scratch_root, arm_name), and produce:

  * one figure PER HEAD (gmm / maf / nf):
        top row  = RF-R2 vs epoch, one subplot per parameter, the 8 sub-arms
                   (norm x arch x aug) overlaid
        bottom row = raw train/val loss vs epoch, one subplot per sub-arm
  * one GLOBAL figure comparing the head-level scores (weak-param rHS bar chart
    grouped by head, plus a printed leaderboard)

The head is taken from each config's compressor.vmim_head; the sub-arm label is
built from the other three factors (norm/arch/aug) parsed from the config path.

USAGE:
  python tools/plot_sweep_by_head.py \
      --configs 'configs/vmim_config/**/*.yaml' \
      --names Fx tau rHS Mmin --out-dir sweep_plots
"""

import argparse
import glob as globmod
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    sys.exit("pyyaml required: pip install pyyaml")

DEFAULT_NAMES = ["Fx", "tau", "rHS", "Mmin"]
HEAD_MAP = {"gmm_old": "gmm", "gmm_new": "gmm", "nf_maf": "maf", "nf_spline": "nf"}


def sub_label(path):
    """norm/arch/aug tag for a sub-arm, from the config path."""
    p = path.replace("\\", "/")
    norm = (re.search(r"norm_(true|false)", p) or [None, "?"])[1]
    arch = (re.search(r"cnn_(conv4d|seblock)_", p) or [None, "?"])[1]
    aug = (re.search(r"aug_(true|false)", p) or [None, "?"])[1]
    return f"{arch}/norm={norm}/aug={aug}"


def load_run(cfg_path):
    """Read a config, resolve its nle dir, load rf + loss histories."""
    cfg = yaml.safe_load(open(cfg_path))
    scratch = cfg.get("scratch_root", "")
    arm = cfg.get("arm_name", "")
    head_raw = cfg.get("compressor", {}).get("vmim_head", "?")
    head = HEAD_MAP.get(head_raw, head_raw)
    nle = os.path.join(scratch, arm, "nle")
    rf_p = os.path.join(nle, "rf_r2_history.npy")
    loss_p = os.path.join(nle, "loss_history.npy")
    rf = loss = None
    if os.path.exists(rf_p):
        rf = np.load(rf_p)
        if rf.ndim == 1:
            rf = rf[None, :]
    else:
        print(f"[WARN] no rf_r2_history for {arm} at {rf_p}", file=sys.stderr)
    if os.path.exists(loss_p):
        loss = np.load(loss_p)
        if loss.ndim == 1:
            loss = loss[None, :]
    return {
        "cfg": cfg_path,
        "head": head,
        "head_raw": head_raw,
        "label": sub_label(cfg_path),
        "arm": arm,
        "rf": rf,
        "loss": loss,
    }


def per_head_figure(head, runs, names, out_dir):
    runs = [r for r in runs if r["rf"] is not None]
    if not runs:
        print(f"[skip] {head}: no runs with rf history", file=sys.stderr)
        return
    n_params = min(r["rf"].shape[1] - 1 for r in runs)
    n_params = min(n_params, len(names))
    n_arms = len(runs)
    ncol = max(n_params, n_arms)
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 3.3 * 2), squeeze=False)
    cmap = plt.get_cmap("tab10")

    # top: RF-R2 per param, arms overlaid
    for p in range(n_params):
        ax = axes[0][p]
        for ri, r in enumerate(runs):
            a = r["rf"]
            ax.plot(a[:, 0], a[:, p + 1], "-o", ms=2.5, color=cmap(ri % 10), label=r["label"])
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.axhline(1, color="grey", lw=0.5, ls="--", alpha=0.4)
        ax.set_title(f"{names[p]}  RF-R\u00b2")
        ax.set_xlabel("epoch")
        ax.set_ylabel("RF R\u00b2")
        ax.set_ylim(-0.3, 1.02)
        ax.grid(alpha=0.25)
        if p == 0:
            ax.legend(fontsize=6, loc="lower right")
    for k in range(n_params, ncol):
        axes[0][k].axis("off")

    # bottom: raw loss per sub-arm
    for ri, r in enumerate(runs):
        ax = axes[1][ri]
        L = r["loss"]
        if L is None:
            ax.text(
                0.5,
                0.5,
                "no loss_history",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="grey",
            )
            ax.set_title(r["label"], fontsize=7)
            ax.axis("off")
            continue
        ep = L[:, 0]
        ax.plot(ep, L[:, 1], "-", color=cmap(ri % 10), label="train")
        ax.plot(ep, L[:, 2], "--", color=cmap(ri % 10), alpha=0.7, label="val")
        if L.shape[1] > 3:  # mark aux->VMIM phase switches
            for s in np.where(np.diff(L[:, 3]) != 0)[0]:
                ax.axvline(ep[s + 1], color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(r["label"], fontsize=7)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    for k in range(n_arms, ncol):
        axes[1][k].axis("off")

    fig.suptitle(
        f"head = {head}   (top: RF-R\u00b2 per param, arms overlaid | "
        f"bottom: raw loss per sub-arm)",
        y=1.005,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, f"sweep_{head}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n_arms} sub-arms)")


def global_figure(all_runs, names, out_dir, metric="last", weak="rHS"):
    widx = names.index(weak) if weak in names else 2
    scored = []
    for r in all_runs:
        if r["rf"] is None or r["rf"].shape[1] - 1 <= widx:
            continue
        col = r["rf"][:, widx + 1]
        s = col[-1] if metric == "last" else col.max()
        scored.append((r["head"], r["label"], float(s)))
    if not scored:
        print("[skip] global: no scored runs", file=sys.stderr)
        return

    heads = sorted({h for h, _, _ in scored})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(8, len(scored) * 0.5), 5))
    x = 0
    xticks = []
    xlabels = []
    for hi, h in enumerate(heads):
        grp = sorted([s for s in scored if s[0] == h], key=lambda z: z[2], reverse=True)
        for _, lbl, s in grp:
            ax.bar(x, s, color=cmap(hi % 10))
            xticks.append(x)
            xlabels.append(f"{h}:{lbl}")
            x += 1
        x += 0.6  # gap between heads
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
    ax.set_ylabel(f"{weak} RF-R\u00b2 ({metric})")
    ax.set_title(f"Global comparison: {weak} recovery by run, grouped by head")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(out_dir, "sweep_global_scores.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # leaderboard + head means
    print(f"\n=== GLOBAL leaderboard ({weak}, {metric}) ===")
    for h, lbl, s in sorted(scored, key=lambda z: z[2], reverse=True):
        print(f"  {s:6.3f}  {h:4s}  {lbl}")
    print("\nhead means:")
    for h in heads:
        vals = [s for hh, _, s in scored if hh == h]
        print(f"  {h:4s}: mean {weak}={np.mean(vals):.3f}  (best {max(vals):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="glob for config YAMLs (use ** for recursive)")
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--metric", default="last", choices=["last", "best"])
    ap.add_argument("--weak", default="rHS")
    ap.add_argument("--out-dir", default="sweep_plots")
    args = ap.parse_args()

    cfgs = sorted(globmod.glob(args.configs, recursive=True))
    if not cfgs:
        sys.exit(f"no configs matched {args.configs}")
    os.makedirs(args.out_dir, exist_ok=True)

    all_runs = [load_run(c) for c in cfgs]
    heads = sorted({r["head"] for r in all_runs})
    print(f"loaded {len(all_runs)} configs across heads: {heads}")

    for h in heads:
        per_head_figure(h, [r for r in all_runs if r["head"] == h], args.names, args.out_dir)
    global_figure(all_runs, args.names, args.out_dir, args.metric, args.weak)


if __name__ == "__main__":
    main()
