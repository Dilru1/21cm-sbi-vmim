#!/usr/bin/env python3
"""Plot Stage 2 NLE training histories, discovering all seed runs per config.

Seeding note
------------
When compressor.tag_arm_with_seed is true, resolve_arm_name() appends
``_s<init_seed>`` to the arm name, so a run lives at:

    {scratch_root}/{arm_name}_s<seed>/nle/<transform>/<model>/loss_history.npy

This plotter takes one OR MORE config YAMLs. For each config it auto-discovers
every seed-tagged run (``{arm_name}_s*``), and overlays all seeds in the same
subplot. Multiple configs each get their own ROW of subplots (one column per
target transform found). Colour encodes the seed; solid = train, dashed = val.

Usage
-----
  # one config, all its seeds overlaid
  python tools/plot_training_nle.py configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml \
      --out noise_ana/loss_nle_n1_jitter.png

  # several configs -> one row each, all seeds overlaid within each
  python tools/plot_training_nle.py \
      configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml \
      configs_seeds/noise/arm_cnn_vmim_jitter_n2.yaml \
      --out noise_ana/loss_nle_n1_n2.png

Optional filters:  --models nsf         (subset of gmm/maf/nsf)
                   --seeds 1 2 3        (subset of discovered seeds)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TRANSFORMS = ["raw_t", "standard_t"]
MODELS = ["gmm", "maf", "nsf"]
TX_TITLE = {"raw_t": "Raw summaries (raw_t)",
            "standard_t": "Standardized summaries (standard_t)"}
LOSS_NAMES = ["loss_history.npy", "stage2_nle_loss_history.npy"]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_and_load_loss(model_dir):
    """Return the loss array (2-D) from a model dir, or None."""
    for name in LOSS_NAMES:
        target = os.path.join(model_dir, name)
        if os.path.exists(target):
            try:
                data = np.load(target)
                if data.ndim == 1:
                    data = data[None, :]
                return data
            except Exception as e:
                print(f"  [WARN] {target} failed to parse: {e}", file=sys.stderr)
    return None


def parse_loss_columns(arr):
    """Handle 3-col (step,train,val) and 5-col (step,phase,split,train,val)."""
    cols = arr.shape[1]
    if cols == 3:
        return arr[:, 0], arr[:, 1], arr[:, 2], None
    if cols >= 5:
        return arr[:, 0], arr[:, 3], arr[:, 4], arr[:, 1]
    return None


def discover_seed_runs(yaml_path):
    """Return (label, {seed: nle_dir}) for every seed-tagged run of this config."""
    if not os.path.exists(yaml_path):
        print(f"[ERROR] config not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)
    cfg = load_yaml(yaml_path)
    scratch_root = cfg.get("scratch_root")
    arm_name = cfg.get("arm_name")
    if not scratch_root or not arm_name:
        print(f"[ERROR] YAML needs scratch_root and arm_name "
              f"(got {scratch_root}, {arm_name})", file=sys.stderr)
        sys.exit(1)
    label = os.path.basename(yaml_path).replace(".yaml", "")

    runs = {}
    # tagged seed runs: {scratch_root}/{arm_name}_s<INT>/nle
    pattern = os.path.join(scratch_root, arm_name + "_s*", "nle")
    for nle_dir in glob.glob(pattern):
        parent = os.path.basename(os.path.dirname(nle_dir))     # e.g. n1_jitter_s3
        m = re.search(r"_s(\d+)$", parent)                      # digits only -> real seed
        if m:
            runs[int(m.group(1))] = nle_dir
    # untagged run (tagging off / legacy): {scratch_root}/{arm_name}/nle
    base = os.path.join(scratch_root, arm_name, "nle")
    if os.path.isdir(base):
        seed = cfg.get("compressor", {}).get("init_seed", None)
        runs.setdefault(seed if seed is not None else "base", base)

    return label, runs


def seed_key(s):
    """Sort ints numerically, put any string keys ('base') last."""
    return (0, s) if isinstance(s, int) else (1, str(s))


def main():
    ap = argparse.ArgumentParser(description="NLE loss plotter (per-config, all seeds overlaid)")
    ap.add_argument("configs", nargs="+", help="one or more config YAML paths")
    ap.add_argument("--out", default="nle_training_comparison.png")
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help="subset of gmm/maf/nsf to plot (default: all present)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="subset of seeds to plot (default: all discovered)")
    args = ap.parse_args()
    models_wanted = [m for m in MODELS if m in set(args.models)]

    # ---- collect: per config -> {tx -> {seed -> {model -> arr}}} ----
    per_config = []
    all_seeds = set()
    active_tx = []
    for yaml_path in args.configs:
        label, runs = discover_seed_runs(yaml_path)
        if args.seeds is not None:
            runs = {s: d for s, d in runs.items() if s in set(args.seeds)}
        found = {sd: None for sd in runs}
        cell = {}
        for seed, nle_dir in runs.items():
            for tx in TRANSFORMS:
                tx_path = os.path.join(nle_dir, tx)
                if not os.path.isdir(tx_path):
                    continue
                for model in models_wanted:
                    arr = find_and_load_loss(os.path.join(tx_path, model))
                    if arr is None:
                        continue
                    cell.setdefault(tx, {}).setdefault(seed, {})[model] = arr
                    all_seeds.add(seed)
                    found[seed] = True
        seeds_ok = sorted([s for s in runs if found.get(s)], key=seed_key)
        print(f"[{label}] discovered seeds={sorted(runs, key=seed_key)}  "
              f"with loss data={seeds_ok}")
        per_config.append((label, cell))
        for tx in cell:
            if tx not in active_tx:
                active_tx.append(tx)

    active_tx = [t for t in TRANSFORMS if t in active_tx]
    if not active_tx:
        print("[ERROR] no loss histories found for any config/seed.", file=sys.stderr)
        sys.exit(1)

    # ---- consistent seed -> colour across all subplots ----
    seeds_sorted = sorted(all_seeds, key=seed_key)
    cmap = plt.get_cmap("tab10")
    seed_color = {s: cmap(i % 10) for i, s in enumerate(seeds_sorted)}

    n_rows, n_cols = len(per_config), len(active_tx)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.4 * n_cols, 4.0 * n_rows), squeeze=False)

    for r, (label, cell) in enumerate(per_config):
        for c, tx in enumerate(active_tx):
            ax = axes[r][c]
            tx_cell = cell.get(tx, {})
            n_models_here = len({m for md in tx_cell.values() for m in md})
            multi_model = n_models_here > 1
            drawn = False
            max_step = 10.0
            phase_drawn = False

            for seed in sorted(tx_cell, key=seed_key):
                color = seed_color[seed]
                for model, arr in sorted(tx_cell[seed].items()):
                    parsed = parse_loss_columns(arr)
                    if parsed is None:
                        print(f"  [WARN] {label}/{tx}/s{seed}/{model}: "
                              f"{arr.shape[1]} cols, skipping")
                        continue
                    steps, train, val, phase = parsed
                    if len(steps):
                        max_step = max(max_step, float(np.max(steps)))
                    lbl = f"s{seed}" + (f"·{model}" if multi_model else "")
                    ax.plot(steps, train, "-", color=color, lw=1.5, label=lbl)
                    ax.plot(steps, val, "--", color=color, lw=1.1, alpha=0.7)
                    drawn = True
                    if phase is not None and not phase_drawn:
                        for s in np.where(np.diff(phase) != 0)[0]:
                            ax.axvline(steps[s + 1], color="grey", lw=0.8,
                                       ls=":", alpha=0.5)
                        phase_drawn = True

            ax.set_title(f"{label}\n{TX_TITLE.get(tx, tx)}",
                         fontsize=10, fontweight="semibold", pad=8)
            ax.set_xlabel("Epoch / chunk step", fontsize=9)
            ax.set_ylabel("NLE loss  (-log q)", fontsize=9)
            ax.grid(True, ls="--", alpha=0.3)
            ax.set_xlim(0, max_step * 1.02)
            if drawn:
                ax.legend(title="seed  (solid=train, dashed=val)",
                          fontsize=8, title_fontsize=8, ncol=max(1, len(seeds_sorted) // 4),
                          loc="upper right", framealpha=0.9)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\n[SUCCESS] wrote {args.out}")


if __name__ == "__main__":
    main()