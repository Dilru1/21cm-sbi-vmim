#!/usr/bin/env python3
"""Stage 4: SBC calibration and constraining power of groups of runs.

  python scripts/stage4_eval.py configs/eval/paper.yaml

Each group is either `config:` (every compressor x NLE seed listed in that config's
seeds:) or `chains:` (explicit chain directories). Every run is evaluated on the
simulations shared by all runs in the file. Writes metrics.csv (per run),
metrics_groups.csv (seed average: geometric mean for GV and sigma, arithmetic for the
calibration ratio; spread = std over seeds, of log10 for GV/sigma) and sbc.png.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from sbi.config import load_config, run_dir  # noqa: E402
from sbi.evaluate import chain_sim, eval_run, list_chains  # noqa: E402


def group_runs(g):
    if "chains" in g:
        return [(os.path.basename(os.path.dirname(d.rstrip("/"))) + "/" + os.path.basename(d.rstrip("/")), d)
                for d in g["chains"]]
    cfg = load_config(os.path.join(ROOT, g["config"]))
    cs_list = [0] if cfg["compressor"]["type"] == "none" else cfg["seeds"]["compressor"]
    return [(f"c{cs}_n{ns}", str(run_dir(cfg, cs) / f"chains_n{ns}"))
            for cs in cs_list for ns in cfg["seeds"]["nle"]]


def n_params_of(d):
    f = next(iter(list_chains(d)), None)
    if f is None:
        sys.exit(f"no chains in {d}")
    return int(np.load(f[:-4] + "_truth.npy").size)


def plot_sbc(res, names, nbins, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = len(names)
    fig, axes = plt.subplots(1, P, figsize=(3.2 * P, 3), sharey=True)
    x = (np.arange(nbins) + 0.5) / nbins
    for label, runs in res.items():
        h = np.mean([r["rank_hist"] for r in runs], axis=0)
        for j in range(P):
            axes[j].step(x, h[j], where="mid", label=label)
    for j, ax in enumerate(axes):
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_title(names[j])
        ax.set_xlabel("rank / N")
    axes[0].set_ylabel("density (seed mean)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    ev = load_config(args.config)
    params, names = ev["params"], ev["param_names"]
    os.makedirs(ev["out"], exist_ok=True)

    groups = {g["label"]: group_runs(g) for g in ev["groups"]}
    sims = set.intersection(*[{chain_sim(f) for f in list_chains(d)} - {None}
                              for runs in groups.values() for _, d in runs])
    print(f"{sum(map(len, groups.values()))} runs, {len(sims)} shared simulations", flush=True)

    res, rows = {}, []
    for label, runs in groups.items():
        res[label] = []
        for run, d in runs:
            r = eval_run(d, n_params_of(d), params, ev, sims)
            if r is None:
                print(f"  [skip] {label} {run}: no usable chains", flush=True)
                continue
            r.update(group=label, run=run, dir=d)
            res[label].append(r)
            rows.append([label, run, r["used"], f"{r['gv']:.6e}"] + [f"{s:.6e}" for s in r["sigma"]]
                        + [f"{c:.3f}" for c in r["calib_ratio"]])
            print(f"  {label:28s} {run:12s} N={r['used']} GV={r['gv']:.3e} "
                  f"calib={[round(c, 2) for c in r['calib_ratio']]}", flush=True)

    head = ["group", "run", "n_used", "GV"] + [f"sigma_{n}" for n in names] + [f"calib_{n}" for n in names]
    with open(os.path.join(ev["out"], "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows([head] + rows)

    ghead = ["group", "n_runs", "GV", "GV_std_log10"]
    for n in names:
        ghead += [f"sigma_{n}", f"sigma_{n}_std_log10"]
    for n in names:
        ghead += [f"calib_{n}", f"calib_{n}_std"]
    grows = []
    print("\n=== group means ===")
    for label, rs in res.items():
        if not rs:
            continue
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        lg = np.log10([r["gv"] for r in rs])
        row = [label, len(rs), f"{10 ** lg.mean():.6g}", f"{sd(lg):.3g}"]
        for j in range(len(params)):
            ls = np.log10([r["sigma"][j] for r in rs])
            row += [f"{10 ** ls.mean():.6g}", f"{sd(ls):.3g}"]
        cal = np.array([r["calib_ratio"] for r in rs])
        for j in range(len(params)):
            row += [f"{cal[:, j].mean():.4g}", f"{sd(cal[:, j]):.3g}"]
        grows.append(row)
        print(f"  {label:28s} runs={len(rs)} GV={10 ** lg.mean():.3e} "
              f"calib={np.round(cal.mean(0), 2).tolist()}")
    with open(os.path.join(ev["out"], "metrics_groups.csv"), "w", newline="") as f:
        csv.writer(f).writerows([ghead] + grows)
    with open(os.path.join(ev["out"], "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot_sbc(res, names, ev["nbins"], os.path.join(ev["out"], "sbc.png"))
    print(f"\n-> {ev['out']}/ metrics.csv metrics_groups.csv metrics.json sbc.png")


if __name__ == "__main__":
    main()
