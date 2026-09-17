#!/usr/bin/env python3
"""Export per-simulation SBC ranks and corner-plot samples for the project website.

Uses the same .list files as stage4_eval.py and the same chain loader as
sbi/validation/eval.py, so the burn-in, thinning and dlogp filter are identical
to the published metrics.

  python export_web.py --list sbc_lists/cnn_vmim_nojit_n1.list --out docs/data

OUTPUT (each arm is self-contained, so arms can be exported one at a time)

  docs/data/manifest.json                   every exported arm, rebuilt on each run
  docs/data/arms/<key>/summary.json         per-sim SBC counts, truth, widths, GV
  docs/data/arms/<key>/sims/sim<ID>.bin     corner samples, uint16 little-endian,
                                            shape (n_keep, 4), decode with lo/hi

Browser-side decoding of one sample:  x = lo[j] + q / 65535 * (hi[j] - lo[j])
SBC bin for nbins bins:               b = min(floor(below[j] / n_samples * nbins), nbins - 1)
which is exactly eval.sbc_rank(), because below/n_samples is the same fraction.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sbi" / "validation"))

from eval import (COMMON_PARAMS, list_chains, load_chain, iqr_sigma,  # noqa: E402
                  common_sims, PATTERN)
from stage4_eval import (parse_list, list_chain_dirs,  # noqa: E402
                         resolve_chains_dir, infer_n_params)

WEB_LABELS = ["log10 fX", "tau", "rH/S", "log10 Mmin", "fesc"]


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def resolve_arms(list_files):
    """Same arm expansion as stage4_eval.main(), without the audit printout."""
    arms = []
    for lf in list_files:
        for name, p, npar, subs in parse_list(lf):
            if any(x == "*" for x in subs):
                dirs = list_chain_dirs(p, name)
            elif subs:
                dirs = [resolve_chains_dir(p, x, name) for x in subs]
            else:
                dirs = [resolve_chains_dir(p, None, name)]
            tag = len(dirs) > 1
            for cdir in dirs:
                aname = f"{name} [{cdir.name}]" if tag else name
                arms.append({"name": aname, "dir": str(cdir),
                             "n_params": npar or infer_n_params(cdir, aname)})
    return arms


def export_arm(arm, out_root, thin, burnin_frac, dlogp, n_keep, sims_keep, seed):
    key = slug(arm["name"])
    arm_dir = Path(out_root) / "arms" / key
    sim_dir = arm_dir / "sims"
    sim_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    P = COMMON_PARAMS

    sims, skipped, dup = {}, 0, 0
    for f in list_chains(arm["dir"]):
        m = PATTERN.search(os.path.basename(f))
        if m is None:
            continue
        sim, row = int(m.group(1)), int(m.group(2))
        if sims_keep is not None and sim not in sims_keep:
            continue
        loaded = load_chain(f, arm["n_params"], burnin_frac, thin, dlogp)
        if loaded is None:
            skipped += 1
            continue
        samples, truth = loaded
        sub = samples[:, P]
        n = sub.shape[0]

        # --- statistics, computed exactly as eval.eval_arm does -----------
        below = [int(np.sum(samples[:, j] < truth[j])) for j in P]
        cov = np.cov(sub, rowvar=False)
        sign, logdet = np.linalg.slogdet(cov)
        gv = float(np.exp(0.5 * logdet)) if sign > 0 and np.isfinite(logdet) else None

        # --- corner samples: random subset, quantised to uint16 ----------
        idx = rng.choice(n, n_keep, replace=n < n_keep)
        keep = sub[idx]
        lo, hi = keep.min(0), keep.max(0)
        span = np.where(hi > lo, hi - lo, 1.0)
        q = np.round((keep - lo) / span * 65535).astype("<u2")
        q.tofile(sim_dir / f"sim{sim}.bin")

        if str(sim) in sims:
            dup += 1
        sims[str(sim)] = {
            "row": row,
            "truth": [round(float(truth[j]), 6) for j in P],
            "n_samples": n,
            "below": below,
            "median": [round(float(np.median(samples[:, j])), 6) for j in P],
            "sigma": [round(float(iqr_sigma(samples[:, j])), 7) for j in P],
            "gv": gv,
            "lo": [float(v) for v in lo],
            "hi": [float(v) for v in hi],
        }

    summary = {
        "key": key, "name": arm["name"], "source_dir": arm["dir"],
        "n_params": arm["n_params"], "params": [WEB_LABELS[j] for j in P],
        "settings": {"thin": thin, "burnin_frac": burnin_frac,
                     "dlogp": dlogp, "n_keep": n_keep},
        "sims": dict(sorted(sims.items(), key=lambda kv: int(kv[0]))),
    }
    with open(arm_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, separators=(",", ":"))

    msg = f"[export] {arm['name']:34s} sims={len(sims):4d} skipped={skipped}"
    if dup:
        msg += f"  WARNING {dup} sims had several chains; the last one was kept"
    print(msg + f"  -> {arm_dir}")
    return summary


def calib_from_summary(summary, nbins, sims_subset=None):
    """Rebuild eval.eval_arm's rank_hist and calib_ratio from the exported counts."""
    P = len(summary["params"])
    hist = np.zeros((P, nbins))
    used = 0
    for sid, s in summary["sims"].items():
        if sims_subset is not None and int(sid) not in sims_subset:
            continue
        for j in range(P):
            frac = s["below"][j] / s["n_samples"]
            hist[j, min(int(frac * nbins), nbins - 1)] += 1
        used += 1
    hist *= nbins / used
    calib = np.mean((hist - 1.0) ** 2, axis=1)
    gvs = [s["gv"] for sid, s in summary["sims"].items()
           if s["gv"] is not None and (sims_subset is None or int(sid) in sims_subset)]
    return calib / ((nbins - 1) / used), float(np.median(gvs)), used


def check_against_metrics(summaries, metrics_path, nbins):
    rep = json.load(open(metrics_path))
    mode = next((m for m in rep if m.startswith("filtered")), next(iter(rep)))
    ref = {r["name"]: r for r in rep[mode]}
    ok = True
    for s in summaries:
        r = ref.get(s["name"])
        if r is None:
            print(f"[check] {s['name']}: not in {metrics_path} ({mode})")
            continue
        ratio, gv, used = calib_from_summary(s, nbins)
        same = (used == r["used"] and np.allclose(ratio, r["calib_ratio"], atol=1e-9)
                and math.isclose(gv, r["gv"], rel_tol=1e-9))
        ok &= same
        print(f"[check] {s['name']}: used {used}/{r['used']}  "
              f"calib {np.round(ratio, 3).tolist()} vs {np.round(r['calib_ratio'], 3).tolist()}  "
              f"GV {gv:.4e} vs {r['gv']:.4e}  -> {'MATCH' if same else 'MISMATCH'}")
    return ok


def write_manifest(out_root):
    arms = []
    for p in sorted((Path(out_root) / "arms").glob("*/summary.json")):
        s = json.load(open(p))
        arms.append({"key": s["key"], "name": s["name"], "n_sims": len(s["sims"]),
                     "settings": s["settings"],
                     "summary": f"arms/{s['key']}/summary.json",
                     "sims": f"arms/{s['key']}/sims/sim{{id}}.bin"})
    manifest = {"params": [WEB_LABELS[j] for j in COMMON_PARAMS], "arms": arms}
    json.dump(manifest, open(Path(out_root) / "manifest.json", "w"), indent=1)
    print(f"[export] manifest: {len(arms)} arms -> {out_root}/manifest.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="append", required=True, dest="lists")
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--thin", type=int, default=11)          # stage4 defaults
    ap.add_argument("--burnin-frac", type=float, default=0.0)
    ap.add_argument("--dlogp", type=float, default=10.0)
    ap.add_argument("--n-keep", type=int, default=1000)
    ap.add_argument("--intersect", action="store_true",
                    help="keep only sims shared by all arms in these lists "
                         "(the browser intersects arms anyway)")
    ap.add_argument("--check", metavar="metrics.json",
                    help="compare rebuilt calibration and GV with a stage4 report")
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = resolve_arms(args.lists)
    sims_keep = common_sims(arms) if args.intersect else None
    summaries = [export_arm(a, args.out, args.thin, args.burnin_frac, args.dlogp,
                            args.n_keep, sims_keep, args.seed) for a in arms]
    write_manifest(args.out)
    if args.check and not check_against_metrics(summaries, args.check, args.nbins):
        sys.exit(1)


if __name__ == "__main__":
    main()
