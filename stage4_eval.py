#!/usr/bin/env python3
"""Stage 4: evaluate and overplot chain directories listed in .list files.

No configs, no arm-name resolution, no family/mode discovery -- you name the
directories, they get compared.

  python stage4_eval.py --list baseline.list --list cnn_mse.list \
                        --out eval_report --dlogp 10 --filtered-only

LIST FORMAT  (comma separated, blank lines and #-comments ignored)

    name, path[, n_params][, subdir][, subdir ...]

    baseline pdf,     /gscratch/ddehiwalage-don/sbi_runs/baseline_pdf
    baseline ps,      /gscratch/ddehiwalage-don/sbi_runs/baseline_ps
    cnn mse seed 80,  /gscratch/.../cnn_mse_up/seed_n1_s80, , standard_t_nsf
    cnn mse,          /gscratch/.../cnn_mse_up/n1/chains, , standard_t_nsf, standard_t_gmm
    cnn mse all,      /gscratch/.../cnn_mse_up/n1/chains, , *

  name      legend label. Anything you like; spaces are fine.
  path      either a directory holding *.dat directly, or an arm root -- in
            which case chains/ is searched one level down. Pointing straight
            at .../chains works too.
  n_params  optional. Omitted -> read from the first *_truth.npy, which is
            exactly the width stage 3 wrote, so you rarely need this.
  subdir    optional, REPEATABLE. Each one is a substring selecting a chains
            subdir, and each becomes its OWN arm. With two or more (or with
            '*', meaning every subdir found) the subdir name is appended to
            the label -- "cnn mse [standard_t_nsf]" -- because arm names key
            the legend and the metrics table and must stay unique.

Arms appear in the output in list order, and lists are concatenated in the
order given on the command line -- that fixes the plot colours, so a figure
regenerated later keeps the same arm->colour mapping.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_list(path):
    """Read one .list file -> [(name, path, n_params|None, [subdir, ...]), ...]."""
    out = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            sys.exit(f"{path}:{lineno}: need at least 'name, path' -- got {raw!r}")
        name, p = parts[0], parts[1]
        npar = int(parts[2]) if len(parts) > 2 and parts[2] else None
        subs = [x for x in parts[3:] if x]          # columns 4+ are subdirs
        out.append((name, p, npar, subs))
    if not out:
        sys.exit(f"{path}: no entries")
    return out


def has_dat(d):
    return next(Path(d).glob("*.dat"), None) is not None


def list_chain_dirs(root, name):
    """Every directory under `root` (or root/chains) that holds *.dat."""
    root = Path(root)
    search = root / "chains" if (root / "chains").is_dir() else root
    found = []
    if has_dat(search):
        found.append(search)
    if search.is_dir():
        found += [d for d in sorted(search.iterdir()) if d.is_dir() and has_dat(d)]
    return found


def resolve_chains_dir(root, subdir, name):
    """Find the directory that actually holds the *.dat files.

    Accepts either a flat directory of chains or an arm root laid out as
    {root}/chains/<scope>_<family>/. With several candidates, `subdir` picks
    one by substring; without it, ambiguity is an error rather than a guess,
    because silently averaging over the wrong NLE family is unrecoverable.
    """
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"[{name}] not a directory: {root}")

    if has_dat(root) and not subdir:
        return root

    cands = list_chain_dirs(root, name)

    if subdir:
        cands = [d for d in cands if subdir.lower() in d.name.lower()]
        if not cands:
            sys.exit(f"[{name}] no chains dir under {root} matching {subdir!r}")

    if not cands:
        sys.exit(f"[{name}] no *.dat found under {root} or {root}/chains/*/")
    if len(cands) > 1:
        opts = "\n    ".join(str(c.relative_to(root)) for c in cands)
        sys.exit(f"[{name}] {len(cands)} candidate chain dirs under {root}:\n"
                 f"    {opts}\n"
                 f"  Add a 4th column to the .list line to pick one, e.g.\n"
                 f"    {name}, {root}, , {cands[0].name}")
    return cands[0]


def infer_n_params(chains_dir, name):
    """Width of the chains, taken from the first *_truth.npy.

    stage3 writes truth as the full theta row for that chain, so its length is
    the chain width by construction -- more reliable than guessing from file
    size, which is divisible by several plausible widths.
    """
    for dat in sorted(Path(chains_dir).glob("*.dat")):
        tp = dat.with_name(dat.stem + "_truth.npy")
        if tp.exists():
            n = int(np.load(tp).reshape(-1).shape[0])
            size = dat.stat().st_size // np.dtype(np.float32).itemsize
            if size % n:
                sys.exit(f"[{name}] {dat.name}: {size} float32 values are not "
                         f"divisible by n_params={n} from {tp.name}. Pass an "
                         f"explicit n_params in the .list.")
            return n
    sys.exit(f"[{name}] no *_truth.npy in {chains_dir}, so n_params cannot be "
             f"inferred and eval.py would skip every chain. Either run "
             f"make_truth_npy.py on this directory, or set n_params in the .list "
             f"(it will still be skipped without truths).")


def audit(chains_dir, n_params, name):
    """Warn about the two silent-drop failure modes before eval.py hits them."""
    dats = sorted(Path(chains_dir).glob("*.dat"))
    no_truth = [d.name for d in dats if not d.with_name(d.stem + "_truth.npy").exists()]
    n_logp = sum(1 for d in dats if d.with_name(d.stem + "_logp.npy").exists())
    if no_truth:
        print(f"  [warn] {len(no_truth)}/{len(dats)} chains have no _truth.npy "
              f"and will be SKIPPED (e.g. {no_truth[0]})", file=sys.stderr)
    if n_logp == 0:
        print(f"  [warn] no _logp.npy: --dlogp filtering is a NO-OP for this arm, "
              f"so it is compared UNFILTERED against filtered arms",
              file=sys.stderr)
    elif n_logp < len(dats):
        print(f"  [warn] only {n_logp}/{len(dats)} chains have _logp.npy; "
              f"filtering is applied inconsistently within this arm",
              file=sys.stderr)
    return len(dats) - len(no_truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="append", default=[], dest="lists",
                    required=True, metavar="FILE.list",
                    help="repeatable; arms are concatenated in the order given")
    ap.add_argument("--out", default="eval_report")
    ap.add_argument("--thin", type=int, default=11)
    ap.add_argument("--dlogp", type=float, default=10.0)
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--burnin-frac", type=float, default=0.0)
    ap.add_argument("--filtered-only", action="store_true")
    ap.add_argument("--no-intersect", action="store_true",
                    help="evaluate each arm on its own sims instead of the "
                         "shared intersection (GV/calibration stop being "
                         "comparable across arms -- diagnostic use only)")
    ap.add_argument("--full-table", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and audit the arms, then stop")

    ap.add_argument("--n-corner", type=int, default=1, help="Number of corner plots to generate")
    args = ap.parse_args()

    arms = []
    seen = {}
    for lf in args.lists:
        if not os.path.isfile(lf):
            sys.exit(f"list file not found: {lf}")
        print(f"\n=== {lf} ===")
        for name, p, npar, subs in parse_list(lf):
            # '*' expands to every chains subdir found under the path
            if any(x == "*" for x in subs):
                dirs = list_chain_dirs(p, name)
                if not dirs:
                    sys.exit(f"[{name}] '*' matched no chains dir under {p}")
            elif subs:
                dirs = [resolve_chains_dir(p, x, name) for x in subs]
            else:
                dirs = [resolve_chains_dir(p, None, name)]

            # one subdir keeps the plain label; several get the subdir appended
            # so every arm name stays unique
            tag = len(dirs) > 1
            for cdir in dirs:
                aname = f"{name} [{cdir.name}]" if tag else name
                n = npar or infer_n_params(cdir, aname)
                n_ok = audit(cdir, n, aname)
                if aname in seen:
                    sys.exit(f"duplicate arm name {aname!r} (in {seen[aname]} "
                             f"and {lf}); names key the legend and the metrics "
                             f"table, so they must be unique")
                seen[aname] = lf
                print(f"  {aname:34s} n_params={n}  usable={n_ok:4d}  {cdir}")
                arms.append({"name": aname, "dir": str(cdir), "n_params": n})

    if not arms:
        sys.exit("no arms")

    os.makedirs(args.out, exist_ok=True)
    arms_path = os.path.join(args.out, "arms.json")
    json.dump(arms, open(arms_path, "w"), indent=2)
    print(f"\n{len(arms)} arms -> {arms_path}")

    if args.dry_run:
        print("--dry-run: stopping before eval.py")
        return

    here = os.path.dirname(os.path.abspath(__file__))

    cmd = [sys.executable, os.path.join(here, "sbi", "validation", "eval.py"),
           "--config", arms_path, "--out", args.out,
           "--thin", str(args.thin), "--dlogp", str(args.dlogp),
           "--nbins", str(args.nbins), "--burnin-frac", str(args.burnin_frac),
           "--n-corner", str(args.n_corner)]
           
    if args.filtered_only:
        cmd.append("--filtered-only")
    if args.no_intersect:
        cmd.append("--no-intersect")
    if args.full_table:
        cmd.append("--full-table")
    subprocess.run(cmd, check=True)

    # eval_arm() returns None for an arm with zero usable chains and it is then
    # dropped from the report without an error, so confirm nothing vanished.
    rep = json.load(open(os.path.join(args.out, "metrics.json")))
    for mode, rs in rep.items():
        got = {r["name"] for r in rs}
        lost = [a["name"] for a in arms if a["name"] not in got]
        print(f"[{mode}] {len(rs)}/{len(arms)} arms produced metrics")
        for n in lost:
            print(f"  DROPPED: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()