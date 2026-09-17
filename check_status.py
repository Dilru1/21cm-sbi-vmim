#!/usr/bin/env python3
"""What is DONE / PARTIAL / MISSING across a compressor x NLE seed grid.

    # explicit grid, same syntax as submit_nle_grid.sh
    python check_status.py configs/vmim_/arm_cnn_vmim_n1_jit_0.05.yaml --seed.compressor 43,44,45 --seed.nle 42,43 --families nsf

    # discover whatever is already on disk
    python check_status.py configs/vmim_/*.yaml --auto-seeds --families nsf

    # single arm, no seed tagging (the pre-ensemble layout)
    python check_status.py configs/vmim_/arm_cnn_vmim_jitter_n1.yaml

Torch-free (sbi.config / sbi.data / sbi.seeding are), so it runs in a second on
a login node while jobs are queued.

LAYOUT
  compressor seed  ->  {scratch_root}/{arm_name}_s<cseed>/       (tag_arm_with_seed)
  NLE seed         ->  {arm}/nle/<scope>/<family>/seed_<nseed>/  (nle.seed_subdir)
  chains           ->  {arm}/chains/<scope>_<family>[_seed_<nseed>]/
Without seeds it falls back to the untagged paths, so it works either way.

WHAT EACH CHECK MEANS
  stage 1   summaries/t.npy exists + row count. For cnn arms it also counts
            leftover part_* shard dirs: their presence means the merge has not
            run, or ran and left stale shards from a different NSHARD (which
            makes merge_export_shards.py refuse next time).
  stage 2   loss_history.npy ROW COUNT, not the checkpoint. save_nle() writes
            model_config.json on the first improving chunk, so a checkpoint
            exists almost immediately and says nothing about completion. A
            finished run has exactly nsplit + nsplit//2 rows.
  stage 3   *.dat count vs the SBC target count, plus a DUPLICATE check, plus a
            check that chains are actually separated by NLE seed -- an
            unpatched stage3_mcmc.py writes every seed into the same
            chains/<scope>_<family>/ with identical filenames, so seeds
            silently OVERWRITE each other.

Exit code 0 when everything requested is DONE, 1 otherwise.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import load_config, load_sbc_targets
from sbi.seeding import resolve_seeds

SIM_RE = re.compile(r"_sim(\d+)_")
DONE, PART, MISS = "DONE", "PARTIAL", "MISSING"


def arm_info(cfg_path, cseed=None):
    """Paths for ONE compressor seed. cseed=None -> untagged layout."""
    c = load_config(cfg_path)
    cc, nc, mc = c.get("compressor", {}), c.get("nle", {}), c.get("mcmc", {})
    name = c["arm_name"]
    if cseed is not None:
        name = f"{name}_s{cseed}"           # mirrors seeding.resolve_arm_name
    elif cc.get("tag_arm_with_seed", False):
        name = f"{name}_s{resolve_seeds(cc)[1]}"
    root = Path(c["scratch_root"]) / name
    mode = str(nc.get("t_scaling")
               or ("standard" if nc.get("standardize", True) else "raw")).lower()
    nsplit = int(nc.get("nsplit", 10))
    return dict(cfg=cfg_path, name=name, type=c.get("arm_type", "cnn"), root=root,
                nle=root / "nle", summ=root / "summaries", chains=root / "chains",
                scope="standard_t" if mode == "standard" else "raw_t",
                default_family=str(nc.get("model", "gmm")).lower(),
                yaml_families=[str(m).lower() for m in (nc.get("models") or [])],
                chunks_expected=nsplit + max(1, nsplit // 2),
                target_path=mc.get("target_path"))


def nle_dir(a, fam, nseed):
    d = a["nle"] / a["scope"] / fam
    return d / f"seed_{nseed}" if nseed is not None else d


def stage1_status(a):
    t = a["summ"] / "t.npy"
    parts = sorted(a["summ"].glob("part_*")) if a["summ"].is_dir() else []
    if not t.exists():
        if parts:
            return PART, f"{len(parts)} part_* dirs, no merged t.npy"
        return MISS, "no summaries/t.npy"
    rows, dim = np.load(t, mmap_mode="r").shape
    note = f"{rows:,} rows x {dim}"
    if parts:
        return PART, note + f"  [WARN {len(parts)} stale part_* dirs]"
    return DONE, note


def stage2_status(a, fam, nseed):
    d = nle_dir(a, fam, nseed)
    cfgf, loss = d / "model_config.json", d / "loss_history.npy"
    if not d.is_dir() or not (cfgf.exists() or loss.exists()):
        return MISS, "not started", d
    n = len(np.load(loss)) if loss.exists() else 0
    exp = a["chunks_expected"]
    if not cfgf.exists():
        return PART, f"{n}/{exp} chunks, no checkpoint yet", d
    if n < exp:
        return PART, f"{n}/{exp} chunks (running or died)", d
    return DONE, f"{n}/{exp} chunks, best val {float(np.load(loss)[:, 2].min()):.4f}", d


def stage3_status(a, fam, nseed, n_targets):
    seeded = a["chains"] / f"{a['scope']}_{fam}_seed_{nseed}" if nseed is not None else None
    plain = a["chains"] / f"{a['scope']}_{fam}"
    d = seeded if (seeded is not None and seeded.is_dir()) else plain
    if not d.is_dir():
        return MISS, "not started", (seeded or plain)
    if nseed is not None and d is plain:
        return PART, (f"only the un-seeded dir {plain.name} exists -- stage3 is not "
                      f"separating chains by NLE seed, so seeds OVERWRITE each other"), d
    files = sorted(d.glob("*.dat"))
    sims = {int(m.group(1)) for f in files if (m := SIM_RE.search(f.name))}
    if len(files) != len(sims):
        return PART, (f"{len(files)} files but {len(sims)} unique sims -- "
                      f"DUPLICATES, rm -rf and re-run"), d
    if n_targets and len(files) < n_targets:
        return PART, f"{len(files)}/{n_targets} chains", d
    return DONE, f"{len(files)} chains", d


def discover_cseeds(cfg_path):
    c = load_config(cfg_path)
    root = Path(c["scratch_root"]) / c["arm_name"]
    out = []
    for p in sorted(root.parent.glob(root.name + "_s*")):
        m = re.search(r"_s(\d+)$", p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return out


def discover_nseeds(a, fam):
    base = a["nle"] / a["scope"] / fam
    if not base.is_dir():
        return []
    out = []
    for p in base.glob("seed_*"):
        m = re.search(r"seed_(\d+)$", p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seed.compressor", dest="cseeds", default=None,
                    help="comma list, e.g. 43,44,45")
    ap.add_argument("--seed.nle", dest="nseeds", default=None, help="comma list")
    ap.add_argument("--auto-seeds", action="store_true",
                    help="discover _s<N> arm dirs and seed_<N> nle dirs on disk")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--expect-chains", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    paths = []
    for p in args.configs:
        paths.extend(sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("no configs matched")

    rows, all_done = [], True
    for p in paths:
        if args.cseeds:
            cseeds = [int(x) for x in args.cseeds.split(",")]
        elif args.auto_seeds:
            cseeds = discover_cseeds(p) or [None]
        else:
            cseeds = [None]

        for cs in cseeds:
            a = arm_info(p, cs)
            fams = args.families or a["yaml_families"] or [a["default_family"]]
            n_targets = args.expect_chains
            if n_targets is None and a["target_path"] and os.path.exists(a["target_path"]):
                try:
                    n_targets = len(load_sbc_targets(a["target_path"]))
                except Exception:
                    pass

            s1, n1 = stage1_status(a)
            print(f"\n{'=' * 78}")
            print(f"{a['name']}   [{a['type']}, scope={a['scope']}]")
            print(f"{'=' * 78}")
            if args.verbose:
                print(f"  cfg  {p}")
                print(f"  root {a['root']}")
            print(f"  stage1  {s1:<8} {n1}")
            if s1 != DONE:
                all_done = False

            for fam in fams:
                if args.nseeds:
                    nseeds = [int(x) for x in args.nseeds.split(",")]
                elif args.auto_seeds:
                    nseeds = discover_nseeds(a, fam) or [None]
                else:
                    nseeds = [None]
                for ns in nseeds:
                    s2, n2, d2 = stage2_status(a, fam, ns)
                    s3, n3, d3 = stage3_status(a, fam, ns, n_targets)
                    if s2 != DONE or s3 != DONE:
                        all_done = False
                    lab = fam + (f"/n{ns}" if ns is not None else "")
                    print(f"  {lab:<11} stage2 {s2:<8} {n2}")
                    print(f"  {'':<11} stage3 {s3:<8} {n3}")
                    if args.verbose:
                        print(f"  {'':<11}        {d2}")
                        print(f"  {'':<11}        {d3}")
                    rows.append(dict(cfg=p, arm=a["name"], cseed=cs, family=fam,
                                     nseed=ns, scope=a["scope"], stage1=s1,
                                     stage2=s2, stage2_note=n2,
                                     stage3=s3, stage3_note=n3))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")

    tot = len(rows)
    ok = sum(1 for r in rows if r["stage2"] == DONE and r["stage3"] == DONE)
    print(f"\n{ok}/{tot} cells complete -- {'ALL DONE' if all_done else 'INCOMPLETE'}")
    sys.exit(0 if all_done else 1)


if __name__ == "__main__":
    main()