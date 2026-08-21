#!/usr/bin/env python3
"""
Usage:

    python check_chains.py /gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter  --expected 908

"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="project root(s), each containing seed_s* dirs")
    ap.add_argument("--expected", type=int, default=908)
    ap.add_argument(
        "--seed-glob",
        default="seed_s*",
        help="glob for seed dirs under each root (default seed_s*)",
    )
    ap.add_argument(
        "--chain-subdir",
        default="chains/standard_t_nsf",
        help="path under each seed dir where .dat files live",
    )
    args = ap.parse_args()

    grand_incomplete = []
    for root in args.roots:
        rp = Path(root)
        print(f"\n=== root: {rp} ===")
        if not rp.exists():
            print("  [WARN] root does not exist")
            continue

        seed_dirs = sorted(d for d in rp.glob(args.seed_glob) if d.is_dir())
        if not seed_dirs:
            print(f"  no dirs matching '{args.seed_glob}'")
            continue

        w = max(len(d.name) for d in seed_dirs)
        n_ok = 0
        for sd in seed_dirs:
            cdir = sd / args.chain_subdir
            n = len(list(cdir.glob("*.dat"))) if cdir.exists() else 0
            if n >= args.expected:
                status, ok = "OK", True
            elif not cdir.exists():
                status, ok = "NO CHAINS DIR", False
            elif n == 0:
                status, ok = "EMPTY", False
            else:
                status, ok = f"PARTIAL ({args.expected - n} short)", False
            n_ok += ok
            if not ok:
                grand_incomplete.append(str(sd))
            print(f"  {sd.name:<{w}}  {n:>6}  {status}")
        print(f"  -> {n_ok}/{len(seed_dirs)} seeds complete (>= {args.expected})")

    if grand_incomplete:
        print(f"\n{len(grand_incomplete)} seed(s) need stage-3 (re)run:")
        for s in grand_incomplete:
            print(f"  {s}")
    else:
        print("\nAll seeds complete across all roots.")


if __name__ == "__main__":
    main()
