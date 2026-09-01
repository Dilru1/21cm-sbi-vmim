#!/usr/bin/env python3
"""
Usage:
    python check_chains.py /gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter --expected 908
"""

import argparse
from pathlib import Path


def get_short_name(subdir_str):
    """Converts 'chains/raw_t_gmm' into 'R_GMM' for compact printing"""
    s = subdir_str.replace("chains/", "")
    s = s.replace("raw_t_", "R_").replace("standard_t_", "S_")
    return s.upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="project root(s)")
    ap.add_argument("--expected", type=int, default=908)
    ap.add_argument("--seed-glob", default="seed_s*")
    ap.add_argument(
        "--chain-subdirs",
        nargs="+",
        default=[
            "chains/raw_t_gmm",
            "chains/raw_t_nsf",
            "chains/standard_t_gmm",
            "chains/standard_t_nsf",
        ],
    )

    args = ap.parse_args()
    grand_incomplete = set()

    for root in args.roots:
        rp = Path(root)
        print(f"\n=== Root: {rp.name} ===")
        if not rp.exists():
            print("  [WARN] root does not exist")
            continue

        seed_dirs = sorted(d for d in rp.glob(args.seed_glob) if d.is_dir())
        if not seed_dirs:
            print(f"  no dirs matching '{args.seed_glob}'")
            continue

        n_ok = 0
        for sd in seed_dirs:
            seed_is_complete = True
            comp_strs = []

            for subdir in args.chain_subdirs:
                cdir = sd / subdir
                short_col = get_short_name(subdir)

                n = len(list(cdir.glob("*.dat"))) if cdir.exists() else 0

                if n >= args.expected:
                    status, ok = "ok", True
                elif not cdir.exists():
                    status, ok = "(NO_DIR)", False
                elif n == 0:
                    status, ok = "(EMPTY)", False
                else:
                    # Shows the actual number of files if it falls short
                    status, ok = f"no({n})", False

                if not ok:
                    seed_is_complete = False

                # Append formatted string for this subdir
                comp_strs.append(f"{short_col}:{status}")

            if seed_is_complete:
                n_ok += 1
            else:
                grand_incomplete.add(str(sd))

            # Print the single compact line for the seed
            print(f"  {sd.name:<9} | " + " | ".join(f"{s:<12}" for s in comp_strs))

        print(
            f"\n  -> {n_ok}/{len(seed_dirs)} seeds fully complete (all subdirs >= {args.expected})"
        )

    if grand_incomplete:
        print(f"\n{len(grand_incomplete)} seed(s) need stage-3 (re)run.")
    else:
        print("\nAll seeds complete across all roots and subdirectories!")


if __name__ == "__main__":
    main()
