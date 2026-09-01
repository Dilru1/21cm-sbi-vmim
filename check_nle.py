#!/usr/bin/env python3
"""
Usage:
    python check_nle.py /gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter --expected-chunks 15
"""

import argparse
from pathlib import Path
import numpy as np


def get_short_name(subdir_str):
    """Converts 'nle/raw_t/gmm' into 'R_GMM' for compact printing"""
    s = subdir_str.replace("nle/", "")
    s = s.replace("raw_t", "R").replace("standard_t", "S")
    s = s.replace("/", "_").upper()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="project root(s)")
    ap.add_argument("--expected-chunks", type=int, default=15)
    ap.add_argument("--seed-glob", default="seed_s*")
    ap.add_argument(
        "--nle-subdirs",
        nargs="+",
        default=[
            "nle/raw_t/gmm",
            "nle/raw_t/maf",
            "nle/raw_t/nsf",
            "nle/standard_t/gmm",
            "nle/standard_t/maf",
            "nle/standard_t/nsf",
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

            for subdir in args.nle_subdirs:
                cdir = sd / subdir
                short_col = get_short_name(subdir)

                model_files = list(cdir.glob("*nle_model_final.pt"))
                loss_file = cdir / "loss_history.npy"

                if not cdir.exists():
                    status, ok = "(NO_DIR)", False
                elif not model_files:
                    status, ok = "(NO_MOD)", False
                elif not loss_file.exists():
                    status, ok = "(NO_LOS)", False
                else:
                    try:
                        loss_data = np.load(loss_file)
                        max_chunk = int(np.max(loss_data[:, 0]))

                        if max_chunk >= args.expected_chunks:
                            status, ok = "ok", True
                        else:
                            status, ok = f"({max_chunk}ch)", False
                    except Exception:
                        status, ok = "(CORRUPT)", False

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
            f"\n  -> {n_ok}/{len(seed_dirs)} seeds fully trained (all subdirs reached {args.expected_chunks} chunks)"
        )

    if grand_incomplete:
        print(f"\n{len(grand_incomplete)} seed(s) need training to resume/finish.")
    else:
        print("\nAll seeds successfully trained!")


if __name__ == "__main__":
    main()
