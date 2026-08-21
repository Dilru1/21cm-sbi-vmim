#!/usr/bin/env python3
"""Check stage-3 completion: for each config, count the .dat SBC chain files in
its chains dir and flag any that are short of the expected count (default 908).

Prints a table and, at the end, the exact list of config paths whose stage-3 is
incomplete -- so you know which to (re)run.

Usage:
    python check_stage3.py configs/**/*.yaml
    python check_stage3.py $(grep -rl "arm_type: cnn" configs)   # subset
    python check_stage3.py --expected 908 --chain-subdir standard_t_nsf configs/*.yaml

--chain-subdir : extra nesting under the arm's chains/ dir, if your pipeline
                 writes into e.g. chains/standard_t_nsf/ rather than chains/ .
                 Leave blank to look directly in chains/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from sbi import load_config, arm_dirs
    HAVE_SBI = True
except Exception:
    HAVE_SBI = False


def chains_dir_for(cfg_path, chain_subdir):
    """Resolve the chains dir for a config the same way the pipeline does."""
    cfg = load_config(cfg_path)
    dirs = arm_dirs(cfg)                      # creates {scratch_root}/{arm_name}/chains
    cdir = Path(dirs["chains"])
    if chain_subdir:
        cdir = cdir / chain_subdir
    return cfg["arm_name"], cdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--expected", type=int, default=908)
    ap.add_argument("--chain-subdir", default="",
                    help="extra nesting under chains/ (e.g. standard_t_nsf)")
    args = ap.parse_args()

    if not HAVE_SBI:
        sys.exit("could not import sbi (load_config/arm_dirs); run from the project root")

    rows, incomplete = [], []
    for cf in args.configs:
        try:
            arm, cdir = chains_dir_for(cf, args.chain_subdir)
        except Exception as e:
            rows.append((cf, "?", -1, f"CONFIG ERROR: {e}"))
            incomplete.append(cf)
            continue

        n = len(list(cdir.glob("*.dat"))) if cdir.exists() else 0
        if n >= args.expected:
            status = "OK"
        elif n == 0:
            status = "MISSING (no chains dir)" if not cdir.exists() else "MISSING (empty)"
            incomplete.append(cf)
        else:
            status = f"PARTIAL ({args.expected - n} short)"
            incomplete.append(cf)
        rows.append((cf, arm, n, status))

    # table
    w = max((len(r[0]) for r in rows), default=20)
    print(f"{'config':<{w}}  {'n_dat':>6}  status")
    print("-" * (w + 6 + 22))
    for cf, arm, n, status in rows:
        ncol = "-" if n < 0 else str(n)
        print(f"{cf:<{w}}  {ncol:>6}  {status}")

    # actionable summary
    print()
    n_ok = sum(1 for _, _, n, s in rows if s == "OK")
    print(f"{n_ok}/{len(rows)} configs complete (>= {args.expected} chains).")
    if incomplete:
        print(f"\n{len(incomplete)} need stage-3 (re)run:")
        for cf in incomplete:
            print(f"  {cf}")
        # ready-to-paste list
        print("\n# paste-able:")
        print("INCOMPLETE=(")
        for cf in incomplete:
            print(f"    {cf}")
        print(")")
    else:
        print("All configs complete.")


if __name__ == "__main__":
    main()