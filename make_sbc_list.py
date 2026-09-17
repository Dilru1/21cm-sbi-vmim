#!/usr/bin/env python3
"""Generate a stage4_eval .list from a config + seed grid, verifying as it goes

    python make_sbc_list.py configs/vmim_/arm_cnn_vmim_n1_jit_0.1.yaml --seed.compressor 43,44,45 --seed.nle 42,43 --families nsf --label "cnn vmim (Jit 0.10  n1)" --out sbc_lists/cnn_vmim_jit_0.1_n1.list
    python make_sbc_list.py configs/vmim_/arm_cnn_vmim_n1_jit_0.05.yaml --seed.compressor 43,44,45 --seed.nle 42,43 --families nsf --label "cnn vmim (Jit 0.05 n1)" --out sbc_lists/cnn_vmim_jit_0p05_n1.list

    python make_sbc_list.py configs/vmim_/arm_cnn_vmim_n1_floor0.yaml --seed.compressor 43,44,45 --seed.nle 42,43 --families nsf --label "no jitter no floor" --out sbc_lists/cnn_vmim_nojit_no_floor.list

    python make_sbc_list.py configs/mlp_/arm_mlp_pdf_ps.yaml --seed.compressor 43,44,45 --seed.nle 42,43 --families nsf --label "mlp pdf+ps" --out sbc_lists/mlp_pdf_ps.list

    # several arms into one list
    python make_sbc_list.py configs/vmim_/arm_cnn_vmim_n1.yaml     --label "no jit n1" ... --out x.list
    python make_sbc_list.py configs/vmim_/arm_cnn_vmim_n1_jit.yaml --label "jit n1"    ... --out x.list --append

WHY NOT WRITE IT BY HAND
------------------------
The 4th column of a .list line is a SUBSTRING and stage4_eval refuses when it
matches more than one chains dir. With the stage3 seed patch the dirs are

    chains/standard_t_nsf_seed_42
    chains/standard_t_nsf_seed_43

so the obvious 'standard_t_nsf' matches BOTH and stage 4 exits. Every line must
name the seed in full. This script emits the full name and, more importantly,
CHECKS each directory exists and holds the expected number of *.dat before
writing the line -- a missing arm otherwise shows up as a stage-4 crash after
you have already queued the job.

It also flags the duplicate-chain failure (files != unique sims), which stage 4
does NOT detect: list_chains globs every *.dat and reads the sim id from the
filename, so a stale run left under different row/noise tags is counted twice,
doubling `used` and silently corrupting the calibration.
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml

SIM_RE = re.compile(r"_sim(\d+)_")


def scope_of(cfg):
    nc = cfg.get("nle", {}) or {}
    mode = str(nc.get("t_scaling")
               or ("standard" if nc.get("standardize", True) else "raw")).lower()
    return "standard_t" if mode == "standard" else "raw_t"


def audit(d, expect):
    """(status, note, n_files) for one chains directory."""
    if not d.is_dir():
        return "MISSING", "no such directory", 0
    files = sorted(d.glob("*.dat"))
    if not files:
        return "MISSING", "no *.dat", 0
    sims = {int(m.group(1)) for f in files if (m := SIM_RE.search(f.name))}
    if len(files) != len(sims):
        return "DUPES", f"{len(files)} files but {len(sims)} unique sims", len(files)
    if expect and len(files) < expect:
        return "SHORT", f"{len(files)}/{expect} chains", len(files)
    return "OK", f"{len(files)} chains", len(files)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--seed.compressor", dest="cseeds", default=None,
                    help="comma list; omit for the un-seeded layout")
    ap.add_argument("--seed.nle", dest="nseeds", default=None, help="comma list")
    ap.add_argument("--families", nargs="+", default=["nsf"])
    ap.add_argument("--label", default=None,
                    help="legend prefix; default is the arm_name")
    ap.add_argument("--out", default=None, help="write here (default: stdout)")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--expect", type=int, default=908,
                    help="expected chains per arm (default 908)")
    ap.add_argument("--skip-bad", action="store_true",
                    help="omit MISSING/SHORT/DUPES arms instead of failing")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scope = scope_of(cfg)
    root0 = Path(cfg["scratch_root"])
    arm0 = cfg["arm_name"]
    label0 = args.label or arm0

    cseeds = [int(x) for x in args.cseeds.split(",")] if args.cseeds else [None]
    nseeds = [int(x) for x in args.nseeds.split(",")] if args.nseeds else [None]

    lines, bad = [], []
    print(f"config : {args.config}")
    print(f"arm    : {arm0}   scope={scope}   families={args.families}")
    print(f"grid   : comp={cseeds}  nle={nseeds}\n")
    print(f"{'status':<8} {'label':<34} {'chains dir':<30} note")
    print("-" * 100)

    for cs in cseeds:
        arm = arm0 if cs is None else f"{arm0}_s{cs}"
        root = root0 / arm
        for ns in nseeds:
            # Base tag for the line entry
            tag_base = label0
            if cs is not None: tag_base += f" c{cs}"
            if ns is not None: tag_base += f" n{ns}"

            subs_for_line = []
            arm_bad = False

            for fam in args.families:
                sub = f"{scope}_{fam}" + (f"_seed_{ns}" if ns is not None else "")
                d = root / "chains" / sub
                
                # Extended tag just for terminal printing
                tag_fam = tag_base
                if len(args.families) > 1: tag_fam += f" [{fam}]"

                # fall back to the un-seeded dir so the audit can SAY that
                # stage3 is not separating seeds, rather than just "missing"
                if ns is not None and not d.is_dir():
                    alt = root / "chains" / f"{scope}_{fam}"
                    if alt.is_dir():
                        st, note, n = ("DUPES", "stage3 did not separate NLE seeds "
                                       "-- patch stage3_mcmc.py and re-run", 0)
                        bad.append((tag_fam, str(alt), note))
                        print(f"{st:<8} {tag_fam:<34} {alt.name:<30} {note}")
                        arm_bad = True
                        continue

                st, note, n = audit(d, args.expect)
                print(f"{st:<8} {tag_fam:<34} {sub:<30} {note}")

                if st == "OK":
                    subs_for_line.append(sub)
                else:
                    bad.append((tag_fam, str(d), note))
                    arm_bad = True

            # If all requested families are OK for this seed pair, emit the joined line
            if subs_for_line and not arm_bad:
                # name, path(arm root), n_params(blank), subdir1, subdir2, ...
                joined_subs = ", ".join(subs_for_line)
                lines.append(f"{tag_base}, {root}, , {joined_subs}")

    print()
    if bad and not args.skip_bad:
        print(f"{len(bad)} arm(s) not usable:")
        for t, d, note in bad:
            print(f"  {t:<34} {note}\n    {d}")
        sys.exit("refusing to write an incomplete list (use --skip-bad to omit them)")
    if bad:
        print(f"[--skip-bad] omitted {len(bad)} arm(s)")

    body = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a" if args.append else "w") as f:
            if args.append:
                f.write("\n")
            f.write(body)
        print(f"{'appended' if args.append else 'wrote'} {len(lines)} lines -> {args.out}")
        print(f"\n  python stage4_eval.py --list {args.out} \\\n"
              f"      --out eval_sbc_out/{Path(args.out).stem} --dlogp 10 --filtered-only")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()