#!/usr/bin/env python3
"""Rebuild and verify the masked param / sim-id files used by the CNN arm.

These two arrays were lost and must be regenerated from the Loreli II source of
truth. They are PER-SIMULATION (length = #available sims, NOT x nb_noise) and
share one order:

    original_sim_ids_masked_from_original.npy : int   (n_avail,)
        original 0..9826 index of each kept simulation, ascending
    astro_params_masked_from_original.npy     : float (n_avail, nb_astro_param)
        RAW astro params (param.dat column order) of those same sims, aligned
        row-for-row to the sim-id array.

The availability mask reproduces the original snippet EXACTLY:
  - a sim is available iff its clean cube is non-all-zero at EVERY redshift,
  - AND sims in the manual bad range [480:500) are forced unavailable.

cubes.py does  clean[:9827][sim_ids]  and  params_norm[idx] ,  so
  astro_params[k]  must equal  param.dat line (sim_ids[k]) ,
which is what `build` writes and `verify` checks.

Usage (on the cluster, where the data lives):
  python tools/build_masked_params.py build \
      --param-file  /data/ddehiwalage-don/data/param.dat \
      --clean-glob  "/data/ddehiwalage-don/data/dtb_data/clean_cubes_z={z}.dat" \
      --redshifts 8.18 10.32 12.06 \
      --out-dir    /data/ddehiwalage-don/data

  python tools/build_masked_params.py verify \
      --param-file /data/ddehiwalage-don/data/param.dat \
      --clean-glob "/data/ddehiwalage-don/data/dtb_data/clean_cubes_z={z}.dat" \
      --redshifts 8.18 10.32 12.06 \
      --out-dir   /data/ddehiwalage-don/data

The two filenames written/read are the ones the configs point at:
  astro_params_masked_from_original.npy
  original_sim_ids_masked_from_original.npy
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np

NB_SIMU_TOTAL = 9827
VOXELS = 32 ** 3
PARAMS_NAME = "astro_params_masked_from_original.npy"
SIMIDS_NAME = "original_sim_ids_masked_from_original.npy"


def read_params(param_file, nb_astro_param, col_offset):
    """One row per sim, in file order. column col_offset..col_offset+nb_astro_param.
    Returns (params_full [n_lines, nb_astro_param], label_col [n_lines])."""
    rows, labels = [], []
    with open(param_file) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            labels.append(p[0])
            rows.append([float(p[col_offset + k]) for k in range(nb_astro_param)])
    return np.asarray(rows, np.float64), labels


def availability_mask(clean_paths, n_total, manual_bad):
    """Reproduce the original flagging: nonzero at every redshift, then force the
    manual bad slice False. Vectorized equivalent of the original per-sim loop."""
    avail = np.ones(n_total, bool)
    per_z = {}
    for path in clean_paths:
        clean = np.fromfile(path, dtype=np.float64).reshape(-1, VOXELS)
        if clean.shape[0] < n_total:
            raise ValueError(f"{path}: only {clean.shape[0]} cubes, need {n_total}")
        clean = clean[:n_total]
        nonzero = np.any(clean != 0.0, axis=1)          # original: count_nonzero(...)>0
        per_z[path] = int(nonzero.sum())
        avail &= nonzero
        del clean
    forced = 0
    if manual_bad is not None:
        lo, hi = manual_bad
        forced = int(avail[lo:hi].sum())               # how many we are turning off
        avail[lo:hi] = False
    return avail, per_z, forced


def build(args):
    clean_paths = [args.clean_glob.format(z=z) for z in args.redshifts]
    for p in [args.param_file, *clean_paths]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    params_full, labels = read_params(args.param_file, args.nb_astro_param, args.col_offset)
    if params_full.shape[0] != NB_SIMU_TOTAL:
        print(f"[WARN] param.dat has {params_full.shape[0]} lines, expected {NB_SIMU_TOTAL}")
    n_total = min(params_full.shape[0], NB_SIMU_TOTAL)

    manual = None if args.no_manual_flag else (args.manual_bad_lo, args.manual_bad_hi)
    avail, per_z, forced = availability_mask(clean_paths, n_total, manual)

    sim_ids = np.where(avail)[0].astype(np.int64)             # original indices, ascending
    astro = params_full[sim_ids].astype(np.float64)           # aligned raw params

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / SIMIDS_NAME, sim_ids)
    np.save(out / PARAMS_NAME, astro)

    report = {
        "n_total": int(n_total),
        "n_available": int(avail.sum()),
        "n_flagged_out": int(n_total - avail.sum()),
        "nonzero_per_redshift": per_z,
        "manual_bad_range": None if manual is None else list(manual),
        "manual_forced_off": forced,
        "label_col_first5": labels[:5],
        "label_col_last5": labels[-5:],
        "sim_ids_shape": list(sim_ids.shape),
        "astro_shape": list(astro.shape),
        "wrote": [str(out / SIMIDS_NAME), str(out / PARAMS_NAME)],
    }
    print(json.dumps(report, indent=2))
    # Quick sanity hints
    seq = labels[:1] == ["0"] or labels[:1] == ["1"]
    if not _labels_monotone(labels):
        print("[WARN] param.dat first column is NOT a clean 0,1,2,... sequence; "
              "the file-order==sim-order assumption from the original code relies on "
              "this. Inspect label_col_first5/last5 above.")
    if abs(int(avail.sum()) - 9120) > 50:
        print(f"[NOTE] available={int(avail.sum())}, you expected ~9120. "
              "If far off, check manual_bad range and that all three clean files are present.")
    return report


def _labels_monotone(labels):
    try:
        vals = [int(float(x)) for x in labels]
    except ValueError:
        return False
    return all(vals[i] + 1 == vals[i + 1] for i in range(len(vals) - 1))


def verify(args):
    out = Path(args.out_dir)
    sip = out / SIMIDS_NAME; pap = out / PARAMS_NAME
    if not sip.exists() or not pap.exists():
        raise FileNotFoundError(f"missing {sip} or {pap}; run `build` first")
    sim_ids = np.load(sip); astro = np.load(pap)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("Structural checks:")
    check("sim_ids is 1-D int", sim_ids.ndim == 1 and np.issubdtype(sim_ids.dtype, np.integer))
    check("astro is 2-D", astro.ndim == 2)
    check("lengths match (per-sim aligned)", len(sim_ids) == astro.shape[0])
    check("astro has nb_astro_param cols", astro.shape[1] == args.nb_astro_param)
    check("sim_ids ascending & unique", np.all(np.diff(sim_ids) > 0))
    check("sim_ids within [0,9827)", sim_ids.min() >= 0 and sim_ids.max() < NB_SIMU_TOTAL)
    check("not per-sample (len << 9827*nb_noise)", len(sim_ids) <= NB_SIMU_TOTAL)
    if args.manual_bad_lo is not None and not args.no_manual_flag:
        bad = set(range(args.manual_bad_lo, args.manual_bad_hi))
        check("manual bad range excluded", not (set(sim_ids.tolist()) & bad))

    # Content cross-check against the real source files, if available
    if Path(args.param_file).exists():
        params_full, labels = read_params(args.param_file, args.nb_astro_param, args.col_offset)
        if astro.shape[0] and sim_ids.max() < params_full.shape[0]:
            recon = params_full[sim_ids]
            check("astro rows == param.dat[sim_ids] (alignment)",
                  np.allclose(recon, astro, atol=1e-6, rtol=0))
        # rebuild the mask and compare the kept set exactly
        clean_paths = [args.clean_glob.format(z=z) for z in args.redshifts]
        if all(Path(p).exists() for p in clean_paths):
            manual = None if args.no_manual_flag else (args.manual_bad_lo, args.manual_bad_hi)
            avail, _, _ = availability_mask(clean_paths, min(params_full.shape[0], NB_SIMU_TOTAL), manual)
            check("sim_ids == recomputed availability mask",
                  np.array_equal(sim_ids, np.where(avail)[0]))
        else:
            print("  [skip] clean cubes not all present; skipped mask re-derivation")
    else:
        print("  [skip] param.dat not present; skipped content cross-check")

    print(f"\nOverall: {'ALL PASS' if ok else 'FAILURES PRESENT'}  "
          f"(n_available={len(sim_ids)}, astro={astro.shape})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--param-file", required=True)
        s.add_argument("--clean-glob", required=True,
                       help='template with {z}, e.g. ".../clean_cubes_z={z}.dat"')
        s.add_argument("--redshifts", nargs="+", required=True)
        s.add_argument("--out-dir", required=True)
        s.add_argument("--nb-astro-param", type=int, default=5)
        s.add_argument("--col-offset", type=int, default=1,
                       help="first param column in param.dat (orig code used 1; col 0 is the label)")
        s.add_argument("--manual-bad-lo", type=int, default=480)
        s.add_argument("--manual-bad-hi", type=int, default=500)
        s.add_argument("--no-manual-flag", action="store_true",
                       help="do NOT force [480:500) off (only if your data already excludes them)")
    args = ap.parse_args()
    (build if args.cmd == "build" else verify)(args)


if __name__ == "__main__":
    main()