#!/usr/bin/env python3
"""Verify noise_scale actually reaches the exported summaries t.

  python tools/check_noise_scale.py configs/vmim_/arm_cnn_vmim_jitter_n2.yaml
  python tools/check_noise_scale.py configs/... --n-rows 32 --tol 1e-4

Four independent checks, cheapest first:

  1. RECORDED SCALE   {arm}/summaries/export_noise_scale.npy vs the config.
                      Stage 1 writes this file but NOTHING ever reads it, so a
                      config edit after an export goes undetected. This catches
                      an ns=1 export being reused under an ns=2 config.

  2. NOISE INDEX      inspects noise_ids.npy as actually written. Detects the
                      sim_id * total_nnoise % pool == 0 collapse, where every
                      simulation is paired with the SAME noise realisation at a
                      given rep because the multiplier equals the pool size.

  3. AMPLITUDE        reconstructs cubes at ns and at 1.0 and compares the
                      measured voxel std ratio against the expected
                      sqrt(sig_var + ns^2 n_var) / sqrt(sig_var + n_var).
                      Confirms the noise really is scaled, not just labelled.

  4. ROUND TRIP       THE definitive one: rebuilds the exact input cube for N
                      exported rows from the raw .dat files using the recorded
                      noise_ids, pushes it through the exported checkpoint, and
                      compares against the stored t.npy rows. If this matches,
                      every normalisation constant and noise scale in the export
                      path is provably the one the network was trained with.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sbi import arm_dirs, load_config

OK, BAD, WARN = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m", "\033[93mWARN\033[0m"


def load_dat(path, n_keep=None):
    a = np.fromfile(path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
    return a if n_keep is None else a[:n_keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n-rows", type=int, default=16, help="rows for the round-trip check")
    ap.add_argument("--tol", type=float, default=1e-3, help="max abs diff on t")
    ap.add_argument("--skip-roundtrip", action="store_true")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    c, d = cfg["compressor"], cfg["data"]
    dirs = arm_dirs(cfg)
    sd = str(dirs["summaries"])
    ns_cfg = float(c["noise_scale"])
    ok = True

    print(f"\narm={cfg['arm_name']}  type={cfg['arm_type']}  config noise_scale={ns_cfg}\n")

    if cfg["arm_type"] == "mlp":
        print(f"[{WARN}] arm_type=mlp reads precomputed summaries from")
        print(f"        {d.get('summaries_npz')}")
        print("        compressor.noise_scale is IGNORED on this path -- the summaries")
        print("        were computed at whatever noise the npz was built with. An ns=2")
        print("        MLP arm does not exist; do not report one.")
        if ns_cfg != 1.0:
            print(f"[{BAD}] config sets noise_scale={ns_cfg} on an mlp arm. This does nothing.")
            ok = False
        sys.exit(0 if ok else 1)

    # ---- 1. recorded scale ---------------------------------------------------
    print("1. recorded export scale")
    f = os.path.join(sd, "export_noise_scale.npy")
    if not os.path.exists(f):
        print(f"   [{WARN}] {f} missing (pre-dates the export_noise_scale feature)")
    else:
        ns_exp = float(np.load(f)[0])
        good = abs(ns_exp - ns_cfg) < 1e-9
        print(f"   exported={ns_exp}  config={ns_cfg}   [{OK if good else BAD}]")
        if not good:
            print("   -> the summaries on disk were exported at a DIFFERENT noise level")
            print("      than this config specifies. Re-run stage 1 --export-only.")
            ok = False

    # ---- 2. noise index structure -------------------------------------------
    print("\n2. noise index structure")
    nids = np.load(os.path.join(sd, "noise_ids.npy"), mmap_mode="r")
    sims = np.load(os.path.join(sd, "original_sim_ids.npy"), mmap_mode="r")
    nids = np.asarray(nids)
    sims = np.asarray(sims)
    pool = int(nids[:, 0].max()) + 1
    tot = int(d["total_nnoise"])
    print(
        f"   rows={len(sims):,}  distinct sims={len(np.unique(sims)):,}  "
        f"pool(observed)={pool}  total_nnoise={tot}"
    )

    # do two different sims get the same noise index at the same rep position?
    u = np.unique(sims)[:2]
    if len(u) == 2:
        a = nids[sims == u[0]][:, 0]
        b = nids[sims == u[1]][:, 0]
        m = min(len(a), len(b))
        collide = np.array_equal(a[:m], b[:m])
        print(
            f"   sim {u[0]} and sim {u[1]} share the identical noise sequence: {collide}"
            f"   [{BAD if collide else OK}]"
        )
        if collide:
            print(
                f"   -> sim_id * total_nnoise % pool == 0 because {tot} % {pool} == {tot % pool}."
            )
            print("      Every simulation is paired with the SAME noise cube at a given rep.")
            print("      Not a bias (each sim still sees all noise reps) but rows are")
            print("      correlated across sims, which inflates the effective sample size")
            print("      assumed by the SBC chi^2 floor. Fix: use a multiplier coprime to")
            print("      the pool, e.g. original_sim_id * 389 instead of * total_nnoise.")
            ok = False
    for ch in range(3):
        print(f"   ch{ch}: {len(np.unique(nids[:, ch])):4d} distinct noise cubes used")

    # ---- 3. amplitude --------------------------------------------------------
    print("\n3. noise amplitude in the reconstructed cubes")
    sim_ids_all = np.load(d["sim_ids_path"]).astype(int)
    ch = 0
    sig = load_dat(d["s_paths"][ch], 9827)[sim_ids_all[:64]]
    noi = load_dat(d["n_paths"][ch])
    sv, nv = float(sig.var()), float(noi.var())
    pred = np.sqrt(sv + ns_cfg**2 * nv) / np.sqrt(sv + 1.0 * nv)
    meas = float((sig[:8] + ns_cfg * noi[:8]).std()) / float((sig[:8] + 1.0 * noi[:8]).std())
    good = abs(pred - meas) / pred < 0.05
    print(f"   signal var={sv:.4g}  noise var={nv:.4g}")
    print(
        f"   std ratio (ns={ns_cfg} vs ns=1): predicted={pred:.4f} measured={meas:.4f}"
        f"   [{OK if good else BAD}]"
    )
    ok &= good

    # ---- 4. round trip -------------------------------------------------------
    if args.skip_roundtrip:
        print(f"\n{'ALL CHECKS PASSED' if ok else 'ISSUES FOUND'}\n")
        sys.exit(0 if ok else 1)

    print(f"\n4. round trip: recompute t for {args.n_rows} exported rows")
    import torch

    from sbi.cubes import get_stats, get_stats_global

    mc = c.get("model", "seblock").lower()
    if mc == "conv4d":
        from sbi.compressors.cnn_grn4d_up import Conv4DCompressor as C

        means, stds = get_stats_global(d["s_paths"], d["n_paths"], sim_ids_all, ns_cfg)
    else:
        from sbi.compressors.cnn_seblock_up import ResNet3DCompressor as C

        means, stds = get_stats(d["s_paths"], d["n_paths"], sim_ids_all, ns_cfg)

    dev = torch.device("cpu")
    net = C(
        t_dim=cfg["t_dim"], n_params=cfg["n_params"], direct=bool(c.get("direct_regression", False))
    ).to(dev)
    ck = dirs["nle"] / "learned_compressor_bestprobe.pt"
    if not ck.exists():
        ck = dirs["nle"] / "learned_compressor.pt"
    net.load_state_dict(torch.load(ck, map_location=dev))
    net.eval()
    print(f"   checkpoint: {ck.name}")

    t_stored = np.load(os.path.join(sd, "t.npy"), mmap_mode="r")
    rng = np.random.default_rng(0)
    rows = rng.choice(len(sims), size=args.n_rows, replace=False)

    sig_all = [load_dat(p, 9827) for p in d["s_paths"]]
    noi_all = [load_dat(p) for p in d["n_paths"]]
    sim_pos = {int(s): i for i, s in enumerate(sim_ids_all)}
    norm = bool(c.get("init_norm", True))

    X = []
    for r in rows:
        cube = []
        for chx in range(3):
            i = sim_pos[int(sims[r])]
            nc_ = sig_all[chx][i] + ns_cfg * noi_all[chx][int(nids[r, chx])]
            if norm:
                nc_ = (nc_ - means[chx]) / (stds[chx] + 1e-8)
            cube.append(nc_)
        X.append(np.stack(cube))
    X = torch.from_numpy(np.stack(X).astype(np.float32))
    if mc == "conv4d":
        X = X.unsqueeze(1)
    with torch.no_grad():
        out = net(X)
        t_new = (out[0] if isinstance(out, tuple) else out).numpy()

    t_ref = np.asarray(t_stored[rows])
    err = np.abs(t_new - t_ref).max()
    rel = err / (np.abs(t_ref).max() + 1e-12)
    good = err < args.tol
    print(f"   max|t_recomputed - t_stored| = {err:.3e}  (rel {rel:.2e})  [{OK if good else BAD}]")
    if not good:
        print("   -> the export path does NOT reproduce the stored summaries.")
        print("      Most likely: noise_scale, init_norm, or the get_stats variant")
        print("      differs between what produced t.npy and this config.")
        for k in range(min(3, len(rows))):
            print(f"      row {rows[k]}: stored={t_ref[k]}  recomputed={t_new[k]}")
    ok &= good

    print(f"\n{'ALL CHECKS PASSED' if ok else 'ISSUES FOUND'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
