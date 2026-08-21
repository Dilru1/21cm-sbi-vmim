#!/usr/bin/env python3
"""Stage 3: SBC MCMC chains for the arm, per NLE family.

  python stage3_mcmc.py configs/arm_cnn_vmim_1.yaml --task-num 0 --task-nb 10
  python stage3_mcmc.py configs/arm_cnn_vmim_1.yaml -o nle.model=nsf ...

Loads the NLE saved by the new stage2 from {arm}/nle/<model>/ (gmm / made /
maf / nsf, chosen by cfg['nle']['model']) and writes chains to
{arm}/chains/<model>/SBC_CHAINS_... so several NLE families trained on the
SAME summaries can be compared side by side in stage 4.

LEGACY fallback: if {arm}/nle/<model>/model_config.json does not exist but
the old {arm}/nle/best_mdn_nle.pt does, the old loading path is used
(requires the original MDNLikelihood class to still exist in sbi.nle) and
chains go to {arm}/chains/gmm_legacy/.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import arm_dirs, load_config, load_sbc_targets, load_source
from sbi.mcmc import run_chain
from sbi.seeding import apply_arm_name

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
torch.set_num_interop_threads(1)


def load_model_new(nle_root, nc, device):
    """New per-family layout: {arm}/nle/<scope>/<model>/ with model_config.json,
    where scope is 'standard_t' or 'raw_t' (nc['standardize'])."""
    import json

    from sbi.nle import load_nle

    kind = nc.get("model", "gmm").lower()
    scope = "standard_t" if bool(nc.get("standardize", True)) else "raw_t"
    base = Path(nle_root) / scope / kind
    # stage2's nle_subdir optionally nests one more level, seed_<init_seed>, when
    # nle.seed_subdir is set (NLE ensembling). Resolve it here so stage3 finds
    # the model in both layouts. If seed_subdir is on, honor the requested seed;
    # otherwise fall back to whatever single seed_* dir exists.
    subdir = base
    if not (subdir / "model_config.json").exists():
        if nc.get("seed_subdir", False):
            iseed = int(nc.get("init_seed", nc.get("seed", 42)))
            cand = base / f"seed_{iseed}"
            if (cand / "model_config.json").exists():
                subdir = cand
        if not (subdir / "model_config.json").exists():
            seed_dirs = sorted(base.glob("seed_*")) if base.is_dir() else []
            hits = [d for d in seed_dirs if (d / "model_config.json").exists()]
            if len(hits) == 1:
                subdir = hits[0]
                print(f"[NLE] using sole seed dir {subdir.name}", flush=True)
            elif len(hits) > 1:
                sys.exit(
                    f"multiple NLE seed dirs under {base}: "
                    f"{[d.name for d in hits]}. Set nle.seed_subdir=true and "
                    f"nle.init_seed=<N> to pick one."
                )
    if not (subdir / "model_config.json").exists():
        return None
    model, t_mean, t_std = load_nle(str(subdir), device)
    with open(subdir / "model_config.json") as f:
        n_params = int(json.load(f)["n_params"])
    print(f"[NLE] scope={scope} family={kind} loaded from {subdir}", flush=True)
    # t_mean/t_std are identity for raw_t, real for standard_t -> the same
    # (t_obs - mean)/std code path is correct for both.
    return model, n_params, t_mean.astype(np.float32), t_std.astype(np.float32), f"{scope}_{kind}"


def load_model_legacy(nle_root, device):
    """Old single-file layout: {arm}/nle/best_mdn_nle.pt (GMM only)."""
    from sbi.nle import MDNLikelihood  # must still exist in sbi/nle.py

    ckpt_path = Path(nle_root) / "best_mdn_nle.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MDNLikelihood(
        int(ckpt["n_params"]),
        int(ckpt["n_summaries"]),
        int(ckpt["n_mix"]),
        int(ckpt["hidden"]),
        ckpt.get("param_style", "semelin"),
        float(ckpt.get("diag_floor", 1e-4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = int(ckpt["n_params"])
    t_mean = t_std = None
    if bool(ckpt.get("t_normalized", False)):
        t_mean = np.load(Path(nle_root) / "t_mean.npy").astype(np.float32)
        t_std = np.load(Path(nle_root) / "t_std.npy").astype(np.float32)
    print(
        f"[NLE] LEGACY gmm loaded from {ckpt_path} "
        f"(style={ckpt.get('param_style')}, t_normalized={t_mean is not None})",
        flush=True,
    )
    return model, n_params, t_mean, t_std, "gmm_legacy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--task-num", type=int, default=0)
    ap.add_argument("--task-nb", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--raw-t",
        dest="standardize",
        action="store_false",
        default=None,
        help="load the NLE trained on RAW t (from {arm}/nle/raw_t/<model>/).",
    )
    ap.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        default=None,
        help="load the standardized-t NLE (default).",
    )
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    # MUST match stage 1/2: tag_arm_with_seed appends _s<init_seed> to arm_name,
    # so the nle/ and chains/ dirs live under e.g. cnn_vmim_noise/n1_s2/. Without
    # this, arm_dirs points at the un-suffixed cnn_vmim_noise/n1/ and the NLE is
    # not found (clean exit) -- or, worse, a stale pre-seeding n1/ dir is loaded.
    apply_arm_name(cfg)
    nc, mc, dirs = cfg["nle"], cfg["mcmc"], arm_dirs(cfg)
    print(f"[stage3] arm={cfg['arm_name']}  nle={dirs['nle']}", flush=True)
    if args.standardize is not None:
        nc["standardize"] = bool(args.standardize)
    else:
        nc["standardize"] = bool(nc.get("standardize", True))
    device = torch.device(args.device)

    loaded = load_model_new(dirs["nle"], nc, device) or load_model_legacy(dirs["nle"], device)
    if loaded is None:
        sys.exit(
            f"no trained NLE found: neither {dirs['nle']}/{nc.get('model', 'gmm')}/"
            f"model_config.json nor {dirs['nle']}/best_mdn_nle.pt exists. "
            f"Run stage2_nle.py first (with the matching -o nle.model=...)."
        )
    model, n_params, t_mean, t_std, kind = loaded

    # chains go into a per-family subdir so families never overwrite each other
    chains_dir = Path(dirs["chains"]) / kind
    chains_dir.mkdir(parents=True, exist_ok=True)

    src = load_source(str(dirs["summaries"]), mmap=True)
    theta_np, t_np, sim_ids, noise_ids = src.theta, src.t, src.sim_ids, src.noise_ids

    rows = []
    for tg in load_sbc_targets(mc["target_path"]):
        r = np.where(sim_ids == tg["sim"])[0]
        if len(r) and tg["offset"] < len(r):
            rows.append(int(r[tg["offset"]]))
    rows = np.array(rows, np.int64)[args.task_num :: args.task_nb]
    print(f"targets this task: {len(rows)} | family={kind} | chains -> {chains_dir}", flush=True)

    for row in rows:
        row, sim = int(row), int(sim_ids[row])
        nv = np.asarray(noise_ids[row])
        tag = str(int(nv)) if nv.ndim == 0 else "_".join(map(str, nv.astype(int).tolist()))
        arm_tag = cfg["arm_name"].replace("/", "_")
        base = chains_dir / f"SBC_CHAINS_{arm_tag}_sim{sim}_row{row}_noise{tag}"

        if base.with_suffix(".dat").exists() and not args.overwrite:
            print("skip", base.name, flush=True)
            continue
        target_t = np.asarray(t_np[row], np.float32)
        if t_mean is not None:
            target_t = ((target_t - t_mean) / t_std).astype(np.float32)
        samples, slogp, accr = run_chain(model, target_t, n_params, device, mc, mc["seed"] + sim)
        if samples.shape[0] == 0:
            print(f"empty sim={sim} row={row}", flush=True)
            continue
        samples.tofile(base.with_suffix(".dat"))
        np.save(f"{base}_logp.npy", slogp)
        np.save(f"{base}_truth.npy", np.asarray(theta_np[row], np.float32))
        np.save(f"{base}_tobs.npy", target_t)
        np.save(
            f"{base}_meta.npy",
            np.array(
                [
                    mc["walkers"],
                    mc["steps"],
                    mc["burnin"],
                    n_params,
                    accr,
                    samples.shape[0],
                    row,
                    sim,
                ],
                np.float32,
            ),
        )
        print(f"saved {base.name} {samples.shape} acc={accr:.3f}", flush=True)


if __name__ == "__main__":
    main()
