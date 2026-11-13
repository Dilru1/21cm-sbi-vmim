#!/usr/bin/env python3
"""Stage 3: MCMC posterior for every SBC target with the stage-2 likelihood.

  python scripts/stage3_mcmc.py CONFIG --cseed 43 --nseed 42 --task 0 --ntasks 6

Target k of mcmc.target_path is row `offset` of simulation `sim` in the summary table.
Each chain is written as chain_sim{sim}_row{row}.dat (float32 samples, post burn-in,
row = step * walkers + walker) with _logp.npy, _truth.npy, _tobs.npy and _meta.npy
[walkers, steps, burnin, n_params, acceptance, n_samples, row, sim]. Existing chains are skipped.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sbi.config import load_config, run_dir, summaries_path  # noqa: E402
from sbi.data import load_sbc_targets, load_summaries  # noqa: E402
from sbi.mcmc import run_chain  # noqa: E402
from sbi.nle import load_nle  # noqa: E402

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
torch.set_num_interop_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--cseed", type=int, default=0)
    ap.add_argument("--nseed", type=int, required=True)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mc = cfg["mcmc"]
    rd = run_dir(cfg, args.cseed)
    out = rd / f"chains_n{args.nseed}"
    out.mkdir(parents=True, exist_ok=True)
    model, P, t_mean, t_std = load_nle(str(rd / f"nle_n{args.nseed}"))

    theta, t, sim_ids, _ = load_summaries(summaries_path(cfg, args.cseed))
    rows = []
    for tg in load_sbc_targets(mc["target_path"]):
        r = np.where(sim_ids == tg["sim"])[0]
        if len(r) and tg["offset"] < len(r):
            rows.append(int(r[tg["offset"]]))
    rows = np.array(rows, np.int64)[args.task::args.ntasks]
    print(f"stage3 {cfg['name']} c{args.cseed} n{args.nseed}: {len(rows)} targets -> {out}", flush=True)

    for row in rows:
        row, sim = int(row), int(sim_ids[row])
        base = out / f"chain_sim{sim}_row{row}"
        if base.with_suffix(".dat").exists():
            print("skip", base.name, flush=True)
            continue
        t_obs = ((np.asarray(t[row], np.float32) - t_mean) / t_std).astype(np.float32)
        samples, logp, acc = run_chain(model, t_obs, P, torch.device("cpu"), mc, mc["seed"] + sim)
        if samples.shape[0] == 0:
            print(f"empty sim={sim} row={row}", flush=True)
            continue
        np.save(f"{base}_logp.npy", logp)
        np.save(f"{base}_truth.npy", np.asarray(theta[row], np.float32))
        np.save(f"{base}_tobs.npy", t_obs)
        np.save(f"{base}_meta.npy", np.array([mc["walkers"], mc["steps"], mc["burnin"], P, acc,
                                              samples.shape[0], row, sim], np.float32))
        samples.tofile(base.with_suffix(".dat"))      # last: a .dat means the chain is complete
        print(f"saved {base.name} {samples.shape} acc={acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
