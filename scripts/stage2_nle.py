#!/usr/bin/env python3
"""Stage 2: train the neural likelihood q(t | theta) on the stage-1 summaries.

  python scripts/stage2_nle.py CONFIG --cseed 43 --nseed 42

Sim-level split (nle.split_seed), t standardised with the training mean/std. Adam;
phase 1: the training rows are cut into nsplit chunks, one epoch per chunk at lr_phase1;
phase 2: nsplit//2 chunks at lr_phase2. The weights with the best chunk validation loss are kept.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sbi.config import load_config, run_dir, set_all_seeds, summaries_path  # noqa: E402
from sbi.data import load_summaries, split_by_sim  # noqa: E402
from sbi.nle import build_nle, save_nle  # noqa: E402

# stage-2 weights are seeded with nseed * 1000 + NSF_SEED_OFFSET (offset kept from the
# original code, where it was derived from the family name "nsf", so results reproduce)
NSF_SEED_OFFSET = 327


def loader(theta, t, bs, shuffle, nw):
    return DataLoader(TensorDataset(torch.tensor(theta), torch.tensor(t)), batch_size=bs,
                      shuffle=shuffle, num_workers=nw, persistent_workers=nw > 0)


def run_pass(model, dl, opt, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for th, t in dl:
            loss = -model.log_prob(th.float(), t.float()).mean()
            if not torch.isfinite(loss):
                continue
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--cseed", type=int, default=0)
    ap.add_argument("--nseed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    nc = cfg["nle"]
    out = run_dir(cfg, args.cseed) / f"nle_n{args.nseed}"
    out.mkdir(parents=True, exist_ok=True)

    theta, t, sim, noise = load_summaries(summaries_path(cfg, args.cseed))
    noise = noise[:, 0] if noise.ndim == 2 else noise
    P, D = theta.shape[1], t.shape[1]
    print(f"stage2 {cfg['name']} c{args.cseed} n{args.nseed}: rows={len(t):,} t_dim={D} n_params={P}",
          flush=True)
    th_tr, t_tr, _, _, th_va, t_va, _, _, train_sims, val_sims = split_by_sim(
        theta, t, sim, noise, nc["val_frac"], nc["split_seed"])
    np.save(out / "train_sims.npy", train_sims)
    np.save(out / "val_sims.npy", val_sims)
    t_mean = t_tr.mean(0).astype(np.float32)
    t_std = (t_tr.std(0) + 1e-8).astype(np.float32)
    t_tr = ((t_tr - t_mean) / t_std).astype(np.float32)
    t_va = ((t_va - t_mean) / t_std).astype(np.float32)
    th_tr, th_va = th_tr.astype(np.float32), th_va.astype(np.float32)

    set_all_seeds(args.nseed * 1000 + NSF_SEED_OFFSET)
    model = build_nle(nc, D, P)
    with open(out / "run_meta.json", "w") as f:
        json.dump({"name": cfg["name"], "cseed": args.cseed, "nseed": args.nseed, **nc}, f, indent=2)

    opt = torch.optim.Adam(model.parameters(), lr=float(nc["lr_phase1"]))
    bs, nw, nsplit = int(nc["batch_size"]), int(nc["num_workers"]), int(nc["nsplit"])
    best, log = float("inf"), []
    for phase, lr, nch in (("p1", float(nc["lr_phase1"]), nsplit), ("p2", float(nc["lr_phase2"]), max(1, nsplit // 2))):
        for g in opt.param_groups:
            g["lr"] = lr
        for j in range(nch):
            a, b = j * len(th_tr) // nch, (j + 1) * len(th_tr) // nch
            va, vb = j * len(th_va) // nch, (j + 1) * len(th_va) // nch
            tr = run_pass(model, loader(th_tr[a:b], t_tr[a:b], bs, True, nw), opt, True)
            vl = run_pass(model, loader(th_va[va:vb], t_va[va:vb], bs, False, nw), opt, False)
            log.append([len(log) + 1, tr, vl])
            np.save(out / "loss_history.npy", np.array(log))
            print(f"[{phase}] chunk {j + 1}/{nch} | train {tr:.4f} | val {vl:.4f}", flush=True)
            if vl < best:
                best = vl
                save_nle(model, str(out), nc, D, P, t_mean, t_std)
    print(f"done. best val NLL {best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
