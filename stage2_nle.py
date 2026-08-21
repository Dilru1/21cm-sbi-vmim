#!/usr/bin/env python3
"""Stage 2: train one or several NLE families q(t | theta) on the SAME
exported summaries, chunked.

  # single family (as before):
  python stage2_nle.py configs/arm.yaml -o nle.model=nsf

  # several families in ONE job, sequentially, sharing data prep:
  python stage2_nle.py configs/arm.yaml --models gmm,maf,nsf
  #   (or set `nle.models: [gmm, maf, nsf]` in the yaml)

  # several families in PARALLEL (one SLURM job each): use
  # slurm/run_nle_families.sh -- true parallelism needs separate GPUs,
  # not one process.

Data loading, sim-split, and t standardization are done ONCE and shared by
every family in the list -- so all families see byte-identical training
data and the identical t_mean/t_std, which is exactly what a fair family
comparison requires. Each family gets a FRESH model, optimizer, and
best-val tracker, and saves to its own {arm}/nle/<model>/ subdir.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import arm_dirs, load_config, split_by_sim
from sbi.nle import build_nle, nle_subdir, save_nle
from sbi.seeding import apply_arm_name, resolve_seeds, set_all_seeds
from sbi.torch_loader import make_loader


def run_pass(model, loader, opt, device, train, tag, log_every=1000):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    tic = time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (th, t) in enumerate(loader):
            th = th.to(device, non_blocking=True).float()
            t = t.to(device, non_blocking=True).float()
            loss = -model.log_prob(th, t).mean()
            if not torch.isfinite(loss):
                print(f"  [WARN] non-finite loss at batch {batch_idx}, skipping", flush=True)
                continue
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.item()
            n += 1
            if train and batch_idx % log_every == 0 and batch_idx > 0:
                dt = time.time() - tic
                tic = time.time()
                eta = (len(loader) - batch_idx) / (log_every / max(dt, 1e-9))
                print(
                    f"  batch {batch_idx}/{len(loader)} | "
                    f"loss {total / max(n, 1):.4f} | eta {eta / 60:.1f} min [{tag}]",
                    flush=True,
                )
    return total / max(n, 1)


def train_family(
    kind,
    nc,
    nle_root,
    device,
    th_tr,
    t_tr,
    th_va,
    t_va,
    t_mean,
    t_std,
    t_dim,
    n_params,
    train_sims,
    val_sims,
    init_seed=0,
    split_seed=42,
):
    """Train ONE family end-to-end (fresh model/opt/best-val), save to its subdir."""
    nc_k = copy.deepcopy(nc)
    nc_k["model"] = kind
    nc_k["init_seed"] = init_seed
    # Re-seed per family: without this, `nsf` gets a different init depending on
    # whether `maf` happened to run before it in the same process, so a family
    # comparison is confounded by RNG-stream position rather than family choice.
    fam_offset = sum(ord(ch) for ch in kind) % 977
    set_all_seeds(init_seed * 1000 + fam_offset, verbose=False)
    kind, model = build_nle(nc_k, t_dim, n_params)
    model.to(device)
    subdir = nle_subdir(nle_root, nc_k)
    print(f"\n===== NLE family '{kind}' -> {subdir} =====", flush=True)
    np.save(os.path.join(subdir, "train_sims.npy"), train_sims)
    np.save(os.path.join(subdir, "val_sims.npy"), val_sims)
    with open(os.path.join(subdir, "run_meta.json"), "w") as f:
        json.dump(
            {"family": kind, "init_seed": init_seed, "split_seed": split_seed, "t_dim": t_dim},
            f,
            indent=2,
        )

    opt = torch.optim.Adam(model.parameters(), lr=float(nc_k["lr_phase1"]))
    pin = device.type == "cuda"
    best_val = float("inf")
    loss_log = []

    nsplit = int(nc_k["nsplit"])
    for pname, lr, nch in [
        ("p1", float(nc_k["lr_phase1"]), nsplit),
        ("p2", float(nc_k["lr_phase2"]), max(1, nsplit // 2)),
    ]:
        for g in opt.param_groups:
            g["lr"] = lr
        for j in range(nch):
            ts, te = j * len(th_tr) // nch, (j + 1) * len(th_tr) // nch
            vs, ve = j * len(th_va) // nch, (j + 1) * len(th_va) // nch
            tl = make_loader(
                th_tr[ts:te], t_tr[ts:te], nc_k["batch_size"], True, nc_k.get("num_workers", 2), pin
            )
            vl = make_loader(
                th_va[vs:ve],
                t_va[vs:ve],
                nc_k["batch_size"],
                False,
                nc_k.get("num_workers", 2),
                pin,
            )
            tr_loss = run_pass(model, tl, opt, device, True, f"{kind} {pname} chunk {j + 1}/{nch}")
            va_loss = run_pass(model, vl, opt, device, False, f"{kind} {pname} val {j + 1}/{nch}")
            loss_log.append([len(loss_log) + 1, tr_loss, va_loss])
            np.save(os.path.join(subdir, "loss_history.npy"), np.array(loss_log))
            print(
                f"[{kind}|{pname}] chunk {j + 1}/{nch} | train {tr_loss:.4f} | val {va_loss:.4f}",
                flush=True,
            )
            if va_loss < best_val:
                best_val = va_loss
                save_nle(model, kind, subdir, nc_k, t_dim, n_params, t_mean, t_std)
                print(f"  saved best ({best_val:.4f}) -> {subdir}", flush=True)

    torch.save(model.state_dict(), os.path.join(subdir, "nle_model_final.pt"))
    print(f"[{kind}] done. best val NLL {best_val:.4f}", flush=True)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument(
        "--models",
        default=None,
        help="comma-separated families to train sequentially in this job, "
        "e.g. gmm,maf,nsf. Overrides yaml nle.models / nle.model.",
    )
    ap.add_argument(
        "--raw-t",
        dest="standardize",
        action="store_false",
        default=None,
        help="train the NLE on RAW t (no per-dim standardization); "
        "outputs go to {arm}/nle/raw_t/<model>/. Default (and "
        "--standardize) uses standardized t -> {arm}/nle/standard_t/<model>/.",
    )
    ap.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        default=None,
        help="force standardized t (this is the default).",
    )
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    nc = cfg["nle"]
    split_seed, init_seed = resolve_seeds(nc)
    # point at the SAME arm tree stage 1 wrote (arm_cnn_s<compressor init_seed>)
    apply_arm_name(cfg)
    dirs = arm_dirs(cfg)
    set_all_seeds(init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # standardize priority: CLI flag > yaml nle.standardize > default True
    if args.standardize is not None:
        nc["standardize"] = bool(args.standardize)
    else:
        nc["standardize"] = bool(nc.get("standardize", True))

    # family list priority: --models flag > yaml nle.models list > yaml nle.model
    if args.models:
        families = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    elif nc.get("models"):
        families = [str(m).lower() for m in nc["models"]]
    else:
        families = [nc.get("model", "gmm").lower()]

    # ---- data prep: ONCE, shared by every family ----
    sd = str(dirs["summaries"])
    theta = np.load(os.path.join(sd, "theta.npy"), mmap_mode="r")
    t = np.load(os.path.join(sd, "t.npy"), mmap_mode="r")
    sim_ids = np.load(os.path.join(sd, "original_sim_ids.npy"), mmap_mode="r")
    noise_ids = np.load(os.path.join(sd, "noise_ids.npy"), mmap_mode="r")
    n_params, t_dim = theta.shape[1], t.shape[1]
    print(
        f"Stage2 arm={cfg['arm_name']} families={families} "
        f"rows={len(theta):,} t_dim={t_dim} n_params={n_params} device={device} "
        f"split_seed={split_seed} init_seed={init_seed} "
        f"seed_subdir={bool(nc.get('seed_subdir', False))}",
        flush=True,
    )

    (th_tr, t_tr, sim_tr, _, th_va, t_va, sim_va, _, train_sims, val_sims) = split_by_sim(
        np.asarray(theta),
        np.asarray(t),
        np.asarray(sim_ids),
        np.asarray(noise_ids)[:, 0] if np.asarray(noise_ids).ndim == 2 else np.asarray(noise_ids),
        nc.get("val_frac", 0.1),
        split_seed,
    )

    if nc["standardize"]:
        t_mean = t_tr.mean(0).astype(np.float32)
        t_std = (t_tr.std(0) + 1e-8).astype(np.float32)
        t_tr = ((t_tr - t_mean) / t_std).astype(np.float32)
        t_va = ((t_va - t_mean) / t_std).astype(np.float32)
        print(
            "[t-scaling] STANDARDIZED: per-dim mean/std removed. "
            "outputs -> {arm}/nle/standard_t/<model>/",
            flush=True,
        )
    else:
        # raw t: save IDENTITY mean/std so stage3's (t_obs - mean)/std is a
        # no-op and the model sees exactly the same coordinates it trained on.
        t_mean = np.zeros(t_dim, np.float32)
        t_std = np.ones(t_dim, np.float32)
        t_tr = t_tr.astype(np.float32)
        t_va = t_va.astype(np.float32)
        print(
            "[t-scaling] RAW t (no standardization; identity mean/std saved). "
            "outputs -> {arm}/nle/raw_t/<model>/",
            flush=True,
        )
    th_tr = th_tr.astype(np.float32)
    th_va = th_va.astype(np.float32)

    # ---- train each family on the identical data ----
    results = {}
    for kind in families:
        results[kind] = train_family(
            kind,
            nc,
            dirs["nle"],
            device,
            th_tr,
            t_tr,
            th_va,
            t_va,
            t_mean,
            t_std,
            t_dim,
            n_params,
            train_sims,
            val_sims,
            init_seed=init_seed,
            split_seed=split_seed,
        )

    scope = "standard_t" if nc["standardize"] else "raw_t"
    print(f"\n===== summary [{scope}] (best val NLL) =====", flush=True)
    for kind, v in results.items():
        print(f"  {kind:6s}  {v:.4f}", flush=True)
    if not nc["standardize"]:
        print(
            "  NOTE: raw-t NLL is in a DIFFERENT coordinate system than "
            "standard-t NLL -- the two are NOT directly comparable as numbers. "
            "Compare via SBC/GV in stage 4, not by this loss.",
            flush=True,
        )


if __name__ == "__main__":
    main()
