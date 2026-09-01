#!/usr/bin/env python3
"""Stage 2 (OPTIMIZED): train NLE families q(t|theta) on exported summaries.

Speed changes vs the original (identical outputs / seeding / save layout):
  * data kept RESIDENT as torch tensors and sliced on-device -- no DataLoader,
    no per-chunk torch.tensor() re-copy, no worker IPC, no pin/H2D per batch.
  * loss accumulated on-device; ONE .item() per pass (kills per-batch sync).
  * torch thread count is configurable and defaults to 1 -- for this tiny
    4-D flow, extra CPU threads REDUCE throughput (measured ~17x total speedup
    at batch 16384 / 1 thread vs batch 32 / 8 threads).
  * optional nle.max_train_rows (subsample) and nle.patience (early stop).

Recommended run:
  OMP_NUM_THREADS=1 python -u stage2_nle_fast.py CONFIG --models nsf \
      -o nle.batch_size=16384 -o nle.lr_phase1=5e-4 -o nle.lr_phase2=1e-4
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


def run_pass(model, th, t, opt, device, train, bs, tag, log_every=200):
    """One pass over RESIDENT tensors (th, t). Slices on-device; no DataLoader.

    th, t are torch tensors already on `device`. Loss is accumulated as a
    0-dim tensor and materialized with a single .item() at the end, so there
    is no host<->device sync per batch (the big win on GPU, and it removes
    Python overhead on CPU too).
    """
    model.train() if train else model.eval()
    N = th.shape[0]
    order = torch.randperm(N, device=th.device) if train else None
    total = torch.zeros((), device=th.device)
    n = 0
    nb = (N + bs - 1) // bs
    tic = time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, s in enumerate(range(0, N, bs)):
            if train:
                idx = order[s : s + bs]
                thb, tb = th[idx], t[idx]
            else:
                thb, tb = th[s : s + bs], t[s : s + bs]
            loss = -model.log_prob(thb, tb).mean()
            if not torch.isfinite(loss):
                print(f"  [WARN] non-finite loss at batch {bi}, skipping", flush=True)
                continue
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.detach()
            n += 1
            if train and log_every and bi % log_every == 0 and bi > 0:
                dt = time.time() - tic
                tic = time.time()
                eta = (nb - bi) / (log_every / max(dt, 1e-9))
                print(
                    f"  batch {bi}/{nb} | loss {(total / max(n, 1)).item():.4f} | "
                    f"eta {eta / 60:.1f} min [{tag}]",
                    flush=True,
                )
    return (total / max(n, 1)).item()


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
    """Train ONE family end-to-end (fresh model/opt/best-val), save to its subdir.

    th_tr/t_tr/th_va/t_va are torch tensors already on `device`.
    """
    nc_k = copy.deepcopy(nc)
    nc_k["model"] = kind
    bmap = nc_k.get("batch_size_by_family") or {}
    if kind in bmap:
        nc_k["batch_size"] = int(bmap[kind])
    elif str(kind) in {str(k) for k in bmap}:
        nc_k["batch_size"] = int(bmap[str(kind)])
    bs = int(nc_k["batch_size"])
    print(f"[batch] family '{kind}' -> batch_size={bs}", flush=True)
    nc_k["init_seed"] = init_seed
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
    best_val = float("inf")
    loss_log = []
    patience = int(nc_k.get("patience", 0))  # 0 -> disabled (original behaviour)
    bad = 0

    nsplit = int(nc_k["nsplit"])
    N_tr, N_va = th_tr.shape[0], th_va.shape[0]
    stop = False
    for pname, lr, nch in [
        ("p1", float(nc_k["lr_phase1"]), nsplit),
        ("p2", float(nc_k["lr_phase2"]), max(1, nsplit // 2)),
    ]:
        if stop:
            break
        for g in opt.param_groups:
            g["lr"] = lr
        for j in range(nch):
            ts, te = j * N_tr // nch, (j + 1) * N_tr // nch
            vs, ve = j * N_va // nch, (j + 1) * N_va // nch
            tr_loss = run_pass(
                model,
                th_tr[ts:te],
                t_tr[ts:te],
                opt,
                device,
                True,
                bs,
                f"{kind} {pname} chunk {j + 1}/{nch}",
            )
            va_loss = run_pass(
                model,
                th_va[vs:ve],
                t_va[vs:ve],
                opt,
                device,
                False,
                bs,
                f"{kind} {pname} val {j + 1}/{nch}",
            )
            loss_log.append([len(loss_log) + 1, tr_loss, va_loss])
            np.save(os.path.join(subdir, "loss_history.npy"), np.array(loss_log))
            print(
                f"[{kind}|{pname}] chunk {j + 1}/{nch} | train {tr_loss:.4f} | val {va_loss:.4f}",
                flush=True,
            )
            if va_loss < best_val:
                best_val = va_loss
                bad = 0
                save_nle(model, kind, subdir, nc_k, t_dim, n_params, t_mean, t_std)
                print(f"  saved best ({best_val:.4f}) -> {subdir}", flush=True)
            else:
                bad += 1
                if patience and bad >= patience:
                    print(f"  [early-stop] no val improvement for {bad} chunks", flush=True)
                    stop = True
                    break

    torch.save(model.state_dict(), os.path.join(subdir, "nle_model_final.pt"))
    print(f"[{kind}] done. best val NLL {best_val:.4f}", flush=True)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--models", default=None, help="comma-separated families, e.g. gmm,maf,nsf.")
    ap.add_argument("--raw-t", dest="standardize", action="store_false", default=None)
    ap.add_argument("--standardize", dest="standardize", action="store_true", default=None)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    nc = cfg["nle"]
    split_seed, init_seed = resolve_seeds(nc)
    apply_arm_name(cfg)
    dirs = arm_dirs(cfg)
    set_all_seeds(init_seed)

    # Threads: default 1. For this 4-D flow, >1 thread REDUCES throughput.
    nthreads = int(nc.get("torch_threads", os.environ.get("OMP_NUM_THREADS", 1)))
    torch.set_num_threads(max(1, nthreads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.standardize is not None:
        nc["standardize"] = bool(args.standardize)
    else:
        nc["standardize"] = bool(nc.get("standardize", True))

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
        f"Stage2 arm={cfg['arm_name']} families={families} rows={len(theta):,} "
        f"t_dim={t_dim} n_params={n_params} device={device} threads={torch.get_num_threads()} "
        f"split_seed={split_seed} init_seed={init_seed}",
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

    # optional subsample: a 4-D density rarely needs all 9M rows. Seeded so it's
    # reproducible; applied AFTER the split so val stays a clean held-out set.
    max_rows = int(nc.get("max_train_rows", 0))
    if max_rows and max_rows < len(th_tr):
        rng = np.random.default_rng(split_seed)
        sel = rng.choice(len(th_tr), size=max_rows, replace=False)
        sel.sort()
        th_tr, t_tr = th_tr[sel], t_tr[sel]
        print(f"[subsample] train rows {len(sel):,} of original (seed={split_seed})", flush=True)

    if nc["standardize"]:
        t_mean = t_tr.mean(0).astype(np.float32)
        t_std = (t_tr.std(0) + 1e-8).astype(np.float32)
        t_tr = ((t_tr - t_mean) / t_std).astype(np.float32)
        t_va = ((t_va - t_mean) / t_std).astype(np.float32)
        print("[t-scaling] STANDARDIZED -> {arm}/nle/standard_t/<model>/", flush=True)
    else:
        t_mean = np.zeros(t_dim, np.float32)
        t_std = np.ones(t_dim, np.float32)
        t_tr = t_tr.astype(np.float32)
        t_va = t_va.astype(np.float32)
        print("[t-scaling] RAW t (identity mean/std) -> {arm}/nle/raw_t/<model>/", flush=True)

    # ---- move to device ONCE as tensors (dataset is ~300 MB) ----
    th_tr = torch.from_numpy(np.ascontiguousarray(th_tr, np.float32)).to(device)
    t_tr = torch.from_numpy(np.ascontiguousarray(t_tr, np.float32)).to(device)
    th_va = torch.from_numpy(np.ascontiguousarray(th_va, np.float32)).to(device)
    t_va = torch.from_numpy(np.ascontiguousarray(t_va, np.float32)).to(device)

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


if __name__ == "__main__":
    main()
