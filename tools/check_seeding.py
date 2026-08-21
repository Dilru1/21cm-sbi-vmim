#!/usr/bin/env python3
"""Verify stage-1 seeding WITHOUT training anything.

  python tools/check_seeding.py configs/arm_cnn.yaml
  python tools/check_seeding.py configs/arm_cnn.yaml --seeds 0 1 2 3
  python tools/check_seeding.py configs/arm_mlp.yaml --with-batches 3

What it asserts:

  A. SPLIT INVARIANCE   train/val sims are IDENTICAL across init_seeds.
                        (If this fails, an init_seed is still leaking into a
                        split and your ensemble members see different data.)
  B. INIT DIVERGENCE    compressor + head weights DIFFER across init_seeds.
                        (If this fails, set_all_seeds is called too late, or
                        after the model is constructed, and every "replica" is
                        the same network.)
  C. REPRODUCIBILITY    re-running the SAME init_seed reproduces the SAME
                        weights bit-for-bit.
                        (If this fails you have an unseeded RNG somewhere and
                        your "seed 1" is not re-runnable.)
  D. AUGMENTATION       (--with-batches N) the first N training batches differ
                        across init_seeds and reproduce within a seed.
                        (Catches the missing-worker_init_fn numpy bug: if
                        batches are identical across seeds, augmentation is
                        NOT following init_seed.)

A/B/C need no data files at all for the CNN arm -- the split is recomputed
with the same 4 lines prepare_cube_loaders uses, which skips the expensive
get_stats() pass over every cube. Only --with-batches touches real data.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sbi import load_config, load_source, split_by_sim
from sbi.compressors.heads import build_head
from sbi.seeding import make_generator, resolve_seeds, seed_worker, set_all_seeds

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def digest(arrays):
    """Stable hash of a list of numpy arrays / tensors."""
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:16]


def report(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)
    return ok


# ---------------------------------------------------------------------------
# split: recomputed exactly as each arm does it, but cheaply
# ---------------------------------------------------------------------------
def get_split_sims(cfg, split_seed):
    c = cfg["compressor"]
    if cfg["arm_type"] == "mlp":
        src = load_source(cfg["data"]["summaries_npz"])
        out = split_by_sim(src.t, src.theta, src.sim_ids, src.noise_ids, c["val_frac"], split_seed)
        return out[8], out[9]  # train_sims, val_sims
    # CNN: mirror prepare_cube_loaders lines 135-140 without get_stats()
    sim_ids = np.load(cfg["data"]["sim_ids_path"]).astype(int)
    rng = np.random.default_rng(split_seed)
    idx = np.arange(len(sim_ids))
    rng.shuffle(idx)
    n_val = int(c["val_frac"] * len(idx))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


# ---------------------------------------------------------------------------
# model construction: same code path as stage1_compress.main()
# ---------------------------------------------------------------------------
def build_models(cfg, device):
    c = cfg["compressor"]
    if cfg["arm_type"] == "mlp":
        from sbi.compressors.mlp import SummaryCompressor

        n_input = int(c.get("_n_input_cached", 0)) or None
        if n_input is None:
            n_input = load_source(cfg["data"]["summaries_npz"]).n_summaries
            c["_n_input_cached"] = n_input
        comp = SummaryCompressor(
            n_input=n_input,
            t_dim=cfg["t_dim"],
            dense_layers=tuple(c["dense_layers"]),
            dropout=c["dropout"],
            activation=c["activation"],
            use_resnet=c["use_resnet"],
            n_params=cfg["n_params"],
        ).to(device)
    else:
        mc = c.get("model", "seblock").lower()
        if mc == "conv4d":
            from sbi.compressors.cnn_grn4d_up import Conv4DCompressor as C
        else:
            from sbi.compressors.cnn_seblock_up import ResNet3DCompressor as C
        comp = C(
            t_dim=cfg["t_dim"],
            n_params=cfg["n_params"],
            direct=bool(c.get("direct_regression", False)),
        ).to(device)
    head = build_head(
        c["vmim_head"], cfg["t_dim"], cfg["n_params"], c["n_mix_head"], c["hidden_head"]
    ).to(device)
    return comp, head


def weight_digest(cfg, init_seed, device):
    """Seed -> build -> hash. Mirrors stage1 ordering exactly."""
    set_all_seeds(init_seed, verbose=False)
    comp, head = build_models(cfg, device)
    ws = [p for p in comp.state_dict().values()] + [p for p in head.state_dict().values()]
    return digest(ws)


# ---------------------------------------------------------------------------
# optional: first N training batches (touches real data)
# ---------------------------------------------------------------------------
def batch_digest(cfg, split_seed, init_seed, n_batches):
    from torch.utils.data import DataLoader

    c = cfg["compressor"]
    set_all_seeds(init_seed, verbose=False)

    if cfg["arm_type"] == "mlp":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from stage1_compress import NoiseResampledDataset

        src = load_source(cfg["data"]["summaries_npz"])
        (t_tr, th_tr, sim_tr, _, _, _, _, _, train_sims, _) = split_by_sim(
            src.t, src.theta, src.sim_ids, src.noise_ids, c["val_frac"], split_seed
        )
        ds = NoiseResampledDataset(t_tr, th_tr, sim_tr, train_sims)
        loader = DataLoader(
            ds,
            batch_size=c["batch_size"],
            shuffle=True,
            num_workers=min(2, c["num_workers"]),
            worker_init_fn=seed_worker,
            generator=make_generator(init_seed),
        )
    else:
        from sbi.cubes import prepare_cube_loaders

        print("    (CNN arm: computing cube stats, this is the slow part...)", flush=True)
        loader, _, _ = prepare_cube_loaders(cfg["data"], c, split_seed, c["val_frac"])

    got = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        got.append(batch[0])
    return digest(got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument(
        "--with-batches",
        type=int,
        default=0,
        metavar="N",
        help="also hash the first N training batches (touches real "
        "data; slow on the CNN arm because of get_stats)",
    )
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    c = cfg["compressor"]
    device = torch.device("cpu")  # CPU is enough and avoids cuDNN nondeterminism
    base_split, _ = resolve_seeds(c)

    print(
        f"\narm={cfg['arm_name']} type={cfg['arm_type']} "
        f"split_seed={base_split} testing init_seeds={args.seeds}\n",
        flush=True,
    )
    ok = True

    # ---- A. split invariance -------------------------------------------------
    print("A. split invariance across init_seeds")
    splits = {}
    for s in args.seeds:
        cfg_s = dict(cfg)
        cfg_s["compressor"] = dict(c)
        cfg_s["compressor"]["init_seed"] = s
        ss, _ = resolve_seeds(cfg_s["compressor"])
        tr, va = get_split_sims(cfg_s, ss)
        splits[s] = (digest([tr]), digest([va]), len(tr), len(va))
        print(
            f"    seed {s}: split_seed={ss} train={splits[s][2]} val={splits[s][3]} "
            f"hash={splits[s][0]}",
            flush=True,
        )
    ref = splits[args.seeds[0]][:2]
    ok &= report(
        "all init_seeds give the identical train/val split",
        all(v[:2] == ref for v in splits.values()),
        "if this FAILS, an init_seed is leaking into a split",
    )

    # ---- B/C. init divergence + reproducibility -----------------------------
    print("\nB. weight divergence across init_seeds")
    digs = {}
    for s in args.seeds:
        digs[s] = weight_digest(cfg, s, device)
        print(f"    seed {s}: weights={digs[s]}", flush=True)
    ok &= report(
        "every init_seed gives DIFFERENT weights",
        len(set(digs.values())) == len(args.seeds),
        "if this FAILS, set_all_seeds runs too late / after model build",
    )

    print("\nC. reproducibility of a single init_seed")
    s0 = args.seeds[0]
    again = weight_digest(cfg, s0, device)
    print(f"    seed {s0}: first={digs[s0]}  repeat={again}", flush=True)
    ok &= report(
        f"re-running init_seed={s0} reproduces the same weights",
        again == digs[s0],
        "if this FAILS, an unseeded RNG is in the init path",
    )

    # ---- D. augmentation stream ---------------------------------------------
    if args.with_batches:
        print(f"\nD. first {args.with_batches} training batches")
        bd = {}
        for s in args.seeds[:2]:
            bd[s] = batch_digest(cfg, base_split, s, args.with_batches)
            print(f"    seed {s}: batches={bd[s]}", flush=True)
        a, b = args.seeds[0], args.seeds[1]
        ok &= report(
            "batch/augmentation stream differs across init_seeds",
            bd[a] != bd[b],
            "if this FAILS, worker_init_fn/generator is not wired in",
        )
        rep = batch_digest(cfg, base_split, a, args.with_batches)
        ok &= report(
            f"batch stream for init_seed={a} is reproducible",
            rep == bd[a],
            "shuffle=True without a seeded generator will fail this",
        )

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
