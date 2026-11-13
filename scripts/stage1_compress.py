#!/usr/bin/env python3
"""Stage 1: train the compressor and export t for every row of the training table.

  python scripts/stage1_compress.py CONFIG --cseed 43                  # train + export
  python scripts/stage1_compress.py CONFIG --cseed 43 --train-only     # CNN: export separately
  python scripts/stage1_compress.py CONFIG --cseed 43 --export-only --shard 0 --nshard 5

compressor.type: mlp  (data.summaries_npz, every noise realisation already in the table)
                 cnn  (data cubes; export = every sim x data.n_noise deterministic noise draws)
The CNN export can be split over the noise axis (--shard/--nshard -> summaries/part_*);
scripts/merge_shards.py joins them.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sbi.compressors import CNNCompressor, GMMHead, MLPCompressor  # noqa: E402
from sbi.config import load_config, run_dir, set_all_seeds  # noqa: E402
from sbi.data import load_summaries, split_by_sim  # noqa: E402
from sbi.train import train  # noqa: E402


class ResampledNoise(Dataset):
    """One item per simulation; each access picks one of its noise realisations at random."""

    def __init__(self, t, theta, sim_ids, sims, n_params):
        self.groups = []
        for s in sims:
            m = sim_ids == s
            self.groups.append((theta[m][0], t[m]))
        self.n_params = n_params

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        th, block = self.groups[i]
        j = np.random.randint(block.shape[0])
        return (torch.from_numpy(block[j].astype(np.float32)),
                torch.from_numpy(th[: self.n_params].astype(np.float32)))


def seed_worker(worker_id):
    import random
    base = torch.initial_seed() % 2 ** 32
    np.random.seed((base + worker_id) % 2 ** 32)
    random.seed(base + worker_id)


def export(model, batches, n_rows, n_params, t_dim, n_ch, device, out):
    os.makedirs(out, exist_ok=True)
    mm = lambda name, dt, shape: np.lib.format.open_memmap(os.path.join(out, name), "w+", dt, shape)
    th_mm, t_mm = mm("theta.npy", np.float32, (n_rows, n_params)), mm("t.npy", np.float32, (n_rows, t_dim))
    sim_mm, nz_mm = mm("original_sim_ids.npy", np.int64, (n_rows,)), mm("noise_ids.npy", np.int64, (n_rows, n_ch))
    model.eval()
    off = 0
    with torch.no_grad():
        for bi, (x, theta, sim, nids) in enumerate(batches):
            k = x.shape[0]
            th_mm[off:off + k] = np.asarray(theta)[:, :n_params]
            t_mm[off:off + k] = model(x.to(device))[0].float().cpu().numpy()
            sim_mm[off:off + k] = np.asarray(sim)
            nz_mm[off:off + k] = np.asarray(nids).reshape(k, -1)
            off += k
            if bi % 100 == 0:
                print(f"  export {off:,}/{n_rows:,}", flush=True)
    for m in (th_mm, t_mm, sim_mm, nz_mm):
        m.flush()
    print(f"exported -> {out}  t {t_mm.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--cseed", type=int, required=True, help="compressor seed (weights, batches, noise draws)")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshard", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c, P = cfg["compressor"], int(cfg["n_params"])
    rd = run_dir(cfg, args.cseed)
    cdir, sdir = rd / "compressor", rd / "summaries"
    cdir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.cseed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direct = c["objective"] == "mse"
    print(f"stage1 {cfg['name']} c{args.cseed} type={c['type']} objective={c['objective']} "
          f"t_dim={c['t_dim']} device={device}", flush=True)

    if c["type"] == "mlp":
        theta, t, sim, noise = load_summaries(cfg["data"]["summaries_npz"])
        (t_tr, th_tr, sim_tr, _, t_va, th_va, sim_va, _, train_sims, val_sims) = split_by_sim(
            t, theta, sim, noise, c["val_frac"], c["split_seed"])
        np.save(cdir / "train_sims.npy", train_sims)
        np.save(cdir / "val_sims.npy", val_sims)
        model = MLPCompressor(t.shape[1], c["t_dim"], P, c["dense_layers"], c["dropout"], direct).to(device)
        g = torch.Generator()
        g.manual_seed(args.cseed)
        kw = dict(batch_size=c["batch_size"], num_workers=c["num_workers"],
                  pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
        tl = DataLoader(ResampledNoise(t_tr, th_tr, sim_tr, train_sims, P), shuffle=True, generator=g, **kw)
        vl = DataLoader(ResampledNoise(t_va, th_va, sim_va, val_sims, P), shuffle=False, **kw)
    elif c["type"] == "cnn":
        from sbi.cubes import ExportTable, cube_loaders
        tl, vl, base = cube_loaders(cfg, c["split_seed"])
        model = CNNCompressor(len(cfg["data"]["signal_paths"]), c["t_dim"], P, direct).to(device)
    else:
        sys.exit(f"compressor.type={c['type']!r}: nothing to do in stage 1")

    if not args.export_only:
        head = None
        if c["objective"] == "vmim":
            head = GMMHead(c["t_dim"], P, c["n_mix_head"], c["hidden_head"], c["sigma_floor"]).to(device)
        with open(cdir / "run_meta.json", "w") as f:
            json.dump({"name": cfg["name"], "cseed": args.cseed, **c}, f, indent=2)
        train(model, head, tl, vl, device, str(cdir), c, P)
        if args.train_only:
            return

    model.load_state_dict(torch.load(cdir / "compressor.pt", map_location=device))
    if c["type"] == "mlp":
        idx = torch.arange(len(t))
        loader = DataLoader(TensorDataset(torch.from_numpy(t), idx), batch_size=c["save_batch_size"])
        noise2 = noise.reshape(len(noise), -1)
        batches = ((xb, theta[i.numpy()], sim[i.numpy()], noise2[i.numpy()]) for xb, i in loader)
        export(model, batches, len(t), P, c["t_dim"], noise2.shape[1], device, str(sdir))
    else:
        n_noise = int(cfg["data"]["n_noise"])
        start, end, out = 0, n_noise, sdir
        if args.nshard:
            per = n_noise // args.nshard
            start = args.shard * per
            end = n_noise if args.shard == args.nshard - 1 else start + per
            out = sdir / f"part_{start:05d}_{end:05d}"
        ds = ExportTable(base, n_noise, start, end)
        nw = min(8, max(0, len(os.sched_getaffinity(0)) - 1))
        loader = DataLoader(ds, batch_size=c["save_batch_size"], num_workers=nw,
                            pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                            prefetch_factor=4 if nw > 0 else None)
        print(f"[export] noise [{start}, {end}) -> {len(ds):,} rows", flush=True)
        export(model, iter(loader), len(ds), P, c["t_dim"], len(base.noises), device, str(out))


if __name__ == "__main__":
    main()
