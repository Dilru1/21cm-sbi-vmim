#!/usr/bin/env python3
"""Join the CNN export shards summaries/part_<start>_<end>/ into summaries/*.npy.

  python scripts/merge_shards.py CONFIG --cseed 43

Rows end up grouped by noise block; stage 2 splits by sim id and shuffles, so only the
row alignment across the four files matters. Shards are removed after the merge.
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sbi.config import load_config, run_dir  # noqa: E402

FILES = ("theta.npy", "t.npy", "original_sim_ids.npy", "noise_ids.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--cseed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    summ = run_dir(cfg, args.cseed) / "summaries"

    parts = sorted((int(p.name.split("_")[1]), int(p.name.split("_")[2]), p)
                   for p in summ.glob("part_*") if p.is_dir())
    cursor = 0
    for s, e, p in parts:
        if s != cursor or not all((p / f).exists() for f in FILES):
            sys.exit(f"incomplete shard set at {p.name} (expected start {cursor})")
        cursor = e
    if cursor != int(cfg["data"]["n_noise"]):
        sys.exit(f"shards cover noise [0, {cursor}), expected {cfg['data']['n_noise']}")

    lens = [np.load(p / "t.npy", mmap_mode="r").shape[0] for _, _, p in parts]
    n = sum(lens)
    for f in FILES:
        first = np.load(parts[0][2] / f, mmap_mode="r")
        out = np.lib.format.open_memmap(summ / f, "w+", first.dtype, (n,) + first.shape[1:])
        off = 0
        for (_, _, p), k in zip(parts, lens):
            src = np.load(p / f, mmap_mode="r")
            for lo in range(0, k, 1_000_000):
                out[off + lo: off + min(lo + 1_000_000, k)] = src[lo: min(lo + 1_000_000, k)]
            off += k
        out.flush()
        print(f"  {f}: {n:,} rows", flush=True)
    for _, _, p in parts:
        shutil.rmtree(p)
    print(f"merged {len(parts)} shards -> {summ}")


if __name__ == "__main__":
    main()
