"""Torch-free data helpers shared across arms."""

from pathlib import Path

import numpy as np


class Source:
    def __init__(self, theta, t, sim_ids, noise_ids):
        self.theta, self.t, self.sim_ids, self.noise_ids = theta, t, sim_ids, noise_ids
        self.n_params, self.n_summaries = theta.shape[1], t.shape[1]


def load_source(path, mmap=True):
    p = Path(path)
    if p.is_file() and p.suffix == ".npz":
        d = np.load(p)
        theta = d["theta"].astype(np.float32)
        t = d["t"].astype(np.float32)
        sim = d["original_sim_ids"]
        noise = d["noise_ids"] if "noise_ids" in d.files else np.zeros(len(sim), np.int64)
    elif p.is_dir():
        mm = "r" if mmap else None
        theta = np.load(p / "theta.npy", mmap_mode=mm).astype(np.float32)
        t = np.load(p / "t.npy", mmap_mode=mm).astype(np.float32)
        sim = np.load(p / "original_sim_ids.npy", mmap_mode=mm)
        npth = p / "noise_ids.npy"
        noise = np.load(npth, mmap_mode=mm) if npth.exists() else np.zeros(len(sim), np.int64)
    else:
        raise FileNotFoundError(f"{path}: not an .npz or memmap dir")
    return Source(theta, t, np.asarray(sim), np.asarray(noise))


def load_sbc_sims(path):
    if path is None or not Path(path).exists():
        return np.array([], np.int64)
    arr = np.load(path, allow_pickle=True)
    return np.array(
        sorted({int((x.item() if hasattr(x, "item") else x)["sim"]) for x in arr}), np.int64
    )


def load_sbc_targets(path):
    arr = np.load(path, allow_pickle=True)
    return [
        {
            "sim": int((x.item() if hasattr(x, "item") else x)["sim"]),
            "offset": int((x.item() if hasattr(x, "item") else x)["offset"]),
        }
        for x in arr
    ]


def split_by_sim(theta, t, sim_ids, noise_ids=None, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sim_ids)
    rng.shuffle(uniq)
    nval = int(len(uniq) * val_frac)
    val, train = set(uniq[:nval]), set(uniq[nval:])
    tr, va = np.isin(sim_ids, list(train)), np.isin(sim_ids, list(val))
    rng2 = np.random.default_rng(seed + 1)
    it, iv = rng2.permutation(int(tr.sum())), rng2.permutation(int(va.sum()))
    ni = noise_ids if noise_ids is not None else np.zeros(len(sim_ids), np.int64)
    return (
        theta[tr][it],
        t[tr][it],
        sim_ids[tr][it],
        ni[tr][it],
        theta[va][iv],
        t[va][iv],
        sim_ids[va][iv],
        ni[va][iv],
        np.array(sorted(train)),
        np.array(sorted(val)),
    )
