"""Summary tables, sim-level train/val split, SBC targets."""
from pathlib import Path

import numpy as np


def load_summaries(path, mmap=True):
    """-> theta (N, P) f32, t (N, D) f32, sim_ids (N,), noise_ids (N,) or (N, C).

    path: an .npz with theta, t, original_sim_ids[, noise_ids], or a directory of
    .npy files with the same names (what stage 1 writes)."""
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(p)
        theta, t, sim = d["theta"], d["t"], d["original_sim_ids"]
        noise = d["noise_ids"] if "noise_ids" in d.files else np.zeros(len(sim), np.int64)
    else:
        mm = "r" if mmap else None
        theta = np.load(p / "theta.npy", mmap_mode=mm)
        t = np.load(p / "t.npy", mmap_mode=mm)
        sim = np.load(p / "original_sim_ids.npy", mmap_mode=mm)
        noise = np.load(p / "noise_ids.npy", mmap_mode=mm)
    return (theta.astype(np.float32), t.astype(np.float32),
            np.asarray(sim), np.asarray(noise))


def split_by_sim(theta, t, sim_ids, noise_ids, val_frac, seed):
    """All rows of a simulation land on the same side. Rows are shuffled."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(sim_ids)
    rng.shuffle(uniq)
    nval = int(len(uniq) * val_frac)
    val, train = set(uniq[:nval]), set(uniq[nval:])
    tr, va = np.isin(sim_ids, list(train)), np.isin(sim_ids, list(val))
    rng2 = np.random.default_rng(seed + 1)
    it, iv = rng2.permutation(int(tr.sum())), rng2.permutation(int(va.sum()))
    return (theta[tr][it], t[tr][it], sim_ids[tr][it], noise_ids[tr][it],
            theta[va][iv], t[va][iv], sim_ids[va][iv], noise_ids[va][iv],
            np.array(sorted(train)), np.array(sorted(val)))


def load_sbc_targets(path):
    """SBC target list: [{'sim': int, 'offset': int}, ...] (offset = row within the sim)."""
    arr = np.load(path, allow_pickle=True)
    out = []
    for x in arr:
        x = x.item() if hasattr(x, "item") else x
        out.append({"sim": int(x["sim"]), "offset": int(x["offset"])})
    return out
