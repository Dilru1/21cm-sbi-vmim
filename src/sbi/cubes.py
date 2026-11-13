"""Field cubes (Path B): signal + scaled noise, per-cube mean removal, standardisation,
symmetry augmentation, and the deterministic export table.

Data layout (data: block of the YAML):
  signal_paths   one raw float64 file per channel (redshift), cubes of box^3, row = simulation
  noise_paths    one raw float64 file per channel, a pool of noise cubes
  params_path    (n_sims, >= n_params) physical parameters, row-aligned with sim_ids_path
  sim_ids_path   (n_sims,) row of each simulation in the signal files
  n_signal       signal files are truncated to their first n_signal cubes before indexing
  params         per-parameter map to [0, 1] (see normalise_params)
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def normalise_params(params, spec):
    """Map each parameter to [0, 1]. spec[i] is one of
         {transform: none}
         {transform: linear, lo, hi[, scale]}      (p/scale - lo) / (hi - lo)
         {transform: log10,  lo, hi[, scale]}      (log10(p/scale) - log10(lo)) / (log10(hi) - log10(lo))
         {transform: log10,  log_lo, log_hi[, scale]}   bounds given directly in log10 units
    lo/hi are in units of p/scale."""
    out = params.copy()
    for i, s in enumerate(spec):
        kind = s.get("transform", "none")
        if kind == "none":
            continue
        x = out[:, i] / float(s["scale"]) if "scale" in s else out[:, i]
        if kind == "linear":
            out[:, i] = (x - float(s["lo"])) / (float(s["hi"]) - float(s["lo"]))
        elif kind == "log10":
            if "log_lo" in s:
                lo, hi = float(s["log_lo"]), float(s["log_hi"])
            else:
                lo, hi = np.log10(float(s["lo"])), np.log10(float(s["hi"]))
            out[:, i] = (np.log10(x) - lo) / (hi - lo)
        else:
            raise ValueError(f"unknown transform {kind!r}")
    return out


def remove_cube_mean(x):
    """Subtract each cube's own 3D mean (interferometers do not see the k=0 mode)."""
    return x - x.mean(axis=(-3, -2, -1), keepdims=True)


def _read_cubes(path, box):
    return np.fromfile(path, dtype=np.float64).astype(np.float32).reshape(-1, box, box, box)


def channel_stats(d, sim_ids, noise_scale):
    """Per-channel mean/std of what the network sees (mean-removed signal + noise):
       std^2 = Var_within(signal) + noise_scale^2 Var_within(noise), mean = 0."""
    means, stds = [], []
    for sp, npth in zip(d["signal_paths"], d["noise_paths"]):
        s = _read_cubes(sp, d["box"])[: d["n_signal"]][sim_ids]
        n = _read_cubes(npth, d["box"])
        s_var = max(float(s.var()) - float(s.reshape(len(s), -1).mean(1).var()), 0.0)
        n_var = max(float(n.var()) - float(n.reshape(len(n), -1).mean(1).var()), 0.0)
        means.append(np.float32(0.0))
        stds.append(np.float32(np.sqrt(s_var + noise_scale ** 2 * n_var + 1e-12)))
        print(f"  {os.path.basename(sp)}: std {stds[-1]:.4e}", flush=True)
    return means, stds


def _lattice_flip(x, axis):
    """Reflection about a lattice point, i -> -i mod N (periodic box)."""
    return torch.roll(torch.flip(x, dims=[axis]), 1, dims=axis)


def augment(x, box):
    """Symmetries shared by signal and noise: flip along the line of sight (axis 1),
    180 deg rotation in the sky plane, and a random periodic shift."""
    if np.random.random() > 0.5:
        x = _lattice_flip(x, 1)
    if np.random.random() > 0.5:
        x = _lattice_flip(_lattice_flip(x, 2), 3)
    shifts = tuple(int(np.random.randint(0, box)) for _ in range(3))
    return torch.roll(x, shifts=shifts, dims=(1, 2, 3))


class CubeDataset(Dataset):
    """Row i = simulation i of sim_ids_path. Each access draws a fresh noise cube per channel."""

    def __init__(self, d, n_params, means, stds, noise_scale, do_augment):
        self.box = int(d["box"])
        self.n_params = n_params
        params = np.load(d["params_path"]).astype(np.float32)
        self.params_norm = normalise_params(params, d["params"])
        self.sim_ids = np.load(d["sim_ids_path"]).astype(int)
        assert len(params) == len(self.sim_ids)
        self.signals = [_read_cubes(p, self.box)[: d["n_signal"]][self.sim_ids]
                        for p in d["signal_paths"]]
        self.noises = [_read_cubes(p, self.box) for p in d["noise_paths"]]
        self.means, self.stds = means, stds
        self.noise_scale = noise_scale
        self.do_augment = do_augment

    def __len__(self):
        return len(self.sim_ids)

    def cube(self, idx, noise_idx):
        chans = []
        for c in range(len(self.signals)):
            x = remove_cube_mean(self.signals[c][idx] + self.noise_scale * self.noises[c][noise_idx[c]])
            chans.append((x - self.means[c]) / (self.stds[c] + 1e-8))
        return torch.from_numpy(np.stack(chans).astype(np.float32))

    def __getitem__(self, idx):
        nidx = [np.random.randint(0, len(n)) for n in self.noises]
        x = self.cube(idx, nidx)
        if self.do_augment:
            x = augment(x, self.box)
        theta = torch.from_numpy(self.params_norm[idx, : self.n_params].astype(np.float32).copy())
        return x, theta, int(self.sim_ids[idx])


def _worker_init(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32) + worker_id)


def cube_loaders(cfg, split_seed):
    """-> train loader, val loader, the un-augmented dataset (for export)."""
    d, c = cfg["data"], cfg["compressor"]
    sim_ids = np.load(d["sim_ids_path"]).astype(int)
    means, stds = channel_stats(d, sim_ids, c["noise_scale"])

    idx = np.arange(len(sim_ids))
    np.random.default_rng(split_seed).shuffle(idx)
    n_val = int(c["val_frac"] * len(idx))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = CubeDataset(d, cfg["n_params"], means, stds, c["noise_scale"], do_augment=True)
    val_ds = CubeDataset(d, cfg["n_params"], means, stds, c["noise_scale"], do_augment=False)

    usable = len(os.sched_getaffinity(0))
    nw = min(int(c["num_workers"]), max(0, usable - 1))
    tl = DataLoader(torch.utils.data.Subset(train_ds, train_idx), batch_size=c["batch_size"],
                    shuffle=True, num_workers=nw, worker_init_fn=_worker_init)
    vl = DataLoader(torch.utils.data.Subset(val_ds, val_idx), batch_size=8, shuffle=False,
                    num_workers=0, pin_memory=True)
    return tl, vl, val_ds


class ExportTable(Dataset):
    """Every simulation x noise realisations [start, end) of n_noise. Deterministic:
    channel c of (sim, rep) uses noise cube (sim_id * n_noise + rep + 10007 c) mod pool,
    so shards over the noise axis reproduce the matching rows of a full export."""

    def __init__(self, base, n_noise, start=0, end=None):
        self.base, self.n_noise = base, n_noise
        self.start, self.end = start, n_noise if end is None else end
        self.width = self.end - self.start

    def __len__(self):
        return len(self.base) * self.width

    def __getitem__(self, i):
        idx, rep = i // self.width, self.start + i % self.width
        sid = int(self.base.sim_ids[idx])
        nidx = [(sid * self.n_noise + rep + 10007 * c) % len(n)
                for c, n in enumerate(self.base.noises)]
        theta = self.base.params_norm[idx, : self.base.n_params].astype(np.float32).copy()
        return self.base.cube(idx, nidx), torch.from_numpy(theta), sid, torch.tensor(nidx)
