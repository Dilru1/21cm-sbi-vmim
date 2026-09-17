import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# CHANGES vs previous version
# ---------------------------
# Theta jitter and the theta2 affine margin are REMOVED from the dataset.
# They now live inside the VMIM head (heads.ThetaPreprocessor), applied
# train-time-only during compressor training. Consequences:
#   * exported theta.npy is PRISTINE -> NLE/MCMC/SBC truths and GV are in the
#     original normalized coordinates, comparable across arms;
#   * dequantization no longer rides on the (unrelated) `augment` flag;
#   * jittered values can no longer escape [0,1] before a logit.
# norm_flag behaviour for CUBES is kept, and is now applied identically in
# training and export (the earlier train/export mismatch).


def worker_init_fn(worker_id):
    process_seed = torch.initial_seed() % (2**32)
    np.random.seed(process_seed + worker_id)


def get_stats_global(signal_paths, noise_paths, sim_ids, noise_scale):
    """Single global (mean, std) across ALL redshifts combined."""
    all_signal_vals = []
    all_noise_var = 0.0
    n_noise_files = 0
    for s_path, n_path in zip(signal_paths, noise_paths):
        signals = np.fromfile(s_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        noises = np.fromfile(n_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        signals = signals[:9827][sim_ids]
        all_signal_vals.append(signals.ravel())
        all_noise_var += noises.var()
        n_noise_files += 1

    combined = np.concatenate(all_signal_vals)
    global_mean = np.float32(combined.mean())
    global_std = np.float32(np.sqrt(combined.var() + (noise_scale**2) * (all_noise_var / n_noise_files) + 1e-12))
    return [global_mean] * 3, [global_std] * 3


def get_stats(signal_paths, noise_paths, sim_ids, noise_scale):
    """Per-redshift statistics, legacy additive variance formula."""
    means, stds = [], []
    for s_path, n_path in zip(signal_paths, noise_paths):
        signals = np.fromfile(s_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        noises = np.fromfile(n_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        signals = signals[:9827][sim_ids]
        mean = np.float32(signals.mean() + noise_scale * noises.mean())
        std = np.float32(np.sqrt(signals.var() + (noise_scale ** 2) * noises.var() + 1e-12))
        means.append(mean)
        stds.append(std)
    return means, stds


def augment_cube(cube, is_6d=False):
    spatial_axes = [3, 4, 5] if is_6d else [1, 2, 3]
    rot_dims = [4, 5] if is_6d else [2, 3]
    for axis in spatial_axes:
        if np.random.random() > 0.5:
            cube = torch.flip(cube, dims=[axis])
    k = np.random.randint(0, 4)
    cube = torch.rot90(cube, k, dims=rot_dims)
    shifts = tuple(np.random.randint(0, 32) for _ in range(3))
    return torch.roll(cube, shifts=shifts, dims=spatial_axes)


class CubeDataset(Dataset):
    def __init__(self, s_paths, n_paths, params_path, sim_ids_path, means, stds,
                 augment, noise_scale, norm_flag, model_choice):
        self.params = np.load(params_path).astype(np.float32)

        self.params_norm = self.params.copy()
        self.params_norm[:, 0] = (np.log10(self.params_norm[:, 0] / 0.00192) - (-1.)) / (1. - (-1.))
        self.params_norm[:, 1] = (np.log10(self.params_norm[:, 1]) - np.log10(738.9)) / (np.log10(10504.7) - np.log10(738.9))
        # column 2 (rHS) is already in [0,1]; left untouched. Dequantization /
        # margin handling happens inside the VMIM head, not here.
        self.params_norm[:, 3] = (self.params_norm[:, 3] - 8.) / (9.6 - 8.)
        self.params_norm[:, 4] = (self.params_norm[:, 4] - 0.05) / (0.5 - 0.05)

        self.sim_ids = np.load(sim_ids_path).astype(int)
        assert len(self.params) == len(self.sim_ids)

        self.signals = [np.fromfile(p, dtype=np.float64).astype(np.float32)
                        .reshape(-1, 32, 32, 32)[:9827][self.sim_ids] for p in s_paths]
        self.noises = [np.fromfile(p, dtype=np.float64).astype(np.float32)
                       .reshape(-1, 32, 32, 32) for p in n_paths]

        self.means = means
        self.stds = stds
        self.augment = augment
        self.noise_scale = noise_scale
        self.norm_flag = bool(norm_flag)
        self.is_6d = (model_choice.lower() == "conv4d")

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        original_sim_id = self.sim_ids[idx]
        stacked_cube = []
        for i in range(3):
            noise_idx = np.random.randint(0, self.noises[i].shape[0])
            noisy_cube = self.signals[i][idx] + self.noise_scale * self.noises[i][noise_idx]
            if self.norm_flag:
                norm_cube = (noisy_cube - self.means[i]) / (self.stds[i] + 1e-8)
            else:
                norm_cube = noisy_cube
            stacked_cube.append(norm_cube)

        data = torch.from_numpy(np.stack(stacked_cube).astype(np.float32))

        if self.is_6d:
            data = data.unsqueeze(0)
            if self.augment:
                data = augment_cube(data.unsqueeze(0), is_6d=True).squeeze(0)
        elif self.augment:
            data = augment_cube(data, is_6d=False)

        # PRISTINE normalized targets -- no jitter, no margin.
        target = torch.from_numpy(self.params_norm[idx, [0, 1, 2, 3]].copy())
        return data, target, int(original_sim_id)


def prepare_cube_loaders(cfg_data, c, seed, val_frac):
    sim_ids = np.load(cfg_data["sim_ids_path"]).astype(int)

    model_choice = c.get("model", "seblock").lower()
    if model_choice == "conv4d":
        means, stds = get_stats_global(cfg_data["s_paths"], cfg_data["n_paths"], sim_ids, c["noise_scale"])
    else:
        means, stds = get_stats(cfg_data["s_paths"], cfg_data["n_paths"], sim_ids, c["noise_scale"])

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(sim_ids))
    rng.shuffle(all_indices)
    n_val = int(val_frac * len(all_indices))
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    norm_flag = c.get("init_norm", True)

    full_train_dataset = CubeDataset(cfg_data["s_paths"], cfg_data["n_paths"], cfg_data["params_path"],
                                     cfg_data["sim_ids_path"], means, stds, augment=c["augment"],
                                     noise_scale=c["noise_scale"], norm_flag=norm_flag, model_choice=model_choice)
    full_val_dataset = CubeDataset(cfg_data["s_paths"], cfg_data["n_paths"], cfg_data["params_path"],
                                   cfg_data["sim_ids_path"], means, stds, augment=False,
                                   noise_scale=c["noise_scale"], norm_flag=norm_flag, model_choice=model_choice)

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

    #allocated_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 6))
    #num_workers = max(1, allocated_cpus - 1)
    try:
        usable = len(os.sched_getaffinity(0))     # cores actually granted (not the env var)
    except AttributeError:
        usable = os.cpu_count() or 1
    # honor the config if set (n1 uses 2); else leave one core for compute. Never exceed usable.
    _cfg_nw = c.get("num_workers", None)
    num_workers = int(_cfg_nw) if _cfg_nw is not None else max(0, usable - 1)
    num_workers = min(num_workers, max(0, usable - 1))
    print(f"--> usable cores={usable}  num_workers={num_workers}", flush=True)
    

    train_loader = DataLoader(train_dataset, batch_size=c["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=False,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader, full_val_dataset


class DeterministicNoisyExport(Dataset):
    """Deterministic NLE summary generation table; exports PRISTINE theta."""
    def __init__(self, base_dataset, n_noise_per_sim, noise_scale, noise_start=0, noise_end=None):
        if isinstance(base_dataset, torch.utils.data.Subset):
            self.base = base_dataset.dataset
        else:
            self.base = base_dataset

        self.total_n_noise_per_sim = n_noise_per_sim
        self.noise_start = noise_start
        self.noise_end = noise_end if noise_end is not None else n_noise_per_sim
        assert 0 <= self.noise_start < self.noise_end <= n_noise_per_sim
        self.n_noise_this_part = self.noise_end - self.noise_start
        self.noise_scale = noise_scale

    def __len__(self):
        return len(self.base) * self.n_noise_this_part

    def __getitem__(self, global_idx):
        sim_idx = global_idx // self.n_noise_this_part
        noise_rep = self.noise_start + (global_idx % self.n_noise_this_part)
        original_sim_id = int(self.base.sim_ids[sim_idx])

        stacked_cube, noise_ids = [], []
        for ch in range(3):
            n_noise_total = self.base.noises[ch].shape[0]
            noise_idx = (original_sim_id * self.total_n_noise_per_sim + noise_rep + 10007 * ch) % n_noise_total
            noisy_cube = self.base.signals[ch][sim_idx] + self.noise_scale * self.base.noises[ch][noise_idx]
            # identical normalization convention to training (norm_flag)
            if self.base.norm_flag:
                norm_cube = (noisy_cube - self.base.means[ch]) / (self.base.stds[ch] + 1e-8)
            else:
                norm_cube = noisy_cube
            stacked_cube.append(norm_cube.astype(np.float32))
            noise_ids.append(noise_idx)

        data = torch.from_numpy(np.stack(stacked_cube, axis=0).astype(np.float32))
        if self.base.is_6d:
            data = data.unsqueeze(0)

        # PRISTINE normalized targets -- downstream coordinates untouched.
        target = torch.from_numpy(self.base.params_norm[sim_idx, :4].copy().astype(np.float32))
        return data, target, original_sim_id, torch.tensor(noise_ids, dtype=torch.long)







