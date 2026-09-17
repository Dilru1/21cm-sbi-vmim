import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def worker_init_fn(worker_id):
    """Ensures each parallel CPU worker process gets a distinct NumPy random seed."""
    process_seed = torch.initial_seed() % (2**32)
    np.random.seed(process_seed + worker_id)

def get_stats(signal_paths, noise_paths, sim_ids, noise_scale):
    """Computes dataset statistics exactly matching the legacy additive variance formula."""
    means = []
    stds = []
    for s_path, n_path in zip(signal_paths, noise_paths):
        signals = np.fromfile(s_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        noises = np.fromfile(n_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        signals = signals[:9827][sim_ids]

        signal_mean = signals.mean()
        signal_var  = signals.var()
        noise_mean  = noises.mean()
        noise_var   = noises.var()
        
        mean = np.float32(signal_mean + noise_scale * noise_mean)
        std  = np.float32(np.sqrt(signal_var + (noise_scale ** 2) * noise_var + 1e-12))
        means.append(mean)
        stds.append(std)
    return means, stds
    
def augment_cube(cube):
    """
    Applies spatial data augmentations to a 6D tensor.
    Incoming cube shape: [1, C, Z, D, H, W] -> [1, 1, 3, 32, 32, 32]
    We isolate operations strictly to spatial axes D, H, W (dims 3, 4, 5).
    """
    # 1. Random Flips across spatial axes D, H, W (axes 3, 4, 5)
    for axis in [3, 4, 5]:
        if np.random.random() > 0.5:
            cube = torch.flip(cube, dims=[axis])
            
    # 2. Random 90-degree rotations in the H-W plane (axes 4 and 5)
    k = np.random.randint(0, 4)
    cube = torch.rot90(cube, k, dims=[4, 5])
    
    # 3. Cyclic Shifting (Periodic Boundary conditions on axes 3, 4, 5)
    shift_d = np.random.randint(0, 32)
    shift_h = np.random.randint(0, 32)
    shift_w = np.random.randint(0, 32)
    
    return torch.roll(cube, shifts=(shift_d, shift_h, shift_w), dims=(3, 4, 5))

class CubeDataset(Dataset):
    def __init__(self, s_paths, n_paths, params_path, sim_ids_path, means, stds, augment, noise_scale):
        self.params = np.load(params_path).astype(np.float32)

        self.params_norm = self.params.copy()
        self.params_norm[:,0] = (np.log10(self.params_norm[:,0] / 0.00192) - (-1.)) / (1. - (-1.))
        self.params_norm[:,1] = (np.log10(self.params_norm[:,1]) - np.log10(738.9)) / (np.log10(10504.7) - np.log10(738.9))
        self.params_norm[:,3] = (self.params_norm[:,3] - 8.) / (9.6 - 8.)
        self.params_norm[:,4] = (self.params_norm[:,4] - 0.05) / (0.5 - 0.05)

        self.sim_ids = np.load(sim_ids_path).astype(int)

        assert len(self.params) == len(self.sim_ids)

        # Pre-load arrays completely into system RAM upfront
        self.signals = [np.fromfile(p, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)[:9827][self.sim_ids] for p in s_paths]
        self.noises = [np.fromfile(p, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32) for p in n_paths]

        self.means = means 
        self.stds = stds   
        self.augment = augment 
        self.noise_scale = noise_scale
        
    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        original_sim_id = self.sim_ids[idx]
        stacked_cube = []
        for i in range(3):
            noise_idx = np.random.randint(0, self.noises[i].shape[0])
            noisy_cube = self.signals[i][idx] + self.noise_scale * self.noises[i][noise_idx]
            norm_cube = (noisy_cube - self.means[i]) / (self.stds[i] + 1e-8)
            stacked_cube.append(norm_cube)
            
        # 1. Stack channels to shape: [3, 32, 32, 32] (where 3 is the Z/redshift axis)
        data = torch.from_numpy(np.stack(stacked_cube).astype(np.float32))

        # 2. Add structural single-channel wrapper dimension -> [C, Z, D, H, W] = [1, 3, 32, 32, 32]
        data = data.unsqueeze(0)

        if self.augment:
            # Temporarily add mock batch channel for 4D/5D spatial tracking adjustments
            data = augment_cube(data.unsqueeze(0)).squeeze(0)

        target = torch.from_numpy(self.params_norm[idx, [0, 1, 2, 3]])
            
        # Returning shape: [1, 3, 32, 32, 32]. Dataloader batching expands it cleanly to 6D: [B, 1, 3, 32, 32, 32]
        return data, target, int(original_sim_id)

def prepare_cube_loaders(cfg_data, c, seed, val_frac):
    sim_ids = np.load(cfg_data["sim_ids_path"]).astype(int)
    means, stds = get_stats(cfg_data["s_paths"], cfg_data["n_paths"], sim_ids, c["noise_scale"])

    print(f"Computed Means: {means}\nComputed Stds: {stds}")

    # Index shuffling partition splitting logic
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(sim_ids))
    rng.shuffle(all_indices)
    
    n_val = int(val_frac * len(all_indices))
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    full_train_dataset = CubeDataset(cfg_data["s_paths"], cfg_data["n_paths"], cfg_data["params_path"], cfg_data["sim_ids_path"], means, stds, augment=c["augment"], noise_scale=c["noise_scale"])
    full_val_dataset   = CubeDataset(cfg_data["s_paths"], cfg_data["n_paths"], cfg_data["params_path"], cfg_data["sim_ids_path"], means, stds, augment=False, noise_scale=c["noise_scale"])

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(full_val_dataset, val_indices)
    
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
    
    #print(f"--> Slurm allocated CPUs: {allocated_cpus} | Dynamically setting num_workers to: {num_workers}")

    train_loader = DataLoader(
        train_dataset, batch_size=c["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=False,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, full_val_dataset

class DeterministicNoisyExport(Dataset):
    """Deterministic NLE summary generation table wrapper supporting 6D models."""
    def __init__(self, base_dataset, n_noise_per_sim, noise_scale, noise_start=0, noise_end=None):
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
            norm_cube = (noisy_cube - self.base.means[ch]) / (self.base.stds[ch] + 1e-8)

            stacked_cube.append(norm_cube.astype(np.float32))
            noise_ids.append(noise_idx)

        # Combines frequencies into Z axis, then wraps a single input channel layout: [1, 3, 32, 32, 32]
        data = torch.from_numpy(np.stack(stacked_cube, axis=0).astype(np.float32)).unsqueeze(0)
        target = torch.from_numpy(self.base.params_norm[sim_idx, :4].astype(np.float32))
        
        return data, target, original_sim_id, torch.tensor(noise_ids, dtype=torch.long)