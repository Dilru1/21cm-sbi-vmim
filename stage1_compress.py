#!/usr/bin/env python3
"""Stage 1: train the arm's compressor and export NLE summaries.

  python stage1_compress.py configs/arm_mlp.yaml
  python stage1_compress.py configs/arm_cnn.yaml --export-only

arm_type=mlp  -> SummaryCompressor on summaries_npz (Case 1)
arm_type=cnn  -> ResNet3DCompressor on cubes        (Case 2)

Both write summaries to {scratch_root}/{arm_name}/summaries/ as a memmap
(theta, t, original_sim_ids, noise_ids). The compressor checkpoint + RF-R2
log live in {arm}/nle/ (kept with the arm). Export noise_scale (CNN) always
equals training noise_scale.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import load_config, arm_dirs, load_source, split_by_sim
#from sbi.compressors.heads import build_head
from sbi.compressors.heads import build_head_from_cfg
#from sbi.train_compressor import run_training
#from sbi.train_compressor_legacy_probe import run_training
from sbi.train_comp_update import run_training
from sbi.seeding import (resolve_seeds, set_all_seeds, seed_worker,
                         make_generator, apply_arm_name)


import torch
#torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 48)))
#torch.set_num_interop_threads(1) 
#torch.backends.mkldnn.enabled = True   
#print("threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())

# in stage1_compress.py, MLP branch — replace the static make_loader(t_tr, ...) with:
class NoiseResampledDataset(torch.utils.data.Dataset):
    """One row per SIM; a fresh random noise rep is drawn each __getitem__.
    Epoch = 9120 samples; noise becomes true augmentation, like the CNN arm."""
    def __init__(self, t_all, th_all, sim_ids_all, sims):
        self.groups = []          # (theta, t_block[1000, n_summ]) per sim
        for s in sims:
            m = sim_ids_all == s
            self.groups.append((th_all[m][0], t_all[m]))
    def __len__(self):
        return len(self.groups)
    def __getitem__(self, i):
        th, block = self.groups[i]
        j = np.random.randint(block.shape[0])
        return torch.from_numpy(block[j].astype(np.float32)), torch.from_numpy(th[:4].astype(np.float32))


def export_memmap(compressor, get_batch_iter, n_rows, device, out_dir, t_dim, noise_scale=None):
    raw = compressor.module if isinstance(compressor, nn.DataParallel) else compressor
    raw.eval()
    theta_mm = np.lib.format.open_memmap(os.path.join(out_dir, "theta.npy"), mode="w+", dtype=np.float32, shape=(n_rows, 4))
    t_mm = np.lib.format.open_memmap(os.path.join(out_dir, "t.npy"), mode="w+", dtype=np.float32, shape=(n_rows, t_dim))
    sim_mm = np.lib.format.open_memmap(os.path.join(out_dir, "original_sim_ids.npy"), mode="w+", dtype=np.int64, shape=(n_rows,))
    noise_mm = np.lib.format.open_memmap(os.path.join(out_dir, "noise_ids.npy"), mode="w+", dtype=np.int64, shape=(n_rows, 3))
    if noise_scale is not None:
        np.save(os.path.join(out_dir, "export_noise_scale.npy"), np.array([noise_scale], np.float32))
    off = 0
    with torch.no_grad():
        for bi, (x, theta, sim, nids) in enumerate(get_batch_iter()):
            x = x.to(device)
            out = raw(x); t = out[0] if isinstance(out, tuple) else out
            k = x.shape[0]
            theta_mm[off:off+k] = np.asarray(theta)[:, :4].astype(np.float32)
            t_mm[off:off+k] = t.detach().float().cpu().numpy().astype(np.float32)
            sim_mm[off:off+k] = np.asarray(sim).astype(np.int64)
            nn_ = np.asarray(nids)
            noise_mm[off:off+k] = nn_.astype(np.int64) if nn_.ndim == 2 else np.repeat(nn_[:, None], 3, 1).astype(np.int64)
            off += k
            if bi % 100 == 0:
                print(f"  export {off:,}/{n_rows:,}", flush=True)
    for mm in (theta_mm, t_mm, sim_mm, noise_mm):
        mm.flush()
    print(f"exported summaries -> {out_dir}  t {t_mm.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--no-export", action="store_true",
                    help="train + RF-R2 probe + loss curves only; skip the memmap export "
                         "(for quick VMIM diagnostics where you just want the plots)")
    ap.add_argument("--ckpt", default="bestprobe", choices=["bestprobe", "bestloss"],
                    help="which checkpoint to export from: bestprobe (best mean RF-R2, "
                         "default -- the one where rHS survived) or bestloss")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    c = cfg["compressor"]
    split_seed, init_seed = resolve_seeds(c)

    apply_arm_name(cfg)          # arm_cnn -> arm_cnn_s<init_seed> if tagging is on
    dirs = arm_dirs(cfg)         # MUST come after the rename

    set_all_seeds(init_seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage1 arm={cfg['arm_name']} split_seed={split_seed} init_seed={init_seed}", flush=True)

    if cfg["arm_type"] == "mlp":
        from sbi.compressors.mlp import SummaryCompressor
        from sbi.torch_loader import make_loader
        src = load_source(cfg["data"]["summaries_npz"])
        n_input = src.n_summaries
        # NOTE: for the MLP arm the "t" slot of split_by_sim carries the RAW
        # hand-crafted summaries and the "theta" slot carries the params.
        (t_tr, th_tr, sim_tr, ntr, t_va, th_va, sim_va, nva, train_sims, val_sims) = split_by_sim(
            src.t, src.theta, src.sim_ids, src.noise_ids, c["val_frac"], split_seed)
        np.save(dirs["nle"] / "train_sims.npy", train_sims)
        np.save(dirs["nle"] / "val_sims.npy", val_sims)
        #compressor = SummaryCompressor(n_input=n_input, t_dim=cfg["t_dim"],
        #                               dense_layers=tuple(c["dense_layers"]), dropout=c["dropout"],
        #                               activation=c["activation"], use_resnet=c["use_resnet"],
        #                               n_params=cfg["n_params"]).to(device)
        #tl = make_loader(t_tr, th_tr, c["batch_size"], True, c["num_workers"], device.type == "cuda")
        #vl = make_loader(t_va, th_va, c["batch_size"], False, c["num_workers"], device.type == "cuda")
        
        compressor = SummaryCompressor(n_input=n_input, t_dim=cfg["t_dim"],
                                       dense_layers=tuple(c["dense_layers"]), dropout=c["dropout"],
                                       activation=c["activation"], use_resnet=c["use_resnet"],
                                       n_params=cfg["n_params"]).to(device)

        # Use the NoiseResampledDataset for true on-the-fly augmentation
        from torch.utils.data import DataLoader
        
        train_ds = NoiseResampledDataset(t_tr, th_tr, sim_tr, train_sims)
        val_ds = NoiseResampledDataset(t_va, th_va, sim_va, val_sims)

        # worker_init_fn is REQUIRED here: NoiseResampledDataset.__getitem__ calls
        # np.random.randint, and torch does NOT seed numpy per worker -- without it
        # every worker inherits the same numpy state and draws identical noise reps.
        tl = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,
                        num_workers=c["num_workers"], pin_memory=(device.type == "cuda"),
                        worker_init_fn=seed_worker, generator=make_generator(init_seed))
        vl = DataLoader(val_ds, batch_size=c["batch_size"], shuffle=False,
                        num_workers=c["num_workers"], pin_memory=(device.type == "cuda"),
                        worker_init_fn=seed_worker)

        
        # loaders yield (summary_x, theta); adapt to (x, theta, sim, nids) shape expected by trainer
        # Wrap loaders to yield the dummy (sim, nids) expected by the trainer
        class _Wrap:
            def __init__(self, loader): self.loader = loader
            def __iter__(self):
                for a, b in self.loader:
                    yield a, b, torch.zeros(len(a)), torch.zeros(len(a), 3)
            def __len__(self): return len(self.loader)
            
        tl, vl = _Wrap(tl), _Wrap(vl)
        export_src = src
    else:

        model_choice = c.get("model", "seblock").lower()

        from sbi.cubes import prepare_cube_loaders, DeterministicNoisyExport
        if model_choice == "seblock":
            from sbi.compressors.cnn_seblock_up import ResNet3DCompressor as comp #for SEBLOCK
            #from sbi.compressors.cnn_seblock import ResNet3DCompressor as comp #for SEBLOCK OLD
            
        
        elif model_choice == "conv4d":
            from sbi.compressors.cnn_grn4d_up import Conv4DCompressor as comp #for grn4d
            #from sbi.compressors.cnn_grn4d import Conv4DCompressor as comp #for grn4d OLD
            
            from sbi.cubes_grn4d import prepare_cube_loaders, DeterministicNoisyExport

        
        from torch.utils.data import DataLoader
        
        tl, vl, full_va = prepare_cube_loaders(cfg["data"], c, split_seed, c["val_frac"])
        compressor = comp(t_dim=cfg["t_dim"], n_params=cfg["n_params"],
                                         direct=bool(c.get("direct_regression", False))).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            compressor = nn.DataParallel(compressor)

    # manifest: lets stage 2/3/4 (and future you) confirm which replica this is
    import json
    with open(dirs["nle"] / "run_meta.json", "w") as f:
        json.dump({"arm_name": cfg["arm_name"], "arm_type": cfg["arm_type"],
                   "split_seed": split_seed, "init_seed": init_seed,
                   "t_dim": cfg["t_dim"], "model": c.get("model", "seblock")}, f, indent=2)

    #head = build_head(c["vmim_head"], cfg["t_dim"], cfg["n_params"], c["n_mix_head"], c["hidden_head"]).to(device)
    head = build_head_from_cfg(c, cfg).to(device)
    h = build_head_from_cfg(cfg["compressor"], cfg)
    print("sigma floor for each parm:", h.sigma_floor, "jitter for each parm", h.pre.jitter) 

    if not args.export_only:
        t0 = time.perf_counter()
        run_training(compressor, head, tl, vl, device, str(dirs["nle"]), c, cfg["n_params"])
        print(f"[TIMER] train {time.perf_counter()-t0:.1f}s", flush=True)

    if args.no_export:
        print("[NO-EXPORT] training + probes done; skipping memmap export. "
              f"RF-R2 history and loss curves saved in {dirs['nle']}", flush=True)
        return

    # reload chosen checkpoint, export
    raw = compressor.module if isinstance(compressor, nn.DataParallel) else compressor
    ckpt_name = "learned_compressor_bestprobe.pt" if args.ckpt == "bestprobe" else "learned_compressor.pt"
    ckpt_path = dirs["nle"] / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = dirs["nle"] / "learned_compressor.pt"
        print(f"[WARN] requested {ckpt_name} missing; falling back to {ckpt_path.name}", flush=True)
    print(f"exporting from checkpoint: {ckpt_path}", flush=True)
    raw.load_state_dict(torch.load(ckpt_path, map_location=device)); raw.eval()

    if cfg["arm_type"] == "mlp":
        from sbi.torch_loader import make_loader
        from torch.utils.data import DataLoader, TensorDataset
        # export ALL rows (every sim, every noise realization already present in npz)
        all_t = torch.tensor(export_src.t, dtype=torch.float32)
        all_theta = export_src.theta; all_sim = export_src.sim_ids; all_noise = export_src.noise_ids
        loader = DataLoader(TensorDataset(all_t, torch.arange(len(all_t))),
                            batch_size=c["save_batch_size"], shuffle=False)
        def it():
            for xb, idx in loader:
                idx = idx.numpy()
                yield xb, all_theta[idx], all_sim[idx], all_noise[idx] if all_noise.ndim == 2 else all_noise[idx]
        export_memmap(compressor, it, len(all_t), device, str(dirs["summaries"]), cfg["t_dim"])
    else:
        from torch.utils.data import DataLoader
        total = cfg["data"]["total_nnoise"]
        ds = DeterministicNoisyExport(full_va, total, c["noise_scale"])
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 6)); nw = min(8, max(0, cpus - 1))
        loader = DataLoader(ds, batch_size=c["save_batch_size"], shuffle=False, num_workers=nw,
                            pin_memory=(device.type == "cuda"))
        def it():
            for x, theta, sim, nids in loader:
                yield x, theta, sim, nids
        export_memmap(compressor, it, len(ds), device, str(dirs["summaries"]), cfg["t_dim"], noise_scale=c["noise_scale"])


if __name__ == "__main__":
    main()