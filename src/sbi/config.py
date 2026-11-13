"""YAML config, run directories and seeding.

One YAML = one experiment. A run is (config, compressor seed, NLE seed):

  {out_root}/{name}/c{cs}/            compressor.type = mlp | cnn
      compressor/                     weights, loss + probe logs, train/val sims
      summaries/                      exported t (theta.npy t.npy original_sim_ids.npy noise_ids.npy)
      nle_n{ns}/                      NLE weights, t_mean/t_std, loss log
      chains_n{ns}/                   MCMC chains
  {out_root}/{name}/raw/              compressor.type = none (summaries read from data.summaries_npz)

${VAR} in any string value is expanded from the environment (see slurm/env.example.sh).
"""
import os
import random
from pathlib import Path

import numpy as np
import yaml


def _expand(x):
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    if isinstance(x, str):
        out = os.path.expandvars(x)
        if "$" in out:
            raise KeyError(f"unset environment variable in config value {x!r}")
        return out
    return x


def load_config(path):
    with open(path) as f:
        return _expand(yaml.safe_load(f))


def run_dir(cfg, cs):
    root = Path(cfg["out_root"]) / cfg["name"]
    return root / "raw" if cfg["compressor"]["type"] == "none" else root / f"c{cs}"


def summaries_path(cfg, cs):
    if cfg["compressor"]["type"] == "none":
        return Path(cfg["data"]["summaries_npz"])
    return run_dir(cfg, cs) / "summaries"


def set_all_seeds(seed, deterministic=True):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
