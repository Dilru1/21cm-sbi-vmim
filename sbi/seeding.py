"""Seed control for ensemble / replica runs.

Two INDEPENDENT seeds, deliberately separated:

  split_seed  -- controls WHICH sims land in train vs val. Must be IDENTICAL
                 across every replica of an ensemble, otherwise members are
                 trained on different data, their exported summaries are not
                 row-comparable, and your SBC targets stop lining up.

  init_seed   -- controls weight initialization, batch order, dropout, and
                 data augmentation (noise draws, flips, rolls). This is the
                 knob you VARY (0,1,2,3,...) to get ensemble members.

Legacy configs that only set `seed` still work: both seeds fall back to it,
which reproduces the old single-realization behaviour exactly.

The split RNGs in data.split_by_sim and cubes.prepare_cube_loaders already use
LOCAL np.random.default_rng(seed) objects, so they are immune to whatever the
global RNG state is. That is why set_all_seeds() can be called before the split
without disturbing it.
"""

import os
import random

import numpy as np


def resolve_seeds(block, legacy_key="seed"):
    """Return (split_seed, init_seed) from a config block.

    Priority: explicit split_seed/init_seed  >  legacy `seed`  >  42/0.
    """
    legacy = block.get(legacy_key, 42)
    split_seed = int(block.get("split_seed", legacy))
    init_seed = int(block.get("init_seed", legacy))
    return split_seed, init_seed


def resolve_arm_name(cfg):
    """The replica-aware arm name. MUST be identical in stages 1, 2 and 3.

    If compressor.tag_arm_with_seed is true, the compressor's init_seed is
    appended: 'arm_cnn' -> 'arm_cnn_s1'. Stage 2 and 3 call this with the SAME
    cfg, so they automatically read the summaries written by the matching
    stage-1 replica instead of silently picking up a different one.
    """
    c = cfg.get("compressor", {})
    name = cfg["arm_name"]
    if c.get("tag_arm_with_seed", False):
        _, comp_init = resolve_seeds(c)
        name = f"{name}_s{comp_init}"
    return name


def apply_arm_name(cfg):
    """Rewrite cfg['arm_name'] in place. Call BEFORE arm_dirs(cfg)."""
    cfg["arm_name"] = resolve_arm_name(cfg)
    return cfg


def set_all_seeds(seed, deterministic=True, verbose=True):
    """Seed python / numpy / torch (CPU + all CUDA devices).

    Call ONCE at the top of a stage, before any model is constructed and before
    any DataLoader iterator is created. Model init consumes the torch global RNG,
    and DataLoader worker base seeds are drawn from it too, so ordering matters.

    deterministic=True additionally forces cuDNN into deterministic mode. This
    makes runs bit-reproducible on the same hardware but can cost 10-30% on 3D
    convs, so it is off by default -- for an ensemble you only need the members
    to be genuinely DIFFERENT and re-runnable, not bit-identical.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        if verbose:
            print(f"[seed] torch unavailable; seeded python+numpy with {seed}", flush=True)
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    if verbose:
        print(f"[seed] global seed={seed} deterministic={deterministic}", flush=True)


def make_generator(seed):
    """torch.Generator for DataLoader(shuffle=True) so batch ORDER follows init_seed.

    Without this, shuffle order comes from the torch global RNG, which is fine
    on a fresh process but drifts if anything upstream consumes randomness.
    """
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id):
    """DataLoader worker_init_fn: give every worker its own numpy/random stream.

    PyTorch seeds `torch` and `random` per worker but does NOT seed numpy. With
    fork start-method every worker inherits the SAME numpy global state, so a
    Dataset whose __getitem__ calls np.random.* (e.g. NoiseResampledDataset
    picking a noise realization, or the cube augmentations) hands out identical
    draws from every worker. Passing this as worker_init_fn fixes that.
    """
    import torch

    base = torch.initial_seed() % (2**32)
    np.random.seed((base + worker_id) % (2**32))
    random.seed(base + worker_id)


def seeded_loader_kwargs(init_seed, shuffle):
    """Convenience: the two kwargs every training DataLoader should carry."""
    kw = {"worker_init_fn": seed_worker}
    if shuffle:
        kw["generator"] = make_generator(init_seed)
    return kw
