"""Config for all arms. One YAML per arm; the scratch2 output tree is derived
from `scratch_root` + `arm_name` so every stage writes to the same organized
place:

  {scratch_root}/{arm_name}/
      summaries/   <- stage1 compressor export (memmap: theta,t,sim,noise)
      nle/         <- stage2 best_mdn_nle.pt + losses
      chains/      <- stage3 SBC_CHAINS_*.dat (+ _truth/_logp/_tobs/_meta)

The NLE and MCMC knobs are shared/identical across arms (that's what makes the
comparison fair); only `arm`, paths, n_params, t_dim and the compressor block
differ per file.
"""
import json
from pathlib import Path

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

DEFAULTS = {
    "arm_name": "arm",
    "arm_type": "mlp",                 # mlp (summaries) | cnn (cubes)
    "scratch_root": "/gscratch/ddehiwalage-don/sbi_runs",
    "n_params": 4,
    "t_dim": 8,

    # ---- inputs ----
    "data": {
        # MLP arm: a single npz of (already-computed) summaries
        "summaries_npz": "",
        # CNN arm: cube + noise .dat lists and param/sim files
        "s_paths": [], "n_paths": [],
        "params_path": "", "sim_ids_path": "",
        "total_nnoise": 1000,
    },

    # ---- stage 1: compressor (shared knobs; arch picked by arm_type) ----
    "compressor": {
        "model": "seblock",
        "init_norm": True,
        "vmim_head": "gmm_old",
        "n_mix_head": 4, "hidden_head": 64,
        "epochs": 60, "patience": 8, "lr": 2.0e-4,
        "batch_size": 16, "save_batch_size": 8192,
        "num_workers": 4, "val_frac": 0.1, "seed": 42,
        "warmup_epochs": 10, "dec_strength": 0.0, "probe_every": 10,
        "param_weights": [1.0, 1.0, 1.0, 1.0],
        # training-mode knobs (see train_compressor.py)
        "aux_only": False, "aux_anneal": False, "aux_anneal_epochs": 8, "lam_aux": 100.0,
        # MLP-specific
        "dense_layers": [128, 128, 128, 128], "dropout": 0.0,
        "activation": "leaky_relu", "use_resnet": False,
        # CNN-specific
        "augment": True, "noise_scale": 1.0,

        # VICReg anti-collapse regularizer knobs
        "lam_var": 0.0,        # forces standard deviation of each dimension up
        "lam_cov": 0.0,        # penalizes correlation/redundancy between dimensions
        "var_target": 0.0,     # the minimum target standard deviation for each t dim

    },

    # ---- stage 2: NLE (IDENTICAL across arms) ----
    "nle": {
        "param_style": "semelin",      # semelin | softplus  (keep same for all arms)
        "diag_floor": 1.0e-4,
        "normalize_t": False,
        "n_mix": 2, "hidden": 64,
        "batch_size": 4, "nsplit": 10, "val_frac": 0.1,
        "num_workers": 4, "seed": 42, "betas": [0.5, 0.999],
        "lr_phase1": 2.0e-4, "lr_phase2": 2.0e-5,
        "target_path": "/gscratch/ddehiwalage-don/sbi_runs/sbc_targets.npy", "exclude_sbc_targets": False,
    },

    # ---- stage 3: MCMC (IDENTICAL across arms) ----
    "mcmc": {
        "target_path": "/gscratch/ddehiwalage-don/sbi_runs/sbc_targets.npy",
        "walkers": 160, "steps": 2000, "burnin": 500, "seed": 13,
        "loglik_batch_size": 8192, "use_original_prior": True,
        "wedge_tau_idx": 1, "wedge_mmin_idx": 3, "filter_dlogp": None,
    },
}


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _coerce(s):
    if s.lower() in ("true", "false"): return s.lower() == "true"
    if s.lower() in ("null", "none"): return None
    for c in (int, float):
        try: return c(s)
        except ValueError: pass
    return s


def _set_dotted(d, dotted, val):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = val


def load_config(path, overrides=None):
    p = Path(path); text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        if not _HAVE_YAML:
            raise RuntimeError("pyyaml missing; use .json or pip install pyyaml")
        user = yaml.safe_load(text)
    else:
        user = json.loads(text)
    cfg = _merge(DEFAULTS, user or {})
    for ov in (overrides or []):
        k, _, v = ov.partition("=")
        _set_dotted(cfg, k.strip(), _coerce(v.strip()))
    return cfg


def arm_dirs(cfg):
    """Return the organized scratch2 subdirs for this arm, creating them."""
    root = Path(cfg["scratch_root"]) / cfg["arm_name"]
    dirs = {
        "root": root,
        "summaries": root / "summaries",
        "nle": root / "nle",
        "chains": root / "chains",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs