"""Shared SBI package. config/data are torch-free; torch-heavy modules load lazily."""

from .config import arm_dirs, load_config
from .data import load_sbc_sims, load_sbc_targets, load_source, split_by_sim

_LAZY = {
    "MDNLikelihood": "nle",
    "nle_loss": "nle",
    "build_head": "compressors.heads",
    "SummaryCompressor": "compressors.mlp",
    "ResNet3DCompressor": "compressors.cnn",
    "run_training": "train_compressor",
    "run_chain": "mcmc",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(name)
