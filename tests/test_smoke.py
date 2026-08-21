"""Import + basic-sanity smoke tests. These run on a bare machine (no torch,
no GPU, no cluster data) so CI can catch broken imports and obvious mistakes
on every push."""

import importlib


def test_torch_free_imports():
    # sbi/__init__ keeps torch-heavy modules lazy; these must import cleanly.
    for mod in ["sbi.config", "sbi.seeding", "sbi.data", "sbi.tracking"]:
        importlib.import_module(mod)


def test_tracking_is_safe_without_wandb():
    from sbi.tracking import init_run

    # No scratch_root on disk / no wandb -> must NOT raise, must return a
    # handle whose methods are safe no-ops.
    cfg = {"arm_name": "unit/test", "scratch_root": "/tmp/sbi_track_test"}
    run = init_run(cfg, stage="unit", tags=["ci"])
    run.log({"loss": 1.23})
    run.summary({"done": True})
    run.finish()
    assert run.active is False  # wandb absent in CI
